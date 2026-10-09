/*
 * treeshap.js
 * Exact TreeSHAP (path-dependent, Lundberg et al. 2018) for XGBoost models saved
 * with booster.save_model("model.json") (tested with XGBoost 1.5.2 JSON format).
 *
 * It reproduces what shap.TreeExplainer / booster.predict(pred_contribs=True)
 * return: one value per feature in log-odds (margin) units, plus the base value.
 *
 * Usage (browser or Node):
 *   const model = TreeSHAP.load(modelJson);
 *   const r = model.explain(x);          // x: array of numbers, NaN = missing
 *   r.base      expected margin E[f(X)]   (the waterfall starts here)
 *   r.phi[i]    SHAP value of feature i   (log-odds)
 *   r.margin    base + sum(phi)  = model margin for x
 *   r.prob      1 / (1 + exp(-margin))    (raw model probability)
 *
 * Supported: numerical splits, missing-value routing (default_left),
 * binary:logistic / reg:squarederror style single-output boosters.
 * Not supported (throws): categorical splits, multi-class, dart weights.
 */
(function (root, factory) {
  if (typeof module === "object" && module.exports) module.exports = factory();
  else root.TreeSHAP = factory();
})(typeof self !== "undefined" ? self : this, function () {
  "use strict";

  var f32 = Math.fround;

  /* ----------------------------- model loading ----------------------------- */

  function load(json) {
    var learner = json.learner;
    var lmp = learner.learner_model_param;
    var gb = learner.gradient_booster;
    if (gb.name !== "gbtree") throw new Error("Only gbtree boosters are supported, got " + gb.name);
    if (parseInt(lmp.num_class || "0", 10) > 1) throw new Error("Multi-class models are not supported");

    var nFeatures = parseInt(lmp.num_feature, 10);
    var objective = learner.objective || {};
    var baseScore = parseFloat(lmp.base_score);
    var isLogistic = /^binary:logistic/.test(objective.name || "") || /^binary:logitraw/.test(objective.name || "");
    // XGBoost stores base_score in probability space for logistic objectives.
    var baseMargin = objective.name === "binary:logistic" ? Math.log(baseScore / (1 - baseScore)) : baseScore;

    var trees = gb.model.trees.map(function (t, k) {
      var n = t.left_children.length;
      var splitType = t.split_type || [];
      for (var i = 0; i < n; i++) {
        if (splitType[i] && splitType[i] !== 0) throw new Error("Categorical splits are not supported (tree " + k + ")");
      }
      var tree = {
        n: n,
        left: Int32Array.from(t.left_children),
        right: Int32Array.from(t.right_children),
        feat: Int32Array.from(t.split_indices),
        // thresholds and leaf values are float32 inside XGBoost
        cond: Float64Array.from(t.split_conditions, f32),
        defLeft: Uint8Array.from(t.default_left, function (b) { return b ? 1 : 0; }),
        cover: Float64Array.from(t.sum_hessian),
        mean: null
      };
      tree.mean = nodeMeans(tree);
      tree.maxDepth = depth(tree, 0);
      return tree;
    });

    var scalePosWeight = null;
    if (objective.reg_loss_param && objective.reg_loss_param.scale_pos_weight != null) {
      scalePosWeight = parseFloat(objective.reg_loss_param.scale_pos_weight);
    }

    var biasTrees = 0;
    trees.forEach(function (t) { biasTrees += t.mean[0]; });

    return {
      nFeatures: nFeatures,
      featureNames: learner.feature_names || null,
      objective: objective.name || null,
      scalePosWeight: scalePosWeight,
      baseMargin: baseMargin,
      base: baseMargin + biasTrees,
      isLogistic: isLogistic,
      trees: trees,
      margin: function (x) { return margin(this, x); },
      explain: function (x) { return explain(this, x); }
    };
  }

  /* Cover-weighted expected leaf value of every node (same as XGBoost's mean values). */
  function nodeMeans(t) {
    var m = new Float64Array(t.n);
    (function rec(i) {
      if (t.left[i] === -1) { m[i] = t.cond[i]; return m[i]; }
      var l = rec(t.left[i]), r = rec(t.right[i]);
      m[i] = (t.cover[t.left[i]] * l + t.cover[t.right[i]] * r) / t.cover[i];
      return m[i];
    })(0);
    return m;
  }

  function depth(t, i) {
    if (t.left[i] === -1) return 0;
    return 1 + Math.max(depth(t, t.left[i]), depth(t, t.right[i]));
  }

  /* ------------------------------- prediction ------------------------------ */

  /* Child taken by XGBoost for value v at node i: left if v < threshold, NaN -> default. */
  function hotChild(t, i, v) {
    if (v !== v) return t.defLeft[i] ? t.left[i] : t.right[i];
    return v < t.cond[i] ? t.left[i] : t.right[i];
  }

  function margin(model, x) {
    var xf = toFloat32(x, model.nFeatures);
    var s = model.baseMargin;
    for (var k = 0; k < model.trees.length; k++) {
      var t = model.trees[k], i = 0;
      while (t.left[i] !== -1) i = hotChild(t, i, xf[t.feat[i]]);
      s += t.cond[i];
    }
    return s;
  }

  function toFloat32(x, n) {
    if (x.length !== n) throw new Error("Expected " + n + " features, got " + x.length);
    var out = new Float64Array(n);
    for (var i = 0; i < n; i++) {
      var v = x[i];
      out[i] = (v === null || v === undefined || v !== v) ? NaN : f32(+v);
    }
    return out;
  }

  /* --------------------------------- TreeSHAP ------------------------------- */

  function explain(model, x) {
    var n = model.nFeatures;
    var xf = toFloat32(x, n);
    var phi = new Float64Array(n);
    for (var k = 0; k < model.trees.length; k++) treeShap(model.trees[k], xf, phi);
    var sum = 0;
    for (var j = 0; j < n; j++) sum += phi[j];
    var m = model.base + sum;
    return { phi: phi, base: model.base, margin: m, prob: 1 / (1 + Math.exp(-m)) };
  }

  function treeShap(t, x, phi) {
    var D = t.maxDepth + 2;
    // Preallocated path arrays, one slice of length (depth+2) per recursion level.
    var feat = new Int32Array(D * D), zero = new Float64Array(D * D),
        one = new Float64Array(D * D), pw = new Float64Array(D * D);

    function extend(o, d, z, w, f) {            // o = offset of this level's path
      feat[o + d] = f; zero[o + d] = z; one[o + d] = w; pw[o + d] = d === 0 ? 1 : 0;
      for (var i = d - 1; i >= 0; i--) {
        pw[o + i + 1] += w * pw[o + i] * (i + 1) / (d + 1);
        pw[o + i] = z * pw[o + i] * (d - i) / (d + 1);
      }
    }

    function unwind(o, d, idx) {
      var w = one[o + idx], z = zero[o + idx], next = pw[o + d];
      for (var i = d - 1; i >= 0; i--) {
        if (w !== 0) {
          var tmp = pw[o + i];
          pw[o + i] = next * (d + 1) / ((i + 1) * w);
          next = tmp - pw[o + i] * z * (d - i) / (d + 1);
        } else {
          pw[o + i] = pw[o + i] * (d + 1) / (z * (d - i));
        }
      }
      for (var j = idx; j < d; j++) {
        feat[o + j] = feat[o + j + 1]; zero[o + j] = zero[o + j + 1]; one[o + j] = one[o + j + 1];
      }
    }

    function unwoundSum(o, d, idx) {
      var w = one[o + idx], z = zero[o + idx], next = pw[o + d], total = 0;
      for (var i = d - 1; i >= 0; i--) {
        if (w !== 0) {
          var tmp = next * (d + 1) / ((i + 1) * w);
          total += tmp;
          next = pw[o + i] - tmp * z * (d - i) / (d + 1);
        } else {
          total += (pw[o + i] / z) / ((d - i) / (d + 1));
        }
      }
      return total;
    }

    // d = unique path depth (can shrink after unwind), level = recursion level (storage slice).
    function recurse(node, d, level, parentOff, pz, po, pf) {
      var o = level * D;                          // this level's slice
      for (var c = 0; c < d; c++) {               // copy parent path
        feat[o + c] = feat[parentOff + c]; zero[o + c] = zero[parentOff + c];
        one[o + c] = one[parentOff + c];  pw[o + c] = pw[parentOff + c];
      }
      extend(o, d, pz, po, pf);

      if (t.left[node] === -1) {                  // leaf
        for (var i = 1; i <= d; i++) {
          var w = unwoundSum(o, d, i);
          phi[feat[o + i]] += w * (one[o + i] - zero[o + i]) * t.cond[node];
        }
        return;
      }

      var f = t.feat[node];
      var hot = hotChild(t, node, x[f]);
      var cold = hot === t.left[node] ? t.right[node] : t.left[node];
      var cov = t.cover[node];
      var hotZero = t.cover[hot] / cov, coldZero = t.cover[cold] / cov;
      var inZero = 1, inOne = 1, dd = d;

      var k = 0;
      for (; k <= dd; k++) if (feat[o + k] === f) break;
      if (k !== dd + 1) {                         // feature already on the path: undo its earlier split
        inZero = zero[o + k]; inOne = one[o + k];
        unwind(o, dd, k); dd -= 1;
      }
      recurse(hot, dd + 1, level + 1, o, hotZero * inZero, inOne, f);
      recurse(cold, dd + 1, level + 1, o, coldZero * inZero, 0, f);
    }

    // A stump (single leaf) contributes nothing.
    if (t.left[0] === -1) return;
    recurse(0, 0, 0, 0, 1, 1, -1);
  }

  return { load: load };
});