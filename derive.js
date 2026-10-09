/*
 * derive.js
 * Rules for features that are calculated from other inputs (read from config.json):
 *   - "delta":  X_Delta = X_value_2nd - X_value_1st   (rounded to the source precision)
 *   - "status": 0 = below range, 1 = in range, 2 = above range
 *               (below if value < low; above if value > high; sex-specific ranges via rule.by)
 * A derived value is blank (null) whenever one of its sources is blank.
 */
(function (root, factory) {
  if (typeof module === "object" && module.exports) module.exports = factory();
  else root.Derive = factory();
})(typeof self !== "undefined" ? self : this, function () {
  "use strict";

  function roundTo(v, d) {
    if (v === null || v === undefined || v !== v) return null;
    // toFixed(12) gives a plain decimal string (no exponent), so appending "e<d>" is always valid.
    var r = Number(Math.round(Number(Number(v).toFixed(12) + "e" + d)) + "e-" + d);
    return r === 0 ? 0 : r;                       // avoid -0
  }

  /* rule: config.features[i].rule ; get(name) -> number | null */
  function evalRule(rule, get) {
    if (rule.kind === "delta") {
      var a = get(rule.minuend), b = get(rule.subtrahend);
      if (a === null || b === null) return null;
      return roundTo(a - b, rule.decimals);
    }
    if (rule.kind === "status") {
      var v = get(rule.source);
      if (v === null) return null;
      var lo = rule.low, hi = rule.high;
      if (rule.by) {
        var s = get(rule.by);
        if (s === null) return null;
        var rg = rule.ranges[String(s)];
        if (!rg) return null;
        lo = rg.low; hi = rg.high;
      }
      return v < lo ? 0 : (v > hi ? 2 : 1);
    }
    throw new Error("Unknown rule kind: " + rule.kind);
  }

  /* The range that applies to a status rule for the given sex code (or null). */
  function rangeFor(rule, sexCode) {
    if (rule.by) {
      if (sexCode === null || sexCode === undefined) return null;
      var rg = rule.ranges[String(sexCode)];
      return rg ? { low: rg.low, high: rg.high } : null;
    }
    return { low: rule.low, high: rule.high };
  }

  /*
   * primary:   { name: number|null } for config.primary_features
   * overrides: { name: number }     values the user typed into derived fields
   * returns    { values, auto }     values = what the model gets, auto = calculated value
   */
  function resolve(config, primary, overrides) {
    var rules = {};
    config.features.forEach(function (f) { if (f.derived) rules[f.name] = f.rule; });
    var vals = {}, auto = {};
    config.primary_features.forEach(function (n) {
      var v = primary[n];
      vals[n] = (v === undefined || v === null || v !== v) ? null : v;
    });
    var get = function (n) { return n in vals ? vals[n] : null; };
    config.derived_features.forEach(function (n) {
      auto[n] = evalRule(rules[n], get);
      var o = overrides && overrides[n];
      vals[n] = (o !== undefined && o !== null && o === o) ? o : auto[n];
    });
    return { values: vals, auto: auto };
  }

  /* Model input vector in the booster's feature order. Blank -> NaN (XGBoost's missing branch). */
  function toVector(config, values) {
    var impute = config.missing_policy === "impute";
    var byName = {};
    config.features.forEach(function (f) { byName[f.name] = f; });
    return config.feature_order.map(function (n) {
      var v = values[n];
      if (v === null || v === undefined || v !== v) return impute ? byName[n].impute_value : NaN;
      return v;
    });
  }

  return { roundTo: roundTo, evalRule: evalRule, rangeFor: rangeFor, resolve: resolve, toVector: toVector };
});