/*
 * test_node.js  -  run from the repository folder:   node test_node.js
 * Checks the JavaScript against the reference values exported from Python:
 *   - derived fields (delta, status) computed from the primary inputs only
 *   - model margin, base value and per-feature SHAP values
 */
const fs = require("fs");
const TreeSHAP = require("./treeshap.js");
const Derive = require("./derive.js");

const read = (f) => JSON.parse(fs.readFileSync(f, "utf8"));
const config = read("config.json");
const model = TreeSHAP.load(read("model.json"));
const cases = read("test_cases.json");

const TOL = 1e-4;
let worst = { margin: 0, base: 0, phi: 0, derived: 0 }, failures = 0;

// Feature order in the model must match the config.
if (model.featureNames && JSON.stringify(model.featureNames) !== JSON.stringify(config.feature_order)) {
  console.error("FAIL: feature order in model.json differs from config.json");
  process.exit(1);
}

cases.forEach((c, i) => {
  const { values } = Derive.resolve(config, c.inputs, {});
  let bad = [];

  // derived values the page computes vs. Python
  for (const [name, expected] of Object.entries(c.derived_expected)) {
    const got = values[name];
    const diff = (expected === null || got === null) ? (expected === got ? 0 : Infinity) : Math.abs(expected - got);
    worst.derived = Math.max(worst.derived, diff);
    if (diff > 1e-6) bad.push(`derived ${name}: js=${got} py=${expected}`);
  }

  const x = Derive.toVector(config, values);
  const r = model.explain(x);
  const m2 = model.margin(x);

  const dm = Math.abs(r.margin - c.margin), db = Math.abs(r.base - c.base), dq = Math.abs(m2 - c.margin);
  worst.margin = Math.max(worst.margin, dm, dq); worst.base = Math.max(worst.base, db);
  if (dm > TOL || dq > TOL) bad.push(`margin: js=${r.margin.toFixed(6)} py=${c.margin}`);
  if (db > TOL) bad.push(`base: js=${r.base.toFixed(6)} py=${c.base}`);

  config.feature_order.forEach((name, j) => {
    const d = Math.abs(r.phi[j] - c.contribs[name]);
    worst.phi = Math.max(worst.phi, d);
    if (d > TOL) bad.push(`phi ${name}: js=${r.phi[j].toFixed(6)} py=${c.contribs[name]}`);
  });

  if (bad.length) { failures++; console.log(`case ${i}: FAIL\n  ` + bad.slice(0, 6).join("\n  ")); }
});

console.log(`\n${cases.length - failures}/${cases.length} cases match (tolerance ${TOL})`);
console.log("largest absolute differences:", worst);
console.log("base value  js:", model.base.toFixed(6), " config:", config.model.base_value_margin.toFixed(6));
process.exit(failures ? 1 : 0);