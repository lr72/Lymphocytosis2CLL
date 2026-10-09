# Progression risk calculator for Lymphocytosis Transformation to CLL


## Files (all in the repository root)

| File | Purpose |
|---|---|
| `index.html` | The calculator page (form, result, per-feature waterfall) |
| `treeshap.js` | Exact TreeSHAP for the XGBoost JSON model |
| `derive.js` | Rules for calculated fields (change, status) |
| `model.json` | Trained XGBoost model (from Python) |
| `config.json` | Features, ranges, derivation rules, calibration (from Python) |
| `test_cases.json` | Reference SHAP values from Python (used by the test only) |
| `test_node.js` | Checks the JavaScript against the Python reference |

## Check before publishing

```bash
node test_node.js        # expect: 30/30 cases match
```

## Try locally

```bash
python -m http.server 8000     # then open http://localhost:8000
```

Opening `index.html` by double-click does not work (browsers block local `fetch`).

## Publish

GitHub repository (public) -> Settings -> Pages -> Deploy from a branch -> `main` / root.

## Things you can edit

* `index.html`, top of the script: `TEXT` (subtitle, disclaimer), `LABS` (display names, short chart names, units), `PLAIN` (names of age, sex, history items).
* `config.json` -> `calibration`: when it is changed from `"none"` to a Platt calibration, the page automatically switches from "Model score (uncalibrated)" to "Estimated probability" and rescales the explanation chart.
* After retraining the model, re-export `model.json`, `config.json`, `test_cases.json` and run `node test_node.js` again.
