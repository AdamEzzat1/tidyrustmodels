# R-versus-Rust verification harness design

## Systems under comparison

| system | what runs                                              | driver |
|--------|--------------------------------------------------------|--------|
| A      | R 4.4.0 + tidymodels 1.5.0 (engines `lm`, `glm`)       | `reference/tidymodels-r/*.R` via Rscript |
| B      | Rust `tidyrustmodels` (engines `linfa`, `smartcore`)   | `cargo test -p tidyrustmodels --test differential` |
| C      | Python `tidyrustmodels` bindings                        | `pytest python/tests/test_parity.py` |
| X      | Independent baselines: exact rational OLS (`fractions.Fraction`), closed-form logistic MLE on 2x2 tables, hand-computed metrics | `reference/independent/*.py` |

System A never runs in the Rust CI job. Its outputs are committed as JSON fixtures with
the producing package versions embedded, so the Rust tests are deterministic and offline.
Regenerating fixtures is an explicit, reviewed action (`just regen-reference`).

## Shared inputs

- Datasets live in `reference/fixtures/datasets/<name>.csv` with a `provenance.json`
  (source URL, license, SHA-256, row count, column types, missing-value policy).
- Row order is the file order. Every system reads the same CSV.
- Split/fold membership is produced once by `reference/fixtures/generate_splits.py`
  (seeded Python `random.Random`, documented algorithm) and written as
  `splits/<dataset>/<scheme>.json`:

```json
{ "scheme": "vfold", "v": 5, "strata": "species", "seed": 20261012,
  "fold_of_row": [0, 3, 1, ...],            // 0-based rows, fold index per row
  "initial_split": { "train": [...], "test": [...] } }
```

R converts to 1-based indices and builds `rsample::manual_rset`; Rust loads
`Folds::from_assignments`.

## Comparison levels and fixture contents

| level | fixture file                       | contents from R                                                              | tolerance |
|-------|------------------------------------|------------------------------------------------------------------------------|-----------|
| 1 preprocessing | `prep_<dataset>.json`    | per-step fitted statistics (means, sds, medians, dummy names), baked matrix for train and test | means/sds: rel 1e-12; baked values: abs 1e-12; names: exact |
| 2 metrics | `metrics_fixed_predictions.json` | metrics computed by yardstick on hand-written truth/estimate arrays (incl. undefined precision case) | abs 1e-12; `NA` must map to `None` |
| 3 estimators | `fit_<dataset>_<model>.json` | coefficients (named), fitted values, test predictions / probabilities, convergence info (`glm$iter`, deviance) | OLS coefficients: rel 1e-8 (both are least squares; condition-number dependent, recorded per dataset); logistic coefficients: rel 1e-6 initially, *measured*, tightened or justified; probabilities abs 1e-7 |
| 4 workflow | `resample_<dataset>_<model>.json` | per-fold metrics and the `collect_metrics()` mean / std_err | per-fold same as level 3 metric; mean/std_err abs 1e-10 given per-fold agreement |
| 5 python | n/a (Python calls Rust)             | Python results vs Rust results on identical arrays                            | bitwise equal |

Discrete outputs (fold membership, feature order, dummy column names, class labels)
require exact equality.

## Tolerance policy

Tolerances are per (operation, dataset) and live in
`reference/fixtures/tolerances.toml`. The file records the *justification* next to
each number (e.g. "OLS on Ames: X'X condition number 3e7, expect ~1e-9 relative"). A
test that needs a looser tolerance than recorded fails; loosening requires editing the
file in a reviewed change with a reason. There is no global default tolerance.

## Independent baselines (System X)

- OLS: exact solutions via `fractions.Fraction` normal equations on small full-rank
  problems (n ≤ 30, p ≤ 4). Both R and Rust must agree with the exact answer to
  ~1e-12 relative. This catches shared mistakes (e.g. both dropping the intercept).
- Logistic: binary predictor / binary outcome tables where the MLE is closed-form:
  `b0 = log(odds | x=0)`, `b1 = log(odds | x=1) - b0`. Also perfectly separable data,
  where both systems must *report* non-convergence rather than silently return
  coefficients.
- Metrics: hand-computed on 6-row arrays; `rsq` vs `rsq_trad` distinguished.
- Standardization: columns with known mean/sd (e.g. 1..10: mean 5.5, sd 3.0276503541).

## Classification of differences

Every difference outside tolerance is triaged into one of:
`defect` (fix), `solver` (same objective, different optimizer; document the bound),
`semantic` (different definition; document in COMPATIBILITY.md and align or expose a
switch), `undefined` (NA/None/NaN behaviour; align with yardstick).

## Datasets for v0.1

| dataset | source | rows × cols | task | why |
|---------|--------|-------------|------|-----|
| synthetic_reg_small | generated, seed 1 | 40 × 4 | regression | exact OLS baseline |
| synthetic_clf_small | generated, seed 2 | 60 × 3 | classification | closed-form logistic checks |
| iris (modeldata / R datasets) | R `datasets::iris` | 150 × 5 | binary (versicolor vs rest) | mixed numeric |
| penguins | R `modeldata::penguins` | 344 × 7 | both | missing values + categoricals |
| ames | R `modeldata::ames` | 2930 × 74 | regression | wide, many categoricals, ill-conditioned |

Larger datasets enter in Phase 5 with provenance records.

## Convergence reporting contract

Rust logistic fits report `{ converged: bool, iterations: u64, final_gradient_norm }`.
R fits report `glm$converged`, `glm$iter`. A fixture where R did not converge is
excluded from coefficient comparison and included in the "both must flag" check.
