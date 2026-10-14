# tidyrustmodels

A Rust-native machine learning framework with tidymodels semantics: recipes that prep on
training data and bake anywhere, estimator specifications that are independent of the
backend that fits them, leakage-safe resampling, yardstick-compatible metrics, and
tuning. Python bindings are an optional interface, not a requirement.

Status: **0.1.0 release candidate** (`docs/releases/0.1.0.md`); the tag waits on an
independent review and on the Linux / macOS CI runs (ROADMAP.md Phase 6). Nothing in
this repository claims parity with or performance relative to R tidymodels unless
VERIFICATION.md or BENCHMARKS.md records the measurement. `docs/FEATURES.md` is the
feature matrix with the verification status of every row.

What is verified today (VERIFICATION.md):

- Preprocessing, OLS, metrics (including yardstick's ROC/PR, F, kappa, MCC and Brier
  families with their multiclass estimators) and resampling match R tidymodels 1.5.0 to
  floating-point round-off on eight reference tasks; multinomial regression matches
  `nnet` to its own convergence residual (3.5e-7) on two more; rpart decision trees are
  reproduced node for node on five tasks; ranger-style random forests agree with R
  within 4 standard errors over 10 seeds on five more; xgboost boosters are
  reproduced tree for tree, including tidymodels' sparse-data path on Ames (bit-exact
  on Windows, where the fixtures were made; within the C runtime's float32 `expf` ulp
  on Linux / macOS, ADR-0017); logistic regression matches to ~1e-8 relative
  (same objective, different optimizer).
- All three engines (`linfa`, `smartcore`, `native`) reproduce exact analytic baselines
  that are independent of R. The default `native` engine ports R's own `lm.fit` and
  `glm.fit` algorithms and agrees with R to round-off (3e-13 relative), including which
  coefficient R reports as `NA` under collinearity; it fits logistic models 3–4× faster
  than the L-BFGS engines (BENCHMARKS.md run F).
- The Python bindings produce bitwise-identical results to the Rust API.

What is measured today (BENCHMARKS.md, one Windows laptop, protocol runs G–K of
2026-10-13/14: 10 cold trials, 30 warm iterations, 10 folds, 20-point or 4 × 5 grids,
R single-threaded on reference BLAS, ratios = R median / native median at 1 thread):
per cold workflow fit the native ports are 3.4–930× faster than R tidymodels for
linear and logistic models (OLS: 930× at 342 rows, 12× at 100k, 6.1× at 1e6; logistic:
3.4× at 1e6), 14–270× for rpart-style trees (13–450× on the warm, in-process fit),
1.9–4.4× for xgboost-style boosting (2.0–4.6× warm), and on par with ranger for random
forests (0.86–1.04×); 10-fold CV 3.7–2200× (linear / logistic), 13–470× (trees),
2.6–7.8× (boosting), 0.98–1.13× (forests: R is faster on one of the two tasks); tuning
grids 13–425× for trees, 4.1× for boosting, 1.1–360× for logistic against glmnet (a
different algorithm; equal at 100k rows); workflow sets 14–270×. R's peak memory is
3.5–70× the Rust binary's at 1 thread (3.1–70× over all cells) and 2.4–5.1× the Python
wrapper's. The Python wrapper reads 0.95–1.03× the Rust binary's time on the 1e6-row
and forest phases; on the other phases above 100 ms it reads 0.57–1.2× the binary's time
(usually faster on sub-second phases; unexplained, see BENCHMARKS.md). Read the
deviations and the disturbance log there before quoting any number; no result
transfers to other hardware. `site/scripts/check_ratios.py` recomputes every range in
this paragraph from the per-trial CSVs.

## Quick start (Rust)

```rust
use tidyrustmodels::prelude::*;

let data = tidyrustmodels::io::read_csv("reference/fixtures/datasets/penguins_sex.csv")?;
let recipe = Recipe::new("sex")
    .step(StepImputeMedian::new(Selector::AllNumericPredictors))
    .step(StepNormalize::new(Selector::AllNumericPredictors))
    .step(StepDummy::new(Selector::AllNominalPredictors));
let wf = Workflow::new(recipe, LogisticRegSpec::new());
// case weights as `add_case_weights()`: .case_weights("w", WeightKind::Frequency)
let folds = vfold_cv(data.nrows(), 5, None, Seed(7))?;
// or bootstraps(n, 25, None, seed), group_vfold_cv(&groups, Some(4), 1, GroupBalance::Groups, seed),
// rolling_origin(n, 60, 12, true, 11, 0) ... (rsample 1.3.2 semantics, ADR-0013)
let cv = wf.fit_resamples(&data, &folds, &MetricSet::classification_defaults(), Parallelism::threads(4))?;
// tuning: tune_grid(..) then .select_best(m) / .select_by_one_std_err(m, &[SortKey::desc("penalty")]),
// wf.finalize(&sel.params).last_fit(&data, &split, &metrics), fit_best(..) (ADR-0014)
// workflow sets: workflow_set(preproc, models, true)?.workflow_map(..); .rank_results(None, true); .fit_best(&data, None) (ADR-0016)
```

See `crates/tidyrustmodels/examples/` for full examples: `penguins_workflow`
(classification end to end), `ames_tuning` (tree tuning, selection rules, `last_fit`,
`fit_best`) and `ames_workflow_sets` (preprocessors x models, ranking, refit), each
run with `cargo run --release --example <name>`; `examples/` holds the Python versions.

## Quick start (Python)

```python
import tidyrustmodels as trm
data = trm.from_pandas(df)
wf = trm.Workflow(trm.Recipe("sex").step_normalize("all_numeric_predictors")
                                    .step_dummy("all_nominal_predictors"),
                  trm.LogisticReg())
# also trm.from_arrow(pa_table), trm.from_polars(pl_df), trm.from_scipy_sparse(csr, names)
folds = trm.vfold_cv(data.nrows, 5, seed=7, strata=list(df["sex"]))
cv = wf.fit_resamples(data, folds, ["accuracy", "mn_log_loss"], threads=4)
```

## Installing

Rust, from the repository (the crates are not yet published to crates.io):

```toml
[dependencies]
tidyrustmodels = { git = "https://github.com/AdamEzzat1/tidyrustmodels", package = "tidyrustmodels" }
# features = ["arrow"] adds RecordBatch <-> Table conversion
```

Python ≥ 3.11, from source (wheels are built by `.github/workflows/release.yml` but not
yet published):

```
python -m venv .venv && .venv/Scripts/activate      # or source .venv/bin/activate
pip install maturin numpy
maturin develop --release                             # or: pip install .
```

Optional extras: `pandas`, `pyarrow`, `polars`, `scipy` (`from_pandas`, `from_arrow`,
`from_polars`, `from_scipy_sparse`). R is needed only to regenerate reference fixtures
and to run the benchmarks; the tests use committed fixtures.

## Building and testing

```
cargo build --workspace --release
cargo test --workspace            # includes the differential suite against committed R fixtures
cargo doc --workspace --no-deps --open   # rendered Rust API documentation
pip install pandas pytest && pytest python/tests
```

## Documents

| file | purpose |
|------|---------|
| ARCHITECTURE.md | crate boundaries, contracts, lifecycle |
| DEPENDENCIES.md | verified dependency facts and decisions |
| COMPATIBILITY.md | tidymodels semantic/feature compatibility, tested vs intended |
| VERIFICATION.md | tests executed and numerical comparison records |
| BENCHMARKS.md | methodology and measured results |
| LIMITATIONS.md | unsupported features, findings, risk register |
| ROADMAP.md | milestones and gates |
| docs/FEATURES.md | supported feature matrix with verification status |
| docs/releases/ | release notes |
| docs/adr/ | architecture decision records |
| docs/specs/ | API spec, verification harness design |
| docs/reviews/ | independent review records |
| reference/ | R reference harness, fixtures, independent baselines |
| benchmarks/ | benchmark drivers and results |

## Layout

```
crates/            Rust workspace crates (core, recipes, models, resample, metrics, workflows, tune, workflowsets, facade)
bindings/python/   PyO3 extension crate
python/            Python package and tests
reference/         R scripts, datasets, shared splits, fixtures, tolerances
benchmarks/        Rust/R drivers, orchestrator, results
examples/          Rust and Python examples
```

License: MIT OR Apache-2.0, at your option (`LICENSE-MIT`, `LICENSE-APACHE`).
