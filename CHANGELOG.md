# Changelog

## 0.1.0-dev (unreleased)

First vertical slice. Everything listed here is backed by tests recorded in
VERIFICATION.md; nothing is claimed beyond those records.

### Added
- Core contracts: `Table`/`Column` with explicit missing-value masks, `Schema` with roles,
  column-major `DesignMatrix`, `Outcome`/`Prediction`, structured `Error`, `Step`/`ModelSpec`
  lifecycle traits, explicit `Seed` and bounded `Parallelism`.
- Recipes: `StepSelect`, `StepImputeMean`, `StepImputeMedian`, `StepNormalize`, `StepDummy`
  (R `make.names` port, recipes column order), prep-on-training/bake-anywhere, versioned JSON.
- Models: `LinearRegSpec` and `LogisticRegSpec` on two engines (`linfa` 0.8.1 default,
  `smartcore` 0.6.16); framework-side prediction from coefficients; aliased-column handling;
  backend-independent gradient convergence check and separation warning.
- Resampling: `initial_split`, `vfold_cv` (optionally stratified), `Folds::from_assignments`,
  numeric strata binning.
- Metrics: rmse, mae, rsq (squared correlation, as yardstick), rsq_trad, accuracy, precision,
  recall, mn_log_loss (eps-clamped, as yardstick); `MetricSet`.
- Workflows: `fit`, `predict`, `fit_resamples` with per-fold prep and serial-identical
  parallel execution; versioned JSON round trip.
- Tuning: `grid_regular`, `grid_random`, `tune_grid`, `show_best`.
- Python bindings (PyO3 0.29, maturin, abi3 ≥ 3.11) with NumPy/pandas input and an exception
  hierarchy; bitwise parity with the Rust API.
- Reference harness against R tidymodels 1.5.0 with committed fixtures, shared split ids,
  per-quantity tolerances, and a differential test (611 comparisons).
- Benchmark harness (Rust and R drivers, orchestrator with peak-RSS sampling, report) and
  one recorded run (`benchmarks/results/2026-10-12-win11-i7-11390H`); see BENCHMARKS.md
  for what it shows, including where tidyrustmodels is slower than R.

### Known limitations
See LIMITATIONS.md. Highlights: binary classification only; general collinearity not
aliased; smartcore engine always fits an intercept and ignores optimizer settings;
unseen categorical levels at bake time are an error.
