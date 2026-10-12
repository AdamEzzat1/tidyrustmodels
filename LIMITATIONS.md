# Limitations and risk register

## Known limitations (v0.1 scope)

- Estimators: linear regression and binary logistic regression only. No multiclass,
  no penalized paths (`glmnet`), no trees.
- Precision: `f64` only.
- Preprocessing: select, mean/median imputation, normalize, dummy. No `step_other`,
  `step_interact`, `step_pca`, splines.
- Resampling: initial split, v-fold (optionally stratified). No bootstraps, repeats,
  grouped or time-based resampling.
- Metrics: rmse, mae, rsq, rsq_trad, accuracy, precision, recall, mn_log_loss. No
  case weights, no ROC AUC.
- Python: NumPy and pandas input. Arrow/Polars input deferred.
- Unseen categorical levels at `bake` time are an error, not NA (stricter than recipes).
- Rank deficiency: only identically-zero columns and (with an intercept) constant
  columns are handled by the framework. General collinearity is passed to the backend;
  linfa then returns a backend error, smartcore returns a minimum-norm solution. R would
  report `NA` for the aliased coefficients in both cases.
- smartcore engine: always fits an intercept; optimizer settings are fixed.
- `make_names` port covers ASCII; non-ASCII level names may differ from R's
  locale-dependent behaviour.

## Findings from verification (facts, not risks)

- linfa-logistic 0.8.1 picks the positive class by **count**, not by label order, in
  contradiction to its documentation. The adapter normalises this (VERIFICATION.md).
- linfa-linear 0.8.1 panics on shape mismatch and errors on singular systems; the
  framework validates and aliases before calling it.
- linfa-logistic does not expose L-BFGS termination status; convergence is verified by
  the framework's own gradient check.

## Risk register

| id | risk | likelihood | impact | status / mitigation |
|----|------|-----------|--------|------------|
| R1 | Logistic coefficients differ from R beyond tolerance (L-BFGS vs IRLS) | was high | medium | **measured**: ≤1.6e-8 relative on all tasks; bound set to 1e-7. Residual: larger n or ill-conditioned problems untested |
| R2 | Ill-conditioned OLS amplifies solver differences | medium | medium | **measured** on Ames (cond 6.5e3 after normalize): 2.6e-13 relative. Unnormalised wide designs untested |
| R3 | Design-matrix copies into backend array types dominate fit time for large n | medium | low–medium | Measure in Phase 5 (D1); zero-copy only with evidence |
| R4 | linfa panics/erors on malformed or singular input; ecosystem last released 2025-12 | medium | medium | Validation and aliasing in the framework; smartcore as second engine |
| R5 | smartcore API churn in patch versions (0.6.16 published 2026-10-12) | medium | low | Exact pin; adapter is ~80 lines |
| R6 | pyo3 0.29 / numpy 0.29 API drift vs older examples online | medium | low | Build from registry README examples; CI builds bindings |
| R7 | R 4.4.0 with packages built under 4.4.3 | low | low | Versions recorded in every fixture |
| R8 | Benchmarks mistaken for equivalent-algorithm comparisons (IRLS vs L-BFGS) | high | high | Category column in every result row; reviewer checks wording |
| R9 | Rayon oversubscription when backends are also threaded | low | medium | Scoped pool; backends are single-threaded |
| R10 | Windows peak-RSS sampling misses short-lived peaks | medium | medium | Use `PeakWorkingSetSize`, which the OS tracks exactly |
| R11 | tidymodels semantics misread | was medium | high | **mitigated**: every row in COMPATIBILITY.md marked "tested" is backed by a fixture comparison |
| R12 | Fixtures drift from the committed datasets | low | medium | `checksums.json`; regeneration is a reviewed change |
