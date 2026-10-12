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

## Risk register (Phase 0)

| id | risk | likelihood | impact | mitigation |
|----|------|-----------|--------|------------|
| R1 | Logistic coefficients differ from R beyond tolerance because L-BFGS stops at a gradient tolerance while IRLS converges on deviance | high | medium | Expose and tighten `gradient_tolerance`; measure on every dataset; Phase 7 candidate native IRLS |
| R2 | Ill-conditioned OLS (Ames with dummies) amplifies solver differences (linfa-linalg SVD/QR vs LINPACK QR) | medium | medium | Per-dataset tolerances tied to measured condition numbers; compare against exact rational baseline on small problems |
| R3 | Design-matrix copies into two different ndarray versions dominate fit time for large n | medium | low–medium | Measure D1 explicitly; zero-copy only with evidence |
| R4 | linfa-linear panics (assert) on malformed shapes; linfa ecosystem last release 2025-12 | medium | medium | Validate before calling; smartcore as second engine; adapters isolated |
| R5 | smartcore publishes frequently (0.6.16 on 2026-10-12); API churn in patch versions | medium | low | Pin exact version; adapter is small |
| R6 | pyo3 0.29 / numpy 0.29 API drift vs older examples online | medium | low | Build from the crate README examples verified in the registry; CI builds bindings |
| R7 | R 4.4.0 with packages built under 4.4.3 could behave differently from a clean 4.4.3 install | low | low | Record versions in every fixture; CI reference job uses a pinned R |
| R8 | Benchmarks mistaken for equivalent-algorithm comparisons (IRLS vs L-BFGS) | high | high | Category column in every result row; reviewer checks wording |
| R9 | Rayon oversubscription when backends are also threaded | low (backends are single-threaded) | medium | Scoped pool with explicit thread count; BLAS threads pinned in R |
| R10 | Windows peak-RSS sampling misses short-lived peaks | medium | medium | Sample at 1 ms; cross-check with `PeakWorkingSetSize` which the OS tracks exactly |
| R11 | tidymodels semantics misread (e.g. `rsq` = squared correlation) | medium | high | Every semantic claim in COMPATIBILITY.md cites the R function body inspected or a fixture test |
