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

## Findings from benchmarking (run `2026-10-12-win11-i7-11390H`, BENCHMARKS.md)

- OLS workflows: 8–240× faster per workflow fit (prep+bake+fit) and 5–57× faster end-to-end than R tidymodels
  on the recorded laptop, 8–50× less memory. Both sides used unoptimised single-threaded
  linear algebra (R's bundled `Rblas.dll`; pure-Rust linfa/smartcore).
- Logistic workflows at 1e5 rows: the L-BFGS backends are **not faster** than R's IRLS
  `glm` in steady state (Rust warm model-only fit 770 ms vs R warm workflow fit 587 ms), tuning is 0.69× the speed of
  glmnet, and single-thread end-to-end wall is slower than R (37.7 s vs 35.0 s). This is
  the measured need for an IRLS solver (ROADMAP.md Phase 7).
- smartcore's SVD least squares is 6–16× slower than linfa's at n ≥ 2930; it remains
  the second engine for cross-checking, not for speed.
- Rust 4-thread scaling on 4 cores: 2.1–2.5× for tuning, 1.4–2.1× for CV at n ≥ 2930,
  a slowdown on 333-row tasks. R's `multisession` 4-worker configuration was slower than
  sequential on every task because worker start-up (8–13 s) dominates.
- The 1 ms RSS sampler did not observe Rust processes that exit in under ~150 ms (81 of
  180 rows) in run A; fixed (exact `PeakWorkingSetSize` via retained handles) and
  re-measured in run B: Rust binary 5–13 MB, Python wrapper 63–69 MB, R 361–410 MB.
- Python wrapper (run B, six tasks up to 10k rows): faster than R tidymodels on every
  phase, cold start 4–10× faster (0.6 s import vs 2.7–3.2 s tidymodels attach), and
  within ~50 µs per call of the Rust binary. Not measured at 100k rows, where the Rust
  binary itself is not faster than R for logistic regression.
- R's model-only `glm` fit is within 1.3× of L-BFGS already at 10k rows (run B).

## Risk register

| id | risk | likelihood | impact | status / mitigation |
|----|------|-----------|--------|------------|
| R1 | Logistic coefficients differ from R beyond tolerance (L-BFGS vs IRLS) | was high | medium | **measured**: ≤1.6e-8 relative on all tasks; bound set to 1e-7. Residual: larger n or ill-conditioned problems untested |
| R2 | Ill-conditioned OLS amplifies solver differences | medium | medium | **measured** on Ames (cond 6.5e3 after normalize): 2.6e-13 relative. Unnormalised wide designs untested |
| R3 | Design-matrix copies into backend array types dominate fit time for large n | was medium | low | **measured**: at 1e5×14, prep+bake 36 ms and load 170 ms vs fit 20 ms (OLS) / 810 ms (logistic); the copy is not the bottleneck. Zero-copy deferred |
| R4 | linfa panics/erors on malformed or singular input; ecosystem last released 2025-12 | medium | medium | Validation and aliasing in the framework; smartcore as second engine |
| R5 | smartcore API churn in patch versions (0.6.16 published 2026-10-12) | medium | low | Exact pin; adapter is ~80 lines |
| R6 | pyo3 0.29 / numpy 0.29 API drift vs older examples online | medium | low | Build from registry README examples; CI builds bindings |
| R7 | R 4.4.0 with packages built under 4.4.3 | low | low | Versions recorded in every fixture |
| R8 | Benchmarks mistaken for equivalent-algorithm comparisons (IRLS vs L-BFGS) | high | high | Category column in every result row; BENCHMARKS.md keeps the two categories in separate tables and states where Rust is slower |
| R9 | Rayon oversubscription when backends are also threaded | low | medium | Scoped pool; backends are single-threaded |
| R10 | Windows peak-RSS sampling misses short-lived peaks | **occurred, fixed** | low | 81/180 Rust rows not captured in run A. Harness now retains a handle per tree process and reads `PeakWorkingSetSize` after exit; run B has no missing cell. POSIX path (`ru_maxrss` wrapper) untested |
| R13 | L-BFGS logistic solver loses to IRLS/glmnet at large n | **measured** | medium | 1e5 rows: Rust model-only warm fit is 0.76× the speed of R's warm workflow fit, tuning 0.69× glmnet. Mitigation is a native IRLS solver (Phase 7), gated on a before/after benchmark |
| R11 | tidymodels semantics misread | was medium | high | **mitigated**: every row in COMPATIBILITY.md marked "tested" is backed by a fixture comparison |
| R12 | Fixtures drift from the committed datasets | low | medium | `checksums.json`; regeneration is a reviewed change |
