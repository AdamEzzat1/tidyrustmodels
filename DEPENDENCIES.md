# Dependency decision record

All facts below were verified on 2026-10-12 from crates.io metadata (`cargo info`,
crates.io API) and by reading the crate sources in the local Cargo registry. Re-verify
before upgrading; do not trust this table for other versions.

## Host environment used for Phase 0

| item          | value                                                    |
|---------------|----------------------------------------------------------|
| CPU           | Intel i7-11390H, 4 cores / 8 threads, 3.4 GHz            |
| RAM           | 31.8 GB                                                  |
| OS            | Windows 11 Pro 10.0.26200                                |
| Rust          | rustc 1.97.1, cargo 1.97.1, stable-x86_64-pc-windows-msvc |
| Python        | 3.11.3; project venv `.venv` with numpy 2.2.6, pandas 2.3.2, pytest 7.2.2, maturin 1.10.x, psutil 7.2.2 (benchmark RSS sampling) |
| R             | 4.4.0 (ucrt); tidymodels packages built under 4.4.3 (warning only) |
| R packages    | tidymodels 1.5.0, parsnip 1.5.0, recipes 1.3.2, rsample 1.3.2, yardstick 1.4.0, tune 2.1.0, dials 1.4.3, workflows 1.3.0, glmnet 4.1-10, arrow 23.0.1.2, jsonlite 2.0.0, bench 1.1.4 |

## Selected Rust dependencies

| crate            | version | license            | last publish | role                        | why |
|------------------|---------|--------------------|--------------|-----------------------------|-----|
| linfa            | 0.8.1   | MIT OR Apache-2.0  | 2025-12-23   | backend trait glue          | Most complete Rust ML ecosystem; `Fit`/`Predict` traits; pure-Rust linalg path |
| linfa-linear     | 0.8.1   | MIT OR Apache-2.0  | 2025-12-23   | OLS engine `linfa`          | Least squares via linfa-linalg (no LAPACK) |
| linfa-logistic   | 0.8.1   | MIT OR Apache-2.0  | 2025-12-23   | logistic engine `linfa`     | L-BFGS, `alpha=0` gives unpenalized MLE, exposes probabilities |
| smartcore        | 0.6.16  | Apache-2.0         | 2026-10-12   | second engine               | Independent implementation (SVD/QR OLS; L-BFGS logistic), MSRV 1.85 |
| rand / rand_chacha | 0.10.3 / 0.10.0 | MIT OR Apache-2.0 | 2026 | seeded RNG             | ChaCha8 is portable and reproducible across platforms |
| rayon            | 1.12.0  | MIT OR Apache-2.0  | 2026         | bounded parallel resampling | Scoped pools; deterministic join order |
| serde + serde_json | 1.0.229 / 1.0.154 | MIT OR Apache-2.0 | 2026 | versioned serialization |  |
| thiserror        | 2.0.21  | MIT OR Apache-2.0  | 2026         | structured errors           |  |
| proptest         | 1.12.0  | MIT OR Apache-2.0  | 2026         | property tests (dev)        | MSRV 1.88 |
| criterion        | 0.8.2   | Apache-2.0 OR MIT  | 2026         | micro-benchmarks (dev)      |  |
| pyo3             | 0.29.3  | MIT OR Apache-2.0  | 2026-09-30   | Python bindings             | `Bound<'py, T>` API, `Python::attach` |
| numpy            | 0.29.0  | BSD-2-Clause       | 2026-06-13   | NumPy interop               | accepts ndarray 0.15 to 0.17 |

All licenses are compatible with the project license (MIT OR Apache-2.0).

## Rejected or deferred

| crate            | version | reason |
|------------------|---------|--------|
| polars           | 0.55.2  | Deferred to optional interop feature (ADR-0003): build size, monthly breaking majors |
| arrow            | 60.0.0  | Same as polars; optional feature later |
| nalgebra         | 0.35.0  | Not needed: backends own their linear algebra |
| faer             | 0.24.4  | Candidate for Phase 7 native solvers only |
| ndarray-linalg   | 0.18.1  | Requires LAPACK (openblas/MKL) build; pure-Rust path suffices for v0.1 |
| linfa-preprocessing | 0.8.1 | Preprocessing is the framework's differentiator and must match recipes semantics exactly (ddof, missing handling, naming); implemented natively |
| approx           | 0.6.0-rc2 | Pre-release; use 0.5.1 in dev-deps |

## Verified backend facts that constrain the design

- linfa-linear `fit` uses `assert_eq!` on `y.len() == n_samples` (panic): adapters
  validate shapes first.
- linfa-logistic objective: `-sum(log sigma(y_i z_i)) + alpha/2 * ||w||^2`, intercept
  excluded from the penalty; probabilities are for the *larger* class under `Ord`.
- linfa-logistic defaults: `alpha = 1.0`, `gradient_tolerance = 1e-4`, `max_iterations = 100`.
  `alpha = 0` is accepted.
- smartcore `LogisticRegression` has no `predict_proba`; `coefficients()` and
  `intercept()` are public, so the adapter computes probabilities.
- smartcore `LinearRegressionParameters::default()` uses SVD; QR available.
- ndarray versions: linfa 0.16, smartcore 0.17, numpy crate 0.15 to 0.17.

## R reference facts that constrain the design

- `yardstick::rsq` is the squared Pearson correlation; `rsq_trad` is `1 - SSE/SST`.
- `yardstick::rmse = sqrt(mean((truth - estimate)^2))`, `mae = mean(|truth - estimate|)`.
- `event_level` defaults to `"first"` for precision, recall, `mn_log_loss`.
- `recipes::step_normalize` uses `stats::var` (n-1 denominator) and `colMeans`,
  both with `na.rm = TRUE` by default.
- `recipes::step_dummy` defaults `one_hot = FALSE` (first level dropped) and names
  columns `paste(var, make.names(level), sep = "_")`.
- `recipes::step_impute_median` uses `stats::median` (R type 7 interpolation).
- `rsample::vfold_cv` assigns folds with `sample(rep(1:v, length.out = n))`;
  stratified folds repeat that within each stratum.
- `stats::glm` control: `epsilon = 1e-8` (relative deviance change), `maxit = 25`.
