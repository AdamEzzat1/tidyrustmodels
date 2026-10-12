# ADR-0005: Initial estimator backends

Date: 2026-10-12 · Status: accepted

## Context
Facts verified by reading crate sources on 2026-10-12 (see DEPENDENCIES.md):
- linfa-linear 0.8.1: OLS via least squares; pure-Rust path through linfa-linalg when
  the `blas` feature is off (no LAPACK needed on Windows MSVC). `fit` *asserts* on
  shape mismatch (panics).
- linfa-logistic 0.8.1: L-BFGS (argmin 0.11, m=10, More-Thuente line search). Default
  `alpha = 1.0` L2 penalty on coefficients, intercept unpenalized; objective is a *sum*
  (not a mean) so `alpha = 0` is exactly the unpenalized MLE. `alpha = 0` passes
  validation (`check_ref` rejects only negative or non-finite). Default
  `gradient_tolerance = 1e-4`, `max_iterations = 100`.
- smartcore 0.6.16: linear regression via SVD (default) or QR; logistic via its own
  L-BFGS (`g_atol = 1e-8`, `max_iter = 1000`), `alpha` default 0, **no `predict_proba`**;
  coefficients and intercept are accessible.

## Decision
- Default engine: `linfa` for both linear and logistic regression.
- Second engine: `smartcore` for both. Its logistic adapter computes probabilities from
  the fitted coefficients (sigmoid of the linear predictor) and declares
  `probabilities_derived_by_adapter = true`.
- For parity with R `glm` (IRLS, `epsilon = 1e-8` on deviance, `maxit = 25`), the linfa
  logistic adapter exposes `gradient_tolerance` and `max_iterations` and uses a
  documented tight default. Observed coefficient agreement is reported, never assumed.

## Consequences
- Two independent implementations of each v0.1 estimator, which lets the differential
  suite triangulate: R vs linfa vs smartcore vs analytic baseline.
- Neither backend ships a native IRLS; coefficient agreement with R is bounded by
  optimizer tolerance and is a measured quantity (see VERIFICATION.md).
