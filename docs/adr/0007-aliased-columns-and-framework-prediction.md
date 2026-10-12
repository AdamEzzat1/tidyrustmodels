# ADR-0007: Aliased columns are handled by the framework; prediction never calls a backend

Date: 2026-10-12 · Status: accepted

## Context
During Phase 4 the Ames 5-fold task produced one fold whose analysis rows lacked a
`Neighborhood` level, so its indicator column was identically zero. R's `lm` pivots
(`dqrls`), reports an `NA` coefficient, and predicts with the remaining terms
(warning "prediction from rank-deficient fit"). linfa-linear 0.8.1 returned
`Matrix is non-invertible`; smartcore's SVD would have returned a minimum-norm solution.
Separately, the Phase 1 closed-form baseline showed linfa-logistic models the class
with the larger *count*, contradicting its documentation.

## Decision
1. Before calling any backend, `linear_reg` and `logistic_reg` drop columns that are
   identically zero, and (when an intercept is fitted) columns that are constant. The
   dropped names go into `FitDiagnostics::aliased`; their coefficient is reported as `0`.
   Predictions are then identical to R's. General collinearity is *not* detected; it is
   left to the backend and documented in LIMITATIONS.md.
2. Backends are used for fitting only. `FittedLinear` / `FittedLogistic` hold plain
   coefficients and compute every prediction and probability themselves. Adapters
   normalise backend conventions (sign, class order) at the fitting boundary.

## Consequences
- Prediction semantics are identical across engines and after deserialization, with no
  backend crate involved.
- `coefficients()` reports `0` where R reports `NA`; the differential test checks that
  any R `NA` coefficient is in the aliased list.
- A pivoted-QR native solver (Phase 7 candidate) would allow `NA`-style reporting for
  general rank deficiency.
