# ADR-0009: A `native` engine that ports R's solvers

Date: 2026-10-12 · Status: accepted

## Context
Phase 7 of the roadmap allowed native algorithms only against a measured need. The
benchmarks (BENCHMARKS.md, runs A and D) and the differential suite recorded two:

1. Logistic regression at 1e5 rows: both L-BFGS backends are slower than R's IRLS
   `glm` in steady state (0.76×) and than glmnet for tuning (0.69×), and parity is
   bounded at ~1e-8 relative by the optimizer difference rather than by arithmetic.
2. Rank deficiency: neither backend pivots. linfa fails on a singular system, smartcore
   returns one of many solutions, and the framework's aliasing rule only covers constant
   and zero columns (ADR-0007). R reports `NA` for a specific column chosen by LINPACK
   `dqrdc2`'s limited pivoting, and users compare coefficient tables.

## Decision
Add `Engine::Native`, always compiled in, with no external numerical dependency:

- `linear_reg`: Householder QR with the `dqrdc2` pivoting rule (`tol = 1e-7`), ported
  from R's `src/appl/dqrdc2.f` and the `dqrls`/`dqrsl` solve. Aliased columns are the
  ones R would report as `NA`; their coefficient is reported as `0` and their name in
  `FitDiagnostics::aliased`, keeping the ADR-0007 contract.
- `logistic_reg`: IRLS exactly as `glm.fit` (stats 4.4.0): the same start values, the
  C-level `logit` link with its `±30` clamp, weighted least squares through the pivoted
  QR with `tol = min(1e-7, epsilon/1000)`, the relative-deviance stopping rule
  (`epsilon = 1e-8`, `maxit = 25`), step halving, and `glm.fit`'s warning wording.
  `converged`, `iterations` and `deviance` therefore mean what `glm$converged`,
  `glm$iter` and `glm$deviance` mean. An unscaled L2 penalty on the slopes is supported
  by row augmentation so tuning still works; it is not glmnet's lambda.
- The framework still computes every prediction from the coefficients (ADR-0005); the
  engine fits only.
- `LogisticRegSpec` gains `irls_epsilon` and `irls_max_iterations`, used by this engine
  only; `max_iterations`/`gradient_tolerance` keep driving the L-BFGS engines.

Default engine: `native`, decided on the evidence in the section below.

## Consequences
- Differential results against R move from ~1e-8 (optimizer-bound) to round-off:
  coefficients ≤ 3.3e-13 relative, probabilities ≤ 4e-15 absolute, `converged` exact
  on every reference task (VERIFICATION.md Phase 7). Native-engine tolerances in
  `reference/fixtures/tolerances.toml` are set 30× above the measured maxima.
- The port is ~400 lines of plain loops with no BLAS; it is readable against the R
  sources it cites, which is the review strategy for it. It is not vectorised, and
  `dnrm2`'s scaled norm is replaced by a plain sum of squares (ulp-level difference,
  measured not assumed).
- `linfa` and `smartcore` remain as independent cross-checks of the objective; the
  analytic baselines run on all three engines.

## Default-engine decision (2026-10-12)
`Engine::default()` and the Python `engine=` default are `native`. Evidence (BENCHMARKS.md
run `…-f`, Rust binary, 1 thread, medians of 5 cold trials, machine idle):

| task | phase | linfa | native | ratio |
|---|---|---|---|---|
| synth_100000_clf | model fit | 585 ms | 145 ms | 4.0× |
| synth_100000_clf | 5-fold CV | 4.62 s | 1.44 s | 3.2× |
| synth_100000_clf | tune 5 × 5 | 28.2 s | 7.4 s | 3.8× |
| synth_10000_clf | model fit / CV / tune | 42 ms / 268 ms / 1.12 s | 13 ms / 99 ms / 0.47 s | 3.2× / 2.7× / 2.4× |
| penguins_clf | model fit / CV / tune | 0.74 / 4.4 / 19 ms | 0.25 / 1.4 / 6.6 ms | 2.9× / 3.1× / 2.9× |
| synth_100000_reg | model fit / CV | 22.8 ms / 359 ms | 22.0 ms / 361 ms | 1.0× |
| ames_reg | model fit / CV | 1.35 ms / 16.3 ms | 2.06 ms / 19.1 ms | 0.65× / 0.85× |
| penguins_reg | model fit / CV | 0.057 / 0.50 ms | 0.043 / 0.53 ms | 1.3× / 0.95× |

Peak memory is the same within 10%. Against R at 100k rows (run `…-d`, same machine):
native's 145 ms model fit versus `glm`'s 708 ms cold / 526 ms warm, and 7.4 s tuning
versus glmnet's 18.2 s, so the solver-bound case recorded in run A no longer holds.
The one regression is Ames OLS (34 columns): the unblocked, unvectorised Householder
loop is 0.7 ms slower per fit than linfa's least squares. Accepted because the native
engine's results are exactly R's and the absolute cost is small; a blocked QR is the
obvious follow-up if wide designs matter.

Run `…-e` (same configuration) is kept on record but was disturbed by concurrent R
and Python processes started from this session; its within-run ratios agree with
run `…-f` (3.7–6× on logistic, ~1× on OLS).
