# Roadmap and milestone gates

Each milestone has objective acceptance criteria. A milestone is **blocked** while any
critical reviewer finding is open. "Done" claims link to VERIFICATION.md records.

## Phase 0 — Research and architecture (2026-10-12) — complete

Delivered: ARCHITECTURE.md, ADR-0001..0006, DEPENDENCIES.md, docs/specs/api-spec.md,
docs/specs/verification-design.md, BENCHMARKS.md (methodology), this file,
LIMITATIONS.md (risk register).

## Phase 1 — Core framework (2026-10-12) — complete, pending independent review

Scope: `core` data contracts, errors, `Step`/`FittedStep`, `ModelSpec`/`FittedModel`,
`Seed`, `Parallelism`; `recipes` with select/impute/normalize/dummy; `models` with
linear + logistic on `linfa` and `smartcore`.

Gate:
- [x] `cargo test --workspace` green; `cargo clippy --workspace --all-targets -D warnings` clean; `cargo fmt --check` clean.
- [x] No `unwrap`/`expect`/`panic!` reachable from public APIs with user data (remaining `expect`s guard invariants checked on the previous line; audited by grep).
- [x] Property tests: normalize mean≈0/sd≈1; dummy rows are indicator rows; imputation fills only missing values (200 cases each).
- [x] Analytic OLS baseline (exact rationals) matched by both engines to rel 1e-10.
- [x] Closed-form logistic baseline matched by both engines to rel 1e-6; separable data flagged.
- [ ] Independent reviewer sign-off recorded.

## Phase 2 — End-to-end ML functionality (2026-10-12) — complete, pending independent review

Gate:
- [x] Resampling invariants tested (disjoint, covering, balanced, stratified balance).
- [x] Serial vs parallel `fit_resamples` and `tune_grid` identical (`assert_eq!` on full results).
- [x] Serialize → deserialize → predict identical (recipe, model, workflow).
- [x] Leakage test: per-fold metrics equal a manual prep-inside-fold computation.
- [x] Examples run end to end: `cargo run --release --example penguins_workflow` and `python examples/penguins_workflow.py` (2026-10-12; both report identical coefficients, CV means and best penalty).
- [ ] Independent reviewer sign-off recorded.

## Phase 3 — Python API (2026-10-12) — complete on Windows, pending CI on Linux/macOS and review

Gate:
- [x] `maturin develop --release` builds on Windows (pyo3 0.29.3, abi3-py311); CI matrix for Linux/macOS added, not yet executed.
- [x] NumPy (C and Fortran order), dict and pandas input paths tested; exception mapping tested (`python/tests/test_parity.py`, 35 tests across 8 tasks × 3 engines).
- [x] Python vs Rust parity bitwise on shared inputs for all 6 tasks × 2 engines (coefficients, predictions, per-fold metrics).
- [ ] Independent reviewer sign-off recorded.

## Phase 4 — Differential verification against R (2026-10-12) — complete, pending independent review

Gate:
- [x] Fixtures for all six v0.1 tasks generated with pinned versions, committed (`reference/fixtures/r/`).
- [x] Levels 1–4 pass under `reference/fixtures/tolerances.toml` (611 comparisons; see VERIFICATION.md).
- [x] Every out-of-tolerance difference triaged: none occurred; three defects found during bring-up are recorded with fixes.
- [x] COMPATIBILITY.md status column populated from test results.
- [ ] Independent reviewer sign-off recorded.

## Phase 5 — Performance evaluation (2026-10-12) — first run complete, protocol gaps open

Gate:
- [x] Runner executes both systems with recorded hardware/config; `benchmarks/results/2026-10-12-win11-i7-11390H/{results.csv,env.json,report.md}` committed (270/270 trials ok).
- [x] Framework-overhead and workflow-level categories reported separately (BENCHMARKS.md).
- [x] Any optimization merged in this phase has a before/after benchmark: none merged; the run is the baseline.
- [ ] Protocol as specified: ≥10 trials, ≥30 warm iterations, 10 folds, 20-point grid, 1e6-row point, allocation counts, power plan recorded. First run used 5/10/5/5, no 1e6, no A1.
- [x] Rust peak memory captured for sub-150 ms processes (`PeakWorkingSetSize` via retained handles; run `…-b`, no missing cell).
- [x] Python wrapper measured as a third system (runs `…-b` and `…-d`, all nine tasks incl. 100k rows; bitwise-identical CV results to the Rust binary).
- [x] Trial interleaving across systems after run `…-c` showed machine-state drift (kept on record, labelled invalid for timing).
- [ ] Run on a second platform (Linux CI runner at minimum).
- [ ] Independent reviewer sign-off recorded.

## Phase 6 — v0.1.0 release — not started

Gate: docs, examples (Rust + Python), feature matrix, compatibility report, benchmark
report, limitations, install instructions, release notes. Reviewer approval.

## Phase 8 — Breadth (started 2026-10-12)

Priorities from the gap analysis, each gated on differential verification where R has
an equivalent: (a) recipe filters and level steps — **done** (zv, nzv, corr, other,
novel, unknown; VERIFICATION.md Phase 8a); (b) classification metrics — **done**
(roc_auc, pr_auc, average_precision, f_meas, sens/spec, bal_accuracy, kap, mcc, j_index,
brier_class, conf_mat, with binary/macro/macro_weighted/micro/hand_till estimators;
VERIFICATION.md Phase 8b); (c) multiclass outcomes — **done** (`MultinomRegSpec` on
three engines, ADR-0010, two reference tasks against nnet; VERIFICATION.md Phase 8c);
(d) tree models — decision tree **done** (rpart 4.1.23 port, five tasks exact against
R, ADR-0011; VERIFICATION.md Phase 8d), random forest **done** (ranger 0.18.0 rules under
the framework's RNG, verified statistically on five tasks, ADR-0011 A; VERIFICATION.md
Phase 8e), boosted tree **done** (xgboost 3.2.1 hist port incl. workflows' sparse-data
decision, five tasks against xgboost's model JSON, ADR-0011 B; VERIFICATION.md Phase 8f); (d) tree models via backends; (e) case weights — **done**
(frequency / importance kinds with tidymodels' per-package rules, native engines for
lm, glm, rpart, ranger and xgboost, weighted recipe statistics and yardstick metrics,
eleven reference tasks, ADR-0012; VERIFICATION.md Phase 8g); (f) bootstrap,
repeated/grouped CV, time-series splits — **done** (rsample 1.3.2 generators as
algorithms, rsplit semantics for `Split`/`Folds`, 17 reference tasks on shared rsets,
exact memberships for the deterministic schemes and exact balance rules on R's
permutations, ADR-0013; VERIFICATION.md Phase 8h); (g) `select_best` /
`finalize_workflow` / `last_fit` — **done** (tune 2.1.0 selection rules incl.
`select_by_one_std_err` / `select_by_pct_loss`, `Workflow::finalize`, `last_fit` on
tune's fold path, `fit_best`; four tuning tasks with shared explicit grids, ADR-0014;
VERIFICATION.md Phase 8i); (h) Arrow/Polars/sparse inputs — **done** (sparse
vectors as a column flag with recipes' propagation rules and parsnip's sparse-matrix
hand-off to xgboost, six tasks against R; Arrow interop behind the `arrow` feature;
Python `from_arrow` / `from_polars` / `from_scipy_sparse` and exports, ADR-0015;
VERIFICATION.md Phase 8j); (i) workflowsets.

## Phase 7 — Native algorithms (2026-10-12) — first two solvers complete, pending review

Only with a written justification per algorithm referencing a measured need.
Measured need on record: BENCHMARKS.md shows L-BFGS logistic at 1e5 rows is 0.76× the
speed of `glm` IRLS in steady state and 0.69× glmnet for tuning; neither backend pivots
for rank deficiency.

Gate:
- [x] `Engine::Native` with a `dqrdc2`-pivoting QR (`lm.fit`) and a `glm.fit` IRLS port (ADR-0009), no external dependency.
- [x] Differential suite on all six tasks: coefficients 3.3e-13 rel, probabilities 4e-15 abs, `converged` exact; tolerances set 30× above measured maxima.
- [x] Analytic baselines and collinearity tests pass on the native engine; R's choice of aliased column reproduced.
- [x] Before/after benchmark recorded (runs `…-e`, `…-f`); `native` made the default engine (ADR-0009).
- [ ] Independent reviewer sign-off recorded.
Candidates visible so far: an IRLS logistic solver (closer match to `glm`, deterministic
iteration count, no L-BFGS bound on parity) and a pivoted-QR least squares (general
rank deficiency with `NA`-style reporting, which neither backend provides).
