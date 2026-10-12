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
- [x] NumPy (C and Fortran order), dict and pandas input paths tested; exception mapping tested (`python/tests/test_parity.py`, 21 tests).
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
- [ ] Rust peak memory captured for sub-150 ms processes (`PeakWorkingSetSize` via retained handle).
- [ ] Run on a second platform (Linux CI runner at minimum).
- [ ] Independent reviewer sign-off recorded.

## Phase 6 — v0.1.0 release — not started

Gate: docs, examples (Rust + Python), feature matrix, compatibility report, benchmark
report, limitations, install instructions, release notes. Reviewer approval.

## Phase 7 — Native algorithms — not started

Only with a written justification per algorithm referencing a measured need.
Measured need on record: BENCHMARKS.md shows L-BFGS logistic at 1e5 rows is 0.76× the
speed of `glm` IRLS in steady state and 0.69× glmnet for tuning.
Candidates visible so far: an IRLS logistic solver (closer match to `glm`, deterministic
iteration count, no L-BFGS bound on parity) and a pivoted-QR least squares (general
rank deficiency with `NA`-style reporting, which neither backend provides).
