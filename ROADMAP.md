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
- [ ] Example `examples/penguins_workflow.rs` runs end to end (facade doctest exists; standalone example pending).
- [ ] Independent reviewer sign-off recorded.

## Phase 3 — Python API — in progress

Gate:
- [ ] `maturin develop` builds on Windows (CI: also Linux, macOS).
- [ ] NumPy, pandas input paths tested; exception mapping tested.
- [ ] Python vs Rust parity bitwise on shared arrays.

## Phase 4 — Differential verification against R (2026-10-12) — complete, pending independent review

Gate:
- [x] Fixtures for all six v0.1 tasks generated with pinned versions, committed (`reference/fixtures/r/`).
- [x] Levels 1–4 pass under `reference/fixtures/tolerances.toml` (611 comparisons; see VERIFICATION.md).
- [x] Every out-of-tolerance difference triaged: none occurred; three defects found during bring-up are recorded with fixes.
- [x] COMPATIBILITY.md status column populated from test results.
- [ ] Independent reviewer sign-off recorded.

## Phase 5 — Performance evaluation — not started

Gate:
- [ ] Runner executes both systems with recorded hardware/config; results.csv + report.md committed under a run id.
- [ ] Framework-overhead and workflow-level categories reported separately.
- [ ] Any optimization merged in this phase has a before/after benchmark and unchanged differential results.

## Phase 6 — v0.1.0 release — not started

Gate: docs, examples (Rust + Python), feature matrix, compatibility report, benchmark
report, limitations, install instructions, release notes. Reviewer approval.

## Phase 7 — Native algorithms — not started

Only with a written justification per algorithm referencing a measured need.
Candidates visible so far: an IRLS logistic solver (closer match to `glm`, deterministic
iteration count, no L-BFGS bound on parity) and a pivoted-QR least squares (general
rank deficiency with `NA`-style reporting, which neither backend provides).
