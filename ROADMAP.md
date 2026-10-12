# Roadmap and milestone gates

Each milestone has objective acceptance criteria. A milestone is **blocked** while any
critical reviewer finding is open. "Done" claims link to VERIFICATION.md records.

## Phase 0 — Research and architecture (2026-10-12) — complete

Delivered: ARCHITECTURE.md, ADR-0001..0006, DEPENDENCIES.md, docs/specs/api-spec.md,
docs/specs/verification-design.md, BENCHMARKS.md (methodology), this file,
LIMITATIONS.md (risk register).

## Phase 1 — Core framework

Scope: `core` data contracts, errors, `Step`/`FittedStep`, `ModelSpec`/`FittedModel`,
`Seed`, `Parallelism`; `recipes` with select/impute/normalize/dummy; `models` with
linear + logistic on `linfa` and `smartcore`.

Gate:
- [ ] `cargo test --workspace` green; `cargo clippy --workspace --all-targets -D warnings` clean; `cargo fmt --check` clean.
- [ ] No `unwrap`/`expect`/`panic!` reachable from public APIs with user data (grep-audited, documented exceptions).
- [ ] Property tests: normalize mean≈0/sd≈1 on training data; dummy columns sum to 1 (one-hot) or ≤1 (drop-first); impute leaves no missing in selected columns.
- [ ] Analytic OLS baseline (exact rationals) matched by both engines to rel 1e-10.
- [ ] Closed-form logistic baseline matched by both engines to rel 1e-6; separable data reports non-convergence.
- [ ] Independent reviewer sign-off recorded.

## Phase 2 — End-to-end ML functionality

Scope: `resample` (initial_split, vfold_cv, stratified, from_assignments), `metrics`
(rmse, mae, rsq, rsq_trad, accuracy, precision, recall, mn_log_loss), `workflows`
(fit, predict, fit_resamples with per-fold prep), `tune` (grid_regular, grid_random,
tune_grid), serialization round trip.

Gate:
- [ ] Resampling invariants property-tested (disjoint, covering, balanced, stratified balance).
- [ ] Serial vs parallel `fit_resamples` bit-identical.
- [ ] Serialize → deserialize → predict identical to original predictions.
- [ ] Leakage test: a step whose statistics would differ if test rows were visible produces the training-only values inside every fold.
- [ ] Example `examples/penguins_workflow.rs` runs end to end.

## Phase 3 — Python API

Gate:
- [ ] `maturin develop` builds on Windows (CI: also Linux, macOS).
- [ ] NumPy, pandas input paths tested; exception mapping tested.
- [ ] Python vs Rust parity bitwise on shared arrays.

## Phase 4 — Differential verification against R

Gate:
- [ ] Fixtures for all five v0.1 datasets generated with pinned versions, committed.
- [ ] Levels 1–4 pass under `reference/fixtures/tolerances.toml` with justifications.
- [ ] Every out-of-tolerance difference triaged (defect / solver / semantic / undefined) in VERIFICATION.md.
- [ ] COMPATIBILITY.md feature matrix populated from test results, not intent.

## Phase 5 — Performance evaluation

Gate:
- [ ] Runner executes both systems with recorded hardware/config; results.csv + report.md committed under a run id.
- [ ] Framework-overhead and workflow-level categories reported separately.
- [ ] Any optimization merged in this phase has a before/after benchmark and unchanged differential results.

## Phase 6 — v0.1.0 release

Gate: docs, examples (Rust + Python), feature matrix, compatibility report, benchmark
report, limitations, install instructions, release notes. Reviewer approval.

## Phase 7 — Native algorithms

Only with a written justification per algorithm referencing a measured need.
Candidates already visible from Phase 0: an IRLS logistic solver (closer semantic match
to `glm`, deterministic iteration count, no L-BFGS tolerance bound on parity).
