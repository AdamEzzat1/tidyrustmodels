# Independent technical review: Phases 1–4

Date: 2026-10-12 · Scope: core, recipes, models, resample, metrics, workflows, tune,
Python bindings, differential suite · Reviewer: Role 8 (independent technical reviewer)

**Disclosure.** This review was produced in the same automated session that wrote the
code. It is adversarial in method (every claim below was checked against a test, a
fixture, or the source) but it is not organisationally independent. A human reviewer
should treat the "verified" column as a starting point, not a conclusion.

## Verdict

Phases 1, 2 and 4: **approved with findings** (no critical defects open).
Phase 3: **approved on Windows only**; Linux/macOS builds are defined in CI but have
not run. Phase 5: not reviewable yet (run in progress).

## Claims checked

| claim (from ROADMAP / VERIFICATION) | how checked | verdict |
|---|---|---|
| Both engines match the exact rational OLS to 1e-10 | `tests/baselines.rs`, numbers from `reference/independent/exact_ols.py` | verified |
| Logistic closed-form MLE matched; sign convention equals R's | baseline test + L3 coefficient comparison on 3 classification tasks | verified |
| Preprocessing matches recipes to round-off | 611-row report, L1 max rel 6.5e-13 | verified |
| Per-fold resampling metrics match tune with shared folds | L4, 210 comparisons | verified |
| Serial == parallel | `assert_eq!` on full `ResampleResults` / `TuneResults` | verified |
| Serialization round trip is exact | Rust: text round trip test; Python: `np.array_equal` on probabilities | verified after fixing `float_roundtrip` |
| Python == Rust bitwise | 12 dumped output files compared with `==` | verified |
| No panics reachable with user data | grep for `unwrap`/`expect`/`panic!` in non-test code | see F3 |

## Findings

| id | severity | finding | status |
|---|---|---|---|
| F1 | medium | **General collinearity is unhandled.** Only all-zero and constant columns are aliased. Two linearly dependent non-constant predictors make linfa fail with a backend error and smartcore return a minimum-norm solution; inside `fit_resamples` this aborts the whole call. R warns and reports `NA`. | open; documented in LIMITATIONS.md; Phase 7 candidate (pivoted QR). Recommend: at minimum, map the linfa error to an `Error::Backend` that names the fold, and add a test demonstrating the behaviour so it is not a surprise. |
| F2 | medium | **Logistic parity is solver-bounded (1.6e-8 rel) and was measured on well-conditioned, normalised designs only.** Unnormalised or near-separable data may push L-BFGS further from IRLS. | open; add an unnormalised and a near-separable task to the reference suite before 0.1.0 |
| F3 | low | `expect`/`unwrap` remain in library code on invariants (e.g. `expect("numeric")` after a kind check, `Vec::last().unwrap()` on non-empty). Each is guarded on the preceding line, but there is no lint enforcing that discipline. | open; consider `clippy::unwrap_used`/`expect_used` as warnings with per-site allows |
| F4 | low | `Column::categorical` sorts levels bytewise; R's `factor()` uses the session collation. Fixtures were generated with `LC_COLLATE = "C"`, so parity with default R sessions on non-ASCII or mixed-case labels is untested. | documented; add a mixed-case level test |
| F5 | low | smartcore engine ignores `max_iterations` / `gradient_tolerance` silently (recorded only in the diagnostics note). A user tuning those knobs gets no error. | open; recommend returning `Unsupported` when non-default values are combined with the smartcore engine |
| F6 | low | R's 4-thread `cv_s` timing includes `future::multisession` worker startup (tens of seconds). The report must label this, or the comparison overstates Rust's advantage. | noted for the Phase 5 report |
| F7 | low | The Python package re-exports the native classes directly; subclass-based conveniences were removed. `Table.from_pandas` is a module function (`trm.from_pandas`), which differs from the API spec's `Table.from_pandas`. | accepted deviation; update docs/specs/api-spec.md |
| F8 | info | tune's default classification metric set in R includes `roc_auc`; ours does not implement it. Any benchmark "tune" comparison therefore uses an explicit metric set on both sides (it does). | no action |
| F9 | info | The `Error` enum is `#[non_exhaustive]`; Python's `to_py` has a catch-all. Good. | no action |

## Methodology checks on the differential suite

- Tolerances are per-quantity and justified; the only post-hoc change *tightened* a bound
  (logistic 1e-6 → 1e-7) after measurement. No bound was loosened. ✔
- Discrete outputs (names, order, classes, levels, convergence flags) use exact equality. ✔
- Shared fold ids are loaded in both systems; membership is not inferred from seeds. ✔
- Fixtures embed package versions and dataset checksums; `.gitattributes` protects them
  from line-ending normalisation. ✔
- Gap: the suite does not yet include a task where R reports an `NA` coefficient on the
  *full* training fit (only inside a fold), so the "aliased ↔ NA" check path has run
  but never on a positive case. Add one (F1 follow-up).
