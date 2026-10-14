# Claims ledger

Every statement on a hand-written page (`site/*.qmd`) that says what the library does,
how fast it is or how it was verified, with the repository line it comes from.
Included content (`site/generated/*`) is a verbatim copy of the named document and is
not listed. `check_claims.py` enforces the numeric part of this ledger mechanically.

## index.qmd

| claim | source |
|---|---|
| recipe / model spec / workflow / resampling / metrics / tuning / workflow sets vocabulary | README.md opening paragraph; docs/FEATURES.md sections |
| one Rust implementation, Python wrapper calls the same code, parity bitwise | docs/FEATURES.md "Framework properties": "Python == Rust bitwise on shared inputs: verified (parity tests on every task)" |
| verified differentially against R, tolerances measured and never silently relaxed | VERIFICATION.md Phase 4; reference/fixtures/tolerances.toml header |
| status 0.1.0-dev, crate and wheels not published | Cargo.toml `version`; README.md "Installing" |
| not affiliated with Posit / tidymodels | project statement (docs/website-prompt.md); no claim of affiliation anywhere in the repository |
| native ports of lm.fit, glm.fit IRLS, rpart, ranger rules, xgboost hist; rsample generators; yardstick; tune_grid; workflow sets | docs/FEATURES.md "Models", "Resampling", "Metrics", "Tuning", "Workflow sets" |
| serial == parallel results | docs/FEATURES.md "Framework properties" |
| NumPy / pandas / Arrow / Polars / SciPy-sparse input | docs/FEATURES.md "Data" |
| forest port on par with ranger, not faster | README.md "What is measured" (0.86–1.04×); BENCHMARKS.md run I |
| nothing transfers to other hardware | README.md "What is measured" last sentence |

## why-rust.qmd

| claim | source |
|---|---|
| parity suite bitwise; examples print the same results in both languages | docs/FEATURES.md "Framework properties"; VERIFICATION.md Phase 6 ("Rust and Python print identical results") |
| driver counts every allocation per phase; forest 580k–2.0M allocations per fit vs a few hundred for linear | BENCHMARKS.md "Measured quantities" A1; run I findings ("580k (Ames) to 2.0M ... where the linear and tree fits make 400–800") |
| 6–930×, 930× / 12× / 6×, 13–450×, 2–4.6×, 3.7–2200×, 13–470×, 2.6–7.8×, 14–270×, 3.5–70× | README.md "What is measured" paragraph (verbatim) |
| semantics ported from installed R package sources | VERIFICATION.md (per phase: "read from the package sources"); docs/adr/0016 context |
| no R interface | the repository has Rust crates and a Python package only (README.md "Layout") |
| covers parts of recipes, parsnip, rsample, yardstick, tune, workflows, workflowsets | docs/FEATURES.md sections; COMPATIBILITY.md "Unsupported in v0.1" |
| forest verified statistically, 4-SE bands over 10 seeds; on par 0.86–1.04× | docs/FEATURES.md "rand_forest ... level 5"; docs/releases/0.1.0.md "ranger forests: statistical (4-SE bands over 10 seeds per side)"; README.md |
| xgboost bit-exact on Windows, expf ulp on Linux / macOS, per-platform bounds (ADR-0017) | docs/adr/0017; VERIFICATION.md cross-platform section; LIMITATIONS.md finding |
| glmnet comparison reported in its own category | BENCHMARKS.md "Two benchmark categories" |
| open: second platform, Linux/macOS xgboost parity, independent review | ROADMAP.md Phase 5 / Phase 6 open items; docs/adr/0017 decision 5 |

## get-started.qmd

| claim | source |
|---|---|
| nothing published to crates.io / PyPI | README.md "Installing" |
| what each example covers | examples/README.md |
| examples run by CI on every push | .github/workflows/ci.yml "Run the release examples" / python job |
| outputs recorded by running them; site build fails on drift | site/data/example-outputs/*, site/scripts/check_examples.py |
| same coefficients, CV accuracy and log loss, best penalty in both languages | site/data/example-outputs/penguins_workflow.{rust,python}.txt (0.9159 / 0.0121, 0.2133 / 0.0282, penalty 0.1) |
| examples pick linfa explicitly; default engine is native | crates/tidyrustmodels/examples/penguins_workflow.rs line `engine(Engine::Linfa)`; docs/FEATURES.md "native (…, default)" |

## features.qmd

| claim | source |
|---|---|
| status definitions | docs/FEATURES.md opening paragraph |

## gaps.qmd

| row | source |
|---|---|
| recipes covered / missing | docs/FEATURES.md "Preprocessing"; COMPATIBILITY.md "Unsupported in v0.1" (`step_zv(group =)`, `step_corr(method = ...)`, `step_filter_missing`); LIMITATIONS.md (unseen levels are an error) |
| parsnip covered / missing | docs/FEATURES.md "Models"; COMPATIBILITY.md rows 35, 97, 100, 101; LIMITATIONS.md first bullet |
| rsample covered / missing | docs/FEATURES.md "Resampling"; LIMITATIONS.md "Resampling" bullet (RNG, empty assessment set, not ported list) |
| yardstick covered / missing | docs/FEATURES.md "Metrics"; COMPATIBILITY.md "Unsupported in v0.1" |
| tune covered / missing | docs/FEATURES.md "Tuning"; LIMITATIONS.md "Tuning" bullet; COMPATIBILITY.md "Unsupported in v0.1" |
| dials | docs/FEATURES.md "Tuning (tune / dials)": regular, random or explicit grids; nothing else listed |
| workflows | docs/FEATURES.md "Data" (case weights), "Models" (sparse toggle), "Framework properties" (JSON round trip); LIMITATIONS.md sparse-decision bullet |
| workflowsets | docs/FEATURES.md "Workflow sets"; LIMITATIONS.md "Workflow sets" bullet; COMPATIBILITY.md row 112 |
| broom | COMPATIBILITY.md row 93 (`tidy(multinom_fit)` std.error / p.value unsupported); `coefficients()` in the examples |
| every "not started" package | absent from docs/FEATURES.md, COMPATIBILITY.md and the workspace (`crates/`) |
| release gated: packaging never run, no reviewer sign-off, no tag; protocol complete on one platform | ROADMAP.md Phase 5 and Phase 6 open items |

## verification.qmd

| claim | source |
|---|---|
| R script writes fixtures; Rust suite compares per quantity under tolerances.toml; fixtures and datasets committed with checksums; no R needed in CI | VERIFICATION.md Phase 4 ("Shared inputs", "Command"); .github/workflows/ci.yml comment "Includes the differential suite ... (no R needed)" |
| bounds justified, set from measured maxima, never silently relaxed; tests fail rather than relax | reference/fixtures/tolerances.toml header |
| discrete outputs compared exactly | VERIFICATION.md tables ("exact" rows); tolerances.toml "Discrete outputs are compared exactly" |
| semantic parity with a solver bound | VERIFICATION.md Phase 4 "Interpretation" |
| forests: ten seeds each side, 4-SE band | docs/releases/0.1.0.md; VERIFICATION.md Phase 8e |
| five levels L1–L5 | VERIFICATION.md report rows (L1–L4); ROADMAP.md Phase 8 (d) "level 5" |
| rpart node for node; tune / rsample deterministic / workflowsets exact | docs/releases/0.1.0.md table |
| xgboost bit for bit on the fixture platform; qualifier | VERIFICATION.md Phase 8f intro + cross-platform section |
| Python parity bitwise, same release profile in CI | VERIFICATION.md cross-platform section last paragraph; LIMITATIONS.md smartcore finding |
| report artifact per OS | .github/workflows/ci.yml upload-artifact step; docs/adr/0017 decision 4 |

## benchmarks.qmd

| claim | source |
|---|---|
| i7-11390H, 31.8 GB RAM, R single-threaded on reference BLAS, ten cold trials, medians | BENCHMARKS.md Phase 5 "Environment"; README.md "What is measured" |
| headline table (every ratio) | README.md "What is measured" paragraph (verbatim ranges) |
| ±5% above 100 ms; 2.4–5× Python wrapper memory | README.md "What is measured" |
| two categories kept apart | BENCHMARKS.md "Two benchmark categories" |
| charts: medians / quartiles over cold trials, invalid run-G configurations excluded, disturbed marked | site/scripts/build_pages.py `INVALID`, `DISTURBED` (from BENCHMARKS.md disturbance log) |
| R 4 threads uses multisession with worker start-up inside the timed phase | BENCHMARKS.md Phase 5 "Environment" |
| A/B with bitwise-identical outputs as acceptance | docs/perf/2026-10-13-rust-opts-ab.md; ROADMAP.md Phase 5 |

## api.qmd

| claim | source |
|---|---|
| crate list and prelude | Cargo workspace members (`crates/`); README.md "Layout" |
| data conventions | python/tidyrustmodels/__init__.py module docstring |
