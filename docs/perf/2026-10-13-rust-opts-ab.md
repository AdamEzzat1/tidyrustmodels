# Rust optimisation pass, A/B against the pre-optimisation baseline (2026-10-13)

A "prove it before merging" pass: the candidate changes the implementation, never the
semantics, and is accepted only where an interleaved A/B shows a gain outside the noise
band with every drift fingerprint bitwise identical.

## Baseline and candidate

| | commit | branch / worktree |
|---|---|---|
| baseline | `29ba04af5305ca3e2b5c433798904eb3b8a365ec` | `baseline-before-rust-opts`, worktree `../tidyrustmodels-baseline` |
| candidate | `04e30c7` (code and report), `main` after the fast-forward merge | `candidate-rust-opts-ab` (merged into `main` after acceptance) |

Both A/B binaries were built from the **same driver source** (`benchmarks/rust/src/main.rs`
of the candidate, copied into the baseline worktree) against the two library versions,
release profile (`lto = "fat"`, `codegen-units = 1`), rustc 1.97.1, in the order
baseline then candidate, on the same machine.

## Environment and commands

Windows 11 Pro 10.0.26200, i7-11390H (4 physical / 8 logical cores), 31.8 GB RAM, AC
power, power plan *Balanced*, Python 3.11.3, R not involved (Rust-only A/B). The host
was not dedicated: other processes ran during the measurements (see the forest task
below); the interleaved order makes the two binaries share that noise.

```
# baseline
git worktree add ../tidyrustmodels-baseline -b baseline-before-rust-opts 29ba04a
cp benchmarks/rust/src/main.rs benchmarks/rust/Cargo.toml ../tidyrustmodels-baseline/benchmarks/rust/...
(cd ../tidyrustmodels-baseline && cargo build --release --manifest-path benchmarks/rust/Cargo.toml)
# candidate
cargo build --release --manifest-path benchmarks/rust/Cargo.toml
# A/B, interleaved A-B / B-A per trial (runner mode added in this pass)
python -I benchmarks/run_benchmarks.py --run-id 2026-10-13-ab-opts1 --systems rust_base,rust \
  --engines native --trials 5 --warm 30 --v 10 --tune-levels 20 --threads 1 \
  --ab-baseline ../tidyrustmodels-baseline/benchmarks/rust/target/release/trm-bench.exe \
  --tasks penguins_reg,ames_reg,synth_10000_reg,synth_10000_clf,synth_100000_reg,penguins_tree_clf,\
ames_tree_reg,ames_xgb_reg,synth_10000_reg_wfs,synth_10000_reg_boot,synth_10000_reg_wfreq,synth_10000_rf_clf
python -I benchmarks/run_benchmarks.py --run-id 2026-10-13-ab-opts1 ... --threads 4 \
  --tasks synth_10000_clf,penguins_tree_clf,ames_tree_reg,synth_10000_reg_wfs
python -I benchmarks/run_benchmarks.py --run-id 2026-10-13-ab-opts1-rf ... --trials 6 --warm 5 \
  --tasks synth_10000_rf_clf,ames_rf_reg
python -I benchmarks/ab_report.py --run-id 2026-10-13-ab-opts1      # -> ab_report.md
# drift and semantics
cargo test --workspace            # unit, property, baselines, differential suite vs R fixtures
pytest python/tests               # Python parity / resampling / tuning / interop
cargo run --release --example ames_tuning / ames_workflow_sets; python examples/...
```

Raw data: `benchmarks/results/2026-10-13-ab-opts1/{results.csv,env.json,ab_report.md}` and
`…-ab-opts1-rf/`. Every trial row carries the driver's **fingerprints**: FNV-1a over the
JSON of the model state, the test-set predictions, the fitted workflow, the per-fold CV
metrics, the tuning results, the selection (parameters, `last_fit` metrics and
predictions) and the workflow-set ranking plus the refit's predictions. serde_json prints
every `f64` in shortest round-trip form, so equal fingerprints mean bitwise-equal values.

## Optimisations attempted

Priority followed the measured phase shares of protocol run G (BENCHMARKS.md): on the
linear workflows prep + bake was 30–60% of a workflow fit, and tuning re-prepped the
recipe for every candidate.

| # | change | where | status |
|---|---|---|---|
| 1 | `Table` stores columns as `Arc<Column>`; `Table::clone` and every step's `bake` (which starts from `data.clone()` and rewrites a few columns) share the untouched columns instead of copying them. `column_shared` / `insert_shared` / `iter_shared` added; `StepDummy` and `StepSelect` use them. JSON unchanged (serde `rc`). | core `data.rs`, recipes | **accepted** |
| 2 | Design matrix built with one copy from borrowed column slices (`DesignMatrix::from_column_slices`) instead of `to_vec()` per column plus a second copy. | core `matrix.rs`, recipes `bake_matrix` | **accepted** |
| 3 | Preprocessing once per resample in `tune_grid`: `Workflow::prepare_fold` (tune's `.fit_pre`: prep, bake both sets, read case weights) and `Workflow::fit_prepared` (`.fit_model`, predict, metrics); `tune_grid` keeps one prepared fold per thread in a lazily built, reference-counted slot freed after the fold's last candidate; jobs are fold-major so the serial path holds one fold at a time. `fit_fold_full` / `last_fit` / `fit_resamples` go through the same two halves. This is tune 2.1.0's own loop structure (`tune_grid_loop_iter`). | workflows, tune | **accepted** |
| 4 | Reuse of the Rayon pool: one pool is already built per `fit_resamples` / `tune_grid` / `workflow_map` call and there is no nested parallelism; nothing to change. | – | not needed |
| 5 | Metric evaluation cache: the metric phase is a small share of a fold (each metric is one pass over the assessment rows); not measured as hot. | – | deferred |
| 6 | Batch tree prediction: `predict` is 0.2–4% of a fold on every task measured; the boosted-tree per-row `Vec<f32>` is real but not hot. | – | deferred |
| 7 | Scratch-buffer reuse in the boosted-tree grower: one histogram per node mirrors xgboost; the model-fit phase is unchanged by this pass and dominates the xgb tasks (tune ratio 1.005). A buffer pool would not change accumulation order but needs its own A/B. | – | deferred, next target |
| 8 | Sparse path: not touched; the sparse flag propagates as before (fingerprints cover the sparse tasks' decisions indirectly through the same code path). | – | deferred |
| 9 | Case weights as `Arc<[f64]>`: weights are read once per fold from the already-selected rows; the copy is one `Vec<f64>` per fold, not measured as hot. | – | deferred |
| 10 | Workflow-set level sharing of prepared folds across models of one preprocessor: `workflow_map` benefits from 3 through `tune_grid`; sharing across entries needs an explicit recipe-identity contract and was not attempted. | – | next target |
| 11 | Name-check / `String` clone micro-optimisations: not measured as hot. | – | deferred |
| 12 | (found by protocol run H, 2026-10-14) `Folds::from_assignments` built each fold's analysis set with a per-row `contains` scan over the assessment rows: quadratic in n, about 2 s at 100k rows and 200 s at 1e6 rows, outside every timed phase (the Rust process spent 231 s on 17 s of measured work). Replaced by one pass over the fold assignments; same ascending complement. A/B `2026-10-14-ab-complement` (6 trials, A-B / B-A): wall at 100k rows 0.34× (3.24 s → 1.09 s), CV and fits within noise, all fingerprints identical; differential suite (exact fold memberships vs rsample) and Python suite pass. | resample `lib.rs` | **accepted** |

Behavioural notes (reviewed by the semantics role): (a) `fit_fold_full` now bakes the
assessment rows before the capability check runs, so on an input that is wrong in both
ways a different error can surface first; results are unaffected. (b) `tune_grid`'s
job order is fold-major instead of candidate-major; results are regrouped by
candidate, so `TuneResults` is identical (the fingerprints prove it). (c) `Table::remove`
clones the column values when another table still shares them.

## A/B results (medians over cold trials; ratio = candidate / baseline, < 1 is better)

Main run `2026-10-13-ab-opts1`: 5 trials per cell, 30 warm iterations, 10 folds, 20-point
grids, 160 of 160 trials ok. "IQR" is the relative interquartile range of the two cells;
a ratio whose absolute difference lies inside the larger IQR is reported as noise.

| task | thr | prep+bake (warm) | workflow fit (warm) | CV | tune / wfs | alloc. CV (MB) | peak WS | drift |
|---|---|---|---|---|---|---|---|---|
| penguins_reg (342) | 1 | 1.16 (noise, IQR 57%) | 1.15 (noise, IQR 43%) | 0.94 (noise) | – | **0.56** | 1.00 | PASS |
| ames_reg (2930) | 1 | **0.66** | **0.86** | **0.88** | – | **0.74** | 1.00 | PASS |
| synth_10000_reg | 1 | **0.57** | **0.73** | **0.71** | – | **0.61** | 0.99 | PASS |
| synth_10000_reg_boot (25 bootstraps) | 1 | **0.71** | **0.79** | **0.79** | – | **0.60** | 1.05 (+0.7 MB) | PASS |
| synth_10000_reg_wfreq (freq. weights) | 1 | **0.65** | **0.82** | **0.81** | – | **0.66** | 1.00 | PASS |
| synth_100000_reg | 1 | **0.70** | **0.81** | **0.82** | – | **0.60** | 1.03 | PASS |
| synth_10000_clf (IRLS, 20×10 tune) | 1 | **0.82** | 0.97 (noise) | **0.75** | tune **0.72** | **0.78** | 0.92 | PASS |
| synth_10000_clf | 4 | 1.19 (noise, IQR 40%) | 1.15 (noise, IQR 29%) | 0.77 (noise, IQR 46%) | tune 0.72 (noise, IQR 30%) | **0.78** | 0.94 | PASS |
| penguins_tree_clf (rpart, 4×5 tune) | 1 | **0.72** | **0.79** | 1.04 (noise) | tune **0.63** | **0.58** | 1.00 | PASS |
| penguins_tree_clf | 4 | 0.63 (noise, IQR 59%) | 0.76 (noise, IQR 68%) | 0.93 (noise) | tune **0.63** | **0.58** | 0.95 | PASS |
| ames_tree_reg (rpart, 4×5 tune) | 1 | 0.65 (noise, IQR 53%) | 0.89 (noise) | 1.01 (noise) | tune **0.84** | **0.74** | 0.93 | PASS |
| ames_tree_reg | 4 | **0.75** | 0.98 (noise) | 1.14 (noise, IQR 23%) | tune 0.79 (noise, IQR 25%) | **0.74** | 0.99 | PASS |
| ames_xgb_reg (50 trees, 4×5 tune) | 1 | **0.60** | **0.95** | 0.91 (noise, IQR 27%) | tune 1.01 (noise) | 0.97 | 0.95 | PASS |
| synth_10000_reg_wfs (lm, tree grid, xgb) | 1 | **0.82** | 0.88 (noise, IQR 22%) | **0.79** | wfs 0.97 (noise) | **0.61** | 0.98 | PASS |
| synth_10000_reg_wfs | 4 | **0.86** | **0.90** | **0.73** | wfs **0.86** | **0.61** | 0.99 | PASS |
| synth_10000_rf_clf (ranger rules, 200 trees) | 1 | 1.07 (noise) | 1.41 (noise, IQR 19–29%) | 1.13 (noise, IQR 26–34%) | – | 1.00 | 1.00 | PASS |

Phases the pass does not touch stayed put: the model-only warm fit is 0.98–1.02 on every
linear, logistic, tree and xgb task (within noise), `predict` differs by microseconds.

Forest re-measurement `2026-10-13-ab-opts1-rf` (6 trials, balanced A-B / B-A, 5 warm
iterations, 24 of 24 ok): ames_rf_reg (2930 rows, 200 trees) prep + bake **0.50**,
model fit 0.80 (inside IQR 28%), CV 1.01, allocations identical, drift PASS.
synth_10000_rf_clf stays undecidable on this host: the same binary swings between
2.5 s and 4.2 s per fit and 36 s and 53 s per CV from one trial to the next in both
directions (per-trial pairs in `results.csv`), the IQRs are 17–35%, the two binaries'
allocation counts and bytes are identical to three digits, and the forest code is not
touched by this pass. It is reported as noise, not as a pass.

## Drift verification

| check | result |
|---|---|
| Fingerprints (model state, predictions, fitted workflow JSON, per-fold CV metrics, tuning results, selection, workflow-set ranking + refit predictions), all 16 task × thread cells, all trials, both binaries | identical in every cell (**PASS**; `ab_report.md`) |
| `cargo test --workspace` on the candidate: unit, property, analytic baselines, collinearity, multinomial, tree, forest and boosted-tree baselines, grouped balance rules, the differential suite against the pinned R fixtures (17541 comparisons, 70 tasks, exact tolerances for trees / xgboost / resample memberships) | all pass, 0 failures |
| Python: `maturin develop --release` + `pytest python/tests` (parity bitwise against the Rust results on shared inputs, resampling ids, tuning selection, interop) | 208 passed, 92 skipped (optional pyarrow / polars / scipy paths), 2026-10-14 |
| Examples `ames_tuning` / `ames_workflow_sets` (Rust and Python) against the numbers recorded in VERIFICATION.md Phase 6 (tuning rows, selection, `last_fit` metrics, workflow-set ranking, refit prediction) | identical to the record in all four runs (2026-10-14) |
| Output schemas, column order, resample membership, sparse / dense decision, RNG streams | unchanged code paths; covered by the differential suite's exact checks and the fingerprints |

## Acceptance

All three changes are accepted: the targeted phases improve by 18–43% (prep + bake),
14–27% (workflow fit at 2930–100k rows), 12–29% (CV on the linear tasks), 16–37%
(tuning for the tree and logistic grids), allocation volume per CV drops 22–44%, peak
working set is unchanged or smaller (one +0.7 MB cell on the bootstrap task, within the
5% band and inside the trial spread), no touched phase regresses outside the noise band,
and every drift fingerprint is identical. Phases where the model fit dominates (xgb
tuning, the 342-row tasks at microsecond level) are unchanged, as expected.

Protocol run H's first attempt (`benchmarks/results/2026-10-13-win11-i7-11390H-h-quadratic-complement/`,
4 trials) is kept as the evidence for change 12; run H itself was re-run on the fix.

## Remaining risks

- The host was shared with other processes during the run; the IQRs above say where a
  ratio is not decidable at n = 5. Cells marked noise are not claimed either way.
- `Table` clones are now cheap, so code that relied on `clone()` to get an independent
  copy for in-place mutation keeps working (copy-on-write through `insert`), but a future
  `&mut Column` accessor must clone the `Arc` first.
- The tuning cache holds one prepared fold per thread; with very wide designs and many
  threads that is `threads × (train + test design)` in memory, the same bound as the
  previous per-job materialisation.

## Recommended next targets

1. Boosted-tree grower buffer pool (histograms, row partitions) with an exact-order
   proof via the fingerprints; the xgb model fit is now the largest untouched phase.
2. Sharing prepared folds across the models of one preprocessor in `workflow_map`.
3. Random-forest fit: profile before touching; the trial spread on this host hides
   anything under ~20%.
4. A dedicated or CI benchmark host (ROADMAP.md Phase 5 second platform) so that
   10%-level decisions do not need 6+ trials.
