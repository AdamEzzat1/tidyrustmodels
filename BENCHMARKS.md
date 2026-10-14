# Benchmarks

Status: runs A–F recorded under `2026-10-12-win11-i7-11390H[-b|-c|-d|-e|-f]` (see Results; runs C and E are disturbed and not used for conclusions). The
methodology below is the target protocol; the Results section lists every point where
the recorded run fell short of it. No number in this file was produced by anything other
than `benchmarks/run_benchmarks.py` on the recorded machine.

## Principle

The benchmark does not assume Rust wins. tidymodels delegates `lm` to LINPACK/LAPACK QR
and `glm` to compiled IRLS; the R-side overhead is orchestration (tibbles, recipes,
tune). The Rust side has its own overhead (design-matrix copies into backend array
types, L-BFGS instead of IRLS). Both are measured, and results are reported per
phase so readers can see *where* time goes.

## Two benchmark categories (kept separate)

1. **Framework-overhead comparison**: same algorithm class and mathematically equivalent
   objective (OLS vs OLS). Differences are attributable to orchestration + solver.
2. **Workflow-level comparison**: different algorithms or optimizers (R IRLS `glm` vs
   linfa L-BFGS; R `glmnet` ridge vs L-BFGS with an unscaled L2 term for tuning).
   Reported as such; never described as an equivalent-algorithm result.

## Measured quantities

| id | quantity                         | how (Rust)                                   | how (R)                                   |
|----|----------------------------------|----------------------------------------------|-------------------------------------------|
| T1 | end-to-end wall clock            | process start to exit, incl. CSV load        | same, `Rscript` process, incl. `library(tidymodels)` |
| T2 | preprocessing (prep + bake)      | `Instant` around `Recipe::prep`/`bake`       | `bench::hires_time()` around `prep()`/`bake()` |
| T3 | model fit                        | around `ModelSpec::fit`                      | around `fit()`                            |
| T4 | predict                          | around `predict` on the test set             | around `predict()`                        |
| T5 | k-fold CV                        | `fit_resamples`                              | `fit_resamples()`                         |
| T6 | tuning (grid × folds)            | `tune_grid`                                  | `tune_grid()`                             |
| T7 | selection + final fit            | `select_best` + `finalize` + `last_fit`      | `select_best()` + `finalize_workflow()` + `last_fit()` |
| T8 | workflow set                     | `workflow_set` + `workflow_map` + `rank_results` + `fit_best` | the same functions in workflowsets |
| M1 | peak resident memory             | process-tree RSS sampled every 1 ms by the Python runner, plus the OS peak working set of the root process (Windows) | same |
| D1 | data conversion overhead         | CSV→`Table` (`load_s`)                       | CSV→tibble (`load_s`)                     |
| A1 | allocations                      | counting global allocator in the driver: every `alloc`/`realloc` call and its bytes, per phase (`wf_fit`, `cv`) | `bench::bench_memory()` on the workflow fit: R-level bytes allocated (Rprofmem; the engines' C-level mallocs are not seen); counts n/a |
| S1 | scaling                          | T3/T5 at n ∈ {1e3, 1e4, 1e5, 1e6} rows       | same                                      |

Tasks are listed in `benchmarks/tasks.py`. Models: `lm` / `glm` vs the native ports
(framework-overhead), rpart vs its port, xgboost vs its port (framework-overhead,
exact algorithms), ranger vs the forest port (workflow-level: different forests),
glmnet ridge vs the native penalised IRLS when tuning logistic regression
(workflow-level). Case weights, bootstraps, repeated CV, tuning selection and workflow
sets each have their own tasks. Grids: a 20-point log-spaced penalty grid for logistic
regression; explicit 4 x 5 grids (`cost_complexity` x `min_n`; `tree_depth` x
`learn_rate`) for trees and boosting, identical values on both sides.

## Fairness controls

- Same machine, same session, nothing else running; CPU governor/plan recorded.
- Trials interleaved across systems (trial 0 of each system, then trial 1, ...) so that a
  drift in machine state during a run affects every system alike; a canary (one Rust
  trial of a reference task) before and after the run, compared against the previous run.
- Rust: `--release`, `lto = "fat"`, `codegen-units = 1`, `target-cpu` recorded (default
  x86-64 baseline unless stated; both settings reported if `native` is used).
- R: version, BLAS/LAPACK in use (`sessionInfo()`, `extSoftVersion()`), package versions.
- Thread counts: R `fit_resamples` run with `future::plan(sequential)` for the
  single-thread class and `plan(multisession, workers = k)` for the parallel class;
  Rust `Parallelism { threads: Some(k) }` with the same `k`. BLAS threads pinned to 1
  in both unless the configuration says otherwise.
- Cold start: fresh process per trial (includes library load). Warm: one process,
  the workload repeated after a discarded warm-up run.
- Repetitions: ≥ 10 cold trials and ≥ 30 warm iterations per configuration; report
  median, interquartile range, min, and the number of trials. Confidence statements
  only from bootstrap intervals over the trials.
- Failures, timeouts (10 min per trial) and out-of-memory are recorded as such; the
  cell is `not run`, never blank.

## Output

`benchmarks/results/<run_id>/results.csv` (one row per trial), `env.json`, and
`report.md` generated by `benchmarks/report.py`. Speedup = R / Rust; Memory ratio =
R / Rust; >1 favours tidyrustmodels *under the recorded conditions*. Numerical parity
cites the VERIFICATION.md level achieved on that workflow; a benchmark row without
parity evidence is marked `unverified`.

## Results

### Run `2026-10-12-win11-i7-11390H`

Environment (from `env.json` and the R/Rust drivers' version output):

| item | value |
|---|---|
| machine | Windows 11 Pro 10.0.26200, Intel i7-11390H (4 cores / 8 threads), 31.8 GB RAM, laptop |
| Rust | rustc 1.97.1, `--release`, `lto = "fat"`, `codegen-units = 1`, default `target-cpu` (x86-64 baseline), git `83358eb` |
| Rust BLAS | none: linfa-linear 0.8.1 pure-Rust least squares; smartcore 0.6.16 pure-Rust SVD; both single-threaded per fit |
| R | 4.4.0 (ucrt), tidymodels 1.5.0, parsnip 1.5.0, recipes 1.3.2, rsample 1.3.2, tune 2.1.0, workflows 1.3.0, glmnet 4.1-10, future 1.70.0 |
| R BLAS/LAPACK | CRAN-bundled reference `Rblas.dll` / `Rlapack.dll` (single-threaded, unoptimised); `sessionInfo()$BLAS` is empty for the bundled build |
| protocol | 5 cold trials per configuration (fresh process), 10 warm iterations inside each process, 75/25 initial split, 5-fold CV (stratified for classification), tuning grid of 5 log-spaced penalties × 5 folds, 10-min timeout |
| trials | 270 of 270 completed with status `ok`; no timeouts, no failures |

**Deviations from the protocol above** (not silently relaxed; they are the gaps):

- 5 cold trials and 10 warm iterations, not ≥ 10 / ≥ 30. The run already took ~45 min
  because every R 4-thread trial pays ~10–30 s of `multisession` worker start-up.
- 5 folds and a 5-point grid, not 10 folds and 20 points (`--v 5 --tune-levels 5`).
- The 1e6-row scaling point (S1) and the allocation counter (A1) were not run or
  implemented. S1 covers 1e3, 1e4, 1e5 rows.
- CPU power plan was not recorded. The machine is a laptop on mains power; thermal
  throttling cannot be excluded for the 100 s R trials.
- The operating session was not idle: documentation edits (no compilation) happened in
  the same desktop session during the run.
- Rust peak memory is **not captured** for processes that exit in under ~150 ms (the
  1 ms sampler never observed them; 81 of 180 Rust rows). Where captured, sampled RSS
  is a lower bound. Fixed in the harness and re-measured for the small tasks in run B.
- `fit_s` was **not like-for-like** in this run: R's timed `fit(wf)` includes recipe prep
  and bake, Rust's `ModelSpec::fit` does not. The tables below therefore put R `fit_s`
  against Rust `prep_s + bake_s + fit_s` (sums of medians, so no IQR). Run B below adds
  an explicit `wf_fit_s` phase to every driver and a model-only `fit_s` to R.
- The benchmark does **not** assert numerical parity. Each system draws its own folds
  from its own RNG, so the per-run CV means differ by up to ~5e-2 relative on the small
  tasks. Parity is established separately with shared fold ids (VERIFICATION.md, Phase 4).

Both systems use unoptimised, single-threaded linear algebra, so the OLS rows compare
orchestration plus a plain solver on each side. An R build linked to OpenBLAS/MKL, or a
Rust backend using BLAS, would change the fit-phase numbers; neither was measured.

#### Framework-overhead category: OLS vs OLS (`lm` vs `linear_reg`), 1 thread

Medians over 5 cold trials; IQR in brackets; times in milliseconds unless marked `s`.
"Rust" is the `linfa` engine; the `smartcore` engine is given separately because its SVD
solver is markedly slower.

| task | rows × cols | R workflow fit (prep+bake+`lm`) | Rust prep+bake+fit (linfa) | Rust prep+bake+fit (smartcore) | R 5-fold CV | Rust CV (linfa) | Rust CV (smartcore) | R wall | Rust wall (linfa) | R peak RSS | Rust peak RSS |
|---|---|---|---|---|---|---|---|---|---|---|---|
| penguins_reg | 342 × 7 | 53.3 [7] | 0.22 | 0.42 | 1060 [310] | 0.72 [0.11] | 1.44 [0.57] | 5.71 s | 0.101 s | 378 MB | not captured |
| synth_1000_reg | 1000 × 14 | 59.2 [6] | 0.55 | 1.98 | 859 [190] | 2.32 [0.17] | 10.2 [1.2] | 5.55 s | 0.062 s | 381 MB | not captured |
| ames_reg | 2930 × 34 | 102 [33] | 4.2 | 35.9 | 1090 [280] | 23.3 [3.1] | 192 [15] | 6.75 s | 0.189 s | 405 MB | 8 MB |
| synth_10000_reg | 10000 × 14 | 78.3 [7] | 6.3 | 18.7 | 906 [83] | 34.8 [2.5] | 96.1 [21] | 6.05 s | 0.197 s | 407 MB | 10 MB |
| synth_100000_reg | 100000 × 14 | 469 [35] | 55.8 | 163 | 3180 [160] | 327 [20] | 946 [100] | 15.3 s | 3.07 s | 603 MB | 74 MB |

Speedups implied by the medians (R / Rust-linfa): workflow fit 8× (100k rows) to ~240×
(342 rows); CV 10× (100k rows) to ~1500× (342 rows); end-to-end wall 5× (100k rows)
to 57× (342 rows). The end-to-end ratio is bounded below by R's `library(tidymodels)`
load, median 2.38 s across all R trials, which is a fixed cost no workload amortises
within a single process.

Reading: at small n the R numbers are orchestration cost (R's workflow fit on 342 rows
takes 53 ms, of which the linear algebra is microseconds); the Rust numbers are prep,
bake and the actual solve. At 100k rows the gap narrows to 8× on the workflow fit
(24× on the model-only solve: 469 ms vs 19.6 ms), which is mostly R's
`model.matrix`/tibble handling plus reference LAPACK versus a pure-Rust least-squares
path. smartcore's SVD is 6–16× slower than linfa's least squares at n ≥ 2930 and is
not the default for that reason.

#### Workflow-level category: logistic regression (`glm` IRLS vs L-BFGS), 1 thread

Different optimisers; the tuning row compares R `glmnet` (coordinate descent, ridge,
standardised penalty) against L-BFGS with an unscaled L2 term over a 5-point grid.
Do not read these as equivalent-algorithm results.

| task | rows × cols | R workflow fit (prep+bake+`glm`) | Rust prep+bake+fit (linfa) | Rust prep+bake+fit (smartcore) | R CV | Rust CV (linfa) | R tune (glmnet) | Rust tune (linfa) | R wall | Rust wall (linfa) |
|---|---|---|---|---|---|---|---|---|---|---|
| iris_clf | 150 × 4 | 47.9 [6] | 0.48 | 0.52 | 969 [140] | 2.48 [1.1] | 2100 [290] | 5.71 [2.2] | 7.11 s | 0.102 s |
| penguins_clf | 333 × 6 | 51.6 [11] | 0.93 | 1.66 | 1340 [180] | 3.99 [0.7] | 2940 [240] | 19.4 [2.8] | 9.40 s | 0.104 s |
| synth_10000_clf | 10000 × 14 | 133 [6] | 41.2 | 58.9 | 1430 [200] | 216 [19] | 2750 [130] | 925 [69] | 10.8 s | 1.68 s |
| synth_100000_clf | 100000 × 14 | 1070 [31] | 857 | 724 | 4690 [680] | 3240 [440] | 15100 [570] | **21900 [40]** | 35.0 s | **37.7 s** |

Reading: the L-BFGS advantage shrinks with n because IRLS converges in 5–8 Newton
steps regardless of n while L-BFGS needs tens of gradient evaluations. At 100k rows the
workflow fit is only 1.25× faster than R's, **tuning is 0.69× (slower than glmnet)**, and the
single-thread end-to-end wall is **slower than R** (37.7 s vs 35.0 s). This is the
measured case for an IRLS solver in Phase 7 (ROADMAP.md); it is not a framework
overhead problem (prep+bake is 46 ms vs R's 640 ms on the same task).

#### Parallel execution, 4 threads

Rust uses a scoped rayon pool over (candidate, fold) jobs. R uses
`future::plan(multisession, workers = 4)`; **its CV and tune times include the start-up
of four R worker processes** (each loads tidymodels), which on this machine is 8–13 s
for the small tasks. The R 4-thread column is therefore a measurement of that
configuration, not of R's steady-state parallel throughput.

| task | R CV 1 thr | R CV 4 thr (incl. worker start) | Rust CV 1 thr | Rust CV 4 thr | R tune 1 thr | R tune 4 thr | Rust tune 1 thr | Rust tune 4 thr |
|---|---|---|---|---|---|---|---|---|
| penguins_clf | 1340 | 13000 | 3.99 | 13.7 | 2940 | 1330 | 19.4 | 18.5 |
| ames_reg | 1090 | 11500 | 23.3 | 14.9 | — | — | — | — |
| synth_10000_clf | 1430 | 9230 | 216 | 105 | 2750 | 1630 | 925 | 364 |
| synth_100000_reg | 3180 | 12000 | 327 | 221 | — | — | — | — |
| synth_100000_clf | 4690 | 32500 | 3240 | 2280 | 15100 | 23800 | 21900 | 10500 |

Rust scaling on 4 threads: tune 2.1–2.5× on the two 10k/100k classification tasks, CV
1.4–2.1× at n ≥ 2930, and a slowdown on the 333-row task (pool start-up exceeds the
4 ms of work). The 100k-row CV is memory-bound on this 4-core laptop (1.4×). R peak
RSS at 4 threads is 1.66–2.46 GB (five R processes), versus 378–712 MB at 1 thread.

#### Memory, 1 thread

| task | R peak RSS | Rust peak RSS (linfa) | Rust OS peak working set | ratio (R / Rust) |
|---|---|---|---|---|
| ames_reg | 405 MB | 8 MB | 9 MB | ~50× |
| synth_10000_reg | 407 MB | 10 MB | 13 MB | ~40× |
| synth_100000_reg | 603 MB | 74 MB | 81 MB | 8× |
| synth_10000_clf | 422 MB | 13 MB | 13 MB | 32× |
| synth_100000_clf | 712 MB | 81 MB | 81 MB | 9× |

R's floor (~375 MB) is the loaded tidymodels namespace set. Rust rows for the 342-row
and smaller tasks are not captured (see deviations).

#### Warm (steady-state) fit, same process, 10 iterations

| task | R warm workflow fit (prep+bake+model) | Rust warm model-only fit (linfa) | note |
|---|---|---|---|
| penguins_reg | 88.9 ms | 0.024 ms | R warm is *slower* than cold (53 ms); not investigated, within R's GC noise |
| ames_reg | 79.7 ms | 2.18 ms | |
| synth_100000_reg | 295 ms | 20.6 ms | R warm 1.6× faster than cold |
| synth_100000_clf | 587 ms | 770 ms | **R `glm` warm is faster than L-BFGS** at 100k rows, and the Rust cell excludes prep+bake (~35 ms), so the true gap is wider |

### What run A does and does not show

- Shown: for OLS workflows the framework is 8–240× faster per workflow fit and 5–57×
  faster end-to-end than R tidymodels on this machine, with 8–50× less memory, and the
  numbers are reproducible from `results.csv`.
- Shown: for logistic workflows at 100k rows the L-BFGS backends are not faster than
  R's IRLS in steady state and are slower than glmnet for tuning. The "Rust is faster"
  claim does not hold there and this file says so.
- Not shown: behaviour on Linux/macOS, with optimised BLAS on either side, at 1e6 rows,
  with ≥ 10 trials, or on a machine without laptop thermal limits.
- Not shown: any claim about algorithms outside OLS and binary logistic regression.

Raw data: `benchmarks/results/2026-10-12-win11-i7-11390H/{results.csv, env.json,
report.md}`; the generated `report.md` has every phase for every configuration.

### Run `2026-10-12-win11-i7-11390H-b`: exact peak memory, like-for-like fit, and the Python wrapper

Same machine, same day, git `40cb3cc` plus the driver changes in this commit. Purpose:
(1) capture peak memory exactly for the short-lived Rust processes that run A missed;
(2) add a `wf_fit` phase (prep + bake + model fit from the raw training table) to every
driver so the fit comparison is like for like, and a model-only `fit` to R; (3) measure
the **Python wrapper** (`benchmarks/python/bench_task.py`: CPython 3.11.3, pandas CSV
parsing, `from_pandas`, PyO3 calls) as a third system. 6 tasks × {1, 4 threads} ×
{R, Rust linfa, Rust smartcore, Python linfa, Python smartcore} × 5 cold trials = 300 of
300 trials ok. The 100k-row tasks were **not** repeated here; nothing below speaks to them.
Protocol gaps from run A (5 trials, 5 folds, 5-point grid, laptop) are unchanged.

Correctness check inside this run: the Python and Rust drivers use the same seeds, and
their CV summaries are bitwise identical in all 24 (task, threads, engine) groups across
every trial. R's differ because it draws its own folds.

Peak memory is now `peak_wset_mb`: the OS-reported `PeakWorkingSetSize` of the largest
process in the tree, read after exit from a retained handle (benchmarks/README.md).

#### Cold start: process wall time and library load, 1 thread (medians, 5 trials)

| task | R wall | of which `library(tidymodels)` | Python wall | of which `import` (numpy, pandas, tidyrustmodels) | Rust binary wall |
|---|---|---|---|---|---|
| penguins_reg | 7.11 s | 3.15 s | 0.839 s | 0.644 s | 0.064 s |
| penguins_clf | 7.70 s | 2.81 s | 0.798 s | 0.578 s | 0.057 s |
| iris_clf | 7.46 s | 2.82 s | 0.862 s | 0.656 s | 0.109 s |
| synth_1000_reg | 6.68 s | 2.96 s | 0.838 s | 0.645 s | 0.072 s |
| ames_reg | 6.75 s | 2.73 s | 0.728 s | 0.509 s | 0.142 s |
| synth_10000_clf | 11.6 s | 2.88 s | 2.95 s | 0.723 s | 2.78 s |

Python's cold start is 4–10× faster than R's end to end; three quarters of it is the
import. The Rust binary is another 10× below Python on the small tasks and equal to it
on synth_10000_clf, where the L-BFGS fits dominate.

#### Workflow fit, CV and tuning, 1 thread, linfa engine (ms, medians)

| task | R `wf_fit` | Python `wf_fit` | Rust `wf_fit` | R CV | Python CV | Rust CV | R tune (glmnet) | Python tune | Rust tune |
|---|---|---|---|---|---|---|---|---|---|
| penguins_reg | 60.2 | 0.165 | 0.111 | 907 | 0.56 | 0.66 | — | — | — |
| penguins_clf | 58.9 | 0.75 | 0.54 | 993 | 4.5 | 3.6 | 1650 | 24.0 | 20.5 |
| iris_clf | 36.9 | 0.36 | 0.34 | 787 | 1.9 | 2.6 | 1390 | 8.7 | 9.5 |
| synth_1000_reg | 80.7 | 0.44 | 0.42 | 927 | 1.6 | 2.7 | — | — | — |
| ames_reg | 80.1 | 2.6 | 2.7 | 1030 | 13.4 | 17.5 | — | — | — |
| synth_10000_clf | 128 | 46.4 | 50.4 | 1410 | 270 | 265 | 2790 | 1100 | 1180 |

Python wrapper vs Rust binary: the wrapper adds ~50 µs to a 342-row workflow fit
(0.165 vs 0.111 ms) and nothing measurable at 10k rows. On three tasks the Python CV
phase was 1.2–1.7× *faster* than the Rust binary's (both are built with the same
`lto = "fat"`, `codegen-units = 1` profile); the cause was not investigated and the two
should be read as equal within this run's noise. `predict` through Python costs
15–30 µs per call versus 2–4 µs native (NumPy array creation), irrelevant at these sizes.

Python wrapper vs R: every phase on every task is faster through the wrapper, by 2.8×
(synth_10000_clf workflow fit) to ~1600× (penguins_reg CV), with cold start 4–10× faster.

#### Model-only fit (R `fit_xy` on the baked design vs Rust `ModelSpec::fit`), 1 thread

| task | R `lm`/`glm` | Rust linfa | Rust smartcore | ratio (R / linfa) |
|---|---|---|---|---|
| penguins_reg | 9.2 ms | 0.064 ms | 0.22 ms | 144× |
| synth_1000_reg | 10.0 ms | 0.21 ms | 1.5 ms | 47× |
| ames_reg | 28.1 ms | 1.8 ms | 25.8 ms | 16× |
| iris_clf | 23.1 ms | 0.47 ms | 0.60 ms | 49× |
| penguins_clf | 11.8 ms | 0.58 ms | 1.4 ms | 20× |
| synth_10000_clf | 60.4 ms | 45.0 ms | 74.5 ms | 1.3× |

At 10k rows the model-only logistic fit is already within 1.3× of `glm`, consistent
with run A's inversion at 100k rows.

#### Exact peak memory (largest process in the tree), medians

| task | R, 1 thr | Python, 1 thr | Rust, 1 thr | R, 4 thr (largest worker) | R, 4 thr (tree sum, sampled) | Python, 4 thr | Rust, 4 thr |
|---|---|---|---|---|---|---|---|
| penguins_reg | 361 MB | 63 MB | 5.2 MB | 363 MB | 1.66 GB | 63 MB | 5.6 MB |
| penguins_clf | 364 MB | 63 MB | 5.3 MB | 366 MB | 1.69 GB | 64 MB | 6.3 MB |
| iris_clf | 361 MB | 63 MB | 5.1 MB | 366 MB | 1.69 GB | 63 MB | 5.5 MB |
| synth_1000_reg | 365 MB | 63 MB | 5.7 MB | 368 MB | 1.67 GB | 64 MB | 6.9 MB |
| ames_reg | 391 MB | 66 MB | 9.0 MB | 395 MB | 1.69 GB | 76 MB | 17 MB |
| synth_10000_clf | 410 MB | 69 MB | 12.6 MB | 425 MB | 1.80 GB | 82 MB | 23 MB |

R's floor is the attached tidymodels namespaces (~360 MB); Python's is CPython + numpy +
pandas (~62 MB); the Rust binary's is ~5 MB. Ratios at 1 thread: R / Python 5.5–6×,
R / Rust 33–70×.

#### What run B adds to the record

- The Python wrapper is faster than R tidymodels on every measured phase and task,
  including cold start, and uses about one sixth of the memory. Run D extends this to
  100k rows.
- The Rust-vs-Python gap is a fixed ~0.6 s import plus tens of microseconds per call.
- The run A memory gaps for the small tasks are closed; no cell in this run is
  "not captured".

Raw data: `benchmarks/results/2026-10-12-win11-i7-11390H-b/{results.csv, env.json, report.md}`.

### Run `2026-10-12-win11-i7-11390H-c`: 100k-row tasks with the Python wrapper — **unstable conditions, retained for the record only**

Same machine, started immediately after run B. 2 tasks × {1, 4 threads} × 5 systems ×
5 trials, 100 of 100 ok, Python and Rust CV summaries bitwise identical in all 8 groups.
**Do not quote its timings.** The machine's state drifted during the run:

- Configurations identical to run A came out 1.3–4× slower for the first ~30 minutes:
  the Rust binary's pure-CPU model fit on synth_100000_reg took 48 ms (run A: 20 ms) and
  its wall 12.5 s (run A: 3.1 s); R's wall was 50 s (run A: 15 s). Trial-to-trial swings
  were 2–3× (Rust linfa walls 7.4–19 s). The slowdown faded mid-run: the Python linfa
  trials on synth_100000_clf fell from 54 s to 32 s within five consecutive trials, and
  the 4-thread rows measured afterwards are close to run A.
- A canary run of the Rust binary after the run matched run A (fit 17–26 ms, CV 270–300
  ms), and no foreign process was consuming CPU at that point. The cause during the run
  was not identified (thermal throttling after 50 minutes of run B, or a background
  task, are the candidates).
- Runs A–C execute all trials of one system before the next system's, so a drift of this
  kind biases *between-system* ratios, not just absolute numbers. The orchestrator now
  interleaves trials across systems by default (`--blocked` restores the old order), and
  run D repeats these tasks under that order.

Raw data: `benchmarks/results/2026-10-12-win11-i7-11390H-c/`.

### Run `2026-10-12-win11-i7-11390H-d`: 100k-row tasks, all three systems, interleaved trials

Same machine, run after a canary confirmed run A conditions (Rust model fit 17–26 ms on
synth_100000_reg). Trials interleaved across systems. 2 tasks × {1, 4 threads} ×
{R, Rust linfa, Rust smartcore, Python linfa, Python smartcore} × 5 cold trials, 100 of
100 ok. Stability check: every Rust and R phase that exists in both runs is within ±15%
of run A (R `fit_s` is now model-only and therefore smaller by design). Python and Rust
CV summaries bitwise identical in all 8 groups.

Caveat on `wall`: the R and Rust drivers' warm loop runs a model-only fit *and* a
workflow fit per iteration; the Python driver can only run the workflow fit. At 100k
rows that is ~7 s less work in the Python process on the classification task, so wall
is not comparable between the Python driver and the other two here. Compare phases.

#### synth_100000_reg (100000 × 14, OLS, framework-overhead category), medians

| phase | R `lm` | Python linfa | Rust linfa | R / Python | Python / Rust |
|---|---|---|---|---|---|
| library load / import | 2.64 s | 0.76 s | — | 3.5× | — |
| CSV load → table | 752 ms | 152 ms | 199 ms | 4.9× | 0.76× |
| workflow fit (1 thr) | 599 ms | 59.9 ms | 62.4 ms | 10.0× | 0.96× |
| model-only fit (1 thr) | 223 ms | n/a | 26 ms | — | — |
| 5-fold CV (1 thr) | 3.46 s | 0.322 s | 0.349 s | 10.7× | 0.92× |
| 5-fold CV (4 thr) | 12.7 s (incl. worker start) | 0.215 s | 0.221 s | 59× | 0.97× |
| predict (25k rows) | 234 ms | 3.8 ms | 0.47 ms | 61× | 8× |
| peak memory (1 thr) | 605 MB | 128 MB | 81 MB | 4.7× | 1.6× |

#### synth_100000_clf (100000 × 14, logistic, workflow-level category), medians

| phase | R `glm` / glmnet | Python linfa | Rust linfa | R / Python | Python / Rust |
|---|---|---|---|---|---|
| library load / import | 2.62 s | 0.62 s | — | 4.2× | — |
| workflow fit (1 thr) | 1.10 s | 0.751 s | 0.723 s | 1.46× | 0.96× |
| model-only fit, cold (1 thr) | 708 ms | n/a | 684 ms | — | — |
| model-only fit, warm (1 thr) | 526 ms | n/a | 657 ms | — (R 1.25× faster than Rust) | — |
| workflow fit, warm (1 thr) | 659 ms | 736 ms | 670 ms | **0.90× (R faster)** | 0.91× |
| 5-fold CV (1 thr) | 5.21 s | 3.37 s | 2.93 s | 1.55× | 0.87× |
| tune, 5 penalties × 5 folds (1 thr) | 18.2 s (glmnet) | 18.8 s | 18.1 s | **0.97× (equal)** | 0.96× |
| 5-fold CV (4 thr) | 14.2 s (incl. worker start) | 1.68 s | 1.82 s | 8.5× | 1.08× |
| tune (4 thr) | 10.3 s (incl. worker start) | 8.34 s | 8.44 s | 1.24× | 1.01× |
| peak memory (1 thr) | 744 MB | 132 MB | 83 MB | 5.6× | 1.6× |

#### What run D adds to the record

- At 100k rows the Python wrapper is still 10× faster than R for the OLS workflow and CV,
  and 4–5× faster to start. The wrapper costs nothing measurable over the Rust binary
  on fit and CV (0.9–1.0×); its only visible overheads are the ~0.6 s import, pandas CSV
  parsing on small files, and a few milliseconds per `predict` for NumPy conversion.
- For logistic regression at 100k rows the advantage is thin: workflow fit 1.5×, CV
  1.5×, tuning equal to glmnet, and in steady state R's IRLS `glm` fits 1.1–1.25× faster
  than L-BFGS. This holds for the Python wrapper and the Rust binary alike; the
  bottleneck is the solver, not the framework or the wrapper. It is the measured case
  for the IRLS solver in ROADMAP.md Phase 7.
- Memory at 100k rows: R 605–744 MB, Python 128–132 MB, Rust 81–83 MB.

Raw data: `benchmarks/results/2026-10-12-win11-i7-11390H-d/{results.csv, env.json, report.md}`.

### Runs `…-e` and `…-f`: `native` engine vs `linfa` (ADR-0009)

Rust binary only, 1 thread, six tasks, interleaved trials, 5 cold trials each, 60 of 60
ok in both runs. Run E was disturbed by R and Python processes started from this
session while it ran (absolute times 1.3–2× above run D); run F repeated it with the
machine idle and its `linfa` rows agree with run D. Decision evidence is run F.

| task | phase | linfa | native | native / linfa speed |
|---|---|---|---|---|
| synth_100000_clf | model fit · warm fit · workflow fit | 585 · 997 · 643 ms | 145 · 208 · 148 ms | 4.0× · 4.8× · 4.3× |
| synth_100000_clf | 5-fold CV · tune (5 × 5) | 4.62 s · 28.2 s | 1.44 s · 7.4 s | 3.2× · 3.8× |
| synth_10000_clf | model fit · CV · tune | 42.2 ms · 268 ms · 1.12 s | 13.3 ms · 99 ms · 0.47 s | 3.2× · 2.7× · 2.4× |
| penguins_clf | model fit · CV · tune | 0.74 · 4.4 · 19.2 ms | 0.25 · 1.4 · 6.6 ms | 2.9× · 3.1× · 2.9× |
| synth_100000_reg | model fit · CV | 22.8 ms · 359 ms | 22.0 ms · 361 ms | 1.0× · 1.0× |
| ames_reg | model fit · CV | 1.35 ms · 16.3 ms | 2.06 ms · 19.1 ms | 0.65× · 0.85× |
| penguins_reg | model fit · CV | 0.057 · 0.50 ms | 0.043 · 0.53 ms | 1.3× · 0.95× |

Peak working set is within 10% on every task (5–87 MB). The native IRLS converges in
5–8 Newton steps where L-BFGS needs tens of gradient passes, which is the whole of the
logistic gain; on OLS both do one direct solve and the unblocked Householder loop is
slower than linfa on the 34-column Ames design.

What this changes in the earlier conclusions: with `native` as the default engine, the
100k-row logistic case is no longer slower than R. Compared with run D's R rows (same
machine, same day): model fit 145 ms vs `glm` 708 ms cold / 526 ms warm (3.6–4.9×),
5-fold CV 1.44 s vs 5.21 s (3.6×), tuning 7.4 s vs glmnet 18.2 s (2.5×). These are
cross-run numbers and carry run-to-run noise of roughly ±15%; a same-run R-vs-native
comparison is the next benchmark to record.

Raw data: `benchmarks/results/2026-10-12-win11-i7-11390H-e/` (disturbed, kept) and
`…-f/{results.csv, env.json, report.md}`.


### Optimisation pass A/B `2026-10-13-ab-opts1` (ROADMAP Phase 5: before/after for merged optimisations)

Three implementation changes (shared table columns, single-copy design matrix,
preprocessing once per resample in `tune_grid`) were merged after an interleaved A/B
against the pre-optimisation library (`29ba04a`), identical driver, 5 trials per cell,
30 warm iterations, 10 folds, 20-point grids, 16 task × thread cells plus a balanced
6-trial forest re-run, every drift fingerprint bitwise identical. Targeted phases:
prep + bake 0.57–0.82×, workflow fit 0.73–0.95× at 2930–100k rows, CV 0.71–0.88× on the
linear tasks, tuning 0.63–0.84× for the tree and logistic grids, allocation volume per
CV 0.56–0.78×; model-only fits unchanged (0.98–1.02×). Full tables, commands, the
rejected / deferred candidates and the risks: `docs/perf/2026-10-13-rust-opts-ab.md`;
raw data `benchmarks/results/2026-10-13-ab-opts1{,-rf}/`. The protocol runs G (above)
predate this pass; runs K, I, J and H run on the merged code.

### Phase 5 protocol runs `2026-10-13-win11-i7-11390H-{g,i,j,h}`

The runs that close the protocol gaps listed in ROADMAP.md Phase 5: G (linear /
logistic core, 2026-10-13, first chain, partly disturbed), K (its three disturbed
configurations re-measured), I (trees, forests, boosting, case weights, resampling
schemes), J (workflow sets) and H (the 1e6-row scaling point), the last four on
2026-10-14 from an idle-gated chain run as a Windows scheduled task (each run starts
only after six consecutive 20 s samples under 15% CPU; the gate's readings are in
`run-phase5.log`). Runs K–H use the optimised library (`docs/perf/2026-10-13-rust-opts-ab.md`);
run G predates it. Per configuration: **10 cold trials**
(fresh process each, trials interleaved across systems), **30 warm iterations** inside
each process, **10-fold CV**, **20-point grids** for logistic tuning and explicit 4 × 5
grids for trees and boosting. Drivers: `benchmarks/r/bench_task.R`,
`benchmarks/rust/src/main.rs` (counting global allocator, A1),
`benchmarks/python/bench_task.py`; tasks in `benchmarks/tasks.py`.

Environment (`env.json`): Windows 11 Pro 10.0.26200, i7-11390H (4 physical / 8 logical
cores), 31.8 GB RAM, AC power, power plan *Balanced*
(`381b4222-f694-41f0-9685-ff5bb260df2e`), rustc 1.97.1, R 4.4.0 with tidymodels 1.5.0,
Python 3.11.3, git commit `172d04c` plus the uncommitted driver changes recorded in the
same commit as this section. R at 4 threads uses `future::plan(multisession)` (four
worker R sessions started inside the timed `cv` phase); Rust and Python use the
framework's thread pool.

Deviations from the protocol as written, stated up front:

- Only one platform (Windows). The Linux CI runner is still open in ROADMAP.md.
- 4-thread configurations were run for all of run G, two tasks of run I and one of run J;
  run H ran at 1 thread with 10 warm iterations (the R trials already take minutes).
- R's allocation figure is R-level bytes (`bench::bench_memory` / Rprofmem) for the
  workflow fit only; it does not see the engines' C-level mallocs and has no count. Rust
  counts every allocation.
- The Rust driver's counting allocator forwards `alloc` and `realloc` to the system
  allocator but not `alloc_zeroed`, so zero-filled buffers in the Rust binary are
  allocated and then memset by hand while the Python extension (no counting) gets
  Windows' zeroed pages. See "Python wrapper vs Rust binary" below.

Disturbance log (the runs were not attended; this is what the per-trial record shows):

- Run G, 2026-10-13 from about 15:51: other processes took the CPU. The per-trial
  timestamps place the onset in `synth_100000_reg` at 1 thread (trials 5–6 are
  10–30% slower on all three systems; trials 0–4 and 7–9 agree with each other), and
  from 15:57 the disturbance is severe: `synth_100000_reg` at 4 threads, trials 2–9,
  run 1.5–5× slower than trials 0–1 (R wall 53 s → 97–244 s, Rust warm workflow fit
  64 → 112–172 ms), and `synth_100000_clf` at 1 thread degrades from trial 1 (R tune
  55 → 66–69 s; Python 57 → 156 s; Rust 56 → 201 s) until R's trial 3 hit the 600 s
  timeout at 16:56. A process snapshot at 16:57 showed four Chrome processes at ~235%
  of one core combined, `svchost` at 41% and `mrt` (Windows Malicious Software Removal
  Tool) at 6.5% while `trm-bench` got 8%. The chain was stopped at 17:04.
- Consequences: `synth_100000_reg` at 1 thread is reported (medians over 10 trials are
  robust to the two slow trials; flagged in the table); `synth_100000_reg` at 4 threads
  and `synth_100000_clf` at both thread counts are **invalid in run G** and were
  re-measured in run K. The remaining 11 configurations of run G ended before 15:51.
- The remainder (A/B, K, I, J, H) was relaunched behind an idle gate (six consecutive
  20 s samples under 10% CPU by per-process CPU-time deltas; Windows' processor-time
  counters read 90–100% on this machine while the process table shows ~1.5 of 8 cores
  busy, so they could not be used).

#### Run G: linear and logistic workflows, 8 tasks × {1, 4 threads} × {R, Rust native, Python native}

480 trials planned, 431 run before the chain was stopped (1 R timeout, see the
disturbance log); 13 configurations valid, 3 re-measured in run K.

Full per-phase tables for the 13 valid configurations (medians; `—` = not measured on
that system; allocation rows: Rust counts and MB requested, R-level MB for the workflow
fit):

#### ames_reg (2930 rows, linear_reg, 1 thread, 10 resamples) — trials: {'ok': 30}, n(R/Py/Rust)=10/10/10

| phase | R | Python wrapper | Rust native | R / Rust | R / Python | Python / Rust |
|---|---|---|---|---|---|---|
| library load / import | 2.17 s | 580 ms | — | — | 3.75× | — |
| CSV load -> table | 10.6 ms | 9.04 ms | 5.54 ms | 1.92× | 1.17× | 1.63× |
| workflow fit, cold | 59.5 ms | 2.54 ms | 3.54 ms | 16.8× | 23.4× | 0.719× |
| workflow fit, warm (median of 30) | 114 ms | 2.4 ms | 3.11 ms | 36.6× | 47.5× | 0.77× |
| model-only fit, cold | 21 ms | — | 2.26 ms | 9.3× | — | — |
| predict | 12 ms | 175 µs | 21.6 µs | 557× | 68.5× | 8.13× |
| resampling (cv) | 2.75 s | 32.1 ms | 43.6 ms | 63.1× | 85.8× | 0.735× |
| wall (cold process) | 14.2 s | 932 ms | 352 ms | 40.2× | 15.2× | 2.64× |
| peak working set | 389 MB | 80.8 MB | 8.77 MB | 44.4× | 4.81× | 9.22× |
| allocations, workflow fit (count) | — | — | 817 | — | — | — |
| allocations, workflow fit (MB) | 8.25 MB | — | 4.01 MB | 2.05× | — | — |
| allocations, cv (count) | — | — | 11,867 | — | — | — |
| allocations, cv (MB) | — | — | 52.1 MB | — | — | — |

#### ames_reg (2930 rows, linear_reg, 4 threads, 10 resamples) — trials: {'ok': 30}, n(R/Py/Rust)=10/10/10

| phase | R | Python wrapper | Rust native | R / Rust | R / Python | Python / Rust |
|---|---|---|---|---|---|---|
| library load / import | 2.14 s | 598 ms | — | — | 3.59× | — |
| CSV load -> table | 9.76 ms | 8.98 ms | 4.14 ms | 2.36× | 1.09× | 2.17× |
| workflow fit, cold | 56.5 ms | 2.67 ms | 3.2 ms | 17.6× | 21.2× | 0.833× |
| workflow fit, warm (median of 30) | 105 ms | 2.57 ms | 3 ms | 35.1× | 41.1× | 0.855× |
| model-only fit, cold | 19.5 ms | — | 2.05 ms | 9.53× | — | — |
| predict | 13.2 ms | 214 µs | 23.4 µs | 565× | 61.9× | 9.13× |
| resampling (cv) | 15.2 s | 18.8 ms | 22.5 ms | 676× | 806× | 0.839× |
| wall (cold process) | 27.6 s | 926 ms | 256 ms | 108× | 29.8× | 3.62× |
| peak working set | 411 MB | 89 MB | 16.4 MB | 25.1× | 4.62× | 5.43× |
| allocations, workflow fit (count) | — | — | 817 | — | — | — |
| allocations, workflow fit (MB) | 8.25 MB | — | 4.01 MB | 2.05× | — | — |
| allocations, cv (count) | — | — | 11,921 | — | — | — |
| allocations, cv (MB) | — | — | 52.1 MB | — | — | — |

#### iris_clf (150 rows, logistic_reg, 1 thread, 10 resamples) — trials: {'ok': 30}, n(R/Py/Rust)=10/10/10

| phase | R | Python wrapper | Rust native | R / Rust | R / Python | Python / Rust |
|---|---|---|---|---|---|---|
| library load / import | 6.07 s | 1.93 s | — | — | 3.15× | — |
| CSV load -> table | 19.4 ms | 8.71 ms | 703 µs | 27.6× | 2.23× | 12.4× |
| workflow fit, cold | 80.9 ms | 230 µs | 148 µs | 549× | 351× | 1.56× |
| workflow fit, warm (median of 30) | 89.8 ms | 99.7 µs | 115 µs | 784× | 902× | 0.87× |
| model-only fit, cold | 45.9 ms | — | 200 µs | 230× | — | — |
| predict | 22.8 ms | 32.4 µs | 10.5 µs | 2.16e+03× | 706× | 3.07× |
| resampling (cv) | 4.07 s | 1.58 ms | 1.82 ms | 2.23e+03× | 2.57e+03× | 0.87× |
| tune | 11 s | 31.9 ms | 30.3 ms | 363× | 346× | 1.05× |
| wall (cold process) | 36 s | 2.55 s | 240 ms | 150× | 14.1× | 10.6× |
| peak working set | 357 MB | 77.3 MB | 5.16 MB | 69.1× | 4.61× | 15× |
| allocations, workflow fit (count) | — | — | 235 | — | — | — |
| allocations, workflow fit (MB) | 0.291 MB | — | 0.102 MB | 2.86× | — | — |
| allocations, cv (count) | — | — | 3,396 | — | — | — |
| allocations, cv (MB) | — | — | 1.34 MB | — | — | — |

tune_candidates: `20`

#### iris_clf (150 rows, logistic_reg, 4 threads, 10 resamples) — trials: {'ok': 30}, n(R/Py/Rust)=10/10/10

| phase | R | Python wrapper | Rust native | R / Rust | R / Python | Python / Rust |
|---|---|---|---|---|---|---|
| library load / import | 3.87 s | 635 ms | — | — | 6.1× | — |
| CSV load -> table | 13.6 ms | 4.13 ms | 445 µs | 30.6× | 3.3× | 9.27× |
| workflow fit, cold | 55.7 ms | 180 µs | 99 µs | 563× | 310× | 1.82× |
| workflow fit, warm (median of 30) | 50.9 ms | 64.3 µs | 66.3 µs | 767× | 792× | 0.968× |
| model-only fit, cold | 35 ms | — | 128 µs | 274× | — | — |
| predict | 15 ms | 24.9 µs | 7.5 µs | 2e+03× | 603× | 3.31× |
| resampling (cv) | 18.6 s | 974 µs | 1.03 ms | 1.8e+04× | 1.91e+04× | 0.947× |
| tune | 2.4 s | 6.11 ms | 6.69 ms | 360× | 394× | 0.913× |
| wall (cold process) | 31.5 s | 892 ms | 102 ms | 308× | 35.3× | 8.73× |
| peak working set | 365 MB | 77.8 MB | 5.87 MB | 62.1× | 4.69× | 13.2× |
| allocations, workflow fit (count) | — | — | 235 | — | — | — |
| allocations, workflow fit (MB) | 0.291 MB | — | 0.102 MB | 2.86× | — | — |
| allocations, cv (count) | — | — | 3,450 | — | — | — |
| allocations, cv (MB) | — | — | 1.37 MB | — | — | — |

tune_candidates: `20`

#### penguins_clf (333 rows, logistic_reg, 1 thread, 10 resamples) — trials: {'ok': 30}, n(R/Py/Rust)=10/10/10

| phase | R | Python wrapper | Rust native | R / Rust | R / Python | Python / Rust |
|---|---|---|---|---|---|---|
| library load / import | 5.63 s | 1.93 s | — | — | 2.92× | — |
| CSV load -> table | 21.8 ms | 15.5 ms | 1.47 ms | 14.8× | 1.41× | 10.5× |
| workflow fit, cold | 173 ms | 533 µs | 497 µs | 347× | 324× | 1.07× |
| workflow fit, warm (median of 30) | 132 ms | 374 µs | 366 µs | 361× | 354× | 1.02× |
| model-only fit, cold | 38.6 ms | — | 451 µs | 85.7× | — | — |
| predict | 25.1 ms | 50.9 µs | 12.6 µs | 1.99e+03× | 494× | 4.04× |
| resampling (cv) | 3.88 s | 6.89 ms | 6.36 ms | 610× | 564× | 1.08× |
| tune | 11.1 s | 109 ms | 190 ms | 58.5× | 102× | 0.572× |
| wall (cold process) | 34.6 s | 2.93 s | 402 ms | 86.1× | 11.8× | 7.29× |
| peak working set | 358 MB | 77.5 MB | 5.34 MB | 67.1× | 4.62× | 14.5× |
| allocations, workflow fit (count) | — | — | 476 | — | — | — |
| allocations, workflow fit (MB) | 1.22 MB | — | 0.378 MB | 3.22× | — | — |
| allocations, cv (count) | — | — | 6,576 | — | — | — |
| allocations, cv (MB) | — | — | 4.97 MB | — | — | — |

tune_candidates: `20`

#### penguins_clf (333 rows, logistic_reg, 4 threads, 10 resamples) — trials: {'ok': 30}, n(R/Py/Rust)=10/10/10

| phase | R | Python wrapper | Rust native | R / Rust | R / Python | Python / Rust |
|---|---|---|---|---|---|---|
| library load / import | 12.2 s | 2.69 s | — | — | 4.54× | — |
| CSV load -> table | 34.8 ms | 14.2 ms | 1.63 ms | 21.3× | 2.44× | 8.72× |
| workflow fit, cold | 205 ms | 577 µs | 529 µs | 388× | 356× | 1.09× |
| workflow fit, warm (median of 30) | 192 ms | 398 µs | 357 µs | 539× | 484× | 1.11× |
| model-only fit, cold | 30.7 ms | — | 491 µs | 62.5× | — | — |
| predict | 28.4 ms | 52.4 µs | 13.1 µs | 2.18e+03× | 542× | 4.02× |
| resampling (cv) | 52.5 s | 6.3 ms | 6.59 ms | 7.97e+03× | 8.33e+03× | 0.957× |
| tune | 7.9 s | 63.1 ms | 64.6 ms | 122× | 125× | 0.976× |
| wall (cold process) | 114 s | 3.88 s | 284 ms | 403× | 29.5× | 13.7× |
| peak working set | 364 MB | 78.4 MB | 6.25 MB | 58.3× | 4.64× | 12.5× |
| allocations, workflow fit (count) | — | — | 476 | — | — | — |
| allocations, workflow fit (MB) | 1.22 MB | — | 0.378 MB | 3.22× | — | — |
| allocations, cv (count) | — | — | 6,626 | — | — | — |
| allocations, cv (MB) | — | — | 4.99 MB | — | — | — |

tune_candidates: `20`

#### penguins_reg (342 rows, linear_reg, 1 thread, 10 resamples) — trials: {'ok': 30}, n(R/Py/Rust)=10/10/10

| phase | R | Python wrapper | Rust native | R / Rust | R / Python | Python / Rust |
|---|---|---|---|---|---|---|
| library load / import | 5.33 s | 1.54 s | — | — | 3.46× | — |
| CSV load -> table | 13 ms | 9.29 ms | 1.28 ms | 10.2× | 1.4× | 7.27× |
| workflow fit, cold | 122 ms | 214 µs | 131 µs | 931× | 569× | 1.64× |
| workflow fit, warm (median of 30) | 123 ms | 77.2 µs | 78.8 µs | 1.56e+03× | 1.59e+03× | 0.979× |
| model-only fit, cold | 15 ms | — | 77.8 µs | 192× | — | — |
| predict | 21.3 ms | 33.2 µs | 8.5 µs | 2.51e+03× | 641× | 3.91× |
| resampling (cv) | 3.8 s | 1.5 ms | 1.76 ms | 2.16e+03× | 2.53e+03× | 0.853× |
| wall (cold process) | 22.8 s | 2.19 s | 129 ms | 177× | 10.4× | 16.9× |
| peak working set | 357 MB | 77.7 MB | 5.14 MB | 69.5× | 4.6× | 15.1× |
| allocations, workflow fit (count) | — | — | 375 | — | — | — |
| allocations, workflow fit (MB) | 0.375 MB | — | 0.189 MB | 1.98× | — | — |
| allocations, cv (count) | — | — | 5,446 | — | — | — |
| allocations, cv (MB) | — | — | 2.65 MB | — | — | — |

#### penguins_reg (342 rows, linear_reg, 4 threads, 10 resamples) — trials: {'ok': 30}, n(R/Py/Rust)=10/10/10

| phase | R | Python wrapper | Rust native | R / Rust | R / Python | Python / Rust |
|---|---|---|---|---|---|---|
| library load / import | 3.33 s | 864 ms | — | — | 3.86× | — |
| CSV load -> table | 9.12 ms | 9.12 ms | 1.04 ms | 8.75× | 1× | 8.75× |
| workflow fit, cold | 79.5 ms | 166 µs | 129 µs | 617× | 479× | 1.29× |
| workflow fit, warm (median of 30) | 80.1 ms | 71.1 µs | 66.6 µs | 1.2e+03× | 1.13e+03× | 1.07× |
| model-only fit, cold | 12.7 ms | — | 83.2 µs | 152× | — | — |
| predict | 15.4 ms | 23.1 µs | 7.6 µs | 2.03e+03× | 668× | 3.04× |
| resampling (cv) | 16.1 s | 1.24 ms | 1.73 ms | 9.34e+03× | 1.3e+04× | 0.718× |
| wall (cold process) | 28.7 s | 1.21 s | 94.7 ms | 303× | 23.7× | 12.8× |
| peak working set | 365 MB | 77.9 MB | 5.61 MB | 65× | 4.68× | 13.9× |
| allocations, workflow fit (count) | — | — | 375 | — | — | — |
| allocations, workflow fit (MB) | 0.375 MB | — | 0.189 MB | 1.98× | — | — |
| allocations, cv (count) | — | — | 5,500 | — | — | — |
| allocations, cv (MB) | — | — | 2.68 MB | — | — | — |

#### synth_100000_reg (100000 rows, linear_reg, 1 thread, 10 resamples) — trials 5–6 mildly disturbed (see log); trials: {'ok': 30}, n(R/Py/Rust)=10/10/10

| phase | R | Python wrapper | Rust native | R / Rust | R / Python | Python / Rust |
|---|---|---|---|---|---|---|
| library load / import | 2.83 s | 989 ms | — | — | 2.86× | — |
| CSV load -> table | 1.12 s | 182 ms | 223 ms | 5.04× | 6.19× | 0.815× |
| workflow fit, cold | 837 ms | 77.6 ms | 67.8 ms | 12.3× | 10.8× | 1.14× |
| workflow fit, warm (median of 30) | 348 ms | 66.1 ms | 74.3 ms | 4.68× | 5.26× | 0.89× |
| model-only fit, cold | 108 ms | — | 24.6 ms | 4.38× | — | — |
| predict | 94.3 ms | 4.88 ms | 441 µs | 214× | 19.3× | 11.1× |
| resampling (cv) | 6.93 s | 927 ms | 920 ms | 7.53× | 7.47× | 1.01× |
| wall (cold process) | 40.4 s | 7.67 s | 8.63 s | 4.68× | 5.27× | 0.888× |
| peak working set | 651 MB | 144 MB | 72.6 MB | 8.96× | 4.51× | 1.99× |
| allocations, workflow fit (count) | — | — | 573 | — | — | — |
| allocations, workflow fit (MB) | 137 MB | — | 93.8 MB | 1.46× | — | — |
| allocations, cv (count) | — | — | 8,036 | — | — | — |
| allocations, cv (MB) | — | — | 1.21e+03 MB | — | — | — |

#### synth_10000_clf (10000 rows, logistic_reg, 1 thread, 10 resamples) — trials: {'ok': 30}, n(R/Py/Rust)=10/10/10

| phase | R | Python wrapper | Rust native | R / Rust | R / Python | Python / Rust |
|---|---|---|---|---|---|---|
| library load / import | 3.51 s | 883 ms | — | — | 3.97× | — |
| CSV load -> table | 91.4 ms | 28.5 ms | 21.5 ms | 4.25× | 3.21× | 1.33× |
| workflow fit, cold | 248 ms | 15.4 ms | 15.5 ms | 16× | 16.1× | 0.992× |
| workflow fit, warm (median of 30) | 200 ms | 21.5 ms | 20.9 ms | 9.58× | 9.31× | 1.03× |
| model-only fit, cold | 99.9 ms | — | 15.4 ms | 6.5× | — | — |
| predict | 32.4 ms | 294 µs | 90 µs | 360× | 110× | 3.27× |
| resampling (cv) | 5.36 s | 215 ms | 192 ms | 27.9× | 25× | 1.12× |
| tune | 12.3 s | 5.79 s | 4.83 s | 2.54× | 2.12× | 1.2× |
| wall (cold process) | 36.9 s | 8.11 s | 6.83 s | 5.39× | 4.54× | 1.19× |
| peak working set | 435 MB | 87.1 MB | 14.6 MB | 29.9× | 5× | 5.97× |
| allocations, workflow fit (count) | — | — | 668 | — | — | — |
| allocations, workflow fit (MB) | 34.8 MB | — | 15.4 MB | 2.25× | — | — |
| allocations, cv (count) | — | — | 9,386 | — | — | — |
| allocations, cv (MB) | — | — | 201 MB | — | — | — |

tune_candidates: `20`

#### synth_10000_clf (10000 rows, logistic_reg, 4 threads, 10 resamples) — trials: {'ok': 30}, n(R/Py/Rust)=10/10/10

| phase | R | Python wrapper | Rust native | R / Rust | R / Python | Python / Rust |
|---|---|---|---|---|---|---|
| library load / import | 3.54 s | 1.09 s | — | — | 3.25× | — |
| CSV load -> table | 68.2 ms | 35.2 ms | 26.9 ms | 2.53× | 1.94× | 1.31× |
| workflow fit, cold | 161 ms | 20.6 ms | 20.4 ms | 7.91× | 7.81× | 1.01× |
| workflow fit, warm (median of 30) | 173 ms | 19.6 ms | 19.4 ms | 8.92× | 8.82× | 1.01× |
| model-only fit, cold | 75.7 ms | — | 16.6 ms | 4.57× | — | — |
| predict | 24.5 ms | 429 µs | 111 µs | 220× | 57× | 3.85× |
| resampling (cv) | 17.4 s | 109 ms | 112 ms | 155× | 160× | 0.97× |
| tune | 4.99 s | 1.99 s | 2.11 s | 2.36× | 2.51× | 0.941× |
| wall (cold process) | 45.7 s | 4.46 s | 3.74 s | 12.2× | 10.3× | 1.19× |
| peak working set | 429 MB | 106 MB | 33.1 MB | 13× | 4.06× | 3.2× |
| allocations, workflow fit (count) | — | — | 668 | — | — | — |
| allocations, workflow fit (MB) | 34.8 MB | — | 15.4 MB | 2.25× | — | — |
| allocations, cv (count) | — | — | 9,440 | — | — | — |
| allocations, cv (MB) | — | — | 201 MB | — | — | — |

tune_candidates: `20`

#### synth_10000_reg (10000 rows, linear_reg, 1 thread, 10 resamples) — trials: {'ok': 30}, n(R/Py/Rust)=10/10/10

| phase | R | Python wrapper | Rust native | R / Rust | R / Python | Python / Rust |
|---|---|---|---|---|---|---|
| library load / import | 2.15 s | 595 ms | — | — | 3.62× | — |
| CSV load -> table | 43.8 ms | 16 ms | 18.2 ms | 2.41× | 2.74× | 0.878× |
| workflow fit, cold | 85.4 ms | 3.97 ms | 5.44 ms | 15.7× | 21.5× | 0.73× |
| workflow fit, warm (median of 30) | 128 ms | 3.57 ms | 4.95 ms | 25.9× | 35.9× | 0.721× |
| model-only fit, cold | 12.6 ms | — | 1.95 ms | 6.49× | — | — |
| predict | 14.3 ms | 357 µs | 30.6 µs | 466× | 39.9× | 11.7× |
| resampling (cv) | 3.14 s | 52.6 ms | 69.3 ms | 45.2× | 59.6× | 0.759× |
| wall (cold process) | 15.5 s | 1.02 s | 498 ms | 31.1× | 15.1× | 2.06× |
| peak working set | 396 MB | 84.7 MB | 12.5 MB | 31.8× | 4.67× | 6.8× |
| allocations, workflow fit (count) | — | — | 553 | — | — | — |
| allocations, workflow fit (MB) | 13.7 MB | — | 9.03 MB | 1.51× | — | — |
| allocations, cv (count) | — | — | 7,886 | — | — | — |
| allocations, cv (MB) | — | — | 124 MB | — | — | — |

#### synth_10000_reg (10000 rows, linear_reg, 4 threads, 10 resamples) — trials: {'ok': 30}, n(R/Py/Rust)=10/10/10

| phase | R | Python wrapper | Rust native | R / Rust | R / Python | Python / Rust |
|---|---|---|---|---|---|---|
| library load / import | 5.17 s | 1.39 s | — | — | 3.73× | — |
| CSV load -> table | 82 ms | 30.7 ms | 29.1 ms | 2.81× | 2.67× | 1.05× |
| workflow fit, cold | 205 ms | 6.03 ms | 8.27 ms | 24.8× | 34× | 0.729× |
| workflow fit, warm (median of 30) | 162 ms | 6.43 ms | 7.7 ms | 21× | 25.1× | 0.834× |
| model-only fit, cold | 32.4 ms | — | 2.86 ms | 11.3× | — | — |
| predict | 37 ms | 620 µs | 36.9 µs | 1e+03× | 59.7× | 16.8× |
| resampling (cv) | 21.2 s | 38.4 ms | 41.9 ms | 505× | 551× | 0.917× |
| wall (cold process) | 37.5 s | 2.15 s | 662 ms | 56.6× | 17.4× | 3.25× |
| peak working set | 411 MB | 96.8 MB | 22.7 MB | 18.1× | 4.25× | 4.27× |
| allocations, workflow fit (count) | — | — | 553 | — | — | — |
| allocations, workflow fit (MB) | 13.7 MB | — | 9.03 MB | 1.51× | — | — |
| allocations, cv (count) | — | — | 7,940 | — | — | — |
| allocations, cv (MB) | — | — | 124 MB | — | — | — |

What run G shows (medians over 10 cold trials; "×" = R time / candidate time):

- **Small data (150–342 rows).** The workflow fit is 350–930× faster than R cold and
  360–1560× warm; 10-fold CV is 610–2200× faster (R 3.8–4.1 s, Rust 1.8–6.4 ms);
  `predict` 2000–2500×. The framework-overhead category is where the Rust implementation
  pays off most: R's cost here is orchestration, not arithmetic.
- **Tuning (20 penalties × 10 folds, workflow-level: glmnet ridge vs native IRLS).**
  iris 11 s vs 30 ms (360×), penguins 11.1 s vs 190 ms (59×), synth_10000_clf 12.3 s
  vs 4.8 s (2.5×). The ratio collapses with n because glmnet's coordinate descent fits a
  whole penalty path per fold while the port runs one IRLS per candidate; this is a
  different algorithm, as the category says.
- **Mid-size data.** ames (2930 rows, `lm`): workflow fit 17× cold / 37× warm, CV 63×,
  peak memory 44×. synth_10000_reg: workflow fit 16× / 26×, CV 45×. synth_10000_clf
  (`glm` IRLS vs the IRLS port): workflow fit 16× cold / 9.6× warm, CV 28×. The warm
  ratio is the one to compare with run D, where the L-BFGS backend was 0.9× of `glm`.
- **4 threads.** R's `multisession` adds 12–49 s of worker start-up to every CV phase, so
  R is slower at 4 threads than at 1 on every task in this run. Rust and Python gain
  1.7–2× on CV and 2.3× on tuning at 10k rows (CV 69 → 36 ms, 192 → 112 ms; tune
  4.83 → 2.11 s) and nothing on the 342-row tasks (1.7 ms either way).
- **Memory.** Peak working set: R 357–435 MB, Python wrapper 77–106 MB, Rust binary
  5–33 MB. Allocation counts (Rust): 235–817 allocations for a workflow fit
  (0.10–15.4 MB requested), 3.4k–12k for 10-fold CV. R allocates 1.5–3.2× the bytes at
  the R level alone for the same workflow fit.
- **100k rows (`synth_100000_reg`, 1 thread).** Workflow fit 12× cold / 4.7× warm
  (R 348 ms vs 74 ms), 10-fold CV 7.5× (6.9 s vs 0.92 s), `predict` 214×, peak memory
  651 vs 73 MB (9×). The gap narrows with n as both sides become arithmetic-bound; the
  framework overhead that dominates the small tasks is a fixed cost.
- **Cold wall time.** R 14–37 s at 1 thread, of which 2.2–6.1 s is attaching tidymodels
  and the rest the 30-iteration warm loop; Python 0.9–8.1 s (0.6–1.9 s import); Rust
  0.09–6.8 s.

**Python wrapper vs Rust binary.** On compute-bound phases (logistic fit, CV and tuning
at 10k rows) the two are equal within ±10%. On the OLS workflows the Python wrapper is
15–28% *faster* than the Rust binary (ames CV 32 vs 44 ms; synth_10000_reg CV 53 vs
69 ms; warm workflow fit 2.4 vs 3.1 ms and 3.6 vs 5.0 ms). Both call the same native
code; the Rust binary additionally counts allocations and, as noted above, memsets every
zero-filled buffer. The planned A/B of the two driver builds was overtaken by the optimisation pass of 2026-10-13 (`docs/perf/2026-10-13-rust-opts-ab.md`), which changed the library under both drivers; it was not run. Runs K, I, J and H use the fixed driver, so their Rust and Python columns are comparable again. For run G, read the Python column as the cleaner timing of the native code on the OLS tasks.

#### Run K: the three configurations invalidated in run G, re-measured idle (2026-10-14 01:14–02:18)

Launched behind the idle gate (loads 8–13% at start), optimised library (commit
`23e5455`, see the A/B section), fixed driver, 90 of 90 trials ok.

Full per-phase tables (medians over 10 cold trials; allocation rows: Rust counts and MB
requested, R-level MB for the workflow fit):

#### synth_100000_clf (100000 rows, logistic_reg, 1 thread, 10 resamples) — trials: {'ok': 30}, n(R/Py/Rust)=10/10/10

| phase | R | Python wrapper | Rust native | R / Rust | R / Python | Python / Rust |
|---|---|---|---|---|---|---|
| library load / import | 2.42 s | 666 ms | — | — | 3.63× | — |
| CSV load -> table | 781 ms | 178 ms | 192 ms | 4.06× | 4.38× | 0.927× |
| workflow fit, cold | 1.13 s | 152 ms | 140 ms | 8.06× | 7.44× | 1.08× |
| workflow fit, warm (median of 30) | 583 ms | 143 ms | 143 ms | 4.09× | 4.07× | 1× |
| model-only fit, cold | 690 ms | — | 125 ms | 5.5× | — | — |
| predict | 87 ms | 3.42 ms | 647 µs | 134× | 25.4× | 5.29× |
| resampling (cv) | 10 s | 2 s | 2.09 s | 4.79× | 5.01× | 0.955× |
| tune | 35.5 s | 33.1 s | 33 s | 1.08× | 1.07× | 1× |
| wall (cold process) | 96.1 s | 43.4 s | 47 s | 2.04× | 2.22× | 0.922× |
| peak working set | 769 MB | 165 MB | 89.6 MB | 8.58× | 4.66× | 1.84× |
| allocations, workflow fit (count) | — | — | 560 | — | — | — |
| allocations, workflow fit (MB) | 347 MB | — | 123 MB | 2.82× | — | — |
| allocations, cv (count) | — | — | 7,656 | — | — | — |
| allocations, cv (MB) | — | — | 1.53e+03 MB | — | — | — |

tune_candidates: `20`

#### synth_100000_clf (100000 rows, logistic_reg, 4 threads, 10 resamples) — trials: {'ok': 30}, n(R/Py/Rust)=10/10/10

| phase | R | Python wrapper | Rust native | R / Rust | R / Python | Python / Rust |
|---|---|---|---|---|---|---|
| library load / import | 2.16 s | 655 ms | — | — | 3.3× | — |
| CSV load -> table | 747 ms | 173 ms | 189 ms | 3.95× | 4.33× | 0.912× |
| workflow fit, cold | 1.08 s | 140 ms | 143 ms | 7.55× | 7.74× | 0.975× |
| workflow fit, warm (median of 30) | 575 ms | 138 ms | 139 ms | 4.14× | 4.16× | 0.996× |
| model-only fit, cold | 697 ms | — | 122 ms | 5.72× | — | — |
| predict | 91.7 ms | 3.21 ms | 741 µs | 124× | 28.6× | 4.34× |
| resampling (cv) | 15.3 s | 918 ms | 1.01 s | 15.1× | 16.6× | 0.907× |
| tune | 15.4 s | 13.9 s | 14.1 s | 1.09× | 1.1× | 0.987× |
| wall (cold process) | 78.7 s | 22.9 s | 26.4 s | 2.98× | 3.44× | 0.867× |
| peak working set | 765 MB | 324 MB | 248 MB | 3.08× | 2.36× | 1.31× |
| allocations, workflow fit (count) | — | — | 560 | — | — | — |
| allocations, workflow fit (MB) | 347 MB | — | 123 MB | 2.82× | — | — |
| allocations, cv (count) | — | — | 7,710 | — | — | — |
| allocations, cv (MB) | — | — | 1.53e+03 MB | — | — | — |

tune_candidates: `20`

#### synth_100000_reg (100000 rows, linear_reg, 4 threads, 10 resamples) — trials: {'ok': 30}, n(R/Py/Rust)=10/10/10

| phase | R | Python wrapper | Rust native | R / Rust | R / Python | Python / Rust |
|---|---|---|---|---|---|---|
| library load / import | 2.52 s | 693 ms | — | — | 3.64× | — |
| CSV load -> table | 829 ms | 132 ms | 206 ms | 4.03× | 6.27× | 0.642× |
| workflow fit, cold | 654 ms | 44.4 ms | 46 ms | 14.2× | 14.7× | 0.966× |
| workflow fit, warm (median of 30) | 287 ms | 45.8 ms | 43 ms | 6.66× | 6.26× | 1.06× |
| model-only fit, cold | 74.3 ms | — | 22.7 ms | 3.27× | — | — |
| predict | 63.7 ms | 2.31 ms | 354 µs | 180× | 27.6× | 6.53× |
| resampling (cv) | 13.6 s | 285 ms | 295 ms | 46× | 47.6× | 0.966× |
| wall (cold process) | 43.7 s | 5.36 s | 5.91 s | 7.38× | 8.15× | 0.907× |
| peak working set | 628 MB | 259 MB | 188 MB | 3.34× | 2.42× | 1.38× |
| allocations, workflow fit (count) | — | — | 465 | — | — | — |
| allocations, workflow fit (MB) | 137 MB | — | 56.4 MB | 2.42× | — | — |
| allocations, cv (count) | — | — | 6,540 | — | — | — |
| allocations, cv (MB) | — | — | 731 MB | — | — | — |

- **100k rows, logistic (`glm` vs the IRLS port).** Workflow fit 8.1× cold / 4.1× warm
  (R 583 ms vs 143 ms), 10-fold CV 4.8× at 1 thread and 15× at 4 threads (R's
  `multisession` 15.3 s vs 1.0 s), peak memory 769 vs 90 MB (8.6×). Tuning 20 penalties
  × 10 folds against glmnet: 1.08× at 1 thread (35.5 vs 33.0 s) and 1.09× at 4 threads
  (15.4 vs 14.1 s): at this size glmnet's path algorithm and one IRLS per candidate
  cost the same, and both scale with the thread count (2.3× for the port, 2.3× for R).
- **100k rows, OLS, 4 threads.** Workflow fit 14× cold / 6.7× warm, CV 46× (R 13.6 s
  including worker start vs 0.29 s), memory 628 vs 188 MB. Rust's CV gains 3.1× from
  4 threads here (0.92 s → 0.29 s, run G 1-thread vs run K).
- The Python wrapper is within 1–8% of the Rust binary on every phase of these three
  configurations, as it was before the driver's `alloc_zeroed` fix only on the
  compute-bound ones.

#### Run I: trees, forests, boosting, case weights, bootstraps and repeated CV (2026-10-14 02:20–06:51, idle)

10 tasks at 1 thread, 10 cold trials each, 30 warm iterations, 10 folds (25 bootstraps,
3 × 10 repeated folds), explicit 4 × 5 grids for the tree (`cost_complexity` × `min_n`)
and the booster (`tree_depth` × `learn_rate`), `select_best` + `finalize` + `last_fit`
on the tuned tasks. 301 trials, 300 ok: R's first trial on `synth_100000_tree_reg`
hit the 600 s timeout on an idle machine (rpart needs about 790 s for that task's
tuning), the task was re-run with a 1500 s timeout and the timed-out row is kept. The
two 4-thread configurations follow in the same run id (`ames_tree_reg`,
`synth_100000_xgb_clf`).

Full per-phase tables (medians over 10 cold trials):

#### ames_rf_reg (2930 rows, rand_forest, 1 thread, 10 resamples) — trials: {'ok': 30}, n(R/Py/Rust)=10/10/10

| phase | R | Python wrapper | Rust native | R / Rust | R / Python | Python / Rust |
|---|---|---|---|---|---|---|
| library load / import | 2.56 s | 716 ms | — | — | 3.58× | — |
| CSV load -> table | 12.6 ms | 10 ms | 5.7 ms | 2.21× | 1.26× | 1.76× |
| workflow fit, cold | 360 ms | 348 ms | 347 ms | 1.04× | 1.04× | 1× |
| workflow fit, warm (median of 30) | 301 ms | 320 ms | 332 ms | 0.908× | 0.941× | 0.965× |
| model-only fit, cold | 330 ms | — | 356 ms | 0.928× | — | — |
| predict | 33.4 ms | 15.5 ms | 15.9 ms | 2.1× | 2.16× | 0.973× |
| resampling (cv) | 5.16 s | 4.32 s | 4.56 s | 1.13× | 1.19× | 0.948× |
| wall (cold process) | 30.5 s | 16.6 s | 26 s | 1.17× | 1.84× | 0.637× |
| peak working set | 421 MB | 96.6 MB | 40.7 MB | 10.3× | 4.36× | 2.37× |
| allocations, workflow fit (count) | — | — | 579,808 | — | — | — |
| allocations, workflow fit (MB) | 7.51 MB | — | 346 MB | 0.0217× | — | — |
| allocations, cv (count) | — | — | 6,542,850 | — | — | — |
| allocations, cv (MB) | — | — | 4.28e+03 MB | — | — | — |

#### ames_tree_reg (2930 rows, decision_tree, 1 thread, 10 resamples) — trials: {'ok': 30}, n(R/Py/Rust)=10/10/10

| phase | R | Python wrapper | Rust native | R / Rust | R / Python | Python / Rust |
|---|---|---|---|---|---|---|
| library load / import | 2.21 s | 623 ms | — | — | 3.55× | — |
| CSV load -> table | 10.4 ms | 10.5 ms | 4.33 ms | 2.4× | 0.987× | 2.43× |
| workflow fit, cold | 133 ms | 3.73 ms | 4.67 ms | 28.5× | 35.7× | 0.799× |
| workflow fit, warm (median of 30) | 113 ms | 3.13 ms | 3.68 ms | 30.8× | 36.2× | 0.852× |
| model-only fit, cold | 85.8 ms | — | 3.47 ms | 24.7× | — | — |
| predict | 13.5 ms | 91.1 µs | 17.9 µs | 758× | 148× | 5.1× |
| resampling (cv) | 2.15 s | 46.3 ms | 52.4 ms | 41.1× | 46.5× | 0.884× |
| tune | 26.1 s | 927 ms | 1.07 s | 24.3× | 28.2× | 0.864× |
| select_best + finalize + last_fit | 308 ms | 6.62 ms | 7.84 ms | 39.3× | 46.5× | 0.845× |
| wall (cold process) | 39 s | 1.84 s | 1.38 s | 28.2× | 21.2× | 1.33× |
| peak working set | 404 MB | 79.8 MB | 8.58 MB | 47.1× | 5.07× | 9.3× |
| allocations, workflow fit (count) | — | — | 757 | — | — | — |
| allocations, workflow fit (MB) | 8.25 MB | — | 3.08 MB | 2.68× | — | — |
| allocations, cv (count) | — | — | 10,506 | — | — | — |
| allocations, cv (MB) | — | — | 39.8 MB | — | — | — |

tune_candidates: `20`; select_params: `[["cost_complexity", 0.0001], ["min_n", 10.0]]`; R select_params: `{"cost_complexity": 0.0001, "min_n": 15}`

#### ames_tree_reg (2930 rows, decision_tree, 4 threads, 10 resamples) — trials: {'ok': 30}, n(R/Py/Rust)=10/10/10

| phase | R | Python wrapper | Rust native | R / Rust | R / Python | Python / Rust |
|---|---|---|---|---|---|---|
| library load / import | 1.92 s | 554 ms | — | — | 3.46× | — |
| CSV load -> table | 8.83 ms | 9.44 ms | 4.03 ms | 2.19× | 0.935× | 2.34× |
| workflow fit, cold | 102 ms | 3.08 ms | 3.47 ms | 29.3× | 33.1× | 0.887× |
| workflow fit, warm (median of 30) | 96.8 ms | 2.74 ms | 2.81 ms | 34.4× | 35.3× | 0.974× |
| model-only fit, cold | 65 ms | — | 3.19 ms | 20.4× | — | — |
| predict | 13.1 ms | 99.3 µs | 13.9 µs | 943× | 131× | 7.17× |
| resampling (cv) | 8.65 s | 20.4 ms | 23 ms | 376× | 425× | 0.886× |
| tune | 9.06 s | 310 ms | 326 ms | 27.8× | 29.2× | 0.95× |
| select_best + finalize + last_fit | 307 ms | 6.18 ms | 5.83 ms | 52.7× | 49.7× | 1.06× |
| wall (cold process) | 28.7 s | 1.12 s | 556 ms | 51.6× | 25.5× | 2.02× |
| peak working set | 395 MB | 87.5 MB | 16.8 MB | 23.6× | 4.52× | 5.22× |
| allocations, workflow fit (count) | — | — | 757 | — | — | — |
| allocations, workflow fit (MB) | 8.25 MB | — | 3.08 MB | 2.68× | — | — |
| allocations, cv (count) | — | — | 10,560 | — | — | — |
| allocations, cv (MB) | — | — | 39.9 MB | — | — | — |

tune_candidates: `20`; select_params: `[["cost_complexity", 0.0001], ["min_n", 10.0]]`; R select_params: `{"cost_complexity": 0.0001, "min_n": 15}`

#### ames_xgb_reg (2930 rows, boost_tree, 1 thread, 10 resamples) — trials: {'ok': 30}, n(R/Py/Rust)=10/10/10

| phase | R | Python wrapper | Rust native | R / Rust | R / Python | Python / Rust |
|---|---|---|---|---|---|---|
| library load / import | 1.9 s | 613 ms | — | — | 3.09× | — |
| CSV load -> table | 9.34 ms | 9 ms | 4 ms | 2.33× | 1.04× | 2.25× |
| workflow fit, cold | 89.4 ms | 17.5 ms | 20.3 ms | 4.39× | 5.1× | 0.862× |
| workflow fit, warm (median of 30) | 75.5 ms | 15.7 ms | 16.6 ms | 4.56× | 4.82× | 0.946× |
| model-only fit, cold | 77.7 ms | — | 21.4 ms | 3.63× | — | — |
| predict | 12.9 ms | 1.09 ms | 831 µs | 15.5× | 11.8× | 1.31× |
| resampling (cv) | 1.71 s | 217 ms | 220 ms | 7.8× | 7.9× | 0.987× |
| tune | 18.3 s | 4.31 s | 4.45 s | 4.12× | 4.25× | 0.969× |
| select_best + finalize + last_fit | 233 ms | 21.7 ms | 22.7 ms | 10.2× | 10.7× | 0.952× |
| wall (cold process) | 28.4 s | 5.83 s | 5.94 s | 4.78× | 4.88× | 0.98× |
| peak working set | 430 MB | 84.7 MB | 14.3 MB | 30× | 5.08× | 5.91× |
| allocations, workflow fit (count) | — | — | 21,794 | — | — | — |
| allocations, workflow fit (MB) | 1.87 MB | — | 42.8 MB | 0.0437× | — | — |
| allocations, cv (count) | — | — | 241,597 | — | — | — |
| allocations, cv (MB) | — | — | 522 MB | — | — | — |

tune_candidates: `20`; select_params: `[["tree_depth", 6.0], ["learn_rate", 0.1]]`; R select_params: `{"tree_depth": 4, "learn_rate": 0.2}`

#### penguins_tree_clf (333 rows, decision_tree, 1 thread, 10 resamples) — trials: {'ok': 30}, n(R/Py/Rust)=10/10/10

| phase | R | Python wrapper | Rust native | R / Rust | R / Python | Python / Rust |
|---|---|---|---|---|---|---|
| library load / import | 2.24 s | 646 ms | — | — | 3.47× | — |
| CSV load -> table | 6.49 ms | 5.68 ms | 926 µs | 7.01× | 1.14× | 6.14× |
| workflow fit, cold | 57.3 ms | 231 µs | 214 µs | 268× | 248× | 1.08× |
| workflow fit, warm (median of 30) | 58.1 ms | 132 µs | 130 µs | 446× | 441× | 1.01× |
| model-only fit, cold | 14.8 ms | — | 182 µs | 81.4× | — | — |
| predict | 9.17 ms | 27.7 µs | 16.4 µs | 559× | 331× | 1.69× |
| resampling (cv) | 1.61 s | 2.31 ms | 3.41 ms | 472× | 695× | 0.679× |
| tune | 15.2 s | 34 ms | 35.8 ms | 425× | 448× | 0.951× |
| select_best + finalize + last_fit | 266 ms | 303 µs | 277 µs | 959× | 876× | 1.09× |
| wall (cold process) | 24.2 s | 846 ms | 67.5 ms | 359× | 28.6× | 12.5× |
| peak working set | 360 MB | 77.4 MB | 5.28 MB | 68.1× | 4.65× | 14.7× |
| allocations, workflow fit (count) | — | — | 423 | — | — | — |
| allocations, workflow fit (MB) | 0.368 MB | — | 0.103 MB | 3.58× | — | — |
| allocations, cv (count) | — | — | 6,487 | — | — | — |
| allocations, cv (MB) | — | — | 1.57 MB | — | — | — |

tune_candidates: `20`; select_params: `[["cost_complexity", 0.0001], ["min_n", 10.0]]`; R select_params: `{"cost_complexity": 0.0001, "min_n": 15}`

#### synth_100000_tree_reg (100000 rows, decision_tree, 1 thread, 10 resamples) — trials: {'timeout': 1, 'ok': 30}, n(R/Py/Rust)=10/10/10

| phase | R | Python wrapper | Rust native | R / Rust | R / Python | Python / Rust |
|---|---|---|---|---|---|---|
| library load / import | 2.32 s | 689 ms | — | — | 3.36× | — |
| CSV load -> table | 841 ms | 126 ms | 190 ms | 4.42× | 6.7× | 0.66× |
| workflow fit, cold | 2.34 s | 173 ms | 164 ms | 14.3× | 13.5× | 1.06× |
| workflow fit, warm (median of 30) | 2.08 s | 154 ms | 160 ms | 12.9× | 13.4× | 0.962× |
| model-only fit, cold | 1.89 s | — | 139 ms | 13.7× | — | — |
| predict | 80 ms | 2.82 ms | 570 µs | 140× | 28.4× | 4.94× |
| resampling (cv) | 27.8 s | 2.18 s | 2.18 s | 12.7× | 12.8× | 0.996× |
| tune | 611 s | 42.4 s | 46.3 s | 13.2× | 14.4× | 0.915× |
| select_best + finalize + last_fit | 4.38 s | 255 ms | 272 ms | 16.1× | 17.2× | 0.937× |
| wall (cold process) | 783 s | 53.6 s | 62.4 s | 12.6× | 14.6× | 0.859× |
| peak working set | 579 MB | 138 MB | 67.2 MB | 8.62× | 4.19× | 2.06× |
| allocations, workflow fit (count) | — | — | 535 | — | — | — |
| allocations, workflow fit (MB) | 136 MB | — | 61 MB | 2.23× | — | — |
| allocations, cv (count) | — | — | 7,221 | — | — | — |
| allocations, cv (MB) | — | — | 787 MB | — | — | — |

tune_candidates: `20`; select_params: `[["cost_complexity", 0.0001], ["min_n", 20.0]]`; R select_params: `{"cost_complexity": 0.0001, "min_n": 2}`

#### synth_100000_xgb_clf (100000 rows, boost_tree, 1 thread, 10 resamples) — trials: {'ok': 30}, n(R/Py/Rust)=10/10/10

| phase | R | Python wrapper | Rust native | R / Rust | R / Python | Python / Rust |
|---|---|---|---|---|---|---|
| library load / import | 1.99 s | 512 ms | — | — | 3.89× | — |
| CSV load -> table | 667 ms | 145 ms | 172 ms | 3.87× | 4.61× | 0.84× |
| workflow fit, cold | 808 ms | 388 ms | 434 ms | 1.86× | 2.08× | 0.896× |
| workflow fit, warm (median of 30) | 774 ms | 391 ms | 392 ms | 1.97× | 1.98× | 0.997× |
| model-only fit, cold | 692 ms | — | 386 ms | 1.79× | — | — |
| predict | 72.6 ms | 45.5 ms | 46.2 ms | 1.57× | 1.6× | 0.985× |
| resampling (cv) | 13 s | 5.08 s | 5.01 s | 2.6× | 2.56× | 1.01× |
| wall (cold process) | 72 s | 22 s | 32.5 s | 2.21× | 3.27× | 0.677× |
| peak working set | 559 MB | 145 MB | 72.2 MB | 7.74× | 3.85× | 2.01× |
| allocations, workflow fit (count) | — | — | 51,074 | — | — | — |
| allocations, workflow fit (MB) | 87.9 MB | — | 674 MB | 0.13× | — | — |
| allocations, cv (count) | — | — | 637,187 | — | — | — |
| allocations, cv (MB) | — | — | 8.3e+03 MB | — | — | — |

#### synth_100000_xgb_clf (100000 rows, boost_tree, 4 threads, 10 resamples) — trials: {'ok': 30}, n(R/Py/Rust)=10/10/10

| phase | R | Python wrapper | Rust native | R / Rust | R / Python | Python / Rust |
|---|---|---|---|---|---|---|
| library load / import | 1.92 s | 518 ms | — | — | 3.71× | — |
| CSV load -> table | 630 ms | 144 ms | 174 ms | 3.63× | 4.38× | 0.828× |
| workflow fit, cold | 813 ms | 420 ms | 424 ms | 1.92× | 1.94× | 0.989× |
| workflow fit, warm (median of 30) | 779 ms | 389 ms | 391 ms | 1.99× | 2× | 0.993× |
| model-only fit, cold | 704 ms | — | 382 ms | 1.84× | — | — |
| predict | 69.4 ms | 48 ms | 42.5 ms | 1.63× | 1.44× | 1.13× |
| resampling (cv) | 14 s | 2.57 s | 2.51 s | 5.57× | 5.43× | 1.02× |
| wall (cold process) | 74.6 s | 19.5 s | 30.1 s | 2.48× | 3.83× | 0.648× |
| peak working set | 570 MB | 238 MB | 165 MB | 3.45× | 2.39× | 1.44× |
| allocations, workflow fit (count) | — | — | 51,074 | — | — | — |
| allocations, workflow fit (MB) | 87.9 MB | — | 674 MB | 0.13× | — | — |
| allocations, cv (count) | — | — | 637,241 | — | — | — |
| allocations, cv (MB) | — | — | 8.3e+03 MB | — | — | — |

#### synth_10000_reg_boot (10000 rows, linear_reg, 1 thread, 25 resamples) — trials: {'ok': 30}, n(R/Py/Rust)=10/10/10

| phase | R | Python wrapper | Rust native | R / Rust | R / Python | Python / Rust |
|---|---|---|---|---|---|---|
| library load / import | 2 s | 560 ms | — | — | 3.57× | — |
| CSV load -> table | 33.7 ms | 16.9 ms | 18.6 ms | 1.81× | 2× | 0.906× |
| workflow fit, cold | 65.4 ms | 3.54 ms | 3.92 ms | 16.7× | 18.5× | 0.903× |
| workflow fit, warm (median of 30) | 57.3 ms | 3.35 ms | 3.65 ms | 15.7× | 17.1× | 0.918× |
| model-only fit, cold | 11.2 ms | — | 1.76 ms | 6.38× | — | — |
| predict | 11.5 ms | 184 µs | 28.4 µs | 403× | 62.4× | 6.46× |
| resampling (cv) | 3.97 s | 127 ms | 139 ms | 28.6× | 31.1× | 0.918× |
| wall (cold process) | 10.9 s | 962 ms | 401 ms | 27.1× | 11.3× | 2.4× |
| peak working set | 433 MB | 87.2 MB | 15.1 MB | 28.7× | 4.97× | 5.78× |
| allocations, workflow fit (count) | — | — | 445 | — | — | — |
| allocations, workflow fit (MB) | 13.7 MB | — | 5.27 MB | 2.59× | — | — |
| allocations, cv (count) | — | — | 15,819 | — | — | — |
| allocations, cv (MB) | — | — | 226 MB | — | — | — |

#### synth_10000_reg_repcv (10000 rows, linear_reg, 1 thread, 30 resamples) — trials: {'ok': 30}, n(R/Py/Rust)=10/10/10

| phase | R | Python wrapper | Rust native | R / Rust | R / Python | Python / Rust |
|---|---|---|---|---|---|---|
| library load / import | 1.9 s | 509 ms | — | — | 3.73× | — |
| CSV load -> table | 35.9 ms | 15.2 ms | 16 ms | 2.24× | 2.37× | 0.948× |
| workflow fit, cold | 63.5 ms | 3.48 ms | 3.46 ms | 18.3× | 18.2× | 1.01× |
| workflow fit, warm (median of 30) | 59.7 ms | 2.97 ms | 3.45 ms | 17.3× | 20.1× | 0.861× |
| model-only fit, cold | 10.7 ms | — | 1.74 ms | 6.18× | — | — |
| predict | 11 ms | 145 µs | 23.1 µs | 477× | 76× | 6.27× |
| resampling (cv) | 4.42 s | 125 ms | 139 ms | 31.9× | 35.2× | 0.904× |
| wall (cold process) | 11.2 s | 917 ms | 446 ms | 25.1× | 12.2× | 2.06× |
| peak working set | 429 MB | 86.7 MB | 14.5 MB | 29.6× | 4.95× | 5.98× |
| allocations, workflow fit (count) | — | — | 445 | — | — | — |
| allocations, workflow fit (MB) | 13.7 MB | — | 5.27 MB | 2.59× | — | — |
| allocations, cv (count) | — | — | 18,979 | — | — | — |
| allocations, cv (MB) | — | — | 227 MB | — | — | — |

#### synth_10000_reg_wfreq (10000 rows, linear_reg, 1 thread, 10 resamples) — trials: {'ok': 30}, n(R/Py/Rust)=10/10/10

| phase | R | Python wrapper | Rust native | R / Rust | R / Python | Python / Rust |
|---|---|---|---|---|---|---|
| library load / import | 1.91 s | 566 ms | — | — | 3.37× | — |
| CSV load -> table | 35.7 ms | 18 ms | 17.3 ms | 2.06× | 1.99× | 1.04× |
| workflow fit, cold | 111 ms | 4.18 ms | 4.7 ms | 23.7× | 26.6× | 0.89× |
| workflow fit, warm (median of 30) | 72.9 ms | 4.04 ms | 4.41 ms | 16.6× | 18× | 0.917× |
| model-only fit, cold | 19.3 ms | — | 1.83 ms | 10.5× | — | — |
| predict | 11.4 ms | 200 µs | 22.7 µs | 502× | 56.8× | 8.84× |
| resampling (cv) | 1.74 s | 53.7 ms | 58.2 ms | 29.9× | 32.4× | 0.922× |
| wall (cold process) | 10.1 s | 931 ms | 380 ms | 26.5× | 10.8× | 2.45× |
| peak working set | 400 MB | 86.5 MB | 13.9 MB | 28.7× | 4.62× | 6.21× |
| allocations, workflow fit (count) | — | — | 545 | — | — | — |
| allocations, workflow fit (MB) | 28.2 MB | — | 7.23 MB | 3.9× | — | — |
| allocations, cv (count) | — | — | 7,516 | — | — | — |
| allocations, cv (MB) | — | — | 106 MB | — | — | — |

#### synth_10000_rf_clf (10000 rows, rand_forest, 1 thread, 10 resamples) — trials: {'ok': 30}, n(R/Py/Rust)=10/10/10

| phase | R | Python wrapper | Rust native | R / Rust | R / Python | Python / Rust |
|---|---|---|---|---|---|---|
| library load / import | 2 s | 583 ms | — | — | 3.43× | — |
| CSV load -> table | 33.4 ms | 21.3 ms | 18.9 ms | 1.77× | 1.57× | 1.13× |
| workflow fit, cold | 1.19 s | 1.34 s | 1.37 s | 0.871× | 0.889× | 0.98× |
| workflow fit, warm (median of 30) | 1.14 s | 1.3 s | 1.33 s | 0.856× | 0.878× | 0.975× |
| model-only fit, cold | 1.26 s | — | 1.33 s | 0.943× | — | — |
| predict | 287 ms | 104 ms | 94.8 ms | 3.03× | 2.75× | 1.1× |
| resampling (cv) | 16.7 s | 16.6 s | 17 s | 0.983× | 1× | 0.979× |
| wall (cold process) | 97.1 s | 61.9 s | 107 s | 0.911× | 1.57× | 0.58× |
| peak working set | 635 MB | 151 MB | 154 MB | 4.12× | 4.2× | 0.982× |
| allocations, workflow fit (count) | — | — | 1,985,278 | — | — | — |
| allocations, workflow fit (MB) | 20.6 MB | — | 1.11e+03 MB | 0.0186× | — | — |
| allocations, cv (count) | — | — | 22,945,667 | — | — | — |
| allocations, cv (MB) | — | — | 1.35e+04 MB | — | — | — |

What run I shows (medians, 1 thread; "×" = R time / Rust time):

- **Decision trees (framework-overhead: rpart 4.1.23 vs its port, exact).** Workflow fit
  28× cold / 31× warm on Ames, 270× / 450× on penguins, 14× / 13× at 100k rows; 10-fold
  CV 41× / 470× / 13×; the 4 × 5 grid over 10 folds 24× / 425× / 13× (100k rows: R
  611 s, Rust 46 s, Python 42 s); `select_best` + `finalize` + `last_fit` 39× / 960× /
  16×. The ratio settles at 13× for large n: that is the port's split search against
  rpart's C code, not orchestration.
- **Boosted trees (framework-overhead: xgboost 3.2.1 hist, 1 thread, vs its port).**
  Ames (50 trees): workflow fit 4.4× cold / 4.6× warm, CV 7.8×, the 4 × 5 grid 4.1×
  (18.3 s vs 4.45 s), selection 10×. 100k rows (50 trees, logistic): fit 1.9× cold /
  2.0× warm, CV 2.6×, `predict` 1.6×. The port's histogram builder is about twice
  xgboost's at 100k rows and the rest is R's overhead.
- **Random forests (workflow-level: ranger 0.18.0 vs the ranger-rules port, 200
  trees).** Not faster. Ames: workflow fit 1.04× cold / 0.91× warm, CV 1.13×; 10k
  classification: fit 0.87× cold / 0.86× warm (R faster), CV 0.98×. Peak memory 421 vs
  41 MB and 635 vs 154 MB. The port makes 580k (Ames) to 2.0M (10k × 200 trees)
  allocations per fit where the linear and tree fits make 400–800: this is the first
  optimisation target the allocation counts point at.
- **Case weights (frequency weights on `lm`, 10k rows).** Workflow fit 24× cold / 17×
  warm, CV 30×; the weighted recipe statistics and weighted metrics cost nothing
  measurable over the unweighted task (run G: 16× / 26× / 45× on the same data; the
  warm fit is 4.4 vs 5.0 ms).
- **Resampling schemes (OLS, 10k rows).** 25 bootstraps: CV 29× (R 3.97 s vs 139 ms);
  3 × 10 repeated folds: 32× (4.42 s vs 139 ms). Per resample the costs are those of
  run G's 10-fold CV.
- **Memory.** Peak working set: R 360–635 MB, Python 77–151 MB, Rust 5–154 MB. The
  forest is the only model where the Rust process exceeds 100 MB.
- **Python wrapper.** Within ±5% of the Rust binary on the forests, the 100k-row
  fits and CV, and xgb at 2930 rows, but **not** on every phase above 100 ms, as an
  earlier version of this bullet claimed (corrected 2026-10-14 after recomputing
  every ratio from `results.csv`): Ames tree tuning 0.86×, 25 bootstraps and 3 × 10
  repeated CV 0.92× / 0.90×, 100k tree tuning 0.92×, 100k xgb cold fit 0.90× (the
  wrapper faster each time). On the
  sub-10 ms phases of the tree tasks it reads 15–32% *faster* than the binary (Ames
  workflow fit 3.7 vs 4.7 ms cold, 3.1 vs 3.7 ms warm; penguins CV 2.3 vs 3.4 ms)
  although both call the same native code and the driver's allocator issue is fixed.
  The leading explanation is that a 50–100 ms Rust process never reaches the clock
  frequency the warmed-up Python process runs at; it is not verified and the Rust
  column is the conservative one. `predict` costs the wrapper 1.3–9× the binary's
  microseconds for the NumPy conversion.

- **4 threads** (run I, 361 trials in all). `ames_tree_reg`: the 4 × 5 grid drops from
  1.07 s to 326 ms in Rust (3.3×) and from 26.1 s to 9.06 s in R (2.9×, R's best
  parallel scaling in these runs because 200 rpart fits amortise the worker start), so
  the ratio stays at 28×; CV 376× because R's 8.65 s is almost entirely worker
  start-up. `synth_100000_xgb_clf`: Rust CV 5.01 s → 2.51 s (2.0×), R 13.0 → 14.0 s
  (no gain), ratio 5.6×; the warm workflow fit is single-threaded on both sides and
  unchanged (779 vs 391 ms).

#### Run J: workflow sets (2026-10-14 07:22–07:36, idle)

Each task's set is one recipe × three models: the task's base model (resampled), a
decision tree with a 3-point `cost_complexity` grid, and a 20-tree booster; `wfs` =
`workflow_map` over the 10 folds + `rank_results(select_best = TRUE)` + `fit_best`.
3 tasks at 1 thread and `penguins_clf_wfs` at 4 threads, 120 of 120 trials ok, and the
ranked winner agrees between R and the ports on every trial (`rec_xgb` on Ames and
penguins, `rec_base` = `lm` on the synthetic data).

Full per-phase tables (medians over 10 cold trials):

#### ames_reg_wfs (2930 rows, linear_reg, 1 thread, 10 resamples) — trials: {'ok': 30}, n(R/Py/Rust)=10/10/10

| phase | R | Python wrapper | Rust native | R / Rust | R / Python | Python / Rust |
|---|---|---|---|---|---|---|
| library load / import | 1.9 s | 533 ms | — | — | 3.56× | — |
| CSV load -> table | 8.3 ms | 9.07 ms | 4.93 ms | 1.68× | 0.915× | 1.84× |
| workflow fit, cold | 54.1 ms | 2.28 ms | 2.81 ms | 19.2× | 23.8× | 0.809× |
| workflow fit, warm (median of 30) | 50.3 ms | 2.42 ms | 2.61 ms | 19.2× | 20.8× | 0.926× |
| model-only fit, cold | 17.1 ms | — | 1.98 ms | 8.62× | — | — |
| predict | 9.82 ms | 108 µs | 12.9 µs | 761× | 91.2× | 8.34× |
| resampling (cv) | 1.35 s | 29.7 ms | 33.9 ms | 40× | 45.6× | 0.876× |
| workflow set (map + rank + fit_best) | 7.26 s | 254 ms | 284 ms | 25.5× | 28.5× | 0.895× |
| wall (cold process) | 14.7 s | 1.03 s | 511 ms | 28.8× | 14.3× | 2.02× |
| peak working set | 411 MB | 82 MB | 10.6 MB | 38.6× | 5.01× | 7.7× |
| allocations, workflow fit (count) | — | — | 679 | — | — | — |
| allocations, workflow fit (MB) | 8.25 MB | — | 2.96 MB | 2.79× | — | — |
| allocations, cv (count) | — | — | 9,747 | — | — | — |
| allocations, cv (MB) | — | — | 38.3 MB | — | — | — |

wfs_best: `rec_xgb`; R wfs_best: `rec_xgb`

#### penguins_clf_wfs (333 rows, logistic_reg, 1 thread, 10 resamples) — trials: {'ok': 30}, n(R/Py/Rust)=10/10/10

| phase | R | Python wrapper | Rust native | R / Rust | R / Python | Python / Rust |
|---|---|---|---|---|---|---|
| library load / import | 1.92 s | 514 ms | — | — | 3.73× | — |
| CSV load -> table | 5.19 ms | 4.72 ms | 842 µs | 6.17× | 1.1× | 5.61× |
| workflow fit, cold | 40.4 ms | 267 µs | 278 µs | 146× | 151× | 0.961× |
| workflow fit, warm (median of 30) | 38.6 ms | 160 µs | 176 µs | 219× | 241× | 0.909× |
| model-only fit, cold | 8.82 ms | — | 275 µs | 32.1× | — | — |
| predict | 7.68 ms | 18.5 µs | 4.35 µs | 1.77e+03× | 415× | 4.25× |
| resampling (cv) | 1.32 s | 2.3 ms | 3.37 ms | 391× | 574× | 0.681× |
| workflow set (map + rank + fit_best) | 6.01 s | 18.9 ms | 22.2 ms | 271× | 318× | 0.85× |
| wall (cold process) | 12.7 s | 677 ms | 54.9 ms | 232× | 18.8× | 12.3× |
| peak working set | 367 MB | 78 MB | 5.88 MB | 62.4× | 4.71× | 13.3× |
| allocations, workflow fit (count) | — | — | 388 | — | — | — |
| allocations, workflow fit (MB) | 1.22 MB | — | 0.288 MB | 4.22× | — | — |
| allocations, cv (count) | — | — | 5,346 | — | — | — |
| allocations, cv (MB) | — | — | 3.82 MB | — | — | — |

wfs_best: `rec_xgb`; R wfs_best: `rec_xgb`

#### penguins_clf_wfs (333 rows, logistic_reg, 4 threads, 10 resamples) — trials: {'ok': 30}, n(R/Py/Rust)=10/10/10

| phase | R | Python wrapper | Rust native | R / Rust | R / Python | Python / Rust |
|---|---|---|---|---|---|---|
| library load / import | 1.92 s | 524 ms | — | — | 3.66× | — |
| CSV load -> table | 6.21 ms | 4.6 ms | 899 µs | 6.91× | 1.35× | 5.11× |
| workflow fit, cold | 43.5 ms | 275 µs | 299 µs | 145× | 158× | 0.92× |
| workflow fit, warm (median of 30) | 40.1 ms | 165 µs | 167 µs | 240× | 243× | 0.989× |
| model-only fit, cold | 9.12 ms | — | 315 µs | 29× | — | — |
| predict | 7.36 ms | 15.8 µs | 4.7 µs | 1.57e+03× | 466× | 3.36× |
| resampling (cv) | 8.29 s | 1.44 ms | 1.87 ms | 4.43e+03× | 5.77e+03× | 0.768× |
| workflow set (map + rank + fit_best) | 3.63 s | 11 ms | 15.3 ms | 238× | 330× | 0.72× |
| wall (cold process) | 18.6 s | 681 ms | 47.1 ms | 395× | 27.3× | 14.5× |
| peak working set | 366 MB | 79.1 MB | 6.95 MB | 52.7× | 4.63× | 11.4× |
| allocations, workflow fit (count) | — | — | 388 | — | — | — |
| allocations, workflow fit (MB) | 1.22 MB | — | 0.288 MB | 4.22× | — | — |
| allocations, cv (count) | — | — | 5,400 | — | — | — |
| allocations, cv (MB) | — | — | 3.85 MB | — | — | — |

wfs_best: `rec_xgb`; R wfs_best: `rec_xgb`

#### synth_10000_reg_wfs (10000 rows, linear_reg, 1 thread, 10 resamples) — trials: {'ok': 30}, n(R/Py/Rust)=10/10/10

| phase | R | Python wrapper | Rust native | R / Rust | R / Python | Python / Rust |
|---|---|---|---|---|---|---|
| library load / import | 1.93 s | 511 ms | — | — | 3.77× | — |
| CSV load -> table | 40 ms | 16.4 ms | 17.4 ms | 2.3× | 2.44× | 0.941× |
| workflow fit, cold | 67.3 ms | 3.56 ms | 3.43 ms | 19.6× | 18.9× | 1.04× |
| workflow fit, warm (median of 30) | 58.2 ms | 3.08 ms | 2.85 ms | 20.5× | 18.9× | 1.08× |
| model-only fit, cold | 12 ms | — | 1.77 ms | 6.77× | — | — |
| predict | 11.5 ms | 183 µs | 24.3 µs | 472× | 62.8× | 7.52× |
| resampling (cv) | 1.77 s | 43.2 ms | 47.6 ms | 37.3× | 41.1× | 0.907× |
| workflow set (map + rank + fit_best) | 10.3 s | 639 ms | 751 ms | 13.7× | 16.1× | 0.851× |
| wall (cold process) | 19 s | 1.45 s | 1.03 s | 18.4× | 13.1× | 1.4× |
| peak working set | 435 MB | 85.7 MB | 13.4 MB | 32.4× | 5.08× | 6.38× |
| allocations, workflow fit (count) | — | — | 445 | — | — | — |
| allocations, workflow fit (MB) | 13.7 MB | — | 5.27 MB | 2.59× | — | — |
| allocations, cv (count) | — | — | 6,336 | — | — | — |
| allocations, cv (MB) | — | — | 75.8 MB | — | — | — |

wfs_best: `rec_base`; R wfs_best: `rec_base`

- **Workflow set, 1 thread.** Ames 25× (R 7.26 s vs 284 ms), penguins 270× (6.01 s vs
  22 ms), 10k rows 14× (10.3 s vs 751 ms). The set is dominated by its booster and tree
  members, so the ratios sit between the tree (24–425×) and xgb (4–8×) results of run I,
  weighted by the 20-tree booster's share.
- **4 threads (penguins).** R 3.63 s (its `multisession` CV phase alone is 8.3 s, all
  worker start-up), Rust 15.3 ms (1.45× from 4 threads on this 333-row set), ratio
  240×.
- The Python wrapper reads 0.72–0.93× of the Rust binary's time on the sub-second
  phases here, the same unresolved pattern as in run I; both report the same winner.

#### Run H: the 1e6-row scaling point (2026-10-14 07:59–09:58)

`synth_1000000_reg` and `synth_1000000_clf` (1e6 × 14 predictors, the synthetic
generator of `benchmarks/make_scaling_data.py`), 1 thread, 10 cold trials, **10** warm
iterations (the protocol's 30 would add 25 s per Rust and 75 s per R trial for no
information), 10-fold CV, no tuning (the 20 × 10 logistic grid alone would take R about
an hour per trial), timeout 1200 s. 60 of 60 trials ok. Second attempt: the first one
(`…-h-quadratic-complement/`, 4 trials) exposed the quadratic fold complement
described in `docs/perf/2026-10-13-rust-opts-ab.md` (Rust wall 231 s on 17 s of timed
phases) and was stopped; this run uses the fixed library. Trials 7–9 of the regression
task (08:16–08:30) ran 1.5–1.8× slower on all three systems (other processes); the
medians over 10 are on the clean side and the IQRs in `report.md` show the spread.

#### synth_1000000_clf (1000000 rows, logistic_reg, 1 thread, 10 resamples) — trials: {'ok': 30}, n(R/Py/Rust)=10/10/10

| phase | R | Python wrapper | Rust native | R / Rust | R / Python | Python / Rust |
|---|---|---|---|---|---|---|
| library load / import | 2.2 s | 539 ms | — | — | 4.08× | — |
| CSV load -> table | 13.8 s | 1.37 s | 1.85 s | 7.47× | 10.1× | 0.74× |
| workflow fit, cold | 8.99 s | 2.65 s | 2.62 s | 3.43× | 3.4× | 1.01× |
| workflow fit, warm (median of 30) | 8.55 s | 2.4 s | 2.5 s | 3.41× | 3.55× | 0.96× |
| model-only fit, cold | 6.19 s | — | 1.95 s | 3.18× | — | — |
| predict | 1.13 s | 44.1 ms | 10.3 ms | 110× | 25.7× | 4.26× |
| resampling (cv) | 111 s | 30.7 s | 29.8 s | 3.72× | 3.61× | 1.03× |
| wall (cold process) | 342 s | 61.6 s | 86.3 s | 3.96× | 5.55× | 0.714× |
| peak working set | 2.94e+03 MB | 941 MB | 832 MB | 3.53× | 3.13× | 1.13× |
| allocations, workflow fit (count) | — | — | 578 | — | — | — |
| allocations, workflow fit (MB) | 3.47e+03 MB | — | 1.21e+03 MB | 2.87× | — | — |
| allocations, cv (count) | — | — | 7,836 | — | — | — |
| allocations, cv (MB) | — | — | 1.5e+04 MB | — | — | — |

#### synth_1000000_reg (1000000 rows, linear_reg, 1 thread, 10 resamples) — trials: {'ok': 30}, n(R/Py/Rust)=10/10/10

| phase | R | Python wrapper | Rust native | R / Rust | R / Python | Python / Rust |
|---|---|---|---|---|---|---|
| library load / import | 2.34 s | 606 ms | — | — | 3.86× | — |
| CSV load -> table | 11.1 s | 1.19 s | 1.93 s | 5.73× | 9.33× | 0.614× |
| workflow fit, cold | 2.94 s | 497 ms | 482 ms | 6.1× | 5.91× | 1.03× |
| workflow fit, warm (median of 30) | 2.45 s | 509 ms | 492 ms | 4.98× | 4.82× | 1.03× |
| model-only fit, cold | 870 ms | — | 244 ms | 3.56× | — | — |
| predict | 766 ms | 20.7 ms | 4.02 ms | 191× | 36.9× | 5.16× |
| resampling (cv) | 36.8 s | 7.06 s | 6.9 s | 5.33× | 5.21× | 1.02× |
| wall (cold process) | 120 s | 15.2 s | 20.2 s | 5.92× | 7.87× | 0.752× |
| peak working set | 2.92e+03 MB | 751 MB | 685 MB | 4.26× | 3.89× | 1.1× |
| allocations, workflow fit (count) | — | — | 480 | — | — | — |
| allocations, workflow fit (MB) | 1.36e+03 MB | — | 543 MB | 2.5× | — | — |
| allocations, cv (count) | — | — | 6,636 | — | — | — |
| allocations, cv (MB) | — | — | 7.11e+03 MB | — | — | — |

- **OLS at 1e6 rows.** Workflow fit 6.1× cold / 5.0× warm (R 2.45 s vs 0.49 s),
  10-fold CV 5.3× (36.8 s vs 6.9 s), `predict` on 250k rows 191×, CSV parsing 5.7×
  (11.1 s vs 1.9 s). Peak working set 2.92 GB vs 685 MB (4.3×); the Rust workflow fit
  requests 543 MB in 480 allocations, the CV 7.1 GB in 6.6k.
- **Logistic at 1e6 rows.** Workflow fit 3.4× (8.6 s vs 2.5 s warm), CV 3.7× (111 s vs
  30 s), memory 2.94 GB vs 832 MB (3.5×); cold wall 342 s vs 86 s.
- **Scaling of the ratio.** For the OLS workflow fit the R / Rust ratio runs 930× (342
  rows), 17× (2930), 16× (10k), 12× (100k), 6× (1e6); for 10-fold CV 2200×, 63×, 45×,
  7.5×, 5.3×. Both sides end up arithmetic-bound and single-threaded on reference BLAS;
  what remains at 1e6 is the port's fused prep + bake + QR against R's model-frame and
  `lm.fit` path.
- **Python wrapper.** Equal to the Rust binary within ±3% on fit and CV; its cold wall
  is 0.71–0.75× the binary's because pandas parses the 1e6-row CSV in 1.2–1.4 s against
  the Rust driver's 1.9 s and the Rust driver also times a model-only fit and bake.
