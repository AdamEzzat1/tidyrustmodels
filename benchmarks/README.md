# Benchmarks

Methodology and results live in `../BENCHMARKS.md`. This directory holds the drivers.

```
rust/                 trm-bench binary: one cold trial of tidyrustmodels per process
r/bench_task.R        one cold trial of R tidymodels per process (same CLI contract)
run_benchmarks.py     orchestrator: spawns trials, samples peak RSS, writes results.csv
report.py             results.csv -> report.md in the BENCHMARKS.md table format
make_scaling_data.py  synthetic datasets at 1e3 / 1e4 / 1e5 rows (gitignored)
results/<run_id>/     results.csv, env.json, report.md (committed per run)
```

Run (from the repository root, inside the project venv):

```
cargo build --release --manifest-path benchmarks/rust/Cargo.toml
python -I benchmarks/make_scaling_data.py --sizes 1000,10000,100000
python -I benchmarks/run_benchmarks.py --run-id <id> --trials 5 --warm 10 \
    --tasks penguins_reg,penguins_clf,iris_clf,ames_reg,synth_1000_reg,synth_10000_reg,synth_100000_reg,synth_10000_clf,synth_100000_clf \
    --threads 1,4
python -I benchmarks/report.py --run-id <id>
```

Both drivers accept the same flags (`--dataset --outcome --type --predictors --steps
--metrics --threads --warm`) and print one `BENCH_RESULT {json}` line. Phases: load,
prep, bake, fit, predict, 5-fold cv, tune (classification only), plus warm in-process
repetitions of prep+bake / fit / predict.

What is *not* controlled, and is therefore reported rather than hidden:
- Split and fold membership differ between R and Rust (each uses its own seeded RNG);
  sizes are identical. The differential suite, not the benchmark, establishes parity.
- R "tune" uses glmnet ridge (tidymodels has no penalised `glm`); Rust tunes an unscaled
  L2 penalty with L-BFGS. Workflow-level comparison only.
- R loads tidymodels in every cold trial (seconds); that is genuine cold-start cost and
  is included in wall time and reported separately as `library_load_s`.
- Peak RSS is sampled every 1 ms over the process tree; processes that exit in under
  ~150 ms are never observed and their memory cell is `not captured`. Known fix (not yet
  done): on Windows read `PeakWorkingSetSize` via `GetProcessMemoryInfo` on the handle
  `subprocess.Popen` keeps open after exit; on Linux use `resource.getrusage(RUSAGE_CHILDREN)`.
- `--threads 4` for R means `future::plan(multisession, workers = 4)`, whose worker
  start-up (each worker loads tidymodels) is inside the timed `cv_s`/`tune_s`.
