# Benchmarks

Methodology and results live in `../BENCHMARKS.md`. This directory holds the drivers.

```
rust/                 trm-bench binary: one cold trial of tidyrustmodels per process
r/bench_task.R        one cold trial of R tidymodels per process (same CLI contract)
python/bench_task.py  one cold trial of the tidyrustmodels *Python wrapper* per process (same contract)
_rusage_wrap.py       POSIX helper: exact child peak RSS via ru_maxrss (unused on Windows)
run_benchmarks.py     orchestrator: spawns trials, records exact peak memory, writes results.csv
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
    --threads 1,4 --systems r,rust,python
python -I benchmarks/report.py --run-id <id>
```

Both drivers accept the same flags (`--dataset --outcome --type --predictors --steps
--metrics --threads --warm`) and print one `BENCH_RESULT {json}` line. Phases: load,
prep, bake, fit (model only, on the baked design; R and Rust), wf_fit (workflow fit from
the raw training table: prep + bake + fit; all three), predict, 5-fold cv, tune
(classification only), plus warm in-process repetitions. The Python driver cannot time
prep/bake/model-fit separately (the Python API has no public bake) and reports them null.

What is *not* controlled, and is therefore reported rather than hidden:
- Split and fold membership differ between R and Rust (each uses its own seeded RNG);
  sizes are identical. The differential suite, not the benchmark, establishes parity.
- R "tune" uses glmnet ridge (tidymodels has no penalised `glm`); Rust tunes an unscaled
  L2 penalty with L-BFGS. Workflow-level comparison only.
- R loads tidymodels in every cold trial (seconds); that is genuine cold-start cost and
  is included in wall time and reported separately as `library_load_s`.
- Peak memory is reported two ways. `peak_wset_mb` is exact: on Windows the orchestrator
  retains a query handle for every process it sees in the tree and reads
  `PeakWorkingSetSize` after exit (needed because `Rscript.exe` and a venv `python.exe`
  are launchers whose work happens in a child); on POSIX a wrapper reports `ru_maxrss` of
  `RUSAGE_CHILDREN` (untested so far). `peak_rss_tree_mb` is the 1 ms-sampled sum over
  the tree and is a lower bound that misses processes shorter than a few samples.
- `--threads 4` for R means `future::plan(multisession, workers = 4)`, whose worker
  start-up (each worker loads tidymodels) is inside the timed `cv_s`/`tune_s`.
