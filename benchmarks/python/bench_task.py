"""One benchmark trial of the tidyrustmodels *Python wrapper* in a fresh process.

Same CLI contract and `BENCH_RESULT {json}` line as benchmarks/rust and benchmarks/r.
Measures what a Python user pays: interpreter start, `import tidyrustmodels` (+ numpy,
pandas), pandas CSV parsing, DataFrame -> Table conversion, and every PyO3 call.

Phase definitions (shared with the other drivers):
  load_s      CSV -> in-memory table of the task's columns (pandas.read_csv + from_pandas)
  wf_fit_s    Workflow.fit on the raw training table (prep + bake + model fit)
  fit_s       model fit on an already-baked design: NOT separable through the Python API
              (Recipe has no public prep/bake), reported as null
  predict_s   FittedWorkflow.predict on the test table (bake + predict)
  cv_s        Workflow.fit_resamples, v folds
  tune_s      tune_grid over a log10 penalty grid (classification only)
  warm        in-process repetitions of wf_fit / predict

Usage:
    python -I benchmarks/python/bench_task.py --dataset <csv> --outcome <col>
        --type regression|classification --predictors a,b,c --steps normalize_numeric,dummy_nominal
        --metrics rmse,rsq --engine linfa|smartcore --threads <k> [--warm <n>] [--v 5]
        [--tune-levels 5] [--seed 1]
"""
from __future__ import annotations

import json
import sys
import time

t_start = time.perf_counter()
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import tidyrustmodels as trm  # noqa: E402

library_load_s = time.perf_counter() - t_start


def arg(name: str, default: str | None = None) -> str | None:
    a = sys.argv
    if name in a:
        i = a.index(name)
        if i + 1 < len(a):
            return a[i + 1]
    return default


def need(name: str) -> str:
    v = arg(name)
    if v is None:
        raise SystemExit(f"missing {name}")
    return v


def build_recipe(outcome: str, steps: list[str]) -> trm.Recipe:
    rec = trm.Recipe(outcome)
    for s in steps:
        if s == "impute_median_numeric":
            rec = rec.step_impute_median("all_numeric_predictors")
        elif s == "impute_mean_numeric":
            rec = rec.step_impute_mean("all_numeric_predictors")
        elif s == "normalize_numeric":
            rec = rec.step_normalize("all_numeric_predictors")
        elif s == "dummy_nominal":
            rec = rec.step_dummy("all_nominal_predictors")
        elif s == "":
            pass
        else:
            raise SystemExit(f"unknown step {s}")
    return rec


def median(xs: list[float]) -> float | None:
    return float(np.median(xs)) if xs else None


def main() -> None:
    dataset = need("--dataset")
    outcome = need("--outcome")
    task_type = need("--type")
    predictors = need("--predictors").split(",")
    steps = arg("--steps", "").split(",")
    metrics = need("--metrics").split(",")
    engine = need("--engine")
    threads = int(need("--threads"))
    warm = int(arg("--warm", "0"))
    v = int(arg("--v", "5"))
    tune_levels = int(arg("--tune-levels", "5"))
    seed = int(arg("--seed", "1"))
    par = threads if threads > 1 else None

    # ---- load (CSV -> DataFrame -> Table) ----
    t0 = time.perf_counter()
    df = pd.read_csv(dataset, keep_default_na=False, na_values=[""])
    df = df[[outcome, *predictors]]
    data = trm.from_pandas(df)
    load_s = time.perf_counter() - t0
    n = data.nrows

    # ---- split (own RNG; not a timed phase) ----
    strata = [None if pd.isna(x) else str(x) for x in df[outcome]] if task_type == "classification" else None
    train_rows, test_rows = trm.initial_split(n, 0.75, seed=seed, strata=strata)
    train = data.select_rows([int(i) for i in train_rows])
    test = data.select_rows([int(i) for i in test_rows])

    recipe = build_recipe(outcome, steps)
    model = trm.LinearReg(engine=engine) if task_type == "regression" else trm.LogisticReg(engine=engine)
    wf = trm.Workflow(recipe, model)

    # ---- workflow fit + predict ----
    t0 = time.perf_counter()
    fitted = wf.fit(train)
    wf_fit_s = time.perf_counter() - t0
    t0 = time.perf_counter()
    pred = fitted.predict(test)
    predict_s = time.perf_counter() - t0
    n_features = len(fitted.feature_names())

    # ---- cross-validation ----
    folds = trm.vfold_cv(n, v, seed=seed + 1, strata=strata)
    t0 = time.perf_counter()
    cv = wf.fit_resamples(data, folds, metrics, threads=par)
    cv_s = time.perf_counter() - t0
    cv_summary = [{"metric": s["metric"], "mean": s["mean"], "std_err": s["std_err"]} for s in cv["summary"]]

    # ---- tuning (classification only) ----
    tune_s = None
    tune_candidates = None
    if task_type == "classification":
        grid = trm.grid_regular([("penalty", -4.0, 0.0, True)], tune_levels)
        t0 = time.perf_counter()
        tuned = trm.tune_grid(wf, data, folds, grid, metrics, threads=par)
        tune_s = time.perf_counter() - t0
        tune_candidates = len(tuned)

    # ---- warm repetitions ----
    w_wffit: list[float] = []
    w_predict: list[float] = []
    for _ in range(warm):
        t0 = time.perf_counter()
        f = wf.fit(train)
        w_wffit.append(time.perf_counter() - t0)
        t0 = time.perf_counter()
        p = f.predict(test)
        w_predict.append(time.perf_counter() - t0)
        del p

    out = {
        "system": "python",
        "engine": engine,
        "threads": threads,
        "n_rows": n,
        "n_train": train.nrows,
        "n_features": n_features,
        "phases": {
            "load_s": load_s, "prep_s": None, "bake_s": None, "fit_s": None, "wf_fit_s": wf_fit_s,
            "predict_s": predict_s, "cv_s": cv_s, "tune_s": tune_s, "library_load_s": library_load_s,
        },
        "cv_summary": cv_summary,
        "tune_candidates": tune_candidates,
        "warm": {
            "iters": warm,
            "prep_bake_s_median": None,
            "fit_s_median": None,
            "wf_fit_s_median": median(w_wffit),
            "predict_s_median": median(w_predict),
        },
        "versions": {
            "python": sys.version.split()[0], "numpy": np.__version__, "pandas": pd.__version__,
            "tidyrustmodels": getattr(trm, "__version__", "unknown"),
        },
    }
    del pred
    print("BENCH_RESULT " + json.dumps(out))


if __name__ == "__main__":
    main()
