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
  cv_s        Workflow.fit_resamples over the chosen resampling scheme
  tune_s      tune_grid (same explicit grids as the other drivers)
  select_s    select_best + finalize + last_fit on the initial split
  wfs_s       workflow set (base model, tuned tree, xgboost): map + rank + fit_best
  warm        in-process repetitions of wf_fit / predict

Usage:
    python -I benchmarks/python/bench_task.py --dataset <csv> --outcome <col>
        --type regression|classification --predictors a,b,c --steps normalize_numeric,dummy_nominal
        --metrics rmse,rsq --engine native|linfa|smartcore --threads <k>
        [--model ...] [--model-args k=v,...] [--case-weights col:kind]
        [--resampling vfold|repeated|bootstraps] [--v 10] [--repeats 3] [--times 25]
        [--tune none|grid] [--tune-levels 20] [--select] [--wfs] [--warm <n>] [--seed 1]
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
        elif s == "zv":
            rec = rec.step_zv("all_predictors")
        elif s == "":
            pass
        else:
            raise SystemExit(f"unknown step {s}")
    return rec


def build_model(name: str, a: dict, engine: str, seed: int):
    if name == "linear_reg":
        return trm.LinearReg(engine=engine)
    if name == "logistic_reg":
        return trm.LogisticReg(engine=engine)
    if name == "decision_tree":
        kw = {k: (int(v) if k in ("tree_depth", "min_n") else v) for k, v in a.items() if k in ("cost_complexity", "tree_depth", "min_n")}
        return trm.DecisionTree(**kw)
    if name == "rand_forest":
        kw = {k: int(v) for k, v in a.items() if k in ("trees", "mtry", "min_n")}
        return trm.RandForest(seed=seed, **kw)
    if name == "boost_tree":
        kw = {k: (int(v) if k in ("trees", "tree_depth") else v) for k, v in a.items() if k in ("trees", "tree_depth", "learn_rate", "min_n")}
        return trm.BoostTree(**kw)
    raise SystemExit(f"unknown model {name}")


def tune_grid_for(name: str, levels: int):
    if name == "logistic_reg":
        return trm.grid_regular([("penalty", -4.0, 0.0, True)], levels)
    if name == "decision_tree":
        return [{"cost_complexity": cp, "min_n": mn} for cp in (1e-4, 1e-3, 1e-2, 1e-1) for mn in (2, 5, 10, 15, 20)]
    if name == "boost_tree":
        return [{"tree_depth": d, "learn_rate": lr} for d in (2, 4, 6, 8) for lr in (0.05, 0.1, 0.2, 0.3, 0.5)]
    return None


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
    model_name = arg("--model", "linear_reg" if task_type == "regression" else "logistic_reg")
    margs = {}
    for kv in (arg("--model-args", "") or "").split(","):
        if "=" in kv:
            k, v = kv.split("=", 1)
            margs[k] = float(v)
    cw = arg("--case-weights")
    cw_col, cw_kind = (None, None)
    if cw:
        cw_col, _, cw_kind = cw.partition(":")
        cw_kind = cw_kind or "frequency"
    resampling = arg("--resampling", "vfold")
    warm = int(arg("--warm", "0"))
    v = int(arg("--v", "5"))
    repeats = int(arg("--repeats", "1"))
    times = int(arg("--times", "25"))
    tune_mode = arg("--tune", "grid")
    tune_levels = int(arg("--tune-levels", "5"))
    do_select = "--select" in sys.argv
    do_wfs = "--wfs" in sys.argv
    seed = int(arg("--seed", "1"))
    par = threads if threads > 1 else None

    # ---- load (CSV -> DataFrame -> Table) ----
    t0 = time.perf_counter()
    df = pd.read_csv(dataset, keep_default_na=False, na_values=[""])
    df = df[[outcome, *predictors, *([cw_col] if cw_col else [])]]
    data = trm.from_pandas(df)
    load_s = time.perf_counter() - t0
    n = data.nrows

    # ---- split (own RNG; not a timed phase) ----
    strata = [None if pd.isna(x) else str(x) for x in df[outcome]] if task_type == "classification" else None
    train_rows, test_rows = trm.initial_split(n, 0.75, seed=seed, strata=strata)
    train_rows = [int(i) for i in train_rows]
    test_rows = [int(i) for i in test_rows]
    train = data.select_rows(train_rows)
    test = data.select_rows(test_rows)

    recipe = build_recipe(outcome, steps)
    model = build_model(model_name, margs, engine, seed)
    wf = trm.Workflow(recipe, model, case_weights=cw_col, weight_kind=cw_kind or "frequency") if cw_col else trm.Workflow(recipe, model)

    # ---- workflow fit + predict ----
    t0 = time.perf_counter()
    fitted = wf.fit(train)
    wf_fit_s = time.perf_counter() - t0
    t0 = time.perf_counter()
    pred = fitted.predict(test)
    predict_s = time.perf_counter() - t0
    n_features = len(fitted.feature_names())

    # ---- resampling ----
    if resampling == "vfold":
        folds = trm.vfold_cv(n, v, seed=seed + 1, strata=strata)
    elif resampling == "repeated":
        folds = trm.vfold_cv(n, v, seed=seed + 1, strata=strata, repeats=repeats)
    elif resampling == "bootstraps":
        folds = trm.bootstraps(n, times, seed=seed + 1, strata=strata)
    else:
        raise SystemExit(f"unknown resampling {resampling}")
    t0 = time.perf_counter()
    cv = wf.fit_resamples(data, folds, metrics, threads=par)
    cv_s = time.perf_counter() - t0
    cv_summary = [{"metric": s["metric"], "mean": s["mean"], "std_err": s["std_err"]} for s in cv["summary"]]

    # ---- tuning + selection ----
    tune_s = None
    tune_candidates = None
    select_s = None
    select_params = None
    grid = tune_grid_for(model_name, tune_levels) if tune_mode == "grid" else None
    if grid is not None:
        t0 = time.perf_counter()
        tuned = trm.tune_grid(wf, data, folds, grid, metrics, threads=par)
        tune_s = time.perf_counter() - t0
        tune_candidates = len(tuned)
        if do_select:
            t0 = time.perf_counter()
            sel = tuned.select_best(metrics[0])
            lf = wf.finalize(sel["params"]).last_fit(data, train_rows, test_rows, metrics)
            select_s = time.perf_counter() - t0
            select_params = sel["params"]
            del lf

    # ---- workflow set ----
    wfs_s = None
    wfs_best = None
    if do_wfs:
        t0 = time.perf_counter()
        ws = trm.workflow_set([("rec", recipe)], [("base", model), ("tree", trm.DecisionTree()), ("xgb", trm.BoostTree(trees=20))])
        ws.option_grid("rec_tree", [{"cost_complexity": cp} for cp in (1e-3, 1e-2, 1e-1)])
        ws.workflow_map(data, folds, metrics, threads=par)
        ranked = ws.rank_results(None, True)
        best = ws.fit_best(data)
        wfs_s = time.perf_counter() - t0
        wfs_best = ranked[0]["wflow_id"]
        del best

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
        "model": model_name,
        "threads": threads,
        "n_rows": n,
        "n_train": train.nrows,
        "n_features": n_features,
        "n_resamples": len(folds),
        "phases": {
            "load_s": load_s, "prep_s": None, "bake_s": None, "fit_s": None, "wf_fit_s": wf_fit_s,
            "predict_s": predict_s, "cv_s": cv_s, "tune_s": tune_s, "select_s": select_s, "wfs_s": wfs_s,
            "library_load_s": library_load_s,
        },
        "cv_summary": cv_summary,
        "tune_candidates": tune_candidates,
        "select_params": select_params,
        "wfs_best": wfs_best,
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
