"""tune's selection rules, finalize_workflow, last_fit and fit_best from Python (ADR-0014).

The differential check against R lives in the Rust suite (Phase 8i); these tests
exercise the bindings on a small synthetic problem and the documented rule semantics.
"""
from __future__ import annotations

import numpy as np
import pytest

import tidyrustmodels as trm


def _problem():
    rng = np.random.default_rng(3)
    x = rng.normal(size=(120, 2))
    y = 1.0 + 2.0 * x[:, 0] - x[:, 1] + 0.3 * x[:, 0] ** 2 + rng.normal(scale=0.4, size=120)
    t = trm.Table.from_dict({"y": y, "a": x[:, 0], "b": x[:, 1]})
    wf = trm.Workflow(trm.Recipe("y"), trm.DecisionTree())
    folds = trm.vfold_cv(120, 4, seed=9)
    grid = [{"cost_complexity": cp, "min_n": mn} for mn in (2, 5, 10) for cp in (0.001, 0.01, 0.1)]
    return t, wf, folds, grid


def test_tune_results_indexing_and_show_best():
    t, wf, folds, grid = _problem()
    res = trm.tune_grid(wf, t, folds, grid, ["rmse", "rsq"], threads=2)
    assert isinstance(res, trm.TuneResults) and len(res) == 9
    assert res[0]["params"] == grid[0]
    assert [c["results"]["summary"][0]["metric"] for c in res] == ["rmse"] * 9
    best = res.show_best("rmse", n=3)
    assert len(best) == 3 and best[0]["mean"] <= best[1]["mean"] <= best[2]["mean"]
    assert best[0]["n"] == 4
    assert res.select_best("rmse")["index"] == best[0]["index"]
    with pytest.raises(trm.InvalidArgumentError):
        res.select_best("mae")


def test_selection_rules_and_final_fits():
    t, wf, folds, grid = _problem()
    res = trm.tune_grid(wf, t, folds, grid, ["rmse"], threads=2)
    best = res.select_best("rmse")
    one_se = res.select_by_one_std_err("rmse", ["desc(cost_complexity)", "desc(min_n)"])
    pct = res.select_by_pct_loss("rmse", ["desc(cost_complexity)", "desc(min_n)"], limit=2.0)
    means = {c["index"]: c["mean"] for c in res.show_best("rmse", n=9)}
    ses = {c["index"]: c["std_err"] for c in res.show_best("rmse", n=9)}
    # the one-SE pick is within one standard error of the best and no more complex
    # (larger cost_complexity, or equal cost_complexity and larger min_n) than any
    # other candidate within the bound
    bound = means[best["index"]] + ses[best["index"]]
    assert means[one_se["index"]] <= bound
    for i, m in means.items():
        if m <= bound:
            assert (grid[i]["cost_complexity"], grid[i]["min_n"]) <= (
                grid[one_se["index"]]["cost_complexity"],
                grid[one_se["index"]]["min_n"],
            )
    assert abs(means[pct["index"]] - means[best["index"]]) / means[best["index"]] * 100 < 2.0
    with pytest.raises(trm.InvalidArgumentError):
        res.select_by_one_std_err("rmse", ["desc(nope)"])
    # finalize + last_fit (tune's fold path) and fit_best (fit on every row)
    train, test = trm.initial_split(120, 0.75, seed=1)
    final = wf.finalize(best["params"])
    lf = final.last_fit(t, list(train), list(test), ["rmse", "mae"])
    assert set(lf["metrics"]) == {"rmse", "mae"} and lf["predictions"].shape == (30,)
    assert isinstance(lf["workflow"], trm.FittedWorkflow)
    assert np.array_equal(lf["workflow"].predict(t.select_rows(list(test))), lf["predictions"])
    fb = res.fit_best(wf, t, "rmse")
    assert fb.predict(t).shape == (120,)
    # a classification last_fit returns classes and probabilities
    cls = ["p" if v > 1.5 else "q" for v in t.column("y")]
    tc = trm.Table.from_dict({"c": cls, "a": t.column("a"), "b": t.column("b")})
    lfc = trm.Workflow(trm.Recipe("c"), trm.DecisionTree()).last_fit(tc, list(train), list(test), ["accuracy"])
    assert set(lfc["predictions"]["class"]) <= {"p", "q"} and lfc["predictions"]["prob"].shape == (30, 2)
