"""Workflow sets (ADR-0016): ids and order as workflowsets builds them, mapping over
shared resamples, collect_metrics / rank_results / fit_best."""
from __future__ import annotations

import numpy as np
import pytest

import tidyrustmodels as trm


def _data(n=48):
    rng = np.random.default_rng(11)
    x = rng.normal(size=n)
    z = rng.normal(size=n)
    y = 1.0 + 2.0 * x - 0.5 * z + rng.normal(scale=0.1, size=n)
    return trm.Table.from_dict({"y": y, "x": x, "z": z})


def test_ids_follow_workflowsets():
    rec = trm.Recipe("y")
    s = trm.workflow_set([("zeta", rec), ("alpha", rec)], [("lm", trm.LinearReg()), ("tree", trm.DecisionTree())])
    assert s.ids == ["zeta_lm", "zeta_tree", "alpha_lm", "alpha_tree"]
    assert len(s) == 4
    assert s.info()[1] == {"wflow_id": "zeta_tree", "preproc": "recipe", "model": "decision_tree", "tuned": False}
    # unnamed entries get class names, duplicated classes numbered by position
    u = trm.workflow_set([(None, rec), (None, rec)], [(None, trm.LinearReg()), (None, trm.DecisionTree())])
    assert u.ids == ["recipe_1_linear_reg", "recipe_1_decision_tree", "recipe_2_linear_reg", "recipe_2_decision_tree"]
    # cross=False pairs element-wise
    p = trm.workflow_set([("a", rec), ("b", rec)], [("lm", trm.LinearReg()), ("tree", trm.DecisionTree())], cross=False)
    assert p.ids == ["a_lm", "b_tree"]
    with pytest.raises(trm.InvalidArgumentError):
        trm.workflow_set([("a", rec), ("b", rec)], [("lm", trm.LinearReg())] * 3, cross=False)
    with pytest.raises(trm.InvalidArgumentError):
        s.collect_metrics()


def test_map_rank_and_fit_best():
    d = _data()
    folds = trm.vfold_cv(d.nrows, 4, seed=3)
    s = trm.workflow_set([("rec", trm.Recipe("y"))], [("lm", trm.LinearReg()), ("tree", trm.DecisionTree())])
    s.option_grid("rec_tree", [{"min_n": 2}, {"min_n": 6}, {"min_n": 10}])
    assert s.info()[1]["tuned"] is True
    s.workflow_map(d, folds, ["rmse", "rsq"])
    cm = s.collect_metrics()
    assert [(r["wflow_id"], r["config"], r["metric"]) for r in cm[:4]] == [
        ("rec_lm", "pre0_mod0_post0", "rmse"),
        ("rec_lm", "pre0_mod0_post0", "rsq"),
        ("rec_tree", "pre0_mod1_post0", "rmse"),
        ("rec_tree", "pre0_mod1_post0", "rsq"),
    ]
    assert len(cm) == 2 + 6 and all(r["n"] == 4 for r in cm)
    rr = s.rank_results("rmse")
    assert rr[0]["wflow_id"] == "rec_lm" and rr[0]["rank"] == 1 and rr[1]["rank"] == 1
    assert [r["rank"] for r in rr] == [1, 1, 2, 2, 3, 3, 4, 4]
    assert s.rank_results() == rr  # default metric: the first of the set
    best = s.rank_results("rsq", select_best=True)
    assert len(best) == 4 and best[0]["wflow_id"] == "rec_lm"
    assert s.best() == {"wflow_id": "rec_lm", "params": {}}
    f = s.fit_best(d)
    assert np.max(np.abs(f.predict(d) - d.column("y"))) < 0.5
    assert isinstance(s.extract_result("rec_tree"), trm.TuneResults)
    assert set(s.extract_result("rec_lm")) == {"per_fold", "summary"}
    assert isinstance(s.extract_workflow("rec_lm"), trm.Workflow)
    with pytest.raises(trm.InvalidArgumentError):
        s.rank_results("mae")
    with pytest.raises(trm.InvalidArgumentError):
        s.extract_result("nope")
