"""Level 5: Python bindings vs Rust (bitwise) and vs the pinned R fixtures.

Prerequisites (run from the repo root):
    maturin develop --release
    cargo test -p tidyrustmodels --test differential   # writes target/rust_outputs/*.json
    pytest python/tests
"""
from __future__ import annotations

import json
import math
import os
from pathlib import Path

import numpy as np
import pytest

import tidyrustmodels as trm

ROOT = Path(__file__).resolve().parents[2]
TASKS = json.loads((ROOT / "reference" / "tasks.json").read_text())["tasks"]
RUST_OUT = ROOT / "target" / "rust_outputs"


def _read_csv_table(path: Path, outcome: str, predictors: list[str]) -> trm.Table:
    pd = pytest.importorskip("pandas")
    df = pd.read_csv(path, keep_default_na=False, na_values=[""])
    return trm.from_pandas(df[[outcome, *predictors]])


def _recipe(task: dict) -> trm.Recipe:
    rec = trm.Recipe(task["outcome"])
    for s in task["steps"]:
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
        elif s == "nzv":
            rec = rec.step_nzv("all_predictors")
        elif s == "corr_numeric":
            rec = rec.step_corr("all_numeric_predictors")
        elif s == "other_nominal":
            rec = rec.step_other("all_nominal_predictors")
        elif s == "novel_nominal":
            rec = rec.step_novel("all_nominal_predictors")
        elif s == "unknown_nominal":
            rec = rec.step_unknown("all_nominal_predictors")
        else:
            raise ValueError(s)
    return rec


def _n_levels(data: trm.Table, outcome: str) -> int:
    return len(set(data.column(outcome)) - {None})


def _model(task, data, engine="native"):
    if task.get("model") == "decision_tree":
        return trm.DecisionTree(**task.get("model_args", {}))
    if task["type"] == "regression":
        return trm.LinearReg(engine=engine)
    if _n_levels(data, task["outcome"]) > 2:
        return trm.MultinomReg(engine=engine)
    return trm.LogisticReg(engine=engine)


def _load_task(task):
    data = _read_csv_table(
        ROOT / "reference" / "fixtures" / "datasets" / f"{task['dataset']}.csv",
        task["outcome"],
        task["predictors"],
    )
    split = json.loads((ROOT / "reference" / "fixtures" / "splits" / f"{task['id']}.json").read_text())
    fx = json.loads((ROOT / "reference" / "fixtures" / "r" / f"{task['id']}.json").read_text())
    return data, split, fx


@pytest.mark.parametrize("task", TASKS, ids=[t["id"] for t in TASKS])
@pytest.mark.parametrize("engine", ["linfa", "smartcore", "native"])
def test_python_matches_rust_bitwise(task, engine):
    """Same binary, same inputs: Python results must equal the Rust test's outputs exactly."""
    if task.get("model") == "decision_tree" and engine != "native":
        pytest.skip("decision_tree is implemented on the native engine only")
    rust_file = RUST_OUT / f"{task['id']}_{engine}.json"
    if not rust_file.exists():
        pytest.skip("run `cargo test -p tidyrustmodels --test differential` first")
    rust = json.loads(rust_file.read_text())
    data, split, _ = _load_task(task)
    train = data.select_rows(split["initial"]["train"])
    test = data.select_rows(split["initial"]["test"])
    model = _model(task, data, engine)
    fitted = trm.Workflow(_recipe(task), model).fit(train)

    coefs = fitted.coefficients()
    if rust["class_coefficients"] is not None:
        assert coefs == rust["class_coefficients"], "class coefficients differ from Rust"
    elif rust["coefficients"] is not None:
        assert list(coefs.keys()) == rust["coefficient_names"]
        assert list(coefs.values()) == rust["coefficients"], "coefficients differ from Rust"
    else:
        assert coefs == {}

    if task["type"] == "regression":
        pred = fitted.predict(test)
        assert pred.tolist() == rust["predicted_test"]
    else:
        proba = fitted.predict_proba(test)
        assert proba.shape[0] == test.nrows
        assert proba[:, 0].tolist() == rust["prob_event_test"]
        assert proba.flatten().tolist() == rust["prob_test"]
        assert fitted.predict(test) == rust["class_test"]

    folds = trm.Folds.from_assignments(split["vfold"]["fold_of_row"])
    res = trm.Workflow(_recipe(task), model).fit_resamples(data, folds, task["metrics"], threads=4)
    for ours, theirs in zip(res["per_fold"], rust["per_fold"]):
        assert ours["id"] == theirs["id"]
        for m in task["metrics"]:
            assert ours[m] == theirs[m], f"{ours['id']} {m}"


@pytest.mark.parametrize("task", TASKS, ids=[t["id"] for t in TASKS])
def test_python_matches_r_within_tolerance(task):
    """Direct check against the R fixture, using the same bounds as the Rust suite."""
    data, split, fx = _load_task(task)
    train = data.select_rows(split["initial"]["train"])
    test = data.select_rows(split["initial"]["test"])
    model = _model(task, data)
    fitted = trm.Workflow(_recipe(task), model).fit(train)
    assert fitted.feature_names() == fx["prep"]["predictor_names"]
    coefs = fitted.coefficients()
    multi = "coefficient_levels" in fx["fit"] and fx["fit"]["coefficient_levels"] is not None
    tree = fx["fit"].get("tree") is not None
    # bounds from reference/fixtures/tolerances.toml (multinom: nnet's residual, Phase 8c)
    rel = 1e-8 if task["type"] == "regression" else (1e-6 if multi else 1e-7)
    sets = ([] if tree else [(lv, coefs[lv], fx["fit"]["coefficients"][lv]) for lv in fx["fit"]["coefficient_levels"]] if multi else [(None, coefs, fx["fit"]["coefficients"])])
    if tree:
        # the tree's test predictions must match rpart to round-off
        if task["type"] == "regression":
            theirs = np.asarray(fx["fit"]["predicted_test"], dtype=float)
            assert np.max(np.abs(fitted.predict(test) - theirs) / np.maximum(1.0, np.abs(theirs))) < 1e-10
        else:
            assert fitted.predict(test) == fx["fit"]["class_test"]
    for _, ours, theirs in sets:
        for name in fx["fit"]["coefficient_names"]:
            r = theirs[name]
            if r is None:
                assert name in fitted.diagnostics()["aliased"]
                continue
            assert math.isclose(ours[name], r, rel_tol=rel, abs_tol=1e-9), name
    if task["type"] == "classification":
        assert tree or fitted.diagnostics()["converged"] is True
        proba = fitted.predict_proba(test)
        levels = fitted.levels(test)
        assert levels == fx["fit"]["outcome_levels"]
        for j, lv in enumerate(levels):
            theirs = np.asarray(fx["fit"]["predicted_test"][f".pred_{lv}"], dtype=float)
            assert np.max(np.abs(proba[:, j] - theirs)) < 1e-7


def test_errors_map_to_exception_hierarchy():
    t = trm.Table.from_dict({"y": np.array([1.0, 2.0, 3.0]), "x": np.array([1.0, 2.0, 3.0])})
    with pytest.raises(trm.SchemaError):
        trm.Recipe("nope").step_normalize("all_numeric_predictors")
        trm.Workflow(trm.Recipe("nope"), trm.LinearReg()).fit(t)
    with pytest.raises(trm.UnsupportedError):
        trm.Workflow(trm.Recipe("y"), trm.LogisticReg()).fit(t)
    with pytest.raises(trm.InvalidArgumentError):
        trm.LinearReg(engine="tensorflow")
    with pytest.raises(trm.InvalidValueError):
        trm.Table.from_dict({"a": [1.0, float("nan")], "b": ["u", "v"]})  # ok: NaN is missing
        tbl = trm.Table.from_dict({"a": ["x", "y"], "y": np.array([0.0, 1.0])})
        trm.Workflow(trm.Recipe("y"), trm.LinearReg()).fit(tbl)  # categorical predictor without dummy
    assert issubclass(trm.BackendError, trm.TidyRustModelsError)


def test_numpy_and_dict_inputs_agree():
    rng = np.random.default_rng(0)
    x = rng.normal(size=(50, 3))
    y = x @ np.array([1.0, -2.0, 0.5]) + 0.3
    a = trm.Table.from_numpy(np.column_stack([y, x]), ["y", "a", "b", "c"])
    b = trm.Table.from_dict({"y": y, "a": x[:, 0], "b": x[:, 1], "c": x[:, 2]})
    fa = trm.Workflow(trm.Recipe("y"), trm.LinearReg()).fit(a)
    fb = trm.Workflow(trm.Recipe("y"), trm.LinearReg()).fit(b)
    assert fa.coefficients() == fb.coefficients()
    assert np.allclose(list(fa.coefficients().values()), [0.3, 1.0, -2.0, 0.5], atol=1e-12)
    # Fortran-order input must give identical results to C-order input.
    fc = trm.Workflow(trm.Recipe("y"), trm.LinearReg()).fit(
        trm.Table.from_numpy(np.asfortranarray(np.column_stack([y, x])), ["y", "a", "b", "c"])
    )
    assert fc.coefficients() == fa.coefficients()


def test_round_trip_and_tuning():
    rng = np.random.default_rng(1)
    x = rng.normal(size=(80, 2))
    p = 1 / (1 + np.exp(-(0.5 + 1.5 * x[:, 0] - x[:, 1])))
    cls = ["yes" if u < q else "no" for u, q in zip(rng.uniform(size=80), p)]
    t = trm.Table.from_dict({"cls": cls, "a": x[:, 0], "b": x[:, 1]})
    wf = trm.Workflow(trm.Recipe("cls").step_normalize("all_numeric_predictors"), trm.LogisticReg())
    fitted = wf.fit(t)
    back = trm.FittedWorkflow.from_json(fitted.to_json())
    assert np.array_equal(back.predict_proba(t), fitted.predict_proba(t))
    folds = trm.vfold_cv(80, 5, seed=7, strata=cls)
    assert len(folds) == 5
    grid = trm.grid_regular([("penalty", -3.0, 0.0, True)], 3)
    res = trm.tune_grid(wf, t, folds, grid, ["accuracy", "mn_log_loss"], threads=2)
    assert len(res) == 3
    assert all(c["results"]["summary"][1]["metric"] == "mn_log_loss" for c in res)


def _close(a, b, tol=1e-12):
    if a is None or b is None:
        return a is None and b is None
    return abs(a - b) <= tol * max(1.0, abs(a), abs(b))


def test_class_metrics_match_r_fixed_arrays():
    """yardstick on fixed arrays (binary with ties, 3-class with an unpredicted and an
    absent level) through the Python entry points."""
    fx = json.loads((ROOT / "reference" / "fixtures" / "r" / "metrics_fixed.json").read_text())
    binary = ["sens", "spec", "precision", "recall", "f_meas", "bal_accuracy", "j_index", "kap", "mcc",
              "accuracy", "roc_auc", "pr_auc", "average_precision", "brier_class", "mn_log_loss"]
    for section in ["binary_full", "binary_ties"]:
        s = fx[section]
        probs = np.column_stack([s["prob_event"], 1.0 - np.asarray(s["prob_event"])])
        ours = trm.class_metrics(s["truth"], s["levels"], binary, estimate=s["estimate"], probabilities=probs)
        for m in binary:
            assert _close(ours[m], s[m]), (section, m, ours[m], s[m])
        assert trm.conf_mat(s["truth"], s["estimate"], s["levels"]) == s["conf_mat"]
    for section in ["multiclass", "multiclass_absent_level"]:
        s = fx[section]
        probs = np.asarray(s["probs"], dtype=float)
        names = ["sens", "spec", "precision", "recall", "f_meas", "bal_accuracy", "j_index"]
        ours = trm.class_metrics(s["truth"], s["levels"], names + ["kap", "mcc", "accuracy", "roc_auc", "pr_auc",
                                 "average_precision", "brier_class", "mn_log_loss"], estimate=s["estimate"], probabilities=probs)
        for m in names:  # yardstick default for > 2 levels is macro
            assert _close(ours[m], s[m]["macro"]), (section, m)
        assert _close(ours["roc_auc"], s["roc_auc"]["hand_till"])
        assert _close(ours["pr_auc"], s["pr_auc"]["macro"])
        assert _close(ours["average_precision"], s["average_precision"]["macro"])
        for m in ["kap", "mcc", "accuracy", "brier_class", "mn_log_loss"]:
            assert _close(ours[m], s[m]), (section, m)
        assert trm.conf_mat(s["truth"], s["estimate"], s["levels"]) == s["conf_mat"]
    # argmax default for `estimate`
    s = fx["multiclass"]
    auto = trm.class_metrics(s["truth"], s["levels"], ["accuracy"], probabilities=np.asarray(s["probs"]))
    assert 0.0 <= auto["accuracy"] <= 1.0
    with pytest.raises(trm.InvalidValueError):
        trm.class_metrics(["q"], ["a", "b"], ["accuracy"], estimate=["a"])
