"""Resampling schemes from Python (ADR-0013): structure and rsample's id conventions.

The membership-level verification against R lives in the Rust differential suite
(Phase 8h); these tests check that the bindings expose the same generators with the
same invariants and that `Folds.from_splits` round-trips a shared rset.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

import tidyrustmodels as trm

ROOT = Path(__file__).resolve().parents[2]


def _groups_of(folds, groups):
    for k in range(len(folds)):
        train, test = folds.split(k)
        test_groups = {groups[i] for i in test}
        assert all(groups[i] not in test_groups for i in train), f"resample {k} straddles a group"


def test_names0_ids():
    assert trm.vfold_cv(23, 5, seed=1).ids == ["Fold1", "Fold2", "Fold3", "Fold4", "Fold5"]
    assert trm.vfold_cv(23, 10, seed=1).ids[0] == "Fold01"
    assert trm.bootstraps(23, 25, seed=1).ids[:2] == ["Bootstrap01", "Bootstrap02"]
    assert trm.mc_cv(23, 0.7, 3, seed=1).ids == ["Resample1", "Resample2", "Resample3"]
    assert trm.loo_cv(4, seed=1).ids == ["Resample1", "Resample2", "Resample3", "Resample4"]
    assert trm.vfold_cv(23, 5, seed=1, repeats=2).ids[5] == "Repeat2.Fold1"
    assert trm.rolling_origin(8, 3, 2).ids == ["Slice1", "Slice2", "Slice3", "Slice4"]


def test_bootstraps_draw_n_rows_and_assess_out_of_bag():
    folds = trm.bootstraps(40, 6, seed=3)
    assert len(folds) == 6 and folds.n == 40 and not folds.is_partition()
    for k in range(6):
        train, test = folds.split(k)
        assert len(train) == 40
        assert sorted(set(range(40)) - set(train.tolist())) == test.tolist()
    # stratified: sorted within each stratum, strata in label order
    strata = ["b"] * 20 + ["a"] * 20
    folds = trm.bootstraps(40, 3, seed=3, strata=strata)
    train, _ = folds.split(0)
    assert all(i >= 20 for i in train[:20]) and all(i < 20 for i in train[20:])
    assert list(train[:20]) == sorted(train[:20])


def test_mc_cv_and_validation_split_sizes():
    folds = trm.mc_cv(40, 0.7, 4, seed=5)
    assert folds.is_partition()
    for k in range(4):
        train, test = folds.split(k)
        assert len(train) == 28 and len(test) == 12
    train, val, test = trm.initial_validation_split(8, 0.5, 0.25, seed=1)
    assert (len(train), len(val), len(test)) == (4, 2, 2)
    assert sorted([*train, *val, *test]) == list(range(8))
    train, test = trm.initial_time_split(8, 0.6)
    assert train.tolist() == [0, 1, 2, 3] and test.tolist() == [4, 5, 6, 7]


def test_grouped_schemes_keep_groups_together():
    groups = [f"g{i % 7}" for i in range(50)]
    for balance in ("groups", "observations"):
        folds = trm.group_vfold_cv(groups, seed=2, v=3, balance=balance)
        assert folds.ids == ["Resample1", "Resample2", "Resample3"]
        assert folds.is_partition()
        _groups_of(folds, groups)
    loo = trm.group_vfold_cv(groups, seed=2)
    assert len(loo) == 7
    rep = trm.group_vfold_cv(groups, seed=2, v=3, repeats=2)
    assert rep.ids[3] == "Repeat2.Resample1"
    boot = trm.group_bootstraps(groups, 4, seed=2)
    _groups_of(boot, groups)
    for k in range(4):
        train, _ = boot.split(k)
        assert list(train) == sorted(train), "rows ascend, repeated per draw"
    mc = trm.group_mc_cv(groups, 0.6, 3, seed=2)
    assert mc.is_partition()
    _groups_of(mc, groups)
    train, test = trm.group_initial_split(groups, 0.7, seed=2)
    assert sorted([*train, *test]) == list(range(50))
    with pytest.raises(trm.InvalidArgumentError):
        trm.group_vfold_cv(groups, seed=2, v=3, balance="rows")


def test_time_series_windows_match_rsample_probe():
    # rsample: n = 8, initial 3, assess 2, cumulative FALSE, skip 1, lag 1
    f = trm.rolling_origin(8, 3, 2, cumulative=False, skip=1, lag=1)
    assert [tuple(map(list, f.split(k))) for k in range(len(f))] == [([0, 1, 2], [2, 3, 4]), ([2, 3, 4], [4, 5, 6])]
    # lookback 2, assess 1..2, complete, step 2, skip 1 -> (2-4|5-6), (4-6|7-8)
    s = trm.sliding_window(8, lookback=2, assess_start=1, assess_stop=2, step=2, skip=1)
    assert [tuple(map(list, s.split(k))) for k in range(len(s))] == [([1, 2, 3], [4, 5]), ([3, 4, 5], [6, 7])]
    assert len(trm.sliding_window(8, lookback=None, assess_stop=2)) == 6
    # index 1 2 2 5 6 9 10 11, lookback 3, assess 1..2: the middle window assesses nothing
    i = trm.sliding_index([1, 2, 2, 5, 6, 9, 10, 11], lookback=3.0, assess_start=1.0, assess_stop=2.0)
    assert len(i) == 3
    assert list(i.split(1)[0]) == [3, 4] and list(i.split(1)[1]) == []


def test_make_strata():
    codes = trm.make_strata([float(v) for v in range(1, 101)], breaks=4)
    assert np.bincount(codes).tolist() == [25, 25, 25, 25]
    assert trm.make_strata([float(v) for v in range(1, 31)], breaks=4).max() == 0  # floor(30 / 20) < 2
    labels = ["a"] * 50 + ["b"] * 45 + ["c"] * 5
    codes = trm.make_strata(labels)
    assert codes.max() == 1 and codes[0] == 0 and codes[60] == 1  # c pooled at random


def test_from_splits_round_trips_a_shared_rset():
    split = json.loads((ROOT / "reference" / "fixtures" / "splits" / "synth_reg_boot.json").read_text())
    rset = split["rset"]
    folds = trm.Folds.from_splits(list(zip(rset["analysis"], rset["assessment"])), rset["ids"], split["n"])
    assert folds.ids == rset["ids"] and len(folds) == 10
    train, test = folds.split(2)
    assert train.tolist() == rset["analysis"][2] and test.tolist() == rset["assessment"][2]
    with pytest.raises(trm.InvalidArgumentError):
        trm.Folds.from_splits([([0, 1], [2])], ["A", "B"], 3)


def test_empty_assessment_set_is_refused():
    data = trm.Table.from_dict({"x": np.arange(10, dtype=float), "y": np.arange(10, dtype=float) * 2.0})
    folds = trm.Folds.from_splits([([0, 1, 2, 3], [])], ["Slice1"], 10)
    wf = trm.Workflow(trm.Recipe("y"), trm.LinearReg())
    with pytest.raises(trm.InvalidArgumentError):
        wf.fit_resamples(data, folds, ["rmse"])
