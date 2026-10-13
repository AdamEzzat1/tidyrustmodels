"""Arrow, Polars, pandas and SciPy-sparse interop (ADR-0015): every path must produce
the same Table, bit for bit, as `Table.from_dict`, and sparse input must reach the
booster as a sparse design (zeros missing) exactly as `with_sparse` does."""
from __future__ import annotations

import numpy as np
import pytest

import tidyrustmodels as trm


def _reference():
    x = np.array([1.5, np.nan, 0.0, 2.25, 0.0], dtype=float)
    n = np.array([3.0, 4.0, 5.0, 6.0, 7.0])
    g = ["a", "b", None, "a", "b"]
    return trm.Table.from_dict({"x": x, "n": n, "g": g}), x, n, g


def _same(a: trm.Table, b: trm.Table) -> None:
    assert a.names == b.names and a.nrows == b.nrows
    for c in a.names:
        va, vb = a.column(c), b.column(c)
        if isinstance(va, np.ndarray):
            assert np.array_equal(va, vb, equal_nan=True), c
        else:
            assert va == vb, c
    assert a.sparse_columns == b.sparse_columns


def test_arrow_round_trip():
    pa = pytest.importorskip("pyarrow")
    ref, x, n, g = _reference()
    tbl = pa.table({"x": pa.array(x), "n": pa.array(n.astype(np.int64)), "g": pa.array(g)})
    t = trm.from_arrow(tbl)
    _same(t, ref)
    # dictionary-encoded strings, booleans and a RecordBatch
    tbl2 = pa.table({"x": pa.array(x), "n": pa.array(n), "g": pa.array(g).dictionary_encode()})
    _same(trm.from_arrow(tbl2), ref)
    _same(trm.from_arrow(tbl.to_batches()[0]), ref)
    back = trm.to_arrow(t)
    assert back.column("x").to_pylist()[1] is None and back.column("g").to_pylist() == g
    _same(trm.from_arrow(back), ref)
    b = trm.from_arrow(pa.table({"flag": pa.array([True, False, None])}))
    assert b.column("flag") == ["True", "False", None]


def test_polars_round_trip():
    pl = pytest.importorskip("polars")
    ref, x, n, g = _reference()
    df = pl.DataFrame({"x": x, "n": n.astype(np.int64), "g": g})
    t = trm.from_polars(df)
    _same(t, ref)
    _same(trm.from_polars(df.with_columns(pl.col("g").cast(pl.Categorical))), ref)
    out = trm.to_polars(t)
    assert out.schema["g"] == pl.String and out["x"].null_count() == 1
    _same(trm.from_polars(out), ref)


def test_pandas_round_trip_and_sparse_dtype():
    pd = pytest.importorskip("pandas")
    ref, x, n, g = _reference()
    df = pd.DataFrame({"x": x, "n": n, "g": g})
    t = trm.from_pandas(df)
    _same(t, ref)
    out = trm.to_pandas(t)
    assert out["g"].tolist()[2] is None and np.isnan(out["x"][1])
    _same(trm.from_pandas(out), ref)
    # a pandas SparseDtype column is a sparse vector
    s = trm.from_pandas(df.assign(x=pd.arrays.SparseArray(np.nan_to_num(x), fill_value=0.0)))
    assert s.sparse_columns == ["x"]


def test_scipy_sparse_input_matches_flagged_dense():
    sp = pytest.importorskip("scipy.sparse")
    rng = np.random.default_rng(5)
    dense = rng.normal(size=(60, 3))
    dense[rng.uniform(size=dense.shape) < 0.7] = 0.0
    y = 1.0 + dense @ np.array([1.0, -2.0, 0.5]) + rng.normal(scale=0.1, size=60)
    csr = sp.csr_matrix(dense)
    t = trm.from_scipy_sparse(csr, ["a", "b", "c"])
    assert t.sparse_columns == ["a", "b", "c"] and np.array_equal(t.column("b"), dense[:, 1])
    flagged = trm.Table.from_dict({"a": dense[:, 0], "b": dense[:, 1], "c": dense[:, 2]}).with_sparse(["a", "b", "c"])
    _same(t, flagged)
    # the booster sees zeros as missing on a sparse design: the fit differs from the
    # dense fit and equals the flagged-dense fit bit for bit
    def with_y(tbl):
        d = {c: tbl.column(c) for c in tbl.names}
        d["y"] = y
        out = trm.Table.from_dict(d)
        return out.with_sparse(tbl.sparse_columns) if tbl.sparse_columns else out
    wf = trm.Workflow(trm.Recipe("y"), trm.BoostTree(trees=20))
    p_sparse = wf.fit(with_y(t)).predict(with_y(t))
    p_flagged = wf.fit(with_y(flagged)).predict(with_y(flagged))
    p_dense = wf.fit(with_y(flagged.with_sparse([]))).predict(with_y(flagged.with_sparse([])))
    assert np.array_equal(p_sparse, p_flagged)
    assert not np.array_equal(p_sparse, p_dense)
    # the new data decides at prediction: the sparse fit reads dense new data with
    # zeros as values (parsnip converts the baked new data again)
    fs = wf.fit(with_y(t))
    assert not np.array_equal(fs.predict(with_y(flagged.with_sparse([]))), p_sparse)
    # step_dummy(sparse = "yes") makes the design sparse on its own
    g = ["u" if v > 0 else "v" for v in dense[:, 0]]
    dd = {"a": dense[:, 0], "b": dense[:, 1], "g": g, "y": y}
    rec_yes = trm.Recipe("y").step_dummy("all_nominal_predictors", sparse="yes")
    rec_no = trm.Recipe("y").step_dummy("all_nominal_predictors", sparse="no")
    bt = trm.BoostTree(trees=20)
    assert not np.array_equal(
        trm.Workflow(rec_yes, bt).fit(trm.Table.from_dict(dd)).predict(trm.Table.from_dict(dd)),
        trm.Workflow(rec_no, bt).fit(trm.Table.from_dict(dd)).predict(trm.Table.from_dict(dd)),
    )
    with pytest.raises(trm.InvalidArgumentError):
        trm.Recipe("y").step_dummy("all_nominal_predictors", sparse="maybe")
    # lm ignores the flag (R densifies for lm)
    lm = trm.Workflow(trm.Recipe("y"), trm.LinearReg())
    assert np.array_equal(lm.fit(with_y(t)).predict(with_y(t)), lm.fit(with_y(flagged.with_sparse([]))).predict(with_y(flagged.with_sparse([]))))
    with pytest.raises(trm.InvalidArgumentError):
        trm.from_scipy_sparse(csr, ["a", "b"])
    with pytest.raises(trm.InvalidArgumentError):
        trm.Table.from_dict({"g": ["a", "b"]}).with_sparse(["g"])
