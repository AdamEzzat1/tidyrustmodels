"""tidyrustmodels: tidymodels-inspired machine learning in Rust, from Python.

All modelling lives in the compiled extension ``tidyrustmodels._native``; this module
re-exports it under stable names and adds pandas conversion, which needs no Rust.

Data conventions
----------------
* Numeric columns are float64; ``NaN`` means missing.
* Categorical columns are ``str`` labels with ``None`` for missing; levels are sorted
  bytewise, like R's ``factor()`` in the C locale. The first level is the event level
  for binary classification metrics (``event_level = "first"``).
* NumPy input is copied once into the Rust column-major layout.
"""
from __future__ import annotations

from typing import Any

import numpy as np

from . import _native
from ._native import (
    BackendError,
    DecisionTree,
    FittedWorkflow,
    Folds,
    InvalidArgumentError,
    InvalidValueError,
    LinearReg,
    LogisticReg,
    MultinomReg,
    RandForest, BoostTree,
    Recipe,
    SchemaError,
    SerializationError,
    Table,
    TidyRustModelsError,
    TuneResults,
    UnsupportedError,
    Workflow,
    __version__,
)

vfold_cv = _native.py_vfold_cv
initial_split = _native.py_initial_split
bootstraps = _native.py_bootstraps
mc_cv = _native.py_mc_cv
loo_cv = _native.py_loo_cv
group_vfold_cv = _native.py_group_vfold_cv
group_bootstraps = _native.py_group_bootstraps
group_mc_cv = _native.py_group_mc_cv
group_initial_split = _native.py_group_initial_split
rolling_origin = _native.py_rolling_origin
sliding_window = _native.py_sliding_window
sliding_index = _native.py_sliding_index
initial_time_split = _native.py_initial_time_split
initial_validation_split = _native.py_initial_validation_split
make_strata = _native.py_make_strata
tune_grid = _native.py_tune_grid
grid_regular = _native.py_grid_regular
class_metrics = _native.py_class_metrics
conf_mat = _native.py_conf_mat

__all__ = [
    "Table",
    "Recipe",
    "LinearReg",
    "LogisticReg",
    "MultinomReg",
    "DecisionTree",
    "RandForest", "BoostTree",
    "Workflow",
    "FittedWorkflow",
    "Folds",
    "vfold_cv",
    "initial_split",
    "bootstraps",
    "mc_cv",
    "loo_cv",
    "group_vfold_cv",
    "group_bootstraps",
    "group_mc_cv",
    "group_initial_split",
    "rolling_origin",
    "sliding_window",
    "sliding_index",
    "initial_time_split",
    "initial_validation_split",
    "make_strata",
    "tune_grid",
    "TuneResults",
    "grid_regular",
    "class_metrics",
    "conf_mat",
    "from_pandas",
    "to_pandas",
    "from_arrow",
    "to_arrow",
    "from_polars",
    "to_polars",
    "from_scipy_sparse",
    "TidyRustModelsError",
    "SchemaError",
    "InvalidValueError",
    "InvalidArgumentError",
    "UnsupportedError",
    "BackendError",
    "SerializationError",
    "__version__",
]


def from_pandas(df: Any) -> Table:
    """Convert a pandas DataFrame to a :class:`Table`, column by column.

    Numeric dtypes become float64 numeric columns (``NaN``/``NA`` -> missing). bool,
    object, string and category dtypes become categorical columns (``NA`` -> missing).
    A ``SparseDtype`` column with fill value 0 becomes a sparse vector (ADR-0015).
    Other dtypes raise :class:`UnsupportedError`.
    """
    import pandas as pd  # optional dependency, imported lazily

    cols: dict[str, Any] = {}
    sparse: list[str] = []
    for name in df.columns:
        s = df[name]
        if isinstance(s.dtype, pd.SparseDtype):
            if s.dtype.fill_value != 0:
                raise UnsupportedError(f"sparse column `{name}` must have fill value 0, got {s.dtype.fill_value}")
            cols[str(name)] = s.sparse.to_dense().to_numpy(dtype=np.float64, na_value=np.nan)
            sparse.append(str(name))
        elif pd.api.types.is_bool_dtype(s):
            cols[str(name)] = [None if pd.isna(v) else str(bool(v)) for v in s]
        elif pd.api.types.is_numeric_dtype(s):
            cols[str(name)] = s.to_numpy(dtype=np.float64, na_value=np.nan)
        elif (
            pd.api.types.is_object_dtype(s)
            or pd.api.types.is_string_dtype(s)
            or isinstance(s.dtype, pd.CategoricalDtype)
        ):
            cols[str(name)] = [None if pd.isna(v) else str(v) for v in s]
        else:
            raise UnsupportedError(f"column `{name}` has unsupported dtype {s.dtype}")
    t = Table.from_dict(cols)
    return t.with_sparse(sparse) if sparse else t


def _columns(table: Table) -> dict[str, Any]:
    return {name: table.column(name) for name in table.names}


def to_pandas(table: Table) -> Any:
    """A pandas DataFrame: float64 columns (``NaN`` for missing) and object columns of
    ``str | None``. The sparse flag is not carried over."""
    import pandas as pd

    return pd.DataFrame({k: (v if isinstance(v, np.ndarray) else pd.Series(v, dtype=object)) for k, v in _columns(table).items()})


def from_arrow(data: Any) -> Table:
    """Convert a pyarrow ``Table`` or ``RecordBatch`` column by column (ADR-0003):
    integer, floating and decimal columns become float64 numeric columns (nulls ->
    missing); string, large_string, dictionary-encoded string and boolean columns
    become categorical columns (``True``/``False`` labels for booleans). Other types
    raise :class:`UnsupportedError`."""
    import pyarrow as pa
    import pyarrow.types as pat

    cols: dict[str, Any] = {}
    for name, col in zip(data.schema.names, data.columns):
        if isinstance(col, pa.ChunkedArray):
            col = col.combine_chunks()
        t = col.type
        if pat.is_dictionary(t):
            col = col.dictionary_decode()
            t = col.type
        if pat.is_integer(t) or pat.is_floating(t) or pat.is_decimal(t):
            cols[name] = col.cast(pa.float64()).to_numpy(zero_copy_only=False)
        elif pat.is_string(t) or pat.is_large_string(t):
            cols[name] = col.to_pylist()
        elif pat.is_boolean(t):
            cols[name] = [None if v is None else str(bool(v)) for v in col.to_pylist()]
        else:
            raise UnsupportedError(f"column `{name}` has unsupported Arrow type {t}")
    return Table.from_dict(cols)


def to_arrow(table: Table) -> Any:
    """A pyarrow ``Table``: float64 columns with nulls for missing values and string
    columns for categoricals."""
    import pyarrow as pa

    arrays = {}
    for name, v in _columns(table).items():
        arrays[name] = pa.array(v, type=pa.float64(), from_pandas=True) if isinstance(v, np.ndarray) else pa.array(v, type=pa.string())
    return pa.table(arrays)


def from_polars(df: Any) -> Table:
    """Convert a polars DataFrame through Arrow (:func:`from_arrow`)."""
    return from_arrow(df.to_arrow())


def to_polars(table: Table) -> Any:
    """A polars DataFrame from :func:`to_arrow`."""
    import polars as pl

    return pl.from_arrow(to_arrow(table))


def from_scipy_sparse(matrix: Any, names: list[str]) -> Table:
    """Numeric columns from a SciPy sparse matrix, each flagged as a sparse vector
    (ADR-0015, like R's sparsevctrs): unstored entries are zero, stored values are
    kept (an explicit stored zero is a zero). The values are held densely; the flag
    makes a design built from them a sparse matrix for xgboost (zeros missing)."""
    import scipy.sparse as sps

    if not sps.issparse(matrix):
        raise InvalidArgumentError("expected a scipy.sparse matrix")
    csc = sps.csc_matrix(matrix)
    if len(names) != csc.shape[1]:
        raise InvalidArgumentError(f"{len(names)} names for {csc.shape[1]} columns")
    cols = {name: np.asarray(csc[:, j].toarray(), dtype=np.float64).ravel() for j, name in enumerate(names)}
    return Table.from_dict(cols).with_sparse(list(names))
