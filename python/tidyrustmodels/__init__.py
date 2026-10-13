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
    "grid_regular",
    "class_metrics",
    "conf_mat",
    "from_pandas",
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
    Other dtypes raise :class:`UnsupportedError`.
    """
    import pandas as pd  # optional dependency, imported lazily

    cols: dict[str, Any] = {}
    for name in df.columns:
        s = df[name]
        if pd.api.types.is_bool_dtype(s):
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
    return Table.from_dict(cols)
