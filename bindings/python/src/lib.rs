//! `tidyrustmodels._native`: PyO3 bindings over the Rust facade crate.
//!
//! Design (ARCHITECTURE.md, "Python layer"):
//! - Every object wraps the corresponding Rust type; no modelling logic lives here.
//! - NumPy input is copied once into the column-major `DesignMatrix`/`Table`.
//! - `Error` maps to an exception hierarchy rooted at `TidyRustModelsError`.
//! - Outputs are NumPy arrays (numeric) or plain Python lists/dicts (labels, metrics).

use numpy::{IntoPyArray, PyArray1, PyArrayMethods, PyReadonlyArray1, PyReadonlyArray2, PyUntypedArrayMethods};
use pyo3::create_exception;
use pyo3::exceptions::PyException;
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList};
use tidyrustmodels::prelude::*;

create_exception!(_native, TidyRustModelsError, PyException, "Base class for tidyrustmodels errors.");
create_exception!(_native, SchemaError, TidyRustModelsError, "Data does not match the fitted schema.");
create_exception!(_native, InvalidValueError, TidyRustModelsError, "A value violates a precondition.");
create_exception!(_native, InvalidArgumentError, TidyRustModelsError, "An argument is out of range.");
create_exception!(_native, UnsupportedError, TidyRustModelsError, "The requested capability is unsupported.");
create_exception!(_native, BackendError, TidyRustModelsError, "A backend crate reported a failure.");
create_exception!(_native, SerializationError, TidyRustModelsError, "Serialization failed.");

fn to_py(e: Error) -> PyErr {
    let msg = e.to_string();
    match e {
        Error::SchemaMismatch(_) => SchemaError::new_err(msg),
        Error::InvalidValue(_) => InvalidValueError::new_err(msg),
        Error::InvalidArgument(_) => InvalidArgumentError::new_err(msg),
        Error::Unsupported(_) => UnsupportedError::new_err(msg),
        Error::Backend { .. } => BackendError::new_err(msg),
        Error::Serialization(_) => SerializationError::new_err(msg),
        _ => TidyRustModelsError::new_err(msg),
    }
}

trait IntoPyResult<T> {
    fn py(self) -> PyResult<T>;
}
impl<T> IntoPyResult<T> for tidyrustmodels::Result<T> {
    fn py(self) -> PyResult<T> {
        self.map_err(to_py)
    }
}

fn parse_engine(engine: &str) -> PyResult<Engine> {
    match engine {
        "linfa" => Ok(Engine::Linfa),
        "smartcore" => Ok(Engine::Smartcore),
        "native" => Ok(Engine::Native),
        other => Err(InvalidArgumentError::new_err(format!("unknown engine `{other}` (use 'linfa' or 'smartcore')"))),
    }
}

fn parse_selector(sel: &Bound<'_, PyAny>) -> PyResult<Selector> {
    if let Ok(s) = sel.extract::<String>() {
        return match s.as_str() {
            "all_predictors" => Ok(Selector::AllPredictors),
            "all_numeric_predictors" => Ok(Selector::AllNumericPredictors),
            "all_nominal_predictors" => Ok(Selector::AllNominalPredictors),
            other => Ok(Selector::columns(&[other])),
        };
    }
    let names: Vec<String> = sel.extract()?;
    Ok(Selector::columns(&names))
}

// ---------------------------------------------------------------- Table

/// A columnar table. Build with `from_dict`, `from_numpy`, or the Python-side
/// `Table.from_pandas` helper.
#[pyclass(name = "Table", module = "tidyrustmodels._native", from_py_object)]
#[derive(Clone)]
struct PyTable {
    inner: Table,
}

#[pymethods]
impl PyTable {
    /// Build from `{name: column}` where a column is a float64 NumPy array / list of
    /// floats (NaN = missing) or a list of `str | None` (categorical).
    #[staticmethod]
    fn from_dict(columns: &Bound<'_, PyDict>) -> PyResult<Self> {
        let mut cols = Vec::with_capacity(columns.len());
        for (k, v) in columns.iter() {
            let name: String = k.extract()?;
            cols.push((name, column_from_py(&v)?));
        }
        Ok(PyTable { inner: Table::new(cols).py()? })
    }

    /// Build numeric columns from a 2-D float64 array (any layout) and column names.
    #[staticmethod]
    fn from_numpy(x: PyReadonlyArray2<'_, f64>, names: Vec<String>) -> PyResult<Self> {
        let shape = x.shape();
        let (n, p) = (shape[0], shape[1]);
        if names.len() != p {
            return Err(InvalidArgumentError::new_err(format!("{} names for {p} columns", names.len())));
        }
        let view = x.as_array();
        let mut cols = Vec::with_capacity(p);
        for (j, name) in names.into_iter().enumerate() {
            let values: Vec<Option<f64>> = (0..n).map(|i| view[[i, j]]).map(|v| if v.is_nan() { None } else { Some(v) }).collect();
            cols.push((name, Column::numeric_with_missing(values).py()?));
        }
        Ok(PyTable { inner: Table::new(cols).py()? })
    }

    #[getter]
    fn nrows(&self) -> usize {
        self.inner.nrows()
    }

    #[getter]
    fn ncols(&self) -> usize {
        self.inner.ncols()
    }

    #[getter]
    fn names(&self) -> Vec<String> {
        self.inner.names().to_vec()
    }

    /// Rows by 0-based index.
    fn select_rows(&self, rows: Vec<usize>) -> PyResult<Self> {
        Ok(PyTable { inner: self.inner.select_rows(&rows).py()? })
    }

    /// Flag numeric columns as sparse vectors (`sparsevctrs::as_sparse_double`, ADR-0015):
    /// values stay dense here, but a design built from any sparse column reaches xgboost
    /// as a sparse matrix, i.e. with every exact zero treated as missing. Other engines
    /// are unaffected (R densifies for lm/glm/rpart/nnet; ranger reads zeros as values).
    fn with_sparse(&self, names: Vec<String>) -> PyResult<Self> {
        let mut cols = Vec::with_capacity(self.inner.ncols());
        for (name, col) in self.inner.iter() {
            let flag = names.iter().any(|n| n == name);
            if flag && col.as_numeric().is_none() {
                return Err(InvalidArgumentError::new_err(format!("column `{name}` is categorical and cannot be sparse")));
            }
            cols.push((name.to_string(), col.clone().with_sparse(flag)));
        }
        for n in &names {
            if self.inner.column(n).is_none() {
                return Err(SchemaError::new_err(format!("no column `{n}`")));
            }
        }
        Ok(PyTable { inner: Table::new(cols).py()? })
    }

    /// Names of the columns flagged as sparse vectors.
    #[getter]
    fn sparse_columns(&self) -> Vec<String> {
        self.inner.iter().filter(|(_, c)| c.is_sparse()).map(|(n, _)| n.to_string()).collect()
    }

    /// Column as a float64 array (NaN for missing) or list of `str | None`.
    fn column<'py>(&self, py: Python<'py>, name: &str) -> PyResult<Bound<'py, PyAny>> {
        let col = self.inner.column(name).ok_or_else(|| SchemaError::new_err(format!("no column `{name}`")))?;
        Ok(match col {
            Column::Numeric { values, .. } => {
                let v: Vec<f64> = values.iter().enumerate().map(|(i, x)| if col.is_missing(i) { f64::NAN } else { *x }).collect();
                v.into_pyarray(py).into_any()
            }
            Column::Categorical { codes, levels, .. } => {
                let v: Vec<Option<&str>> = codes.iter().enumerate().map(|(i, c)| if col.is_missing(i) { None } else { Some(levels[*c as usize].as_str()) }).collect();
                PyList::new(py, v)?.into_any()
            }
        })
    }

    fn __repr__(&self) -> String {
        format!("Table({} rows x {} cols: {:?})", self.inner.nrows(), self.inner.ncols(), self.inner.names())
    }
}

fn column_from_py(v: &Bound<'_, PyAny>) -> PyResult<Column> {
    if let Ok(arr) = v.extract::<PyReadonlyArray1<'_, f64>>() {
        let values: Vec<Option<f64>> = arr.as_array().iter().map(|x| if x.is_nan() { None } else { Some(*x) }).collect();
        return Column::numeric_with_missing(values).py();
    }
    if let Ok(values) = v.extract::<Vec<Option<f64>>>() {
        let values = values.into_iter().map(|x| x.filter(|f| !f.is_nan())).collect();
        return Column::numeric_with_missing(values).py();
    }
    let labels: Vec<Option<String>> = v.extract().map_err(|_| {
        InvalidArgumentError::new_err("column must be a float64 array, a list of floats, or a list of str | None")
    })?;
    let mut levels: Vec<String> = labels.iter().flatten().cloned().collect();
    levels.sort();
    levels.dedup();
    let refs: Vec<Option<&str>> = labels.iter().map(|l| l.as_deref()).collect();
    Column::categorical_with_levels_opt(&refs, levels).py()
}

// ---------------------------------------------------------------- Recipe

/// A recipe: outcome name plus ordered preprocessing steps.
#[pyclass(name = "Recipe", module = "tidyrustmodels._native", from_py_object)]
#[derive(Clone)]
struct PyRecipe {
    inner: Recipe,
}

#[pymethods]
impl PyRecipe {
    #[new]
    fn new(outcome: &str) -> Self {
        PyRecipe { inner: Recipe::new(outcome) }
    }

    fn step_normalize(&self, selector: &Bound<'_, PyAny>) -> PyResult<Self> {
        Ok(PyRecipe { inner: self.inner.clone().step(StepNormalize::new(parse_selector(selector)?)) })
    }

    fn step_impute_mean(&self, selector: &Bound<'_, PyAny>) -> PyResult<Self> {
        Ok(PyRecipe { inner: self.inner.clone().step(StepImputeMean::new(parse_selector(selector)?)) })
    }

    fn step_impute_median(&self, selector: &Bound<'_, PyAny>) -> PyResult<Self> {
        Ok(PyRecipe { inner: self.inner.clone().step(StepImputeMedian::new(parse_selector(selector)?)) })
    }

    /// `sparse` is recipes' `sparse = "auto" | "yes" | "no"` (ADR-0015): "yes" flags the
    /// indicator columns as sparse vectors (the design is then sparse for xgboost, i.e.
    /// zeros are missing); "auto" leaves it to the workflow's sparse-data decision.
    #[pyo3(signature = (selector, one_hot = false, sparse = "auto"))]
    fn step_dummy(&self, selector: &Bound<'_, PyAny>, one_hot: bool, sparse: &str) -> PyResult<Self> {
        let sparse = match sparse {
            "auto" => SparseOutput::Auto,
            "yes" => SparseOutput::Yes,
            "no" => SparseOutput::No,
            other => {
                return Err(InvalidArgumentError::new_err(format!(
                    "sparse must be \"auto\", \"yes\" or \"no\" (got \"{other}\")"
                )))
            }
        };
        Ok(PyRecipe {
            inner: self
                .inner
                .clone()
                .step(StepDummy::new(parse_selector(selector)?).one_hot(one_hot).sparse(sparse)),
        })
    }

    fn step_select(&self, selector: &Bound<'_, PyAny>) -> PyResult<Self> {
        Ok(PyRecipe { inner: self.inner.clone().step(StepSelect::new(parse_selector(selector)?)) })
    }

    fn step_zv(&self, selector: &Bound<'_, PyAny>) -> PyResult<Self> {
        Ok(PyRecipe { inner: self.inner.clone().step(StepZv::new(parse_selector(selector)?)) })
    }

    #[pyo3(signature = (selector, freq_cut = 19.0, unique_cut = 10.0))]
    fn step_nzv(&self, selector: &Bound<'_, PyAny>, freq_cut: f64, unique_cut: f64) -> PyResult<Self> {
        Ok(PyRecipe {
            inner: self.inner.clone().step(StepNzv::new(parse_selector(selector)?).freq_cut(freq_cut).unique_cut(unique_cut)),
        })
    }

    #[pyo3(signature = (selector, threshold = 0.9))]
    fn step_corr(&self, selector: &Bound<'_, PyAny>, threshold: f64) -> PyResult<Self> {
        Ok(PyRecipe { inner: self.inner.clone().step(StepCorr::new(parse_selector(selector)?).threshold(threshold)) })
    }

    #[pyo3(signature = (selector, threshold = 0.05, other = "other"))]
    fn step_other(&self, selector: &Bound<'_, PyAny>, threshold: f64, other: &str) -> PyResult<Self> {
        Ok(PyRecipe {
            inner: self.inner.clone().step(StepOther::new(parse_selector(selector)?).threshold(threshold).other(other)),
        })
    }

    #[pyo3(signature = (selector, new_level = "new"))]
    fn step_novel(&self, selector: &Bound<'_, PyAny>, new_level: &str) -> PyResult<Self> {
        Ok(PyRecipe { inner: self.inner.clone().step(StepNovel::new(parse_selector(selector)?).new_level(new_level)) })
    }

    #[pyo3(signature = (selector, new_level = "unknown"))]
    fn step_unknown(&self, selector: &Bound<'_, PyAny>, new_level: &str) -> PyResult<Self> {
        Ok(PyRecipe { inner: self.inner.clone().step(StepUnknown::new(parse_selector(selector)?).new_level(new_level)) })
    }

    #[getter]
    fn outcome(&self) -> String {
        self.inner.outcome().to_string()
    }
}

// ---------------------------------------------------------------- Models

/// `linear_reg`.
#[pyclass(name = "LinearReg", module = "tidyrustmodels._native", from_py_object)]
#[derive(Clone)]
struct PyLinearReg {
    inner: LinearRegSpec,
}

#[pymethods]
impl PyLinearReg {
    #[new]
    #[pyo3(signature = (engine = "native", intercept = true))]
    fn new(engine: &str, intercept: bool) -> PyResult<Self> {
        Ok(PyLinearReg { inner: LinearRegSpec::new().engine(parse_engine(engine)?).intercept(intercept) })
    }
}

/// `logistic_reg`.
#[pyclass(name = "LogisticReg", module = "tidyrustmodels._native", from_py_object)]
#[derive(Clone)]
struct PyLogisticReg {
    inner: LogisticRegSpec,
}

#[pymethods]
impl PyLogisticReg {
    #[new]
    #[pyo3(signature = (engine = "native", intercept = true, penalty = 0.0, max_iterations = 1000, gradient_tolerance = 1e-8))]
    fn new(engine: &str, intercept: bool, penalty: f64, max_iterations: u64, gradient_tolerance: f64) -> PyResult<Self> {
        Ok(PyLogisticReg {
            inner: LogisticRegSpec::new()
                .engine(parse_engine(engine)?)
                .intercept(intercept)
                .penalty(penalty)
                .max_iterations(max_iterations)
                .gradient_tolerance(gradient_tolerance),
        })
    }
}

/// `parsnip::multinom_reg()`: multinomial regression (ADR-0010). The first outcome level
/// is the baseline; `coefficients()` on the fitted workflow returns one dict per other
/// level. `penalty` is the unscaled L2 term of the Rust API (not nnet's `decay`).
#[pyclass(name = "MultinomReg", module = "tidyrustmodels._native", from_py_object)]
#[derive(Clone)]
struct PyMultinomReg {
    inner: MultinomRegSpec,
}

#[pymethods]
impl PyMultinomReg {
    #[new]
    #[pyo3(signature = (engine = "native", intercept = true, penalty = 0.0, epsilon = 1e-10, newton_max_iterations = 100, max_iterations = 1000, gradient_tolerance = 1e-8))]
    fn new(
        engine: &str,
        intercept: bool,
        penalty: f64,
        epsilon: f64,
        newton_max_iterations: usize,
        max_iterations: u64,
        gradient_tolerance: f64,
    ) -> PyResult<Self> {
        Ok(PyMultinomReg {
            inner: MultinomRegSpec::new()
                .engine(parse_engine(engine)?)
                .intercept(intercept)
                .penalty(penalty)
                .epsilon(epsilon)
                .newton_max_iterations(newton_max_iterations)
                .max_iterations(max_iterations)
                .gradient_tolerance(gradient_tolerance),
        })
    }
}

/// `parsnip::decision_tree()` with engine `rpart` (native port, ADR-0011):
/// `cost_complexity` = `cp`, `tree_depth` = `maxdepth`, `min_n` = `minsplit`.
#[pyclass(name = "DecisionTree", module = "tidyrustmodels._native", from_py_object)]
#[derive(Clone)]
struct PyDecisionTree {
    inner: DecisionTreeSpec,
}

#[pymethods]
impl PyDecisionTree {
    #[new]
    #[pyo3(signature = (cost_complexity = 0.01, tree_depth = 30, min_n = 20))]
    fn new(cost_complexity: f64, tree_depth: usize, min_n: usize) -> PyResult<Self> {
        Ok(PyDecisionTree {
            inner: DecisionTreeSpec::new().cost_complexity(cost_complexity).tree_depth(tree_depth).min_n(min_n),
        })
    }
}

/// `parsnip::rand_forest()` with engine `ranger` semantics (native, ADR-0011):
/// `mtry` (default `floor(sqrt(p))`), `trees` (500), `min_n` (5 regression / 10
/// classification), `seed` for the per-tree streams. Stochastic: compare forests
/// statistically, not tree by tree.
#[pyclass(name = "RandForest", module = "tidyrustmodels._native", from_py_object)]
#[derive(Clone)]
struct PyRandForest {
    inner: RandForestSpec,
}

#[pymethods]
impl PyRandForest {
    #[new]
    #[pyo3(signature = (mtry = None, trees = 500, min_n = None, seed = 0))]
    fn new(mtry: Option<usize>, trees: usize, min_n: Option<usize>, seed: u64) -> PyResult<Self> {
        let mut s = RandForestSpec::new().trees(trees).seed(Seed(seed));
        if let Some(m) = mtry {
            s = s.mtry(m);
        }
        if let Some(m) = min_n {
            s = s.min_n(m);
        }
        Ok(PyRandForest { inner: s })
    }
}

/// `parsnip::boost_tree()` with engine `xgboost` semantics (native port of xgboost
/// 3.2.1's hist booster, ADR-0011): `trees` (15), `tree_depth` (6), `learn_rate`
/// (0.3), `min_n` (1), `loss_reduction` (0), `lambda` (1), `alpha` (0);
/// `zeros_as_missing` None = tidymodels' behaviour (a Workflow applies workflows'
/// sparse-data toggle to the recipe's dummy step, and zeros are missing exactly when
/// the design is sparse, at fit and at predict); True = always missing; False = never
/// toggle (a sparse design still reads zeros as missing).
#[pyclass(name = "BoostTree", module = "tidyrustmodels._native", from_py_object)]
#[derive(Clone)]
struct PyBoostTree {
    inner: BoostTreeSpec,
}

#[pymethods]
impl PyBoostTree {
    #[new]
    #[pyo3(signature = (trees = 15, tree_depth = 6, learn_rate = 0.3, min_n = 1.0, loss_reduction = 0.0, lambda = 1.0, alpha = 0.0, zeros_as_missing = None))]
    #[allow(clippy::too_many_arguments)]
    fn new(
        trees: usize,
        tree_depth: usize,
        learn_rate: f64,
        min_n: f64,
        loss_reduction: f64,
        lambda: f64,
        alpha: f64,
        zeros_as_missing: Option<bool>,
    ) -> PyResult<Self> {
        let mut s = BoostTreeSpec::new()
            .trees(trees)
            .tree_depth(tree_depth)
            .learn_rate(learn_rate)
            .min_n(min_n)
            .loss_reduction(loss_reduction)
            .lambda(lambda)
            .alpha(alpha);
        if let Some(z) = zeros_as_missing {
            s = s.zeros_as_missing(z);
        }
        Ok(PyBoostTree { inner: s })
    }
}

fn model_from_py(model: &Bound<'_, PyAny>) -> PyResult<Box<dyn ModelSpec>> {
    if let Ok(m) = model.extract::<PyBoostTree>() {
        return Ok(Box::new(m.inner));
    }
    if let Ok(m) = model.extract::<PyRandForest>() {
        return Ok(Box::new(m.inner));
    }
    if let Ok(m) = model.extract::<PyDecisionTree>() {
        return Ok(Box::new(m.inner));
    }
    if let Ok(m) = model.extract::<PyMultinomReg>() {
        return Ok(Box::new(m.inner));
    }
    if let Ok(m) = model.extract::<PyLinearReg>() {
        return Ok(Box::new(m.inner));
    }
    if let Ok(m) = model.extract::<PyLogisticReg>() {
        return Ok(Box::new(m.inner));
    }
    Err(InvalidArgumentError::new_err("model must be a LinearReg, LogisticReg, MultinomReg, DecisionTree, RandForest or BoostTree"))
}

// ---------------------------------------------------------------- Resampling

/// Cross-validation folds.
#[pyclass(name = "Folds", module = "tidyrustmodels._native", from_py_object)]
#[derive(Clone)]
struct PyFolds {
    inner: Folds,
}

#[pymethods]
impl PyFolds {
    /// Build from a fold index per row.
    #[staticmethod]
    fn from_assignments(fold_of_row: Vec<u32>) -> PyResult<Self> {
        Ok(PyFolds { inner: Folds::from_assignments(&fold_of_row).py()? })
    }

    /// Build from explicit `(analysis_rows, assessment_rows)` pairs with their ids over
    /// `n` rows (rsample's `manual_rset()`): analysis rows keep their order and may
    /// repeat, assessment rows are sorted and may be empty.
    #[staticmethod]
    fn from_splits(splits: Vec<(Vec<usize>, Vec<usize>)>, ids: Vec<String>, n: usize) -> PyResult<Self> {
        let splits = splits
            .into_iter()
            .map(|(a, s)| Split::new(a, s, n))
            .collect::<Result<Vec<_>>>()
            .py()?;
        Ok(PyFolds { inner: Folds::from_splits(splits, ids, n).py()? })
    }

    #[getter]
    fn ids(&self) -> Vec<String> {
        self.inner.ids.clone()
    }

    /// Number of rows the resamples index.
    #[getter]
    fn n(&self) -> usize {
        self.inner.n
    }

    /// Whether every resample partitions the rows (v-fold style).
    fn is_partition(&self) -> bool {
        self.inner.is_partition()
    }

    /// Fold index per row.
    fn assignments<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<u32>> {
        self.inner.assignments().into_pyarray(py)
    }

    /// `(train_rows, test_rows)` for fold `k`.
    fn split<'py>(&self, py: Python<'py>, k: usize) -> PyResult<(Bound<'py, PyArray1<u64>>, Bound<'py, PyArray1<u64>>)> {
        let s = self.inner.splits.get(k).ok_or_else(|| InvalidArgumentError::new_err(format!("no fold {k}")))?;
        let tr: Vec<u64> = s.train.iter().map(|&i| i as u64).collect();
        let te: Vec<u64> = s.test.iter().map(|&i| i as u64).collect();
        Ok((tr.into_pyarray(py), te.into_pyarray(py)))
    }

    fn __len__(&self) -> usize {
        self.inner.len()
    }
}

fn strata_codes(strata: Option<Vec<Option<String>>>) -> PyResult<Option<Vec<u32>>> {
    match strata {
        None => Ok(None),
        Some(labels) => {
            let refs: Vec<Option<&str>> = labels.iter().map(|l| l.as_deref()).collect();
            let mut levels: Vec<String> = labels.iter().flatten().cloned().collect();
            levels.sort();
            levels.dedup();
            let col = Column::categorical_with_levels_opt(&refs, levels).py()?;
            if col.n_missing() > 0 {
                return Err(InvalidValueError::new_err("strata must not contain missing values"));
            }
            Ok(Some(col.as_categorical().map(|(c, _)| c.to_vec()).unwrap_or_default()))
        }
    }
}

fn ids_pair<'py>(py: Python<'py>, s: &Split) -> (Bound<'py, PyArray1<u64>>, Bound<'py, PyArray1<u64>>) {
    let tr: Vec<u64> = s.train.iter().map(|&i| i as u64).collect();
    let te: Vec<u64> = s.test.iter().map(|&i| i as u64).collect();
    (tr.into_pyarray(py), te.into_pyarray(py))
}

/// Group labels (one per row, none missing) as codes.
fn group_codes(groups: Vec<Option<String>>) -> PyResult<Vec<u32>> {
    strata_codes(Some(groups))?.ok_or_else(|| InvalidArgumentError::new_err("groups must not be empty"))
}

/// rsample-style v-fold cross-validation (seeded, optionally stratified by labels,
/// `repeats` independent assignments with ids `Repeat1.Fold1`, ...).
#[pyfunction]
#[pyo3(signature = (n, v, seed, strata = None, repeats = 1))]
fn py_vfold_cv(n: usize, v: usize, seed: u64, strata: Option<Vec<Option<String>>>, repeats: usize) -> PyResult<PyFolds> {
    let codes = strata_codes(strata)?;
    let inner = if repeats > 1 {
        vfold_cv_repeated(n, v, repeats, codes.as_deref(), Seed(seed)).py()?
    } else {
        vfold_cv(n, v, codes.as_deref(), Seed(seed)).py()?
    };
    Ok(PyFolds { inner })
}

/// rsample-style bootstrap resamples: `times` draws of `n` rows with replacement
/// (sorted within each stratum when stratified); the assessment set is out-of-bag.
#[pyfunction]
#[pyo3(signature = (n, times, seed, strata = None))]
fn py_bootstraps(n: usize, times: usize, seed: u64, strata: Option<Vec<Option<String>>>) -> PyResult<PyFolds> {
    let codes = strata_codes(strata)?;
    Ok(PyFolds { inner: bootstraps(n, times, codes.as_deref(), Seed(seed)).py()? })
}

/// rsample-style Monte-Carlo cross-validation: `times` random `prop` splits.
#[pyfunction]
#[pyo3(signature = (n, prop, times, seed, strata = None))]
fn py_mc_cv(n: usize, prop: f64, times: usize, seed: u64, strata: Option<Vec<Option<String>>>) -> PyResult<PyFolds> {
    let codes = strata_codes(strata)?;
    Ok(PyFolds { inner: mc_cv(n, prop, times, codes.as_deref(), Seed(seed)).py()? })
}

/// rsample-style leave-one-out cross-validation (ids `Resample1`, ...).
#[pyfunction]
fn py_loo_cv(n: usize, seed: u64) -> PyResult<PyFolds> {
    Ok(PyFolds { inner: loo_cv(n, Seed(seed)).py()? })
}

/// rsample-style grouped v-fold cross-validation: every row of a group stays on one
/// side; `v=None` leaves one group out per resample; `balance` is `"groups"` or
/// `"observations"`.
#[pyfunction]
#[pyo3(signature = (groups, seed, v = None, repeats = 1, balance = "groups"))]
fn py_group_vfold_cv(
    groups: Vec<Option<String>>,
    seed: u64,
    v: Option<usize>,
    repeats: usize,
    balance: &str,
) -> PyResult<PyFolds> {
    let balance = match balance {
        "groups" => GroupBalance::Groups,
        "observations" => GroupBalance::Observations,
        other => return Err(InvalidArgumentError::new_err(format!("balance must be 'groups' or 'observations', got {other:?}"))),
    };
    let codes = group_codes(groups)?;
    Ok(PyFolds { inner: group_vfold_cv(&codes, v, repeats, balance, Seed(seed)).py()? })
}

/// rsample-style grouped bootstraps: groups drawn with replacement until their rows
/// cover the data; out-of-group rows assess.
#[pyfunction]
fn py_group_bootstraps(groups: Vec<Option<String>>, times: usize, seed: u64) -> PyResult<PyFolds> {
    let codes = group_codes(groups)?;
    Ok(PyFolds { inner: group_bootstraps(&codes, times, Seed(seed)).py()? })
}

/// rsample-style grouped Monte-Carlo cross-validation.
#[pyfunction]
fn py_group_mc_cv(groups: Vec<Option<String>>, prop: f64, times: usize, seed: u64) -> PyResult<PyFolds> {
    let codes = group_codes(groups)?;
    Ok(PyFolds { inner: group_mc_cv(&codes, prop, times, Seed(seed)).py()? })
}

/// rsample-style grouped initial split; returns `(train_rows, test_rows)`.
#[pyfunction]
fn py_group_initial_split<'py>(
    py: Python<'py>,
    groups: Vec<Option<String>>,
    prop: f64,
    seed: u64,
) -> PyResult<(Bound<'py, PyArray1<u64>>, Bound<'py, PyArray1<u64>>)> {
    let codes = group_codes(groups)?;
    Ok(ids_pair(py, &group_initial_split(&codes, prop, Seed(seed)).py()?))
}

/// rsample's `rolling_origin()`: analysis windows of `initial` rows (growing when
/// `cumulative`), assessed on the next `assess` rows, advancing `skip + 1` rows.
#[pyfunction]
#[pyo3(signature = (n, initial, assess, cumulative = true, skip = 0, lag = 0))]
fn py_rolling_origin(n: usize, initial: usize, assess: usize, cumulative: bool, skip: usize, lag: usize) -> PyResult<PyFolds> {
    Ok(PyFolds { inner: rolling_origin(n, initial, assess, cumulative, skip, lag).py()? })
}

/// rsample's `sliding_window()`: per row, `lookback` preceding rows (`None` = all,
/// rsample's `Inf`) assessed on rows `assess_start..=assess_stop` ahead.
#[pyfunction]
#[pyo3(signature = (n, lookback = Some(0), assess_start = 1, assess_stop = 1, complete = true, step = 1, skip = 0))]
fn py_sliding_window(
    n: usize,
    lookback: Option<usize>,
    assess_start: usize,
    assess_stop: usize,
    complete: bool,
    step: usize,
    skip: usize,
) -> PyResult<PyFolds> {
    Ok(PyFolds { inner: sliding_window(n, lookback, assess_start, assess_stop, complete, step, skip).py()? })
}

/// rsample's `sliding_index()` over a sorted numeric index: windows in index units.
#[pyfunction]
#[pyo3(signature = (index, lookback = Some(0.0), assess_start = 1.0, assess_stop = 1.0, complete = true, step = 1, skip = 0))]
fn py_sliding_index(
    index: Vec<f64>,
    lookback: Option<f64>,
    assess_start: f64,
    assess_stop: f64,
    complete: bool,
    step: usize,
    skip: usize,
) -> PyResult<PyFolds> {
    Ok(PyFolds { inner: sliding_index(&index, lookback, assess_start, assess_stop, complete, step, skip).py()? })
}

/// rsample's `initial_time_split()`: the first `floor(n * prop)` rows train.
#[pyfunction]
fn py_initial_time_split<'py>(py: Python<'py>, n: usize, prop: f64) -> PyResult<(Bound<'py, PyArray1<u64>>, Bound<'py, PyArray1<u64>>)> {
    Ok(ids_pair(py, &initial_time_split(n, prop).py()?))
}

/// rsample's `initial_validation_split()`: `(train_rows, validation_rows, test_rows)`
/// with proportions `prop_train` and `prop_val` of all rows.
#[pyfunction]
#[pyo3(signature = (n, prop_train, prop_val, seed, strata = None))]
fn py_initial_validation_split<'py>(
    py: Python<'py>,
    n: usize,
    prop_train: f64,
    prop_val: f64,
    seed: u64,
    strata: Option<Vec<Option<String>>>,
) -> PyResult<(Bound<'py, PyArray1<u64>>, Bound<'py, PyArray1<u64>>, Bound<'py, PyArray1<u64>>)> {
    let codes = strata_codes(strata)?;
    let s = initial_validation_split(n, (prop_train, prop_val), codes.as_deref(), Seed(seed)).py()?;
    let conv = |v: &[usize]| v.iter().map(|&i| i as u64).collect::<Vec<u64>>().into_pyarray(py);
    Ok((conv(&s.train), conv(&s.validation), conv(&s.test)))
}

/// rsample's `make_strata()`: stratum codes for numeric values (`breaks` quantile bins,
/// at most `floor(n / depth)`) or labels (classes below `pool` are pooled at random).
#[pyfunction]
#[pyo3(signature = (values, breaks = 4, pool = 0.1, depth = 20, seed = 0))]
fn py_make_strata<'py>(
    py: Python<'py>,
    values: &Bound<'py, PyAny>,
    breaks: usize,
    pool: f64,
    depth: usize,
    seed: u64,
) -> PyResult<Bound<'py, PyArray1<u32>>> {
    let codes = if let Ok(nums) = values.extract::<Vec<Option<f64>>>() {
        let opt: Vec<Option<f64>> = nums.into_iter().map(|v| v.filter(|x| !x.is_nan())).collect();
        make_strata_numeric(&opt, breaks, depth, Seed(seed)).py()?
    } else {
        let labels: Vec<Option<String>> = values.extract()?;
        let mut levels: Vec<String> = labels.iter().flatten().cloned().collect();
        levels.sort();
        levels.dedup();
        let codes: Vec<Option<u32>> = labels
            .iter()
            .map(|l| l.as_ref().map(|l| levels.iter().position(|x| x == l).unwrap() as u32))
            .collect();
        make_strata_categorical(&codes, &levels, pool, Seed(seed)).py()?
    };
    Ok(codes.into_pyarray(py))
}

/// rsample-style initial split; returns `(train_rows, test_rows)`.
#[pyfunction]
#[pyo3(signature = (n, prop, seed, strata = None))]
fn py_initial_split<'py>(
    py: Python<'py>,
    n: usize,
    prop: f64,
    seed: u64,
    strata: Option<Vec<Option<String>>>,
) -> PyResult<(Bound<'py, PyArray1<u64>>, Bound<'py, PyArray1<u64>>)> {
    let codes = strata_codes(strata)?;
    Ok(ids_pair(py, &initial_split(n, prop, codes.as_deref(), Seed(seed)).py()?))
}

// ---------------------------------------------------------------- Workflow

/// Recipe + model.
#[pyclass(name = "Workflow", module = "tidyrustmodels._native")]
struct PyWorkflow {
    inner: Workflow,
}

#[pymethods]
impl PyWorkflow {
    /// `case_weights`: name of a numeric column of the training data holding case
    /// weights; `weight_kind` is `"frequency"` (whole numbers; used by unsupervised
    /// recipe steps, the model and resampling metrics) or `"importance"` (model only),
    /// as `workflows::add_case_weights()` with `hardhat::frequency_weights()` /
    /// `hardhat::importance_weights()`.
    #[new]
    #[pyo3(signature = (recipe, model, case_weights = None, weight_kind = "frequency"))]
    fn new(
        recipe: PyRecipe,
        model: &Bound<'_, PyAny>,
        case_weights: Option<String>,
        weight_kind: &str,
    ) -> PyResult<Self> {
        let mut wf = Workflow::from_boxed(recipe.inner, model_from_py(model)?);
        if let Some(col) = case_weights {
            wf = wf.case_weights(col, weight_kind_from_str(weight_kind)?);
        }
        Ok(PyWorkflow { inner: wf })
    }

    fn fit(&self, data: &PyTable) -> PyResult<PyFittedWorkflow> {
        Ok(PyFittedWorkflow { inner: self.inner.fit(&data.inner).py()? })
    }

    /// `tune::finalize_workflow()`: the workflow with the tuning parameters `params`
    /// (`{name: value}`) set on the model.
    fn finalize(&self, params: &Bound<'_, PyDict>) -> PyResult<PyWorkflow> {
        let ps = param_set(params)?;
        Ok(PyWorkflow { inner: self.inner.finalize(&ps).py()? })
    }

    /// `tune::last_fit()`: fit on `train_rows` exactly as a resample is fitted (no
    /// sparse-data toggle; frequency case weights reach the metrics), predict and score
    /// `test_rows`. Returns `{"workflow": FittedWorkflow, "metrics": {name: value},
    /// "predictions": array (regression) | {"class": [...], "prob": array}}`.
    fn last_fit<'py>(
        &self,
        py: Python<'py>,
        data: &PyTable,
        train_rows: Vec<usize>,
        test_rows: Vec<usize>,
        metrics: Vec<String>,
    ) -> PyResult<Bound<'py, PyDict>> {
        let set = MetricSet::from_names(&metrics).py()?;
        let split = Split::new(train_rows, test_rows, data.inner.nrows()).py()?;
        let lf = self.inner.last_fit(&data.inner, &split, &set).py()?;
        let d = PyDict::new(py);
        d.set_item("workflow", PyFittedWorkflow { inner: lf.workflow }.into_pyobject(py)?)?;
        let m = PyDict::new(py);
        for v in &lf.metrics {
            m.set_item(&v.name, v.value)?;
        }
        d.set_item("metrics", m)?;
        d.set_item("predictions", prediction_to_py(py, &lf.predictions)?)?;
        Ok(d)
    }

    /// Fit on each fold (prep inside the fold) and evaluate `metrics` (yardstick names).
    /// Returns `{"per_fold": [{"id", metrics...}], "summary": [{"metric","mean","std_err","n"}]}`.
    #[pyo3(signature = (data, folds, metrics, threads = None))]
    fn fit_resamples<'py>(
        &self,
        py: Python<'py>,
        data: &PyTable,
        folds: &PyFolds,
        metrics: Vec<String>,
        threads: Option<usize>,
    ) -> PyResult<Bound<'py, PyDict>> {
        let set = MetricSet::from_names(&metrics).py()?;
        let par = match threads {
            Some(k) => Parallelism::threads(k),
            None => Parallelism::SERIAL,
        };
        let res = self.inner.fit_resamples(&data.inner, &folds.inner, &set, par).py()?;
        resample_results_to_py(py, &res)
    }
}

fn weight_kind_from_str(s: &str) -> PyResult<WeightKind> {
    match s {
        "frequency" => Ok(WeightKind::Frequency),
        "importance" => Ok(WeightKind::Importance),
        other => Err(InvalidArgumentError::new_err(format!(
            "weight_kind must be \"frequency\" or \"importance\", got {other:?}"
        ))),
    }
}

fn resample_results_to_py<'py>(py: Python<'py>, res: &ResampleResults) -> PyResult<Bound<'py, PyDict>> {
    let out = PyDict::new(py);
    let per_fold = PyList::empty(py);
    for f in &res.per_fold {
        let d = PyDict::new(py);
        d.set_item("id", &f.id)?;
        for m in &f.metrics {
            d.set_item(&m.name, m.value)?;
        }
        per_fold.append(d)?;
    }
    let summary = PyList::empty(py);
    for s in &res.summary {
        let d = PyDict::new(py);
        d.set_item("metric", &s.name)?;
        d.set_item("mean", s.mean)?;
        d.set_item("std_err", s.std_err)?;
        d.set_item("n", s.n)?;
        summary.append(d)?;
    }
    out.set_item("per_fold", per_fold)?;
    out.set_item("summary", summary)?;
    Ok(out)
}

/// A fitted workflow.
#[pyclass(name = "FittedWorkflow", module = "tidyrustmodels._native")]
struct PyFittedWorkflow {
    inner: FittedWorkflow,
}

#[pymethods]
impl PyFittedWorkflow {
    /// Regression: float64 array. Classification: predicted class labels (list of str).
    fn predict<'py>(&self, py: Python<'py>, data: &PyTable) -> PyResult<Bound<'py, PyAny>> {
        match self.inner.predict(&data.inner).py()? {
            Prediction::Numeric(v) => Ok(v.into_pyarray(py).into_any()),
            Prediction::Class { codes, levels, .. } => {
                let labels: Vec<&str> = codes.iter().map(|c| levels[*c as usize].as_str()).collect();
                Ok(PyList::new(py, labels)?.into_any())
            }
        }
    }

    /// Classification only: `n x n_levels` float64 array, columns in `levels` order.
    fn predict_proba<'py>(&self, py: Python<'py>, data: &PyTable) -> PyResult<Bound<'py, PyAny>> {
        match self.inner.predict(&data.inner).py()? {
            Prediction::Class { probabilities: Some(p), .. } => {
                let (n, k) = (p.nrows(), p.ncols());
                Ok(PyArray1::from_vec(py, p.to_row_major()).reshape([n, k])?.into_any())
            }
            Prediction::Class { probabilities: None, .. } => Err(UnsupportedError::new_err("model has no class probabilities")),
            Prediction::Numeric(_) => Err(UnsupportedError::new_err("predict_proba is for classification models")),
        }
    }

    /// Outcome levels for classification models, else `None`.
    fn levels(&self, data: &PyTable) -> PyResult<Option<Vec<String>>> {
        Ok(match self.inner.predict(&data.inner).py()? {
            Prediction::Class { levels, .. } => Some(levels),
            Prediction::Numeric(_) => None,
        })
    }

    /// `{"(Intercept)": float | None, feature: float, ...}` in design-matrix order; for a
    /// multinomial model, `{level: {...}}` with one such dict per non-baseline level.
    fn coefficients<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        let d = PyDict::new(py);
        let fill = |d: &Bound<'py, PyDict>, c: Coefficients| -> PyResult<()> {
            d.set_item("(Intercept)", c.intercept)?;
            for (name, v) in c.terms {
                d.set_item(name, v)?;
            }
            Ok(())
        };
        if let Some(sets) = self.inner.model().class_coefficients() {
            for (level, c) in sets {
                let inner = PyDict::new(py);
                fill(&inner, c)?;
                d.set_item(level, inner)?;
            }
        } else if let Some(c) = self.inner.model().coefficients() {
            fill(&d, c)?;
        }
        Ok(d)
    }

    /// Feature names the model was fitted on.
    fn feature_names(&self) -> Vec<String> {
        self.inner.model().feature_names().to_vec()
    }

    /// Solver diagnostics as a dict.
    fn diagnostics<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        let dg = self.inner.model().diagnostics();
        let d = PyDict::new(py);
        d.set_item("converged", dg.converged)?;
        d.set_item("iterations", dg.iterations)?;
        d.set_item("final_gradient_norm", dg.final_gradient_norm)?;
        d.set_item("deviance", dg.deviance)?;
        d.set_item("warnings", dg.warnings)?;
        d.set_item("aliased", dg.aliased)?;
        d.set_item("note", dg.note)?;
        Ok(d)
    }

    /// Backend facts: engine, crate_version, solver.
    fn backend<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        let b = self.inner.model().backend();
        let d = PyDict::new(py);
        d.set_item("engine", b.engine)?;
        d.set_item("crate_version", b.crate_version)?;
        d.set_item("solver", b.solver)?;
        Ok(d)
    }

    /// Versioned JSON string.
    fn to_json(&self) -> PyResult<String> {
        serde_json::to_string(&self.inner.to_json().py()?).map_err(|e| SerializationError::new_err(e.to_string()))
    }

    #[staticmethod]
    fn from_json(text: &str) -> PyResult<Self> {
        let v: serde_json::Value = serde_json::from_str(text).map_err(|e| SerializationError::new_err(e.to_string()))?;
        Ok(PyFittedWorkflow { inner: FittedWorkflow::from_json(&v).py()? })
    }
}

// ---------------------------------------------------------------- Tuning

/// Evaluate a grid of `{param: value}` dicts with `fit_resamples` per candidate.
fn param_set(params: &Bound<'_, PyDict>) -> PyResult<Vec<(String, f64)>> {
    params.iter().map(|(k, v)| Ok((k.extract::<String>()?, v.extract::<f64>()?))).collect()
}

fn param_set_to_py<'py>(py: Python<'py>, params: &[(String, f64)]) -> PyResult<Bound<'py, PyDict>> {
    let p = PyDict::new(py);
    for (k, v) in params {
        p.set_item(k, v)?;
    }
    Ok(p)
}

fn prediction_to_py<'py>(py: Python<'py>, pred: &Prediction) -> PyResult<Bound<'py, PyAny>> {
    Ok(match pred {
        Prediction::Numeric(v) => v.clone().into_pyarray(py).into_any(),
        Prediction::Class {
            codes,
            levels,
            probabilities,
        } => {
            let d = PyDict::new(py);
            d.set_item("class", codes.iter().map(|c| levels[*c as usize].clone()).collect::<Vec<_>>())?;
            match probabilities {
                Some(p) => {
                    let (n, k) = (p.nrows(), p.ncols());
                    d.set_item("prob", PyArray1::from_vec(py, p.to_row_major()).reshape([n, k])?)?;
                }
                None => d.set_item("prob", py.None())?,
            }
            d.into_any()
        }
    })
}

fn sort_keys(sort: &[String]) -> PyResult<Vec<SortKey>> {
    sort.iter().map(|k| SortKey::parse(k).py()).collect()
}

fn selection_to_py<'py>(py: Python<'py>, sel: &Selection) -> PyResult<Bound<'py, PyDict>> {
    let d = PyDict::new(py);
    d.set_item("index", sel.index)?;
    d.set_item("config", sel.config)?;
    d.set_item("params", param_set_to_py(py, &sel.params)?)?;
    Ok(d)
}

/// Results of `tune_grid`: one entry per grid candidate (`len()`, indexing gives
/// `{"params", "results"}`), plus tune's selection rules.
#[pyclass(name = "TuneResults", module = "tidyrustmodels._native")]
struct PyTuneResults {
    inner: TuneResults,
}

#[pymethods]
impl PyTuneResults {
    fn __len__(&self) -> usize {
        self.inner.candidates.len()
    }

    /// tune's `.config` number of every candidate (grid sorted on the parameters in
    /// parsnip's argument order; identical candidates share a number).
    fn configs(&self) -> Vec<usize> {
        self.inner.configs()
    }

    fn __getitem__<'py>(&self, py: Python<'py>, i: usize) -> PyResult<Bound<'py, PyDict>> {
        let (params, r) = self
            .inner
            .candidates
            .get(i)
            .ok_or_else(|| pyo3::exceptions::PyIndexError::new_err("candidate index out of range"))?;
        let d = PyDict::new(py);
        d.set_item("params", param_set_to_py(py, params)?)?;
        d.set_item("results", resample_results_to_py(py, r)?)?;
        Ok(d)
    }

    /// `tune::show_best()`: the top `n` candidates for `metric` as
    /// `{"index", "params", "mean", "std_err", "n"}`, best first (ties in grid order).
    #[pyo3(signature = (metric, n = 5))]
    fn show_best<'py>(&self, py: Python<'py>, metric: &str, n: usize) -> PyResult<Bound<'py, PyList>> {
        let m = Metric::from_name(metric).py()?;
        let out = PyList::empty(py);
        for b in self.inner.show_best(m, n) {
            let d = PyDict::new(py);
            d.set_item("index", b.index)?;
            d.set_item("config", b.config)?;
            d.set_item("params", param_set_to_py(py, &b.params)?)?;
            d.set_item("mean", b.mean)?;
            d.set_item("std_err", b.std_err)?;
            d.set_item("n", b.n)?;
            out.append(d)?;
        }
        Ok(out)
    }

    /// `tune::select_best()`: `{"index", "params"}` of the best candidate.
    fn select_best<'py>(&self, py: Python<'py>, metric: &str) -> PyResult<Bound<'py, PyDict>> {
        let m = Metric::from_name(metric).py()?;
        selection_to_py(py, &self.inner.select_best(m).py()?)
    }

    /// `tune::select_by_one_std_err(x, ..., metric)`: `sort` lists the `...`
    /// (`"param"` or `"desc(param)"`, simplest model first).
    fn select_by_one_std_err<'py>(&self, py: Python<'py>, metric: &str, sort: Vec<String>) -> PyResult<Bound<'py, PyDict>> {
        let m = Metric::from_name(metric).py()?;
        selection_to_py(py, &self.inner.select_by_one_std_err(m, &sort_keys(&sort)?).py()?)
    }

    /// `tune::select_by_pct_loss(x, ..., metric, limit)`.
    #[pyo3(signature = (metric, sort, limit = 2.0))]
    fn select_by_pct_loss<'py>(&self, py: Python<'py>, metric: &str, sort: Vec<String>, limit: f64) -> PyResult<Bound<'py, PyDict>> {
        let m = Metric::from_name(metric).py()?;
        selection_to_py(py, &self.inner.select_by_pct_loss(m, &sort_keys(&sort)?, limit).py()?)
    }

    /// `tune::fit_best()`: the best candidate for `metric` refitted on all of `data`
    /// with `Workflow.fit` (sparse-data decision applied, as parsnip's `fit()`).
    fn fit_best(&self, workflow: &PyWorkflow, data: &PyTable, metric: &str) -> PyResult<PyFittedWorkflow> {
        let m = Metric::from_name(metric).py()?;
        Ok(PyFittedWorkflow { inner: fit_best(&self.inner, &workflow.inner, &data.inner, m).py()? })
    }
}

#[pyfunction]
#[pyo3(signature = (workflow, data, folds, grid, metrics, threads = None))]
fn py_tune_grid<'py>(
    workflow: &PyWorkflow,
    data: &PyTable,
    folds: &PyFolds,
    grid: Vec<Bound<'py, PyDict>>,
    metrics: Vec<String>,
    threads: Option<usize>,
) -> PyResult<PyTuneResults> {
    let points = grid
        .iter()
        .map(|d| {
            d.iter().map(|(k, v)| Ok((k.extract::<String>()?, v.extract::<f64>()?))).collect::<PyResult<Vec<_>>>()
        })
        .collect::<PyResult<Vec<_>>>()?;
    let g = Grid { points };
    let set = MetricSet::from_names(&metrics).py()?;
    let par = match threads {
        Some(k) => Parallelism::threads(k),
        None => Parallelism::SERIAL,
    };
    let res = tune_grid(&workflow.inner, &data.inner, &folds.inner, &g, &set, par).py()?;
    Ok(PyTuneResults { inner: res })
}

/// yardstick metrics on class labels and, for probability metrics, a probability matrix
/// with one column per level in `levels` order. `estimate` defaults to the arg-max of
/// each probability row. Returns `{name: value}` with `None` where yardstick gives `NA`;
/// estimators follow yardstick's defaults (binary / macro / hand_till for roc_auc;
/// macro when `case_weights` are given, as yardstick). `case_weights` are yardstick's
/// `case_weights =` (one non-negative weight per row).
#[pyfunction]
#[pyo3(signature = (truth, levels, metrics, estimate = None, probabilities = None, case_weights = None))]
fn py_class_metrics<'py>(
    py: Python<'py>,
    truth: Vec<String>,
    levels: Vec<String>,
    metrics: Vec<String>,
    estimate: Option<Vec<String>>,
    probabilities: Option<PyReadonlyArray2<'py, f64>>,
    case_weights: Option<Vec<f64>>,
) -> PyResult<Bound<'py, PyDict>> {
    let code = |labels: &[String]| -> PyResult<Vec<u32>> {
        labels
            .iter()
            .map(|l| {
                levels
                    .iter()
                    .position(|x| x == l)
                    .map(|i| i as u32)
                    .ok_or_else(|| InvalidValueError::new_err(format!("label `{l}` is not one of the levels")))
            })
            .collect()
    };
    let t = code(&truth)?;
    let probs = match probabilities {
        Some(a) => {
            let shape = a.shape();
            let view = a.as_array();
            let mut flat = Vec::with_capacity(shape[0] * shape[1]);
            for i in 0..shape[0] {
                for j in 0..shape[1] {
                    flat.push(view[[i, j]]);
                }
            }
            Some(DesignMatrix::from_row_major(&flat, shape[0], shape[1], levels.clone()).py()?)
        }
        None => None,
    };
    let e = match (&estimate, &probs) {
        (Some(e), _) => code(e)?,
        (None, Some(p)) => (0..p.nrows())
            .map(|i| {
                let row = p.row(i);
                let mut best = 0;
                for (j, v) in row.iter().enumerate() {
                    if *v > row[best] {
                        best = j;
                    }
                }
                best as u32
            })
            .collect(),
        (None, None) => return Err(InvalidArgumentError::new_err("give `estimate` labels or `probabilities`")),
    };
    let set = MetricSet::from_names(&metrics).py()?;
    let outcome = Outcome::Class { codes: t, levels: levels.clone() };
    let pred = Prediction::Class { codes: e, levels, probabilities: probs };
    let d = PyDict::new(py);
    for v in set.evaluate_weighted(&outcome, &pred, case_weights.as_deref()).py()? {
        d.set_item(v.name, v.value)?;
    }
    Ok(d)
}

/// Confusion matrix as nested lists, rows = prediction, columns = truth (yardstick's
/// `conf_mat()$table` orientation), in `levels` order.
#[pyfunction]
fn py_conf_mat(truth: Vec<String>, estimate: Vec<String>, levels: Vec<String>) -> PyResult<Vec<Vec<f64>>> {
    let code = |labels: &[String]| -> PyResult<Vec<u32>> {
        labels
            .iter()
            .map(|l| {
                levels
                    .iter()
                    .position(|x| x == l)
                    .map(|i| i as u32)
                    .ok_or_else(|| InvalidValueError::new_err(format!("label `{l}` is not one of the levels")))
            })
            .collect()
    };
    let cm = ConfusionMatrix::from_codes(&code(&truth)?, &code(&estimate)?, &levels).py()?;
    let k = cm.n_levels();
    Ok((0..k).map(|r| (0..k).map(|c| cm.count(r, c)).collect()).collect())
}

/// `dials::grid_regular` over `(name, lower, upper, log10)` ranges.
#[pyfunction]
fn py_grid_regular<'py>(py: Python<'py>, ranges: Vec<(String, f64, f64, bool)>, levels: usize) -> PyResult<Bound<'py, PyList>> {
    let rs: Vec<ParamRange> = ranges
        .into_iter()
        .map(|(name, lo, hi, log10)| if log10 { ParamRange::log10(&name, lo, hi) } else { ParamRange::linear(&name, lo, hi) })
        .collect();
    let g = grid_regular(&rs, levels).py()?;
    let out = PyList::empty(py);
    for p in g.points {
        let d = PyDict::new(py);
        for (k, v) in p {
            d.set_item(k, v)?;
        }
        out.append(d)?;
    }
    Ok(out)
}

#[pymodule]
fn _native(m: &Bound<'_, PyModule>) -> PyResult<()> {
    let py = m.py();
    m.add_class::<PyTable>()?;
    m.add_class::<PyRecipe>()?;
    m.add_class::<PyLinearReg>()?;
    m.add_class::<PyLogisticReg>()?;
    m.add_class::<PyMultinomReg>()?;
    m.add_class::<PyDecisionTree>()?;
    m.add_class::<PyRandForest>()?;
    m.add_class::<PyBoostTree>()?;
    m.add_class::<PyFolds>()?;
    m.add_class::<PyTuneResults>()?;
    m.add_class::<PyWorkflow>()?;
    m.add_class::<PyFittedWorkflow>()?;
    m.add_function(wrap_pyfunction!(py_vfold_cv, m)?)?;
    m.add_function(wrap_pyfunction!(py_initial_split, m)?)?;
    m.add_function(wrap_pyfunction!(py_bootstraps, m)?)?;
    m.add_function(wrap_pyfunction!(py_mc_cv, m)?)?;
    m.add_function(wrap_pyfunction!(py_loo_cv, m)?)?;
    m.add_function(wrap_pyfunction!(py_group_vfold_cv, m)?)?;
    m.add_function(wrap_pyfunction!(py_group_bootstraps, m)?)?;
    m.add_function(wrap_pyfunction!(py_group_mc_cv, m)?)?;
    m.add_function(wrap_pyfunction!(py_group_initial_split, m)?)?;
    m.add_function(wrap_pyfunction!(py_rolling_origin, m)?)?;
    m.add_function(wrap_pyfunction!(py_sliding_window, m)?)?;
    m.add_function(wrap_pyfunction!(py_sliding_index, m)?)?;
    m.add_function(wrap_pyfunction!(py_initial_time_split, m)?)?;
    m.add_function(wrap_pyfunction!(py_initial_validation_split, m)?)?;
    m.add_function(wrap_pyfunction!(py_make_strata, m)?)?;
    m.add_function(wrap_pyfunction!(py_tune_grid, m)?)?;
    m.add_function(wrap_pyfunction!(py_grid_regular, m)?)?;
    m.add_function(wrap_pyfunction!(py_class_metrics, m)?)?;
    m.add_function(wrap_pyfunction!(py_conf_mat, m)?)?;
    m.add("TidyRustModelsError", py.get_type::<TidyRustModelsError>())?;
    m.add("SchemaError", py.get_type::<SchemaError>())?;
    m.add("InvalidValueError", py.get_type::<InvalidValueError>())?;
    m.add("InvalidArgumentError", py.get_type::<InvalidArgumentError>())?;
    m.add("UnsupportedError", py.get_type::<UnsupportedError>())?;
    m.add("BackendError", py.get_type::<BackendError>())?;
    m.add("SerializationError", py.get_type::<SerializationError>())?;
    m.add("__version__", env!("CARGO_PKG_VERSION"))?;
    Ok(())
}
