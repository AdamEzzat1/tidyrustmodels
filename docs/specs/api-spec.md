# Initial API specification (v0.1 vertical slice)

This is the contract the vertical slice implements. Names are final for v0.1 unless an
ADR changes them.

## tidyrustmodels-core

```rust
pub enum Role { Predictor, Outcome, Id, Ignored }
pub enum ColumnKind { Numeric, Categorical }

pub struct Column { /* Numeric{values, valid} | Categorical{codes, levels, valid} */ }
pub struct Table   { columns: Vec<(String, Column)> }      // equal-length, unique names
pub struct Schema  { fields: Vec<Field { name, kind, role }> }

pub struct DesignMatrix { data: Vec<f64> /* column-major */, nrows, ncols, names: Vec<String> }
pub enum Outcome { Numeric(Vec<f64>), Class { codes: Vec<u32>, levels: Vec<String> } }

pub enum Error { SchemaMismatch{..}, InvalidValue{..}, Unsupported{..},
                 Backend{ engine, source }, Serialization(..), InvalidArgument(..) }
pub type Result<T> = std::result::Result<T, Error>;

pub trait Step: Debug + Send + Sync { fn prep(&self, training: &Table, schema: &Schema) -> Result<Box<dyn FittedStep>>; }
pub trait FittedStep: Debug + Send + Sync { fn bake(&self, data: &Table) -> Result<Table>; fn output_schema(&self) -> &Schema; }

pub enum Capability { Regression, BinaryClassification, ClassProbabilities, Coefficients }
pub struct BackendInfo { engine: &'static str, crate_version: &'static str, probabilities_derived_by_adapter: bool }

pub trait ModelSpec: Debug + Send + Sync {
    fn capabilities(&self) -> &'static [Capability];
    fn backend(&self) -> BackendInfo;
    fn fit(&self, x: &DesignMatrix, y: &Outcome) -> Result<Box<dyn FittedModel>>;
}
pub enum Prediction { Numeric(Vec<f64>), Class { codes: Vec<u32>, probabilities: Option<DesignMatrix> } }
pub trait FittedModel: Debug + Send + Sync {
    fn predict(&self, x: &DesignMatrix) -> Result<Prediction>;
    fn coefficients(&self) -> Option<Coefficients>;   // intercept + named slopes
    fn feature_names(&self) -> &[String];
}

pub struct Seed(pub u64);           // every stochastic API takes a Seed
pub struct Parallelism { pub threads: Option<usize> }   // None = serial
```

## tidyrustmodels-recipes

```rust
pub struct Recipe { steps: Vec<Box<dyn Step>>, outcome: String }
impl Recipe {
    pub fn new(outcome: &str) -> Self;
    pub fn step(self, s: impl Step + 'static) -> Self;
    pub fn prep(&self, training: &Table) -> Result<FittedRecipe>;
}
impl FittedRecipe { pub fn bake(&self, data: &Table) -> Result<(DesignMatrix, Option<Outcome>)>; }

// Steps (names mirror recipes):
StepSelect::columns(&[..]) / StepSelect::all_numeric_predictors()
StepImputeMean::new(selector) / StepImputeMedian::new(selector)
StepNormalize::new(selector)          // (x - mean) / sd, sd with n-1, na.rm
StepDummy::new(selector).one_hot(bool) // default false (drop first level)
```

Semantics: `prep` computes statistics on training rows only. `bake` on data with an
unseen categorical level is an error (recipes warns and yields NA; we choose the strict
behaviour and document it in COMPATIBILITY.md).

## tidyrustmodels-models

```rust
pub enum Engine { Linfa, Smartcore }
pub struct LinearRegSpec  { engine: Engine, intercept: bool }
pub struct LogisticRegSpec { engine: Engine, intercept: bool, penalty: f64 /* L2 alpha, 0 = MLE */,
                             max_iterations: u64, gradient_tolerance: f64 }
```

## tidyrustmodels-resample

```rust
// an rsample rsplit (ADR-0013): analysis rows in draw order (repeats allowed),
// assessment rows sorted (may be empty); rows may be in neither set
pub struct Split { pub train: Vec<usize>, pub test: Vec<usize> }
impl Split {
    pub fn from_ids(train, test, n) -> Result<Split>;   // strict partition (initial split)
    pub fn new(train, test, n) -> Result<Split>;        // rsample::make_splits(list(analysis, assessment))
    pub fn is_partition(&self, n: usize) -> bool;
}
// an rset: rsample ids follow names0 (Fold1, Fold01, Bootstrap01, Resample1, Slice1, Repeat1.Fold1)
pub struct Folds { pub splits: Vec<Split>, pub ids: Vec<String>, pub n: usize }
impl Folds {
    pub fn from_assignments(fold_of_row: &[u32]) -> Result<Folds>;              // v-fold from a fold index per row
    pub fn from_splits(splits: Vec<Split>, ids: Vec<String>, n: usize) -> Result<Folds>;   // manual_rset
    pub fn assignments(&self) -> Vec<u32>; pub fn is_partition(&self) -> bool;
}
pub fn names0(num: usize, prefix: &str) -> Vec<String>;
// random schemes (framework RNG; rsample's algorithms)
pub fn initial_split(n, prop, strata: Option<&[u32]>, seed) -> Result<Split>;
pub fn initial_validation_split(n, (prop_train, prop_val), strata, seed) -> Result<ThreeWaySplit>;  // .validation_set()
pub fn vfold_cv(n, v, strata, seed) -> Result<Folds>;
pub fn vfold_cv_repeated(n, v, repeats, strata, seed) -> Result<Folds>;
pub fn bootstraps(n, times, strata, seed) -> Result<Folds>;      // assessment = out-of-bag
pub fn mc_cv(n, prop, times, strata, seed) -> Result<Folds>;
pub fn loo_cv(n, seed) -> Result<Folds>;
pub fn group_vfold_cv(groups: &[u32], v: Option<usize>, repeats, balance: GroupBalance, seed) -> Result<Folds>;
pub fn group_bootstraps(groups, times, seed) -> Result<Folds>;
pub fn group_mc_cv(groups, prop, times, seed) -> Result<Folds>;
pub fn group_initial_split(groups, prop, seed) -> Result<Split>;
pub fn balance_observations(table, perm, v, n_obs) -> Vec<usize>;   // rsample's rules on a given permutation
pub fn balance_prop(table, draws, prop) -> Vec<usize>;
// deterministic schemes (reproduce rsample exactly)
pub fn rolling_origin(n, initial, assess, cumulative, skip, lag) -> Result<Folds>;
pub fn sliding_window(n, lookback: Option<usize>, assess_start, assess_stop, complete, step, skip) -> Result<Folds>;
pub fn sliding_index(index: &[f64], lookback: Option<f64>, assess_start, assess_stop, complete, step, skip) -> Result<Folds>;
pub fn initial_time_split(n, prop) -> Result<Split>;
// strata
pub fn make_strata_categorical(codes: &[Option<u32>], levels, pool, seed) -> Result<Vec<u32>>;
pub fn make_strata_numeric(values: &[Option<f64>], breaks, depth, seed) -> Result<Vec<u32>>;
pub fn strata_from_numeric(values: &[f64], breaks) -> Result<Vec<u32>>;
```

Invariants (unit-tested): v-fold assessment sets partition `0..n` with sizes differing
by ≤ 1 per stratum; bootstraps draw `n` rows and assess the out-of-bag rows; grouped
schemes never place a group on both sides; time-series windows equal rsample's
(checked against R, Phase 8h).

## tidyrustmodels-metrics

```rust
pub fn rmse(truth: &[f64], estimate: &[f64]) -> Result<f64>;
pub fn mae(..); pub fn rsq(..) /* squared correlation */; pub fn rsq_trad(..);
pub fn accuracy(truth: &[u32], estimate: &[u32]) -> Result<f64>;
pub fn precision(truth, estimate, event: u32) -> Result<Option<f64>>;  // None when undefined (yardstick NA)
pub fn recall(..) -> Result<Option<f64>>;
pub fn mn_log_loss(truth: &[u32], prob_event: &[f64], event: u32) -> Result<f64>;
pub struct MetricSet(Vec<Metric>);  // evaluates a Prediction + Outcome into Vec<(name, value)>
```

## tidyrustmodels-workflows

```rust
pub struct Workflow { recipe: Recipe, model: Box<dyn ModelSpec>, case_weights: Option<(String, WeightKind)> }
impl Workflow {
    pub fn case_weights(self, column: &str, kind: WeightKind) -> Self;   // workflows::add_case_weights (ADR-0012)
    pub fn fit(&self, training: &Table) -> Result<FittedWorkflow>;
    pub fn fit_fold(&self, train: &Table, test: &Table, metrics: &MetricSet) -> Result<Vec<MetricValue>>;  // one fold as tune runs it
    pub fn fit_resamples(&self, data: &Table, folds: &Folds, metrics: &MetricSet, par: Parallelism)
        -> Result<ResampleResults>;   // prep inside each fold
}
impl FittedWorkflow { pub fn predict(&self, data: &Table) -> Result<Prediction>; }
pub struct ResampleResults { pub per_fold: Vec<Vec<(String, f64)>>, pub summary: Vec<(String, f64 /*mean*/, f64 /*std_err*/)> }
```

## tidyrustmodels-tune

```rust
pub struct Grid { points: Vec<ParamSet> }   // ParamSet = Vec<(String, f64)>
pub fn grid_regular(ranges: &[(name, lo, hi)], levels: usize) -> Grid;
pub fn grid_random(ranges, size, seed) -> Grid;
pub fn tune_grid(workflow: &Workflow, data: &Table, folds: &Folds, grid: &Grid,
                 metrics: &MetricSet, par: Parallelism) -> Result<TuneResults>;
impl TuneResults { pub fn show_best(&self, metric: &str, maximize: bool, n: usize) -> Vec<(ParamSet, f64, f64)>; }
```

The tunable parameters in v0.1 are `penalty` for `LogisticRegSpec` (so tuning is
exercised end-to-end) and `intercept` (boolean, encoded 0/1).

## Python (`tidyrustmodels`)

```python
tbl = trm.from_pandas(df)                       # or trm.Table.from_numpy(X, names=[...]) / Table.from_dict({...})
rec = trm.Recipe("y").step_normalize("all_numeric_predictors").step_dummy(["species"])
wf  = trm.Workflow(rec, trm.LinearReg(engine="linfa"))   # case_weights="w", weight_kind="frequency"|"importance" (native engine)
fit = wf.fit(tbl); fit.predict(tbl) -> np.ndarray   # classification: list[str]; fit.predict_proba(tbl) -> np.ndarray
folds = trm.vfold_cv(tbl.nrows, 5, seed=42, strata=list(df["y"]))
res = wf.fit_resamples(tbl, folds, ["rmse", "rsq"], threads=4)
```

Implemented deviation (review F7): `from_pandas` is a module-level function because
PyO3 classes are final and a Python subclass cannot be returned from native
constructors; selectors are strings or lists of column names.

Exceptions: `TidyRustModelsError` (base) → `SchemaError`, `InvalidValueError`,
`UnsupportedError`, `BackendError`.
