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
pub struct Split { pub train: Vec<usize>, pub test: Vec<usize> }
pub fn initial_split(n: usize, prop: f64, strata: Option<&[u32]>, seed: Seed) -> Result<Split>;
pub struct Folds { pub splits: Vec<Split>, pub ids: Vec<String> }
pub fn vfold_cv(n: usize, v: usize, strata: Option<&[u32]>, seed: Seed) -> Result<Folds>;
impl Folds { pub fn from_assignments(fold_of_row: &[u32]) -> Result<Folds>; }   // fixture loader
```

Invariants (property-tested): train ∩ test = ∅, train ∪ test = 0..n, |fold sizes| differ
by ≤ 1, stratified folds preserve per-stratum balance within ±1.

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
pub struct Workflow { recipe: Recipe, model: Box<dyn ModelSpec> }
impl Workflow {
    pub fn fit(&self, training: &Table) -> Result<FittedWorkflow>;
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
tbl = trm.Table.from_pandas(df)                 # or from_numpy(X, names=[...])
rec = trm.Recipe(outcome="y").step_normalize(trm.all_numeric_predictors()).step_dummy(["species"])
wf  = trm.Workflow(rec, trm.LinearReg(engine="linfa"))
fit = wf.fit(tbl); fit.predict(tbl) -> np.ndarray
folds = trm.vfold_cv(tbl, v=5, strata="y", seed=42)
res = wf.fit_resamples(tbl, folds, metrics=["rmse", "rsq"], threads=4)
```

Exceptions: `TidyRustModelsError` (base) → `SchemaError`, `InvalidValueError`,
`UnsupportedError`, `BackendError`.
