//! Workflows: a recipe plus a model specification, fitted together and evaluated
//! across resamples with the recipe re-prepped inside every fold.

#![forbid(unsafe_code)]
#![warn(missing_docs, rust_2018_idioms)]

use serde::{Deserialize, Serialize};
use tidyrustmodels_core::sparsity::should_use_sparsity;
use tidyrustmodels_core::{
    Capability, CaseWeights, DesignMatrix, Error, FittedModel, ModelSpec, Outcome, Parallelism, Prediction,
    Result, Table, WeightKind,
};
use tidyrustmodels_metrics::{MetricSet, MetricValue};
use tidyrustmodels_models::fitted_model_from_state;
use tidyrustmodels_recipes::{FittedRecipe, Recipe};
use tidyrustmodels_resample::{Folds, Split};

/// Serialization format version for fitted workflows.
pub const WORKFLOW_FORMAT_VERSION: u32 = 1;

/// What `tune::last_fit()` returns: the workflow fitted on the analysis rows, its
/// predictions for the assessment rows (in the split's assessment order) and the
/// metrics on them.
#[derive(Debug)]
pub struct LastFit {
    /// The fitted workflow (`extract_workflow()`).
    pub workflow: FittedWorkflow,
    /// Predictions for the assessment rows (`collect_predictions()`).
    pub predictions: Prediction,
    /// Metrics on the assessment rows (`collect_metrics()`).
    pub metrics: Vec<MetricValue>,
}

/// A resampling fold with its preprocessing done: what tune's `.fit_pre` step
/// produces once per resample before the model candidates are fitted on it
/// (tune 2.1.0 `tune_grid_loop_iter` estimates the preprocessor once per resample
/// when only model parameters are tuned). Built by [`Workflow::prepare_fold`]; it
/// depends on the recipe, the case-weight specification and the rows, never on the
/// model, so every candidate of one workflow can share it.
#[derive(Debug)]
pub struct PreparedFold {
    recipe: FittedRecipe,
    x_train: DesignMatrix,
    y_train: Outcome,
    w_train: Option<CaseWeights>,
    x_test: DesignMatrix,
    y_test: Outcome,
    w_test: Option<CaseWeights>,
}

impl PreparedFold {
    /// Rows of the analysis set.
    pub fn n_train(&self) -> usize {
        self.x_train.nrows()
    }

    /// Rows of the assessment set.
    pub fn n_test(&self) -> usize {
        self.x_test.nrows()
    }
}

/// An unfitted workflow.
#[derive(Clone, Debug)]
pub struct Workflow {
    recipe: Recipe,
    model: Box<dyn ModelSpec>,
    case_weights: Option<(String, WeightKind)>,
}

impl Workflow {
    /// Combine a recipe and a model specification.
    pub fn new(recipe: Recipe, model: impl ModelSpec + 'static) -> Self {
        Workflow {
            recipe,
            model: Box::new(model),
            case_weights: None,
        }
    }

    /// Combine with an already boxed specification (used by tuning).
    pub fn from_boxed(recipe: Recipe, model: Box<dyn ModelSpec>) -> Self {
        Workflow {
            recipe,
            model,
            case_weights: None,
        }
    }

    /// `workflows::add_case_weights()`: a numeric column of the training data holds
    /// case weights of the given kind. The column gets the recipe role
    /// `case_weights` (never a predictor); frequency weights are used by the
    /// unsupervised recipe steps, by the model and, in `fit_resamples`, by the
    /// metrics (tune's `.use_case_weights_with_yardstick()`); importance weights reach
    /// the model only. Engines that take no weights fail at fit time with parsnip's
    /// message. Prediction data need not carry the column.
    pub fn case_weights(mut self, column: impl Into<String>, kind: WeightKind) -> Self {
        self.case_weights = Some((column.into(), kind));
        self
    }

    /// The case-weights column and kind, if set.
    pub fn case_weights_spec(&self) -> Option<(&str, WeightKind)> {
        self.case_weights.as_ref().map(|(c, k)| (c.as_str(), *k))
    }

    /// The case weights of `data`'s rows (`None` when the workflow has none).
    pub fn case_weights_of(&self, data: &Table) -> Result<Option<CaseWeights>> {
        match &self.case_weights {
            None => Ok(None),
            Some((col, kind)) => Ok(Some(extract_case_weights(data, col, *kind)?)),
        }
    }

    /// The recipe.
    pub fn recipe(&self) -> &Recipe {
        &self.recipe
    }

    /// The model specification.
    pub fn model(&self) -> &dyn ModelSpec {
        self.model.as_ref()
    }

    /// Prep the recipe on `training`, then fit the model on the baked design matrix.
    /// Like `workflows::fit()`, this applies the sparse-data decision
    /// (`workflows:::toggle_sparsity`: for a model with a `ModelSpec::sparse_engine`,
    /// `core::sparsity::should_use_sparsity` on the recipe's sparsity estimate sets
    /// `step_dummy`'s sparse output, ADR-0015); `fit_resamples` does not, like tune.
    pub fn fit(&self, training: &Table) -> Result<FittedWorkflow> {
        self.fit_with(training, true)
    }

    fn fit_with(&self, training: &Table, toggle_sparsity: bool) -> Result<FittedWorkflow> {
        // workflows::toggle_sparsity: engines that take sparse input decide from the
        // recipe's sparsity estimate whether sparse-capable steps emit sparse columns;
        // the toggled recipe is what the fitted workflow keeps, so prediction bakes
        // new data the same way
        let toggled;
        let recipe_spec = match self.model.sparse_engine() {
            Some(engine) if toggle_sparsity => {
                let est = self.recipe.estimate_sparsity(training);
                toggled = self
                    .recipe
                    .toggle_sparse(should_use_sparsity(est, engine, training.nrows()));
                &toggled
            }
            _ => &self.recipe,
        };
        let recipe = recipe_spec.prep_with_weights(training, self.case_weights_spec())?;
        let weights = self.case_weights_of(training)?;
        let (x, y) = recipe.bake_matrix(training)?;
        let y = y.ok_or_else(|| {
            Error::SchemaMismatch(format!(
                "outcome `{}` is missing from the training data",
                recipe.outcome()
            ))
        })?;
        check_capabilities(self.model.as_ref(), &y)?;
        let model = self.model.fit_maybe_weighted(&x, &y, weights.as_ref())?;
        Ok(FittedWorkflow { recipe, model })
    }

    /// `tune::finalize_workflow()`: the same workflow with the tuning parameters
    /// `params` (`(name, value)` pairs understood by `ModelSpec::with_param`) set on
    /// the model specification; an unknown parameter is an error.
    pub fn finalize(&self, params: &[(String, f64)]) -> Result<Workflow> {
        let mut spec = self.model.clone_box();
        for (name, value) in params {
            spec = spec.with_param(name, *value)?;
        }
        Ok(Workflow {
            recipe: self.recipe.clone(),
            model: spec,
            case_weights: self.case_weights.clone(),
        })
    }

    /// One resampling fold as tune 2.1.0 runs it: fit on `train` through
    /// `.fit_pre` / `.fit_model` (no sparse-data toggle), predict `test`, and evaluate
    /// `metrics`, weighted by the assessment rows' case weights when those are
    /// frequency weights (`.use_case_weights_with_yardstick()`).
    pub fn fit_fold(&self, train: &Table, test: &Table, metrics: &MetricSet) -> Result<Vec<MetricValue>> {
        Ok(self.fit_fold_full(train, test, metrics)?.metrics)
    }

    /// [`Workflow::fit_fold`] keeping the fitted workflow and the assessment
    /// predictions (what `tune::last_fit()` returns).
    pub fn fit_fold_full(&self, train: &Table, test: &Table, metrics: &MetricSet) -> Result<LastFit> {
        let fold = self.prepare_fold(train, test)?;
        let (model, predictions, metrics) = self.fit_prepared_parts(&fold, metrics)?;
        Ok(LastFit {
            workflow: FittedWorkflow {
                recipe: fold.recipe,
                model,
            },
            predictions,
            metrics,
        })
    }

    /// The preprocessing half of a fold fit (tune's `.fit_pre`): prep the recipe on
    /// `train` without the sparse-data toggle, bake both sets and read the case
    /// weights. The result serves every model candidate that shares this recipe and
    /// case-weight specification; [`Workflow::fit_prepared`] does the model half.
    pub fn prepare_fold(&self, train: &Table, test: &Table) -> Result<PreparedFold> {
        if test.nrows() == 0 {
            // rsample warns when a bootstrap or grouped resample has no assessment rows
            // and tune cannot score it; refuse instead of returning empty metrics
            return Err(Error::InvalidArgument(
                "the assessment set has no rows; choose more resamples or a different scheme".into(),
            ));
        }
        let recipe = self.recipe.prep_with_weights(train, self.case_weights_spec())?;
        let w_train = self.case_weights_of(train)?;
        let (x_train, y_train) = recipe.bake_matrix(train)?;
        let y_train = y_train.ok_or_else(|| {
            Error::SchemaMismatch(format!(
                "outcome `{}` is missing from the training data",
                recipe.outcome()
            ))
        })?;
        let (x_test, y_test) = recipe.bake_matrix(test)?;
        let y_test =
            y_test.ok_or_else(|| Error::SchemaMismatch("assessment rows lack the outcome".into()))?;
        let w_test = match self.case_weights_of(test)? {
            Some(w) if w.used_by_metrics() => Some(w),
            _ => None,
        };
        Ok(PreparedFold {
            recipe,
            x_train,
            y_train,
            w_train,
            x_test,
            y_test,
            w_test,
        })
    }

    /// The model half of a fold fit (tune's `.fit_model`, then predict and score):
    /// fit this workflow's model on the prepared analysis design, predict the
    /// assessment design and evaluate `metrics` (frequency case weights applied).
    /// `fold` must come from [`Workflow::prepare_fold`] on a workflow with the same
    /// recipe and case weights (the tuning candidates of one workflow qualify).
    pub fn fit_prepared(&self, fold: &PreparedFold, metrics: &MetricSet) -> Result<Vec<MetricValue>> {
        Ok(self.fit_prepared_parts(fold, metrics)?.2)
    }

    fn fit_prepared_parts(
        &self,
        fold: &PreparedFold,
        metrics: &MetricSet,
    ) -> Result<(Box<dyn FittedModel>, Prediction, Vec<MetricValue>)> {
        check_capabilities(self.model.as_ref(), &fold.y_train)?;
        let model = self
            .model
            .fit_maybe_weighted(&fold.x_train, &fold.y_train, fold.w_train.as_ref())?;
        let predictions = model.predict(&fold.x_test)?;
        let metrics = metrics.evaluate_weighted(
            &fold.y_test,
            &predictions,
            fold.w_test.as_ref().map(|w| w.values()),
        )?;
        Ok((model, predictions, metrics))
    }

    /// `tune::last_fit()`: fit on the analysis rows of `split` exactly as a resample
    /// is fitted (tune's `.fit_pre` / `.fit_model` path: no sparse-data toggle;
    /// frequency case weights reach the metrics), then predict and score the
    /// assessment rows. The fitted workflow is kept (`collect_predictions()` /
    /// `extract_workflow()` in tune).
    pub fn last_fit(&self, data: &Table, split: &Split, metrics: &MetricSet) -> Result<LastFit> {
        let train = data.select_rows(&split.train)?;
        let test = data.select_rows(&split.test)?;
        self.fit_fold_full(&train, &test, metrics)
    }

    /// Fit on each fold's analysis rows (prep included), predict the assessment rows,
    /// and evaluate `metrics`. With `Parallelism::threads(k)` folds run on a scoped
    /// Rayon pool; the result is identical to the serial run.
    pub fn fit_resamples(
        &self,
        data: &Table,
        folds: &Folds,
        metrics: &MetricSet,
        par: Parallelism,
    ) -> Result<ResampleResults> {
        if folds.n != data.nrows() {
            return Err(Error::InvalidArgument(format!(
                "folds were built for {} rows but data has {}",
                folds.n,
                data.nrows()
            )));
        }
        let jobs: Vec<usize> = (0..folds.len()).collect();
        let per_fold = parallel_map(jobs, par, |k| {
            let id = &folds.ids[k];
            let split = &folds.splits[k];
            let train = data.select_rows(&split.train)?;
            let test = data.select_rows(&split.test)?;
            // tune 2.1.0 fits folds through .fit_pre/.fit_model, which skip
            // workflows' sparsity toggle; mirror that so resampling metrics match R's
            let metrics = self
                .fit_fold(&train, &test, metrics)
                .map_err(|e| in_fold(e, id))?;
            Ok(FoldMetrics {
                id: id.clone(),
                metrics,
            })
        })?;
        Ok(ResampleResults::from_folds(per_fold))
    }
}

/// Read a case-weights column (`workflows:::fit.action_case_weights`: numeric, complete,
/// and valid for its kind; frequency weights must be whole numbers as
/// `hardhat::frequency_weights()` requires).
pub fn extract_case_weights(data: &Table, column: &str, kind: WeightKind) -> Result<CaseWeights> {
    let col = data
        .column(column)
        .ok_or_else(|| Error::SchemaMismatch(format!("case weights column `{column}` not found")))?;
    let values = col
        .as_numeric()
        .ok_or_else(|| Error::InvalidValue(format!("case weights column `{column}` must be numeric")))?;
    if col.n_missing() > 0 {
        return Err(Error::InvalidValue(format!(
            "case weights column `{column}` has {} missing values",
            col.n_missing()
        )));
    }
    CaseWeights::new(kind, values.to_vec())
}

/// Prefix an error raised inside a resampling fold with the fold id, so a failure in
/// one of many folds is attributable (review finding F1).
pub fn in_fold(e: Error, id: &str) -> Error {
    match e {
        Error::SchemaMismatch(m) => Error::SchemaMismatch(format!("{id}: {m}")),
        Error::InvalidValue(m) => Error::InvalidValue(format!("{id}: {m}")),
        Error::InvalidArgument(m) => Error::InvalidArgument(format!("{id}: {m}")),
        Error::Unsupported(m) => Error::Unsupported(format!("{id}: {m}")),
        Error::Backend { engine, message } => Error::Backend {
            engine,
            message: format!("{id}: {message}"),
        },
        Error::Serialization(m) => Error::Serialization(format!("{id}: {m}")),
        other => other,
    }
}

/// Reject a model/outcome pairing before any backend work.
pub fn check_capabilities(model: &dyn ModelSpec, y: &Outcome) -> Result<()> {
    let caps = model.capabilities();
    match y {
        Outcome::Numeric(_) if !caps.contains(&Capability::Regression) => Err(Error::Unsupported(format!(
            "`{}` does not support regression (numeric outcome)",
            model.name()
        ))),
        Outcome::Class { levels, .. }
            if levels.len() == 2 && !caps.contains(&Capability::BinaryClassification) =>
        {
            Err(Error::Unsupported(format!(
                "`{}` does not support binary classification",
                model.name()
            )))
        }
        Outcome::Class { levels, .. }
            if levels.len() > 2 && !caps.contains(&Capability::MulticlassClassification) =>
        {
            Err(Error::Unsupported(format!(
                "`{}` does not support {}-level classification",
                model.name(),
                levels.len()
            )))
        }
        Outcome::Class { levels, .. } if levels.len() < 2 => Err(Error::Unsupported(format!(
            "`{}` needs at least two outcome levels",
            model.name()
        ))),
        _ => Ok(()),
    }
}

/// Run `f` over `items`, serially or on a scoped Rayon pool of `k` threads. Output
/// order always matches input order, and errors are reported in input order.
pub fn parallel_map<T, R, F>(items: Vec<T>, par: Parallelism, f: F) -> Result<Vec<R>>
where
    T: Send,
    R: Send,
    F: Fn(T) -> Result<R> + Sync + Send,
{
    match par.threads {
        None => items.into_iter().map(f).collect(),
        Some(k) => {
            use rayon::prelude::*;
            let pool = rayon::ThreadPoolBuilder::new()
                .num_threads(k)
                .build()
                .map_err(|e| Error::InvalidArgument(format!("cannot build thread pool: {e}")))?;
            pool.install(|| items.into_par_iter().map(f).collect::<Vec<Result<R>>>())
                .into_iter()
                .collect()
        }
    }
}

/// A fitted workflow: prepped recipe plus fitted model.
#[derive(Debug)]
pub struct FittedWorkflow {
    recipe: FittedRecipe,
    model: Box<dyn FittedModel>,
}

impl FittedWorkflow {
    /// Bake `data` and predict.
    pub fn predict(&self, data: &Table) -> Result<Prediction> {
        let (x, _) = self.recipe.bake_matrix(data)?;
        self.model.predict(&x)
    }

    /// The prepped recipe.
    pub fn recipe(&self) -> &FittedRecipe {
        &self.recipe
    }

    /// The fitted model.
    pub fn model(&self) -> &dyn FittedModel {
        self.model.as_ref()
    }

    /// Versioned JSON envelope.
    pub fn to_json(&self) -> Result<serde_json::Value> {
        let env = Envelope {
            format_version: WORKFLOW_FORMAT_VERSION,
            recipe: self.recipe.to_json()?,
            model: self.model.state()?,
        };
        serde_json::to_value(env).map_err(|e| Error::Serialization(e.to_string()))
    }

    /// Rebuild from [`FittedWorkflow::to_json`].
    pub fn from_json(value: &serde_json::Value) -> Result<Self> {
        let env: Envelope =
            serde_json::from_value(value.clone()).map_err(|e| Error::Serialization(e.to_string()))?;
        if env.format_version != WORKFLOW_FORMAT_VERSION {
            return Err(Error::Serialization(format!(
                "fitted workflow format version {} is not supported (expected {WORKFLOW_FORMAT_VERSION})",
                env.format_version
            )));
        }
        Ok(FittedWorkflow {
            recipe: FittedRecipe::from_json(&env.recipe)?,
            model: fitted_model_from_state(&env.model)?,
        })
    }
}

#[derive(Serialize, Deserialize)]
struct Envelope {
    format_version: u32,
    recipe: serde_json::Value,
    model: serde_json::Value,
}

/// Metrics for one fold.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct FoldMetrics {
    /// Fold id.
    pub id: String,
    /// Metric values in metric-set order.
    pub metrics: Vec<MetricValue>,
}

/// `tune::collect_metrics()`-style summary of one metric.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct MetricSummary {
    /// Metric name.
    pub name: String,
    /// Mean over folds where the metric was defined.
    pub mean: Option<f64>,
    /// Standard error `sd / sqrt(n)` over defined folds (`None` if n < 2).
    pub std_err: Option<f64>,
    /// Number of folds where the metric was defined.
    pub n: usize,
}

/// Per-fold metrics plus their summary.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct ResampleResults {
    /// One entry per fold.
    pub per_fold: Vec<FoldMetrics>,
    /// One entry per metric.
    pub summary: Vec<MetricSummary>,
}

impl ResampleResults {
    /// Summarise per-fold metrics (mean and standard error over defined values, as
    /// `tune::collect_metrics()` reports).
    pub fn from_folds(per_fold: Vec<FoldMetrics>) -> Self {
        let names: Vec<String> = per_fold
            .first()
            .map(|f| f.metrics.iter().map(|m| m.name.clone()).collect())
            .unwrap_or_default();
        let summary = names
            .iter()
            .map(|name| {
                let vals: Vec<f64> = per_fold
                    .iter()
                    .flat_map(|f| {
                        f.metrics
                            .iter()
                            .filter(|m| &m.name == name)
                            .filter_map(|m| m.value)
                    })
                    .collect();
                let n = vals.len();
                let mean = (n > 0).then(|| vals.iter().sum::<f64>() / n as f64);
                let std_err = (n > 1).then(|| {
                    let m = mean.expect("n > 0");
                    let var = vals.iter().map(|v| (v - m) * (v - m)).sum::<f64>() / (n as f64 - 1.0);
                    (var / n as f64).sqrt()
                });
                MetricSummary {
                    name: name.clone(),
                    mean,
                    std_err,
                    n,
                }
            })
            .collect();
        ResampleResults { per_fold, summary }
    }

    /// Summary for one metric by name.
    pub fn metric(&self, name: &str) -> Option<&MetricSummary> {
        self.summary.iter().find(|m| m.name == name)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tidyrustmodels_core::{Column, Seed};
    use tidyrustmodels_metrics::Metric;
    use tidyrustmodels_models::{LinearRegSpec, LogisticRegSpec};
    use tidyrustmodels_recipes::{Selector, StepDummy, StepNormalize};
    use tidyrustmodels_resample::vfold_cv;

    fn data(n: usize) -> Table {
        let x: Vec<f64> = (0..n)
            .map(|i| (i as f64 * 0.37).sin() * 3.0 + i as f64 * 0.01)
            .collect();
        let g: Vec<&str> = (0..n).map(|i| if i % 3 == 0 { "a" } else { "b" }).collect();
        let y: Vec<f64> = x
            .iter()
            .enumerate()
            .map(|(i, v)| 2.0 * v + if i % 3 == 0 { 1.0 } else { 0.0 } + (i as f64 * 1.3).cos())
            .collect();
        let cls: Vec<&str> = y.iter().map(|v| if *v > 0.5 { "hi" } else { "lo" }).collect();
        Table::new(vec![
            ("y", Column::numeric(y).unwrap()),
            ("x", Column::numeric(x).unwrap()),
            ("g", Column::categorical(&g).unwrap()),
            ("cls", Column::categorical(&cls).unwrap()),
        ])
        .unwrap()
    }

    #[test]
    fn fit_predict_and_round_trip() {
        let mut t = data(40);
        t.remove("cls");
        let wf = Workflow::new(
            Recipe::new("y")
                .step(StepNormalize::new(Selector::AllNumericPredictors))
                .step(StepDummy::new(Selector::AllNominalPredictors)),
            LinearRegSpec::new(),
        );
        let fitted = wf.fit(&t).unwrap();
        let p = fitted.predict(&t).unwrap();
        // Round-trip through *text*, not just `Value`: float parsing fidelity depends on
        // serde_json's `float_roundtrip` feature, which the workspace enables.
        let text = serde_json::to_string(&fitted.to_json().unwrap()).unwrap();
        let back = FittedWorkflow::from_json(&serde_json::from_str(&text).unwrap()).unwrap();
        assert_eq!(p, back.predict(&t).unwrap());
        assert_eq!(fitted.model().feature_names(), &["x", "g_b"]);
    }

    #[test]
    fn capability_mismatch_is_rejected_before_fitting() {
        let mut t = data(20);
        t.remove("y");
        let wf = Workflow::new(
            Recipe::new("cls").step(StepDummy::new(Selector::columns(&["g"]))),
            LinearRegSpec::new(),
        );
        assert!(matches!(wf.fit(&t), Err(Error::Unsupported(_))));
    }

    #[test]
    fn serial_and_parallel_resampling_are_identical() {
        let mut t = data(60);
        t.remove("y");
        let wf = Workflow::new(
            Recipe::new("cls")
                .step(StepNormalize::new(Selector::AllNumericPredictors))
                .step(StepDummy::new(Selector::columns(&["g"]))),
            LogisticRegSpec::new(),
        );
        let folds = vfold_cv(60, 5, None, Seed(11)).unwrap();
        let metrics = MetricSet::new(vec![Metric::Accuracy, Metric::MnLogLoss]);
        let serial = wf
            .fit_resamples(&t, &folds, &metrics, Parallelism::SERIAL)
            .unwrap();
        let parallel = wf
            .fit_resamples(&t, &folds, &metrics, Parallelism::threads(4))
            .unwrap();
        assert_eq!(serial, parallel);
        assert_eq!(serial.per_fold.len(), 5);
        let acc = serial.metric("accuracy").unwrap();
        assert_eq!(acc.n, 5);
        assert!(acc.std_err.is_some());
    }

    /// Frequency weights mean repeated rows: a weighted workflow (recipe statistics,
    /// model and resampling metrics) must reproduce the unweighted workflow on the
    /// data with every row replicated by its weight.
    #[test]
    fn frequency_weights_reproduce_replicated_rows() {
        let mut t = data(30);
        t.remove("cls");
        let w: Vec<f64> = (0..30).map(|i| ((i * 7) % 4) as f64).collect(); // includes zeros
        let mut tw = t.clone();
        tw.insert("w", Column::numeric(w.clone()).unwrap()).unwrap();
        let rows: Vec<usize> = (0..30)
            .flat_map(|i| std::iter::repeat_n(i, w[i] as usize))
            .collect();
        let rep = t.select_rows(&rows).unwrap();
        let rec = Recipe::new("y")
            .step(StepNormalize::new(Selector::AllNumericPredictors))
            .step(StepDummy::new(Selector::AllNominalPredictors));
        let wf = Workflow::new(rec.clone(), LinearRegSpec::new()).case_weights("w", WeightKind::Frequency);
        let plain = Workflow::new(rec, LinearRegSpec::new());
        let a = wf.fit(&tw).unwrap();
        let b = plain.fit(&rep).unwrap();
        // the weights column is not a predictor and prediction data need not carry it
        assert_eq!(a.model().feature_names(), &["x", "g_b"]);
        let pa = match a.predict(&t).unwrap() {
            Prediction::Numeric(p) => p,
            _ => unreachable!(),
        };
        let pb = match b.predict(&t).unwrap() {
            Prediction::Numeric(p) => p,
            _ => unreachable!(),
        };
        for (u, v) in pa.iter().zip(&pb) {
            assert!((u - v).abs() < 1e-9, "{u} vs {v}");
        }
        assert!(a
            .model()
            .diagnostics()
            .note
            .unwrap()
            .contains("case weights (frequency)"));
        // weighted metrics on one fold equal the unweighted metrics on replicated rows
        let m = MetricSet::new(vec![Metric::Rmse, Metric::Rsq]);
        let fold = wf.fit_fold(&tw, &tw, &m).unwrap();
        let fold_rep = plain.fit_fold(&rep, &rep, &m).unwrap();
        for (u, v) in fold.iter().zip(&fold_rep) {
            assert!(
                (u.value.unwrap() - v.value.unwrap()).abs() < 1e-9,
                "{u:?} vs {v:?}"
            );
        }
    }

    #[test]
    fn importance_weights_skip_recipe_steps_and_unsupported_engines_fail_like_parsnip() {
        let mut t = data(30);
        t.remove("cls");
        t.insert(
            "w",
            Column::numeric((0..30).map(|i| 0.5 + (i % 3) as f64).collect()).unwrap(),
        )
        .unwrap();
        let rec = Recipe::new("y")
            .step(StepNormalize::new(Selector::columns(&["x"])))
            .step(StepDummy::new(Selector::columns(&["g"])));
        let imp = Workflow::new(rec.clone(), LinearRegSpec::new())
            .case_weights("w", WeightKind::Importance)
            .fit(&t)
            .unwrap();
        let mut no_w = t.clone();
        no_w.remove("w");
        let plain = Workflow::new(rec.clone(), LinearRegSpec::new())
            .fit(&no_w)
            .unwrap();
        // importance weights do not reach step_normalize: identical statistics (the
        // schemas differ only by the weights column's role)
        assert_eq!(
            imp.recipe().steps()[0].state().unwrap()["stats"],
            plain.recipe().steps()[0].state().unwrap()["stats"]
        );
        // ... but do reach the model: same features, different coefficients
        assert_eq!(imp.model().feature_names(), plain.model().feature_names());
        assert_ne!(
            imp.model().coefficients().unwrap(),
            plain.model().coefficients().unwrap()
        );
        // nnet multinom takes no weights (parsnip's message), nor do non-native engines
        let mut c = data(30);
        c.remove("y");
        c.insert("w", Column::numeric(vec![1.0; 30]).unwrap()).unwrap();
        let err = Workflow::new(
            Recipe::new("cls").step(StepDummy::new(Selector::columns(&["g"]))),
            tidyrustmodels_models::MultinomRegSpec::new(),
        )
        .case_weights("w", WeightKind::Frequency)
        .fit(&c)
        .unwrap_err();
        assert!(err
            .to_string()
            .contains("Case weights are not enabled by the underlying model implementation."));
        let err = Workflow::new(
            Recipe::new("y").step(StepDummy::new(Selector::columns(&["g"]))),
            LinearRegSpec::new().engine(tidyrustmodels_models::Engine::Smartcore),
        )
        .case_weights("w", WeightKind::Importance)
        .fit(&t)
        .unwrap_err();
        assert!(matches!(err, Error::Unsupported(_)), "{err}");
    }

    #[test]
    fn prep_happens_inside_each_fold() {
        // A column whose training mean differs per fold: the normalize step must not
        // see assessment rows. We detect leakage by comparing with a manual per-fold prep.
        let mut t = data(30);
        t.remove("cls");
        let rec = Recipe::new("y")
            .step(StepNormalize::new(Selector::columns(&["x"])))
            .step(StepDummy::new(Selector::columns(&["g"])));
        let wf = Workflow::new(rec.clone(), LinearRegSpec::new());
        let folds = vfold_cv(30, 3, None, Seed(5)).unwrap();
        let res = wf
            .fit_resamples(
                &t,
                &folds,
                &MetricSet::new(vec![Metric::Rmse]),
                Parallelism::SERIAL,
            )
            .unwrap();
        for (k, split) in folds.splits.iter().enumerate() {
            let train = t.select_rows(&split.train).unwrap();
            let test = t.select_rows(&split.test).unwrap();
            let f = wf.fit(&train).unwrap();
            let (xt, yt) = f.recipe().bake_matrix(&test).unwrap();
            let pred = f.model().predict(&xt).unwrap();
            let manual = Metric::Rmse.compute(&yt.unwrap(), &pred).unwrap();
            assert_eq!(res.per_fold[k].metrics[0].value, manual);
        }
    }
}
