//! Workflows: a recipe plus a model specification, fitted together and evaluated
//! across resamples with the recipe re-prepped inside every fold.

#![forbid(unsafe_code)]
#![warn(missing_docs, rust_2018_idioms)]

use serde::{Deserialize, Serialize};
use tidyrustmodels_core::{
    Capability, Error, FittedModel, ModelSpec, Outcome, Parallelism, Prediction, Result, Table,
};
use tidyrustmodels_metrics::{MetricSet, MetricValue};
use tidyrustmodels_models::fitted_model_from_state;
use tidyrustmodels_recipes::{FittedRecipe, Recipe};
use tidyrustmodels_resample::Folds;

/// Serialization format version for fitted workflows.
pub const WORKFLOW_FORMAT_VERSION: u32 = 1;

/// An unfitted workflow.
#[derive(Clone, Debug)]
pub struct Workflow {
    recipe: Recipe,
    model: Box<dyn ModelSpec>,
}

impl Workflow {
    /// Combine a recipe and a model specification.
    pub fn new(recipe: Recipe, model: impl ModelSpec + 'static) -> Self {
        Workflow {
            recipe,
            model: Box::new(model),
        }
    }

    /// Combine with an already boxed specification (used by tuning).
    pub fn from_boxed(recipe: Recipe, model: Box<dyn ModelSpec>) -> Self {
        Workflow { recipe, model }
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
    /// (`ModelSpec::resolve_sparsity`); `fit_resamples` does not, like tune.
    pub fn fit(&self, training: &Table) -> Result<FittedWorkflow> {
        self.fit_with(training, true)
    }

    fn fit_with(&self, training: &Table, resolve_sparsity: bool) -> Result<FittedWorkflow> {
        let recipe = self.recipe.prep(training)?;
        let (x, y) = recipe.bake_matrix(training)?;
        let y = y.ok_or_else(|| {
            Error::SchemaMismatch(format!(
                "outcome `{}` is missing from the training data",
                recipe.outcome()
            ))
        })?;
        check_capabilities(self.model.as_ref(), &y)?;
        // workflows::toggle_sparsity: models whose engine takes sparse input decide
        // from the recipe's sparsity estimate (see ModelSpec::resolve_sparsity)
        let spec = if resolve_sparsity {
            let est = self.recipe.estimate_sparsity(training);
            self.model
                .resolve_sparsity(est, training.nrows())
                .unwrap_or_else(|| self.model.clone_box())
        } else {
            self.model.clone_box()
        };
        let model = spec.fit(&x, &y)?;
        Ok(FittedWorkflow { recipe, model })
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
            let fitted = self.fit_with(&train, false).map_err(|e| in_fold(e, id))?;
            let (x_test, y_test) = fitted.recipe.bake_matrix(&test).map_err(|e| in_fold(e, id))?;
            let y_test =
                y_test.ok_or_else(|| Error::SchemaMismatch("assessment rows lack the outcome".into()))?;
            let pred = fitted.model.predict(&x_test).map_err(|e| in_fold(e, id))?;
            Ok(FoldMetrics {
                id: id.clone(),
                metrics: metrics.evaluate(&y_test, &pred)?,
            })
        })?;
        Ok(ResampleResults::from_folds(per_fold))
    }
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
