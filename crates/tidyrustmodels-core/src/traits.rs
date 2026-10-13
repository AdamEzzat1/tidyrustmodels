//! Lifecycle traits: preprocessing steps and model specifications.

use crate::data::{Schema, Table};
use crate::error::{Error, Result};
use crate::matrix::DesignMatrix;
use crate::outcome::{Coefficients, FitDiagnostics, Outcome, Prediction};
use serde::{Deserialize, Serialize};
use std::fmt::Debug;

/// An unfitted preprocessing step. `prep` sees training rows only.
pub trait Step: Debug + Send + Sync {
    /// Stable identifier used in serialized recipes (e.g. `"normalize"`).
    fn name(&self) -> &'static str;

    /// Estimate the step's state from training data. Must not mutate `self`.
    fn prep(&self, training: &Table, schema: &Schema) -> Result<Box<dyn FittedStep>>;

    /// recipes' `.recipes_estimate_sparsity` contribution of a step that can emit
    /// sparse columns (`sparse = "auto"`): one `(n_cols, sparsity)` pair per selected
    /// column, where `n_cols` columns of that sparsity replace the column. Default:
    /// none (the step is dense).
    fn sparse_adjustments(&self, _training: &Table, _schema: &Schema) -> Vec<(usize, f64)> {
        Vec::new()
    }

    /// Clone into a boxed trait object.
    fn clone_box(&self) -> Box<dyn Step>;
}

impl Clone for Box<dyn Step> {
    fn clone(&self) -> Self {
        self.clone_box()
    }
}

/// A fitted preprocessing step. `bake` never refits.
pub trait FittedStep: Debug + Send + Sync {
    /// Same identifier as the unfitted step.
    fn name(&self) -> &'static str;

    /// Apply the fitted transformation to any table with a compatible schema.
    fn bake(&self, data: &Table) -> Result<Table>;

    /// Schema of the table `bake` produces.
    fn output_schema(&self) -> &Schema;

    /// Serializable fitted state for round-tripping. Built-in steps return a tagged
    /// JSON object; custom steps may return `Error::Unsupported`.
    fn state(&self) -> Result<serde_json::Value>;
}

/// A capability a model declares before fitting.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum Capability {
    /// Numeric outcomes.
    Regression,
    /// Two-level categorical outcomes.
    BinaryClassification,
    /// Categorical outcomes with three or more levels (ADR-0010).
    MulticlassClassification,
    /// Returns a full probability matrix for classification.
    ClassProbabilities,
    /// Exposes named linear coefficients.
    Coefficients,
}

/// Facts about the backend implementing a model.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct BackendInfo {
    /// Engine identifier (`linfa`, `smartcore`, `native`).
    pub engine: String,
    /// Version of the backend crate compiled in.
    pub crate_version: String,
    /// Solver description (e.g. `"L-BFGS (argmin), More-Thuente line search"`).
    pub solver: String,
}

impl BackendInfo {
    /// Backends are used for *fitting only*: every prediction and probability is
    /// computed by the framework from the fitted coefficients, so prediction
    /// semantics are identical across engines and serialized models do not depend on
    /// a backend crate.
    pub const PREDICTIONS_COMPUTED_BY_FRAMEWORK: bool = true;
}

/// An unfitted model specification.
pub trait ModelSpec: Debug + Send + Sync {
    /// Stable identifier (`"linear_reg"`, `"logistic_reg"`).
    fn name(&self) -> &'static str;

    /// Declared capabilities. Workflows check these before fitting.
    fn capabilities(&self) -> &'static [Capability];

    /// Which backend will do the fitting.
    fn backend(&self) -> BackendInfo;

    /// Fit on a design matrix and outcome. Must validate shapes and outcome kind
    /// before delegating to a backend.
    fn fit(&self, x: &DesignMatrix, y: &Outcome) -> Result<Box<dyn FittedModel>>;

    /// Names of parameters that `with_param` accepts (used by tuning).
    fn tunable(&self) -> &'static [&'static str] {
        &[]
    }

    /// Return a copy with one parameter changed. Default: nothing is tunable.
    fn with_param(&self, name: &str, _value: f64) -> Result<Box<dyn ModelSpec>> {
        Err(Error::Unsupported(format!(
            "`{}` has no tunable parameter `{name}`",
            self.name()
        )))
    }

    /// workflows' sparse-data hook: given the estimated sparsity of the baked design
    /// and the number of training rows (`workflows:::toggle_sparsity`), return the
    /// specification to fit instead, or `None` to fit `self` unchanged. Only models
    /// whose R engine accepts sparse matrices (`allow_sparse_x`) and whose behaviour
    /// depends on it override this.
    fn resolve_sparsity(&self, _estimated_sparsity: f64, _n_rows: usize) -> Option<Box<dyn ModelSpec>> {
        None
    }

    /// Clone into a boxed trait object.
    fn clone_box(&self) -> Box<dyn ModelSpec>;
}

impl Clone for Box<dyn ModelSpec> {
    fn clone(&self) -> Self {
        self.clone_box()
    }
}

/// A fitted model. Immutable; `predict` has no side effects.
pub trait FittedModel: Debug + Send + Sync {
    /// Predict for rows whose features match `feature_names()` in order.
    fn predict(&self, x: &DesignMatrix) -> Result<Prediction>;

    /// Feature names the model was fitted on.
    fn feature_names(&self) -> &[String];

    /// Linear coefficients, when the model exposes a single set of them.
    fn coefficients(&self) -> Option<Coefficients> {
        None
    }

    /// Per-level coefficient sets for multiclass linear models: one entry per
    /// non-baseline level `(level, coefficients)` in level order, relative to the first
    /// level, as `coef(nnet::multinom)` reports them (ADR-0010). `None` otherwise.
    fn class_coefficients(&self) -> Option<Vec<(String, Coefficients)>> {
        None
    }

    /// Solver diagnostics recorded at fit time.
    fn diagnostics(&self) -> FitDiagnostics {
        FitDiagnostics::default()
    }

    /// Backend that produced this fit.
    fn backend(&self) -> BackendInfo;

    /// Serializable fitted state (coefficients etc.), backend-independent.
    fn state(&self) -> Result<serde_json::Value>;
}

/// Verify that a design matrix matches a fitted feature list.
pub fn check_features(expected: &[String], x: &DesignMatrix) -> Result<()> {
    if expected != x.names() {
        return Err(Error::SchemaMismatch(format!(
            "model fitted on features {:?} but received {:?}",
            expected,
            x.names()
        )));
    }
    Ok(())
}
