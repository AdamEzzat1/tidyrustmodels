//! `logistic_reg`: binary logistic regression by maximum likelihood, optionally with an
//! L2 penalty on the slopes.

use crate::backends::{self, Engine};
use crate::diag::{logistic_gradient_norm, probabilities_saturated};
use crate::fitted::FittedLogistic;
use crate::linear::validate_shapes;
use serde::{Deserialize, Serialize};
use tidyrustmodels_core::{
    BackendInfo, Capability, DesignMatrix, Error, FittedModel, ModelSpec, Outcome, Result,
};

/// Binary logistic regression. With `penalty = 0` the objective is the binomial
/// negative log-likelihood, the same objective as `parsnip::logistic_reg()` with engine
/// `glm`; the optimizer differs (L-BFGS here, IRLS in R), so agreement is bounded by
/// `gradient_tolerance` and is measured, not assumed.
///
/// `penalty` is an unscaled L2 term `penalty/2 * ||beta||^2` on the slopes (intercept
/// unpenalised). It is **not** glmnet's `lambda`, which is scaled by `1/n` and mixes
/// L1/L2; see COMPATIBILITY.md.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct LogisticRegSpec {
    engine: Engine,
    intercept: bool,
    penalty: f64,
    max_iterations: u64,
    gradient_tolerance: f64,
}

impl Default for LogisticRegSpec {
    fn default() -> Self {
        LogisticRegSpec {
            engine: Engine::default(),
            intercept: true,
            penalty: 0.0,
            max_iterations: 1000,
            gradient_tolerance: 1e-8,
        }
    }
}

impl LogisticRegSpec {
    /// Unpenalised MLE on the default engine.
    pub fn new() -> Self {
        Self::default()
    }

    /// Choose the fitting backend.
    pub fn engine(mut self, engine: Engine) -> Self {
        self.engine = engine;
        self
    }

    /// Whether to fit an intercept (default `true`).
    pub fn intercept(mut self, intercept: bool) -> Self {
        self.intercept = intercept;
        self
    }

    /// Unscaled L2 penalty on the slopes (default `0`).
    pub fn penalty(mut self, penalty: f64) -> Self {
        self.penalty = penalty;
        self
    }

    /// Optimizer iteration cap (default 1000).
    pub fn max_iterations(mut self, n: u64) -> Self {
        self.max_iterations = n;
        self
    }

    /// Optimizer gradient tolerance passed to the backend (default `1e-8`). The
    /// framework re-checks the gradient after fitting (see `FitDiagnostics`).
    pub fn gradient_tolerance(mut self, tol: f64) -> Self {
        self.gradient_tolerance = tol;
        self
    }

    /// Current penalty.
    pub fn penalty_value(&self) -> f64 {
        self.penalty
    }

    /// Selected engine.
    pub fn engine_kind(&self) -> Engine {
        self.engine
    }

    fn validate(&self) -> Result<()> {
        if !self.penalty.is_finite() || self.penalty < 0.0 {
            return Err(Error::InvalidArgument(format!(
                "penalty must be finite and >= 0, got {}",
                self.penalty
            )));
        }
        if !self.gradient_tolerance.is_finite() || self.gradient_tolerance <= 0.0 {
            return Err(Error::InvalidArgument(
                "gradient_tolerance must be finite and > 0".into(),
            ));
        }
        if self.max_iterations == 0 {
            return Err(Error::InvalidArgument("max_iterations must be >= 1".into()));
        }
        Ok(())
    }
}

/// Options forwarded to backends.
#[derive(Clone, Copy, Debug)]
pub(crate) struct LogisticOptions {
    pub intercept: bool,
    pub penalty: f64,
    pub max_iterations: u64,
    pub gradient_tolerance: f64,
}

impl ModelSpec for LogisticRegSpec {
    fn name(&self) -> &'static str {
        "logistic_reg"
    }

    fn capabilities(&self) -> &'static [Capability] {
        &[
            Capability::BinaryClassification,
            Capability::ClassProbabilities,
            Capability::Coefficients,
        ]
    }

    fn backend(&self) -> BackendInfo {
        backends::info(self.engine, "logistic_reg")
    }

    fn fit(&self, x: &DesignMatrix, y: &Outcome) -> Result<Box<dyn FittedModel>> {
        self.validate()?;
        let (codes, levels) = y
            .as_class()
            .ok_or_else(|| Error::Unsupported("logistic_reg requires a categorical outcome".into()))?;
        if levels.len() != 2 {
            return Err(Error::Unsupported(format!(
                "logistic_reg is binary; outcome has {} levels",
                levels.len()
            )));
        }
        validate_shapes(x, codes.len(), self.intercept)?;
        let n1 = codes.iter().filter(|c| **c == 1).count();
        if n1 == 0 || n1 == codes.len() {
            return Err(Error::InvalidValue(
                "logistic_reg needs both outcome levels present in the training data".into(),
            ));
        }
        let opts = LogisticOptions {
            intercept: self.intercept,
            penalty: self.penalty,
            max_iterations: self.max_iterations,
            gradient_tolerance: self.gradient_tolerance,
        };
        let mut out = backends::fit_logistic(self.engine, x, codes, opts)?;
        let gnorm = logistic_gradient_norm(x, codes, out.intercept, &out.coefs, self.penalty);
        out.diagnostics.final_gradient_norm = Some(gnorm);
        // Scale-aware convergence verdict: gradient entries are sums over n rows.
        let scale = (x.nrows() as f64).max(1.0);
        out.diagnostics.converged = Some(gnorm <= self.gradient_tolerance * scale);
        if probabilities_saturated(x, out.intercept, &out.coefs) {
            out.diagnostics
                .warnings
                .push("fitted probabilities numerically 0 or 1 occurred (possible separation)".into());
        }
        Ok(Box::new(FittedLogistic::new(
            out.intercept,
            out.coefs,
            x.names().to_vec(),
            levels.to_vec(),
            self.backend(),
            out.diagnostics,
        )?))
    }

    fn tunable(&self) -> &'static [&'static str] {
        &["penalty"]
    }

    fn with_param(&self, name: &str, value: f64) -> Result<Box<dyn ModelSpec>> {
        match name {
            "penalty" => {
                let s = self.clone().penalty(value);
                s.validate()?;
                Ok(Box::new(s))
            }
            other => Err(Error::Unsupported(format!(
                "logistic_reg has no tunable parameter `{other}`"
            ))),
        }
    }

    fn clone_box(&self) -> Box<dyn ModelSpec> {
        Box::new(self.clone())
    }
}
