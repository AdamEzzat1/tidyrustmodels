//! `multinom_reg`: multinomial logistic regression by maximum likelihood, optionally
//! with an L2 penalty on the slopes (ADR-0010).

use crate::alias::{drop_aliased, expand};
use crate::backends::{self, Engine};
use crate::diag::multinom_gradient_norm;
use crate::fitted::FittedMultinom;
use crate::linear::validate_shapes;
use serde::{Deserialize, Serialize};
use tidyrustmodels_core::{
    BackendInfo, Capability, DesignMatrix, Error, FittedModel, ModelSpec, Outcome, Result,
};

/// Multinomial logistic regression with the first outcome level as the baseline, the
/// parametrisation of `parsnip::multinom_reg()` with engine `nnet`. With `penalty = 0`
/// the objective is the multinomial negative log-likelihood; the `native` engine solves
/// it by Newton-Raphson to the deviance criterion `epsilon`, the L-BFGS engines to their
/// gradient tolerance. Two-level outcomes are accepted (the fit then equals
/// `logistic_reg` for the second level).
///
/// `penalty` is an unscaled L2 term `penalty/2 * ||beta||^2` on the slopes of every
/// level (intercepts unpenalised). It is **not** `nnet`'s `decay` (which also penalises
/// intercepts) nor glmnet's `lambda`; see COMPATIBILITY.md.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct MultinomRegSpec {
    engine: Engine,
    intercept: bool,
    penalty: f64,
    max_iterations: u64,
    gradient_tolerance: f64,
    epsilon: f64,
    newton_max_iterations: usize,
}

impl Default for MultinomRegSpec {
    fn default() -> Self {
        MultinomRegSpec {
            engine: Engine::default(),
            intercept: true,
            penalty: 0.0,
            max_iterations: 1000,
            gradient_tolerance: 1e-8,
            epsilon: 1e-10,
            newton_max_iterations: 100,
        }
    }
}

impl MultinomRegSpec {
    /// Unpenalised MLE on the default engine.
    pub fn new() -> Self {
        Self::default()
    }

    /// Choose the fitting backend.
    pub fn engine(mut self, engine: Engine) -> Self {
        self.engine = engine;
        self
    }

    /// Whether to fit intercepts (default `true`).
    pub fn intercept(mut self, intercept: bool) -> Self {
        self.intercept = intercept;
        self
    }

    /// Unscaled L2 penalty on the slopes (default `0`).
    pub fn penalty(mut self, penalty: f64) -> Self {
        self.penalty = penalty;
        self
    }

    /// L-BFGS iteration cap for the `linfa` engine (default 1000).
    pub fn max_iterations(mut self, n: u64) -> Self {
        self.max_iterations = n;
        self
    }

    /// L-BFGS gradient tolerance for the `linfa` engine (default `1e-8`).
    pub fn gradient_tolerance(mut self, tol: f64) -> Self {
        self.gradient_tolerance = tol;
        self
    }

    /// Relative deviance change at which the `native` Newton solver stops (default
    /// `1e-10`).
    pub fn epsilon(mut self, epsilon: f64) -> Self {
        self.epsilon = epsilon;
        self
    }

    /// Newton iteration cap for the `native` engine (default 100).
    pub fn newton_max_iterations(mut self, n: usize) -> Self {
        self.newton_max_iterations = n;
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
        if !self.epsilon.is_finite() || self.epsilon <= 0.0 {
            return Err(Error::InvalidArgument("epsilon must be finite and > 0".into()));
        }
        if self.max_iterations == 0 || self.newton_max_iterations == 0 {
            return Err(Error::InvalidArgument("iteration caps must be >= 1".into()));
        }
        Ok(())
    }
}

/// Options forwarded to backends.
#[derive(Clone, Copy, Debug)]
pub(crate) struct MultinomOptions {
    pub intercept: bool,
    pub penalty: f64,
    pub max_iterations: u64,
    pub gradient_tolerance: f64,
    pub epsilon: f64,
    pub newton_max_iterations: usize,
}

impl ModelSpec for MultinomRegSpec {
    fn name(&self) -> &'static str {
        "multinom_reg"
    }

    fn capabilities(&self) -> &'static [Capability] {
        &[
            Capability::BinaryClassification,
            Capability::MulticlassClassification,
            Capability::ClassProbabilities,
            Capability::Coefficients,
        ]
    }

    fn backend(&self) -> BackendInfo {
        backends::info(self.engine, "multinom_reg")
    }

    fn fit(&self, x: &DesignMatrix, y: &Outcome) -> Result<Box<dyn FittedModel>> {
        self.validate()?;
        let (codes, levels) = y
            .as_class()
            .ok_or_else(|| Error::Unsupported("multinom_reg requires a categorical outcome".into()))?;
        let k = levels.len();
        if k < 2 {
            return Err(Error::Unsupported(
                "multinom_reg needs at least two outcome levels".into(),
            ));
        }
        validate_shapes(x, codes.len(), self.intercept)?;
        let mut counts = vec![0usize; k];
        for c in codes {
            counts[*c as usize] += 1;
        }
        if let Some(i) = counts.iter().position(|c| *c == 0) {
            return Err(Error::InvalidValue(format!(
                "multinom_reg needs every outcome level present in the training data; `{}` has no rows",
                levels[i]
            )));
        }
        let opts = MultinomOptions {
            intercept: self.intercept,
            penalty: self.penalty,
            max_iterations: self.max_iterations,
            gradient_tolerance: self.gradient_tolerance,
            epsilon: self.epsilon,
            newton_max_iterations: self.newton_max_iterations,
        };
        let (xr, keep, aliased) = drop_aliased(x, self.intercept)?;
        let mut out = backends::fit_multinom(self.engine, &xr, codes, k, opts)?;
        out.coefs = out.coefs.iter().map(|c| expand(c, &keep, x.ncols())).collect();
        let mut all_aliased = aliased;
        all_aliased.append(&mut out.diagnostics.aliased);
        out.diagnostics.aliased = all_aliased;
        let gnorm = multinom_gradient_norm(x, codes, k, out.intercepts.as_deref(), &out.coefs, self.penalty);
        out.diagnostics.final_gradient_norm = Some(gnorm);
        let scale = (x.nrows() as f64).max(1.0);
        if out.diagnostics.converged.is_none() {
            out.diagnostics.converged = Some(gnorm <= self.gradient_tolerance * scale);
        }
        let fitted = FittedMultinom::new(
            out.intercepts,
            out.coefs,
            x.names().to_vec(),
            levels.to_vec(),
            self.backend(),
            out.diagnostics,
        )?;
        Ok(Box::new(fitted))
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
                "multinom_reg has no tunable parameter `{other}`"
            ))),
        }
    }

    fn clone_box(&self) -> Box<dyn ModelSpec> {
        Box::new(self.clone())
    }
}
