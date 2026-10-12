//! `linear_reg`: ordinary least squares.

use crate::backends::{self, Engine};
use crate::fitted::FittedLinear;
use serde::{Deserialize, Serialize};
use tidyrustmodels_core::{
    BackendInfo, Capability, DesignMatrix, Error, FittedModel, ModelSpec, Outcome, Result,
};

/// Ordinary least squares, equivalent in objective to `parsnip::linear_reg()` with
/// engine `lm`.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct LinearRegSpec {
    engine: Engine,
    intercept: bool,
}

impl Default for LinearRegSpec {
    fn default() -> Self {
        LinearRegSpec {
            engine: Engine::default(),
            intercept: true,
        }
    }
}

impl LinearRegSpec {
    /// Default engine, with intercept.
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

    /// Selected engine.
    pub fn engine_kind(&self) -> Engine {
        self.engine
    }
}

impl ModelSpec for LinearRegSpec {
    fn name(&self) -> &'static str {
        "linear_reg"
    }

    fn capabilities(&self) -> &'static [Capability] {
        &[Capability::Regression, Capability::Coefficients]
    }

    fn backend(&self) -> BackendInfo {
        backends::info(self.engine, "linear_reg")
    }

    fn fit(&self, x: &DesignMatrix, y: &Outcome) -> Result<Box<dyn FittedModel>> {
        let y = y
            .as_numeric()
            .ok_or_else(|| Error::Unsupported("linear_reg requires a numeric outcome".into()))?;
        validate_shapes(x, y.len(), self.intercept)?;
        let out = backends::fit_linear(self.engine, x, y, self.intercept)?;
        Ok(Box::new(FittedLinear::new(
            out.intercept,
            out.coefs,
            x.names().to_vec(),
            self.backend(),
            out.diagnostics,
        )?))
    }

    fn clone_box(&self) -> Box<dyn ModelSpec> {
        Box::new(self.clone())
    }
}

/// Shared precondition checks run before any backend call (several backends panic
/// instead of returning errors on malformed shapes).
pub(crate) fn validate_shapes(x: &DesignMatrix, n_y: usize, intercept: bool) -> Result<()> {
    if x.nrows() != n_y {
        return Err(Error::InvalidArgument(format!(
            "design matrix has {} rows but outcome has {n_y}",
            x.nrows()
        )));
    }
    if x.nrows() == 0 {
        return Err(Error::InvalidArgument("cannot fit on zero rows".into()));
    }
    let n_params = x.ncols() + usize::from(intercept);
    if n_params == 0 {
        return Err(Error::InvalidArgument(
            "no features and no intercept: nothing to fit".into(),
        ));
    }
    if x.nrows() < n_params {
        return Err(Error::InvalidArgument(format!(
            "{} rows is fewer than the {n_params} parameters to estimate",
            x.nrows()
        )));
    }
    Ok(())
}
