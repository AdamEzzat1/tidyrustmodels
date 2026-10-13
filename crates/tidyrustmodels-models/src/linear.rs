//! `linear_reg`: ordinary least squares.

use crate::alias::{drop_aliased, expand};
use crate::backends::{self, Engine};
use crate::fitted::FittedLinear;
use serde::{Deserialize, Serialize};
use tidyrustmodels_core::{
    BackendInfo, Capability, CaseWeights, DesignMatrix, Error, FittedModel, ModelSpec, Outcome, Result,
};

/// Ordinary least squares, equivalent in objective to `parsnip::linear_reg()` with
/// engine `lm`. The default engine (`native`) is a port of `lm.fit`'s pivoted QR.
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
        if self.engine == Engine::Native {
            &[
                Capability::Regression,
                Capability::Coefficients,
                Capability::CaseWeights,
            ]
        } else {
            &[Capability::Regression, Capability::Coefficients]
        }
    }

    fn backend(&self) -> BackendInfo {
        backends::info(self.engine, "linear_reg")
    }

    fn fit(&self, x: &DesignMatrix, y: &Outcome) -> Result<Box<dyn FittedModel>> {
        self.fit_impl(x, y, None)
    }

    /// Weighted least squares as `lm(weights = )` (`lm.wfit`): native engine only.
    fn fit_weighted(
        &self,
        x: &DesignMatrix,
        y: &Outcome,
        weights: &CaseWeights,
    ) -> Result<Box<dyn FittedModel>> {
        self.fit_impl(x, y, Some(weights))
    }

    fn clone_box(&self) -> Box<dyn ModelSpec> {
        Box::new(self.clone())
    }
}

impl LinearRegSpec {
    fn fit_impl(
        &self,
        x: &DesignMatrix,
        y: &Outcome,
        weights: Option<&CaseWeights>,
    ) -> Result<Box<dyn FittedModel>> {
        let y = y
            .as_numeric()
            .ok_or_else(|| Error::Unsupported("linear_reg requires a numeric outcome".into()))?;
        validate_shapes(x, y.len(), self.intercept)?;
        check_weights(weights, y.len())?;
        let (xr, keep, aliased) = drop_aliased(x, self.intercept)?;
        let mut out = match weights {
            None => backends::fit_linear(self.engine, &xr, y, self.intercept)?,
            Some(w) => backends::fit_linear_weighted(self.engine, &xr, y, self.intercept, w.values())?,
        };
        if let Some(w) = weights {
            out.diagnostics.note = Some(format!(
                "{}; case weights ({})",
                out.diagnostics.note.unwrap_or_default(),
                w.kind().as_str()
            ));
        }
        // Constant/zero columns dropped before the fit come first, then any column the
        // backend itself judged aliased (native pivoted QR).
        let mut all_aliased = aliased;
        all_aliased.append(&mut out.diagnostics.aliased);
        out.diagnostics.aliased = all_aliased;
        Ok(Box::new(FittedLinear::new(
            out.intercept,
            expand(&out.coefs, &keep, x.ncols()),
            x.names().to_vec(),
            self.backend(),
            out.diagnostics,
        )?))
    }
}

/// Case weights must be one per row.
pub(crate) fn check_weights(weights: Option<&CaseWeights>, n: usize) -> Result<()> {
    match weights {
        Some(w) if w.len() != n => Err(Error::InvalidArgument(format!(
            "{} case weights for {n} rows",
            w.len()
        ))),
        _ => Ok(()),
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
