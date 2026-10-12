//! Adapter for `linfa-linear` / `linfa-logistic` 0.8.1 (ndarray 0.16).

use super::FitOutput;
use crate::logistic::LogisticOptions;
use linfa::prelude::*;
use linfa_linear::LinearRegression;
use linfa_logistic::LogisticRegression;
use ndarray16::{Array1, Array2, ShapeBuilder};
use tidyrustmodels_core::{DesignMatrix, Error, FitDiagnostics, Result};

const ENGINE: &str = "linfa";

/// One O(n·p) copy into a Fortran-order ndarray 0.16 array (ADR-0002).
fn records(x: &DesignMatrix) -> Result<Array2<f64>> {
    Array2::from_shape_vec((x.nrows(), x.ncols()).f(), x.data().to_vec())
        .map_err(|e| Error::backend(ENGINE, e))
}

/// OLS through `linfa_linear::LinearRegression` (least squares, pure-Rust path).
pub(crate) fn fit_linear(x: &DesignMatrix, y: &[f64], intercept: bool) -> Result<FitOutput> {
    let ds = Dataset::new(records(x)?, Array1::from(y.to_vec()));
    let model = LinearRegression::new()
        .with_intercept(intercept)
        .fit(&ds)
        .map_err(|e| Error::backend(ENGINE, e))?;
    Ok(FitOutput {
        intercept: intercept.then(|| model.intercept()),
        coefs: model.params().to_vec(),
        diagnostics: FitDiagnostics {
            note: Some("direct least-squares solve".into()),
            ..Default::default()
        },
    })
}

/// Logistic regression through `linfa_logistic::LogisticRegression` (L-BFGS).
///
/// Targets are passed as `0`/`1`. linfa-logistic 0.8.1's documentation says the
/// positive class is the larger label, but `label_classes` actually picks the class
/// with the **larger count** (ties: the first class encountered). The adapter therefore
/// reads `labels().pos.class` after fitting and negates every coefficient when linfa
/// modelled class `0`, which is an exact transformation (logit P(y=0) = -logit P(y=1)).
/// The closed-form baseline test pins this behaviour.
pub(crate) fn fit_logistic(x: &DesignMatrix, y: &[u32], opts: LogisticOptions) -> Result<FitOutput> {
    let targets: Array1<usize> = y.iter().map(|&c| c as usize).collect();
    let ds = Dataset::new(records(x)?, targets);
    let model = LogisticRegression::default()
        .alpha(opts.penalty)
        .with_intercept(opts.intercept)
        .max_iterations(opts.max_iterations)
        .gradient_tolerance(opts.gradient_tolerance)
        .fit(&ds)
        .map_err(|e| Error::backend(ENGINE, e))?;
    let sign = if model.labels().pos.class == 1 { 1.0 } else { -1.0 };
    Ok(FitOutput {
        intercept: opts.intercept.then(|| sign * model.intercept()),
        coefs: model.params().iter().map(|b| sign * b).collect(),
        diagnostics: FitDiagnostics {
            note: Some("linfa does not expose L-BFGS termination status; see final_gradient_norm".into()),
            ..Default::default()
        },
    })
}
