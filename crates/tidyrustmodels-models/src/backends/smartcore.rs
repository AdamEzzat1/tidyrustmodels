//! Adapter for `smartcore` 0.6.16 using its native `DenseMatrix` (no ndarray).

use super::{relative_to_baseline, FitOutput, MultiFitOutput};
use crate::logistic::LogisticOptions;
use crate::multinom::MultinomOptions;
use smartcore::linalg::basic::arrays::Array;
use smartcore::linalg::basic::matrix::DenseMatrix;
use smartcore::linear::linear_regression::{LinearRegression, LinearRegressionParameters};
use smartcore::linear::logistic_regression::{LogisticRegression, LogisticRegressionParameters};
use tidyrustmodels_core::{DesignMatrix, Error, FitDiagnostics, Result};

const ENGINE: &str = "smartcore";

/// `DenseMatrix::new` accepts a column-major buffer directly, so this is one `Vec`
/// clone with no re-layout.
fn matrix(x: &DesignMatrix) -> Result<DenseMatrix<f64>> {
    DenseMatrix::new(x.nrows(), x.ncols(), x.data().to_vec(), true).map_err(|e| Error::backend(ENGINE, e))
}

/// OLS through `smartcore::linear::LinearRegression` (SVD by default).
///
/// smartcore always fits an intercept; `intercept = false` is unsupported on this
/// engine.
pub(crate) fn fit_linear(x: &DesignMatrix, y: &[f64], intercept: bool) -> Result<FitOutput> {
    if !intercept {
        return Err(Error::Unsupported(
            "engine `smartcore` always fits an intercept for linear_reg".into(),
        ));
    }
    let xm = matrix(x)?;
    let model = LinearRegression::fit(&xm, &y.to_vec(), LinearRegressionParameters::default())
        .map_err(|e| Error::backend(ENGINE, e))?;
    let coef = model.coefficients();
    let (r, c) = coef.shape();
    if r * c != x.ncols() {
        return Err(Error::Backend {
            engine: ENGINE.into(),
            message: format!("coefficient matrix is {r}x{c} for {} features", x.ncols()),
        });
    }
    let coefs: Vec<f64> = (0..x.ncols())
        .map(|j| {
            if r == 1 {
                *coef.get((0, j))
            } else {
                *coef.get((j, 0))
            }
        })
        .collect();
    Ok(FitOutput {
        intercept: Some(*model.intercept()),
        coefs,
        diagnostics: FitDiagnostics {
            note: Some("direct SVD least-squares solve".into()),
            ..Default::default()
        },
    })
}

/// Logistic regression through `smartcore::linear::LogisticRegression` (L-BFGS with
/// fixed internal settings: `g_atol = 1e-8`, `max_iter = 1000`).
///
/// smartcore always fits an intercept and ignores `max_iterations` /
/// `gradient_tolerance`; both facts are recorded in the diagnostics note.
pub(crate) fn fit_logistic(x: &DesignMatrix, y: &[u32], opts: LogisticOptions) -> Result<FitOutput> {
    if !opts.intercept {
        return Err(Error::Unsupported(
            "engine `smartcore` always fits an intercept for logistic_reg".into(),
        ));
    }
    let xm = matrix(x)?;
    let params = LogisticRegressionParameters::default().with_alpha(opts.penalty);
    let model = LogisticRegression::fit(&xm, &y.to_vec(), params).map_err(|e| Error::backend(ENGINE, e))?;
    let coef = model.coefficients();
    let (r, c) = coef.shape();
    if r * c != x.ncols() {
        return Err(Error::Backend {
            engine: ENGINE.into(),
            message: format!("coefficient matrix is {r}x{c} for {} features", x.ncols()),
        });
    }
    let coefs: Vec<f64> = (0..x.ncols())
        .map(|j| {
            if r == 1 {
                *coef.get((0, j))
            } else {
                *coef.get((j, 0))
            }
        })
        .collect();
    let intercept = *model.intercept().get((0, 0));
    Ok(FitOutput {
        intercept: Some(intercept),
        coefs,
        diagnostics: FitDiagnostics {
            note: Some(
                "smartcore L-BFGS uses fixed settings (g_atol=1e-8, max_iter=1000); \
                 max_iterations/gradient_tolerance are ignored on this engine"
                    .into(),
            ),
            ..Default::default()
        },
    })
}

/// Multinomial fit through smartcore's softmax model (`k x (p + 1)` weights),
/// converted to the baseline-relative parametrisation.
pub(crate) fn fit_multinom(
    x: &DesignMatrix,
    y: &[u32],
    k: usize,
    opts: MultinomOptions,
) -> Result<MultiFitOutput> {
    if !opts.intercept {
        return Err(Error::Unsupported(
            "engine `smartcore` always fits an intercept for multinom_reg".into(),
        ));
    }
    if k == 2 {
        // smartcore switches to its binary objective for two classes; reuse that path.
        let b = fit_logistic(
            x,
            y,
            LogisticOptions {
                intercept: true,
                penalty: opts.penalty,
                max_iterations: opts.max_iterations,
                gradient_tolerance: opts.gradient_tolerance,
                irls_epsilon: 1e-8,
                irls_max_iterations: 25,
            },
        )?;
        return Ok(MultiFitOutput {
            intercepts: Some(vec![b.intercept.unwrap_or(0.0)]),
            coefs: vec![b.coefs],
            diagnostics: b.diagnostics,
        });
    }
    let xm = matrix(x)?;
    let params = LogisticRegressionParameters::default().with_alpha(opts.penalty);
    let model = LogisticRegression::fit(&xm, &y.to_vec(), params).map_err(|e| Error::backend(ENGINE, e))?;
    let coef = model.coefficients();
    let (r, c) = coef.shape();
    if r != k || c != x.ncols() {
        return Err(Error::Backend {
            engine: ENGINE.into(),
            message: format!(
                "coefficient matrix is {r}x{c} for {k} classes and {} features",
                x.ncols()
            ),
        });
    }
    let classes = model.classes();
    let mut intercepts = vec![0.0; k];
    let mut coefs = vec![vec![0.0; x.ncols()]; k];
    for cl in 0..k {
        let row = classes
            .iter()
            .position(|v| *v as usize == cl)
            .ok_or_else(|| Error::Backend {
                engine: ENGINE.into(),
                message: format!("class {cl} missing from the fitted model"),
            })?;
        intercepts[cl] = *model.intercept().get((row, 0));
        coefs[cl] = (0..x.ncols()).map(|j| *coef.get((row, j))).collect();
    }
    let mut out = relative_to_baseline(&intercepts, &coefs, true);
    out.diagnostics.note = Some(
        "smartcore L-BFGS uses fixed settings (g_atol=1e-8, max_iter=1000); \
         max_iterations/gradient_tolerance are ignored on this engine"
            .into(),
    );
    Ok(out)
}
