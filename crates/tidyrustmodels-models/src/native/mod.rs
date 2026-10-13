//! The `native` engine: pure-Rust solvers that reproduce R's algorithms rather than
//! merely its objectives (ADR-0009).
//!
//! * `linear_reg`: Householder QR with `dqrdc2` pivoting, `tol = 1e-7`, as `lm.fit`.
//! * `logistic_reg`: IRLS as `glm.fit`, weighted QR with `tol = 1e-11`.
//!
//! No external numerical dependency is involved; the code is small enough to read
//! against the R sources it cites.

pub mod forest;
pub mod irls;
pub mod multinom;
pub mod qr;
pub mod rpart;
pub mod xgb;

use crate::backends::{FitOutput, MultiFitOutput};
use crate::multinom::MultinomOptions;
use tidyrustmodels_core::{DesignMatrix, Error, FitDiagnostics, Result};

const ENGINE: &str = "native";

/// `[1 | X]` (or `X`) as a column-major buffer.
fn design(x: &DesignMatrix, intercept: bool) -> (Vec<f64>, usize) {
    let n = x.nrows();
    let p = x.ncols() + usize::from(intercept);
    let mut buf = Vec::with_capacity(n * p);
    if intercept {
        buf.resize(n, 1.0);
    }
    buf.extend_from_slice(x.data());
    (buf, p)
}

/// Split a design-order coefficient vector into intercept and slopes.
fn split(coef: Vec<f64>, intercept: bool) -> (Option<f64>, Vec<f64>) {
    if intercept {
        let mut it = coef.into_iter();
        let b0 = it.next();
        (b0, it.collect())
    } else {
        (None, coef)
    }
}

/// Names of aliased design columns, mapped back to feature names.
fn aliased_names(x: &DesignMatrix, intercept: bool, aliased: &[usize]) -> Vec<String> {
    let off = usize::from(intercept);
    aliased
        .iter()
        .map(|&j| {
            if j >= off {
                x.names()[j - off].clone()
            } else {
                "(Intercept)".into()
            }
        })
        .collect()
}

/// OLS through the pivoted QR (`lm.fit` semantics, including which collinear column is
/// aliased).
pub(crate) fn fit_linear(x: &DesignMatrix, y: &[f64], intercept: bool) -> Result<FitOutput> {
    fit_linear_weighted(x, y, intercept, None)
}

/// Weighted least squares as `lm.wfit`: rows with zero weight are dropped, the rest
/// (design and response) are scaled by `sqrt(w)` before the same pivoted QR.
pub(crate) fn fit_linear_weighted(
    x: &DesignMatrix,
    y: &[f64],
    intercept: bool,
    weights: Option<&[f64]>,
) -> Result<FitOutput> {
    let (buf, p) = design(x, intercept);
    let n = x.nrows();
    let (buf, yw, n_ok) = match weights {
        None => (buf, y.to_vec(), n),
        Some(w) => {
            let ok: Vec<usize> = (0..n).filter(|&i| w[i] > 0.0).collect();
            if ok.is_empty() {
                return Err(Error::InvalidValue("all case weights are zero".into()));
            }
            let mut b = Vec::with_capacity(ok.len() * p);
            for j in 0..p {
                b.extend(ok.iter().map(|&i| buf[j * n + i] * w[i].sqrt()));
            }
            let yw = ok.iter().map(|&i| y[i] * w[i].sqrt()).collect();
            (b, yw, ok.len())
        }
    };
    let qr = qr::PivotedQr::decompose(buf, n_ok, p, 1e-7);
    let coef = qr.solve(&yw);
    let aliased = aliased_names(x, intercept, qr.aliased());
    let (b0, coefs) = split(coef, intercept);
    let dropped = n - n_ok;
    Ok(FitOutput {
        intercept: b0,
        coefs,
        diagnostics: FitDiagnostics {
            aliased,
            note: Some(format!(
                "pivoted Householder QR (dqrdc2, tol 1e-7), rank {}{}",
                qr.rank(),
                if weights.is_some() {
                    format!("; weighted as lm.wfit ({dropped} zero-weight rows dropped)")
                } else {
                    String::new()
                }
            )),
            ..Default::default()
        },
    })
}

/// Settings for the IRLS solver, set from `LogisticRegSpec`.
#[derive(Clone, Copy, Debug)]
pub(crate) struct NativeLogisticOptions {
    pub intercept: bool,
    pub penalty: f64,
    pub epsilon: f64,
    pub maxit: usize,
}

/// Logistic regression through IRLS (`glm.fit` semantics).
pub(crate) fn fit_logistic(x: &DesignMatrix, y: &[u32], opts: NativeLogisticOptions) -> Result<FitOutput> {
    fit_logistic_weighted(x, y, opts, None)
}

/// IRLS with prior weights (`glm.fit(weights = )`), see `irls::fit_weighted`.
pub(crate) fn fit_logistic_weighted(
    x: &DesignMatrix,
    y: &[u32],
    opts: NativeLogisticOptions,
    weights: Option<&[f64]>,
) -> Result<FitOutput> {
    let (buf, p) = design(x, opts.intercept);
    let fit = irls::fit_weighted(
        &buf,
        x.nrows(),
        p,
        y,
        weights,
        irls::IrlsOptions {
            intercept: opts.intercept,
            penalty: opts.penalty,
            epsilon: opts.epsilon,
            maxit: opts.maxit,
        },
    )
    .map_err(|e| Error::Backend {
        engine: ENGINE.into(),
        message: e.to_string(),
    })?;
    let aliased = aliased_names(x, opts.intercept, &fit.aliased);
    let (b0, coefs) = split(fit.coef, opts.intercept);
    Ok(FitOutput {
        intercept: b0,
        coefs,
        diagnostics: FitDiagnostics {
            converged: Some(fit.converged),
            iterations: Some(fit.iter as u64),
            deviance: Some(fit.deviance),
            warnings: fit.warnings,
            aliased,
            note: Some(format!(
                "IRLS as glm.fit (epsilon {:e}, maxit {}){}{}",
                opts.epsilon,
                opts.maxit,
                if weights.is_some() { "; prior weights" } else { "" },
                if fit.boundary { "; stopped at boundary" } else { "" }
            )),
            ..Default::default()
        },
    })
}

/// Multinomial regression by Newton-Raphson (see `multinom`).
pub(crate) fn fit_multinom(
    x: &DesignMatrix,
    y: &[u32],
    k: usize,
    opts: MultinomOptions,
) -> Result<MultiFitOutput> {
    let (buf, p) = design(x, opts.intercept);
    let fit = multinom::fit(
        &buf,
        x.nrows(),
        p,
        y,
        k,
        multinom::MultinomOptions {
            intercept: opts.intercept,
            penalty: opts.penalty,
            epsilon: opts.epsilon,
            maxit: opts.newton_max_iterations,
        },
    )
    .map_err(|e| Error::Backend {
        engine: ENGINE.into(),
        message: e.to_string(),
    })?;
    let aliased = aliased_names(x, opts.intercept, &fit.aliased);
    let mut intercepts = opts.intercept.then(Vec::new);
    let mut coefs = Vec::with_capacity(k - 1);
    for c in 0..k - 1 {
        let (b0, slopes) = split(fit.coef[c * p..(c + 1) * p].to_vec(), opts.intercept);
        if let (Some(v), Some(b)) = (intercepts.as_mut(), b0) {
            v.push(b);
        }
        coefs.push(slopes);
    }
    Ok(MultiFitOutput {
        intercepts,
        coefs,
        diagnostics: FitDiagnostics {
            converged: Some(fit.converged),
            iterations: Some(fit.iter as u64),
            deviance: Some(fit.deviance),
            warnings: fit.warnings,
            aliased,
            note: Some("Newton-Raphson, exact Hessian, deviance criterion".into()),
            ..Default::default()
        },
    })
}
