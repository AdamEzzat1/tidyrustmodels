//! The `native` engine: pure-Rust solvers that reproduce R's algorithms rather than
//! merely its objectives (ADR-0009).
//!
//! * `linear_reg`: Householder QR with `dqrdc2` pivoting, `tol = 1e-7`, as `lm.fit`.
//! * `logistic_reg`: IRLS as `glm.fit`, weighted QR with `tol = 1e-11`.
//!
//! No external numerical dependency is involved; the code is small enough to read
//! against the R sources it cites.

pub mod irls;
pub mod multinom;
pub mod qr;

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
    let (buf, p) = design(x, intercept);
    let qr = qr::PivotedQr::decompose(buf, x.nrows(), p, 1e-7);
    let coef = qr.solve(y);
    let aliased = aliased_names(x, intercept, qr.aliased());
    let (b0, coefs) = split(coef, intercept);
    Ok(FitOutput {
        intercept: b0,
        coefs,
        diagnostics: FitDiagnostics {
            aliased,
            note: Some(format!(
                "pivoted Householder QR (dqrdc2, tol 1e-7), rank {}",
                qr.rank()
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
    let (buf, p) = design(x, opts.intercept);
    let fit = irls::fit(
        &buf,
        x.nrows(),
        p,
        y,
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
                "IRLS as glm.fit (epsilon {:e}, maxit {}){}",
                opts.epsilon,
                opts.maxit,
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
