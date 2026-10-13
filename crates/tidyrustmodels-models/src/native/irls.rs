//! Iteratively reweighted least squares for binomial logistic regression, mirroring
//! R's `glm.fit` (stats 4.4.0) step for step:
//!
//! * start: `mu = (y + 0.5) / 2`, `eta = logit(mu)`;
//! * each iteration: `z = eta + (y - mu) / mu.eta`, `w = sqrt(mu.eta² / (mu (1 - mu)))`,
//!   weighted least squares through the pivoted QR with `tol = min(1e-7, eps / 1000)`;
//! * convergence when `|dev - devold| / (|dev| + 0.1) < epsilon` (default `1e-8`), at
//!   most `maxit` (default 25) iterations;
//! * step halving when the deviance becomes non-finite, as `glm.fit` does;
//! * the C-level `logit` link clamps `eta` to `±30` before `exp`, which is reproduced so
//!   saturated fits land on the same numbers.
//!
//! Prior weights (`glm.fit(weights = )`) are supported by `fit_weighted`: see its doc.
//!
//! An optional unscaled L2 penalty on the slopes is handled by row augmentation
//! (`sqrt(penalty)` rows), which `glm` itself does not offer; the convergence criterion
//! then uses the penalised deviance.

use super::qr::PivotedQr;

const THRESH: f64 = 30.0;

/// `C_logit_linkinv`.
fn linkinv(eta: f64) -> f64 {
    let tmp = if eta < -THRESH {
        f64::EPSILON
    } else if eta > THRESH {
        1.0 / f64::EPSILON
    } else {
        eta.exp()
    };
    tmp / (1.0 + tmp)
}

/// `C_logit_mu_eta`.
fn mu_eta(eta: f64) -> f64 {
    if !(-THRESH..=THRESH).contains(&eta) {
        f64::EPSILON
    } else {
        let e = eta.exp();
        let opexp = 1.0 + e;
        e / (opexp * opexp)
    }
}

fn y_log_y(y: f64, mu: f64) -> f64 {
    if y > 0.0 {
        y * (y / mu).ln()
    } else {
        0.0
    }
}

/// `binomial()$dev.resids` summed with prior weights (`C_binomial_dev_resids`):
/// `2 * wt * (y log(y / mu) + (1 - y) log((1 - y) / (1 - mu)))`.
fn deviance(y: &[f64], mu: &[f64], wt: &[f64]) -> f64 {
    y.iter()
        .zip(mu)
        .zip(wt)
        .map(|((&yi, &mi), &wi)| 2.0 * wi * (y_log_y(yi, mi) + y_log_y(1.0 - yi, 1.0 - mi)))
        .sum()
}

/// Solver settings.
#[derive(Clone, Copy, Debug)]
pub struct IrlsOptions {
    /// Fit an intercept (first design column of ones).
    pub intercept: bool,
    /// Unscaled L2 penalty on the slopes (`0` = plain MLE).
    pub penalty: f64,
    /// `glm.control(epsilon)`.
    pub epsilon: f64,
    /// `glm.control(maxit)`.
    pub maxit: usize,
}

/// What the solver reports.
#[derive(Clone, Debug)]
pub struct IrlsFit {
    /// Coefficients in design-column order (intercept first when fitted); aliased
    /// columns are `0`.
    pub coef: Vec<f64>,
    /// Final (penalised) deviance.
    pub deviance: f64,
    /// Iterations run (`glm$iter`).
    pub iter: usize,
    /// `glm$converged`.
    pub converged: bool,
    /// Whether step halving was needed (`glm$boundary`).
    pub boundary: bool,
    /// Design-column indices judged aliased by the final weighted QR.
    pub aliased: Vec<usize>,
    /// Warnings in `glm.fit`'s wording.
    pub warnings: Vec<String>,
}

/// Errors the solver can raise (mirroring `glm.fit`'s `stop()` calls).
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum IrlsError {
    /// `glm.fit` stops when the first iteration's deviance is non-finite.
    NoValidCoefficients,
    /// Step halving failed to produce a finite deviance.
    CannotCorrectStepSize,
}

impl std::fmt::Display for IrlsError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            IrlsError::NoValidCoefficients => {
                write!(
                    f,
                    "no valid set of coefficients has been found: please supply starting values"
                )
            }
            IrlsError::CannotCorrectStepSize => write!(f, "inner loop 1; cannot correct step size"),
        }
    }
}

/// Fit on a column-major `n × p` design (intercept column included by the caller when
/// `opts.intercept`) with 0/1 outcomes and unit prior weights.
pub fn fit(x: &[f64], n: usize, p: usize, y: &[u32], opts: IrlsOptions) -> Result<IrlsFit, IrlsError> {
    fit_weighted(x, n, p, y, None, opts)
}

/// [`fit`] with prior weights, as `glm.fit(weights = )`: the start is
/// `mustart = (weights * y + 0.5) / (weights + 1)`, the deviance is weighted, rows
/// with zero weight are left out of every weighted least-squares solve
/// (`good <- weights > 0`) while their `eta` and `mu` are still updated, and
/// non-integer `weights * y` raise glm's "non-integer #successes" warning.
pub fn fit_weighted(
    x: &[f64],
    n: usize,
    p: usize,
    y: &[u32],
    weights: Option<&[f64]>,
    opts: IrlsOptions,
) -> Result<IrlsFit, IrlsError> {
    assert_eq!(x.len(), n * p);
    assert_eq!(y.len(), n);
    let prior: Vec<f64> = match weights {
        Some(w) => {
            assert_eq!(w.len(), n);
            w.to_vec()
        }
        None => vec![1.0; n],
    };
    let yf: Vec<f64> = y.iter().map(|&v| f64::from(v)).collect();
    let tol = (1e-7f64).min(opts.epsilon / 1000.0);
    let penalised = opts.penalty > 0.0;
    let first_slope = usize::from(opts.intercept);
    // `good <- (weights > 0) & (mu.eta.val != 0)`; mu.eta is never 0 under the clamp
    let good: Vec<usize> = (0..n).filter(|&i| prior[i] > 0.0).collect();
    let n_good = good.len();
    let n_aug = if penalised {
        n_good + (p - first_slope)
    } else {
        n_good
    };
    let sqrt_pen = opts.penalty.sqrt();
    let mut warnings = Vec::new();
    if weights.is_some()
        && yf
            .iter()
            .zip(&prior)
            .any(|(yi, w)| (w * yi - (w * yi).round()).abs() > 0.001)
    {
        warnings.push("non-integer #successes in a binomial glm!".into());
    }

    let pen_term = |coef: &[f64]| -> f64 {
        if penalised {
            opts.penalty * coef[first_slope..].iter().map(|b| b * b).sum::<f64>()
        } else {
            0.0
        }
    };
    let eta_of = |coef: &[f64]| -> Vec<f64> {
        let mut eta = vec![0.0; n];
        for (j, &b) in coef.iter().enumerate() {
            if b != 0.0 {
                for (e, v) in eta.iter_mut().zip(&x[j * n..(j + 1) * n]) {
                    *e += b * v;
                }
            }
        }
        eta
    };

    let mut mu: Vec<f64> = yf
        .iter()
        .zip(&prior)
        .map(|(&v, &w)| (w * v + 0.5) / (w + 1.0))
        .collect();
    let mut eta: Vec<f64> = mu.iter().map(|&m| (m / (1.0 - m)).ln()).collect();
    let mut devold = deviance(&yf, &mu, &prior);
    let mut coef = vec![0.0; p];
    let mut coefold: Option<Vec<f64>> = None;
    let mut converged = false;
    let mut boundary = false;
    let mut aliased = Vec::new();
    let mut dev = devold;
    let mut iter = 0;

    for it in 1..=opts.maxit {
        iter = it;
        let me: Vec<f64> = eta.iter().map(|&e| mu_eta(e)).collect();
        let z: Vec<f64> = good.iter().map(|&i| eta[i] + (yf[i] - mu[i]) / me[i]).collect();
        // `w <- sqrt((weights[good] * mu.eta.val[good]^2) / variance(mu)[good])`
        let w: Vec<f64> = good
            .iter()
            .map(|&i| ((prior[i] * (me[i] * me[i])) / (mu[i] * (1.0 - mu[i]))).sqrt())
            .collect();
        let mut xw = Vec::with_capacity(n_aug * p);
        for j in 0..p {
            let col = &x[j * n..(j + 1) * n];
            xw.extend(good.iter().zip(&w).map(|(&i, wi)| col[i] * wi));
            if penalised {
                for s in first_slope..p {
                    xw.push(if s == j { sqrt_pen } else { 0.0 });
                }
            }
        }
        let mut zw: Vec<f64> = z.iter().zip(&w).map(|(zi, wi)| zi * wi).collect();
        if penalised {
            zw.resize(n_aug, 0.0);
        }
        let qr = PivotedQr::decompose(xw, n_aug, p, tol);
        let start = qr.solve(&zw);
        if start.iter().any(|b| !b.is_finite()) {
            warnings.push(format!("non-finite coefficients at iteration {it}"));
            converged = false;
            break;
        }
        aliased = qr.aliased().to_vec();
        let mut start = start;
        eta = eta_of(&start);
        mu = eta.iter().map(|&e| linkinv(e)).collect();
        dev = deviance(&yf, &mu, &prior) + pen_term(&start);
        boundary = false;
        if !dev.is_finite() {
            let Some(old) = coefold.as_ref() else {
                return Err(IrlsError::NoValidCoefficients);
            };
            warnings.push("step size truncated due to divergence".into());
            let mut ii = 1;
            while !dev.is_finite() {
                if ii > opts.maxit {
                    return Err(IrlsError::CannotCorrectStepSize);
                }
                ii += 1;
                for (s, o) in start.iter_mut().zip(old) {
                    *s = (*s + o) / 2.0;
                }
                eta = eta_of(&start);
                mu = eta.iter().map(|&e| linkinv(e)).collect();
                dev = deviance(&yf, &mu, &prior) + pen_term(&start);
            }
            boundary = true;
        }
        // validmu/valideta always hold for the clamped logit link.
        if (dev - devold).abs() / (dev.abs() + 0.1) < opts.epsilon {
            converged = true;
            coef = start;
            break;
        }
        devold = dev;
        coef = start.clone();
        coefold = Some(start);
    }
    if !converged {
        warnings.push("glm.fit: algorithm did not converge".into());
    }
    if boundary {
        warnings.push("glm.fit: algorithm stopped at boundary value".into());
    }
    Ok(IrlsFit {
        coef,
        deviance: dev,
        iter,
        converged,
        boundary,
        aliased,
        warnings,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    fn opts() -> IrlsOptions {
        IrlsOptions {
            intercept: true,
            penalty: 0.0,
            epsilon: 1e-8,
            maxit: 25,
        }
    }

    #[test]
    fn closed_form_mle() {
        // x in {0,1}: P(y=1|0) = 1/4, P(y=1|1) = 3/4 -> b0 = log(1/3), b1 = log 9.
        let n = 8;
        let mut x = vec![1.0; n];
        x.extend([0., 0., 0., 0., 1., 1., 1., 1.]);
        let y = [0, 0, 0, 1, 1, 1, 1, 0];
        let f = fit(&x, n, 2, &y, opts()).unwrap();
        assert!(f.converged);
        assert_relative_eq!(f.coef[0], (1.0f64 / 3.0).ln(), max_relative = 1e-9);
        assert_relative_eq!(f.coef[1], 9.0f64.ln(), max_relative = 1e-9);
        // deviance = -2 loglik = -2 * 8 * (1/4 log 1/4 + 3/4 log 3/4)
        let ll = 8.0 * (0.25 * 0.25f64.ln() + 0.75 * 0.75f64.ln());
        assert_relative_eq!(f.deviance, -2.0 * ll, max_relative = 1e-9);
        assert!(f.iter >= 3 && f.iter <= 6, "iter = {}", f.iter);
    }

    #[test]
    fn separable_data_warns_and_saturates_like_glm() {
        let n = 8;
        let mut x = vec![1.0; n];
        x.extend([0., 1., 2., 3., 10., 11., 12., 13.]);
        let y = [0, 0, 0, 0, 1, 1, 1, 1];
        let f = fit(&x, n, 2, &y, opts()).unwrap();
        // glm converges (deviance criterion) with huge coefficients; the saturation
        // warning is raised by the framework from the fitted probabilities.
        assert!(f.coef[1] > 1.0);
        assert!(f.deviance < 1e-6);
    }

    #[test]
    fn frequency_weights_equal_replicated_rows_and_zero_weights_drop_rows() {
        let n = 8;
        let mut x = vec![1.0; n];
        x.extend([0., 0., 0., 0., 1., 1., 1., 1.]);
        let y = [0, 0, 0, 1, 1, 1, 1, 0];
        let w = [2.0, 1.0, 0.0, 3.0, 1.0, 2.0, 1.0, 1.0];
        let f = fit_weighted(&x, n, 2, &y, Some(&w), opts()).unwrap();
        // replicate rows by their weights
        let mut xr = Vec::new();
        let mut xr1 = Vec::new();
        let mut yr = Vec::new();
        for i in 0..n {
            for _ in 0..w[i] as usize {
                xr.push(1.0);
                xr1.push(x[n + i]);
                yr.push(y[i]);
            }
        }
        let m = yr.len();
        xr.extend(xr1);
        let g = fit(&xr, m, 2, &yr, opts()).unwrap();
        // same MLE; the iterates differ (mustart depends on the weights), so agreement
        // is to the deviance convergence criterion, not to the last bit
        assert_relative_eq!(f.coef[0], g.coef[0], epsilon = 1e-7, max_relative = 1e-7);
        assert_relative_eq!(f.coef[1], g.coef[1], epsilon = 1e-7, max_relative = 1e-7);
        assert_relative_eq!(f.deviance, g.deviance, max_relative = 1e-9);
        assert!(f.warnings.iter().all(|w| !w.contains("non-integer")));
        let h = fit_weighted(&x, n, 2, &y, Some(&[0.5; 8]), opts()).unwrap();
        assert!(h.warnings.iter().any(|w| w.contains("non-integer #successes")));
    }

    #[test]
    fn penalty_shrinks_and_aliased_column_is_zero() {
        let n = 8;
        let mut x = vec![1.0; n];
        x.extend([0., 0., 0., 0., 1., 1., 1., 1.]);
        x.extend([0., 0., 0., 0., 2., 2., 2., 2.]); // 2 × the previous column
        let y = [0, 0, 0, 1, 1, 1, 1, 0];
        let f = fit(&x, n, 3, &y, opts()).unwrap();
        assert_eq!(f.aliased, vec![2]);
        assert_eq!(f.coef[2], 0.0);
        assert_relative_eq!(f.coef[1], 9.0f64.ln(), max_relative = 1e-9);
        let g = fit(
            &x[..2 * n],
            n,
            2,
            &y,
            IrlsOptions {
                penalty: 2.0,
                ..opts()
            },
        )
        .unwrap();
        assert!(g.converged && g.coef[1].abs() < 9.0f64.ln());
    }
}
