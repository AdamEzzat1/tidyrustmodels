//! Multinomial logistic regression by Newton-Raphson with the exact Hessian.
//!
//! Parametrisation is R's `nnet::multinom` / `multinom_reg()`: the first outcome level is
//! the baseline with `eta = 0`, and each other level `c` has its own coefficient vector
//! `theta_c` so that `P(y = c | x) = exp(x theta_c) / (1 + sum_d exp(x theta_d))`.
//! Unlike `nnet` (BFGS on the same objective, stopped by `reltol`), this solver iterates
//! Newton steps with step halving until the relative change in deviance is below
//! `epsilon`, so it reaches the maximum-likelihood solution to round-off when the
//! data are not separable. The objective is the deviance `-2 log L` plus
//! `penalty * ||slopes||^2` (an unscaled L2 term, not `nnet`'s `decay`).
//!
//! Rank deficiency is handled as in `lm.fit`: a pivoted Householder QR of the design
//! (`dqrdc2`, tol 1e-7) decides which columns are aliased; they get coefficient 0 in
//! every level and are reported, which is where this port deliberately differs from
//! `nnet` (which keeps the unidentified parameters).

use super::qr::PivotedQr;

/// Solver settings.
#[derive(Clone, Copy, Debug)]
pub struct MultinomOptions {
    /// Whether column 0 of the design is the intercept (unpenalised).
    pub intercept: bool,
    /// Unscaled L2 penalty on the slopes (deviance scale: `penalty * sum(beta^2)`).
    pub penalty: f64,
    /// Relative deviance change below which the fit is declared converged.
    pub epsilon: f64,
    /// Maximum Newton iterations.
    pub maxit: usize,
}

/// Solver result.
#[derive(Clone, Debug)]
pub struct MultinomFit {
    /// `(k - 1) * p` coefficients, class-major: `coef[(c - 1) * p + j]` for level `c >= 1`.
    pub coef: Vec<f64>,
    /// Final objective: `-2 log L + penalty * ||slopes||^2`.
    pub deviance: f64,
    /// Newton iterations taken.
    pub iter: usize,
    /// Whether the deviance criterion was met within `maxit`.
    pub converged: bool,
    /// Design columns found aliased by the pivoted QR (0-based, including the intercept
    /// column when present).
    pub aliased: Vec<usize>,
    /// Warnings in the spirit of `glm.fit`.
    pub warnings: Vec<String>,
}

/// Solver failures.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum MultinomError {
    /// Step halving could not find a deviance decrease.
    CannotCorrectStepSize,
    /// The Newton system was singular even after regularisation.
    SingularHessian,
}

impl std::fmt::Display for MultinomError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            MultinomError::CannotCorrectStepSize => write!(f, "step halving failed to reduce the deviance"),
            MultinomError::SingularHessian => {
                write!(f, "the Hessian of the multinomial likelihood is singular")
            }
        }
    }
}

/// Class probabilities for every row, class-major (`out[c * n + i]`), from the
/// class-major coefficients over a column-major design `x` (`n × p`).
pub fn probabilities(x: &[f64], n: usize, p: usize, k: usize, theta: &[f64]) -> Vec<f64> {
    let mut eta = vec![0.0; k * n]; // class 0 stays 0
    for c in 1..k {
        let th = &theta[(c - 1) * p..c * p];
        let out = &mut eta[c * n..(c + 1) * n];
        for (j, b) in th.iter().enumerate() {
            if *b != 0.0 {
                for (e, v) in out.iter_mut().zip(&x[j * n..(j + 1) * n]) {
                    *e += b * v;
                }
            }
        }
    }
    let mut pi = eta;
    for i in 0..n {
        let mut m = 0.0f64;
        for c in 0..k {
            m = m.max(pi[c * n + i]);
        }
        let mut s = 0.0;
        for c in 0..k {
            let e = (pi[c * n + i] - m).exp();
            pi[c * n + i] = e;
            s += e;
        }
        for c in 0..k {
            pi[c * n + i] /= s;
        }
    }
    pi
}

fn objective(pi: &[f64], n: usize, y: &[u32], theta: &[f64], p: usize, opts: MultinomOptions) -> f64 {
    let mut nll = 0.0;
    for (i, yi) in y.iter().enumerate() {
        nll -= pi[*yi as usize * n + i].ln();
    }
    2.0 * nll + penalty_term(theta, p, opts)
}

fn penalty_term(theta: &[f64], p: usize, opts: MultinomOptions) -> f64 {
    if opts.penalty == 0.0 {
        return 0.0;
    }
    let first = usize::from(opts.intercept);
    let mut s = 0.0;
    for block in theta.chunks(p) {
        s += block[first..].iter().map(|b| b * b).sum::<f64>();
    }
    opts.penalty * s
}

/// Cholesky solve of the symmetric positive definite `a x = b` (`a` row-major `m × m`),
/// in place in `a`; returns `None` if a pivot is not positive.
fn cholesky_solve(a: &mut [f64], m: usize, b: &[f64]) -> Option<Vec<f64>> {
    for j in 0..m {
        let mut d = a[j * m + j];
        for k in 0..j {
            d -= a[j * m + k] * a[j * m + k];
        }
        if d.is_nan() || d <= 0.0 {
            return None;
        }
        let d = d.sqrt();
        a[j * m + j] = d;
        for i in j + 1..m {
            let mut s = a[i * m + j];
            for k in 0..j {
                s -= a[i * m + k] * a[j * m + k];
            }
            a[i * m + j] = s / d;
        }
    }
    // forward: L z = b
    let mut z = b.to_vec();
    for i in 0..m {
        for k in 0..i {
            z[i] -= a[i * m + k] * z[k];
        }
        z[i] /= a[i * m + i];
    }
    // backward: L^T x = z
    for i in (0..m).rev() {
        for k in i + 1..m {
            z[i] -= a[k * m + i] * z[k];
        }
        z[i] /= a[i * m + i];
    }
    Some(z)
}

/// Fit the multinomial model. `x` is column-major `n × p`, `y` holds codes `0..k`.
pub fn fit(
    x: &[f64],
    n: usize,
    p: usize,
    y: &[u32],
    k: usize,
    opts: MultinomOptions,
) -> Result<MultinomFit, MultinomError> {
    assert_eq!(x.len(), n * p);
    assert_eq!(y.len(), n);
    assert!(k >= 2);
    // Aliased columns from the same pivoted QR rule as lm.fit.
    let qr = PivotedQr::decompose(x.to_vec(), n, p, 1e-7);
    let mut aliased = qr.aliased().to_vec();
    aliased.sort_unstable();
    let keep: Vec<usize> = (0..p).filter(|j| !aliased.contains(j)).collect();
    let q = keep.len();
    let mut xr = Vec::with_capacity(n * q);
    for &j in &keep {
        xr.extend_from_slice(&x[j * n..(j + 1) * n]);
    }
    let first_slope = usize::from(opts.intercept && keep.first() == Some(&0));
    let ropts = MultinomOptions {
        intercept: first_slope == 1,
        ..opts
    };
    let m = (k - 1) * q;
    let mut theta = vec![0.0; m];
    let mut pi = probabilities(&xr, n, q, k, &theta);
    let mut dev = objective(&pi, n, y, &theta, q, ropts);
    let mut converged = false;
    let mut iter = 0;
    let mut warnings = Vec::new();
    while iter < opts.maxit {
        iter += 1;
        // Gradient of the objective / 2 (negative log-likelihood + penalty/2 ||b||^2).
        let mut g = vec![0.0; m];
        for c in 1..k {
            for (jj, _) in keep.iter().enumerate() {
                let col = &xr[jj * n..(jj + 1) * n];
                let mut s = 0.0;
                for i in 0..n {
                    let yi = if y[i] as usize == c { 1.0 } else { 0.0 };
                    s += col[i] * (pi[c * n + i] - yi);
                }
                let idx = (c - 1) * q + jj;
                if jj >= first_slope {
                    s += opts.penalty * theta[idx];
                }
                g[idx] = s;
            }
        }
        // Hessian, block (c, d) = X' diag(pi_c (delta_cd - pi_d)) X.
        let mut h = vec![0.0; m * m];
        let mut w = vec![0.0; n];
        for c in 1..k {
            for d in c..k {
                for i in 0..n {
                    let pc = pi[c * n + i];
                    let pd = pi[d * n + i];
                    w[i] = if c == d { pc * (1.0 - pc) } else { -pc * pd };
                }
                for a in 0..q {
                    let ca = &xr[a * n..(a + 1) * n];
                    for b in a..q {
                        let cb = &xr[b * n..(b + 1) * n];
                        let mut s = 0.0;
                        for i in 0..n {
                            s += w[i] * ca[i] * cb[i];
                        }
                        let (ra, rb) = ((c - 1) * q + a, (d - 1) * q + b);
                        h[ra * m + rb] = s;
                        h[rb * m + ra] = s;
                        if c != d {
                            let (ra2, rb2) = ((c - 1) * q + b, (d - 1) * q + a);
                            h[ra2 * m + rb2] = s;
                            h[rb2 * m + ra2] = s;
                        }
                    }
                }
            }
        }
        if opts.penalty > 0.0 {
            for c in 1..k {
                for jj in first_slope..q {
                    let idx = (c - 1) * q + jj;
                    h[idx * m + idx] += opts.penalty;
                }
            }
        }
        // Solve H delta = g, regularising the diagonal if the Hessian is numerically
        // singular (separable data make it so as the fit saturates).
        let trace: f64 = (0..m).map(|i| h[i * m + i]).sum::<f64>() / m as f64;
        let mut delta = None;
        let mut jitter = 0.0;
        for attempt in 0..6 {
            let mut hh = h.clone();
            if attempt > 0 {
                jitter = if jitter == 0.0 {
                    1e-10 * trace.max(f64::MIN_POSITIVE)
                } else {
                    jitter * 100.0
                };
                for i in 0..m {
                    hh[i * m + i] += jitter;
                }
            }
            if let Some(d) = cholesky_solve(&mut hh, m, &g) {
                delta = Some(d);
                break;
            }
        }
        let delta = delta.ok_or(MultinomError::SingularHessian)?;
        if jitter > 0.0 {
            warnings.push(format!(
                "Hessian regularised with jitter {jitter:.1e} at iteration {iter}"
            ));
        }
        // Step halving on the objective.
        let mut step = 1.0;
        let mut accepted = None;
        for _ in 0..30 {
            let cand: Vec<f64> = theta.iter().zip(&delta).map(|(t, d)| t - step * d).collect();
            let pc = probabilities(&xr, n, q, k, &cand);
            let dc = objective(&pc, n, y, &cand, q, ropts);
            if dc.is_finite() && dc <= dev + 1e-10 * dev.abs() {
                accepted = Some((cand, pc, dc));
                break;
            }
            step /= 2.0;
        }
        let (cand, pc, dc) = accepted.ok_or(MultinomError::CannotCorrectStepSize)?;
        let rel = (dev - dc).abs() / (dc.abs() + 0.1);
        theta = cand;
        pi = pc;
        dev = dc;
        if rel < opts.epsilon {
            converged = true;
            break;
        }
    }
    if !converged {
        warnings.push(format!("algorithm did not converge in {} iterations", opts.maxit));
    }
    let eps = 10.0 * f64::EPSILON;
    if pi.iter().any(|p| *p < eps || *p > 1.0 - eps) {
        warnings.push("fitted probabilities numerically 0 or 1 occurred".into());
    }
    // Expand to the full p columns per level.
    let mut coef = vec![0.0; (k - 1) * p];
    for c in 1..k {
        for (jj, &j) in keep.iter().enumerate() {
            coef[(c - 1) * p + j] = theta[(c - 1) * q + jj];
        }
    }
    Ok(MultinomFit {
        coef,
        deviance: dev,
        iter,
        converged,
        aliased,
        warnings,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;

    fn opts() -> MultinomOptions {
        MultinomOptions {
            intercept: true,
            penalty: 0.0,
            epsilon: 1e-12,
            maxit: 50,
        }
    }

    #[test]
    fn saturated_design_reproduces_group_proportions() {
        // Intercept + one dummy (group B vs A). Counts: A: 4/2/2 ; B: 1/3/4 over 3 levels.
        let y: Vec<u32> = vec![0, 0, 0, 0, 1, 1, 2, 2, 0, 1, 1, 1, 2, 2, 2, 2];
        let g: Vec<f64> = (0..16).map(|i| if i < 8 { 0.0 } else { 1.0 }).collect();
        let n = 16;
        let mut x = vec![1.0; n];
        x.extend(g);
        let f = fit(&x, n, 2, &y, 3, opts()).unwrap();
        assert!(f.converged);
        // group A: log(2/4) for level 1 and level 2
        assert_abs_diff_eq!(f.coef[0], (2.0f64 / 4.0).ln(), epsilon = 1e-9);
        assert_abs_diff_eq!(f.coef[2], (2.0f64 / 4.0).ln(), epsilon = 1e-9);
        // group B: log(3/1), log(4/1) relative to A's
        assert_abs_diff_eq!(f.coef[0] + f.coef[1], (3.0f64 / 1.0).ln(), epsilon = 1e-9);
        assert_abs_diff_eq!(f.coef[2] + f.coef[3], (4.0f64 / 1.0).ln(), epsilon = 1e-9);
        // deviance = -2 * sum n_gc log(n_gc / n_g)
        let expect = -2.0
            * ((4.0 * (4.0f64 / 8.0).ln() + 2.0 * (2.0f64 / 8.0).ln() * 2.0)
                + (1.0 * (1.0f64 / 8.0).ln() + 3.0 * (3.0f64 / 8.0).ln() + 4.0 * (4.0f64 / 8.0).ln()));
        assert_abs_diff_eq!(f.deviance, expect, epsilon = 1e-9);
        assert!(f.aliased.is_empty());
    }

    #[test]
    fn two_levels_match_binary_logistic_closed_form() {
        // x in {0, 1}: P(y=1|x=0) = 1/4, P(y=1|x=1) = 3/4
        let y = [0u32, 0, 0, 1, 1, 1, 1, 0];
        let mut x = vec![1.0; 8];
        x.extend([0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0]);
        let f = fit(&x, 8, 2, &y, 2, opts()).unwrap();
        let b0 = (1.0f64 / 3.0).ln();
        assert_abs_diff_eq!(f.coef[0], b0, epsilon = 1e-9);
        assert_abs_diff_eq!(f.coef[1], 3.0f64.ln() - b0, epsilon = 1e-9);
    }

    #[test]
    fn collinear_column_is_aliased_and_zero() {
        let y = [0u32, 1, 2, 0, 1, 2, 0, 1, 2, 0];
        let n = 10;
        let x1: Vec<f64> = (0..n).map(|i| i as f64 * 0.3 - 1.0).collect();
        let x2: Vec<f64> = x1.iter().map(|v| 2.0 * v).collect();
        let mut x = vec![1.0; n];
        x.extend(x1);
        x.extend(x2);
        let f = fit(&x, n, 3, &y, 3, opts()).unwrap();
        assert_eq!(f.aliased, vec![2]);
        assert_eq!(f.coef[2], 0.0);
        assert_eq!(f.coef[5], 0.0);
        assert!(f.converged);
    }

    #[test]
    fn probabilities_are_stable_for_large_eta() {
        let x = vec![1.0, 1.0, 1000.0, -1000.0];
        let pi = probabilities(&x, 2, 2, 2, &[0.0, 1.0]);
        assert!((pi[0] + pi[2] - 1.0).abs() < 1e-15);
        assert!(pi[2] == 1.0 && pi[3] < 1e-300);
    }
}
