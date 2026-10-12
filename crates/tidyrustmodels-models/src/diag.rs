//! Backend-independent convergence diagnostics for logistic regression.
//!
//! Backends do not reliably expose their termination status, so the framework
//! evaluates the objective gradient at the returned coefficients itself. The check is
//! O(n·p) and runs once per fit.

use crate::fitted::sigmoid;
use tidyrustmodels_core::DesignMatrix;

/// Infinity norm of the gradient of the (optionally L2-penalised, intercept
/// unpenalised) negative binomial log-likelihood, in *sum* form:
/// `g_j = sum_i (p_i - y_i) x_ij + penalty * beta_j`, `g_0 = sum_i (p_i - y_i)`.
///
/// `y` is 0/1 for the second level; `coefs` are the log-odds coefficients of that level.
pub fn logistic_gradient_norm(
    x: &DesignMatrix,
    y: &[u32],
    intercept: Option<f64>,
    coefs: &[f64],
    penalty: f64,
) -> f64 {
    let n = x.nrows();
    let mut resid = vec![intercept.unwrap_or(0.0); n];
    for (j, b) in coefs.iter().enumerate() {
        for (r, v) in resid.iter_mut().zip(x.column(j)) {
            *r += b * v;
        }
    }
    for (r, yi) in resid.iter_mut().zip(y) {
        *r = sigmoid(*r) - f64::from(*yi);
    }
    let mut norm: f64 = if intercept.is_some() {
        resid.iter().sum::<f64>().abs()
    } else {
        0.0
    };
    for (j, b) in coefs.iter().enumerate() {
        let g: f64 = resid.iter().zip(x.column(j)).map(|(r, v)| r * v).sum::<f64>() + penalty * b;
        norm = norm.max(g.abs());
    }
    norm
}

/// R's `glm` warning condition: any fitted probability within `10 * eps` of 0 or 1,
/// which usually indicates (quasi-)complete separation.
pub fn probabilities_saturated(x: &DesignMatrix, intercept: Option<f64>, coefs: &[f64]) -> bool {
    let eps = 10.0 * f64::EPSILON;
    let n = x.nrows();
    let mut eta = vec![intercept.unwrap_or(0.0); n];
    for (j, b) in coefs.iter().enumerate() {
        for (e, v) in eta.iter_mut().zip(x.column(j)) {
            *e += b * v;
        }
    }
    eta.iter().map(|z| sigmoid(*z)).any(|p| p < eps || p > 1.0 - eps)
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;

    #[test]
    fn gradient_is_zero_at_closed_form_mle() {
        // x in {0,1}; y|x=0 ~ 1 of 4, y|x=1 ~ 3 of 4 -> b0 = log(1/3), b1 = log(3) - b0.
        let x = DesignMatrix::from_columns(vec![("x".into(), vec![0., 0., 0., 0., 1., 1., 1., 1.])]).unwrap();
        let y = [0, 0, 0, 1, 1, 1, 1, 0];
        let b0 = (1.0f64 / 3.0).ln();
        let b1 = 3.0f64.ln() - b0;
        assert_abs_diff_eq!(
            logistic_gradient_norm(&x, &y, Some(b0), &[b1], 0.0),
            0.0,
            epsilon = 1e-14
        );
        assert!(logistic_gradient_norm(&x, &y, Some(0.0), &[0.0], 0.0) > 0.1);
        assert!(!probabilities_saturated(&x, Some(b0), &[b1]));
        assert!(probabilities_saturated(&x, Some(-60.0), &[120.0]));
    }
}
