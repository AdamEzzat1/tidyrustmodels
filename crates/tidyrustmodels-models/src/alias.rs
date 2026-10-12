//! Detection of aliased design-matrix columns before fitting.
//!
//! R's `lm`/`glm` use a pivoted QR and report `NA` for coefficients that cannot be
//! estimated; predictions then ignore those terms. Neither linfa nor smartcore pivots:
//! linfa's least squares fails outright on a singular system. The framework therefore
//! drops the two cases that arise in ordinary workflows, an identically zero column
//! (a dummy level absent from the analysis rows of a fold) and a constant column when an
//! intercept is fitted, records them in `FitDiagnostics::aliased`, and reports their
//! coefficient as 0. General collinearity is still left to the backend.

use tidyrustmodels_core::{DesignMatrix, Result};

/// Returns `(reduced matrix, kept column indices, aliased column names)`.
pub fn drop_aliased(x: &DesignMatrix, intercept: bool) -> Result<(DesignMatrix, Vec<usize>, Vec<String>)> {
    let mut keep = Vec::with_capacity(x.ncols());
    let mut aliased = Vec::new();
    for j in 0..x.ncols() {
        let col = x.column(j);
        let first = col.first().copied().unwrap_or(0.0);
        let constant = col.iter().all(|v| *v == first);
        let drop = constant && (intercept || first == 0.0);
        if drop {
            aliased.push(x.names()[j].clone());
        } else {
            keep.push(j);
        }
    }
    if aliased.is_empty() {
        return Ok((x.clone(), keep, aliased));
    }
    Ok((x.select_columns(&keep)?, keep, aliased))
}

/// Expand coefficients of the reduced fit back to the full column list (0 for aliased).
pub fn expand(coefs: &[f64], keep: &[usize], ncols: usize) -> Vec<f64> {
    let mut out = vec![0.0; ncols];
    for (k, c) in keep.iter().zip(coefs) {
        out[*k] = *c;
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn drops_zero_and_constant_columns() {
        let x = DesignMatrix::from_columns(vec![
            ("a".into(), vec![1.0, 2.0, 3.0]),
            ("zero".into(), vec![0.0, 0.0, 0.0]),
            ("const".into(), vec![5.0, 5.0, 5.0]),
        ])
        .unwrap();
        let (r, keep, aliased) = drop_aliased(&x, true).unwrap();
        assert_eq!(r.names(), &["a"]);
        assert_eq!(keep, vec![0]);
        assert_eq!(aliased, vec!["zero", "const"]);
        let (r, _, aliased) = drop_aliased(&x, false).unwrap();
        assert_eq!(r.names(), &["a", "const"]);
        assert_eq!(aliased, vec!["zero"]);
        assert_eq!(expand(&[2.0], &[0], 3), vec![2.0, 0.0, 0.0]);
    }
}
