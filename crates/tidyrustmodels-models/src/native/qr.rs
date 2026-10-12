//! Householder QR with LINPACK `dqrdc2` *limited* column pivoting.
//!
//! This is the decomposition behind R's `lm.fit` (`tol = 1e-7`) and `glm.fit`
//! (`tol = min(1e-7, epsilon / 1000)`), ported from `src/appl/dqrdc2.f` and the parts
//! of `dqrsl.f` / `dqrls.f` that the solve needs. The pivoting rule is not "largest
//! norm first": columns are processed in order and a column is moved to the end only
//! when its remaining norm has fallen below `tol` times its original norm. That rule is
//! what makes R report `NA` for the *later* of two collinear columns, and reproducing it
//! is the point of this module.
//!
//! Storage is column-major, like the rest of the framework and like Fortran.

/// A pivoted QR factorisation of an `n × p` matrix.
#[derive(Clone, Debug)]
pub struct PivotedQr {
    /// Column-major `n × p`: `R` on and above the diagonal, Householder vectors below.
    qr: Vec<f64>,
    /// Leading element of each Householder vector (LINPACK `qraux`).
    qraux: Vec<f64>,
    /// `pivot[j]` is the original index of the column now at position `j`.
    pivot: Vec<usize>,
    /// Number of columns judged linearly independent.
    rank: usize,
    n: usize,
    p: usize,
}

fn norm(v: &[f64]) -> f64 {
    // LINPACK dnrm2 uses a scaled sum; a plain sum of squares in f64 differs at the
    // ulp level only, which the differential suite measures rather than assumes.
    v.iter().map(|x| x * x).sum::<f64>().sqrt()
}

impl PivotedQr {
    /// Factorise a column-major `n × p` matrix (consumed) with tolerance `tol`.
    pub fn decompose(mut x: Vec<f64>, n: usize, p: usize, tol: f64) -> PivotedQr {
        assert_eq!(x.len(), n * p, "buffer is not n × p");
        let col = |j: usize| j * n;
        let mut qraux = vec![0.0; p];
        let mut work1 = vec![0.0; p];
        let mut work2 = vec![0.0; p];
        let mut pivot: Vec<usize> = (0..p).collect();
        for j in 0..p {
            let nrm = norm(&x[col(j)..col(j) + n]);
            qraux[j] = nrm;
            work1[j] = nrm;
            work2[j] = if nrm == 0.0 { 1.0 } else { nrm };
        }
        let lup = n.min(p);
        // Fortran `k` (1-based sentinel): one more than the number of columns kept so far.
        let mut k = p + 1;
        for l in 0..lup {
            // Cycle negligible columns to the end until a usable one sits at position l.
            while l + 1 < k && qraux[l] < work2[l] * tol {
                let saved: Vec<f64> = x[col(l)..col(l) + n].to_vec();
                for j in l + 1..p {
                    x.copy_within(col(j)..col(j) + n, col(j - 1));
                }
                x[col(p - 1)..col(p - 1) + n].copy_from_slice(&saved);
                let (pi, t, tt, ttt) = (pivot[l], qraux[l], work1[l], work2[l]);
                for j in l + 1..p {
                    pivot[j - 1] = pivot[j];
                    qraux[j - 1] = qraux[j];
                    work1[j - 1] = work1[j];
                    work2[j - 1] = work2[j];
                }
                pivot[p - 1] = pi;
                qraux[p - 1] = t;
                work1[p - 1] = tt;
                work2[p - 1] = ttt;
                k -= 1;
            }
            if l + 1 == n {
                // Last row: nothing below the diagonal to annihilate.
                continue;
            }
            let mut nrmxl = norm(&x[col(l) + l..col(l) + n]);
            if nrmxl == 0.0 {
                continue;
            }
            if x[col(l) + l] != 0.0 {
                nrmxl = nrmxl.copysign(x[col(l) + l]);
            }
            for i in l..n {
                x[col(l) + i] /= nrmxl;
            }
            x[col(l) + l] += 1.0;
            for j in l + 1..p {
                let dot: f64 = (l..n).map(|i| x[col(l) + i] * x[col(j) + i]).sum();
                let t = -dot / x[col(l) + l];
                for i in l..n {
                    x[col(j) + i] += t * x[col(l) + i];
                }
                if qraux[j] != 0.0 {
                    let ratio = x[col(j) + l].abs() / qraux[j];
                    let tt = (1.0 - ratio * ratio).max(0.0);
                    if tt.abs() < 1e-6 {
                        // Doug Bates' 1999 criterion: recompute the norm exactly.
                        qraux[j] = norm(&x[col(j) + l + 1..col(j) + n]);
                        work1[j] = qraux[j];
                    } else {
                        qraux[j] *= tt.sqrt();
                    }
                }
            }
            qraux[l] = x[col(l) + l];
            x[col(l) + l] = -nrmxl;
        }
        let rank = (k - 1).min(n);
        PivotedQr {
            qr: x,
            qraux,
            pivot,
            rank,
            n,
            p,
        }
    }

    /// Rank determined by the pivoting rule.
    pub fn rank(&self) -> usize {
        self.rank
    }

    /// Original indices of the columns judged aliased (positions `rank..p`).
    pub fn aliased(&self) -> &[usize] {
        &self.pivot[self.rank..]
    }

    /// Column permutation (`pivot[j]` = original index of column `j`).
    pub fn pivot(&self) -> &[usize] {
        &self.pivot
    }

    /// `Qᵀ y` using the first `rank` reflections (LINPACK `dqrsl`, as `dqrls` calls it).
    pub fn qty(&self, y: &[f64]) -> Vec<f64> {
        assert_eq!(y.len(), self.n);
        let n = self.n;
        let mut qty = y.to_vec();
        let ju = self.rank.min(n.saturating_sub(1));
        for j in 0..ju {
            if self.qraux[j] == 0.0 {
                continue;
            }
            let c = j * n;
            // Householder vector: qraux[j] at row j, stored entries below.
            let v = &self.qr[c + j + 1..c + n];
            let dot = self.qraux[j] * qty[j] + v.iter().zip(&qty[j + 1..]).map(|(a, b)| a * b).sum::<f64>();
            let t = -dot / self.qraux[j];
            qty[j] += t * self.qraux[j];
            for (q, a) in qty[j + 1..].iter_mut().zip(v) {
                *q += t * a;
            }
        }
        qty
    }

    /// Least-squares coefficients in the **original** column order; aliased columns get
    /// `0`, exactly as `dqrls` leaves the unused components (`lm.fit` then reports them
    /// as `NA`).
    pub fn solve(&self, y: &[f64]) -> Vec<f64> {
        let n = self.n;
        let k = self.rank;
        let mut b: Vec<f64> = self.qty(y)[..k].to_vec();
        for j in (0..k).rev() {
            let diag = self.qr[j * n + j];
            b[j] /= diag;
            let bj = b[j];
            for (i, bi) in b.iter_mut().enumerate().take(j) {
                *bi -= self.qr[j * n + i] * bj;
            }
        }
        let mut out = vec![0.0; self.p];
        for (j, &bj) in b.iter().enumerate() {
            out[self.pivot[j]] = bj;
        }
        out
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    fn colmajor(cols: &[&[f64]]) -> (Vec<f64>, usize, usize) {
        let n = cols[0].len();
        let mut v = Vec::with_capacity(n * cols.len());
        for c in cols {
            v.extend_from_slice(c);
        }
        (v, n, cols.len())
    }

    #[test]
    fn full_rank_matches_exact_rational_ols() {
        // reference/independent/exact_ols.py: beta = -3445/1683, 1300/1683, 1895/1683
        let ones = [1.0; 12];
        let x1 = [1., 2., 3., 4., 5., 6., 7., 8., 9., 10., 11., 12.];
        let x2 = [5., 3., 8., 1., 9., 2., 7., 4., 6., 0., 10., 11.];
        let y = [3., 4., 9., 2., 13., 5., 12., 8., 11., 6., 17., 20.];
        let (v, n, p) = colmajor(&[&ones, &x1, &x2]);
        let qr = PivotedQr::decompose(v, n, p, 1e-7);
        assert_eq!(qr.rank(), 3);
        let b = qr.solve(&y);
        assert_relative_eq!(b[0], -3445.0 / 1683.0, max_relative = 1e-13);
        assert_relative_eq!(b[1], 1300.0 / 1683.0, max_relative = 1e-13);
        assert_relative_eq!(b[2], 1895.0 / 1683.0, max_relative = 1e-13);
    }

    #[test]
    fn later_collinear_column_is_aliased_like_r() {
        // R: lm(y ~ a + b + c) with b = 2a reports NA for `b`, keeps a and c.
        // coef from R 4.4.0: (Intercept) 1, a 0.5, b NA, c -1  (y = 1 + 0.5 a - c exactly)
        let ones = [1.0; 6];
        let a = [1., 2., 3., 4., 5., 6.];
        let b: Vec<f64> = a.iter().map(|v| 2.0 * v).collect();
        let c = [0.3, -1.2, 0.8, 2.2, -0.5, 1.1];
        let y: Vec<f64> = a.iter().zip(&c).map(|(x, z)| 1.0 + 0.5 * x - z).collect();
        let (v, n, p) = colmajor(&[&ones, &a, &b, &c]);
        let qr = PivotedQr::decompose(v, n, p, 1e-7);
        assert_eq!(qr.rank(), 3);
        assert_eq!(qr.aliased(), &[2]);
        let coef = qr.solve(&y);
        assert_relative_eq!(coef[0], 1.0, epsilon = 1e-12);
        assert_relative_eq!(coef[1], 0.5, epsilon = 1e-12);
        assert_eq!(coef[2], 0.0);
        assert_relative_eq!(coef[3], -1.0, epsilon = 1e-12);
        // With the collinear pair reversed, the *later* one (now `a`) is the aliased one.
        let (v, n, p) = colmajor(&[&ones, &b, &a, &c]);
        let qr = PivotedQr::decompose(v, n, p, 1e-7);
        assert_eq!(qr.aliased(), &[2]);
        let coef = qr.solve(&y);
        assert_relative_eq!(coef[1], 0.25, epsilon = 1e-12);
    }

    #[test]
    fn zero_column_and_square_system() {
        let ones = [1.0; 3];
        let z = [0.0; 3];
        let a = [1., 2., 4.];
        let y = [2., 3., 5.];
        let (v, n, p) = colmajor(&[&ones, &z, &a]);
        let qr = PivotedQr::decompose(v, n, p, 1e-7);
        assert_eq!(qr.rank(), 2);
        assert_eq!(qr.aliased(), &[1]);
        let coef = qr.solve(&y);
        assert_relative_eq!(coef[0], 1.0, epsilon = 1e-12);
        assert_relative_eq!(coef[2], 1.0, epsilon = 1e-12);
        // n == p: exact interpolation, including the last-row branch.
        let (v, n, p) = colmajor(&[&[1., 1.], &[0., 1.]]);
        let qr = PivotedQr::decompose(v, n, p, 1e-7);
        let coef = qr.solve(&[3.0, 5.0]);
        assert_relative_eq!(coef[0], 3.0, epsilon = 1e-12);
        assert_relative_eq!(coef[1], 2.0, epsilon = 1e-12);
    }
}
