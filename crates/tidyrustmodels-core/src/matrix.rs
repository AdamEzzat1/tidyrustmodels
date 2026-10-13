//! Dense column-major design matrix with feature names.

use crate::error::{Error, Result};
use serde::{Deserialize, Serialize};

/// A dense `f64` matrix stored column-major (Fortran order) with one name per column.
///
/// Column-major storage makes per-feature preprocessing contiguous and converts to R
/// and to `ndarray`'s `.f()` layout without a transpose. Row-major copies are
/// produced on demand for row-oriented consumers (NumPy C-order).
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct DesignMatrix {
    data: Vec<f64>,
    nrows: usize,
    ncols: usize,
    names: Vec<String>,
    /// Whether any source column was a sparse vector (ADR-0015): parsnip then hands
    /// the engine a sparse matrix (`maybe_sparse_matrix`), which xgboost reads with
    /// every exact zero missing. The storage here stays dense.
    #[serde(default)]
    sparse: bool,
}

impl DesignMatrix {
    /// Mark the matrix as sparse (see the field doc); `from_columns` and the other
    /// constructors build dense matrices.
    pub fn with_sparse(mut self, sparse: bool) -> Self {
        self.sparse = sparse;
        self
    }

    /// Whether the matrix stands for a sparse matrix in R (ADR-0015).
    pub fn is_sparse(&self) -> bool {
        self.sparse
    }

    /// Build from named columns of equal length. Rejects non-finite values.
    pub fn from_columns(columns: Vec<(String, Vec<f64>)>) -> Result<Self> {
        let ncols = columns.len();
        let nrows = columns.first().map(|c| c.1.len()).unwrap_or(0);
        let mut data = Vec::with_capacity(nrows * ncols);
        let mut names = Vec::with_capacity(ncols);
        for (name, col) in columns {
            if col.len() != nrows {
                return Err(Error::InvalidArgument(format!(
                    "column `{name}` has {} rows, expected {nrows}",
                    col.len()
                )));
            }
            if let Some(i) = col.iter().position(|v| !v.is_finite()) {
                return Err(Error::InvalidValue(format!(
                    "design matrix column `{name}` has a non-finite value at row {i}"
                )));
            }
            if names.contains(&name) {
                return Err(Error::InvalidArgument(format!("duplicate feature name `{name}`")));
            }
            data.extend_from_slice(&col);
            names.push(name);
        }
        Ok(DesignMatrix {
            data,
            nrows,
            ncols,
            names,
            sparse: false,
        })
    }

    /// Build from a row-major buffer (`data[i * ncols + j]`).
    pub fn from_row_major(data: &[f64], nrows: usize, ncols: usize, names: Vec<String>) -> Result<Self> {
        if data.len() != nrows * ncols {
            return Err(Error::InvalidArgument(format!(
                "row-major buffer has {} elements, expected {}",
                data.len(),
                nrows * ncols
            )));
        }
        if names.len() != ncols {
            return Err(Error::InvalidArgument(format!(
                "{} names supplied for {ncols} columns",
                names.len()
            )));
        }
        let mut cm = vec![0.0; nrows * ncols];
        for i in 0..nrows {
            for j in 0..ncols {
                cm[j * nrows + i] = data[i * ncols + j];
            }
        }
        if let Some(k) = cm.iter().position(|v| !v.is_finite()) {
            return Err(Error::InvalidValue(format!(
                "design matrix has a non-finite value at row {}, column {}",
                k % nrows,
                k / nrows
            )));
        }
        Ok(DesignMatrix {
            data: cm,
            nrows,
            ncols,
            names,
            sparse: false,
        })
    }

    /// Build from a column-major buffer, taking ownership (no copy).
    pub fn from_column_major(data: Vec<f64>, nrows: usize, ncols: usize, names: Vec<String>) -> Result<Self> {
        if data.len() != nrows * ncols {
            return Err(Error::InvalidArgument(format!(
                "column-major buffer has {} elements, expected {}",
                data.len(),
                nrows * ncols
            )));
        }
        if names.len() != ncols {
            return Err(Error::InvalidArgument(format!(
                "{} names supplied for {ncols} columns",
                names.len()
            )));
        }
        if let Some(k) = data.iter().position(|v| !v.is_finite()) {
            return Err(Error::InvalidValue(format!(
                "design matrix has a non-finite value at row {}, column {}",
                k % nrows,
                k / nrows
            )));
        }
        Ok(DesignMatrix {
            data,
            nrows,
            ncols,
            names,
            sparse: false,
        })
    }

    /// Rows.
    pub fn nrows(&self) -> usize {
        self.nrows
    }

    /// Columns.
    pub fn ncols(&self) -> usize {
        self.ncols
    }

    /// Feature names in column order.
    pub fn names(&self) -> &[String] {
        &self.names
    }

    /// The column-major buffer.
    pub fn data(&self) -> &[f64] {
        &self.data
    }

    /// Consume into the column-major buffer.
    pub fn into_data(self) -> Vec<f64> {
        self.data
    }

    /// One column as a slice.
    pub fn column(&self, j: usize) -> &[f64] {
        &self.data[j * self.nrows..(j + 1) * self.nrows]
    }

    /// Element `(i, j)`.
    pub fn get(&self, i: usize, j: usize) -> f64 {
        self.data[j * self.nrows + i]
    }

    /// A fresh row-major copy.
    pub fn to_row_major(&self) -> Vec<f64> {
        let mut out = vec![0.0; self.nrows * self.ncols];
        for j in 0..self.ncols {
            let col = self.column(j);
            for (i, v) in col.iter().enumerate() {
                out[i * self.ncols + j] = *v;
            }
        }
        out
    }

    /// Row `i` as an owned vector.
    pub fn row(&self, i: usize) -> Vec<f64> {
        (0..self.ncols).map(|j| self.get(i, j)).collect()
    }

    /// Select rows by index.
    pub fn select_rows(&self, rows: &[usize]) -> Result<DesignMatrix> {
        if let Some(bad) = rows.iter().find(|&&r| r >= self.nrows) {
            return Err(Error::InvalidArgument(format!(
                "row index {bad} out of range for {} rows",
                self.nrows
            )));
        }
        let mut data = Vec::with_capacity(rows.len() * self.ncols);
        for j in 0..self.ncols {
            let col = self.column(j);
            data.extend(rows.iter().map(|&r| col[r]));
        }
        Ok(DesignMatrix {
            data,
            nrows: rows.len(),
            ncols: self.ncols,
            names: self.names.clone(),
            sparse: false,
        })
    }

    /// Keep only the given columns (by index), in the given order.
    pub fn select_columns(&self, cols: &[usize]) -> Result<DesignMatrix> {
        if let Some(bad) = cols.iter().find(|&&c| c >= self.ncols) {
            return Err(Error::InvalidArgument(format!(
                "column index {bad} out of range for {} columns",
                self.ncols
            )));
        }
        let mut data = Vec::with_capacity(self.nrows * cols.len());
        for &c in cols {
            data.extend_from_slice(self.column(c));
        }
        Ok(DesignMatrix {
            data,
            nrows: self.nrows,
            ncols: cols.len(),
            names: cols.iter().map(|&c| self.names[c].clone()).collect(),
            sparse: false,
        })
    }

    /// Check that another matrix has the same feature names in the same order.
    pub fn check_same_features(&self, other: &DesignMatrix) -> Result<()> {
        if self.names != other.names {
            return Err(Error::SchemaMismatch(format!(
                "feature names differ: fitted on {:?}, received {:?}",
                self.names, other.names
            )));
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn row_major_round_trip() {
        let rm = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let m = DesignMatrix::from_row_major(&rm, 2, 3, vec!["a".into(), "b".into(), "c".into()]).unwrap();
        assert_eq!(m.column(0), &[1.0, 4.0]);
        assert_eq!(m.get(1, 2), 6.0);
        assert_eq!(m.to_row_major(), rm.to_vec());
        assert_eq!(m.row(1), vec![4.0, 5.0, 6.0]);
    }

    #[test]
    fn rejects_non_finite() {
        assert!(DesignMatrix::from_columns(vec![("a".into(), vec![1.0, f64::INFINITY])]).is_err());
    }

    #[test]
    fn select_rows_keeps_layout() {
        let m = DesignMatrix::from_columns(vec![
            ("a".into(), vec![1.0, 2.0, 3.0]),
            ("b".into(), vec![4.0, 5.0, 6.0]),
        ])
        .unwrap();
        let s = m.select_rows(&[2, 0]).unwrap();
        assert_eq!(s.column(0), &[3.0, 1.0]);
        assert_eq!(s.column(1), &[6.0, 4.0]);
    }
}
