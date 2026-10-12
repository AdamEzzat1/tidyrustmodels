//! Outcome, prediction, and fitted-model summary types.

use crate::data::Column;
use crate::error::{Error, Result};
use crate::matrix::DesignMatrix;
use serde::{Deserialize, Serialize};

/// The target variable of a modelling task.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub enum Outcome {
    /// Regression target.
    Numeric(Vec<f64>),
    /// Classification target. `levels[0]` is the event level for binary tasks.
    Class {
        /// Code per row into `levels`.
        codes: Vec<u32>,
        /// Level labels.
        levels: Vec<String>,
    },
}

impl Outcome {
    /// Build from a column; missing values are rejected.
    pub fn from_column(col: &Column) -> Result<Self> {
        if col.n_missing() > 0 {
            return Err(Error::InvalidValue(format!(
                "outcome has {} missing values; drop or impute them before modelling",
                col.n_missing()
            )));
        }
        Ok(match col {
            Column::Numeric { values, .. } => {
                if let Some(i) = values.iter().position(|v| !v.is_finite()) {
                    return Err(Error::InvalidValue(format!("outcome is non-finite at row {i}")));
                }
                Outcome::Numeric(values.clone())
            }
            Column::Categorical { codes, levels, .. } => Outcome::Class {
                codes: codes.clone(),
                levels: levels.clone(),
            },
        })
    }

    /// Number of rows.
    pub fn len(&self) -> usize {
        match self {
            Outcome::Numeric(v) => v.len(),
            Outcome::Class { codes, .. } => codes.len(),
        }
    }

    /// Whether there are zero rows.
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Select rows by index.
    pub fn select_rows(&self, rows: &[usize]) -> Result<Outcome> {
        let n = self.len();
        if let Some(bad) = rows.iter().find(|&&r| r >= n) {
            return Err(Error::InvalidArgument(format!(
                "row index {bad} out of range for {n} rows"
            )));
        }
        Ok(match self {
            Outcome::Numeric(v) => Outcome::Numeric(rows.iter().map(|&r| v[r]).collect()),
            Outcome::Class { codes, levels } => Outcome::Class {
                codes: rows.iter().map(|&r| codes[r]).collect(),
                levels: levels.clone(),
            },
        })
    }

    /// Numeric values, if regression.
    pub fn as_numeric(&self) -> Option<&[f64]> {
        match self {
            Outcome::Numeric(v) => Some(v),
            Outcome::Class { .. } => None,
        }
    }

    /// Codes and levels, if classification.
    pub fn as_class(&self) -> Option<(&[u32], &[String])> {
        match self {
            Outcome::Class { codes, levels } => Some((codes, levels)),
            Outcome::Numeric(_) => None,
        }
    }
}

/// Model output for a set of rows.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub enum Prediction {
    /// Regression predictions.
    Numeric(Vec<f64>),
    /// Hard class predictions plus, when the model supports it, a probability matrix
    /// with one column per level in `levels` order.
    Class {
        /// Predicted code per row.
        codes: Vec<u32>,
        /// Level labels (same order as training).
        levels: Vec<String>,
        /// Probability matrix `nrows × levels.len()`; `None` if unsupported.
        probabilities: Option<DesignMatrix>,
    },
}

impl Prediction {
    /// Number of rows.
    pub fn len(&self) -> usize {
        match self {
            Prediction::Numeric(v) => v.len(),
            Prediction::Class { codes, .. } => codes.len(),
        }
    }

    /// Whether there are zero rows.
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Probability of the event level (`levels[0]`) per row, if available.
    pub fn event_probability(&self) -> Option<&[f64]> {
        match self {
            Prediction::Class {
                probabilities: Some(p),
                ..
            } => Some(p.column(0)),
            _ => None,
        }
    }
}

/// Named linear coefficients of a fitted model.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Coefficients {
    /// Intercept, if the model has one.
    pub intercept: Option<f64>,
    /// `(feature name, coefficient)` in design-matrix column order.
    pub terms: Vec<(String, f64)>,
}

/// Optimizer/solver diagnostics recorded at fit time.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize, Default)]
pub struct FitDiagnostics {
    /// Whether the solver reported convergence (`None` for direct solvers).
    pub converged: Option<bool>,
    /// Iterations used (`None` when the backend does not expose it).
    pub iterations: Option<u64>,
    /// Infinity norm of the objective gradient at the returned coefficients, computed
    /// by the framework independently of the backend (`None` for direct solvers).
    pub final_gradient_norm: Option<f64>,
    /// Warnings in the spirit of R's `glm` (e.g. fitted probabilities numerically 0 or 1).
    pub warnings: Vec<String>,
    /// Columns dropped before fitting because they were constant (aliased with the
    /// intercept) or identically zero. Their coefficient is reported as 0; R reports `NA`.
    pub aliased: Vec<String>,
    /// Free-form note from the adapter (e.g. termination reason).
    pub note: Option<String>,
}
