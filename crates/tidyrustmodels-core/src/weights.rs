//! Case weights, as tidymodels classifies them (`hardhat::frequency_weights()` and
//! `hardhat::importance_weights()`): frequency weights are non-negative integers
//! that stand for repeated rows and are used by unsupervised recipe steps and, in
//! resampling, by the metrics; importance weights are non-negative doubles that only
//! the model sees (recipes 1.3.2 `are_weights_used()`, tune 2.1.0
//! `.use_case_weights_with_yardstick()`).

use crate::error::{Error, Result};
use serde::{Deserialize, Serialize};

/// The kind of case weights.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum WeightKind {
    /// `hardhat::frequency_weights()`: integer counts of identical rows.
    Frequency,
    /// `hardhat::importance_weights()`: relative importance, model use only.
    Importance,
}

impl WeightKind {
    /// Name as tidymodels prints it.
    pub fn as_str(&self) -> &'static str {
        match self {
            WeightKind::Frequency => "frequency",
            WeightKind::Importance => "importance",
        }
    }
}

/// One weight per training row.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct CaseWeights {
    kind: WeightKind,
    values: Vec<f64>,
}

impl CaseWeights {
    /// Frequency weights: finite, non-negative integers.
    pub fn frequency(values: Vec<f64>) -> Result<Self> {
        if let Some(i) = values
            .iter()
            .position(|v| !v.is_finite() || *v < 0.0 || v.fract() != 0.0)
        {
            return Err(Error::InvalidValue(format!(
                "frequency weights must be non-negative integers (row {i} is {})",
                values[i]
            )));
        }
        Ok(CaseWeights {
            kind: WeightKind::Frequency,
            values,
        })
    }

    /// Importance weights: finite and non-negative.
    pub fn importance(values: Vec<f64>) -> Result<Self> {
        if let Some(i) = values.iter().position(|v| !v.is_finite() || *v < 0.0) {
            return Err(Error::InvalidValue(format!(
                "importance weights must be finite and non-negative (row {i} is {})",
                values[i]
            )));
        }
        Ok(CaseWeights {
            kind: WeightKind::Importance,
            values,
        })
    }

    /// Build from a kind and values.
    pub fn new(kind: WeightKind, values: Vec<f64>) -> Result<Self> {
        match kind {
            WeightKind::Frequency => Self::frequency(values),
            WeightKind::Importance => Self::importance(values),
        }
    }

    /// The kind.
    pub fn kind(&self) -> WeightKind {
        self.kind
    }

    /// The weights.
    pub fn values(&self) -> &[f64] {
        &self.values
    }

    /// Number of rows.
    pub fn len(&self) -> usize {
        self.values.len()
    }

    /// Whether there are no rows.
    pub fn is_empty(&self) -> bool {
        self.values.is_empty()
    }

    /// recipes: unsupervised steps (centering, scaling, imputation, filters) use
    /// frequency weights only (`are_weights_used(wts, unsupervised = TRUE)`).
    pub fn used_by_steps(&self) -> bool {
        self.kind == WeightKind::Frequency
    }

    /// tune: metrics on resamples are weighted by frequency weights only
    /// (`.use_case_weights_with_yardstick()`).
    pub fn used_by_metrics(&self) -> bool {
        self.kind == WeightKind::Frequency
    }

    /// The weights of a subset of rows.
    pub fn select_rows(&self, rows: &[usize]) -> Result<Self> {
        let mut values = Vec::with_capacity(rows.len());
        for &r in rows {
            let v = self.values.get(r).ok_or_else(|| {
                Error::InvalidArgument(format!("row {r} out of range for {} case weights", self.len()))
            })?;
            values.push(*v);
        }
        Ok(CaseWeights {
            kind: self.kind,
            values,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn kinds_validate_and_decide_their_uses() {
        assert!(CaseWeights::frequency(vec![1.0, 2.5]).is_err());
        assert!(CaseWeights::frequency(vec![1.0, -1.0]).is_err());
        let f = CaseWeights::frequency(vec![1.0, 0.0, 3.0]).unwrap();
        assert!(f.used_by_steps() && f.used_by_metrics());
        let i = CaseWeights::importance(vec![0.5, 2.5]).unwrap();
        assert!(!i.used_by_steps() && !i.used_by_metrics());
        assert!(CaseWeights::importance(vec![f64::NAN]).is_err());
        assert_eq!(f.select_rows(&[2, 0]).unwrap().values(), &[3.0, 1.0]);
    }
}
