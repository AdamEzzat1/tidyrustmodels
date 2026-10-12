//! Backend-independent fitted models. Prediction lives here, not in the backends.

use serde::{Deserialize, Serialize};
use tidyrustmodels_core::traits::check_features;
use tidyrustmodels_core::{
    BackendInfo, Coefficients, DesignMatrix, Error, FitDiagnostics, FittedModel, Prediction, Result,
};

/// Serialization format version for fitted models.
pub const MODEL_FORMAT_VERSION: u32 = 1;

/// Linear predictor `eta = intercept + X beta`, computed column-wise over the
/// column-major design matrix.
fn linear_predictor(x: &DesignMatrix, intercept: Option<f64>, coefs: &[f64]) -> Vec<f64> {
    let mut eta = vec![intercept.unwrap_or(0.0); x.nrows()];
    for (j, b) in coefs.iter().enumerate() {
        let col = x.column(j);
        for (e, v) in eta.iter_mut().zip(col) {
            *e += b * v;
        }
    }
    eta
}

/// Numerically stable logistic function.
pub fn sigmoid(z: f64) -> f64 {
    if z >= 0.0 {
        1.0 / (1.0 + (-z).exp())
    } else {
        let e = z.exp();
        e / (1.0 + e)
    }
}

/// A fitted linear regression.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct FittedLinear {
    model: String,
    format_version: u32,
    /// Intercept, if fitted.
    pub intercept: Option<f64>,
    /// One coefficient per feature.
    pub coefs: Vec<f64>,
    names: Vec<String>,
    backend: BackendInfo,
    diagnostics: FitDiagnostics,
}

impl FittedLinear {
    /// Assemble from backend output.
    pub fn new(
        intercept: Option<f64>,
        coefs: Vec<f64>,
        names: Vec<String>,
        backend: BackendInfo,
        diagnostics: FitDiagnostics,
    ) -> Result<Self> {
        if coefs.len() != names.len() {
            return Err(Error::InvalidArgument(format!(
                "{} coefficients for {} features",
                coefs.len(),
                names.len()
            )));
        }
        if coefs.iter().chain(intercept.iter()).any(|c| !c.is_finite()) {
            return Err(Error::Backend {
                engine: backend.engine.clone(),
                message: "returned non-finite coefficients".into(),
            });
        }
        Ok(FittedLinear {
            model: "linear_reg".into(),
            format_version: MODEL_FORMAT_VERSION,
            intercept,
            coefs,
            names,
            backend,
            diagnostics,
        })
    }
}

impl FittedModel for FittedLinear {
    fn predict(&self, x: &DesignMatrix) -> Result<Prediction> {
        check_features(&self.names, x)?;
        Ok(Prediction::Numeric(linear_predictor(
            x,
            self.intercept,
            &self.coefs,
        )))
    }

    fn feature_names(&self) -> &[String] {
        &self.names
    }

    fn coefficients(&self) -> Option<Coefficients> {
        Some(Coefficients {
            intercept: self.intercept,
            terms: self
                .names
                .iter()
                .cloned()
                .zip(self.coefs.iter().copied())
                .collect(),
        })
    }

    fn diagnostics(&self) -> FitDiagnostics {
        self.diagnostics.clone()
    }

    fn backend(&self) -> BackendInfo {
        self.backend.clone()
    }

    fn state(&self) -> Result<serde_json::Value> {
        serde_json::to_value(self).map_err(|e| Error::Serialization(e.to_string()))
    }
}

/// A fitted binary logistic regression.
///
/// Coefficients are for the log-odds of `levels[1]`, matching R's `glm` with a
/// two-level factor outcome. The event level for metrics remains `levels[0]`
/// (ADR-0004); the probability matrix carries both columns.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct FittedLogistic {
    model: String,
    format_version: u32,
    /// Intercept, if fitted.
    pub intercept: Option<f64>,
    /// One coefficient per feature (log-odds of `levels[1]`).
    pub coefs: Vec<f64>,
    names: Vec<String>,
    /// Outcome levels in training order.
    pub levels: Vec<String>,
    backend: BackendInfo,
    diagnostics: FitDiagnostics,
}

impl FittedLogistic {
    /// Assemble from backend output.
    pub fn new(
        intercept: Option<f64>,
        coefs: Vec<f64>,
        names: Vec<String>,
        levels: Vec<String>,
        backend: BackendInfo,
        diagnostics: FitDiagnostics,
    ) -> Result<Self> {
        if coefs.len() != names.len() {
            return Err(Error::InvalidArgument(format!(
                "{} coefficients for {} features",
                coefs.len(),
                names.len()
            )));
        }
        if levels.len() != 2 {
            return Err(Error::InvalidArgument(
                "logistic regression needs exactly two levels".into(),
            ));
        }
        if coefs.iter().chain(intercept.iter()).any(|c| !c.is_finite()) {
            return Err(Error::Backend {
                engine: backend.engine.clone(),
                message: "returned non-finite coefficients".into(),
            });
        }
        Ok(FittedLogistic {
            model: "logistic_reg".into(),
            format_version: MODEL_FORMAT_VERSION,
            intercept,
            coefs,
            names,
            levels,
            backend,
            diagnostics,
        })
    }

    /// Probability of `levels[1]` per row.
    pub fn prob_second_level(&self, x: &DesignMatrix) -> Result<Vec<f64>> {
        check_features(&self.names, x)?;
        Ok(linear_predictor(x, self.intercept, &self.coefs)
            .into_iter()
            .map(sigmoid)
            .collect())
    }
}

impl FittedModel for FittedLogistic {
    fn predict(&self, x: &DesignMatrix) -> Result<Prediction> {
        let p1 = self.prob_second_level(x)?;
        let n = p1.len();
        let mut data = Vec::with_capacity(2 * n);
        data.extend(p1.iter().map(|p| 1.0 - p));
        data.extend_from_slice(&p1);
        let probabilities = DesignMatrix::from_column_major(data, n, 2, self.levels.clone())?;
        // parsnip picks the level with the larger probability; ties go to the first level.
        let codes = p1.iter().map(|p| u32::from(*p > 0.5)).collect();
        Ok(Prediction::Class {
            codes,
            levels: self.levels.clone(),
            probabilities: Some(probabilities),
        })
    }

    fn feature_names(&self) -> &[String] {
        &self.names
    }

    fn coefficients(&self) -> Option<Coefficients> {
        Some(Coefficients {
            intercept: self.intercept,
            terms: self
                .names
                .iter()
                .cloned()
                .zip(self.coefs.iter().copied())
                .collect(),
        })
    }

    fn diagnostics(&self) -> FitDiagnostics {
        self.diagnostics.clone()
    }

    fn backend(&self) -> BackendInfo {
        self.backend.clone()
    }

    fn state(&self) -> Result<serde_json::Value> {
        serde_json::to_value(self).map_err(|e| Error::Serialization(e.to_string()))
    }
}

/// A fitted multinomial regression (ADR-0010).
///
/// `levels[0]` is the baseline; `coefs[c - 1]` / `intercepts[c - 1]` give the log-odds
/// of `levels[c]` against the baseline, as `coef(nnet::multinom)` reports them.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct FittedMultinom {
    model: String,
    format_version: u32,
    /// Intercept per non-baseline level, if fitted.
    pub intercepts: Option<Vec<f64>>,
    /// Slopes per non-baseline level (each one coefficient per feature).
    pub coefs: Vec<Vec<f64>>,
    names: Vec<String>,
    /// Outcome levels in training order.
    pub levels: Vec<String>,
    backend: BackendInfo,
    diagnostics: FitDiagnostics,
}

impl FittedMultinom {
    /// Assemble from backend output.
    pub fn new(
        intercepts: Option<Vec<f64>>,
        coefs: Vec<Vec<f64>>,
        names: Vec<String>,
        levels: Vec<String>,
        backend: BackendInfo,
        diagnostics: FitDiagnostics,
    ) -> Result<Self> {
        if levels.len() < 2 {
            return Err(Error::InvalidArgument(
                "multinomial regression needs at least two levels".into(),
            ));
        }
        if coefs.len() != levels.len() - 1 || intercepts.as_ref().is_some_and(|b| b.len() != levels.len() - 1)
        {
            return Err(Error::InvalidArgument(format!(
                "{} coefficient sets for {} levels",
                coefs.len(),
                levels.len()
            )));
        }
        for c in &coefs {
            if c.len() != names.len() {
                return Err(Error::InvalidArgument(format!(
                    "{} coefficients for {} features",
                    c.len(),
                    names.len()
                )));
            }
        }
        if coefs
            .iter()
            .flatten()
            .chain(intercepts.iter().flatten())
            .any(|c| !c.is_finite())
        {
            return Err(Error::Backend {
                engine: backend.engine.clone(),
                message: "returned non-finite coefficients".into(),
            });
        }
        Ok(FittedMultinom {
            model: "multinom_reg".into(),
            format_version: MODEL_FORMAT_VERSION,
            intercepts,
            coefs,
            names,
            levels,
            backend,
            diagnostics,
        })
    }

    /// Probability matrix (`n × levels`), softmax over `eta_0 = 0` and the per-level
    /// linear predictors, computed stably.
    pub fn probabilities(&self, x: &DesignMatrix) -> Result<DesignMatrix> {
        check_features(&self.names, x)?;
        let n = x.nrows();
        let k = self.levels.len();
        let etas: Vec<Vec<f64>> = (0..k - 1)
            .map(|c| linear_predictor(x, self.intercepts.as_ref().map(|b| b[c]), &self.coefs[c]))
            .collect();
        let mut data = vec![0.0; n * k];
        for i in 0..n {
            let m = etas.iter().map(|e| e[i]).fold(0.0f64, f64::max);
            let mut s = (-m).exp();
            data[i] = s;
            for c in 1..k {
                let e = (etas[c - 1][i] - m).exp();
                data[c * n + i] = e;
                s += e;
            }
            for c in 0..k {
                data[c * n + i] /= s;
            }
        }
        DesignMatrix::from_column_major(data, n, k, self.levels.clone())
    }
}

impl FittedModel for FittedMultinom {
    fn predict(&self, x: &DesignMatrix) -> Result<Prediction> {
        let p = self.probabilities(x)?;
        let (n, k) = (p.nrows(), p.ncols());
        // parsnip/nnet: the level with the largest probability; ties go to the first.
        let codes = (0..n)
            .map(|i| {
                let mut best = 0;
                for c in 1..k {
                    if p.get(i, c) > p.get(i, best) {
                        best = c;
                    }
                }
                best as u32
            })
            .collect();
        Ok(Prediction::Class {
            codes,
            levels: self.levels.clone(),
            probabilities: Some(p),
        })
    }

    fn feature_names(&self) -> &[String] {
        &self.names
    }

    fn class_coefficients(&self) -> Option<Vec<(String, Coefficients)>> {
        Some(
            (1..self.levels.len())
                .map(|c| {
                    (
                        self.levels[c].clone(),
                        Coefficients {
                            intercept: self.intercepts.as_ref().map(|b| b[c - 1]),
                            terms: self
                                .names
                                .iter()
                                .cloned()
                                .zip(self.coefs[c - 1].iter().copied())
                                .collect(),
                        },
                    )
                })
                .collect(),
        )
    }

    fn diagnostics(&self) -> FitDiagnostics {
        self.diagnostics.clone()
    }

    fn backend(&self) -> BackendInfo {
        self.backend.clone()
    }

    fn state(&self) -> Result<serde_json::Value> {
        serde_json::to_value(self).map_err(|e| Error::Serialization(e.to_string()))
    }
}

/// Rebuild a fitted model from its `state()` JSON.
pub fn fitted_model_from_state(value: &serde_json::Value) -> Result<Box<dyn FittedModel>> {
    let tag = value
        .get("model")
        .and_then(|v| v.as_str())
        .ok_or_else(|| Error::Serialization("fitted model state lacks a `model` tag".into()))?;
    let version = value.get("format_version").and_then(|v| v.as_u64()).unwrap_or(0);
    if version != u64::from(MODEL_FORMAT_VERSION) {
        return Err(Error::Serialization(format!(
            "fitted model format version {version} is not supported (expected {MODEL_FORMAT_VERSION})"
        )));
    }
    let de = |e: serde_json::Error| Error::Serialization(format!("model `{tag}`: {e}"));
    Ok(match tag {
        "linear_reg" => Box::new(serde_json::from_value::<FittedLinear>(value.clone()).map_err(de)?),
        "logistic_reg" => Box::new(serde_json::from_value::<FittedLogistic>(value.clone()).map_err(de)?),
        "multinom_reg" => Box::new(serde_json::from_value::<FittedMultinom>(value.clone()).map_err(de)?),
        "decision_tree" => {
            Box::new(serde_json::from_value::<crate::tree::FittedDecisionTree>(value.clone()).map_err(de)?)
        }
        other => return Err(Error::Serialization(format!("unknown model `{other}`"))),
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;

    fn info() -> BackendInfo {
        BackendInfo {
            engine: "test".into(),
            crate_version: "0".into(),
            solver: "none".into(),
        }
    }

    #[test]
    fn linear_prediction_and_feature_check() {
        let m = FittedLinear::new(
            Some(1.0),
            vec![2.0, -1.0],
            vec!["a".into(), "b".into()],
            info(),
            Default::default(),
        )
        .unwrap();
        let x = DesignMatrix::from_columns(vec![("a".into(), vec![1.0, 2.0]), ("b".into(), vec![0.0, 4.0])])
            .unwrap();
        match m.predict(&x).unwrap() {
            Prediction::Numeric(p) => assert_eq!(p, vec![3.0, 1.0]),
            _ => panic!(),
        }
        let wrong =
            DesignMatrix::from_columns(vec![("b".into(), vec![1.0]), ("a".into(), vec![0.0])]).unwrap();
        assert!(matches!(m.predict(&wrong), Err(Error::SchemaMismatch(_))));
    }

    #[test]
    fn logistic_probabilities_sum_to_one_and_event_is_first_level() {
        let m = FittedLogistic::new(
            Some(0.0),
            vec![1.0],
            vec!["a".into()],
            vec!["no".into(), "yes".into()],
            info(),
            Default::default(),
        )
        .unwrap();
        let x = DesignMatrix::from_columns(vec![("a".into(), vec![0.0, 10.0, -10.0])]).unwrap();
        match m.predict(&x).unwrap() {
            Prediction::Class {
                codes,
                probabilities: Some(p),
                levels,
            } => {
                assert_eq!(levels, vec!["no", "yes"]);
                assert_eq!(codes, vec![0, 1, 0]);
                for i in 0..3 {
                    assert_abs_diff_eq!(p.get(i, 0) + p.get(i, 1), 1.0, epsilon = 1e-15);
                }
                assert_abs_diff_eq!(p.get(0, 1), 0.5);
                assert!(p.get(1, 1) > 0.999);
            }
            _ => panic!(),
        }
    }

    #[test]
    fn sigmoid_is_stable() {
        assert_eq!(sigmoid(1000.0), 1.0);
        assert!(sigmoid(-1000.0) >= 0.0 && sigmoid(-1000.0) < 1e-300);
        assert_abs_diff_eq!(sigmoid(0.0), 0.5);
    }

    #[test]
    fn state_round_trip() {
        let m = FittedLogistic::new(
            Some(0.5),
            vec![1.0, 2.0],
            vec!["a".into(), "b".into()],
            vec!["no".into(), "yes".into()],
            info(),
            Default::default(),
        )
        .unwrap();
        let back = fitted_model_from_state(&m.state().unwrap()).unwrap();
        let x = DesignMatrix::from_columns(vec![("a".into(), vec![0.3]), ("b".into(), vec![-0.2])]).unwrap();
        assert_eq!(m.predict(&x).unwrap(), back.predict(&x).unwrap());
        assert!(
            fitted_model_from_state(&serde_json::json!({"model": "linear_reg", "format_version": 7}))
                .is_err()
        );
    }
}
