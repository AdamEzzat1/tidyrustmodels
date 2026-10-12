//! yardstick-compatible metrics.
//!
//! Every function returns `Ok(None)` where yardstick returns `NA` (undefined
//! correlation, undefined precision/recall) and `Err` for malformed inputs. The event
//! level is the first outcome level (`event_level = "first"`, ADR-0004).
//! Definitions were verified against yardstick 1.4.0 function bodies
//! (see DEPENDENCIES.md).

#![forbid(unsafe_code)]
#![warn(missing_docs, rust_2018_idioms)]

use serde::{Deserialize, Serialize};
use tidyrustmodels_core::{Error, Outcome, Prediction, Result};

fn check_len(truth: usize, estimate: usize) -> Result<()> {
    if truth != estimate {
        return Err(Error::InvalidArgument(format!(
            "truth has {truth} rows but estimate has {estimate}"
        )));
    }
    if truth == 0 {
        return Err(Error::InvalidArgument("metrics need at least one row".into()));
    }
    Ok(())
}

fn check_finite(name: &str, v: &[f64]) -> Result<()> {
    if let Some(i) = v.iter().position(|x| !x.is_finite()) {
        return Err(Error::InvalidValue(format!("{name} is non-finite at row {i}")));
    }
    Ok(())
}

/// Root mean squared error: `sqrt(mean((truth - estimate)^2))`.
pub fn rmse(truth: &[f64], estimate: &[f64]) -> Result<Option<f64>> {
    check_len(truth.len(), estimate.len())?;
    check_finite("truth", truth)?;
    check_finite("estimate", estimate)?;
    let mse = truth
        .iter()
        .zip(estimate)
        .map(|(t, e)| (t - e) * (t - e))
        .sum::<f64>()
        / truth.len() as f64;
    Ok(Some(mse.sqrt()))
}

/// Mean absolute error.
pub fn mae(truth: &[f64], estimate: &[f64]) -> Result<Option<f64>> {
    check_len(truth.len(), estimate.len())?;
    check_finite("truth", truth)?;
    check_finite("estimate", estimate)?;
    Ok(Some(
        truth
            .iter()
            .zip(estimate)
            .map(|(t, e)| (t - e).abs())
            .sum::<f64>()
            / truth.len() as f64,
    ))
}

/// yardstick's `rsq`: the **squared Pearson correlation** between truth and estimate.
/// `None` when either vector is constant or has fewer than two rows.
pub fn rsq(truth: &[f64], estimate: &[f64]) -> Result<Option<f64>> {
    check_len(truth.len(), estimate.len())?;
    check_finite("truth", truth)?;
    check_finite("estimate", estimate)?;
    let n = truth.len();
    if n < 2 {
        return Ok(None);
    }
    let mt = truth.iter().sum::<f64>() / n as f64;
    let me = estimate.iter().sum::<f64>() / n as f64;
    let (mut sxy, mut sxx, mut syy) = (0.0, 0.0, 0.0);
    for (t, e) in truth.iter().zip(estimate) {
        let (dt, de) = (t - mt, e - me);
        sxy += dt * de;
        sxx += dt * dt;
        syy += de * de;
    }
    if sxx == 0.0 || syy == 0.0 {
        return Ok(None);
    }
    let r = sxy / (sxx.sqrt() * syy.sqrt());
    Ok(Some(r * r))
}

/// Traditional R²: `1 - SS_res / SS_tot`. `None` when truth is constant.
pub fn rsq_trad(truth: &[f64], estimate: &[f64]) -> Result<Option<f64>> {
    check_len(truth.len(), estimate.len())?;
    check_finite("truth", truth)?;
    check_finite("estimate", estimate)?;
    let mt = truth.iter().sum::<f64>() / truth.len() as f64;
    let ss_res: f64 = truth.iter().zip(estimate).map(|(t, e)| (t - e) * (t - e)).sum();
    let ss_tot: f64 = truth.iter().map(|t| (t - mt) * (t - mt)).sum();
    if ss_tot == 0.0 {
        return Ok(None);
    }
    Ok(Some(1.0 - ss_res / ss_tot))
}

/// Fraction of rows where the predicted class equals the truth.
pub fn accuracy(truth: &[u32], estimate: &[u32]) -> Result<Option<f64>> {
    check_len(truth.len(), estimate.len())?;
    let hits = truth.iter().zip(estimate).filter(|(t, e)| t == e).count();
    Ok(Some(hits as f64 / truth.len() as f64))
}

/// Binary precision for `event`: TP / (TP + FP). `None` when nothing was predicted as
/// the event (yardstick returns `NA` with a warning).
pub fn precision(truth: &[u32], estimate: &[u32], event: u32) -> Result<Option<f64>> {
    check_len(truth.len(), estimate.len())?;
    let tp = truth
        .iter()
        .zip(estimate)
        .filter(|(t, e)| **t == event && **e == event)
        .count();
    let predicted = estimate.iter().filter(|e| **e == event).count();
    Ok(if predicted == 0 {
        None
    } else {
        Some(tp as f64 / predicted as f64)
    })
}

/// Binary recall for `event`: TP / (TP + FN). `None` when the event never occurs.
pub fn recall(truth: &[u32], estimate: &[u32], event: u32) -> Result<Option<f64>> {
    check_len(truth.len(), estimate.len())?;
    let tp = truth
        .iter()
        .zip(estimate)
        .filter(|(t, e)| **t == event && **e == event)
        .count();
    let actual = truth.iter().filter(|t| **t == event).count();
    Ok(if actual == 0 {
        None
    } else {
        Some(tp as f64 / actual as f64)
    })
}

/// yardstick's `mn_log_loss` for a binary outcome: `-mean(log p(truth))` where
/// `prob_event` is the probability of `event` and probabilities are clamped to
/// `[eps, 1 - eps]` with `eps = f64::EPSILON` (verified in `mn_log_loss_multiclass`).
pub fn mn_log_loss(truth: &[u32], prob_event: &[f64], event: u32) -> Result<Option<f64>> {
    check_len(truth.len(), prob_event.len())?;
    check_finite("prob_event", prob_event)?;
    if let Some(i) = prob_event.iter().position(|p| *p < 0.0 || *p > 1.0) {
        return Err(Error::InvalidValue(format!(
            "probability out of [0, 1] at row {i}"
        )));
    }
    let eps = f64::EPSILON;
    let total: f64 = truth
        .iter()
        .zip(prob_event)
        .map(|(t, p)| {
            let p_truth = if *t == event { *p } else { 1.0 - *p };
            -p_truth.clamp(eps, 1.0 - eps).ln()
        })
        .sum();
    Ok(Some(total / truth.len() as f64))
}

/// Whether larger values are better.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum Direction {
    /// Larger is better (accuracy, rsq, ...).
    Maximize,
    /// Smaller is better (rmse, log loss, ...).
    Minimize,
}

/// A named metric that can be evaluated on an `Outcome` / `Prediction` pair.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Metric {
    /// [`rmse`]
    Rmse,
    /// [`mae`]
    Mae,
    /// [`rsq`]
    Rsq,
    /// [`rsq_trad`]
    RsqTrad,
    /// [`accuracy`]
    Accuracy,
    /// [`precision`] with the first level as event
    Precision,
    /// [`recall`] with the first level as event
    Recall,
    /// [`mn_log_loss`] with the first level as event
    MnLogLoss,
}

impl Metric {
    /// yardstick name.
    pub fn name(self) -> &'static str {
        match self {
            Metric::Rmse => "rmse",
            Metric::Mae => "mae",
            Metric::Rsq => "rsq",
            Metric::RsqTrad => "rsq_trad",
            Metric::Accuracy => "accuracy",
            Metric::Precision => "precision",
            Metric::Recall => "recall",
            Metric::MnLogLoss => "mn_log_loss",
        }
    }

    /// Parse a yardstick name.
    pub fn from_name(name: &str) -> Result<Metric> {
        Ok(match name {
            "rmse" => Metric::Rmse,
            "mae" => Metric::Mae,
            "rsq" => Metric::Rsq,
            "rsq_trad" => Metric::RsqTrad,
            "accuracy" => Metric::Accuracy,
            "precision" => Metric::Precision,
            "recall" => Metric::Recall,
            "mn_log_loss" => Metric::MnLogLoss,
            other => return Err(Error::InvalidArgument(format!("unknown metric `{other}`"))),
        })
    }

    /// Optimisation direction.
    pub fn direction(self) -> Direction {
        match self {
            Metric::Rmse | Metric::Mae | Metric::MnLogLoss => Direction::Minimize,
            _ => Direction::Maximize,
        }
    }

    /// Evaluate on an outcome/prediction pair. The event level is code 0.
    pub fn compute(self, truth: &Outcome, pred: &Prediction) -> Result<Option<f64>> {
        match (self, truth, pred) {
            (Metric::Rmse, Outcome::Numeric(t), Prediction::Numeric(e)) => rmse(t, e),
            (Metric::Mae, Outcome::Numeric(t), Prediction::Numeric(e)) => mae(t, e),
            (Metric::Rsq, Outcome::Numeric(t), Prediction::Numeric(e)) => rsq(t, e),
            (Metric::RsqTrad, Outcome::Numeric(t), Prediction::Numeric(e)) => rsq_trad(t, e),
            (Metric::Accuracy, Outcome::Class { codes: t, .. }, Prediction::Class { codes: e, .. }) => {
                accuracy(t, e)
            }
            (Metric::Precision, Outcome::Class { codes: t, .. }, Prediction::Class { codes: e, .. }) => {
                precision(t, e, 0)
            }
            (Metric::Recall, Outcome::Class { codes: t, .. }, Prediction::Class { codes: e, .. }) => {
                recall(t, e, 0)
            }
            (Metric::MnLogLoss, Outcome::Class { codes: t, .. }, p @ Prediction::Class { .. }) => {
                let probs = p.event_probability().ok_or_else(|| {
                    Error::Unsupported(
                        "mn_log_loss needs class probabilities, which this model did not produce".into(),
                    )
                })?;
                mn_log_loss(t, probs, 0)
            }
            _ => Err(Error::Unsupported(format!(
                "metric `{}` does not apply to this outcome/prediction kind",
                self.name()
            ))),
        }
    }
}

/// One evaluated metric.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct MetricValue {
    /// yardstick name.
    pub name: String,
    /// `None` where yardstick would return `NA`.
    pub value: Option<f64>,
}

/// An ordered set of metrics (`yardstick::metric_set`).
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct MetricSet(pub Vec<Metric>);

impl MetricSet {
    /// Build from metrics.
    pub fn new(metrics: Vec<Metric>) -> Self {
        MetricSet(metrics)
    }

    /// Build from yardstick names.
    pub fn from_names<S: AsRef<str>>(names: &[S]) -> Result<Self> {
        Ok(MetricSet(
            names
                .iter()
                .map(|n| Metric::from_name(n.as_ref()))
                .collect::<Result<_>>()?,
        ))
    }

    /// tune's regression default: rmse, rsq.
    pub fn regression_defaults() -> Self {
        MetricSet(vec![Metric::Rmse, Metric::Rsq])
    }

    /// tune's classification default for probability models: accuracy, mn_log_loss
    /// (tune also reports roc_auc, which v0.1 does not implement).
    pub fn classification_defaults() -> Self {
        MetricSet(vec![Metric::Accuracy, Metric::MnLogLoss])
    }

    /// Metrics in order.
    pub fn metrics(&self) -> &[Metric] {
        &self.0
    }

    /// Evaluate every metric.
    pub fn evaluate(&self, truth: &Outcome, pred: &Prediction) -> Result<Vec<MetricValue>> {
        self.0
            .iter()
            .map(|m| {
                Ok(MetricValue {
                    name: m.name().into(),
                    value: m.compute(truth, pred)?,
                })
            })
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;

    #[test]
    fn regression_metrics_hand_computed() {
        let t = [1.0, 2.0, 3.0, 4.0];
        let e = [1.5, 1.5, 3.5, 3.5];
        assert_abs_diff_eq!(rmse(&t, &e).unwrap().unwrap(), 0.5);
        assert_abs_diff_eq!(mae(&t, &e).unwrap().unwrap(), 0.5);
        // rsq = cor^2: cor(t, e) = 0.894427..., squared = 0.8
        assert_abs_diff_eq!(rsq(&t, &e).unwrap().unwrap(), 0.8, epsilon = 1e-12);
        // rsq_trad = 1 - 1.0/5.0 = 0.8 here as well; distinguish with another case:
        let e2 = [2.0, 3.0, 4.0, 5.0]; // perfectly correlated, biased
        assert_abs_diff_eq!(rsq(&t, &e2).unwrap().unwrap(), 1.0, epsilon = 1e-12);
        assert_abs_diff_eq!(
            rsq_trad(&t, &e2).unwrap().unwrap(),
            1.0 - 4.0 / 5.0,
            epsilon = 1e-12
        );
        assert_eq!(rsq(&[1.0, 1.0], &[1.0, 2.0]).unwrap(), None);
        assert!(rmse(&t, &e[..3]).is_err());
    }

    #[test]
    fn classification_metrics_event_first() {
        let t = [0, 0, 1, 1, 0];
        let e = [0, 1, 1, 1, 0];
        assert_abs_diff_eq!(accuracy(&t, &e).unwrap().unwrap(), 0.8);
        // event = 0: TP = 2 (rows 0, 4), predicted 0 = 2 -> precision 1; actual 0 = 3 -> recall 2/3
        assert_abs_diff_eq!(precision(&t, &e, 0).unwrap().unwrap(), 1.0);
        assert_abs_diff_eq!(recall(&t, &e, 0).unwrap().unwrap(), 2.0 / 3.0);
        assert_eq!(precision(&[1, 1], &[1, 1], 0).unwrap(), None);
        assert_eq!(recall(&[1, 1], &[0, 0], 0).unwrap(), None);
    }

    #[test]
    fn log_loss_matches_yardstick_including_clamping() {
        // yardstick: mn_log_loss(truth = a,b ; estimate = 0, 0.5) = 18.3684 (eps-clamped)
        let v = mn_log_loss(&[0, 1], &[0.0, 0.5], 0).unwrap().unwrap();
        assert_abs_diff_eq!(v, 18.3684, epsilon = 5e-5);
        let v = mn_log_loss(&[0, 1], &[0.9, 0.2], 0).unwrap().unwrap();
        assert_abs_diff_eq!(v, -(0.9f64.ln() + 0.8f64.ln()) / 2.0, epsilon = 1e-15);
        assert!(mn_log_loss(&[0], &[1.5], 0).is_err());
    }

    #[test]
    fn metric_set_dispatch() {
        let set = MetricSet::from_names(&["rmse", "rsq"]).unwrap();
        let out = set
            .evaluate(
                &Outcome::Numeric(vec![1.0, 2.0]),
                &Prediction::Numeric(vec![1.0, 2.0]),
            )
            .unwrap();
        assert_eq!(out[0].name, "rmse");
        assert_eq!(out[0].value, Some(0.0));
        assert!(MetricSet::from_names(&["nope"]).is_err());
        assert!(set
            .evaluate(
                &Outcome::Numeric(vec![1.0]),
                &Prediction::Class {
                    codes: vec![0],
                    levels: vec![],
                    probabilities: None
                }
            )
            .is_err());
    }
}
