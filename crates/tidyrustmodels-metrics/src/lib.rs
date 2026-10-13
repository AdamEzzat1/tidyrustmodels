//! yardstick-compatible metrics.
//!
//! Every function returns `Ok(None)` where yardstick returns `NA` (undefined
//! correlation, undefined precision/recall, no events for a curve) and `Err` for
//! malformed inputs. The event level is the first outcome level (`event_level =
//! "first"`, ADR-0004). Definitions were transcribed from the yardstick 1.4.0 function
//! bodies (see DEPENDENCIES.md): regression and binary helpers here, confusion-table
//! metrics in [`class`], threshold curves and probability metrics in [`curves`].

#![forbid(unsafe_code)]
#![warn(missing_docs, rust_2018_idioms)]

pub mod class;
pub mod curves;

pub use class::{weighted_mean, ConfusionMatrix, Estimator, KapWeighting};
pub use curves::{
    average_precision, average_precision_multiclass, average_precision_multiclass_weighted,
    average_precision_weighted, binary_threshold_curve, binary_threshold_curve_weighted, brier_class,
    brier_class_multiclass, brier_class_multiclass_weighted, brier_class_weighted, mn_log_loss_multiclass,
    mn_log_loss_multiclass_weighted, pr_auc, pr_auc_multiclass, pr_auc_multiclass_weighted, pr_auc_weighted,
    pr_curve, roc_auc, roc_auc_multiclass, roc_auc_multiclass_weighted, roc_auc_weighted, roc_curve,
    trapezoid_auc, PrCurve, RocCurve, ThresholdCurve,
};
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
    rmse_weighted(truth, estimate, None)
}

/// `rmse` with optional case weights: `sqrt(weighted.mean((truth - estimate)^2, w))`.
pub fn rmse_weighted(truth: &[f64], estimate: &[f64], w: Option<&[f64]>) -> Result<Option<f64>> {
    check_len(truth.len(), estimate.len())?;
    check_finite("truth", truth)?;
    check_finite("estimate", estimate)?;
    check_weights(truth.len(), w)?;
    let errors: Vec<f64> = truth
        .iter()
        .zip(estimate)
        .map(|(t, e)| (t - e) * (t - e))
        .collect();
    Ok(Some(yardstick_mean(&errors, w).sqrt()))
}

fn check_weights(n: usize, w: Option<&[f64]>) -> Result<()> {
    if let Some(w) = w {
        check_len(n, w.len())?;
        check_finite("case_weights", w)?;
        if let Some(i) = w.iter().position(|v| *v < 0.0) {
            return Err(Error::InvalidValue(format!("negative case weight at row {i}")));
        }
    }
    Ok(())
}

/// `yardstick_mean`: `mean(x)`, or `stats::weighted.mean(x, w)`, which is
/// `sum((x * w)[w != 0]) / sum(w)`.
pub fn yardstick_mean(x: &[f64], w: Option<&[f64]>) -> f64 {
    match w {
        None => x.iter().sum::<f64>() / x.len() as f64,
        Some(w) => {
            let num: f64 = x
                .iter()
                .zip(w)
                .filter(|(_, wi)| **wi != 0.0)
                .map(|(xi, wi)| xi * wi)
                .sum();
            num / w.iter().sum::<f64>()
        }
    }
}

/// `yardstick_sum`: `sum(x)` or `sum(x * w)`.
pub fn yardstick_sum(x: &[f64], w: Option<&[f64]>) -> f64 {
    match w {
        None => x.iter().sum(),
        Some(w) => x.iter().zip(w).map(|(xi, wi)| xi * wi).sum(),
    }
}

/// Mean absolute error.
pub fn mae(truth: &[f64], estimate: &[f64]) -> Result<Option<f64>> {
    mae_weighted(truth, estimate, None)
}

/// `mae` with optional case weights: `weighted.mean(|truth - estimate|, w)`.
pub fn mae_weighted(truth: &[f64], estimate: &[f64], w: Option<&[f64]>) -> Result<Option<f64>> {
    check_len(truth.len(), estimate.len())?;
    check_finite("truth", truth)?;
    check_finite("estimate", estimate)?;
    check_weights(truth.len(), w)?;
    let errors: Vec<f64> = truth.iter().zip(estimate).map(|(t, e)| (t - e).abs()).collect();
    Ok(Some(yardstick_mean(&errors, w)))
}

/// yardstick's `rsq`: the **squared Pearson correlation** between truth and estimate.
/// `None` when either vector is constant or has fewer than two rows.
pub fn rsq(truth: &[f64], estimate: &[f64]) -> Result<Option<f64>> {
    rsq_weighted(truth, estimate, None)
}

/// `rsq` with optional case weights: the squared correlation from
/// `stats::cov.wt(cbind(truth, estimate), wt = w, cor = TRUE, method = "unbiased")`
/// (`yardstick_cor`), `NA` for fewer than two rows or a constant truth or estimate.
pub fn rsq_weighted(truth: &[f64], estimate: &[f64], w: Option<&[f64]>) -> Result<Option<f64>> {
    check_len(truth.len(), estimate.len())?;
    check_finite("truth", truth)?;
    check_finite("estimate", estimate)?;
    check_weights(truth.len(), w)?;
    let n = truth.len();
    if n < 2 {
        return Ok(None);
    }
    let constant = |v: &[f64]| v.iter().all(|x| *x == v[0]);
    if constant(truth) || constant(estimate) {
        return Ok(None);
    }
    match w {
        None => {
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
        Some(w) => {
            // cov.wt: wt <- wt / sum(wt); center <- colSums(wt * x);
            // x <- sqrt(wt) * (x - center); cov <- crossprod(x) / (1 - sum(wt^2))
            let sw: f64 = w.iter().sum();
            if sw == 0.0 {
                return Ok(None);
            }
            let wn: Vec<f64> = w.iter().map(|v| v / sw).collect();
            let mt: f64 = wn.iter().zip(truth).map(|(a, b)| a * b).sum();
            let me: f64 = wn.iter().zip(estimate).map(|(a, b)| a * b).sum();
            let (mut sxy, mut sxx, mut syy) = (0.0, 0.0, 0.0);
            for ((t, e), wi) in truth.iter().zip(estimate).zip(&wn) {
                let (dt, de) = (wi.sqrt() * (t - mt), wi.sqrt() * (e - me));
                sxy += dt * de;
                sxx += dt * dt;
                syy += de * de;
            }
            let denom = 1.0 - wn.iter().map(|v| v * v).sum::<f64>();
            let (cxy, cxx, cyy) = (sxy / denom, sxx / denom, syy / denom);
            let r = (1.0 / cxx.sqrt()) * cxy * (1.0 / cyy.sqrt());
            Ok(Some(r * r))
        }
    }
}

/// Traditional R²: `1 - SS_res / SS_tot`. `None` when truth is constant.
pub fn rsq_trad(truth: &[f64], estimate: &[f64]) -> Result<Option<f64>> {
    rsq_trad_weighted(truth, estimate, None)
}

/// `rsq_trad` with optional case weights: `1 - SS_res / SS_tot` with weighted sums
/// around the weighted mean of `truth`.
pub fn rsq_trad_weighted(truth: &[f64], estimate: &[f64], w: Option<&[f64]>) -> Result<Option<f64>> {
    check_len(truth.len(), estimate.len())?;
    check_finite("truth", truth)?;
    check_finite("estimate", estimate)?;
    check_weights(truth.len(), w)?;
    let mt = yardstick_mean(truth, w);
    let res: Vec<f64> = truth
        .iter()
        .zip(estimate)
        .map(|(t, e)| (t - e) * (t - e))
        .collect();
    let tot: Vec<f64> = truth.iter().map(|t| (t - mt) * (t - mt)).collect();
    let ss_res = yardstick_sum(&res, w);
    let ss_tot = yardstick_sum(&tot, w);
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
    mn_log_loss_weighted(truth, prob_event, event, None)
}

/// `mn_log_loss` with optional case weights (`weighted.mean` of the row losses).
pub fn mn_log_loss_weighted(
    truth: &[u32],
    prob_event: &[f64],
    event: u32,
    w: Option<&[f64]>,
) -> Result<Option<f64>> {
    check_len(truth.len(), prob_event.len())?;
    check_finite("prob_event", prob_event)?;
    check_weights(truth.len(), w)?;
    if let Some(i) = prob_event.iter().position(|p| *p < 0.0 || *p > 1.0) {
        return Err(Error::InvalidValue(format!(
            "probability out of [0, 1] at row {i}"
        )));
    }
    let eps = f64::EPSILON;
    let loss: Vec<f64> = truth
        .iter()
        .zip(prob_event)
        .map(|(t, p)| {
            let p_truth = if *t == event { *p } else { 1.0 - *p };
            -p_truth.clamp(eps, 1.0 - eps).ln()
        })
        .collect();
    Ok(Some(yardstick_mean(&loss, w)))
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
    /// [`roc_auc`] (binary) / Hand-Till (multiclass)
    RocAuc,
    /// [`pr_auc`] (binary) / macro (multiclass)
    PrAuc,
    /// [`average_precision`] (binary) / macro (multiclass)
    AveragePrecision,
    /// `ConfusionMatrix::f_meas` with `beta = 1`
    FMeas,
    /// `ConfusionMatrix::sens`
    Sens,
    /// `ConfusionMatrix::spec`
    Spec,
    /// `ConfusionMatrix::bal_accuracy`
    BalAccuracy,
    /// `ConfusionMatrix::kap` unweighted
    Kap,
    /// `ConfusionMatrix::mcc`
    Mcc,
    /// `ConfusionMatrix::j_index`
    JIndex,
    /// [`brier_class`]
    BrierClass,
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
            Metric::RocAuc => "roc_auc",
            Metric::PrAuc => "pr_auc",
            Metric::AveragePrecision => "average_precision",
            Metric::FMeas => "f_meas",
            Metric::Sens => "sens",
            Metric::Spec => "spec",
            Metric::BalAccuracy => "bal_accuracy",
            Metric::Kap => "kap",
            Metric::Mcc => "mcc",
            Metric::JIndex => "j_index",
            Metric::BrierClass => "brier_class",
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
            "roc_auc" => Metric::RocAuc,
            "pr_auc" => Metric::PrAuc,
            "average_precision" => Metric::AveragePrecision,
            "f_meas" => Metric::FMeas,
            "sens" | "sensitivity" => Metric::Sens,
            "spec" | "specificity" => Metric::Spec,
            "bal_accuracy" => Metric::BalAccuracy,
            "kap" => Metric::Kap,
            "mcc" => Metric::Mcc,
            "j_index" => Metric::JIndex,
            "brier_class" => Metric::BrierClass,
            other => return Err(Error::InvalidArgument(format!("unknown metric `{other}`"))),
        })
    }

    /// Optimisation direction.
    pub fn direction(self) -> Direction {
        match self {
            Metric::Rmse | Metric::Mae | Metric::MnLogLoss | Metric::BrierClass => Direction::Minimize,
            _ => Direction::Maximize,
        }
    }

    /// Whether this metric needs class probabilities.
    pub fn needs_probabilities(self) -> bool {
        matches!(
            self,
            Metric::MnLogLoss
                | Metric::RocAuc
                | Metric::PrAuc
                | Metric::AveragePrecision
                | Metric::BrierClass
        )
    }

    /// Evaluate on an outcome/prediction pair with yardstick's default estimator: the
    /// first level is the event for two-level outcomes; with more levels, class metrics
    /// use `macro`, `roc_auc` uses `hand_till`, `pr_auc`/`average_precision` use `macro`,
    /// and `kap`/`mcc`/`brier_class`/`mn_log_loss`/`accuracy` use their multiclass forms.
    pub fn compute(self, truth: &Outcome, pred: &Prediction) -> Result<Option<f64>> {
        self.compute_weighted(truth, pred, None)
    }

    /// [`compute`](Self::compute) with optional case weights, as yardstick's
    /// `case_weights`: weighted means and sums, a weighted confusion table, weighted
    /// threshold curves; the multiclass `roc_auc` default becomes `macro`, since
    /// `hand_till` cannot take weights.
    pub fn compute_weighted(
        self,
        truth: &Outcome,
        pred: &Prediction,
        w: Option<&[f64]>,
    ) -> Result<Option<f64>> {
        match (truth, pred) {
            (Outcome::Numeric(t), Prediction::Numeric(e)) => match self {
                Metric::Rmse => rmse_weighted(t, e, w),
                Metric::Mae => mae_weighted(t, e, w),
                Metric::Rsq => rsq_weighted(t, e, w),
                Metric::RsqTrad => rsq_trad_weighted(t, e, w),
                _ => Err(self.not_applicable()),
            },
            (
                Outcome::Class { codes: t, levels },
                Prediction::Class {
                    codes: e,
                    probabilities,
                    ..
                },
            ) => {
                if matches!(self, Metric::Rmse | Metric::Mae | Metric::Rsq | Metric::RsqTrad) {
                    return Err(self.not_applicable());
                }
                let k = levels.len();
                let est = Estimator::default_for(k);
                if self.needs_probabilities() {
                    let p = probabilities.as_ref().ok_or_else(|| {
                        Error::Unsupported(format!(
                            "{} needs class probabilities, which this model did not produce",
                            self.name()
                        ))
                    })?;
                    if p.ncols() != k {
                        return Err(Error::InvalidArgument(format!(
                            "probability matrix has {} columns but the outcome has {k} levels",
                            p.ncols()
                        )));
                    }
                    if k == 2 {
                        let pe = p.column(0);
                        return match self {
                            Metric::MnLogLoss => mn_log_loss_weighted(t, pe, 0, w),
                            Metric::RocAuc => roc_auc_weighted(t, pe, 0, w),
                            Metric::PrAuc => pr_auc_weighted(t, pe, 0, w),
                            Metric::AveragePrecision => average_precision_weighted(t, pe, 0, w),
                            Metric::BrierClass => brier_class_weighted(t, pe, 0, w),
                            _ => unreachable!(),
                        };
                    }
                    // finalize_estimator_roc_auc: hand_till without weights, macro with
                    let roc_est = if w.is_some() {
                        Estimator::Macro
                    } else {
                        Estimator::HandTill
                    };
                    return match self {
                        Metric::MnLogLoss => mn_log_loss_multiclass_weighted(t, p, w),
                        Metric::RocAuc => roc_auc_multiclass_weighted(t, p, roc_est, w),
                        Metric::PrAuc => pr_auc_multiclass_weighted(t, p, est, w),
                        Metric::AveragePrecision => average_precision_multiclass_weighted(t, p, est, w),
                        Metric::BrierClass => brier_class_multiclass_weighted(t, p, w),
                        _ => unreachable!(),
                    };
                }
                check_len(t.len(), e.len())?;
                let cm = ConfusionMatrix::from_codes_weighted(t, e, levels, w)?;
                match self {
                    Metric::Accuracy => Ok(cm.accuracy()),
                    Metric::Precision => cm.precision(est),
                    Metric::Recall | Metric::Sens => cm.sens(est),
                    Metric::Spec => cm.spec(est),
                    Metric::FMeas => cm.f_meas(est, 1.0),
                    Metric::BalAccuracy => cm.bal_accuracy(est),
                    Metric::JIndex => cm.j_index(est),
                    Metric::Kap => Ok(cm.kap(KapWeighting::None)),
                    Metric::Mcc => Ok(cm.mcc()),
                    _ => unreachable!(),
                }
            }
            _ => Err(self.not_applicable()),
        }
    }

    fn not_applicable(self) -> Error {
        Error::Unsupported(format!(
            "metric `{}` does not apply to this outcome/prediction kind",
            self.name()
        ))
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

    /// tune's classification default for probability models:
    /// `metric_set(accuracy, roc_auc, brier_class)` (tune 1.2+).
    pub fn classification_defaults() -> Self {
        MetricSet(vec![Metric::Accuracy, Metric::RocAuc, Metric::BrierClass])
    }

    /// Metrics in order.
    pub fn metrics(&self) -> &[Metric] {
        &self.0
    }

    /// Evaluate every metric.
    pub fn evaluate(&self, truth: &Outcome, pred: &Prediction) -> Result<Vec<MetricValue>> {
        self.evaluate_weighted(truth, pred, None)
    }

    /// Evaluate every metric with optional case weights.
    pub fn evaluate_weighted(
        &self,
        truth: &Outcome,
        pred: &Prediction,
        w: Option<&[f64]>,
    ) -> Result<Vec<MetricValue>> {
        self.0
            .iter()
            .map(|m| {
                Ok(MetricValue {
                    name: m.name().into(),
                    value: m.compute_weighted(truth, pred, w)?,
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
