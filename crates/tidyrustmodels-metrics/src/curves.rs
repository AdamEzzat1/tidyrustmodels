//! Threshold curves and probability metrics with yardstick 1.4.0 semantics
//! (`binary_threshold_curve`, `roc_curve_binary`, `pr_curve_binary`, `auc`,
//! `roc_auc_hand_till`, `one_vs_all_impl`, `brier_ind`, `mn_log_loss_multiclass`).
//!
//! Binary functions take coded truth and the probability of `event`; multiclass
//! functions take a probability matrix with one column per level in level order.

use crate::class::{weighted_mean, Estimator};
use tidyrustmodels_core::{DesignMatrix, Error, Result};

fn check_prob(truth: usize, prob: &[f64]) -> Result<()> {
    if truth != prob.len() {
        return Err(Error::InvalidArgument(format!(
            "truth has {truth} rows but estimate has {}",
            prob.len()
        )));
    }
    if truth == 0 {
        return Err(Error::InvalidArgument("metrics need at least one row".into()));
    }
    if let Some(i) = prob.iter().position(|p| !p.is_finite()) {
        return Err(Error::InvalidValue(format!("estimate is non-finite at row {i}")));
    }
    Ok(())
}

/// Cumulative true/false positives at each distinct threshold, thresholds in
/// decreasing order (`binary_threshold_curve`).
#[derive(Clone, Debug, PartialEq)]
pub struct ThresholdCurve {
    /// Distinct estimate values, decreasing.
    pub threshold: Vec<f64>,
    /// Events with estimate `>=` the threshold.
    pub tp: Vec<f64>,
    /// Non-events with estimate `>=` the threshold.
    pub fp: Vec<f64>,
}

/// `binary_threshold_curve`: sort by estimate decreasing (stable), cumulate events
/// and non-events, and keep the values at the last row of each tie group.
pub fn binary_threshold_curve(is_event: &[bool], estimate: &[f64]) -> ThresholdCurve {
    binary_threshold_curve_weighted(is_event, estimate, None)
}

/// `binary_threshold_curve` with case weights: rows of zero weight are dropped and
/// the cumulative counts become cumulative weights.
pub fn binary_threshold_curve_weighted(
    is_event: &[bool],
    estimate: &[f64],
    w: Option<&[f64]>,
) -> ThresholdCurve {
    let mut idx: Vec<usize> = (0..estimate.len())
        .filter(|i| w.is_none_or(|w| w[*i] != 0.0))
        .collect();
    // R: order(estimate, decreasing = TRUE); ties keep their original order, which
    // does not affect the per-group cumulative sums used below.
    idx.sort_by(|a, b| estimate[*b].partial_cmp(&estimate[*a]).unwrap());
    let (mut threshold, mut tp, mut fp) = (Vec::new(), Vec::new(), Vec::new());
    let (mut ctp, mut cfp) = (0.0, 0.0);
    for (pos, &i) in idx.iter().enumerate() {
        let wi = w.map_or(1.0, |w| w[i]);
        if is_event[i] {
            ctp += wi;
        } else {
            cfp += wi;
        }
        let last_of_group = pos + 1 == idx.len() || estimate[idx[pos + 1]] != estimate[i];
        if last_of_group {
            threshold.push(estimate[i]);
            tp.push(ctp);
            fp.push(cfp);
        }
    }
    ThresholdCurve { threshold, tp, fp }
}

/// `auc(x, y)`: trapezoidal area under `y` over weakly increasing `x`.
pub fn trapezoid_auc(x: &[f64], y: &[f64]) -> Result<f64> {
    if x.len() != y.len() {
        return Err(Error::InvalidArgument("x and y must have the same length".into()));
    }
    if x.windows(2).any(|w| w[1] < w[0]) {
        return Err(Error::InvalidArgument(
            "x must already be in weakly increasing order".into(),
        ));
    }
    let mut auc = 0.0;
    for i in 1..x.len() {
        let dx = x[i] - x[i - 1];
        let height = (y[i - 1] + y[i]) / 2.0;
        auc += height * dx;
    }
    Ok(auc)
}

/// A binary ROC curve as `roc_curve()` returns it: thresholds increasing with
/// `-Inf`/`Inf` end points, sensitivity from 1 to 0, specificity from 0 to 1.
#[derive(Clone, Debug, PartialEq)]
pub struct RocCurve {
    /// `.threshold`
    pub threshold: Vec<f64>,
    /// `specificity`
    pub specificity: Vec<f64>,
    /// `sensitivity`
    pub sensitivity: Vec<f64>,
}

fn event_flags(truth: &[u32], event: u32) -> (Vec<bool>, usize, usize) {
    let flags: Vec<bool> = truth.iter().map(|t| *t == event).collect();
    let n_event = flags.iter().filter(|f| **f).count();
    (flags, n_event, truth.len() - n_event)
}

/// `roc_curve_binary`. Errors when truth has no event or no control, as R does.
pub fn roc_curve(truth: &[u32], prob_event: &[f64], event: u32) -> Result<RocCurve> {
    roc_curve_weighted(truth, prob_event, event, None)
}

/// `roc_curve` with optional case weights.
pub fn roc_curve_weighted(
    truth: &[u32],
    prob_event: &[f64],
    event: u32,
    w: Option<&[f64]>,
) -> Result<RocCurve> {
    check_prob(truth.len(), prob_event)?;
    let (flags, n_event, n_control) = event_flags(truth, event);
    if n_event == 0 {
        return Err(Error::InvalidValue(
            "no event observations were detected in truth".into(),
        ));
    }
    if n_control == 0 {
        return Err(Error::InvalidValue(
            "no control observations were detected in truth".into(),
        ));
    }
    let c = binary_threshold_curve_weighted(&flags, prob_event, w);
    let tp_last = c.tp[c.tp.len() - 1];
    let fp_last = c.fp[c.fp.len() - 1];
    let m = c.threshold.len();
    let mut threshold = Vec::with_capacity(m + 2);
    let mut sensitivity = Vec::with_capacity(m + 2);
    let mut specificity = Vec::with_capacity(m + 2);
    threshold.push(f64::NEG_INFINITY);
    sensitivity.push(1.0);
    specificity.push(0.0);
    for i in (0..m).rev() {
        threshold.push(c.threshold[i]);
        sensitivity.push(c.tp[i] / tp_last);
        specificity.push(1.0 - c.fp[i] / fp_last);
    }
    threshold.push(f64::INFINITY);
    sensitivity.push(0.0);
    specificity.push(1.0);
    Ok(RocCurve {
        threshold,
        specificity,
        sensitivity,
    })
}

/// `roc_auc` (binary, event first): trapezoid over the ROC curve. `None` when truth
/// has no event or no control (yardstick warns and returns `NA`).
pub fn roc_auc(truth: &[u32], prob_event: &[f64], event: u32) -> Result<Option<f64>> {
    roc_auc_weighted(truth, prob_event, event, None)
}

/// `roc_auc` with optional case weights.
pub fn roc_auc_weighted(
    truth: &[u32],
    prob_event: &[f64],
    event: u32,
    w: Option<&[f64]>,
) -> Result<Option<f64>> {
    check_prob(truth.len(), prob_event)?;
    let (_, n_event, n_control) = event_flags(truth, event);
    if n_event == 0 || n_control == 0 {
        return Ok(None);
    }
    let c = roc_curve_weighted(truth, prob_event, event, w)?;
    Ok(Some(trapezoid_auc(&c.specificity, &c.sensitivity)?))
}

/// A binary precision-recall curve as `pr_curve()` returns it: thresholds decreasing
/// starting at `Inf`, recall from 0, precision from 1.
#[derive(Clone, Debug, PartialEq)]
pub struct PrCurve {
    /// `.threshold`
    pub threshold: Vec<f64>,
    /// `recall`
    pub recall: Vec<f64>,
    /// `precision`
    pub precision: Vec<f64>,
}

/// `pr_curve_binary`. With no events the recall is `0/0` in R; here it is an error.
pub fn pr_curve(truth: &[u32], prob_event: &[f64], event: u32) -> Result<PrCurve> {
    pr_curve_weighted(truth, prob_event, event, None)
}

/// `pr_curve` with optional case weights.
pub fn pr_curve_weighted(
    truth: &[u32],
    prob_event: &[f64],
    event: u32,
    w: Option<&[f64]>,
) -> Result<PrCurve> {
    check_prob(truth.len(), prob_event)?;
    let (flags, n_event, _) = event_flags(truth, event);
    if n_event == 0 {
        return Err(Error::InvalidValue("there are 0 event cases in truth".into()));
    }
    let c = binary_threshold_curve_weighted(&flags, prob_event, w);
    let tp_last = c.tp[c.tp.len() - 1];
    let m = c.threshold.len();
    let mut threshold = Vec::with_capacity(m + 1);
    let mut recall = Vec::with_capacity(m + 1);
    let mut precision = Vec::with_capacity(m + 1);
    threshold.push(f64::INFINITY);
    recall.push(0.0);
    precision.push(1.0);
    for i in 0..m {
        threshold.push(c.threshold[i]);
        recall.push(c.tp[i] / tp_last);
        precision.push(c.tp[i] / (c.tp[i] + c.fp[i]));
    }
    Ok(PrCurve {
        threshold,
        recall,
        precision,
    })
}

/// `pr_auc` (binary, event first): trapezoid over the PR curve. `None` when truth has
/// no event (R: `NaN`).
pub fn pr_auc(truth: &[u32], prob_event: &[f64], event: u32) -> Result<Option<f64>> {
    pr_auc_weighted(truth, prob_event, event, None)
}

/// `pr_auc` with optional case weights.
pub fn pr_auc_weighted(
    truth: &[u32],
    prob_event: &[f64],
    event: u32,
    w: Option<&[f64]>,
) -> Result<Option<f64>> {
    check_prob(truth.len(), prob_event)?;
    if event_flags(truth, event).1 == 0 {
        return Ok(None);
    }
    let c = pr_curve_weighted(truth, prob_event, event, w)?;
    Ok(Some(trapezoid_auc(&c.recall, &c.precision)?))
}

/// `average_precision` (binary, event first): `sum(diff(recall) * precision[-1])`.
/// `None` when truth has no event (R: `NaN`).
pub fn average_precision(truth: &[u32], prob_event: &[f64], event: u32) -> Result<Option<f64>> {
    average_precision_weighted(truth, prob_event, event, None)
}

/// `average_precision` with optional case weights.
pub fn average_precision_weighted(
    truth: &[u32],
    prob_event: &[f64],
    event: u32,
    w: Option<&[f64]>,
) -> Result<Option<f64>> {
    check_prob(truth.len(), prob_event)?;
    if event_flags(truth, event).1 == 0 {
        return Ok(None);
    }
    let c = pr_curve_weighted(truth, prob_event, event, w)?;
    let mut s = 0.0;
    for i in 1..c.recall.len() {
        s += (c.recall[i] - c.recall[i - 1]) * c.precision[i];
    }
    Ok(Some(s))
}

/// `brier_class` (binary, event first): `brier_ind` on the one-hot truth and the
/// `(p, 1 - p)` matrix, i.e. `mean((y - p)^2 + ((1 - y) - (1 - p))^2) / 2`, written in
/// R's order of operations (uniform softmax case weights).
pub fn brier_class(truth: &[u32], prob_event: &[f64], event: u32) -> Result<Option<f64>> {
    brier_class_weighted(truth, prob_event, event, None)
}

/// `brier_ind`'s weighting: the case weights are passed through a softmax
/// (`exp(w - max(w)) / sum(exp(w - max(w)))`) before weighting the squared residuals.
fn brier_weights(n: usize, w: Option<&[f64]>) -> Vec<f64> {
    match w {
        None => vec![1.0 / n as f64; n],
        Some(w) => {
            let mx = w.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
            let e: Vec<f64> = w.iter().map(|v| (v - mx).exp()).collect();
            let s: f64 = e.iter().sum();
            e.iter().map(|v| v / s).collect()
        }
    }
}

/// `brier_class` with optional case weights.
pub fn brier_class_weighted(
    truth: &[u32],
    prob_event: &[f64],
    event: u32,
    w: Option<&[f64]>,
) -> Result<Option<f64>> {
    check_prob(truth.len(), prob_event)?;
    check_unit(prob_event)?;
    let bw = brier_weights(truth.len(), w);
    let mut s = 0.0;
    for ((t, p), wi) in truth.iter().zip(prob_event).zip(&bw) {
        let y = if *t == event { 1.0 } else { 0.0 };
        let r = (y - p) * (y - p) + ((1.0 - y) - (1.0 - p)) * ((1.0 - y) - (1.0 - p));
        s += r * wi;
    }
    Ok(Some(s / (2.0 * bw.iter().sum::<f64>())))
}

fn check_unit(p: &[f64]) -> Result<()> {
    if let Some(i) = p.iter().position(|p| *p < 0.0 || *p > 1.0) {
        return Err(Error::InvalidValue(format!(
            "probability out of [0, 1] at row {i}"
        )));
    }
    Ok(())
}

fn check_matrix(truth: &[u32], probs: &DesignMatrix, n_levels: usize) -> Result<()> {
    if probs.nrows() != truth.len() {
        return Err(Error::InvalidArgument(format!(
            "truth has {} rows but the probability matrix has {}",
            truth.len(),
            probs.nrows()
        )));
    }
    if probs.ncols() != n_levels {
        return Err(Error::InvalidArgument(format!(
            "the probability matrix has {} columns but truth has {n_levels} levels",
            probs.ncols()
        )));
    }
    if truth.is_empty() {
        return Err(Error::InvalidArgument("metrics need at least one row".into()));
    }
    if let Some(i) = truth.iter().position(|t| *t as usize >= n_levels) {
        return Err(Error::InvalidValue(format!("class code out of range at row {i}")));
    }
    if let Some(i) = probs.data().iter().position(|p| !p.is_finite()) {
        return Err(Error::InvalidValue(format!(
            "probability matrix is non-finite at element {i}"
        )));
    }
    Ok(())
}

/// Truth frequencies per level (`yardstick_truth_table`).
fn truth_table(truth: &[u32], n_levels: usize, w: Option<&[f64]>) -> Vec<f64> {
    let mut t = vec![0.0; n_levels];
    for (i, c) in truth.iter().enumerate() {
        t[*c as usize] += w.map_or(1.0, |w| w[i]);
    }
    t
}

fn one_vs_all_weights(truth: &[u32], n_levels: usize, est: Estimator, w: Option<&[f64]>) -> Result<Vec<f64>> {
    Ok(match est {
        Estimator::Macro => vec![1.0 / n_levels as f64; n_levels],
        Estimator::MacroWeighted => {
            let tt = truth_table(truth, n_levels, w);
            let s: f64 = tt.iter().sum();
            tt.iter().map(|c| c / s).collect()
        }
        other => {
            return Err(Error::InvalidArgument(format!(
                "estimator `{}` is not available for this probability metric",
                other.name()
            )))
        }
    })
}

/// A binary probability metric `(truth, prob_event, event)`.
type BinaryProbMetric = fn(&[u32], &[f64], u32, Option<&[f64]>) -> Result<Option<f64>>;

/// `one_vs_all_impl` + `weighted.mean(out_vec, w)` (no `na.rm`): apply a binary
/// probability metric per level with that level as the event and average.
fn one_vs_all(
    truth: &[u32],
    probs: &DesignMatrix,
    n_levels: usize,
    est: Estimator,
    f: BinaryProbMetric,
    w: Option<&[f64]>,
) -> Result<Option<f64>> {
    check_matrix(truth, probs, n_levels)?;
    let lw = one_vs_all_weights(truth, n_levels, est, w)?;
    let mut vals = Vec::with_capacity(n_levels);
    for lvl in 0..n_levels {
        vals.push(f(truth, probs.column(lvl), lvl as u32, w)?);
    }
    Ok(weighted_mean(&vals, &lw, false))
}

/// Multiclass `roc_auc`: `hand_till` (default), `macro` or `macro_weighted`. For
/// `hand_till`, levels absent from truth are ignored (yardstick warns); with fewer
/// than two remaining levels the result is `None`.
pub fn roc_auc_multiclass(truth: &[u32], probs: &DesignMatrix, est: Estimator) -> Result<Option<f64>> {
    roc_auc_multiclass_weighted(truth, probs, est, None)
}

/// `roc_auc_multiclass` with optional case weights; `hand_till` cannot take weights
/// ("Can't specify both `estimator = 'hand_till'` and `case_weights`").
pub fn roc_auc_multiclass_weighted(
    truth: &[u32],
    probs: &DesignMatrix,
    est: Estimator,
    w: Option<&[f64]>,
) -> Result<Option<f64>> {
    let k = probs.ncols();
    check_matrix(truth, probs, k)?;
    if est != Estimator::HandTill {
        return one_vs_all(truth, probs, k, est, roc_auc_weighted, w);
    }
    if w.is_some() {
        return Err(Error::InvalidArgument(
            "Can't specify both `estimator = 'hand_till'` and `case_weights`.".into(),
        ));
    }
    let present: Vec<usize> = truth_table(truth, k, None)
        .iter()
        .enumerate()
        .filter(|(_, c)| **c > 0.0)
        .map(|(i, _)| i)
        .collect();
    let c = present.len();
    if c < 2 {
        return Ok(None);
    }
    let multiplier = 2.0 / (c as f64 * (c as f64 - 1.0));
    let mut sum = 0.0;
    for (a, &i) in present.iter().enumerate() {
        for &j in &present[a + 1..] {
            let a_ij = roc_auc_subset(i, j, truth, probs)?;
            let a_ji = roc_auc_subset(j, i, truth, probs)?;
            sum += (a_ij + a_ji) / 2.0;
        }
    }
    Ok(Some(multiplier * sum))
}

/// `roc_auc_subset(lvl1, lvl2)`: binary AUC of `P(lvl1)` on the rows whose truth is
/// `lvl1` or `lvl2`, with `lvl1` as the event.
fn roc_auc_subset(lvl1: usize, lvl2: usize, truth: &[u32], probs: &DesignMatrix) -> Result<f64> {
    let col = probs.column(lvl1);
    let (mut t, mut p) = (Vec::new(), Vec::new());
    for (i, c) in truth.iter().enumerate() {
        let c = *c as usize;
        if c == lvl1 || c == lvl2 {
            t.push(if c == lvl1 { 0 } else { 1 });
            p.push(col[i]);
        }
    }
    roc_auc(&t, &p, 0).map(|v| v.expect("both levels present in the subset"))
}

/// Multiclass `pr_auc`: `macro` (default) or `macro_weighted` one-vs-all average.
pub fn pr_auc_multiclass(truth: &[u32], probs: &DesignMatrix, est: Estimator) -> Result<Option<f64>> {
    pr_auc_multiclass_weighted(truth, probs, est, None)
}

/// `pr_auc_multiclass` with optional case weights.
pub fn pr_auc_multiclass_weighted(
    truth: &[u32],
    probs: &DesignMatrix,
    est: Estimator,
    w: Option<&[f64]>,
) -> Result<Option<f64>> {
    one_vs_all(truth, probs, probs.ncols(), est, pr_auc_weighted, w)
}

/// Multiclass `average_precision`: `macro` (default) or `macro_weighted`.
pub fn average_precision_multiclass(
    truth: &[u32],
    probs: &DesignMatrix,
    est: Estimator,
) -> Result<Option<f64>> {
    average_precision_multiclass_weighted(truth, probs, est, None)
}

/// `average_precision_multiclass` with optional case weights.
pub fn average_precision_multiclass_weighted(
    truth: &[u32],
    probs: &DesignMatrix,
    est: Estimator,
    w: Option<&[f64]>,
) -> Result<Option<f64>> {
    one_vs_all(truth, probs, probs.ncols(), est, average_precision_weighted, w)
}

/// Multiclass `brier_class`: `mean(rowSums((onehot - probs)^2)) / 2` in R's order of
/// operations.
pub fn brier_class_multiclass(truth: &[u32], probs: &DesignMatrix) -> Result<Option<f64>> {
    brier_class_multiclass_weighted(truth, probs, None)
}

/// `brier_class_multiclass` with optional case weights (softmax-normalised as in
/// `brier_ind`).
pub fn brier_class_multiclass_weighted(
    truth: &[u32],
    probs: &DesignMatrix,
    w: Option<&[f64]>,
) -> Result<Option<f64>> {
    let k = probs.ncols();
    check_matrix(truth, probs, k)?;
    check_unit(probs.data())?;
    let bw = brier_weights(truth.len(), w);
    let mut s = 0.0;
    for ((i, t), wi) in truth.iter().enumerate().zip(&bw) {
        let mut r = 0.0;
        for j in 0..k {
            let y = if *t as usize == j { 1.0 } else { 0.0 };
            let d = y - probs.get(i, j);
            r += d * d;
        }
        s += r * wi;
    }
    Ok(Some(s / (2.0 * bw.iter().sum::<f64>())))
}

/// Multiclass `mn_log_loss`: `-mean(log(p[truth]))` with `p` clamped to
/// `[eps, 1 - eps]`, `eps = .Machine$double.eps`.
pub fn mn_log_loss_multiclass(truth: &[u32], probs: &DesignMatrix) -> Result<Option<f64>> {
    mn_log_loss_multiclass_weighted(truth, probs, None)
}

/// `mn_log_loss_multiclass` with optional case weights.
pub fn mn_log_loss_multiclass_weighted(
    truth: &[u32],
    probs: &DesignMatrix,
    w: Option<&[f64]>,
) -> Result<Option<f64>> {
    let k = probs.ncols();
    check_matrix(truth, probs, k)?;
    check_unit(probs.data())?;
    let eps = f64::EPSILON;
    let loss: Vec<f64> = truth
        .iter()
        .enumerate()
        .map(|(i, t)| -probs.get(i, *t as usize).clamp(eps, 1.0 - eps).ln())
        .collect();
    Ok(Some(crate::yardstick_mean(&loss, w)))
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;

    #[test]
    fn threshold_curve_groups_ties_at_last_row() {
        // estimate 0.9 (event), 0.5 (event), 0.5 (control), 0.1 (control)
        let c = binary_threshold_curve(&[true, true, false, false], &[0.9, 0.5, 0.5, 0.1]);
        assert_eq!(c.threshold, vec![0.9, 0.5, 0.1]);
        assert_eq!(c.tp, vec![1.0, 2.0, 2.0]);
        assert_eq!(c.fp, vec![0.0, 1.0, 2.0]);
    }

    #[test]
    fn roc_auc_is_mann_whitney_with_half_credit_for_ties() {
        let t = [0, 0, 1, 1];
        let p = [0.9, 0.5, 0.5, 0.1];
        // pairs (event, control): (0.9,0.5)=1, (0.9,0.1)=1, (0.5,0.5)=0.5, (0.5,0.1)=1 -> 3.5/4
        assert_abs_diff_eq!(roc_auc(&t, &p, 0).unwrap().unwrap(), 0.875, epsilon = 1e-15);
        assert_eq!(roc_auc(&[1, 1], &[0.2, 0.3], 0).unwrap(), None);
        let c = roc_curve(&t, &p, 0).unwrap();
        assert_eq!(c.threshold.len(), 5);
        assert_eq!(c.sensitivity, vec![1.0, 1.0, 1.0, 0.5, 0.0]);
        assert_eq!(c.specificity, vec![0.0, 0.0, 0.5, 1.0, 1.0]);
    }

    #[test]
    fn pr_metrics_hand_computed() {
        let t = [0, 0, 1, 1];
        let p = [0.9, 0.5, 0.5, 0.1];
        let c = pr_curve(&t, &p, 0).unwrap();
        assert_eq!(c.recall, vec![0.0, 0.5, 1.0, 1.0]);
        assert_eq!(c.precision, vec![1.0, 1.0, 2.0 / 3.0, 0.5]);
        // trapezoid: 0.5*(1+1)/2 + 0.5*(1+2/3)/2 + 0 = 0.5 + 5/12
        assert_abs_diff_eq!(
            pr_auc(&t, &p, 0).unwrap().unwrap(),
            0.5 + 5.0 / 12.0,
            epsilon = 1e-15
        );
        // average precision: 0.5*1 + 0.5*(2/3) + 0*0.5
        assert_abs_diff_eq!(
            average_precision(&t, &p, 0).unwrap().unwrap(),
            0.5 + 1.0 / 3.0,
            epsilon = 1e-15
        );
        assert_eq!(pr_auc(&[1, 1], &[0.2, 0.3], 0).unwrap(), None);
    }

    #[test]
    fn brier_binary_equals_mean_squared_error_of_event_probability() {
        let t = [0, 1, 0];
        let p = [0.8, 0.3, 0.4];
        let expect = (0.04 + 0.09 + 0.36) / 3.0;
        assert_abs_diff_eq!(brier_class(&t, &p, 0).unwrap().unwrap(), expect, epsilon = 1e-15);
    }

    #[test]
    fn multiclass_probability_metrics() {
        let truth = [0u32, 1, 2, 1];
        let probs = DesignMatrix::from_row_major(
            &[0.7, 0.2, 0.1, 0.2, 0.5, 0.3, 0.1, 0.3, 0.6, 0.4, 0.4, 0.2],
            4,
            3,
            vec!["a".into(), "b".into(), "c".into()],
        )
        .unwrap();
        // hand_till: all pairwise subset AUCs are 1 here -> 1
        assert_abs_diff_eq!(
            roc_auc_multiclass(&truth, &probs, Estimator::HandTill)
                .unwrap()
                .unwrap(),
            1.0
        );
        let macro_ = roc_auc_multiclass(&truth, &probs, Estimator::Macro)
            .unwrap()
            .unwrap();
        assert_abs_diff_eq!(macro_, 1.0);
        let ll = mn_log_loss_multiclass(&truth, &probs).unwrap().unwrap();
        assert_abs_diff_eq!(
            ll,
            -(0.7f64.ln() + 0.5f64.ln() + 0.6f64.ln() + 0.4f64.ln()) / 4.0,
            epsilon = 1e-15
        );
        let br = brier_class_multiclass(&truth, &probs).unwrap().unwrap();
        let rows = [
            (0.3f64).powi(2) + 0.04 + 0.01,
            0.04 + 0.25 + 0.09,
            0.01 + 0.09 + 0.16,
            0.16 + 0.36 + 0.04,
        ];
        assert_abs_diff_eq!(br, rows.iter().sum::<f64>() / 4.0 / 2.0, epsilon = 1e-15);
        assert!(pr_auc_multiclass(&truth, &probs, Estimator::Micro).is_err());
    }
}
