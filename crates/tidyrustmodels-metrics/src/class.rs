//! Confusion-table metrics with yardstick 1.4.0 semantics.
//!
//! The table has the orientation of `yardstick_table()`: **rows are predictions,
//! columns are truth** (`table(Prediction = estimate, Truth = truth)`). Every formula
//! below is a transcription of the corresponding `*_binary` / `*_multiclass` /
//! `*_table_impl` body in yardstick (read from the installed package; see
//! VERIFICATION.md Phase 8b), including which denominators yield `NA` and whether the
//! multiclass average drops those `NA`s (`weighted.mean(..., na.rm = TRUE)`) or
//! propagates them.
//!
//! Binary metrics use the **first level as the event** (`event_level = "first"`,
//! ADR-0004).

use serde::{Deserialize, Serialize};
use tidyrustmodels_core::{Error, Result};

/// How a class metric is averaged across levels (yardstick's `estimator`).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Estimator {
    /// Two levels; the first level is the event.
    Binary,
    /// One-vs-all per level, averaged with equal weights.
    Macro,
    /// One-vs-all per level, averaged with weights proportional to the level's truth
    /// frequency.
    MacroWeighted,
    /// Numerators and denominators summed over levels before dividing.
    Micro,
    /// Hand & Till (2001) pairwise average; `roc_auc` only.
    HandTill,
}

impl Estimator {
    /// yardstick's name.
    pub fn name(self) -> &'static str {
        match self {
            Estimator::Binary => "binary",
            Estimator::Macro => "macro",
            Estimator::MacroWeighted => "macro_weighted",
            Estimator::Micro => "micro",
            Estimator::HandTill => "hand_till",
        }
    }

    /// Parse yardstick's name.
    pub fn from_name(name: &str) -> Result<Estimator> {
        Ok(match name {
            "binary" => Estimator::Binary,
            "macro" => Estimator::Macro,
            "macro_weighted" => Estimator::MacroWeighted,
            "micro" => Estimator::Micro,
            "hand_till" => Estimator::HandTill,
            other => return Err(Error::InvalidArgument(format!("unknown estimator `{other}`"))),
        })
    }

    /// yardstick's default for a class metric: `binary` for two levels, else `macro`
    /// (`finalize_estimator_default.factor`).
    pub fn default_for(n_levels: usize) -> Estimator {
        if n_levels == 2 {
            Estimator::Binary
        } else {
            Estimator::Macro
        }
    }
}

/// Weighting for Cohen's kappa (`kap(weighting = )`).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum KapWeighting {
    /// Unweighted: every disagreement counts 1.
    None,
    /// `|i - j|` for levels `i`, `j` in level order.
    Linear,
    /// `(i - j)^2`.
    Quadratic,
}

/// A confusion matrix in yardstick's orientation (rows = prediction, columns = truth),
/// stored as `f64` counts like `yardstick_table()`.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct ConfusionMatrix {
    levels: Vec<String>,
    /// Row-major `k × k`: `counts[pred * k + truth]`.
    counts: Vec<f64>,
}

impl ConfusionMatrix {
    /// Cross-tabulate coded truth and estimate over `levels` (codes index `levels`).
    /// At least two levels are required, as in yardstick.
    pub fn from_codes(truth: &[u32], estimate: &[u32], levels: &[String]) -> Result<Self> {
        Self::from_codes_weighted(truth, estimate, levels, None)
    }

    /// `hardhat::weighted_table`: each cell is the sum of the case weights of its rows.
    pub fn from_codes_weighted(
        truth: &[u32],
        estimate: &[u32],
        levels: &[String],
        w: Option<&[f64]>,
    ) -> Result<Self> {
        let k = levels.len();
        if let Some(w) = w {
            if w.len() != truth.len() {
                return Err(Error::InvalidArgument(format!(
                    "truth has {} rows but case_weights has {}",
                    truth.len(),
                    w.len()
                )));
            }
        }
        if k < 2 {
            return Err(Error::InvalidArgument("truth must have at least 2 levels".into()));
        }
        if truth.len() != estimate.len() {
            return Err(Error::InvalidArgument(format!(
                "truth has {} rows but estimate has {}",
                truth.len(),
                estimate.len()
            )));
        }
        let mut counts = vec![0.0; k * k];
        for (i, (t, e)) in truth.iter().zip(estimate).enumerate() {
            let (t, e) = (*t as usize, *e as usize);
            if t >= k || e >= k {
                return Err(Error::InvalidValue(format!(
                    "class code out of range at row {i} (levels: {k})"
                )));
            }
            counts[e * k + t] += w.map_or(1.0, |w| w[i]);
        }
        Ok(ConfusionMatrix {
            levels: levels.to_vec(),
            counts,
        })
    }

    /// Build from explicit counts (row-major, rows = prediction).
    pub fn from_counts(levels: Vec<String>, counts: Vec<f64>) -> Result<Self> {
        let k = levels.len();
        if k < 2 || counts.len() != k * k {
            return Err(Error::InvalidArgument(format!(
                "need k >= 2 levels and k*k counts; got {k} levels and {} counts",
                counts.len()
            )));
        }
        if counts.iter().any(|c| !c.is_finite() || *c < 0.0) {
            return Err(Error::InvalidValue("counts must be finite and >= 0".into()));
        }
        Ok(ConfusionMatrix { levels, counts })
    }

    /// Level labels in order (first = event for binary metrics).
    pub fn levels(&self) -> &[String] {
        &self.levels
    }

    /// Number of levels.
    pub fn n_levels(&self) -> usize {
        self.levels.len()
    }

    /// Count of rows predicted as `pred` whose truth is `truth`.
    pub fn count(&self, pred: usize, truth: usize) -> f64 {
        self.counts[pred * self.levels.len() + truth]
    }

    /// Row-major counts (`[pred][truth]`).
    pub fn counts(&self) -> &[f64] {
        &self.counts
    }

    /// Total count.
    pub fn total(&self) -> f64 {
        self.counts.iter().sum()
    }

    /// Diagonal (correct predictions per level).
    pub fn diag(&self) -> Vec<f64> {
        (0..self.n_levels()).map(|i| self.count(i, i)).collect()
    }

    /// Row sums: number predicted as each level (`rowSums`).
    pub fn pred_totals(&self) -> Vec<f64> {
        let k = self.n_levels();
        (0..k).map(|p| (0..k).map(|t| self.count(p, t)).sum()).collect()
    }

    /// Column sums: number truly in each level (`colSums`).
    pub fn truth_totals(&self) -> Vec<f64> {
        let k = self.n_levels();
        (0..k).map(|t| (0..k).map(|p| self.count(p, t)).sum()).collect()
    }

    fn check_table_estimator(&self, est: Estimator) -> Result<()> {
        match est {
            Estimator::Binary if self.n_levels() != 2 => Err(Error::InvalidArgument(format!(
                "estimator is binary, only two class truth factors are allowed; {} levels were provided",
                self.n_levels()
            ))),
            Estimator::HandTill => Err(Error::InvalidArgument(
                "estimator `hand_till` applies to roc_auc only".into(),
            )),
            _ => Ok(()),
        }
    }

    /// `get_weights()`: averaging weights per level.
    fn weights(&self, est: Estimator) -> Vec<f64> {
        let k = self.n_levels();
        match est {
            Estimator::MacroWeighted => {
                let tt = self.truth_totals();
                let s: f64 = tt.iter().sum();
                tt.iter().map(|c| c / s).collect()
            }
            _ => vec![1.0 / k as f64; k],
        }
    }

    /// `sum(diag(x)) / sum(x)`; `None` for an empty table.
    pub fn accuracy(&self) -> Option<f64> {
        let n = self.total();
        if n == 0.0 {
            None
        } else {
            Some(self.diag().iter().sum::<f64>() / n)
        }
    }

    // ---- per-level numerator/denominator vectors (yardstick `*_multiclass`) ----

    /// Precision per level: `diag / rowSums`, `None` where nothing was predicted as
    /// that level.
    fn precision_classes(&self) -> Vec<Option<f64>> {
        ratio_classes(&self.diag(), &self.pred_totals())
    }

    /// Recall / sensitivity per level: `diag / colSums`, `None` where the level never
    /// occurs in truth.
    fn recall_classes(&self) -> Vec<Option<f64>> {
        ratio_classes(&self.diag(), &self.truth_totals())
    }

    /// Specificity per level: `tn / (tn + fp)` with
    /// `tn = n - (rowSum + colSum - diag)`, `fp = rowSum - diag`.
    fn spec_classes(&self) -> Vec<Option<f64>> {
        let n = self.total();
        let tp = self.diag();
        let tpfp = self.pred_totals();
        let tpfn = self.truth_totals();
        let mut numer = Vec::with_capacity(tp.len());
        let mut denom = Vec::with_capacity(tp.len());
        for i in 0..tp.len() {
            let tn = n - (tpfp[i] + tpfn[i] - tp[i]);
            let fp = tpfp[i] - tp[i];
            numer.push(tn);
            denom.push(tn + fp);
        }
        ratio_classes(&numer, &denom)
    }

    /// Apply yardstick's micro rule to per-level (numer, denom) pairs: sum the defined
    /// numerators and denominators, then divide once.
    fn micro(&self, numer: &[f64], denom: &[f64]) -> Option<f64> {
        let (mut sn, mut sd) = (0.0, 0.0);
        for (n, d) in numer.iter().zip(denom) {
            if *d > 0.0 {
                sn += n;
                sd += d;
            }
        }
        finite_ratio(sn, sd)
    }

    fn precision_est(&self, est: Estimator) -> Vec<Option<f64>> {
        if est == Estimator::Micro {
            vec![self.micro(&self.diag(), &self.pred_totals())]
        } else {
            self.precision_classes()
        }
    }

    fn recall_est(&self, est: Estimator) -> Vec<Option<f64>> {
        if est == Estimator::Micro {
            vec![self.micro(&self.diag(), &self.truth_totals())]
        } else {
            self.recall_classes()
        }
    }

    fn spec_est(&self, est: Estimator) -> Vec<Option<f64>> {
        if est == Estimator::Micro {
            let n = self.total();
            let tp = self.diag();
            let tpfp = self.pred_totals();
            let tpfn = self.truth_totals();
            let numer: Vec<f64> = (0..tp.len()).map(|i| n - (tpfp[i] + tpfn[i] - tp[i])).collect();
            let denom: Vec<f64> = (0..tp.len()).map(|i| numer[i] + tpfp[i] - tp[i]).collect();
            vec![self.micro(&numer, &denom)]
        } else {
            self.spec_classes()
        }
    }

    /// Average per-level values as `*_table_impl` does: binary takes the event level,
    /// micro is already a scalar, macro/macro_weighted use `stats::weighted.mean`
    /// (with or without `na.rm = TRUE`, as the metric specifies).
    fn average(&self, est: Estimator, values: &[Option<f64>], na_rm: bool) -> Option<f64> {
        match est {
            Estimator::Binary => values[0],
            Estimator::Micro => values[0],
            _ => weighted_mean(values, &self.weights(est), na_rm),
        }
    }

    /// `sens` / `recall` (identical in yardstick): `TP / (TP + FN)` for the event level,
    /// or the `estimator` average of the per-level recalls (`na.rm = TRUE`).
    pub fn sens(&self, est: Estimator) -> Result<Option<f64>> {
        self.check_table_estimator(est)?;
        Ok(self.average(est, &self.recall_est(est), true))
    }

    /// Alias of [`ConfusionMatrix::sens`].
    pub fn recall(&self, est: Estimator) -> Result<Option<f64>> {
        self.sens(est)
    }

    /// `spec`: `TN / (TN + FP)` for the event level (binary: the second level's
    /// recall), or the `estimator` average (`na.rm = TRUE`).
    pub fn spec(&self, est: Estimator) -> Result<Option<f64>> {
        self.check_table_estimator(est)?;
        if est == Estimator::Binary {
            // spec_binary: numer = data[neg, neg]; denom = sum(data[, neg]).
            return Ok(self.recall_classes()[1]);
        }
        Ok(self.average(est, &self.spec_est(est), true))
    }

    /// `precision`: `TP / (TP + FP)` for the event level, or the `estimator` average
    /// (`na.rm = TRUE`).
    pub fn precision(&self, est: Estimator) -> Result<Option<f64>> {
        self.check_table_estimator(est)?;
        Ok(self.average(est, &self.precision_est(est), true))
    }

    /// `f_meas(beta)`: `(1 + β²) · P · R / (β² · P + R)` per level, `0` where both
    /// precision and recall are zero, `NA` where either is undefined; averaged with
    /// `na.rm = TRUE`.
    pub fn f_meas(&self, est: Estimator, beta: f64) -> Result<Option<f64>> {
        self.check_table_estimator(est)?;
        if beta.is_nan() || beta < 0.0 {
            return Err(Error::InvalidArgument("beta must be a number >= 0".into()));
        }
        let p = self.precision_est(est);
        let r = self.recall_est(est);
        let f: Vec<Option<f64>> = p
            .iter()
            .zip(&r)
            .map(|(p, r)| match (p, r) {
                (Some(p), Some(r)) => {
                    if *p == 0.0 && *r == 0.0 {
                        Some(0.0)
                    } else {
                        Some((1.0 + beta * beta) * p * r / ((beta * beta * p) + r))
                    }
                }
                _ => None,
            })
            .collect();
        Ok(self.average(est, &f, true))
    }

    /// `bal_accuracy`: `(sens + spec) / 2`. Multiclass: per level, then averaged
    /// **without** `na.rm` (`bal_accuracy_table_impl`), so one undefined level gives `NA`.
    pub fn bal_accuracy(&self, est: Estimator) -> Result<Option<f64>> {
        self.check_table_estimator(est)?;
        if est == Estimator::Binary {
            return Ok(match (self.sens(est)?, self.spec(est)?) {
                (Some(s), Some(p)) => Some((s + p) / 2.0),
                _ => None,
            });
        }
        let r = self.recall_est(est);
        let s = self.spec_est(est);
        let v: Vec<Option<f64>> = r
            .iter()
            .zip(&s)
            .map(|(r, s)| match (r, s) {
                (Some(r), Some(s)) => Some((r + s) / 2.0),
                _ => None,
            })
            .collect();
        Ok(self.average(est, &v, false))
    }

    /// `j_index` (Youden's J): `sens + spec - 1`; multiclass averaged with `na.rm = TRUE`.
    pub fn j_index(&self, est: Estimator) -> Result<Option<f64>> {
        self.check_table_estimator(est)?;
        if est == Estimator::Binary {
            return Ok(match (self.sens(est)?, self.spec(est)?) {
                (Some(s), Some(p)) => Some(s + p - 1.0),
                _ => None,
            });
        }
        let r = self.recall_est(est);
        let s = self.spec_est(est);
        let v: Vec<Option<f64>> = r
            .iter()
            .zip(&s)
            .map(|(r, s)| match (r, s) {
                (Some(r), Some(s)) => Some(r + s - 1.0),
                _ => None,
            })
            .collect();
        Ok(self.average(est, &v, true))
    }

    /// Cohen's `kap`: `1 - sum(w * data) / sum(w * expected)` with
    /// `expected = outer(rowSums, colSums) / n` and the weighting matrix of
    /// `make_weighting_matrix`. `None` when the chance disagreement is zero (R: `NaN`).
    pub fn kap(&self, weighting: KapWeighting) -> Option<f64> {
        let n = self.total();
        let rs = self.pred_totals();
        let cs = self.truth_totals();
        let w = |i: usize, j: usize| -> f64 {
            let d = (i as f64 - j as f64).abs();
            match weighting {
                KapWeighting::None => {
                    if i == j {
                        0.0
                    } else {
                        1.0
                    }
                }
                KapWeighting::Linear => d,
                KapWeighting::Quadratic => d * d,
            }
        };
        let (mut disagree, mut chance) = (0.0, 0.0);
        for (i, ri) in rs.iter().enumerate() {
            for (j, cj) in cs.iter().enumerate() {
                disagree += w(i, j) * self.count(i, j);
                chance += w(i, j) * (ri * cj / n);
            }
        }
        finite_ratio(disagree, chance).map(|r| 1.0 - r)
    }

    /// Matthews correlation coefficient. Binary (`mcc_binary`): `NA` when any of
    /// `TP+FP`, `TP+FN`, `TN+FP`, `TN+FN` is zero. Multiclass: the Gorodkin (2004)
    /// generalisation used by yardstick's C implementation; `None` when its
    /// denominator is zero.
    pub fn mcc(&self) -> Option<f64> {
        let k = self.n_levels();
        if k == 2 {
            let (tp, tn) = (self.count(0, 0), self.count(1, 1));
            let (fp, fn_) = (self.count(0, 1), self.count(1, 0));
            let (d1, d2, d3, d4) = (tp + fp, tp + fn_, tn + fp, tn + fn_);
            if d1 == 0.0 || d2 == 0.0 || d3 == 0.0 || d4 == 0.0 {
                return None;
            }
            return Some((tp * tn - fp * fn_) / (d1 * d2 * d3 * d4).sqrt());
        }
        let n = self.total();
        let c: f64 = self.diag().iter().sum();
        let ps = self.pred_totals();
        let ts = self.truth_totals();
        let pt: f64 = ps.iter().zip(&ts).map(|(p, t)| p * t).sum();
        let pp: f64 = ps.iter().map(|p| p * p).sum();
        let tt: f64 = ts.iter().map(|t| t * t).sum();
        let numer = c * n - pt;
        let denom = ((n * n - pp) * (n * n - tt)).sqrt();
        finite_ratio(numer, denom)
    }
}

fn ratio_classes(numer: &[f64], denom: &[f64]) -> Vec<Option<f64>> {
    numer
        .iter()
        .zip(denom)
        .map(|(n, d)| if *d <= 0.0 { None } else { Some(n / d) })
        .collect()
}

fn finite_ratio(numer: f64, denom: f64) -> Option<f64> {
    if denom == 0.0 {
        None
    } else {
        Some(numer / denom)
    }
}

/// `stats::weighted.mean(x, w, na.rm)`: with `na.rm`, drop `NA` entries and their
/// weights; then `sum((x * w)[w != 0]) / sum(w)`, so an `NA` with **zero weight** is
/// ignored (an absent level under `macro_weighted`) while any other `NA` gives `NA`.
/// All-`NA` (or zero total weight) is `None`.
pub fn weighted_mean(values: &[Option<f64>], w: &[f64], na_rm: bool) -> Option<f64> {
    let (mut s, mut sw) = (0.0, 0.0);
    for (v, w) in values.iter().zip(w) {
        match v {
            Some(v) => {
                if *w != 0.0 {
                    s += v * w;
                }
                sw += w;
            }
            None => {
                if na_rm || *w == 0.0 {
                    continue;
                }
                return None;
            }
        }
    }
    finite_ratio(s, sw)
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;

    fn lv(n: usize) -> Vec<String> {
        (0..n).map(|i| format!("l{i}")).collect()
    }

    #[test]
    fn binary_table_metrics_hand_computed() {
        // truth a,a,b,b,a,b ; estimate a,b,b,b,a,a  (event = a = 0)
        let t = [0, 0, 1, 1, 0, 1];
        let e = [0, 1, 1, 1, 0, 0];
        let cm = ConfusionMatrix::from_codes(&t, &e, &lv(2)).unwrap();
        // rows = pred, cols = truth: [a,a]=2 [a,b]=1 [b,a]=1 [b,b]=2
        assert_eq!(cm.counts(), &[2.0, 1.0, 1.0, 2.0]);
        let b = Estimator::Binary;
        assert_abs_diff_eq!(cm.sens(b).unwrap().unwrap(), 2.0 / 3.0);
        assert_abs_diff_eq!(cm.spec(b).unwrap().unwrap(), 2.0 / 3.0);
        assert_abs_diff_eq!(cm.precision(b).unwrap().unwrap(), 2.0 / 3.0);
        assert_abs_diff_eq!(cm.f_meas(b, 1.0).unwrap().unwrap(), 2.0 / 3.0);
        assert_abs_diff_eq!(cm.bal_accuracy(b).unwrap().unwrap(), 2.0 / 3.0);
        assert_abs_diff_eq!(cm.j_index(b).unwrap().unwrap(), 1.0 / 3.0);
        // kappa: po = 4/6, pe = (3*3 + 3*3)/36 = 0.5 -> (2/3 - 1/2)/(1/2) = 1/3
        assert_abs_diff_eq!(cm.kap(KapWeighting::None).unwrap(), 1.0 / 3.0, epsilon = 1e-15);
        // mcc = (4 - 1)/sqrt(3*3*3*3) = 1/3
        assert_abs_diff_eq!(cm.mcc().unwrap(), 1.0 / 3.0, epsilon = 1e-15);
        assert_eq!(cm.accuracy(), Some(4.0 / 6.0));
    }

    #[test]
    fn binary_undefined_cases_are_none() {
        // nothing predicted as the event -> precision NA; truth has no event -> sens NA
        let cm = ConfusionMatrix::from_counts(lv(2), vec![0.0, 0.0, 0.0, 2.0]).unwrap();
        let b = Estimator::Binary;
        assert_eq!(cm.precision(b).unwrap(), None);
        assert_eq!(cm.sens(b).unwrap(), None);
        assert_eq!(cm.spec(b).unwrap(), Some(1.0));
        assert_eq!(cm.f_meas(b, 1.0).unwrap(), None);
        assert_eq!(cm.bal_accuracy(b).unwrap(), None);
        assert_eq!(cm.mcc(), None);
        assert_eq!(cm.kap(KapWeighting::None), None);
        assert!(cm.sens(Estimator::HandTill).is_err());
        assert!(ConfusionMatrix::from_counts(lv(3), vec![0.0; 9])
            .unwrap()
            .sens(b)
            .is_err());
    }

    #[test]
    fn f_meas_both_zero_is_zero() {
        // event predicted once, wrongly; event occurs once, missed -> P = 0, R = 0
        let cm = ConfusionMatrix::from_counts(lv(2), vec![0.0, 1.0, 1.0, 1.0]).unwrap();
        assert_eq!(cm.f_meas(Estimator::Binary, 1.0).unwrap(), Some(0.0));
    }

    #[test]
    fn multiclass_averages_follow_yardstick() {
        // 3 levels; level 2 never predicted -> precision NA for it (dropped by na.rm)
        // rows = pred, cols = truth
        let cm = ConfusionMatrix::from_counts(
            lv(3),
            vec![
                3.0, 1.0, 1.0, //
                0.0, 2.0, 1.0, //
                0.0, 0.0, 0.0,
            ],
        )
        .unwrap();
        let prec = cm.precision_classes();
        assert_eq!(prec, vec![Some(0.6), Some(2.0 / 3.0), None]);
        // macro drops the NA: mean(0.6, 2/3)
        assert_abs_diff_eq!(
            cm.precision(Estimator::Macro).unwrap().unwrap(),
            (0.6 + 2.0 / 3.0) / 2.0,
            epsilon = 1e-15
        );
        // macro_weighted: weights colSums/n = 3/8, 3/8, 2/8 ; NA dropped and renormalised
        let w = (0.6 * 3.0 / 8.0 + (2.0 / 3.0) * 3.0 / 8.0) / (6.0 / 8.0);
        assert_abs_diff_eq!(
            cm.precision(Estimator::MacroWeighted).unwrap().unwrap(),
            w,
            epsilon = 1e-15
        );
        // micro: sum diag / sum defined denominators = 5 / 8
        assert_abs_diff_eq!(cm.precision(Estimator::Micro).unwrap().unwrap(), 5.0 / 8.0);
        // recall per class: 3/3, 2/3, 0/2
        assert_abs_diff_eq!(
            cm.sens(Estimator::Macro).unwrap().unwrap(),
            (1.0 + 2.0 / 3.0) / 3.0,
            epsilon = 1e-15
        );
        // bal_accuracy propagates NA? recall and spec are all defined here -> a value
        assert!(cm.bal_accuracy(Estimator::Macro).unwrap().is_some());
        // j_index macro = mean(sens + spec - 1)
        let spec = cm.spec_classes();
        let j: f64 = (0..3)
            .map(|i| cm.recall_classes()[i].unwrap() + spec[i].unwrap() - 1.0)
            .sum::<f64>()
            / 3.0;
        assert_abs_diff_eq!(cm.j_index(Estimator::Macro).unwrap().unwrap(), j, epsilon = 1e-15);
    }

    #[test]
    fn weighted_mean_semantics() {
        assert_eq!(weighted_mean(&[Some(1.0), None], &[0.5, 0.5], false), None);
        assert_eq!(weighted_mean(&[Some(1.0), None], &[0.5, 0.5], true), Some(1.0));
        assert_eq!(weighted_mean(&[None, None], &[0.5, 0.5], true), None);
        // R: sum((x * w)[w != 0]) / sum(w) -> a zero-weight NA is ignored
        assert_eq!(weighted_mean(&[Some(1.0), None], &[1.0, 0.0], false), Some(1.0));
    }
}
