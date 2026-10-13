//! Built-in preprocessing steps.

mod dummy;
mod filter;
mod impute;
mod levels;
mod normalize;
mod select;

pub use dummy::{FittedDummy, SparseOutput, StepDummy};
pub use filter::{FittedCorr, FittedRemoveCols, StepCorr, StepNzv, StepZv};
pub use impute::{FittedImpute, ImputeStat, StepImputeMean, StepImputeMedian};
pub use levels::{FittedLevelAdd, FittedOther, OtherObject, StepNovel, StepOther, StepUnknown};
pub use normalize::{FittedNormalize, StepNormalize};
pub use select::{FittedSelect, StepSelect};

use tidyrustmodels_core::{Error, FittedStep, Result, Role, Schema, Table, WeightKind};

/// Rebuild a built-in fitted step from its serialized `state()`.
pub fn fitted_step_from_state(value: &serde_json::Value) -> Result<Box<dyn FittedStep>> {
    let tag = value
        .get("step")
        .and_then(|v| v.as_str())
        .ok_or_else(|| Error::Serialization("fitted step state lacks a `step` tag".into()))?;
    let de = |msg: serde_json::Error| Error::Serialization(format!("step `{tag}`: {msg}"));
    Ok(match tag {
        "normalize" => Box::new(serde_json::from_value::<FittedNormalize>(value.clone()).map_err(de)?),
        "impute_mean" | "impute_median" => {
            Box::new(serde_json::from_value::<FittedImpute>(value.clone()).map_err(de)?)
        }
        "dummy" => Box::new(serde_json::from_value::<FittedDummy>(value.clone()).map_err(de)?),
        "select" => Box::new(serde_json::from_value::<FittedSelect>(value.clone()).map_err(de)?),
        "zv" | "nzv" => Box::new(serde_json::from_value::<FittedRemoveCols>(value.clone()).map_err(de)?),
        "corr" => Box::new(serde_json::from_value::<FittedCorr>(value.clone()).map_err(de)?),
        "other" => Box::new(serde_json::from_value::<FittedOther>(value.clone()).map_err(de)?),
        "novel" | "unknown" => Box::new(serde_json::from_value::<FittedLevelAdd>(value.clone()).map_err(de)?),
        other => return Err(Error::Serialization(format!("unknown built-in step `{other}`"))),
    })
}

/// Shared helper: the mean of the valid entries of a numeric column (`colMeans(na.rm = TRUE)`).
pub(crate) fn mean_valid(values: &[f64], valid: Option<&[bool]>) -> Option<f64> {
    let mut sum = 0.0;
    let mut n = 0usize;
    for (i, v) in values.iter().enumerate() {
        if valid.is_none_or(|m| m[i]) {
            sum += v;
            n += 1;
        }
    }
    if n == 0 {
        None
    } else {
        Some(sum / n as f64)
    }
}

/// Shared helper: sample variance with n-1 denominator over valid entries
/// (`stats::var(na.rm = TRUE)`). Two-pass for numerical stability.
pub(crate) fn var_valid(values: &[f64], valid: Option<&[bool]>) -> Option<f64> {
    let mean = mean_valid(values, valid)?;
    let mut ss = 0.0;
    let mut n = 0usize;
    for (i, v) in values.iter().enumerate() {
        if valid.is_none_or(|m| m[i]) {
            let d = v - mean;
            ss += d * d;
            n += 1;
        }
    }
    if n < 2 {
        None
    } else {
        Some(ss / (n as f64 - 1.0))
    }
}

/// Shared helper: `stats::median(na.rm = TRUE)` (type-7, i.e. average of the two
/// middle order statistics for even n).
pub(crate) fn median_valid(values: &[f64], valid: Option<&[bool]>) -> Option<f64> {
    let mut v: Vec<f64> = values
        .iter()
        .enumerate()
        .filter(|(i, _)| valid.is_none_or(|m| m[*i]))
        .map(|(_, x)| *x)
        .collect();
    if v.is_empty() {
        return None;
    }
    v.sort_by(|a, b| a.partial_cmp(b).expect("finite values are totally ordered"));
    let n = v.len();
    Some(if n % 2 == 1 {
        v[n / 2]
    } else {
        (v[n / 2 - 1] + v[n / 2]) / 2.0
    })
}

// ---------------------------------------------------------------------------------
// Case weights (recipes 1.3.2: `get_case_weights()` + `are_weights_used(unsupervised =
// TRUE)`): unsupervised steps use *frequency* weights only; importance weights are
// ignored by every step and only reach the model.
// ---------------------------------------------------------------------------------

/// The frequency case weights a step may use: the values of the column with role
/// `case_weights` when its kind is frequency, else `None`.
pub(crate) fn step_weights<'a>(training: &'a Table, schema: &Schema) -> Result<Option<&'a [f64]>> {
    let mut found: Option<(&str, WeightKind)> = None;
    for f in &schema.fields {
        if let Role::CaseWeights(kind) = f.role {
            if found.is_some() {
                return Err(Error::InvalidArgument(
                    "recipes: more than one column has the `case_weights` role".into(),
                ));
            }
            found = Some((f.name.as_str(), kind));
        }
    }
    match found {
        Some((name, WeightKind::Frequency)) => {
            let col = training.column(name).ok_or_else(|| {
                Error::SchemaMismatch(format!("case weights column `{name}` not in training data"))
            })?;
            let v = col.as_numeric().ok_or_else(|| {
                Error::InvalidValue(format!(
                    "`{name}` has a `case_weights` role and should be numeric"
                ))
            })?;
            if col.n_missing() > 0 || v.iter().any(|w| !w.is_finite() || *w < 0.0) {
                return Err(Error::InvalidValue(format!(
                    "case weights column `{name}` must be finite and non-negative with no missing values"
                )));
            }
            Ok(Some(v))
        }
        _ => Ok(None),
    }
}

/// `recipes:::wt_calcs(statistic = "mean" | "var")` on one column: rows with a missing
/// value are dropped (`vec_detect_complete`), then `stats::cov.wt(x, wt)` with the
/// weights normalised to sum 1: centre `sum(wt * x)`, unbiased variance
/// `sum(wt * (x - centre)^2) / (1 - sum(wt^2))`. Errors when the weights are all zero,
/// as `cov.wt` does.
pub(crate) fn cov_wt_valid(values: &[f64], valid: Option<&[bool]>, w: &[f64]) -> Result<Option<(f64, f64)>> {
    let rows: Vec<usize> = (0..values.len())
        .filter(|i| valid.is_none_or(|m| m[*i]))
        .collect();
    if rows.is_empty() {
        return Ok(None);
    }
    let s: f64 = rows.iter().map(|&i| w[i]).sum();
    if s == 0.0 {
        return Err(Error::InvalidValue(
            "weights must be non-negative and not all zero".into(),
        ));
    }
    let wt: Vec<f64> = rows.iter().map(|&i| w[i] / s).collect();
    let center: f64 = rows.iter().zip(&wt).map(|(&i, wi)| wi * values[i]).sum();
    let ss: f64 = rows
        .iter()
        .zip(&wt)
        .map(|(&i, wi)| {
            let d = wi.sqrt() * (values[i] - center);
            d * d
        })
        .sum();
    let denom = 1.0 - wt.iter().map(|v| v * v).sum::<f64>();
    Ok(Some((center, ss / denom)))
}

/// `recipes:::weighted_median_impl` over the complete rows: sort by value, take the
/// first value whose cumulative weight share exceeds one half.
pub(crate) fn weighted_median_valid(
    values: &[f64],
    valid: Option<&[bool]>,
    w: &[f64],
) -> Result<Option<f64>> {
    let mut rows: Vec<usize> = (0..values.len())
        .filter(|i| valid.is_none_or(|m| m[*i]))
        .collect();
    if rows.is_empty() {
        return Ok(None);
    }
    // `order(x)`: stable radix order on the values
    rows.sort_by(|a, b| values[*a].partial_cmp(&values[*b]).expect("finite values"));
    let total: f64 = rows.iter().map(|&i| w[i]).sum();
    if total == 0.0 {
        return Err(Error::InvalidValue(
            "weights must be non-negative and not all zero".into(),
        ));
    }
    let mut cum = 0.0;
    for &i in &rows {
        cum += w[i];
        if cum / total > 0.5 {
            return Ok(Some(values[i]));
        }
    }
    Ok(Some(values[*rows.last().expect("non-empty")]))
}

/// `recipes:::wt_calcs(statistic = "cor")`: listwise-complete rows over all columns,
/// then `cov.wt(cor = TRUE)`. Entries whose variance is zero come out undefined.
pub(crate) fn weighted_cor_matrix(cols: &[(&[f64], Option<&[bool]>)], w: &[f64]) -> Result<Vec<Option<f64>>> {
    let p = cols.len();
    let n = cols.first().map(|c| c.0.len()).unwrap_or(0);
    let rows: Vec<usize> = (0..n)
        .filter(|i| cols.iter().all(|(_, v)| v.is_none_or(|m| m[*i])))
        .collect();
    let s: f64 = rows.iter().map(|&i| w[i]).sum();
    if rows.is_empty() || s == 0.0 {
        return Err(Error::InvalidValue(
            "weights must be non-negative and not all zero".into(),
        ));
    }
    let wt: Vec<f64> = rows.iter().map(|&i| w[i] / s).collect();
    let denom = 1.0 - wt.iter().map(|v| v * v).sum::<f64>();
    let centred: Vec<Vec<f64>> = cols
        .iter()
        .map(|(v, _)| {
            let c: f64 = rows.iter().zip(&wt).map(|(&i, wi)| wi * v[i]).sum();
            rows.iter()
                .zip(&wt)
                .map(|(&i, wi)| wi.sqrt() * (v[i] - c))
                .collect()
        })
        .collect();
    let mut cov = vec![0.0; p * p];
    for i in 0..p {
        for j in 0..=i {
            let c = centred[i]
                .iter()
                .zip(&centred[j])
                .map(|(a, b)| a * b)
                .sum::<f64>()
                / denom;
            cov[i * p + j] = c;
            cov[j * p + i] = c;
        }
    }
    let is: Vec<f64> = (0..p).map(|i| 1.0 / cov[i * p + i].sqrt()).collect();
    // `Is * cov * rep(Is, each = p)`, computed once per pair and mirrored so the matrix
    // is exactly symmetric (R's elementwise product can differ by one ulp between
    // (i, j) and (j, i), which would decide `corr_filter`'s tie-break by round-off)
    let mut out = vec![None; p * p];
    for i in 0..p {
        for j in 0..i {
            let r = is[i] * cov[i * p + j] * is[j];
            let v = r.is_finite().then_some(r);
            out[i * p + j] = v;
            out[j * p + i] = v;
        }
        // cov.wt leaves the diagonal as `Is_i * cov_ii * Is_i`, i.e. one up to round-off
        // (equal for every column in R's arithmetic on the reference task); the exact
        // one removes a dependency of `corr_filter`'s tie-break (`colMeans(abs(x))`)
        // on the last bit of a different summation order
        out[i * p + i] = Some(1.0);
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;

    #[test]
    fn stats_match_r_definitions() {
        let x: Vec<f64> = (1..=10).map(|i| i as f64).collect();
        assert_abs_diff_eq!(mean_valid(&x, None).unwrap(), 5.5);
        // R: var(1:10) = 9.166667
        assert_abs_diff_eq!(var_valid(&x, None).unwrap(), 9.166666666666666, epsilon = 1e-12);
        assert_abs_diff_eq!(median_valid(&x, None).unwrap(), 5.5);
        assert_abs_diff_eq!(median_valid(&x[..9], None).unwrap(), 5.0);
        let valid = [true, false, true, true, true, true, true, true, true, true];
        assert_abs_diff_eq!(mean_valid(&x, Some(&valid)).unwrap(), (55.0 - 2.0) / 9.0);
        assert!(var_valid(&[1.0], None).is_none());
        assert!(mean_valid(&[], None).is_none());
    }
}
