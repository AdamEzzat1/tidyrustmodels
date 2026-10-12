//! Built-in preprocessing steps.

mod dummy;
mod impute;
mod normalize;
mod select;

pub use dummy::{FittedDummy, StepDummy};
pub use impute::{FittedImpute, ImputeStat, StepImputeMean, StepImputeMedian};
pub use normalize::{FittedNormalize, StepNormalize};
pub use select::{FittedSelect, StepSelect};

use tidyrustmodels_core::{Error, FittedStep, Result};

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
