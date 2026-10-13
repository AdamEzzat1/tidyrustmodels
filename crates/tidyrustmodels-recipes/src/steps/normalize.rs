//! `step_normalize`: centre and scale numeric columns using training statistics.

use super::{cov_wt_valid, mean_valid, step_weights, var_valid};
use crate::selector::Selector;
use serde::{Deserialize, Serialize};
use tidyrustmodels_core::{Column, ColumnKind, Error, FittedStep, Result, Schema, Step, Table};

/// Centre by the training mean and scale by the training standard deviation
/// (`stats::var`, n-1 denominator, missing values ignored). A zero or undefined
/// standard deviation is replaced by 1, as recipes' `sd_check` does. With frequency
/// case weights the statistics come from `stats::cov.wt` (weighted centre, unbiased
/// weighted variance), as recipes' `averages()` / `variances()` do.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct StepNormalize {
    selector: Selector,
}

impl StepNormalize {
    /// Normalize the selected columns.
    pub fn new(selector: Selector) -> Self {
        StepNormalize { selector }
    }
}

/// Fitted state of [`StepNormalize`].
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct FittedNormalize {
    step: String,
    /// `(column, mean, sd)` in schema order.
    pub stats: Vec<(String, f64, f64)>,
    schema: Schema,
}

impl Step for StepNormalize {
    fn name(&self) -> &'static str {
        "normalize"
    }

    fn prep(&self, training: &Table, schema: &Schema) -> Result<Box<dyn FittedStep>> {
        let cols = self.selector.resolve(schema)?;
        let wts = step_weights(training, schema)?;
        let mut stats = Vec::with_capacity(cols.len());
        for name in cols {
            let col = training
                .column(&name)
                .ok_or_else(|| Error::SchemaMismatch(format!("column `{name}` not in training data")))?;
            let values = col.as_numeric().ok_or_else(|| {
                Error::InvalidValue(format!(
                    "step_normalize requires numeric columns; `{name}` is categorical"
                ))
            })?;
            let (mean, sd) = match wts {
                Some(w) => {
                    let (m, v) = cov_wt_valid(values, col.validity(), w)?.ok_or_else(|| {
                        Error::InvalidValue(format!("column `{name}` has no non-missing training values"))
                    })?;
                    (m, Some(v.sqrt()))
                }
                None => (
                    mean_valid(values, col.validity()).ok_or_else(|| {
                        Error::InvalidValue(format!("column `{name}` has no non-missing training values"))
                    })?,
                    var_valid(values, col.validity()).map(f64::sqrt),
                ),
            };
            // recipes::sd_check: sd == 0 (or NA) -> 1 with a warning.
            let sd = match sd {
                Some(s) if s > 0.0 => s,
                _ => 1.0,
            };
            stats.push((name, mean, sd));
        }
        Ok(Box::new(FittedNormalize {
            step: "normalize".into(),
            stats,
            schema: schema.clone(),
        }))
    }

    fn clone_box(&self) -> Box<dyn Step> {
        Box::new(self.clone())
    }
}

impl FittedStep for FittedNormalize {
    fn name(&self) -> &'static str {
        "normalize"
    }

    fn bake(&self, data: &Table) -> Result<Table> {
        let mut out = data.clone();
        for (name, mean, sd) in &self.stats {
            let col = data
                .column(name)
                .ok_or_else(|| Error::SchemaMismatch(format!("column `{name}` missing at bake time")))?;
            if col.kind() != ColumnKind::Numeric {
                return Err(Error::SchemaMismatch(format!(
                    "column `{name}` must be numeric at bake time"
                )));
            }
            let values = col.as_numeric().expect("kind checked above");
            let valid = col.validity().map(|v| v.to_vec());
            let scaled: Vec<f64> = values
                .iter()
                .enumerate()
                .map(|(i, v)| {
                    if valid.as_ref().is_none_or(|m| m[i]) {
                        (v - mean) / sd
                    } else {
                        0.0
                    }
                })
                .collect();
            out.insert(
                name.clone(),
                Column::Numeric {
                    values: scaled,
                    valid,
                    // centring materialises a sparse vector in recipes
                    sparse: false,
                },
            )?;
        }
        Ok(out)
    }

    fn output_schema(&self) -> &Schema {
        &self.schema
    }

    fn state(&self) -> Result<serde_json::Value> {
        serde_json::to_value(self).map_err(|e| Error::Serialization(e.to_string()))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;
    use tidyrustmodels_core::Role;

    #[test]
    fn normalize_uses_training_stats_only_and_handles_missing() {
        let train = Table::new(vec![
            (
                "x",
                Column::numeric_with_missing(vec![Some(1.0), Some(2.0), None, Some(3.0)]).unwrap(),
            ),
            ("y", Column::numeric(vec![0.0; 4]).unwrap()),
        ])
        .unwrap();
        let schema = train.schema_with_outcome("y", Role::Predictor).unwrap();
        let fitted = StepNormalize::new(Selector::AllNumericPredictors)
            .prep(&train, &schema)
            .unwrap();
        let test = Table::new(vec![
            ("x", Column::numeric(vec![2.0, 10.0]).unwrap()),
            ("y", Column::numeric(vec![0.0; 2]).unwrap()),
        ])
        .unwrap();
        let baked = fitted.bake(&test).unwrap();
        let x = baked.column("x").unwrap().as_numeric().unwrap();
        // mean 2, sd 1 from training rows [1,2,3]
        assert_abs_diff_eq!(x[0], 0.0);
        assert_abs_diff_eq!(x[1], 8.0);
        // outcome untouched
        assert_eq!(baked.column("y").unwrap().as_numeric().unwrap(), &[0.0, 0.0]);
    }

    #[test]
    fn zero_variance_column_gets_sd_one() {
        let t = Table::new(vec![("x", Column::numeric(vec![5.0, 5.0, 5.0]).unwrap())]).unwrap();
        let schema = Schema {
            fields: vec![tidyrustmodels_core::Field {
                name: "x".into(),
                kind: ColumnKind::Numeric,
                role: Role::Predictor,
            }],
        };
        let f = StepNormalize::new(Selector::AllPredictors)
            .prep(&t, &schema)
            .unwrap();
        let b = f.bake(&t).unwrap();
        assert_eq!(b.column("x").unwrap().as_numeric().unwrap(), &[0.0, 0.0, 0.0]);
    }
}
