//! `step_impute_mean` / `step_impute_median`: fill missing numeric values with a
//! statistic estimated on the training data.

use super::{cov_wt_valid, mean_valid, median_valid, step_weights, weighted_median_valid};
use crate::selector::Selector;
use serde::{Deserialize, Serialize};
use tidyrustmodels_core::{Column, Error, FittedStep, Result, Schema, Step, Table};

/// Which statistic an imputation step uses.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum ImputeStat {
    /// Training mean of non-missing values.
    Mean,
    /// Training median (type 7) of non-missing values.
    Median,
}

/// Impute with the training mean (`step_impute_mean`, `trim = 0`).
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct StepImputeMean {
    selector: Selector,
}

/// Impute with the training median (`step_impute_median`).
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct StepImputeMedian {
    selector: Selector,
}

impl StepImputeMean {
    /// Impute the selected numeric columns.
    pub fn new(selector: Selector) -> Self {
        StepImputeMean { selector }
    }
}

impl StepImputeMedian {
    /// Impute the selected numeric columns.
    pub fn new(selector: Selector) -> Self {
        StepImputeMedian { selector }
    }
}

/// Fitted state shared by both imputation steps.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct FittedImpute {
    step: String,
    stat: ImputeStat,
    /// `(column, fill value)`.
    pub fills: Vec<(String, f64)>,
    schema: Schema,
}

fn prep_impute(
    stat: ImputeStat,
    selector: &Selector,
    training: &Table,
    schema: &Schema,
) -> Result<FittedImpute> {
    let cols = selector.resolve(schema)?;
    // frequency case weights: `averages()` (cov.wt centre) / `weighted_median_impl`
    let wts = step_weights(training, schema)?;
    let mut fills = Vec::with_capacity(cols.len());
    for name in cols {
        let col = training
            .column(&name)
            .ok_or_else(|| Error::SchemaMismatch(format!("column `{name}` not in training data")))?;
        let values = col.as_numeric().ok_or_else(|| {
            Error::InvalidValue(format!(
                "imputation requires numeric columns; `{name}` is categorical"
            ))
        })?;
        let fill = match (stat, wts) {
            (ImputeStat::Mean, None) => mean_valid(values, col.validity()),
            (ImputeStat::Median, None) => median_valid(values, col.validity()),
            (ImputeStat::Mean, Some(w)) => cov_wt_valid(values, col.validity(), w)?.map(|(m, _)| m),
            (ImputeStat::Median, Some(w)) => weighted_median_valid(values, col.validity(), w)?,
        }
        .ok_or_else(|| Error::InvalidValue(format!("column `{name}` has no non-missing training values")))?;
        fills.push((name, fill));
    }
    Ok(FittedImpute {
        step: match stat {
            ImputeStat::Mean => "impute_mean".into(),
            ImputeStat::Median => "impute_median".into(),
        },
        stat,
        fills,
        schema: schema.clone(),
    })
}

impl Step for StepImputeMean {
    fn name(&self) -> &'static str {
        "impute_mean"
    }
    fn prep(&self, training: &Table, schema: &Schema) -> Result<Box<dyn FittedStep>> {
        Ok(Box::new(prep_impute(
            ImputeStat::Mean,
            &self.selector,
            training,
            schema,
        )?))
    }
    fn clone_box(&self) -> Box<dyn Step> {
        Box::new(self.clone())
    }
}

impl Step for StepImputeMedian {
    fn name(&self) -> &'static str {
        "impute_median"
    }
    fn prep(&self, training: &Table, schema: &Schema) -> Result<Box<dyn FittedStep>> {
        Ok(Box::new(prep_impute(
            ImputeStat::Median,
            &self.selector,
            training,
            schema,
        )?))
    }
    fn clone_box(&self) -> Box<dyn Step> {
        Box::new(self.clone())
    }
}

impl FittedStep for FittedImpute {
    fn name(&self) -> &'static str {
        match self.stat {
            ImputeStat::Mean => "impute_mean",
            ImputeStat::Median => "impute_median",
        }
    }

    fn bake(&self, data: &Table) -> Result<Table> {
        let mut out = data.clone();
        for (name, fill) in &self.fills {
            let col = data
                .column(name)
                .ok_or_else(|| Error::SchemaMismatch(format!("column `{name}` missing at bake time")))?;
            let values = col.as_numeric().ok_or_else(|| {
                Error::SchemaMismatch(format!("column `{name}` must be numeric at bake time"))
            })?;
            if let Some(valid) = col.validity() {
                let filled: Vec<f64> = values
                    .iter()
                    .zip(valid)
                    .map(|(v, ok)| if *ok { *v } else { *fill })
                    .collect();
                out.insert(
                    name.clone(),
                    Column::Numeric {
                        values: filled,
                        valid: None,
                    },
                )?;
            }
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
    use tidyrustmodels_core::Role;

    #[test]
    fn median_and_mean_fill_from_training_only() {
        let train = Table::new(vec![
            (
                "x",
                Column::numeric_with_missing(vec![Some(1.0), Some(10.0), None, Some(4.0)]).unwrap(),
            ),
            ("y", Column::numeric(vec![0.0; 4]).unwrap()),
        ])
        .unwrap();
        let schema = train.schema_with_outcome("y", Role::Predictor).unwrap();
        let med = StepImputeMedian::new(Selector::columns(&["x"]))
            .prep(&train, &schema)
            .unwrap();
        let mean = StepImputeMean::new(Selector::columns(&["x"]))
            .prep(&train, &schema)
            .unwrap();
        let test = Table::new(vec![
            (
                "x",
                Column::numeric_with_missing(vec![None, Some(100.0)]).unwrap(),
            ),
            ("y", Column::numeric(vec![0.0; 2]).unwrap()),
        ])
        .unwrap();
        let bm = med.bake(&test).unwrap();
        assert_eq!(bm.column("x").unwrap().as_numeric().unwrap(), &[4.0, 100.0]);
        assert_eq!(bm.column("x").unwrap().n_missing(), 0);
        let bmean = mean.bake(&test).unwrap();
        assert_eq!(bmean.column("x").unwrap().as_numeric().unwrap(), &[5.0, 100.0]);
    }
}
