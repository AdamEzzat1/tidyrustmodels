//! Recipe: an ordered list of steps, prepped on training data and baked anywhere.

use crate::steps::fitted_step_from_state;
use serde::{Deserialize, Serialize};
use tidyrustmodels_core::{
    ColumnKind, DesignMatrix, Error, FittedStep, Outcome, Result, Role, Schema, Step, Table, WeightKind,
};

/// Serialization format version for fitted recipes.
pub const RECIPE_FORMAT_VERSION: u32 = 1;

/// An unfitted recipe.
#[derive(Clone, Debug)]
pub struct Recipe {
    outcome: String,
    steps: Vec<Box<dyn Step>>,
}

impl Recipe {
    /// A recipe predicting `outcome`; every other column starts as a predictor.
    pub fn new(outcome: impl Into<String>) -> Self {
        Recipe {
            outcome: outcome.into(),
            steps: Vec::new(),
        }
    }

    /// Append a step.
    pub fn step(mut self, step: impl Step + 'static) -> Self {
        self.steps.push(Box::new(step));
        self
    }

    /// Outcome column name.
    pub fn outcome(&self) -> &str {
        &self.outcome
    }

    /// Steps in order.
    pub fn steps(&self) -> &[Box<dyn Step>] {
        &self.steps
    }

    /// recipes' `.recipes_estimate_sparsity()` (recipes 1.3.2): the share of exact
    /// zeros in the numeric columns of the template (all columns, outcome included),
    /// adjusted for the columns that sparse-capable steps (`step_dummy`) will create.
    /// workflows feeds this to `ModelSpec::resolve_sparsity`. R samples 1000 rows for
    /// the zero count when the template is larger; this counts every row.
    pub fn estimate_sparsity(&self, template: &Table) -> f64 {
        let n_rows = template.nrows();
        if n_rows == 0 {
            return 0.0;
        }
        let mut n_cols = template.ncols() as f64;
        let n_rows_f = n_rows as f64;
        let zeros: f64 = template
            .iter()
            .map(|(_, c)| match c.as_numeric() {
                Some(v) => v
                    .iter()
                    .enumerate()
                    .filter(|(i, x)| **x == 0.0 && !c.is_missing(*i))
                    .count() as f64,
                None => 0.0,
            })
            .sum();
        let est = zeros / (n_rows_f * n_cols);
        let mut zeroes = est * n_rows_f * n_cols;
        if let Ok(schema) = template.schema_with_outcome(&self.outcome, Role::Predictor) {
            for step in &self.steps {
                for (cols, sparsity) in step.sparse_adjustments(template, &schema) {
                    zeroes += n_rows_f * sparsity * cols as f64;
                    n_cols = n_cols + cols as f64 - 1.0;
                }
            }
        }
        zeroes / (n_rows_f * n_cols)
    }

    /// Estimate every step on `training` in sequence (each step sees the output of the
    /// previous one, exactly as recipes' `prep`).
    pub fn prep(&self, training: &Table) -> Result<FittedRecipe> {
        self.prep_with_weights(training, None)
    }

    /// `prep` with one column given the `case_weights` role (recipes assigns it to a
    /// `hardhat` case-weights column automatically). The column is never a predictor;
    /// unsupervised steps read it when the kind is `WeightKind::Frequency`, and the
    /// importance kind only reaches the model. At bake time the column may be absent.
    pub fn prep_with_weights(
        &self,
        training: &Table,
        case_weights: Option<(&str, WeightKind)>,
    ) -> Result<FittedRecipe> {
        let input_schema = training.schema_with_roles(&self.outcome, Role::Predictor, case_weights)?;
        let mut schema = input_schema.clone();
        let mut current = training.clone();
        let mut fitted: Vec<Box<dyn FittedStep>> = Vec::with_capacity(self.steps.len());
        for step in &self.steps {
            let f = step.prep(&current, &schema)?;
            current = f.bake(&current)?;
            schema = f.output_schema().clone();
            fitted.push(f);
        }
        Ok(FittedRecipe {
            outcome: self.outcome.clone(),
            steps: fitted,
            input_schema,
            output_schema: schema,
        })
    }
}

/// A prepped recipe. Immutable.
#[derive(Debug)]
pub struct FittedRecipe {
    outcome: String,
    steps: Vec<Box<dyn FittedStep>>,
    input_schema: Schema,
    output_schema: Schema,
}

impl FittedRecipe {
    /// Apply all steps to `data`. The outcome column may be absent (prediction time).
    pub fn bake(&self, data: &Table) -> Result<Table> {
        for f in &self.input_schema.fields {
            if f.role == Role::Outcome {
                continue;
            }
            match data.column(&f.name) {
                // case weights are only needed to prep; `predict()` data carry none
                None if matches!(f.role, Role::CaseWeights(_)) => continue,
                None => return Err(Error::SchemaMismatch(format!("column `{}` is missing", f.name))),
                Some(c) if c.kind() != f.kind => {
                    return Err(Error::SchemaMismatch(format!(
                        "column `{}` has kind {:?}, expected {:?}",
                        f.name,
                        c.kind(),
                        f.kind
                    )))
                }
                Some(_) => {}
            }
        }
        let mut current = data.clone();
        for step in &self.steps {
            current = step.bake(&current)?;
        }
        Ok(current)
    }

    /// Bake and assemble the design matrix from predictor-role numeric columns, plus
    /// the outcome if present. Errors if a predictor is still categorical or has
    /// missing values, naming the column and the step that would fix it.
    pub fn bake_matrix(&self, data: &Table) -> Result<(DesignMatrix, Option<Outcome>)> {
        let baked = self.bake(data)?;
        let mut cols = Vec::new();
        for f in &self.output_schema.fields {
            if f.role != Role::Predictor {
                continue;
            }
            let col = baked
                .column(&f.name)
                .ok_or_else(|| Error::SchemaMismatch(format!("predictor `{}` missing after bake", f.name)))?;
            if col.kind() == ColumnKind::Categorical {
                return Err(Error::InvalidValue(format!(
                    "predictor `{}` is categorical; add StepDummy before modelling",
                    f.name
                )));
            }
            if col.n_missing() > 0 {
                return Err(Error::InvalidValue(format!(
                    "predictor `{}` has {} missing values; add an imputation step before modelling",
                    f.name,
                    col.n_missing()
                )));
            }
            cols.push((f.name.clone(), col.as_numeric().expect("numeric").to_vec()));
        }
        let x = if cols.is_empty() {
            DesignMatrix::from_column_major(Vec::new(), baked.nrows(), 0, Vec::new())?
        } else {
            DesignMatrix::from_columns(cols)?
        };
        let y = match baked.column(&self.outcome) {
            Some(c) => Some(Outcome::from_column(c)?),
            None => None,
        };
        Ok((x, y))
    }

    /// Outcome column name.
    pub fn outcome(&self) -> &str {
        &self.outcome
    }

    /// Schema of the training table.
    pub fn input_schema(&self) -> &Schema {
        &self.input_schema
    }

    /// Schema after all steps.
    pub fn output_schema(&self) -> &Schema {
        &self.output_schema
    }

    /// Fitted steps in order.
    pub fn steps(&self) -> &[Box<dyn FittedStep>] {
        &self.steps
    }

    /// Feature names the design matrix will have, in order.
    pub fn feature_names(&self) -> Vec<String> {
        self.output_schema
            .names_with_role(Role::Predictor)
            .into_iter()
            .map(String::from)
            .collect()
    }

    /// Serialize to a versioned JSON envelope. Only built-in steps are supported.
    pub fn to_json(&self) -> Result<serde_json::Value> {
        let steps = self.steps.iter().map(|s| s.state()).collect::<Result<Vec<_>>>()?;
        let env = FittedRecipeEnvelope {
            format_version: RECIPE_FORMAT_VERSION,
            outcome: self.outcome.clone(),
            input_schema: self.input_schema.clone(),
            output_schema: self.output_schema.clone(),
            steps,
        };
        serde_json::to_value(env).map_err(|e| Error::Serialization(e.to_string()))
    }

    /// Rebuild from [`FittedRecipe::to_json`] output.
    pub fn from_json(value: &serde_json::Value) -> Result<Self> {
        let env: FittedRecipeEnvelope =
            serde_json::from_value(value.clone()).map_err(|e| Error::Serialization(e.to_string()))?;
        if env.format_version != RECIPE_FORMAT_VERSION {
            return Err(Error::Serialization(format!(
                "fitted recipe format version {} is not supported (expected {RECIPE_FORMAT_VERSION})",
                env.format_version
            )));
        }
        let steps = env
            .steps
            .iter()
            .map(fitted_step_from_state)
            .collect::<Result<Vec<_>>>()?;
        Ok(FittedRecipe {
            outcome: env.outcome,
            steps,
            input_schema: env.input_schema,
            output_schema: env.output_schema,
        })
    }
}

#[derive(Serialize, Deserialize)]
struct FittedRecipeEnvelope {
    format_version: u32,
    outcome: String,
    input_schema: Schema,
    output_schema: Schema,
    steps: Vec<serde_json::Value>,
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::selector::Selector;
    use crate::steps::{StepDummy, StepImputeMedian, StepNormalize};
    use approx::assert_abs_diff_eq;
    use tidyrustmodels_core::Column;

    fn penguin_like() -> Table {
        Table::new(vec![
            ("mass", Column::numeric(vec![10.0, 20.0, 30.0, 40.0]).unwrap()),
            (
                "flipper",
                Column::numeric_with_missing(vec![Some(1.0), None, Some(3.0), Some(5.0)]).unwrap(),
            ),
            ("species", Column::categorical(&["a", "b", "c", "a"]).unwrap()),
        ])
        .unwrap()
    }

    #[test]
    fn end_to_end_prep_bake_matrix() {
        let rec = Recipe::new("mass")
            .step(StepImputeMedian::new(Selector::AllNumericPredictors))
            .step(StepNormalize::new(Selector::AllNumericPredictors))
            .step(StepDummy::new(Selector::AllNominalPredictors));
        let fitted = rec.prep(&penguin_like()).unwrap();
        assert_eq!(fitted.feature_names(), vec!["flipper", "species_b", "species_c"]);
        let (x, y) = fitted.bake_matrix(&penguin_like()).unwrap();
        assert_eq!(x.ncols(), 3);
        assert_eq!(x.nrows(), 4);
        // flipper imputed with median(1,3,5)=3 then normalized: values [1,3,3,5], mean 3, sd sqrt(8/3)
        let sd = (8.0f64 / 3.0).sqrt();
        assert_abs_diff_eq!(x.get(0, 0), (1.0 - 3.0) / sd, epsilon = 1e-12);
        assert_abs_diff_eq!(x.get(1, 0), 0.0, epsilon = 1e-12);
        assert_eq!(y.unwrap().as_numeric().unwrap(), &[10.0, 20.0, 30.0, 40.0]);
    }

    #[test]
    fn bake_without_outcome_and_schema_checks() {
        let rec = Recipe::new("mass")
            .step(StepImputeMedian::new(Selector::AllNumericPredictors))
            .step(StepDummy::new(Selector::AllNominalPredictors));
        let fitted = rec.prep(&penguin_like()).unwrap();
        let mut t = penguin_like();
        t.remove("mass");
        let (x, y) = fitted.bake_matrix(&t).unwrap();
        assert!(y.is_none());
        assert_eq!(x.nrows(), 4);
        t.remove("species");
        assert!(matches!(fitted.bake(&t), Err(Error::SchemaMismatch(_))));
    }

    #[test]
    fn categorical_predictor_without_dummy_is_an_error() {
        let fitted = Recipe::new("mass")
            .step(StepImputeMedian::new(Selector::AllNumericPredictors))
            .prep(&penguin_like())
            .unwrap();
        let err = fitted.bake_matrix(&penguin_like()).unwrap_err();
        assert!(err.to_string().contains("StepDummy"));
    }

    #[test]
    fn json_round_trip_preserves_bake() {
        let rec = Recipe::new("mass")
            .step(StepImputeMedian::new(Selector::AllNumericPredictors))
            .step(StepNormalize::new(Selector::AllNumericPredictors))
            .step(StepDummy::new(Selector::AllNominalPredictors));
        let fitted = rec.prep(&penguin_like()).unwrap();
        let json = fitted.to_json().unwrap();
        let back = FittedRecipe::from_json(&json).unwrap();
        let (a, _) = fitted.bake_matrix(&penguin_like()).unwrap();
        let (b, _) = back.bake_matrix(&penguin_like()).unwrap();
        assert_eq!(a, b);
        let mut bad = json.clone();
        bad["format_version"] = serde_json::json!(99);
        assert!(FittedRecipe::from_json(&bad).is_err());
    }
}
