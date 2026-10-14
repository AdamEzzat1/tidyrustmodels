//! `step_select`: keep only the named predictors (outcome and non-predictor roles are
//! always retained).

use crate::selector::Selector;
use serde::{Deserialize, Serialize};
use tidyrustmodels_core::{Error, FittedStep, Result, Role, Schema, Step, Table};

/// Keep the selected predictor columns and drop every other predictor.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct StepSelect {
    selector: Selector,
}

impl StepSelect {
    /// Keep the selected predictors.
    pub fn new(selector: Selector) -> Self {
        StepSelect { selector }
    }

    /// Keep exactly these predictor columns.
    pub fn columns<S: AsRef<str>>(names: &[S]) -> Self {
        StepSelect {
            selector: Selector::columns(names),
        }
    }
}

/// Fitted state of [`StepSelect`].
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct FittedSelect {
    step: String,
    /// Columns kept, in schema order.
    pub keep: Vec<String>,
    schema: Schema,
}

impl Step for StepSelect {
    fn name(&self) -> &'static str {
        "select"
    }

    fn prep(&self, _training: &Table, schema: &Schema) -> Result<Box<dyn FittedStep>> {
        let selected = self.selector.resolve(schema)?;
        let fields: Vec<_> = schema
            .fields
            .iter()
            .filter(|f| f.role != Role::Predictor || selected.contains(&f.name))
            .cloned()
            .collect();
        Ok(Box::new(FittedSelect {
            step: "select".into(),
            keep: fields.iter().map(|f| f.name.clone()).collect(),
            schema: Schema { fields },
        }))
    }

    fn clone_box(&self) -> Box<dyn Step> {
        Box::new(self.clone())
    }
}

impl FittedStep for FittedSelect {
    fn name(&self) -> &'static str {
        "select"
    }

    fn bake(&self, data: &Table) -> Result<Table> {
        let mut out = Table::with_nrows(data.nrows());
        for name in &self.keep {
            // The outcome may legitimately be absent at prediction time.
            match data.column_shared(name) {
                Some(c) => out.insert_shared(name.clone(), c)?,
                None if self.schema.field(name).is_some_and(|f| f.role == Role::Outcome) => {}
                None => {
                    return Err(Error::SchemaMismatch(format!(
                        "column `{name}` missing at bake time"
                    )))
                }
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
    use tidyrustmodels_core::Column;

    #[test]
    fn keeps_selected_predictors_and_outcome() {
        let t = Table::new(vec![
            ("y", Column::numeric(vec![1.0]).unwrap()),
            ("a", Column::numeric(vec![1.0]).unwrap()),
            ("b", Column::numeric(vec![1.0]).unwrap()),
        ])
        .unwrap();
        let s = t.schema_with_outcome("y", Role::Predictor).unwrap();
        let f = StepSelect::columns(&["b"]).prep(&t, &s).unwrap();
        assert_eq!(f.bake(&t).unwrap().names(), &["y", "b"]);
        let no_outcome = Table::new(vec![
            ("a", Column::numeric(vec![1.0]).unwrap()),
            ("b", Column::numeric(vec![2.0]).unwrap()),
        ])
        .unwrap();
        assert_eq!(f.bake(&no_outcome).unwrap().names(), &["b"]);
    }
}
