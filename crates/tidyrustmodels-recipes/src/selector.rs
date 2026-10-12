//! Column selectors, a small subset of recipes' tidyselect helpers.

use serde::{Deserialize, Serialize};
use tidyrustmodels_core::{ColumnKind, Error, Result, Role, Schema};

/// Which columns a step operates on. Resolved against the schema at `prep` time.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum Selector {
    /// Explicit column names; every name must exist.
    Columns(Vec<String>),
    /// All predictor-role columns.
    AllPredictors,
    /// All numeric predictor-role columns (`all_numeric_predictors()`).
    AllNumericPredictors,
    /// All categorical predictor-role columns (`all_nominal_predictors()`).
    AllNominalPredictors,
}

impl Selector {
    /// Convenience constructor for explicit names.
    pub fn columns<S: AsRef<str>>(names: &[S]) -> Self {
        Selector::Columns(names.iter().map(|s| s.as_ref().to_string()).collect())
    }

    /// Resolve to concrete column names in schema order.
    pub fn resolve(&self, schema: &Schema) -> Result<Vec<String>> {
        match self {
            Selector::Columns(names) => {
                for n in names {
                    if schema.field(n).is_none() {
                        return Err(Error::SchemaMismatch(format!("selected column `{n}` not found")));
                    }
                }
                // Preserve schema order, like tidyselect.
                Ok(schema
                    .fields
                    .iter()
                    .filter(|f| names.contains(&f.name))
                    .map(|f| f.name.clone())
                    .collect())
            }
            Selector::AllPredictors => Ok(by_role_kind(schema, None)),
            Selector::AllNumericPredictors => Ok(by_role_kind(schema, Some(ColumnKind::Numeric))),
            Selector::AllNominalPredictors => Ok(by_role_kind(schema, Some(ColumnKind::Categorical))),
        }
    }
}

fn by_role_kind(schema: &Schema, kind: Option<ColumnKind>) -> Vec<String> {
    schema
        .fields
        .iter()
        .filter(|f| f.role == Role::Predictor && kind.is_none_or(|k| f.kind == k))
        .map(|f| f.name.clone())
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use tidyrustmodels_core::Field;

    fn schema() -> Schema {
        Schema {
            fields: vec![
                Field {
                    name: "y".into(),
                    kind: ColumnKind::Numeric,
                    role: Role::Outcome,
                },
                Field {
                    name: "a".into(),
                    kind: ColumnKind::Numeric,
                    role: Role::Predictor,
                },
                Field {
                    name: "b".into(),
                    kind: ColumnKind::Categorical,
                    role: Role::Predictor,
                },
                Field {
                    name: "id".into(),
                    kind: ColumnKind::Numeric,
                    role: Role::Id,
                },
            ],
        }
    }

    #[test]
    fn resolves_by_role_and_kind() {
        let s = schema();
        assert_eq!(Selector::AllPredictors.resolve(&s).unwrap(), vec!["a", "b"]);
        assert_eq!(Selector::AllNumericPredictors.resolve(&s).unwrap(), vec!["a"]);
        assert_eq!(Selector::AllNominalPredictors.resolve(&s).unwrap(), vec!["b"]);
    }

    #[test]
    fn explicit_names_keep_schema_order_and_must_exist() {
        let s = schema();
        assert_eq!(
            Selector::columns(&["b", "a"]).resolve(&s).unwrap(),
            vec!["a", "b"]
        );
        assert!(Selector::columns(&["zzz"]).resolve(&s).is_err());
    }
}
