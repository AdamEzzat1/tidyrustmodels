//! `step_dummy`: encode categorical columns as indicator columns.

use crate::names::dummy_name;
use crate::selector::Selector;
use serde::{Deserialize, Serialize};
use tidyrustmodels_core::{Column, ColumnKind, Error, Field, FittedStep, Result, Schema, Step, Table};

/// Indicator encoding. With `one_hot = false` (the recipes default) the first level is
/// the reference and gets no column (treatment contrasts); with `one_hot = true` every
/// level gets a column.
///
/// Missing categorical values propagate as missing indicator values (recipes yields
/// `NA`). An unseen level at bake time is an error (recipes warns and yields `NA`);
/// see COMPATIBILITY.md.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct StepDummy {
    selector: Selector,
    one_hot: bool,
}

impl StepDummy {
    /// Dummy-encode the selected categorical columns (drop-first by default).
    pub fn new(selector: Selector) -> Self {
        StepDummy {
            selector,
            one_hot: false,
        }
    }

    /// Keep a column for every level.
    pub fn one_hot(mut self, one_hot: bool) -> Self {
        self.one_hot = one_hot;
        self
    }
}

/// Fitted state of [`StepDummy`].
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct FittedDummy {
    step: String,
    one_hot: bool,
    /// `(source column, training levels, output column names)`.
    pub encodings: Vec<(String, Vec<String>, Vec<String>)>,
    schema: Schema,
}

impl Step for StepDummy {
    fn name(&self) -> &'static str {
        "dummy"
    }

    fn prep(&self, training: &Table, schema: &Schema) -> Result<Box<dyn FittedStep>> {
        let cols = self.selector.resolve(schema)?;
        let mut encodings = Vec::with_capacity(cols.len());
        for name in &cols {
            let col = training
                .column(name)
                .ok_or_else(|| Error::SchemaMismatch(format!("column `{name}` not in training data")))?;
            let (_, levels) = col.as_categorical().ok_or_else(|| {
                Error::InvalidValue(format!(
                    "step_dummy requires categorical columns; `{name}` is numeric"
                ))
            })?;
            let kept: &[String] = if self.one_hot {
                levels
            } else {
                &levels[1.min(levels.len())..]
            };
            let out_names: Vec<String> = kept.iter().map(|l| dummy_name(name, l)).collect();
            encodings.push((name.clone(), levels.to_vec(), out_names));
        }
        // Output schema: replace each encoded column (in place) by its indicator columns.
        let mut fields = Vec::with_capacity(schema.fields.len());
        for f in &schema.fields {
            match encodings.iter().find(|(n, _, _)| n == &f.name) {
                Some((_, _, outs)) => {
                    for o in outs {
                        fields.push(Field {
                            name: o.clone(),
                            kind: ColumnKind::Numeric,
                            role: f.role,
                        });
                    }
                }
                None => fields.push(f.clone()),
            }
        }
        for f in &fields {
            if fields.iter().filter(|g| g.name == f.name).count() > 1 {
                return Err(Error::InvalidArgument(format!(
                    "dummy column name `{}` collides with an existing column",
                    f.name
                )));
            }
        }
        Ok(Box::new(FittedDummy {
            step: "dummy".into(),
            one_hot: self.one_hot,
            encodings,
            schema: Schema { fields },
        }))
    }

    fn clone_box(&self) -> Box<dyn Step> {
        Box::new(self.clone())
    }
}

impl FittedStep for FittedDummy {
    fn name(&self) -> &'static str {
        "dummy"
    }

    fn bake(&self, data: &Table) -> Result<Table> {
        let mut out = Table::with_nrows(data.nrows());
        for (name, col) in data.iter() {
            match self.encodings.iter().find(|(n, _, _)| n == name) {
                None => out.insert(name, col.clone())?,
                Some((_, levels, out_names)) => {
                    let (codes, bake_levels) = col.as_categorical().ok_or_else(|| {
                        Error::SchemaMismatch(format!("column `{name}` must be categorical at bake time"))
                    })?;
                    // Map bake-time codes to training-level indices by label.
                    let remap: Vec<Option<usize>> = bake_levels
                        .iter()
                        .map(|l| levels.iter().position(|t| t == l))
                        .collect();
                    let valid = col.validity();
                    let offset = if self.one_hot { 0 } else { 1 };
                    let mut cols: Vec<Vec<f64>> = vec![vec![0.0; codes.len()]; out_names.len()];
                    for (i, code) in codes.iter().enumerate() {
                        if valid.is_some_and(|v| !v[i]) {
                            continue;
                        }
                        let train_idx = remap[*code as usize].ok_or_else(|| {
                            Error::InvalidValue(format!(
                                "column `{name}` has level `{}` at row {i} that was not seen in training",
                                bake_levels[*code as usize]
                            ))
                        })?;
                        if train_idx >= offset {
                            cols[train_idx - offset][i] = 1.0;
                        }
                    }
                    for (o, values) in out_names.iter().zip(cols) {
                        out.insert(
                            o.clone(),
                            Column::Numeric {
                                values,
                                valid: valid.map(|v| v.to_vec()),
                            },
                        )?;
                    }
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
    use tidyrustmodels_core::Role;

    fn table() -> (Table, Schema) {
        let t = Table::new(vec![
            ("y", Column::numeric(vec![0.0; 4]).unwrap()),
            ("species", Column::categorical(&["b", "a", "c", "b"]).unwrap()),
        ])
        .unwrap();
        let s = t.schema_with_outcome("y", Role::Predictor).unwrap();
        (t, s)
    }

    #[test]
    fn drop_first_by_default_with_recipes_names() {
        let (t, s) = table();
        let f = StepDummy::new(Selector::AllNominalPredictors)
            .prep(&t, &s)
            .unwrap();
        let b = f.bake(&t).unwrap();
        assert_eq!(b.names(), &["y", "species_b", "species_c"]);
        assert_eq!(
            b.column("species_b").unwrap().as_numeric().unwrap(),
            &[1.0, 0.0, 0.0, 1.0]
        );
        assert_eq!(
            b.column("species_c").unwrap().as_numeric().unwrap(),
            &[0.0, 0.0, 1.0, 0.0]
        );
        let names: Vec<&str> = f.output_schema().fields.iter().map(|f| f.name.as_str()).collect();
        assert_eq!(names, vec!["y", "species_b", "species_c"]);
    }

    #[test]
    fn one_hot_keeps_all_levels() {
        let (t, s) = table();
        let f = StepDummy::new(Selector::columns(&["species"]))
            .one_hot(true)
            .prep(&t, &s)
            .unwrap();
        let b = f.bake(&t).unwrap();
        assert_eq!(b.names(), &["y", "species_a", "species_b", "species_c"]);
    }

    #[test]
    fn unseen_level_is_an_error_and_missing_propagates() {
        let (t, s) = table();
        let f = StepDummy::new(Selector::columns(&["species"]))
            .prep(&t, &s)
            .unwrap();
        let bad = Table::new(vec![
            ("y", Column::numeric(vec![0.0]).unwrap()),
            ("species", Column::categorical(&["zebra"]).unwrap()),
        ])
        .unwrap();
        assert!(matches!(f.bake(&bad), Err(Error::InvalidValue(_))));
        let miss = Table::new(vec![
            ("y", Column::numeric(vec![0.0, 0.0]).unwrap()),
            (
                "species",
                Column::categorical_with_levels_opt(
                    &[None, Some("c")],
                    vec!["a".into(), "b".into(), "c".into()],
                )
                .unwrap(),
            ),
        ])
        .unwrap();
        let b = f.bake(&miss).unwrap();
        assert!(b.column("species_b").unwrap().is_missing(0));
        assert_eq!(b.column("species_c").unwrap().as_numeric().unwrap()[1], 1.0);
    }
}
