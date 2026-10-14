//! `step_dummy`: encode categorical columns as indicator columns.

use crate::names::dummy_name;
use crate::selector::Selector;
use serde::{Deserialize, Serialize};
use std::sync::Arc;
use tidyrustmodels_core::{Column, ColumnKind, Error, Field, FittedStep, Result, Role, Schema, Step, Table};

/// Indicator encoding. With `one_hot = false` (the recipes default) the first level is
/// the reference and gets no column (treatment contrasts); with `one_hot = true` every
/// level gets a column. As in recipes, the indicator columns are appended after the
/// remaining columns.
///
/// Missing categorical values propagate as missing indicator values (recipes yields
/// `NA`). An unseen level at bake time is an error (recipes warns and yields `NA`);
/// see COMPATIBILITY.md.
///
/// `sparse` is recipes' `sparse = "auto" | "yes" | "no"` argument (ADR-0015): with
/// `Yes` the indicator columns are flagged as sparse vectors, which makes the design
/// sparse for xgboost (zeros become missing values); `Auto` (the default) bakes dense
/// columns unless a `Workflow::fit` toggles it, as `workflows:::toggle_sparsity` does.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct StepDummy {
    selector: Selector,
    one_hot: bool,
    #[serde(default)]
    sparse: SparseOutput,
}

/// recipes' `sparse` argument of `step_dummy()`.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub enum SparseOutput {
    /// Dense unless a workflow's sparse-data decision toggles it.
    #[default]
    Auto,
    /// Indicator columns are sparse vectors.
    Yes,
    /// Indicator columns are dense.
    No,
}

impl StepDummy {
    /// Dummy-encode the selected categorical columns (drop-first by default).
    pub fn new(selector: Selector) -> Self {
        StepDummy {
            selector,
            one_hot: false,
            sparse: SparseOutput::Auto,
        }
    }

    /// Keep a column for every level.
    pub fn one_hot(mut self, one_hot: bool) -> Self {
        self.one_hot = one_hot;
        self
    }

    /// Set recipes' `sparse` argument (see [`SparseOutput`]).
    pub fn sparse(mut self, sparse: SparseOutput) -> Self {
        self.sparse = sparse;
        self
    }

    /// The `sparse` setting.
    pub fn sparse_output(&self) -> SparseOutput {
        self.sparse
    }
}

/// Fitted state of [`StepDummy`].
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct FittedDummy {
    step: String,
    one_hot: bool,
    /// Whether the indicator columns are flagged as sparse vectors.
    #[serde(default)]
    pub sparse: bool,
    /// `(source column, training levels, output column names)`.
    pub encodings: Vec<(String, Vec<String>, Vec<String>)>,
    schema: Schema,
}

impl Step for StepDummy {
    fn name(&self) -> &'static str {
        "dummy"
    }

    /// `.recipes_estimate_sparsity.step_dummy`: each selected factor with `k` levels
    /// becomes `k - 1` (or `k` with one-hot) columns of sparsity `1 - 1/k`.
    fn sparse_adjustments(&self, training: &Table, schema: &Schema) -> Vec<(usize, f64)> {
        let Ok(cols) = self.selector.resolve(schema) else {
            return Vec::new();
        };
        cols.iter()
            .filter_map(|name| training.column(name).and_then(|c| c.as_categorical()))
            .map(|(_, levels)| {
                let k = levels.len();
                (
                    if self.one_hot { k } else { k.saturating_sub(1) },
                    1.0 - 1.0 / k as f64,
                )
            })
            .collect()
    }

    fn toggle_sparse(&self, choice: bool) -> Option<Box<dyn Step>> {
        match self.sparse {
            SparseOutput::Auto => Some(Box::new(self.clone().sparse(if choice {
                SparseOutput::Yes
            } else {
                SparseOutput::No
            }))),
            _ => None,
        }
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
        // Output schema: recipes removes the encoded columns and appends the indicator
        // columns after the remaining columns, in encoding order.
        let mut fields: Vec<Field> = schema
            .fields
            .iter()
            .filter(|f| !encodings.iter().any(|(n, _, _)| n == &f.name))
            .cloned()
            .collect();
        for (name, _, outs) in &encodings {
            let role = schema.field(name).map(|f| f.role).unwrap_or(Role::Predictor);
            for o in outs {
                fields.push(Field {
                    name: o.clone(),
                    kind: ColumnKind::Numeric,
                    role,
                });
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
            sparse: self.sparse == SparseOutput::Yes,
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
        for (name, col) in data.iter_shared() {
            if !self.encodings.iter().any(|(n, _, _)| n == name) {
                out.insert_shared(name, Arc::clone(col))?;
            }
        }
        for (name, levels, out_names) in &self.encodings {
            let col = data
                .column(name)
                .ok_or_else(|| Error::SchemaMismatch(format!("column `{name}` missing at bake time")))?;
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
                        // sparse = "yes" emits sparse vectors (recipes 1.3.2); "auto" and
                        // "no" bake dense indicators
                        sparse: self.sparse,
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
    fn sparse_output_is_toggled_like_recipes_and_flags_the_indicators() {
        let (t, s) = table();
        let auto = StepDummy::new(Selector::AllNominalPredictors);
        // "auto" bakes dense columns; the toggle sets "yes" / "no"
        assert!(!auto
            .prep(&t, &s)
            .unwrap()
            .bake(&t)
            .unwrap()
            .column("species_b")
            .unwrap()
            .is_sparse());
        let yes = auto.toggle_sparse(true).unwrap();
        let b = yes.prep(&t, &s).unwrap().bake(&t).unwrap();
        assert!(b.column("species_b").unwrap().is_sparse() && b.column("species_c").unwrap().is_sparse());
        assert!(!b.column("y").unwrap().is_sparse());
        let no = auto.toggle_sparse(false).unwrap();
        assert!(!no
            .prep(&t, &s)
            .unwrap()
            .bake(&t)
            .unwrap()
            .column("species_b")
            .unwrap()
            .is_sparse());
        // an explicit setting is left alone (.recipes_toggle_sparse_args)
        assert!(auto
            .clone()
            .sparse(SparseOutput::No)
            .toggle_sparse(true)
            .is_none());
        assert!(auto
            .clone()
            .sparse(SparseOutput::Yes)
            .toggle_sparse(false)
            .is_none());
        // the fitted state round-trips the flag
        let state = yes.prep(&t, &s).unwrap().state().unwrap();
        let f: FittedDummy = serde_json::from_value(state).unwrap();
        assert!(f.sparse);
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
