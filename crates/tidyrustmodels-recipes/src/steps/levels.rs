//! Categorical level handling: `step_other`, `step_novel`, `step_unknown`.
//! Definitions follow recipes 1.3.2 (`keep_levels`, `bake.step_other`,
//! `bake.step_novel`, `bake.step_unknown`), read from the installed package source.

use crate::selector::Selector;
use serde::{Deserialize, Serialize};
use tidyrustmodels_core::{Column, Error, FittedStep, Result, Schema, Step, Table};

/// Labels of a categorical column per row (`None` = missing), plus its levels.
fn labels_of<'a>(name: &str, col: &'a Column) -> Result<(Vec<Option<&'a str>>, &'a [String])> {
    let (codes, levels) = col.as_categorical().ok_or_else(|| {
        Error::InvalidValue(format!(
            "this step requires categorical columns; `{name}` is numeric"
        ))
    })?;
    let valid = col.validity();
    let labels = codes
        .iter()
        .enumerate()
        .map(|(i, c)| {
            if valid.is_none_or(|m| m[i]) {
                Some(levels[*c as usize].as_str())
            } else {
                None
            }
        })
        .collect();
    Ok((labels, levels))
}

fn get_col<'a>(table: &'a Table, name: &str, when: &str) -> Result<&'a Column> {
    table
        .column(name)
        .ok_or_else(|| Error::SchemaMismatch(format!("column `{name}` missing at {when} time")))
}

// ------------------------------------------------------------------ step_other

/// `step_other`: collapse infrequent levels into one `other` level. With
/// `threshold < 1` a level is infrequent when its share of the non-missing training
/// rows is below `threshold` (default `0.05`); with `threshold >= 1` when its count is
/// below `threshold`. If every level is infrequent, the most frequent one is kept. If no
/// level is infrequent the column is left untouched (no `other` level is added), as in
/// recipes. Missing values stay missing.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct StepOther {
    selector: Selector,
    threshold: f64,
    other: String,
}

impl StepOther {
    /// recipes' defaults: `threshold = 0.05`, `other = "other"`.
    pub fn new(selector: Selector) -> Self {
        StepOther {
            selector,
            threshold: 0.05,
            other: "other".into(),
        }
    }

    /// Proportion (`< 1`) or count (`>= 1`) below which a level is collapsed.
    pub fn threshold(mut self, v: f64) -> Self {
        self.threshold = v;
        self
    }

    /// Label of the collapsed level.
    pub fn other(mut self, label: impl Into<String>) -> Self {
        self.other = label.into();
        self
    }
}

/// Per-column state of [`StepOther`] (recipes' `keep_levels` object).
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct OtherObject {
    /// Column name.
    pub column: String,
    /// Levels retained, in original level order.
    pub keep: Vec<String>,
    /// Whether any level was collapsed (if not, bake leaves the column untouched).
    pub collapse: bool,
    /// Label of the collapsed level.
    pub other: String,
}

/// Fitted state of [`StepOther`].
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct FittedOther {
    step: String,
    /// One entry per selected column.
    pub objects: Vec<OtherObject>,
    schema: Schema,
}

impl Step for StepOther {
    fn name(&self) -> &'static str {
        "other"
    }

    fn prep(&self, training: &Table, schema: &Schema) -> Result<Box<dyn FittedStep>> {
        if !self.threshold.is_finite() || self.threshold < 0.0 {
            return Err(Error::InvalidArgument(
                "step_other: threshold must be >= 0".into(),
            ));
        }
        if self.threshold >= 1.0 && self.threshold.fract() != 0.0 {
            return Err(Error::InvalidArgument(
                "step_other: a threshold >= 1 must be a whole number (a count)".into(),
            ));
        }
        let mut objects = Vec::new();
        for name in self.selector.resolve(schema)? {
            let col = get_col(training, &name, "prep")?;
            let (labels, levels) = labels_of(&name, col)?;
            let mut counts = vec![0usize; levels.len()];
            let mut n_valid = 0usize;
            for l in labels.iter().flatten() {
                counts[levels.iter().position(|lv| lv == l).expect("level")] += 1;
                n_valid += 1;
            }
            // xtab sorted decreasing (stable for ties, as R's `sort` on a table).
            let mut order: Vec<usize> = (0..levels.len()).collect();
            order.sort_by(|a, b| counts[*b].cmp(&counts[*a]));
            let value = |k: usize| -> f64 {
                if self.threshold < 1.0 {
                    counts[k] as f64 / n_valid.max(1) as f64
                } else {
                    counts[k] as f64
                }
            };
            let dropped: Vec<usize> = order
                .iter()
                .copied()
                .filter(|&k| value(k) < self.threshold)
                .collect();
            let mut keepers: Vec<usize> = order.iter().copied().filter(|k| !dropped.contains(k)).collect();
            if keepers.is_empty() {
                keepers.push(order[0]);
            }
            let keep: Vec<String> = levels
                .iter()
                .enumerate()
                .filter(|(k, _)| keepers.contains(k))
                .map(|(_, l)| l.clone())
                .collect();
            if keep.contains(&self.other) {
                return Err(Error::InvalidArgument(format!(
                    "step_other: the level `{}` of `{name}` is already a retained level; choose another `other` label",
                    self.other
                )));
            }
            objects.push(OtherObject {
                column: name,
                keep,
                collapse: !dropped.is_empty(),
                other: self.other.clone(),
            });
        }
        Ok(Box::new(FittedOther {
            step: "other".into(),
            objects,
            schema: schema.clone(),
        }))
    }

    fn clone_box(&self) -> Box<dyn Step> {
        Box::new(self.clone())
    }
}

impl FittedStep for FittedOther {
    fn name(&self) -> &'static str {
        "other"
    }

    fn bake(&self, data: &Table) -> Result<Table> {
        let mut out = data.clone();
        for o in &self.objects {
            let col = get_col(data, &o.column, "bake")?;
            if !o.collapse {
                continue;
            }
            let (labels, _) = labels_of(&o.column, col)?;
            let mapped: Vec<Option<&str>> = labels
                .iter()
                .map(|l| {
                    l.map(|s| {
                        if o.keep.iter().any(|k| k == s) {
                            s
                        } else {
                            o.other.as_str()
                        }
                    })
                })
                .collect();
            let mut levels = o.keep.clone();
            levels.push(o.other.clone());
            out.insert(
                o.column.clone(),
                Column::categorical_with_levels_opt(&mapped, levels)?,
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

// ------------------------------------------------------------------ step_novel / step_unknown

/// `step_novel`: add a level (`"new"` by default) to each selected categorical column
/// so that labels unseen in training are coded to it at bake time instead of failing.
/// Missing values stay missing. Prep fails if the training data already contains the
/// label.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct StepNovel {
    selector: Selector,
    new_level: String,
}

impl StepNovel {
    /// recipes' default label `"new"`.
    pub fn new(selector: Selector) -> Self {
        StepNovel {
            selector,
            new_level: "new".into(),
        }
    }

    /// Label for unseen values.
    pub fn new_level(mut self, label: impl Into<String>) -> Self {
        self.new_level = label.into();
        self
    }
}

/// `step_unknown`: recode missing values of each selected categorical column to a level
/// (`"unknown"` by default). Labels unseen in training become missing at bake time
/// (recipes warns and yields `NA`). Prep fails if the training data already contains
/// the label.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct StepUnknown {
    selector: Selector,
    new_level: String,
}

impl StepUnknown {
    /// recipes' default label `"unknown"`.
    pub fn new(selector: Selector) -> Self {
        StepUnknown {
            selector,
            new_level: "unknown".into(),
        }
    }

    /// Label for missing values.
    pub fn new_level(mut self, label: impl Into<String>) -> Self {
        self.new_level = label.into();
        self
    }
}

/// Fitted state shared by [`StepNovel`] and [`StepUnknown`]: training levels per column
/// plus the added label.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct FittedLevelAdd {
    step: String,
    /// `(column, training levels)`.
    pub objects: Vec<(String, Vec<String>)>,
    /// The added level.
    pub new_level: String,
    schema: Schema,
}

fn prep_level_add(
    step: &str,
    selector: &Selector,
    new_level: &str,
    training: &Table,
    schema: &Schema,
) -> Result<Box<dyn FittedStep>> {
    if new_level.is_empty() {
        return Err(Error::InvalidArgument(format!(
            "step_{step}: new_level must not be empty"
        )));
    }
    let mut objects = Vec::new();
    for name in selector.resolve(schema)? {
        let col = get_col(training, &name, "prep")?;
        let (_, levels) = labels_of(&name, col)?;
        if levels.iter().any(|l| l == new_level) {
            return Err(Error::InvalidArgument(format!(
                "step_{step}: column `{name}` already contains the level `{new_level}`"
            )));
        }
        objects.push((name, levels.to_vec()));
    }
    Ok(Box::new(FittedLevelAdd {
        step: step.into(),
        objects,
        new_level: new_level.into(),
        schema: schema.clone(),
    }))
}

impl Step for StepNovel {
    fn name(&self) -> &'static str {
        "novel"
    }

    fn prep(&self, training: &Table, schema: &Schema) -> Result<Box<dyn FittedStep>> {
        prep_level_add("novel", &self.selector, &self.new_level, training, schema)
    }

    fn clone_box(&self) -> Box<dyn Step> {
        Box::new(self.clone())
    }
}

impl Step for StepUnknown {
    fn name(&self) -> &'static str {
        "unknown"
    }

    fn prep(&self, training: &Table, schema: &Schema) -> Result<Box<dyn FittedStep>> {
        prep_level_add("unknown", &self.selector, &self.new_level, training, schema)
    }

    fn clone_box(&self) -> Box<dyn Step> {
        Box::new(self.clone())
    }
}

impl FittedStep for FittedLevelAdd {
    fn name(&self) -> &'static str {
        if self.step == "novel" {
            "novel"
        } else {
            "unknown"
        }
    }

    fn bake(&self, data: &Table) -> Result<Table> {
        let mut out = data.clone();
        let novel = self.step == "novel";
        for (name, train_levels) in &self.objects {
            let col = get_col(data, name, "bake")?;
            let (labels, _) = labels_of(name, col)?;
            let mut levels = train_levels.clone();
            levels.push(self.new_level.clone());
            let mapped: Vec<Option<&str>> = labels
                .iter()
                .map(|l| match l {
                    Some(s) if train_levels.iter().any(|t| t == s) => Some(*s),
                    // novel: unseen -> new level, missing stays missing
                    Some(_) if novel => Some(self.new_level.as_str()),
                    None if novel => None,
                    // unknown: missing -> new level, unseen -> missing
                    None => Some(self.new_level.as_str()),
                    Some(_) => None,
                })
                .collect();
            out.insert(
                name.clone(),
                Column::categorical_with_levels_opt(&mapped, levels)?,
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
    use tidyrustmodels_core::Role;

    fn table() -> (Table, Schema) {
        let g = ["a", "a", "a", "a", "a", "a", "b", "b", "c", "d"];
        let t = Table::new(vec![
            ("y", Column::numeric(vec![0.0; 10]).unwrap()),
            ("g", Column::categorical(&g).unwrap()),
        ])
        .unwrap();
        let s = t.schema_with_outcome("y", Role::Predictor).unwrap();
        (t, s)
    }

    fn labels(col: &Column) -> Vec<Option<String>> {
        let (codes, levels) = col.as_categorical().unwrap();
        codes
            .iter()
            .enumerate()
            .map(|(i, c)| {
                if col.is_missing(i) {
                    None
                } else {
                    Some(levels[*c as usize].clone())
                }
            })
            .collect()
    }

    #[test]
    fn other_collapses_rare_levels_by_proportion_and_count() {
        let (t, s) = table();
        // shares: a .6, b .2, c .1, d .1 -> threshold .15 drops c, d
        let f = StepOther::new(Selector::columns(&["g"]))
            .threshold(0.15)
            .prep(&t, &s)
            .unwrap();
        let b = f.bake(&t).unwrap();
        let (_, levels) = b.column("g").unwrap().as_categorical().unwrap();
        assert_eq!(levels, &["a", "b", "other"]);
        assert_eq!(labels(b.column("g").unwrap())[9].as_deref(), Some("other"));
        // count threshold 2: c and d (1 each) collapse
        let f = StepOther::new(Selector::columns(&["g"]))
            .threshold(2.0)
            .prep(&t, &s)
            .unwrap();
        let baked = f.bake(&t).unwrap();
        let (_, levels) = baked.column("g").unwrap().as_categorical().unwrap();
        assert_eq!(levels, &["a", "b", "other"]);
        // nothing rare at the default 0.05 -> column untouched, no `other` level
        let f = StepOther::new(Selector::columns(&["g"])).prep(&t, &s).unwrap();
        let baked = f.bake(&t).unwrap();
        let (_, levels) = baked.column("g").unwrap().as_categorical().unwrap();
        assert_eq!(levels, &["a", "b", "c", "d"]);
        // all rare -> most frequent kept
        let f = StepOther::new(Selector::columns(&["g"]))
            .threshold(0.99)
            .prep(&t, &s)
            .unwrap();
        let baked = f.bake(&t).unwrap();
        let (_, levels) = baked.column("g").unwrap().as_categorical().unwrap();
        assert_eq!(levels, &["a", "other"]);
        assert!(StepOther::new(Selector::columns(&["g"]))
            .threshold(0.15)
            .other("a")
            .prep(&t, &s)
            .is_err());
    }

    #[test]
    fn novel_and_unknown_handle_unseen_and_missing() {
        let (t, s) = table();
        let novel = StepNovel::new(Selector::columns(&["g"])).prep(&t, &s).unwrap();
        let unknown = StepUnknown::new(Selector::columns(&["g"])).prep(&t, &s).unwrap();
        let test = Table::new(vec![
            ("y", Column::numeric(vec![0.0; 3]).unwrap()),
            (
                "g",
                Column::categorical_with_levels_opt(
                    &[Some("b"), Some("zebra"), None],
                    vec!["b".into(), "zebra".into()],
                )
                .unwrap(),
            ),
        ])
        .unwrap();
        let b = novel.bake(&test).unwrap();
        let col = b.column("g").unwrap();
        assert_eq!(col.as_categorical().unwrap().1, &["a", "b", "c", "d", "new"]);
        assert_eq!(labels(col), vec![Some("b".into()), Some("new".into()), None]);
        let b = unknown.bake(&test).unwrap();
        let col = b.column("g").unwrap();
        assert_eq!(col.as_categorical().unwrap().1, &["a", "b", "c", "d", "unknown"]);
        assert_eq!(labels(col), vec![Some("b".into()), None, Some("unknown".into())]);
        assert!(StepNovel::new(Selector::columns(&["g"]))
            .new_level("a")
            .prep(&t, &s)
            .is_err());
    }
}
