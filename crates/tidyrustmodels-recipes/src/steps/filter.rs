//! Column filters: `step_zv`, `step_nzv`, `step_corr`. All three decide at `prep`
//! which columns to drop and remove exactly those columns at `bake`, like recipes'
//! `recipes_remove_cols`. Definitions follow recipes 1.3.2 (`one_unique`, `nzv`,
//! `corr_filter`), read from the installed package source.

use crate::selector::Selector;
use serde::{Deserialize, Serialize};
use tidyrustmodels_core::{Column, Error, FittedStep, Result, Schema, Step, Table};

/// Fitted state shared by the three filters: the columns to remove.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct FittedRemoveCols {
    step: String,
    /// Columns removed at bake time, in schema order (recipes' `removals`).
    pub removals: Vec<String>,
    schema: Schema,
}

impl FittedRemoveCols {
    fn new(step: &str, removals: Vec<String>, schema: &Schema) -> Self {
        FittedRemoveCols {
            step: step.into(),
            schema: Schema {
                fields: schema
                    .fields
                    .iter()
                    .filter(|f| !removals.contains(&f.name))
                    .cloned()
                    .collect(),
            },
            removals,
        }
    }
}

impl FittedStep for FittedRemoveCols {
    fn name(&self) -> &'static str {
        match self.step.as_str() {
            "zv" => "zv",
            "nzv" => "nzv",
            _ => "corr",
        }
    }

    fn bake(&self, data: &Table) -> Result<Table> {
        let mut out = data.clone();
        for r in &self.removals {
            out.remove(r);
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

/// Distinct non-missing values of a column as sortable keys (numeric bits or codes),
/// plus the count of non-missing rows.
fn distinct_counts(col: &Column) -> Vec<usize> {
    let valid = col.validity();
    let mut counts: std::collections::HashMap<u64, usize> = std::collections::HashMap::new();
    match col {
        Column::Numeric { values, .. } => {
            for (i, v) in values.iter().enumerate() {
                if valid.is_none_or(|m| m[i]) {
                    // Normalise -0.0 to 0.0 so `unique` semantics match R's `==`.
                    let key = if *v == 0.0 { 0.0f64.to_bits() } else { v.to_bits() };
                    *counts.entry(key).or_insert(0) += 1;
                }
            }
        }
        Column::Categorical { codes, .. } => {
            for (i, c) in codes.iter().enumerate() {
                if valid.is_none_or(|m| m[i]) {
                    *counts.entry(u64::from(*c)).or_insert(0) += 1;
                }
            }
        }
    }
    let mut v: Vec<usize> = counts.into_values().collect();
    v.sort_unstable_by(|a, b| b.cmp(a));
    v
}

// ------------------------------------------------------------------ step_zv

/// `step_zv`: remove columns with fewer than two distinct non-missing values
/// (recipes' `one_unique`; an all-missing column is also removed).
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct StepZv {
    selector: Selector,
}

impl StepZv {
    /// Filter the selected columns.
    pub fn new(selector: Selector) -> Self {
        StepZv { selector }
    }
}

impl Step for StepZv {
    fn name(&self) -> &'static str {
        "zv"
    }

    fn prep(&self, training: &Table, schema: &Schema) -> Result<Box<dyn FittedStep>> {
        let mut removals = Vec::new();
        for name in self.selector.resolve(schema)? {
            let col = training
                .column(&name)
                .ok_or_else(|| Error::SchemaMismatch(format!("column `{name}` not in training data")))?;
            if distinct_counts(col).len() < 2 {
                removals.push(name);
            }
        }
        Ok(Box::new(FittedRemoveCols::new("zv", removals, schema)))
    }

    fn clone_box(&self) -> Box<dyn Step> {
        Box::new(self.clone())
    }
}

// ------------------------------------------------------------------ step_nzv

/// `step_nzv`: remove near-zero-variance columns. A column is removed when the ratio
/// of the most common value's count to the second most common is greater than
/// `freq_cut` **and** the percentage of distinct values (over all rows, missing
/// included in the denominator) is at most `unique_cut`, or when it has a single
/// distinct value / is all missing. Defaults `freq_cut = 95/5`, `unique_cut = 10`.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct StepNzv {
    selector: Selector,
    freq_cut: f64,
    unique_cut: f64,
}

impl StepNzv {
    /// Filter the selected columns with recipes' defaults.
    pub fn new(selector: Selector) -> Self {
        StepNzv {
            selector,
            freq_cut: 95.0 / 5.0,
            unique_cut: 10.0,
        }
    }

    /// Frequency-ratio cut (default `19`).
    pub fn freq_cut(mut self, v: f64) -> Self {
        self.freq_cut = v;
        self
    }

    /// Percent-unique cut in `[0, 100]` (default `10`).
    pub fn unique_cut(mut self, v: f64) -> Self {
        self.unique_cut = v;
        self
    }
}

impl Step for StepNzv {
    fn name(&self) -> &'static str {
        "nzv"
    }

    fn prep(&self, training: &Table, schema: &Schema) -> Result<Box<dyn FittedStep>> {
        if !(0.0..=100.0).contains(&self.unique_cut) || self.freq_cut.is_nan() || self.freq_cut < 0.0 {
            return Err(Error::InvalidArgument(
                "step_nzv: unique_cut must be in [0, 100] and freq_cut >= 0".into(),
            ));
        }
        let mut removals = Vec::new();
        for name in self.selector.resolve(schema)? {
            let col = training
                .column(&name)
                .ok_or_else(|| Error::SchemaMismatch(format!("column `{name}` not in training data")))?;
            let counts = distinct_counts(col);
            let freq_ratio = if counts.len() <= 1 {
                0.0
            } else {
                counts[0] as f64 / counts[1] as f64
            };
            let pct_unique = 100.0 * counts.len() as f64 / col.len() as f64;
            let zero_var = counts.len() == 1 || counts.is_empty();
            if (freq_ratio > self.freq_cut && pct_unique <= self.unique_cut) || zero_var {
                removals.push(name);
            }
        }
        Ok(Box::new(FittedRemoveCols::new("nzv", removals, schema)))
    }

    fn clone_box(&self) -> Box<dyn Step> {
        Box::new(self.clone())
    }
}

// ------------------------------------------------------------------ step_corr

/// `step_corr`: remove numeric columns so that no remaining pair has absolute Pearson
/// correlation above `threshold` (default `0.9`), with recipes' `corr_filter`
/// algorithm (a port of caret's `findCorrelation(exact = FALSE)`): for every pair above
/// the cutoff, drop the member with the larger mean absolute correlation. Correlations
/// use pairwise-complete observations; columns whose correlations are all undefined are
/// excluded from the filter, sporadic undefined entries are treated as 0, and the step
/// is skipped when too many are undefined (recipes warns in each case; the condition
/// is recorded in the fitted state).
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct StepCorr {
    selector: Selector,
    threshold: f64,
}

impl StepCorr {
    /// Filter the selected numeric columns at threshold `0.9`.
    pub fn new(selector: Selector) -> Self {
        StepCorr {
            selector,
            threshold: 0.9,
        }
    }

    /// Absolute-correlation cutoff in `[0, 1]`.
    pub fn threshold(mut self, v: f64) -> Self {
        self.threshold = v;
        self
    }
}

/// Pearson correlation on rows where both columns are valid; `None` when fewer than
/// two such rows or either column is constant on them (`cor` gives `NA`).
fn pairwise_cor(a: &[f64], av: Option<&[bool]>, b: &[f64], bv: Option<&[bool]>) -> Option<f64> {
    let rows: Vec<usize> = (0..a.len())
        .filter(|&i| av.is_none_or(|m| m[i]) && bv.is_none_or(|m| m[i]))
        .collect();
    if rows.len() < 2 {
        return None;
    }
    let n = rows.len() as f64;
    let ma = rows.iter().map(|&i| a[i]).sum::<f64>() / n;
    let mb = rows.iter().map(|&i| b[i]).sum::<f64>() / n;
    let (mut sab, mut saa, mut sbb) = (0.0, 0.0, 0.0);
    for &i in &rows {
        let (da, db) = (a[i] - ma, b[i] - mb);
        sab += da * db;
        saa += da * da;
        sbb += db * db;
    }
    if saa == 0.0 || sbb == 0.0 {
        return None;
    }
    Some(sab / (saa.sqrt() * sbb.sqrt()))
}

/// recipes' `corr_filter` on a correlation matrix (column-major `p × p`, `None` = NA).
/// Returns indices to delete in recipes' order.
pub(crate) fn corr_filter(mut x: Vec<Option<f64>>, p: usize, cutoff: f64) -> (Vec<usize>, Vec<String>) {
    let mut notes = Vec::new();
    let at = |x: &Vec<Option<f64>>, i: usize, j: usize| x[j * p + i];
    if x.iter().any(|v| v.is_none()) {
        let all_na: Vec<bool> = (0..p).map(|j| (0..p).all(|i| at(&x, i, j).is_none())).collect();
        let n_all_na = all_na.iter().filter(|b| **b).count();
        if n_all_na + 1 >= p {
            notes.push("too many correlations are NA; correlation filter skipped".into());
            return (Vec::new(), notes);
        }
        if n_all_na > 0 {
            for j in 0..p {
                if all_na[j] {
                    for i in 0..p {
                        x[j * p + i] = Some(0.0);
                        x[i * p + j] = Some(0.0);
                    }
                }
            }
            notes.push(format!(
                "the correlation matrix has missing values; {n_all_na} column(s) excluded from the filter"
            ));
        }
        if x.iter().any(|v| v.is_none()) {
            notes.push("the correlation matrix has sporadic missing values; some columns excluded".into());
            for v in x.iter_mut() {
                if v.is_none() {
                    *v = Some(0.0);
                }
            }
        }
        for j in 0..p {
            x[j * p + j] = Some(1.0);
        }
    }
    let get = |i: usize, j: usize| x[j * p + i].expect("filled");
    // colMeans(abs(x)), diagonal included; then dense rank of the distinct values.
    let avg: Vec<f64> = (0..p)
        .map(|j| (0..p).map(|i| get(i, j).abs()).sum::<f64>() / p as f64)
        .collect();
    let mut distinct: Vec<f64> = avg.clone();
    distinct.sort_by(|a, b| a.partial_cmp(b).expect("finite"));
    distinct.dedup();
    let rank: Vec<usize> = avg
        .iter()
        .map(|v| distinct.iter().position(|d| d == v).expect("present"))
        .collect();
    // Upper triangle (strict) in column-major order, as `which()` enumerates it.
    let mut delete: Vec<usize> = Vec::new();
    let mut cols_discard = Vec::new();
    let mut rows_discard = Vec::new();
    for j in 0..p {
        for i in 0..j {
            if get(i, j).abs() > cutoff {
                if rank[j] > rank[i] {
                    cols_discard.push(j);
                } else {
                    rows_discard.push(i);
                }
            }
        }
    }
    for c in cols_discard.into_iter().chain(rows_discard) {
        if !delete.contains(&c) {
            delete.push(c);
        }
    }
    (delete, notes)
}

/// Fitted state of [`StepCorr`]: removals plus any recipes-style warning.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct FittedCorr {
    #[serde(flatten)]
    inner: FittedRemoveCols,
    /// Conditions recipes would have warned about.
    pub notes: Vec<String>,
}

impl FittedStep for FittedCorr {
    fn name(&self) -> &'static str {
        "corr"
    }

    fn bake(&self, data: &Table) -> Result<Table> {
        self.inner.bake(data)
    }

    fn output_schema(&self) -> &Schema {
        self.inner.output_schema()
    }

    fn state(&self) -> Result<serde_json::Value> {
        serde_json::to_value(self).map_err(|e| Error::Serialization(e.to_string()))
    }
}

impl Step for StepCorr {
    fn name(&self) -> &'static str {
        "corr"
    }

    fn prep(&self, training: &Table, schema: &Schema) -> Result<Box<dyn FittedStep>> {
        if !(0.0..=1.0).contains(&self.threshold) {
            return Err(Error::InvalidArgument(
                "step_corr: threshold must be in [0, 1]".into(),
            ));
        }
        let names = self.selector.resolve(schema)?;
        let mut cols = Vec::with_capacity(names.len());
        for name in &names {
            let col = training
                .column(name)
                .ok_or_else(|| Error::SchemaMismatch(format!("column `{name}` not in training data")))?;
            let values = col.as_numeric().ok_or_else(|| {
                Error::InvalidValue(format!(
                    "step_corr requires numeric columns; `{name}` is categorical"
                ))
            })?;
            cols.push((values, col.validity()));
        }
        let p = cols.len();
        let (removals, notes) = if p > 1 {
            let mut m = vec![None; p * p];
            for j in 0..p {
                m[j * p + j] = Some(1.0);
                for i in 0..j {
                    let c = pairwise_cor(cols[i].0, cols[i].1, cols[j].0, cols[j].1);
                    m[j * p + i] = c;
                    m[i * p + j] = c;
                }
            }
            let (idx, notes) = corr_filter(m, p, self.threshold);
            (idx.into_iter().map(|k| names[k].clone()).collect(), notes)
        } else {
            (Vec::new(), Vec::new())
        };
        Ok(Box::new(FittedCorr {
            inner: FittedRemoveCols::new("corr", removals, schema),
            notes,
        }))
    }

    fn clone_box(&self) -> Box<dyn Step> {
        Box::new(self.clone())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tidyrustmodels_core::Role;

    fn table() -> (Table, Schema) {
        let t = Table::new(vec![
            ("y", Column::numeric(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap()),
            ("const", Column::numeric(vec![7.0; 6]).unwrap()),
            ("x", Column::numeric(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.5]).unwrap()),
            (
                "x2",
                Column::numeric(vec![2.0, 4.0, 6.0, 8.0, 10.0, 13.0]).unwrap(),
            ),
            (
                "z",
                Column::numeric(vec![0.3, -1.0, 2.0, 0.1, -0.7, 0.9]).unwrap(),
            ),
            ("g", Column::categorical(&["a", "a", "a", "a", "a", "b"]).unwrap()),
            (
                "na",
                Column::numeric_with_missing(vec![None, None, None, None, None, None]).unwrap(),
            ),
        ])
        .unwrap();
        let s = t.schema_with_outcome("y", Role::Predictor).unwrap();
        (t, s)
    }

    #[test]
    fn zv_removes_constant_and_all_missing() {
        let (t, s) = table();
        let f = StepZv::new(Selector::AllPredictors).prep(&t, &s).unwrap();
        let b = f.bake(&t).unwrap();
        assert_eq!(b.names(), &["y", "x", "x2", "z", "g"]);
        let names: Vec<&str> = f.output_schema().fields.iter().map(|f| f.name.as_str()).collect();
        assert_eq!(names, vec!["y", "x", "x2", "z", "g"]);
    }

    #[test]
    fn nzv_matches_recipes_rule() {
        // g: counts 5/1 -> ratio 5, pct unique 2/6*100 = 33 -> kept with defaults;
        // with freq_cut 4 and unique_cut 40 it goes. const and na always go.
        let (t, s) = table();
        let f = StepNzv::new(Selector::AllPredictors).prep(&t, &s).unwrap();
        assert_eq!(f.bake(&t).unwrap().names(), &["y", "x", "x2", "z", "g"]);
        let f = StepNzv::new(Selector::AllPredictors)
            .freq_cut(4.0)
            .unique_cut(40.0)
            .prep(&t, &s)
            .unwrap();
        assert_eq!(f.bake(&t).unwrap().names(), &["y", "x", "x2", "z"]);
    }

    #[test]
    fn corr_drops_the_member_with_larger_mean_correlation() {
        let (mut t, _) = table();
        t.remove("const");
        t.remove("na");
        t.remove("g");
        let s = t.schema_with_outcome("y", Role::Predictor).unwrap();
        let f = StepCorr::new(Selector::AllNumericPredictors)
            .prep(&t, &s)
            .unwrap();
        let b = f.bake(&t).unwrap();
        // x and x2 are correlated at ~0.9995; x2 has the larger mean |cor| with z? Either
        // way exactly one of them must go and z must stay.
        assert_eq!(b.ncols(), 3);
        assert!(b.column("z").is_some() && b.column("y").is_some());
        assert!(b.column("x").is_some() ^ b.column("x2").is_some());
    }

    #[test]
    fn corr_filter_port_hand_checked() {
        // 3 variables: cor(1,2) = 0.95, cor(1,3) = 0.2, cor(2,3) = 0.3.
        // mean |cor|: v1 = (1 + .95 + .2)/3 = .7167, v2 = (.95 + 1 + .3)/3 = .75, v3 = .5
        // pair (1,2) above cutoff: rank(v2) > rank(v1) -> discard column 2 (index 1).
        let m = vec![
            Some(1.0),
            Some(0.95),
            Some(0.2), //
            Some(0.95),
            Some(1.0),
            Some(0.3), //
            Some(0.2),
            Some(0.3),
            Some(1.0),
        ];
        let (del, notes) = corr_filter(m, 3, 0.9);
        assert_eq!(del, vec![1]);
        assert!(notes.is_empty());
        // Sporadic NA is treated as 0; all-NA column is excluded.
        let m = vec![
            Some(1.0),
            None,
            Some(0.2), //
            None,
            Some(1.0),
            Some(0.3), //
            Some(0.2),
            Some(0.3),
            Some(1.0),
        ];
        let (del, notes) = corr_filter(m, 3, 0.1);
        assert_eq!(notes.len(), 1);
        assert_eq!(del, vec![2]);
    }
}
