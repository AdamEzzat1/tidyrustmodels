//! Owned columnar table with explicit missing-value masks.
//!
//! Missing values are always represented by a validity mask, never by sentinel values.
//! Numeric columns reject non-finite values at construction unless the caller opts in
//! through [`Column::numeric_allow_nonfinite`].

use crate::error::{Error, Result};
use serde::{Deserialize, Serialize};

/// The kind of a column.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum ColumnKind {
    /// 64-bit floating point.
    Numeric,
    /// Integer codes into an ordered list of levels.
    Categorical,
}

/// The modelling role of a column, mirroring recipes roles.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum Role {
    /// Used as a model input.
    Predictor,
    /// The variable being predicted.
    Outcome,
    /// Carried through but never modelled.
    Id,
    /// Dropped from modelling.
    Ignored,
}

/// One typed column.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub enum Column {
    /// Floating-point values with an optional validity mask (`None` = all valid).
    Numeric {
        /// Values; entries whose mask bit is `false` are unspecified.
        values: Vec<f64>,
        /// `Some(mask)` where `mask[i] == false` means missing.
        valid: Option<Vec<bool>>,
    },
    /// Categorical codes into `levels`, with an optional validity mask.
    Categorical {
        /// Code per row; `codes[i] < levels.len()` whenever valid.
        codes: Vec<u32>,
        /// Level labels in order. `levels[0]` is the event level for binary outcomes.
        levels: Vec<String>,
        /// `Some(mask)` where `mask[i] == false` means missing.
        valid: Option<Vec<bool>>,
    },
}

impl Column {
    /// A numeric column with no missing values. Rejects non-finite entries.
    pub fn numeric(values: Vec<f64>) -> Result<Self> {
        if let Some(i) = values.iter().position(|v| !v.is_finite()) {
            return Err(Error::InvalidValue(format!(
                "numeric column contains a non-finite value at row {i}; use `numeric_with_missing` for missing data"
            )));
        }
        Ok(Column::Numeric { values, valid: None })
    }

    /// A numeric column that permits `NaN`/infinite entries (they are *values*, not
    /// missing). Preprocessing steps still reject non-finite inputs where their
    /// mathematics requires it.
    pub fn numeric_allow_nonfinite(values: Vec<f64>) -> Self {
        Column::Numeric { values, valid: None }
    }

    /// A numeric column from optional values; `None` becomes missing.
    pub fn numeric_with_missing(values: Vec<Option<f64>>) -> Result<Self> {
        let mut out = Vec::with_capacity(values.len());
        let mut valid = Vec::with_capacity(values.len());
        let mut any_missing = false;
        for (i, v) in values.into_iter().enumerate() {
            match v {
                Some(x) if x.is_finite() => {
                    out.push(x);
                    valid.push(true);
                }
                Some(_) => {
                    return Err(Error::InvalidValue(format!(
                        "numeric column contains a non-finite value at row {i}"
                    )))
                }
                None => {
                    out.push(0.0);
                    valid.push(false);
                    any_missing = true;
                }
            }
        }
        Ok(Column::Numeric {
            values: out,
            valid: if any_missing { Some(valid) } else { None },
        })
    }

    /// A categorical column whose levels are the distinct labels sorted bytewise.
    ///
    /// R's `factor()` sorts levels in the session locale; for parity fixtures supply
    /// explicit levels with [`Column::categorical_with_levels`].
    pub fn categorical<S: AsRef<str>>(labels: &[S]) -> Result<Self> {
        let mut levels: Vec<String> = labels.iter().map(|s| s.as_ref().to_string()).collect();
        levels.sort();
        levels.dedup();
        let opt: Vec<Option<&str>> = labels.iter().map(|s| Some(s.as_ref())).collect();
        Self::categorical_with_levels_opt(&opt, levels)
    }

    /// A categorical column with caller-specified level order.
    pub fn categorical_with_levels<S: AsRef<str>>(labels: &[S], levels: Vec<String>) -> Result<Self> {
        let opt: Vec<Option<&str>> = labels.iter().map(|s| Some(s.as_ref())).collect();
        Self::categorical_with_levels_opt(&opt, levels)
    }

    /// A categorical column with explicit levels and optional (missing) labels.
    pub fn categorical_with_levels_opt(labels: &[Option<&str>], levels: Vec<String>) -> Result<Self> {
        if levels.is_empty() {
            return Err(Error::InvalidArgument(
                "categorical column needs at least one level".into(),
            ));
        }
        {
            let mut sorted = levels.clone();
            sorted.sort();
            sorted.dedup();
            if sorted.len() != levels.len() {
                return Err(Error::InvalidArgument("categorical levels must be unique".into()));
            }
        }
        let mut codes = Vec::with_capacity(labels.len());
        let mut valid = Vec::with_capacity(labels.len());
        let mut any_missing = false;
        for (i, l) in labels.iter().enumerate() {
            match l {
                Some(s) => match levels.iter().position(|lv| lv == s) {
                    Some(c) => {
                        codes.push(c as u32);
                        valid.push(true);
                    }
                    None => {
                        return Err(Error::InvalidValue(format!(
                            "label `{s}` at row {i} is not one of the declared levels"
                        )))
                    }
                },
                None => {
                    codes.push(0);
                    valid.push(false);
                    any_missing = true;
                }
            }
        }
        Ok(Column::Categorical {
            codes,
            levels,
            valid: if any_missing { Some(valid) } else { None },
        })
    }

    /// Number of rows.
    pub fn len(&self) -> usize {
        match self {
            Column::Numeric { values, .. } => values.len(),
            Column::Categorical { codes, .. } => codes.len(),
        }
    }

    /// Whether the column has zero rows.
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// The column kind.
    pub fn kind(&self) -> ColumnKind {
        match self {
            Column::Numeric { .. } => ColumnKind::Numeric,
            Column::Categorical { .. } => ColumnKind::Categorical,
        }
    }

    /// The validity mask, if any rows are missing.
    pub fn validity(&self) -> Option<&[bool]> {
        match self {
            Column::Numeric { valid, .. } | Column::Categorical { valid, .. } => valid.as_deref(),
        }
    }

    /// Whether row `i` is missing.
    pub fn is_missing(&self, i: usize) -> bool {
        self.validity().map(|v| !v[i]).unwrap_or(false)
    }

    /// Count of missing rows.
    pub fn n_missing(&self) -> usize {
        self.validity()
            .map(|v| v.iter().filter(|b| !**b).count())
            .unwrap_or(0)
    }

    /// Numeric values (regardless of validity) or `None` for categorical columns.
    pub fn as_numeric(&self) -> Option<&[f64]> {
        match self {
            Column::Numeric { values, .. } => Some(values),
            Column::Categorical { .. } => None,
        }
    }

    /// Categorical codes and levels, or `None` for numeric columns.
    pub fn as_categorical(&self) -> Option<(&[u32], &[String])> {
        match self {
            Column::Categorical { codes, levels, .. } => Some((codes, levels)),
            Column::Numeric { .. } => None,
        }
    }

    /// Select rows by index (duplicates allowed, order preserved).
    pub fn select_rows(&self, rows: &[usize]) -> Result<Column> {
        let n = self.len();
        if let Some(bad) = rows.iter().find(|&&r| r >= n) {
            return Err(Error::InvalidArgument(format!(
                "row index {bad} out of range for {n} rows"
            )));
        }
        let sub_valid =
            |valid: &Option<Vec<bool>>| valid.as_ref().map(|v| rows.iter().map(|&r| v[r]).collect());
        Ok(match self {
            Column::Numeric { values, valid } => Column::Numeric {
                values: rows.iter().map(|&r| values[r]).collect(),
                valid: sub_valid(valid),
            },
            Column::Categorical { codes, levels, valid } => Column::Categorical {
                codes: rows.iter().map(|&r| codes[r]).collect(),
                levels: levels.clone(),
                valid: sub_valid(valid),
            },
        })
    }
}

/// A named, typed column with a modelling role.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct Field {
    /// Column name.
    pub name: String,
    /// Column kind.
    pub kind: ColumnKind,
    /// Modelling role.
    pub role: Role,
}

/// Ordered list of fields describing a table's modelling layout.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize, Default)]
pub struct Schema {
    /// Fields in column order.
    pub fields: Vec<Field>,
}

impl Schema {
    /// Look up a field by name.
    pub fn field(&self, name: &str) -> Option<&Field> {
        self.fields.iter().find(|f| f.name == name)
    }

    /// Names of fields with the given role, in order.
    pub fn names_with_role(&self, role: Role) -> Vec<&str> {
        self.fields
            .iter()
            .filter(|f| f.role == role)
            .map(|f| f.name.as_str())
            .collect()
    }

    /// Verify a table has every field of this schema with the same kind. Extra
    /// columns in the table are allowed (they are ignored downstream).
    pub fn check_table(&self, table: &Table) -> Result<()> {
        for f in &self.fields {
            match table.column(&f.name) {
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
        Ok(())
    }
}

/// An owned table of equal-length columns with unique names.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Table {
    names: Vec<String>,
    columns: Vec<Column>,
    nrows: usize,
}

impl Table {
    /// Build a table; all columns must have equal length and unique names.
    pub fn new(columns: Vec<(impl Into<String>, Column)>) -> Result<Self> {
        let mut names = Vec::with_capacity(columns.len());
        let mut cols = Vec::with_capacity(columns.len());
        let mut nrows: Option<usize> = None;
        for (name, col) in columns {
            let name: String = name.into();
            if names.contains(&name) {
                return Err(Error::InvalidArgument(format!("duplicate column name `{name}`")));
            }
            match nrows {
                None => nrows = Some(col.len()),
                Some(n) if n != col.len() => {
                    return Err(Error::InvalidArgument(format!(
                        "column `{name}` has {} rows, expected {n}",
                        col.len()
                    )))
                }
                Some(_) => {}
            }
            names.push(name);
            cols.push(col);
        }
        Ok(Table {
            names,
            columns: cols,
            nrows: nrows.unwrap_or(0),
        })
    }

    /// An empty table with zero columns and `nrows` rows.
    pub fn with_nrows(nrows: usize) -> Self {
        Table {
            names: Vec::new(),
            columns: Vec::new(),
            nrows,
        }
    }

    /// Number of rows.
    pub fn nrows(&self) -> usize {
        self.nrows
    }

    /// Number of columns.
    pub fn ncols(&self) -> usize {
        self.columns.len()
    }

    /// Column names in order.
    pub fn names(&self) -> &[String] {
        &self.names
    }

    /// Column by name.
    pub fn column(&self, name: &str) -> Option<&Column> {
        self.names
            .iter()
            .position(|n| n == name)
            .map(|i| &self.columns[i])
    }

    /// Iterate `(name, column)` pairs in order.
    pub fn iter(&self) -> impl Iterator<Item = (&str, &Column)> {
        self.names.iter().map(String::as_str).zip(self.columns.iter())
    }

    /// Add or replace a column.
    pub fn insert(&mut self, name: impl Into<String>, col: Column) -> Result<()> {
        let name = name.into();
        if col.len() != self.nrows && !(self.columns.is_empty()) {
            return Err(Error::InvalidArgument(format!(
                "column `{name}` has {} rows, expected {}",
                col.len(),
                self.nrows
            )));
        }
        if self.columns.is_empty() {
            self.nrows = col.len();
        }
        match self.names.iter().position(|n| *n == name) {
            Some(i) => self.columns[i] = col,
            None => {
                self.names.push(name);
                self.columns.push(col);
            }
        }
        Ok(())
    }

    /// Remove a column by name (no-op if absent).
    pub fn remove(&mut self, name: &str) -> Option<Column> {
        let i = self.names.iter().position(|n| n == name)?;
        self.names.remove(i);
        Some(self.columns.remove(i))
    }

    /// Select rows by index into a new table.
    pub fn select_rows(&self, rows: &[usize]) -> Result<Table> {
        let mut cols = Vec::with_capacity(self.columns.len());
        for c in &self.columns {
            cols.push(c.select_rows(rows)?);
        }
        Ok(Table {
            names: self.names.clone(),
            columns: cols,
            nrows: rows.len(),
        })
    }

    /// A schema with every column assigned `role`, except `outcome` which gets
    /// `Role::Outcome`.
    pub fn schema_with_outcome(&self, outcome: &str, role: Role) -> Result<Schema> {
        if self.column(outcome).is_none() {
            return Err(Error::SchemaMismatch(format!(
                "outcome column `{outcome}` not found"
            )));
        }
        Ok(Schema {
            fields: self
                .iter()
                .map(|(n, c)| Field {
                    name: n.to_string(),
                    kind: c.kind(),
                    role: if n == outcome { Role::Outcome } else { role },
                })
                .collect(),
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn numeric_rejects_nan() {
        assert!(Column::numeric(vec![1.0, f64::NAN]).is_err());
        assert!(Column::numeric(vec![1.0, 2.0]).is_ok());
    }

    #[test]
    fn missing_mask_is_explicit() {
        let c = Column::numeric_with_missing(vec![Some(1.0), None, Some(3.0)]).unwrap();
        assert_eq!(c.n_missing(), 1);
        assert!(c.is_missing(1));
        assert!(!c.is_missing(0));
        let all = Column::numeric_with_missing(vec![Some(1.0)]).unwrap();
        assert!(all.validity().is_none());
    }

    #[test]
    fn categorical_levels_sorted_and_coded() {
        let c = Column::categorical(&["b", "a", "b"]).unwrap();
        let (codes, levels) = c.as_categorical().unwrap();
        assert_eq!(levels, &["a".to_string(), "b".to_string()]);
        assert_eq!(codes, &[1, 0, 1]);
    }

    #[test]
    fn categorical_rejects_unknown_label() {
        let r = Column::categorical_with_levels(&["a", "z"], vec!["a".into(), "b".into()]);
        assert!(matches!(r, Err(Error::InvalidValue(_))));
    }

    #[test]
    fn table_requires_equal_lengths_and_unique_names() {
        let a = Column::numeric(vec![1.0, 2.0]).unwrap();
        let b = Column::numeric(vec![1.0]).unwrap();
        assert!(Table::new(vec![("a", a.clone()), ("b", b)]).is_err());
        assert!(Table::new(vec![("a", a.clone()), ("a", a)]).is_err());
    }

    #[test]
    fn select_rows_preserves_mask_and_order() {
        let c = Column::numeric_with_missing(vec![Some(1.0), None, Some(3.0)]).unwrap();
        let t = Table::new(vec![("x", c)]).unwrap();
        let s = t.select_rows(&[2, 1, 2]).unwrap();
        let x = s.column("x").unwrap();
        assert_eq!(x.as_numeric().unwrap(), &[3.0, 0.0, 3.0]);
        assert!(x.is_missing(1));
        assert!(t.select_rows(&[3]).is_err());
    }
}
