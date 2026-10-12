//! Minimal CSV reader producing a [`Table`].
//!
//! Handles RFC 4180 quoting (`"..."` with `""` escapes, embedded commas/newlines).
//! Column kinds are inferred: a column whose non-empty fields all parse as `f64` is
//! numeric; anything else is categorical with levels sorted bytewise. Empty fields are
//! missing. This is what the reference datasets need; it is not a general CSV engine.

use std::path::Path;
use tidyrustmodels_core::{Column, Error, Result, Table};

/// Read a CSV file into a table.
pub fn read_csv(path: impl AsRef<Path>) -> Result<Table> {
    let text = std::fs::read_to_string(path.as_ref())
        .map_err(|e| Error::InvalidArgument(format!("cannot read {}: {e}", path.as_ref().display())))?;
    parse_csv(&text)
}

/// Parse CSV text into a table. The first record is the header.
pub fn parse_csv(text: &str) -> Result<Table> {
    let records = split_records(text)?;
    let mut it = records.into_iter();
    let header = it
        .next()
        .ok_or_else(|| Error::InvalidArgument("empty CSV".into()))?;
    let ncols = header.len();
    let mut cols: Vec<Vec<String>> = vec![Vec::new(); ncols];
    for (r, rec) in it.enumerate() {
        if rec.len() == 1 && rec[0].is_empty() {
            continue; // trailing blank line
        }
        if rec.len() != ncols {
            return Err(Error::InvalidArgument(format!(
                "CSV row {} has {} fields, expected {ncols}",
                r + 2,
                rec.len()
            )));
        }
        for (c, v) in rec.into_iter().enumerate() {
            cols[c].push(v);
        }
    }
    let mut out = Vec::with_capacity(ncols);
    for (name, raw) in header.into_iter().zip(cols) {
        out.push((name, infer_column(&raw)?));
    }
    Table::new(out)
}

fn infer_column(raw: &[String]) -> Result<Column> {
    let parsed: Vec<Option<f64>> = raw
        .iter()
        .map(|s| {
            if s.is_empty() {
                None
            } else {
                s.trim().parse::<f64>().ok()
            }
        })
        .collect();
    let numeric = raw.iter().zip(&parsed).all(|(s, p)| s.is_empty() || p.is_some());
    // An entirely empty column has no type evidence; R's read.csv yields an all-NA
    // logical column. The closest representation here is numeric with every row missing
    // (filters such as `StepZv` then remove it, as `step_zv` does in R).
    if numeric {
        Column::numeric_with_missing(parsed)
    } else {
        let mut levels: Vec<String> = raw.iter().filter(|s| !s.is_empty()).cloned().collect();
        levels.sort();
        levels.dedup();
        if levels.is_empty() {
            return Err(Error::InvalidValue("column is entirely empty".into()));
        }
        let labels: Vec<Option<&str>> = raw
            .iter()
            .map(|s| if s.is_empty() { None } else { Some(s.as_str()) })
            .collect();
        Column::categorical_with_levels_opt(&labels, levels)
    }
}

fn split_records(text: &str) -> Result<Vec<Vec<String>>> {
    let mut records = Vec::new();
    let mut record = Vec::new();
    let mut field = String::new();
    let mut in_quotes = false;
    let mut chars = text.chars().peekable();
    while let Some(c) = chars.next() {
        if in_quotes {
            match c {
                '"' => {
                    if chars.peek() == Some(&'"') {
                        field.push('"');
                        chars.next();
                    } else {
                        in_quotes = false;
                    }
                }
                _ => field.push(c),
            }
        } else {
            match c {
                '"' => in_quotes = true,
                ',' => record.push(std::mem::take(&mut field)),
                '\r' => {}
                '\n' => {
                    record.push(std::mem::take(&mut field));
                    records.push(std::mem::take(&mut record));
                }
                _ => field.push(c),
            }
        }
    }
    if in_quotes {
        return Err(Error::InvalidArgument("unterminated quoted field".into()));
    }
    if !field.is_empty() || !record.is_empty() {
        record.push(field);
        records.push(record);
    }
    Ok(records)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_quotes_missing_and_types() {
        let t = parse_csv("\"a\",b,\"c\"\n1.5,\"x, y\",\n,\"z\"\"q\",3\n").unwrap();
        assert_eq!(t.nrows(), 2);
        let a = t.column("a").unwrap();
        assert_eq!(a.as_numeric().unwrap()[0], 1.5);
        assert!(a.is_missing(1));
        let (codes, levels) = t.column("b").unwrap().as_categorical().unwrap();
        assert_eq!(levels, &["x, y".to_string(), "z\"q".to_string()]);
        assert_eq!(codes, &[0, 1]);
        assert!(t.column("c").unwrap().is_missing(0));
    }
}
