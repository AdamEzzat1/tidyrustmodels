//! Arrow interop (feature `arrow`; ADR-0003, ADR-0015): column-wise conversion
//! between an [`arrow_array::RecordBatch`] and a [`Table`]. Numeric Arrow types
//! (integers, floats) become `f64` columns with nulls as missing values; `Utf8`,
//! `LargeUtf8`, dictionary-encoded strings and booleans become categorical columns
//! (levels sorted bytewise, as every other constructor). Nothing is zero-copy: the
//! `Table` owns its data.

use arrow_array::cast::AsArray;
use arrow_array::types::{
    Float32Type, Float64Type, Int16Type, Int32Type, Int64Type, Int8Type, UInt16Type, UInt32Type, UInt64Type,
    UInt8Type,
};
use arrow_array::{Array, ArrayRef, ArrowPrimitiveType, Float64Array, RecordBatch, StringArray};
use arrow_schema::{DataType, Field, Schema};
use std::sync::Arc;
use tidyrustmodels_core::{Column, Error, Result, Table};

fn numeric<T: ArrowPrimitiveType>(a: &ArrayRef, to_f64: impl Fn(T::Native) -> f64) -> Column {
    let p = a.as_primitive::<T>();
    let values: Vec<Option<f64>> = (0..p.len())
        .map(|i| {
            if p.is_null(i) {
                None
            } else {
                Some(to_f64(p.value(i)))
            }
        })
        .collect();
    Column::numeric_with_missing(values).expect("finite or missing")
}

fn labels(a: &ArrayRef) -> Result<Vec<Option<String>>> {
    Ok(match a.data_type() {
        DataType::Utf8 => a
            .as_string::<i32>()
            .iter()
            .map(|v| v.map(str::to_string))
            .collect(),
        DataType::LargeUtf8 => a
            .as_string::<i64>()
            .iter()
            .map(|v| v.map(str::to_string))
            .collect(),
        DataType::Boolean => a
            .as_boolean()
            .iter()
            .map(|v| {
                v.map(|b| {
                    if b {
                        "True".to_string()
                    } else {
                        "False".to_string()
                    }
                })
            })
            .collect(),
        DataType::Dictionary(key, value) if matches!(**value, DataType::Utf8 | DataType::LargeUtf8) => {
            let dict_labels = |keys: Vec<Option<usize>>| -> Result<Vec<Option<String>>> {
                let values = match &**value {
                    DataType::Utf8 => a
                        .as_any_dictionary()
                        .values()
                        .as_string::<i32>()
                        .iter()
                        .map(|v| v.map(str::to_string))
                        .collect::<Vec<_>>(),
                    _ => a
                        .as_any_dictionary()
                        .values()
                        .as_string::<i64>()
                        .iter()
                        .map(|v| v.map(str::to_string))
                        .collect::<Vec<_>>(),
                };
                Ok(keys
                    .into_iter()
                    .map(|k| k.and_then(|k| values[k].clone()))
                    .collect())
            };
            let keys: Vec<Option<usize>> = match **key {
                DataType::Int8 => a
                    .as_dictionary::<Int8Type>()
                    .keys()
                    .iter()
                    .map(|k| k.map(|k| k as usize))
                    .collect(),
                DataType::Int16 => a
                    .as_dictionary::<Int16Type>()
                    .keys()
                    .iter()
                    .map(|k| k.map(|k| k as usize))
                    .collect(),
                DataType::Int32 => a
                    .as_dictionary::<Int32Type>()
                    .keys()
                    .iter()
                    .map(|k| k.map(|k| k as usize))
                    .collect(),
                DataType::Int64 => a
                    .as_dictionary::<Int64Type>()
                    .keys()
                    .iter()
                    .map(|k| k.map(|k| k as usize))
                    .collect(),
                DataType::UInt8 => a
                    .as_dictionary::<UInt8Type>()
                    .keys()
                    .iter()
                    .map(|k| k.map(|k| k as usize))
                    .collect(),
                DataType::UInt16 => a
                    .as_dictionary::<UInt16Type>()
                    .keys()
                    .iter()
                    .map(|k| k.map(|k| k as usize))
                    .collect(),
                DataType::UInt32 => a
                    .as_dictionary::<UInt32Type>()
                    .keys()
                    .iter()
                    .map(|k| k.map(|k| k as usize))
                    .collect(),
                DataType::UInt64 => a
                    .as_dictionary::<UInt64Type>()
                    .keys()
                    .iter()
                    .map(|k| k.map(|k| k as usize))
                    .collect(),
                ref other => return Err(Error::Unsupported(format!("dictionary key type {other}"))),
            };
            dict_labels(keys)?
        }
        other => {
            return Err(Error::Unsupported(format!(
                "Arrow type {other} is not a string or boolean column"
            )))
        }
    })
}

/// One Arrow column as a [`Column`].
pub fn column_from_arrow(a: &ArrayRef) -> Result<Column> {
    Ok(match a.data_type() {
        DataType::Float64 => numeric::<Float64Type>(a, |v| v),
        DataType::Float32 => numeric::<Float32Type>(a, f64::from),
        DataType::Int8 => numeric::<Int8Type>(a, f64::from),
        DataType::Int16 => numeric::<Int16Type>(a, f64::from),
        DataType::Int32 => numeric::<Int32Type>(a, f64::from),
        DataType::Int64 => numeric::<Int64Type>(a, |v| v as f64),
        DataType::UInt8 => numeric::<UInt8Type>(a, f64::from),
        DataType::UInt16 => numeric::<UInt16Type>(a, f64::from),
        DataType::UInt32 => numeric::<UInt32Type>(a, f64::from),
        DataType::UInt64 => numeric::<UInt64Type>(a, |v| v as f64),
        DataType::Utf8 | DataType::LargeUtf8 | DataType::Boolean | DataType::Dictionary(_, _) => {
            let labels = labels(a)?;
            let mut levels: Vec<String> = labels.iter().flatten().cloned().collect();
            levels.sort();
            levels.dedup();
            let refs: Vec<Option<&str>> = labels.iter().map(|l| l.as_deref()).collect();
            Column::categorical_with_levels_opt(&refs, levels)?
        }
        other => return Err(Error::Unsupported(format!("Arrow type {other}"))),
    })
}

/// A [`Table`] from a record batch (one column per field, in schema order).
pub fn table_from_record_batch(batch: &RecordBatch) -> Result<Table> {
    let mut cols = Vec::with_capacity(batch.num_columns());
    for (field, array) in batch.schema().fields().iter().zip(batch.columns()) {
        let col = column_from_arrow(array)
            .map_err(|e| Error::Unsupported(format!("column `{}`: {e}", field.name())))?;
        cols.push((field.name().to_string(), col));
    }
    Table::new(cols)
}

/// A record batch from a table: `Float64` columns with nulls for missing values and
/// `Utf8` columns for categoricals. The sparse-vector flag is not carried over.
pub fn record_batch_from_table(table: &Table) -> Result<RecordBatch> {
    let mut fields = Vec::with_capacity(table.ncols());
    let mut arrays: Vec<ArrayRef> = Vec::with_capacity(table.ncols());
    for (name, col) in table.iter() {
        match col.as_numeric() {
            Some(values) => {
                let a = Float64Array::from_iter((0..values.len()).map(|i| {
                    if col.is_missing(i) {
                        None
                    } else {
                        Some(values[i])
                    }
                }));
                fields.push(Field::new(name, DataType::Float64, col.n_missing() > 0));
                arrays.push(Arc::new(a));
            }
            None => {
                let (codes, levels) = col.as_categorical().expect("categorical");
                let a = StringArray::from_iter((0..codes.len()).map(|i| {
                    if col.is_missing(i) {
                        None
                    } else {
                        Some(levels[codes[i] as usize].clone())
                    }
                }));
                fields.push(Field::new(name, DataType::Utf8, col.n_missing() > 0));
                arrays.push(Arc::new(a));
            }
        }
    }
    RecordBatch::try_new(Arc::new(Schema::new(fields)), arrays)
        .map_err(|e| Error::InvalidArgument(format!("cannot build a record batch: {e}")))
}

#[cfg(test)]
mod tests {
    use super::*;
    use arrow_array::{BooleanArray, DictionaryArray, Int32Array, Int64Array};

    #[test]
    fn round_trip_numeric_string_dictionary_and_boolean() {
        let x = Float64Array::from(vec![Some(1.5), None, Some(0.0)]);
        let n = Int64Array::from(vec![3, 4, 5]);
        let g = StringArray::from(vec![Some("b"), Some("a"), None]);
        let keys = Int32Array::from(vec![Some(1), Some(0), None]);
        let d =
            DictionaryArray::<Int32Type>::try_new(keys, Arc::new(StringArray::from(vec!["a", "b"]))).unwrap();
        let b = BooleanArray::from(vec![Some(true), Some(false), None]);
        let schema = Schema::new(vec![
            Field::new("x", DataType::Float64, true),
            Field::new("n", DataType::Int64, false),
            Field::new("g", DataType::Utf8, true),
            Field::new("d", d.data_type().clone(), true),
            Field::new("b", DataType::Boolean, true),
        ]);
        let batch = RecordBatch::try_new(
            Arc::new(schema),
            vec![Arc::new(x), Arc::new(n), Arc::new(g), Arc::new(d), Arc::new(b)],
        )
        .unwrap();
        let t = table_from_record_batch(&batch).unwrap();
        assert_eq!(t.names(), ["x", "n", "g", "d", "b"]);
        let x = t.column("x").unwrap();
        assert_eq!(x.as_numeric().unwrap()[0], 1.5);
        assert!(x.is_missing(1));
        assert_eq!(t.column("n").unwrap().as_numeric().unwrap(), &[3.0, 4.0, 5.0]);
        let (codes, levels) = t.column("g").unwrap().as_categorical().unwrap();
        assert_eq!(levels, &["a".to_string(), "b".to_string()]);
        assert_eq!(codes[0], 1);
        assert!(t.column("g").unwrap().is_missing(2));
        // the dictionary column decodes to the same labels as the plain string column
        assert_eq!(t.column("d").unwrap().as_categorical().unwrap().0, codes);
        let (bcodes, blevels) = t.column("b").unwrap().as_categorical().unwrap();
        assert_eq!(blevels, &["False".to_string(), "True".to_string()]);
        assert_eq!(bcodes[0], 1);
        // back to Arrow and again to a table: identical
        let back = record_batch_from_table(&t).unwrap();
        assert_eq!(back.schema().field(2).data_type(), &DataType::Utf8);
        assert_eq!(table_from_record_batch(&back).unwrap(), t);
        // unsupported types are refused with the column name
        let lst = arrow_array::ListArray::from_iter_primitive::<Int32Type, _, _>(vec![Some(vec![Some(1)])]);
        let bad = RecordBatch::try_new(
            Arc::new(Schema::new(vec![Field::new("l", lst.data_type().clone(), true)])),
            vec![Arc::new(lst)],
        )
        .unwrap();
        let err = table_from_record_batch(&bad).unwrap_err().to_string();
        assert!(err.contains("`l`"), "{err}");
    }
}
