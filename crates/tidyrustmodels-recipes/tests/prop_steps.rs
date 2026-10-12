//! Property tests for preprocessing invariants (ROADMAP Phase 1 gate).

use proptest::prelude::*;
use tidyrustmodels_core::{Column, Role, Step, Table};
use tidyrustmodels_recipes::{Selector, StepDummy, StepImputeMean, StepImputeMedian, StepNormalize};

fn numeric_with_missing() -> impl Strategy<Value = Vec<Option<f64>>> {
    prop::collection::vec(prop::option::weighted(0.8, -1e3..1e3f64), 3..60)
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(200))]

    /// Standardized training columns have mean ≈ 0 and sd ≈ 1 over their valid rows
    /// whenever the training sd is non-zero.
    #[test]
    fn normalize_centres_and_scales_training_data(values in numeric_with_missing()) {
        let n_valid = values.iter().filter(|v| v.is_some()).count();
        prop_assume!(n_valid >= 2);
        let distinct: std::collections::BTreeSet<i64> = values.iter().flatten().map(|v| v.to_bits() as i64).collect();
        prop_assume!(distinct.len() >= 2);
        let t = Table::new(vec![
            ("x", Column::numeric_with_missing(values.clone()).unwrap()),
            ("y", Column::numeric(vec![0.0; values.len()]).unwrap()),
        ]).unwrap();
        let schema = t.schema_with_outcome("y", Role::Predictor).unwrap();
        let fitted = StepNormalize::new(Selector::AllNumericPredictors).prep(&t, &schema).unwrap();
        let baked = fitted.bake(&t).unwrap();
        let col = baked.column("x").unwrap();
        let vals = col.as_numeric().unwrap();
        let valid: Vec<f64> = vals.iter().enumerate().filter(|(i, _)| !col.is_missing(*i)).map(|(_, v)| *v).collect();
        let m = valid.iter().sum::<f64>() / valid.len() as f64;
        let var = valid.iter().map(|v| (v - m).powi(2)).sum::<f64>() / (valid.len() as f64 - 1.0);
        let scale = values.iter().flatten().map(|v| v.abs()).fold(1.0, f64::max);
        prop_assert!(m.abs() < 1e-9 * scale, "mean {m}");
        prop_assert!((var - 1.0).abs() < 1e-6, "var {var}");
        prop_assert_eq!(col.n_missing(), values.iter().filter(|v| v.is_none()).count());
    }

    /// Imputation leaves no missing values and never touches valid entries.
    #[test]
    fn imputation_fills_only_missing(values in numeric_with_missing(), use_median in any::<bool>()) {
        prop_assume!(values.iter().any(|v| v.is_some()));
        let t = Table::new(vec![
            ("x", Column::numeric_with_missing(values.clone()).unwrap()),
            ("y", Column::numeric(vec![0.0; values.len()]).unwrap()),
        ]).unwrap();
        let schema = t.schema_with_outcome("y", Role::Predictor).unwrap();
        let fitted = if use_median {
            StepImputeMedian::new(Selector::columns(&["x"])).prep(&t, &schema).unwrap()
        } else {
            StepImputeMean::new(Selector::columns(&["x"])).prep(&t, &schema).unwrap()
        };
        let baked = fitted.bake(&t).unwrap();
        let col = baked.column("x").unwrap();
        prop_assert_eq!(col.n_missing(), 0);
        let out = col.as_numeric().unwrap();
        let lo = values.iter().flatten().cloned().fold(f64::INFINITY, f64::min);
        let hi = values.iter().flatten().cloned().fold(f64::NEG_INFINITY, f64::max);
        for (o, v) in out.iter().zip(&values) {
            match v {
                Some(x) => prop_assert_eq!(o, x),
                None => prop_assert!(*o >= lo - 1e-9 && *o <= hi + 1e-9, "fill {o} outside [{lo}, {hi}]"),
            }
        }
    }

    /// One-hot rows sum to exactly 1; drop-first rows sum to 0 or 1, and the dropped
    /// level is exactly the first training level.
    #[test]
    fn dummy_rows_are_indicator_rows(labels in prop::collection::vec(0u8..5, 2..50), one_hot in any::<bool>()) {
        let names: Vec<String> = labels.iter().map(|l| format!("L{l}")).collect();
        let t = Table::new(vec![
            ("y", Column::numeric(vec![0.0; labels.len()]).unwrap()),
            ("g", Column::categorical(&names).unwrap()),
        ]).unwrap();
        let schema = t.schema_with_outcome("y", Role::Predictor).unwrap();
        let fitted = StepDummy::new(Selector::columns(&["g"])).one_hot(one_hot).prep(&t, &schema).unwrap();
        let baked = fitted.bake(&t).unwrap();
        let dummies: Vec<&str> = baked.names().iter().filter(|n| n.starts_with("g_")).map(String::as_str).collect();
        let (_, levels) = t.column("g").unwrap().as_categorical().unwrap();
        prop_assert_eq!(dummies.len(), if one_hot { levels.len() } else { levels.len() - 1 });
        if !one_hot {
            let first = format!("g_{}", levels[0]);
            prop_assert!(!dummies.contains(&first.as_str()));
        }
        for (i, name) in names.iter().enumerate() {
            let s: f64 = dummies.iter().map(|d| baked.column(d).unwrap().as_numeric().unwrap()[i]).sum();
            if one_hot {
                prop_assert_eq!(s, 1.0);
            } else {
                let is_first = *name == levels[0];
                prop_assert_eq!(s, if is_first { 0.0 } else { 1.0 });
            }
        }
    }
}
