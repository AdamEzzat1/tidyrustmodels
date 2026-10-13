//! Boosted-tree baselines independent of R: parsnip's defaults, a closed-form stump,
//! the binary class rule, probability rows, validation, tuning, state round trip.
#![allow(clippy::needless_range_loop)]

use tidyrustmodels_core::{DesignMatrix, Error, ModelSpec, Outcome, Prediction};
use tidyrustmodels_models::{fitted_model_from_state, BoostTreeSpec, Engine, FittedBoostTree};

fn step_data(n: usize) -> (DesignMatrix, Outcome) {
    let x: Vec<f64> = (0..n).map(|i| i as f64).collect();
    let noise: Vec<f64> = (0..n).map(|i| ((i * 7) % 5) as f64).collect();
    let y: Vec<f64> = (0..n).map(|i| if i < n / 2 { 0.0 } else { 10.0 }).collect();
    (
        DesignMatrix::from_columns(vec![("x".into(), x), ("noise".into(), noise)]).unwrap(),
        Outcome::Numeric(y),
    )
}

#[test]
fn defaults_follow_parsnip_and_one_stump_matches_the_closed_form() {
    let (x, y) = step_data(10);
    let f = BoostTreeSpec::new().fit(&x, &y).unwrap();
    assert_eq!(
        f.diagnostics().note.unwrap(),
        "15 rounds, max_depth 6, eta 0.3, min_child_weight 1, gamma 0, zeros as missing: false"
    );
    // one round, depth 1: base = mean = 5, leaf = eta * 25 / (5 + 1) on each side
    let s = BoostTreeSpec::new().trees(1).tree_depth(1).fit(&x, &y).unwrap();
    match s.predict(&x).unwrap() {
        Prediction::Numeric(p) => {
            let leaf = 0.3f32 * (25.0f32 / 6.0f32);
            assert!((p[0] - f64::from(5.0f32 - leaf)).abs() < 1e-6, "{}", p[0]);
            assert!((p[9] - f64::from(5.0f32 + leaf)).abs() < 1e-6, "{}", p[9]);
        }
        _ => panic!(),
    }
    // fifteen rounds get close to the step
    match f.predict(&x).unwrap() {
        Prediction::Numeric(p) => {
            assert!(p[0] < 0.2, "{}", p[0]);
            assert!(p[9] > 9.8, "{}", p[9]);
        }
        _ => panic!(),
    }
}

#[test]
fn binary_uses_the_first_level_as_the_event_and_the_half_rule() {
    let n = 40;
    let x = DesignMatrix::from_columns(vec![("x".into(), (0..n).map(|i| i as f64).collect())]).unwrap();
    let y = Outcome::Class {
        codes: (0..n).map(|i| if i < 20 { 0 } else { 1 }).collect(),
        levels: vec!["yes".into(), "no".into()],
    };
    let f = BoostTreeSpec::new().fit(&x, &y).unwrap();
    match f.predict(&x).unwrap() {
        Prediction::Class {
            codes,
            levels,
            probabilities: Some(p),
        } => {
            assert_eq!(levels, vec!["yes", "no"]);
            assert!(p.get(2, 0) > 0.9, "{}", p.get(2, 0));
            assert!(p.get(37, 0) < 0.1, "{}", p.get(37, 0));
            for i in 0..n {
                assert_eq!(p.get(i, 0) + p.get(i, 1), 1.0);
                assert_eq!(codes[i], if p.get(i, 0) >= 0.5 { 0 } else { 1 });
            }
        }
        _ => panic!(),
    }
}

#[test]
fn multiclass_probabilities_sum_to_one_and_pick_the_largest() {
    let n = 60;
    let x = DesignMatrix::from_columns(vec![
        ("x".into(), (0..n).map(|i| i as f64).collect()),
        ("z".into(), (0..n).map(|i| ((i * 3) % 7) as f64).collect()),
    ])
    .unwrap();
    let y = Outcome::Class {
        codes: (0..n).map(|i| (i / 20) as u32).collect(),
        levels: vec!["a".into(), "b".into(), "c".into()],
    };
    let f = BoostTreeSpec::new().trees(10).tree_depth(3).fit(&x, &y).unwrap();
    let state: FittedBoostTree = serde_json::from_value(f.state().unwrap()).unwrap();
    assert_eq!(state.booster.trees.len(), 30);
    assert_eq!(state.booster.base_score.len(), 3);
    match f.predict(&x).unwrap() {
        Prediction::Class {
            codes,
            probabilities: Some(p),
            ..
        } => {
            for i in 0..n {
                let s: f64 = (0..3).map(|j| p.get(i, j)).sum();
                assert!((s - 1.0).abs() < 1e-6);
                let best = (0..3)
                    .max_by(|a, b| p.get(i, *a).partial_cmp(&p.get(i, *b)).unwrap())
                    .unwrap();
                assert_eq!(codes[i] as usize, best);
            }
            assert_eq!(codes[5], 0);
            assert_eq!(codes[55], 2);
        }
        _ => panic!(),
    }
}

#[test]
fn parameters_validate_tune_and_round_trip() {
    let (x, y) = step_data(20);
    assert!(matches!(
        BoostTreeSpec::new().engine(Engine::Linfa).fit(&x, &y),
        Err(Error::Unsupported(_))
    ));
    assert!(matches!(
        BoostTreeSpec::new().trees(0).fit(&x, &y),
        Err(Error::InvalidArgument(_))
    ));
    assert!(matches!(
        BoostTreeSpec::new().learn_rate(-0.1).fit(&x, &y),
        Err(Error::InvalidArgument(_))
    ));
    assert!(BoostTreeSpec::new().with_param("mtry", 2.0).is_err());
    let tuned = BoostTreeSpec::new()
        .with_param("trees", 3.0)
        .unwrap()
        .with_param("learn_rate", 0.5)
        .unwrap()
        .with_param("loss_reduction", 1.0)
        .unwrap();
    assert_eq!(
        tuned.tunable(),
        // parsnip's argument order (tune's .config numbering, ADR-0014)
        &["trees", "min_n", "tree_depth", "learn_rate", "loss_reduction"]
    );
    let f = tuned.fit(&x, &y).unwrap();
    assert_eq!(
        f.diagnostics().note.unwrap(),
        "3 rounds, max_depth 6, eta 0.5, min_child_weight 1, gamma 1, zeros as missing: false"
    );
    let back = fitted_model_from_state(&f.state().unwrap()).unwrap();
    assert_eq!(f.predict(&x).unwrap(), back.predict(&x).unwrap());
    // min_n is capped at the number of rows, as parsnip does
    let g = BoostTreeSpec::new().trees(1).min_n(1000.0).fit(&x, &y).unwrap();
    assert!(g.diagnostics().note.unwrap().contains("min_child_weight 20"));
    let state: FittedBoostTree = serde_json::from_value(g.state().unwrap()).unwrap();
    assert_eq!(
        state.booster.trees[0].len(),
        1,
        "no split can satisfy min_child_weight = n"
    );
}

#[test]
fn constant_outcome_gives_single_leaf_trees_predicting_the_mean() {
    let x = DesignMatrix::from_columns(vec![("x".into(), (0..30).map(|i| i as f64).collect())]).unwrap();
    let y = Outcome::Numeric(vec![2.5; 30]);
    let f = BoostTreeSpec::new().trees(4).fit(&x, &y).unwrap();
    assert_eq!(f.predict(&x).unwrap(), Prediction::Numeric(vec![2.5; 30]));
    let state: FittedBoostTree = serde_json::from_value(f.state().unwrap()).unwrap();
    assert!(state.booster.trees.iter().all(|t| t.len() == 1));
}

#[test]
fn sparse_rule_follows_the_design_and_can_be_forced() {
    // zeros only in the upper step: read as values they sit left of every other x and
    // need their own leaf; read as missing they join the upper leaf by default direction
    let n = 20;
    let xv: Vec<f64> = (0..n)
        .map(|i| {
            if i >= n / 2 && i % 2 == 0 {
                0.0
            } else {
                (i + 1) as f64
            }
        })
        .collect();
    let noise: Vec<f64> = (0..n).map(|i| ((i * 7) % 5 + 1) as f64).collect();
    let x = DesignMatrix::from_columns(vec![("x".into(), xv), ("noise".into(), noise)]).unwrap();
    let y = Outcome::Numeric((0..n).map(|i| if i < n / 2 { 0.0 } else { 10.0 }).collect());
    let spec = BoostTreeSpec::new();
    // Auto lets a workflow toggle the recipe for xgboost; a forced policy does not
    assert_eq!(spec.sparse_engine(), Some("xgboost"));
    assert_eq!(spec.clone().zeros_as_missing(true).sparse_engine(), None);
    assert_eq!(spec.clone().zeros_as_missing(false).sparse_engine(), None);
    let note = |f: &dyn tidyrustmodels_core::FittedModel| f.diagnostics().note.unwrap();
    // a dense design: zeros are values; a sparse design (dgCMatrix in R): missing
    let dense = spec.fit(&x, &y).unwrap();
    assert!(note(dense.as_ref()).ends_with("zeros as missing: false"));
    let xs = x.clone().with_sparse(true);
    let sparse = spec.fit(&xs, &y).unwrap();
    assert!(note(sparse.as_ref()).ends_with("zeros as missing: true"));
    assert_ne!(dense.predict(&x).unwrap(), sparse.predict(&xs).unwrap());
    // at prediction the new data decides (parsnip converts the baked new data again):
    // the sparse fit reads a dense design with zeros as values, a sparse one as missing
    let state: FittedBoostTree = serde_json::from_value(sparse.state().unwrap()).unwrap();
    assert!(state.booster.zero_as_missing);
    let manual = |zam: bool| -> Prediction {
        Prediction::Numeric(
            (0..x.nrows())
                .map(|i| f64::from(state.booster.predict_with(|f| x.get(i, f) as f32, zam)[0]))
                .collect(),
        )
    };
    assert_eq!(sparse.predict(&x).unwrap(), manual(false));
    assert_eq!(sparse.predict(&xs).unwrap(), manual(true));
    assert_ne!(manual(false), manual(true));
    // forced: zeros are missing on any design, at fit and at predict
    let forced = spec.clone().zeros_as_missing(true).fit(&x, &y).unwrap();
    assert!(note(forced.as_ref()).ends_with("zeros as missing: true"));
    assert_eq!(forced.predict(&x).unwrap(), forced.predict(&xs).unwrap());
    // Dense only skips the toggle: a sparse design is still a dgCMatrix
    let no_toggle = spec.clone().zeros_as_missing(false).fit(&xs, &y).unwrap();
    assert!(note(no_toggle.as_ref()).ends_with("zeros as missing: true"));
    // a state written before ADR-0015 (no `forced_zeros_missing`) keeps its fitted rule
    let mut old = sparse.state().unwrap();
    old.as_object_mut().unwrap().remove("forced_zeros_missing");
    let old = fitted_model_from_state(&old).unwrap();
    assert_eq!(old.predict(&x).unwrap(), manual(true));
}
