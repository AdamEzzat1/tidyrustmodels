//! Random-forest baselines independent of R: ranger's default resolution, determinism
//! per seed, sanity of predictions, parameter plumbing and serialization.

use tidyrustmodels_core::{DesignMatrix, Error, ModelSpec, Outcome, Prediction, Seed};
use tidyrustmodels_models::{fitted_model_from_state, Engine, FittedRandForest, RandForestSpec};

fn step_data(n: usize) -> (DesignMatrix, Outcome) {
    let x1: Vec<f64> = (0..n).map(|i| i as f64 / n as f64).collect();
    let x2: Vec<f64> = (0..n).map(|i| ((i * 13) % 17) as f64).collect();
    let y: Vec<f64> = x1.iter().map(|v| if *v < 0.5 { 1.0 } else { 3.0 }).collect();
    (
        DesignMatrix::from_columns(vec![("x1".into(), x1), ("x2".into(), x2)]).unwrap(),
        Outcome::Numeric(y),
    )
}

#[test]
fn defaults_follow_ranger_and_fits_are_deterministic_per_seed() {
    let (x, y) = step_data(120);
    let a = RandForestSpec::new().trees(40).fit(&x, &y).unwrap();
    let b = RandForestSpec::new().trees(40).fit(&x, &y).unwrap();
    assert_eq!(a.predict(&x).unwrap(), b.predict(&x).unwrap());
    let c = RandForestSpec::new().trees(40).seed(Seed(1)).fit(&x, &y).unwrap();
    assert_ne!(a.predict(&x).unwrap(), c.predict(&x).unwrap());
    // mtry = floor(sqrt(2)) = 1, min.node.size 5 for regression
    assert_eq!(a.diagnostics().note.unwrap(), "40 trees, mtry 1, min.node.size 5");
    match a.predict(&x).unwrap() {
        Prediction::Numeric(p) => {
            assert!((p[10] - 1.0).abs() < 0.1);
            assert!((p[110] - 3.0).abs() < 0.1);
        }
        _ => panic!(),
    }
}

#[test]
fn probability_forest_defaults_and_class_prediction() {
    let n = 150;
    let x = DesignMatrix::from_columns(vec![
        ("x".into(), (0..n).map(|i| i as f64).collect()),
        ("noise".into(), (0..n).map(|i| ((i * 7) % 11) as f64).collect()),
        ("noise2".into(), (0..n).map(|i| ((i * 3) % 5) as f64).collect()),
    ])
    .unwrap();
    let y = Outcome::Class {
        codes: (0..n)
            .map(|i| {
                if i < 50 {
                    0
                } else if i < 100 {
                    1
                } else {
                    2
                }
            })
            .collect(),
        levels: vec!["a".into(), "b".into(), "c".into()],
    };
    let f = RandForestSpec::new().trees(60).fit(&x, &y).unwrap();
    // mtry = floor(sqrt(3)) = 1, min.node.size 10 for probability forests
    assert_eq!(
        f.diagnostics().note.unwrap(),
        "60 trees, mtry 1, min.node.size 10"
    );
    match f.predict(&x).unwrap() {
        Prediction::Class {
            codes,
            probabilities: Some(p),
            levels,
        } => {
            assert_eq!(levels, vec!["a", "b", "c"]);
            assert_eq!(codes[25], 0);
            assert_eq!(codes[125], 2);
            for i in 0..n {
                let s: f64 = (0..3).map(|j| p.get(i, j)).sum();
                assert!((s - 1.0).abs() < 1e-12);
            }
            assert!(p.get(25, 0) > 0.6, "{}", p.get(25, 0));
        }
        _ => panic!(),
    }
}

#[test]
fn parameters_validate_tune_and_round_trip() {
    let (x, y) = step_data(60);
    assert!(matches!(
        RandForestSpec::new().engine(Engine::Smartcore).fit(&x, &y),
        Err(Error::Unsupported(_))
    ));
    assert!(matches!(
        RandForestSpec::new().mtry(3).fit(&x, &y),
        Err(Error::InvalidArgument(_))
    ));
    assert!(RandForestSpec::new().with_param("trees", 0.0).is_err());
    let tuned = RandForestSpec::new()
        .with_param("mtry", 2.0)
        .unwrap()
        .with_param("min_n", 3.0)
        .unwrap();
    assert_eq!(tuned.tunable(), &["mtry", "trees", "min_n"]);
    let f = RandForestSpec::new()
        .trees(10)
        .mtry(2)
        .min_n(3)
        .fit(&x, &y)
        .unwrap();
    assert_eq!(f.diagnostics().note.unwrap(), "10 trees, mtry 2, min.node.size 3");
    let back = fitted_model_from_state(&f.state().unwrap()).unwrap();
    assert_eq!(f.predict(&x).unwrap(), back.predict(&x).unwrap());
    let state: FittedRandForest = serde_json::from_value(f.state().unwrap()).unwrap();
    assert_eq!(state.forest.trees.len(), 10);
    // every tree's leaves hold one value for regression
    assert!(state
        .forest
        .trees
        .iter()
        .all(|t| t.leaf.iter().filter(|l| !l.is_empty()).all(|l| l.len() == 1)));
}

#[test]
fn pure_or_tiny_nodes_become_leaves() {
    // constant outcome: every tree is a single leaf predicting the constant
    let x = DesignMatrix::from_columns(vec![("x".into(), (0..30).map(|i| i as f64).collect())]).unwrap();
    let y = Outcome::Numeric(vec![2.5; 30]);
    let f = RandForestSpec::new().trees(5).fit(&x, &y).unwrap();
    assert_eq!(f.predict(&x).unwrap(), Prediction::Numeric(vec![2.5; 30]));
    let state: FittedRandForest = serde_json::from_value(f.state().unwrap()).unwrap();
    assert!(state.forest.trees.iter().all(|t| t.split_var.len() == 1));
}
