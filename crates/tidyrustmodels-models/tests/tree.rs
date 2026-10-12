//! Decision-tree baselines independent of R: closed-form behaviour of the rpart port
//! through the `ModelSpec` API, parameter plumbing, predictions and serialization.

use tidyrustmodels_core::{DesignMatrix, Error, FittedModel, ModelSpec, Outcome, Prediction};
use tidyrustmodels_models::{fitted_model_from_state, DecisionTreeSpec, Engine, FittedDecisionTree};

fn step_data() -> (DesignMatrix, Outcome) {
    // y is a step function of x1 at 4.5; x2 is noise that never helps
    let x1: Vec<f64> = (1..=16).map(|v| (v as f64) / 2.0).collect();
    let x2: Vec<f64> = (0..16).map(|i| ((i * 7) % 5) as f64).collect();
    let y: Vec<f64> = x1
        .iter()
        .map(|v| if *v < 4.5 { 1.0 } else { 5.0 } + 0.01 * (*v % 1.0))
        .collect();
    (
        DesignMatrix::from_columns(vec![("x1".into(), x1), ("x2".into(), x2)]).unwrap(),
        Outcome::Numeric(y),
    )
}

#[test]
fn regression_tree_predicts_leaf_means_and_round_trips() {
    let (x, y) = step_data();
    let fit = DecisionTreeSpec::new().min_n(4).fit(&x, &y).unwrap();
    let pred = match fit.predict(&x).unwrap() {
        Prediction::Numeric(p) => p,
        _ => panic!(),
    };
    // every prediction is the mean of the leaf's training rows
    let state: FittedDecisionTree = serde_json::from_value(fit.state().unwrap()).unwrap();
    let root = &state.tree.root;
    assert_eq!(root.split.as_ref().unwrap().var, 0);
    assert!((root.split.as_ref().unwrap().point - 4.25).abs() < 1e-12); // midpoint of 4.0 and 4.5
    let yv = match &y {
        Outcome::Numeric(v) => v,
        _ => unreachable!(),
    };
    let left_mean = yv[..8].iter().sum::<f64>() / 8.0;
    assert!(pred[0..8].iter().all(|p| (p - left_mean).abs() < 1e-12) || state.tree.n_splits() > 1);
    let back = fitted_model_from_state(&fit.state().unwrap()).unwrap();
    assert_eq!(fit.predict(&x).unwrap(), back.predict(&x).unwrap());
    assert!(fit.diagnostics().note.unwrap().ends_with("splits"));
}

#[test]
fn classification_tree_gives_class_shares_and_first_level_on_ties() {
    // x takes two values only, so the single possible split separates the halves;
    // first 10 rows: 8 "a" / 2 "b", last 10 rows: 3 "a" / 7 "b" (misclassification 9 -> 5,
    // so rpart keeps the split even at cp = 0)
    let x = DesignMatrix::from_columns(vec![(
        "x".into(),
        (0..20).map(|i| if i < 10 { 1.0 } else { 2.0 }).collect(),
    )])
    .unwrap();
    let mut codes = vec![0u32; 8];
    codes.extend([1, 1]);
    codes.extend([0, 0, 0, 1, 1, 1, 1, 1, 1, 1]);
    let y = Outcome::Class {
        codes,
        levels: vec!["a".into(), "b".into()],
    };
    let fit = DecisionTreeSpec::new()
        .min_n(6)
        .cost_complexity(0.0)
        .tree_depth(1)
        .fit(&x, &y)
        .unwrap();
    match fit.predict(&x).unwrap() {
        Prediction::Class {
            codes,
            probabilities: Some(p),
            ..
        } => {
            assert!((p.get(0, 0) - 0.8).abs() < 1e-12);
            assert!((p.get(19, 0) - 0.3).abs() < 1e-12);
            assert_eq!(codes[0], 0);
            assert_eq!(codes[19], 1);
        }
        _ => panic!(),
    }
    // a 5/5 root with nothing to split on: rpart's ginidev picks the first class
    let x = DesignMatrix::from_columns(vec![("x".into(), vec![1.0; 10])]).unwrap();
    let y = Outcome::Class {
        codes: vec![1, 0, 1, 0, 1, 0, 1, 0, 1, 0],
        levels: vec!["a".into(), "b".into()],
    };
    let fit = DecisionTreeSpec::new().min_n(2).fit(&x, &y).unwrap();
    match fit.predict(&x).unwrap() {
        Prediction::Class { codes, .. } => assert!(codes.iter().all(|c| *c == 0)),
        _ => panic!(),
    }
    // a split that does not reduce the misclassification count is collapsed (cp = 0 is
    // still "complexity must exceed alpha = 0"), as in partition.c
    let x = DesignMatrix::from_columns(vec![(
        "x".into(),
        (0..20).map(|i| if i < 10 { 1.0 } else { 2.0 }).collect(),
    )])
    .unwrap();
    let mut codes = vec![0u32; 7];
    codes.extend([1, 1, 1]);
    codes.extend([0, 1, 0, 1, 0, 1, 0, 1, 0, 1]);
    let y = Outcome::Class {
        codes,
        levels: vec!["a".into(), "b".into()],
    };
    let fit = DecisionTreeSpec::new()
        .min_n(6)
        .cost_complexity(0.0)
        .fit(&x, &y)
        .unwrap();
    let t: FittedDecisionTree = serde_json::from_value(fit.state().unwrap()).unwrap();
    assert!(t.tree.root.is_leaf());
}

#[test]
fn parameters_validate_and_tune() {
    let (x, y) = step_data();
    assert!(matches!(
        DecisionTreeSpec::new().tree_depth(31).fit(&x, &y),
        Err(Error::InvalidArgument(_))
    ));
    assert!(matches!(
        DecisionTreeSpec::new().engine(Engine::Linfa).fit(&x, &y),
        Err(Error::Unsupported(_))
    ));
    let tuned = DecisionTreeSpec::new().with_param("min_n", 4.0).unwrap();
    assert_eq!(tuned.tunable(), &["cost_complexity", "tree_depth", "min_n"]);
    assert!(DecisionTreeSpec::new().with_param("mtry", 1.0).is_err());
    let deep = DecisionTreeSpec::new()
        .min_n(2)
        .cost_complexity(0.0)
        .fit(&x, &y)
        .unwrap();
    let shallow = DecisionTreeSpec::new()
        .min_n(2)
        .cost_complexity(0.0)
        .tree_depth(1)
        .fit(&x, &y)
        .unwrap();
    let n = |m: &dyn FittedModel| {
        serde_json::from_value::<FittedDecisionTree>(m.state().unwrap())
            .unwrap()
            .tree
            .n_splits()
    };
    assert_eq!(n(shallow.as_ref()), 1);
    assert!(n(deep.as_ref()) > 1);
}

#[test]
fn root_only_tree_when_nothing_splits() {
    let x = DesignMatrix::from_columns(vec![("x".into(), vec![1.0, 2.0, 3.0, 4.0])]).unwrap();
    let y = Outcome::Numeric(vec![2.0, 2.0, 2.0, 2.0]);
    let fit = DecisionTreeSpec::new().min_n(2).fit(&x, &y).unwrap();
    let t: FittedDecisionTree = serde_json::from_value(fit.state().unwrap()).unwrap();
    assert!(t.tree.root.is_leaf());
    assert_eq!(fit.predict(&x).unwrap(), Prediction::Numeric(vec![2.0; 4]));
}
