//! Analytic baselines independent of both R and the backend crates
//! (docs/specs/verification-design.md, System X).

use approx::assert_relative_eq;
use tidyrustmodels_core::{DesignMatrix, Error, ModelSpec, Outcome, Prediction};
use tidyrustmodels_models::{Engine, LinearRegSpec, LogisticRegSpec};

const ENGINES: [Engine; 3] = [Engine::Linfa, Engine::Smartcore, Engine::Native];

/// Dataset and exact solution from reference/independent/exact_ols.py
/// (beta = -3445/1683, 1300/1683, 1895/1683).
fn exact_ols_data() -> (DesignMatrix, Outcome, [f64; 3], Vec<f64>) {
    let x1 = [1., 2., 3., 4., 5., 6., 7., 8., 9., 10., 11., 12.];
    let x2 = [5., 3., 8., 1., 9., 2., 7., 4., 6., 0., 10., 11.];
    let y = vec![3., 4., 9., 2., 13., 5., 12., 8., 11., 6., 17., 20.];
    let x = DesignMatrix::from_columns(vec![("x1".into(), x1.to_vec()), ("x2".into(), x2.to_vec())]).unwrap();
    let beta = [-2.0469399881164585, 0.7724301841948901, 1.1259655377302435];
    let fitted = vec![
        4.35531788472965,
        2.8758169934640523,
        9.27807486631016,
        2.168746286393345,
        11.948900772430184,
        4.839572192513369,
        11.241830065359476,
        8.636363636363637,
        11.660724896019014,
        5.677361853832442,
        17.70944741532977,
        19.607843137254903,
    ];
    (x, Outcome::Numeric(y), beta, fitted)
}

#[test]
fn ols_matches_exact_rational_solution_on_every_engine() {
    let (x, y, beta, fitted) = exact_ols_data();
    for engine in ENGINES {
        let m = LinearRegSpec::new().engine(engine).fit(&x, &y).unwrap();
        let c = m.coefficients().unwrap();
        assert_relative_eq!(c.intercept.unwrap(), beta[0], max_relative = 1e-10);
        assert_relative_eq!(c.terms[0].1, beta[1], max_relative = 1e-10);
        assert_relative_eq!(c.terms[1].1, beta[2], max_relative = 1e-10);
        assert_eq!(c.terms[0].0, "x1");
        match m.predict(&x).unwrap() {
            Prediction::Numeric(p) => {
                for (a, b) in p.iter().zip(&fitted) {
                    assert_relative_eq!(a, b, max_relative = 1e-10);
                }
            }
            _ => panic!("regression must predict numbers"),
        }
        assert_eq!(m.backend().engine, engine.as_str());
    }
}

#[test]
fn ols_without_intercept_recovers_exact_slope() {
    // y = 2x exactly, no intercept.
    let x = DesignMatrix::from_columns(vec![("x".into(), vec![1., 2., 3., 4.])]).unwrap();
    let y = Outcome::Numeric(vec![2., 4., 6., 8.]);
    let m = LinearRegSpec::new().intercept(false).fit(&x, &y).unwrap();
    let c = m.coefficients().unwrap();
    assert!(c.intercept.is_none());
    assert_relative_eq!(c.terms[0].1, 2.0, max_relative = 1e-12);
    // smartcore cannot drop the intercept: must be a clean Unsupported error.
    let r = LinearRegSpec::new()
        .engine(Engine::Smartcore)
        .intercept(false)
        .fit(&x, &y);
    assert!(matches!(r, Err(Error::Unsupported(_))));
}

/// Binary predictor, binary outcome: the MLE is closed-form.
/// x = 0: 1 of 4 are "yes";  x = 1: 3 of 4 are "yes".
/// b0 = log(1/3), b1 = log(3) - b0 = log(9).
fn closed_form_logistic_data() -> (DesignMatrix, Outcome, f64, f64) {
    let x = DesignMatrix::from_columns(vec![("x".into(), vec![0., 0., 0., 0., 1., 1., 1., 1.])]).unwrap();
    let y = Outcome::Class {
        codes: vec![0, 0, 0, 1, 1, 1, 1, 0],
        levels: vec!["no".into(), "yes".into()],
    };
    let b0 = (1.0f64 / 3.0).ln();
    let b1 = 9.0f64.ln();
    (x, y, b0, b1)
}

#[test]
fn logistic_matches_closed_form_mle_on_every_engine() {
    let (x, y, b0, b1) = closed_form_logistic_data();
    for engine in ENGINES {
        let m = LogisticRegSpec::new().engine(engine).fit(&x, &y).unwrap();
        let c = m.coefficients().unwrap();
        // Coefficients are for log-odds of levels[1] = "yes", matching R's glm.
        assert_relative_eq!(c.intercept.unwrap(), b0, max_relative = 1e-6);
        assert_relative_eq!(c.terms[0].1, b1, max_relative = 1e-6);
        let d = m.diagnostics();
        assert_eq!(d.converged, Some(true), "engine {engine:?}: {d:?}");
        assert!(d.final_gradient_norm.unwrap() < 1e-6, "engine {engine:?}: {d:?}");
        assert!(d.warnings.is_empty());
        match m.predict(&x).unwrap() {
            Prediction::Class {
                codes,
                probabilities: Some(p),
                levels,
            } => {
                assert_eq!(levels, vec!["no", "yes"]);
                // P(yes | x=0) = 1/4, P(yes | x=1) = 3/4; event column (levels[0]) is the complement.
                assert_relative_eq!(p.get(0, 1), 0.25, max_relative = 1e-6);
                assert_relative_eq!(p.get(4, 1), 0.75, max_relative = 1e-6);
                assert_relative_eq!(p.get(0, 0), 0.75, max_relative = 1e-6);
                assert_eq!(codes, vec![0, 0, 0, 0, 1, 1, 1, 1]);
            }
            _ => panic!("classification must return probabilities"),
        }
    }
}

#[test]
fn engines_agree_on_a_two_feature_logistic_problem() {
    let x = DesignMatrix::from_columns(vec![
        (
            "a".into(),
            vec![0.1, 0.4, 0.35, 0.8, 0.9, 0.2, 0.7, 0.6, 0.3, 0.5, 0.15, 0.95],
        ),
        (
            "b".into(),
            vec![1.0, 0.0, 1.0, 1.0, 0.0, 0.0, 1.0, 0.0, 1.0, 1.0, 0.0, 1.0],
        ),
    ])
    .unwrap();
    // Deliberately *not* linearly separable (rows 2, 4 and 9 cross the `a` boundary).
    let y = Outcome::Class {
        codes: vec![0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 0, 1],
        levels: vec!["n".into(), "y".into()],
    };
    let a = LogisticRegSpec::new().engine(Engine::Linfa).fit(&x, &y).unwrap();
    let b = LogisticRegSpec::new()
        .engine(Engine::Smartcore)
        .fit(&x, &y)
        .unwrap();
    let c = LogisticRegSpec::new().engine(Engine::Native).fit(&x, &y).unwrap();
    let (ca, cb, cc) = (
        a.coefficients().unwrap(),
        b.coefficients().unwrap(),
        c.coefficients().unwrap(),
    );
    assert_relative_eq!(ca.intercept.unwrap(), cb.intercept.unwrap(), max_relative = 1e-5);
    assert_relative_eq!(ca.intercept.unwrap(), cc.intercept.unwrap(), max_relative = 1e-5);
    for ((ta, tb), tc) in ca.terms.iter().zip(&cb.terms).zip(&cc.terms) {
        assert_relative_eq!(ta.1, tb.1, max_relative = 1e-5);
        assert_relative_eq!(ta.1, tc.1, max_relative = 1e-5);
    }
    assert!(a.diagnostics().converged.unwrap() && b.diagnostics().converged.unwrap());
    let dc = c.diagnostics();
    assert_eq!(dc.converged, Some(true));
    assert!(dc.iterations.is_some() && dc.deviance.is_some(), "{dc:?}");
}

#[test]
fn separable_data_is_flagged_not_hidden() {
    let x = DesignMatrix::from_columns(vec![("x".into(), vec![0., 1., 2., 3., 10., 11., 12., 13.])]).unwrap();
    let y = Outcome::Class {
        codes: vec![0, 0, 0, 0, 1, 1, 1, 1],
        levels: vec!["a".into(), "b".into()],
    };
    for engine in ENGINES {
        match LogisticRegSpec::new().engine(engine).fit(&x, &y) {
            Ok(m) => {
                let d = m.diagnostics();
                assert!(
                    !d.warnings.is_empty() || d.converged == Some(false),
                    "engine {engine:?} returned a silent fit on separable data: {d:?}"
                );
            }
            Err(Error::Backend { .. }) => {} // an explicit backend error is acceptable
            Err(e) => panic!("unexpected error kind: {e}"),
        }
    }
}

#[test]
fn penalty_shrinks_coefficients_and_is_tunable() {
    let (x, y, _, b1) = closed_form_logistic_data();
    let spec = LogisticRegSpec::new();
    assert_eq!(spec.tunable(), &["penalty"]);
    let shrunk = spec.with_param("penalty", 2.0).unwrap().fit(&x, &y).unwrap();
    assert!(shrunk.coefficients().unwrap().terms[0].1.abs() < b1.abs());
    assert!(spec.with_param("penalty", -1.0).is_err());
    assert!(spec.with_param("nope", 1.0).is_err());
}

#[test]
fn malformed_inputs_are_errors_not_panics() {
    let x = DesignMatrix::from_columns(vec![("x".into(), vec![1., 2., 3.])]).unwrap();
    for engine in ENGINES {
        let lin = LinearRegSpec::new().engine(engine);
        assert!(matches!(
            lin.fit(&x, &Outcome::Numeric(vec![1., 2.])),
            Err(Error::InvalidArgument(_))
        ));
        assert!(matches!(
            lin.fit(
                &x,
                &Outcome::Class {
                    codes: vec![0, 1, 0],
                    levels: vec!["a".into(), "b".into()]
                }
            ),
            Err(Error::Unsupported(_))
        ));
        let wide =
            DesignMatrix::from_columns(vec![("a".into(), vec![1., 2.]), ("b".into(), vec![3., 4.])]).unwrap();
        assert!(matches!(
            lin.fit(&wide, &Outcome::Numeric(vec![1., 2.])),
            Err(Error::InvalidArgument(_))
        ));
        let log = LogisticRegSpec::new().engine(engine);
        assert!(matches!(
            log.fit(
                &x,
                &Outcome::Class {
                    codes: vec![0, 0, 0],
                    levels: vec!["a".into(), "b".into()]
                }
            ),
            Err(Error::InvalidValue(_))
        ));
        assert!(matches!(
            log.fit(
                &x,
                &Outcome::Class {
                    codes: vec![0, 1, 2],
                    levels: vec!["a".into(), "b".into(), "c".into()]
                }
            ),
            Err(Error::Unsupported(_))
        ));
    }
}
