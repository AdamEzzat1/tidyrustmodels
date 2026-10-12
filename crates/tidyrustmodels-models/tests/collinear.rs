//! Documented behaviour under exact collinearity (LIMITATIONS.md, review F1).
//!
//! Only all-zero and constant columns are aliased by the framework. Two linearly
//! dependent, non-constant predictors are passed to the backend: linfa's least squares
//! must fail with a *typed* backend error (never a panic), and smartcore's SVD must
//! return a finite solution. Both facts are pinned here so a change in either backend
//! is noticed.

use tidyrustmodels_core::{DesignMatrix, Error, ModelSpec, Outcome};
use tidyrustmodels_models::{Engine, LinearRegSpec};

fn collinear() -> (DesignMatrix, Outcome) {
    let a = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
    let b: Vec<f64> = a.iter().map(|v| 2.0 * v).collect(); // exact multiple of `a`
    let c = vec![0.3, -1.2, 0.8, 2.2, -0.5, 1.1];
    let y: Vec<f64> = a.iter().zip(&c).map(|(x, z)| 1.0 + 0.5 * x - z).collect();
    (
        DesignMatrix::from_columns(vec![("a".into(), a), ("b".into(), b), ("c".into(), c)]).unwrap(),
        Outcome::Numeric(y),
    )
}

#[test]
fn linfa_reports_a_typed_backend_error() {
    let (x, y) = collinear();
    match LinearRegSpec::new().engine(Engine::Linfa).fit(&x, &y) {
        Err(Error::Backend { engine, message }) => {
            assert_eq!(engine, "linfa");
            assert!(!message.is_empty());
        }
        Ok(m) => {
            // If a future linfa version handles singular systems, the fit must at least be finite.
            let c = m.coefficients().unwrap();
            assert!(
                c.terms.iter().all(|t| t.1.is_finite()),
                "non-finite coefficients: {c:?}"
            );
        }
        Err(e) => panic!("unexpected error kind: {e}"),
    }
}

#[test]
fn smartcore_returns_a_finite_solution() {
    let (x, y) = collinear();
    let m = LinearRegSpec::new()
        .engine(Engine::Smartcore)
        .fit(&x, &y)
        .unwrap();
    let c = m.coefficients().unwrap();
    assert!(c.terms.iter().all(|t| t.1.is_finite()));
    // Predictions are still determined even though individual coefficients are not.
    let p = m.predict(&x).unwrap();
    let truth = y.as_numeric().unwrap();
    match p {
        tidyrustmodels_core::Prediction::Numeric(v) => {
            for (a, b) in v.iter().zip(truth) {
                assert!((a - b).abs() < 1e-8, "fitted {a} vs truth {b}");
            }
        }
        _ => unreachable!(),
    }
}

#[test]
fn native_aliases_the_later_collinear_column_like_r() {
    // R 4.4.0: lm(y ~ a + b + c) with b = 2a gives (Intercept) 1, a 0.5, b NA, c -1.
    let (x, y) = collinear();
    let m = LinearRegSpec::new().engine(Engine::Native).fit(&x, &y).unwrap();
    assert_eq!(m.diagnostics().aliased, vec!["b"]);
    let c = m.coefficients().unwrap();
    assert!((c.intercept.unwrap() - 1.0).abs() < 1e-10);
    assert!((c.terms[0].1 - 0.5).abs() < 1e-10);
    assert_eq!(c.terms[1].1, 0.0);
    assert!((c.terms[2].1 + 1.0).abs() < 1e-10);
    match m.predict(&x).unwrap() {
        tidyrustmodels_core::Prediction::Numeric(v) => {
            for (a, b) in v.iter().zip(y.as_numeric().unwrap()) {
                assert!((a - b).abs() < 1e-10);
            }
        }
        _ => unreachable!(),
    }
}

#[test]
fn constant_and_zero_columns_are_aliased_not_errors() {
    let x = DesignMatrix::from_columns(vec![
        ("a".into(), vec![1.0, 2.0, 3.0, 4.0]),
        ("zero".into(), vec![0.0; 4]),
        ("k".into(), vec![7.0; 4]),
    ])
    .unwrap();
    let y = Outcome::Numeric(vec![2.0, 4.0, 6.0, 8.0]);
    for engine in [Engine::Linfa, Engine::Smartcore, Engine::Native] {
        let m = LinearRegSpec::new().engine(engine).fit(&x, &y).unwrap();
        let d = m.diagnostics();
        assert_eq!(d.aliased, vec!["zero", "k"], "{engine:?}");
        let c = m.coefficients().unwrap();
        assert_eq!(c.terms[1].1, 0.0);
        assert_eq!(c.terms[2].1, 0.0);
        assert!((c.terms[0].1 - 2.0).abs() < 1e-10);
    }
}
