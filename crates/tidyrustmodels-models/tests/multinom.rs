//! Multinomial regression baselines independent of R: closed-form fits on saturated
//! designs, agreement between the three engines, the two-level case against
//! `logistic_reg`, and serialization fidelity.

use approx::assert_abs_diff_eq;
use tidyrustmodels_core::{DesignMatrix, FittedModel, ModelSpec, Outcome, Prediction};
use tidyrustmodels_models::{fitted_model_from_state, Engine, LogisticRegSpec, MultinomRegSpec};

const ENGINES: [Engine; 3] = [Engine::Linfa, Engine::Smartcore, Engine::Native];

fn levels(names: &[&str]) -> Vec<String> {
    names.iter().map(|s| s.to_string()).collect()
}

/// Intercept + one dummy: the MLE probabilities are the group proportions, so the
/// baseline-relative coefficients are log-ratios of counts.
#[test]
fn saturated_design_matches_closed_form_on_every_engine() {
    // group 0 (g = 0): a 4, b 2, c 2 ; group 1 (g = 1): a 1, b 3, c 4
    let codes: Vec<u32> = vec![0, 0, 0, 0, 1, 1, 2, 2, 0, 1, 1, 1, 2, 2, 2, 2];
    let g: Vec<f64> = (0..16).map(|i| if i < 8 { 0.0 } else { 1.0 }).collect();
    let x = DesignMatrix::from_columns(vec![("g".into(), g)]).unwrap();
    let y = Outcome::Class {
        codes,
        levels: levels(&["a", "b", "c"]),
    };
    for engine in ENGINES {
        let fit = MultinomRegSpec::new().engine(engine).fit(&x, &y).unwrap();
        let sets = fit.class_coefficients().unwrap();
        assert_eq!(sets[0].0, "b");
        assert_eq!(sets[1].0, "c");
        let tol = if engine == Engine::Native { 1e-9 } else { 1e-5 };
        let (b0_b, b_g_b) = (sets[0].1.intercept.unwrap(), sets[0].1.terms[0].1);
        let (b0_c, b_g_c) = (sets[1].1.intercept.unwrap(), sets[1].1.terms[0].1);
        assert_abs_diff_eq!(b0_b, (2.0f64 / 4.0).ln(), epsilon = tol);
        assert_abs_diff_eq!(b0_c, (2.0f64 / 4.0).ln(), epsilon = tol);
        assert_abs_diff_eq!(b0_b + b_g_b, 3.0f64.ln(), epsilon = tol);
        assert_abs_diff_eq!(b0_c + b_g_c, 4.0f64.ln(), epsilon = tol);
        // probabilities are the group proportions
        match fit.predict(&x).unwrap() {
            Prediction::Class {
                probabilities: Some(p),
                codes,
                ..
            } => {
                assert_abs_diff_eq!(p.get(0, 0), 0.5, epsilon = tol);
                assert_abs_diff_eq!(p.get(8, 2), 0.5, epsilon = tol);
                assert_eq!(codes[0], 0);
                assert_eq!(codes[8], 2);
                for i in 0..16 {
                    assert_abs_diff_eq!(p.get(i, 0) + p.get(i, 1) + p.get(i, 2), 1.0, epsilon = 1e-12);
                }
            }
            _ => panic!(),
        }
        let d = fit.diagnostics();
        assert!(d.final_gradient_norm.unwrap() < 1e-5, "{engine:?}: {d:?}");
        if engine == Engine::Native {
            assert_eq!(d.converged, Some(true));
            assert!(d.deviance.is_some());
        }
    }
}

#[test]
fn engines_agree_on_overlapping_classes() {
    // deterministic pseudo-random design, three classes with overlap
    let n = 90;
    let mut s = 12345u64;
    let mut next = || {
        s = s
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        ((s >> 11) as f64) / ((1u64 << 53) as f64)
    };
    let mut x1 = Vec::new();
    let mut x2 = Vec::new();
    let mut codes = Vec::new();
    for i in 0..n {
        let c = (i % 3) as u32;
        let (mx, my) = [(0.0, 0.0), (1.2, 0.3), (0.3, 1.4)][c as usize];
        x1.push(mx + 2.0 * next() - 1.0);
        x2.push(my + 2.0 * next() - 1.0);
        codes.push(c);
    }
    let x = DesignMatrix::from_columns(vec![("x1".into(), x1), ("x2".into(), x2)]).unwrap();
    let y = Outcome::Class {
        codes,
        levels: levels(&["a", "b", "c"]),
    };
    let native = MultinomRegSpec::new().fit(&x, &y).unwrap();
    let nat = native.class_coefficients().unwrap();
    for engine in [Engine::Linfa, Engine::Smartcore] {
        let fit = MultinomRegSpec::new().engine(engine).fit(&x, &y).unwrap();
        let other = fit.class_coefficients().unwrap();
        for (a, b) in nat.iter().zip(&other) {
            assert_eq!(a.0, b.0);
            assert_abs_diff_eq!(a.1.intercept.unwrap(), b.1.intercept.unwrap(), epsilon = 1e-4);
            for (ta, tb) in a.1.terms.iter().zip(&b.1.terms) {
                assert_abs_diff_eq!(ta.1, tb.1, epsilon = 1e-4);
            }
        }
    }
    // the native deviance is the minimum: evaluate the objective at the L-BFGS solutions
    let dev = |m: &dyn FittedModel| -> f64 {
        match m.predict(&x).unwrap() {
            Prediction::Class {
                probabilities: Some(p),
                ..
            } => {
                let (codes, _) = y.as_class().unwrap();
                -2.0 * codes
                    .iter()
                    .enumerate()
                    .map(|(i, c)| p.get(i, *c as usize).ln())
                    .sum::<f64>()
            }
            _ => panic!(),
        }
    };
    let d_native = dev(native.as_ref());
    assert_abs_diff_eq!(d_native, native.diagnostics().deviance.unwrap(), epsilon = 1e-9);
    for engine in [Engine::Linfa, Engine::Smartcore] {
        let fit = MultinomRegSpec::new().engine(engine).fit(&x, &y).unwrap();
        assert!(dev(fit.as_ref()) >= d_native - 1e-9);
    }
}

#[test]
fn two_levels_reduce_to_logistic_regression() {
    let x = DesignMatrix::from_columns(vec![(
        "x".into(),
        vec![0.1, 0.4, -0.3, 1.2, 0.8, -1.1, 0.0, 0.5, 1.5, -0.7],
    )])
    .unwrap();
    let y = Outcome::Class {
        codes: vec![0, 0, 0, 1, 1, 0, 1, 0, 1, 0],
        levels: levels(&["no", "yes"]),
    };
    // glm's default deviance criterion (1e-8) leaves ~4e-8 in the parameters; tighten it.
    let logit = LogisticRegSpec::new()
        .irls_epsilon(1e-14)
        .fit(&x, &y)
        .unwrap()
        .coefficients()
        .unwrap();
    for engine in ENGINES {
        let m = MultinomRegSpec::new().engine(engine).fit(&x, &y).unwrap();
        let sets = m.class_coefficients().unwrap();
        assert_eq!(sets.len(), 1);
        assert_eq!(sets[0].0, "yes");
        let tol = if engine == Engine::Native { 1e-8 } else { 1e-5 };
        assert_abs_diff_eq!(
            sets[0].1.intercept.unwrap(),
            logit.intercept.unwrap(),
            epsilon = tol
        );
        assert_abs_diff_eq!(sets[0].1.terms[0].1, logit.terms[0].1, epsilon = tol);
    }
}

#[test]
fn aliased_and_absent_levels_are_handled() {
    let x1 = vec![0.1, 0.4, -0.3, 1.2, 0.8, -1.1, 0.0, 0.5, 1.5];
    let x2: Vec<f64> = x1.iter().map(|v| 3.0 * v).collect();
    let x = DesignMatrix::from_columns(vec![("x1".into(), x1), ("x2".into(), x2)]).unwrap();
    let y = Outcome::Class {
        codes: vec![0, 1, 2, 0, 1, 2, 0, 1, 2],
        levels: levels(&["a", "b", "c"]),
    };
    let m = MultinomRegSpec::new().fit(&x, &y).unwrap();
    assert_eq!(m.diagnostics().aliased, vec!["x2".to_string()]);
    for (_, c) in m.class_coefficients().unwrap() {
        assert_eq!(c.terms[1].1, 0.0);
    }
    let absent = Outcome::Class {
        codes: vec![0, 1, 0, 1, 0, 1, 0, 1, 0],
        levels: levels(&["a", "b", "c"]),
    };
    assert!(MultinomRegSpec::new().fit(&x, &absent).is_err());
}

#[test]
fn state_round_trip_is_exact() {
    let x = DesignMatrix::from_columns(vec![("x".into(), vec![0.1, 0.4, -0.3, 1.2, 0.8, -1.1])]).unwrap();
    let y = Outcome::Class {
        codes: vec![0, 1, 2, 0, 1, 2],
        levels: levels(&["a", "b", "c"]),
    };
    let m = MultinomRegSpec::new().penalty(0.5).fit(&x, &y).unwrap();
    let back = fitted_model_from_state(&m.state().unwrap()).unwrap();
    assert_eq!(m.predict(&x).unwrap(), back.predict(&x).unwrap());
    assert_eq!(m.class_coefficients(), back.class_coefficients());
}
