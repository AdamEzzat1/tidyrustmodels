//! Differential verification against pinned R tidymodels fixtures
//! (docs/specs/verification-design.md). Runs offline; fixtures are committed.
//!
//! Every comparison is recorded in a report written to `target/differential_report.md`
//! so VERIFICATION.md can cite measured differences, not intentions.

#![allow(clippy::too_many_arguments)]

use serde_json::Value;
use std::collections::BTreeMap;
use std::path::{Path, PathBuf};
use tidyrustmodels::io::read_csv;
use tidyrustmodels::prelude::*;

fn root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../..")
        .canonicalize()
        .unwrap()
}

fn load_json(rel: &str) -> Value {
    let p = root().join(rel);
    serde_json::from_str(&std::fs::read_to_string(&p).unwrap_or_else(|e| panic!("{}: {e}", p.display())))
        .unwrap()
}

// ---------- tolerances.toml (tiny parser: [section] + key = number) ----------

struct Tolerances {
    defaults: BTreeMap<String, f64>,
    tasks: BTreeMap<String, BTreeMap<String, f64>>,
}

impl Tolerances {
    fn load() -> Self {
        let text = std::fs::read_to_string(root().join("reference/fixtures/tolerances.toml")).unwrap();
        let mut defaults = BTreeMap::new();
        let mut tasks: BTreeMap<String, BTreeMap<String, f64>> = BTreeMap::new();
        let mut section = String::new();
        for line in text.lines() {
            let line = line.split('#').next().unwrap().trim();
            if line.is_empty() {
                continue;
            }
            if let Some(s) = line.strip_prefix('[').and_then(|l| l.strip_suffix(']')) {
                section = s.to_string();
                continue;
            }
            let (k, v) = line.split_once('=').expect("key = value");
            let v: f64 = v.trim().parse().expect("number");
            if section == "defaults" {
                defaults.insert(k.trim().to_string(), v);
            } else if let Some(t) = section.strip_prefix("tasks.") {
                tasks
                    .entry(t.to_string())
                    .or_default()
                    .insert(k.trim().to_string(), v);
            }
        }
        Tolerances { defaults, tasks }
    }

    fn get(&self, task: &str, key: &str) -> f64 {
        self.tasks
            .get(task)
            .and_then(|t| t.get(key))
            .or_else(|| self.defaults.get(key))
            .copied()
            .unwrap_or_else(|| panic!("no tolerance `{key}`"))
    }

    fn pair(&self, task: &str, base: &str) -> (f64, f64) {
        let rel = self.get(task, &format!("{base}_rel"));
        let abs = self
            .tasks
            .get(task)
            .and_then(|t| t.get(&format!("{base}_abs")))
            .or_else(|| self.defaults.get(&format!("{base}_abs")))
            .copied()
            .unwrap_or(0.0);
        (rel, abs)
    }
}

// ---------- report ----------

struct Row {
    task: String,
    engine: String,
    level: &'static str,
    quantity: String,
    n: usize,
    max_abs: f64,
    max_rel: f64,
    bound: String,
    pass: bool,
}

struct Report {
    rows: Vec<Row>,
}

impl Report {
    fn compare(
        &mut self,
        task: &str,
        engine: &str,
        level: &'static str,
        quantity: &str,
        ours: &[f64],
        theirs: &[f64],
        (rel, abs): (f64, f64),
    ) {
        assert_eq!(
            ours.len(),
            theirs.len(),
            "{task}/{engine}/{quantity}: length mismatch"
        );
        let mut max_abs: f64 = 0.0;
        let mut max_rel: f64 = 0.0;
        let mut pass = true;
        for (a, b) in ours.iter().zip(theirs) {
            let d = (a - b).abs();
            let scale = a.abs().max(b.abs());
            let r = if scale > 0.0 { d / scale } else { 0.0 };
            max_abs = max_abs.max(d);
            max_rel = max_rel.max(r);
            if !(d <= abs || r <= rel) {
                pass = false;
            }
        }
        self.rows.push(Row {
            task: task.into(),
            engine: engine.into(),
            level,
            quantity: quantity.into(),
            n: ours.len(),
            max_abs,
            max_rel,
            bound: format!("rel {rel:.0e} / abs {abs:.0e}"),
            pass,
        });
    }

    fn exact<T: PartialEq + std::fmt::Debug>(
        &mut self,
        task: &str,
        engine: &str,
        level: &'static str,
        quantity: &str,
        ours: &T,
        theirs: &T,
    ) {
        let pass = ours == theirs;
        self.rows.push(Row {
            task: task.into(),
            engine: engine.into(),
            level,
            quantity: quantity.into(),
            n: 1,
            max_abs: if pass { 0.0 } else { f64::NAN },
            max_rel: if pass { 0.0 } else { f64::NAN },
            bound: "exact".into(),
            pass,
        });
        if !pass {
            eprintln!(
                "EXACT MISMATCH {task}/{engine}/{quantity}:\n  ours   = {ours:?}\n  theirs = {theirs:?}"
            );
        }
    }

    fn write_and_assert(&self) {
        let mut md = String::from("| task | engine | level | quantity | n | max abs diff | max rel diff | bound | pass |\n|---|---|---|---|---|---|---|---|---|\n");
        for r in &self.rows {
            md.push_str(&format!(
                "| {} | {} | {} | {} | {} | {:.2e} | {:.2e} | {} | {} |\n",
                r.task,
                r.engine,
                r.level,
                r.quantity,
                r.n,
                r.max_abs,
                r.max_rel,
                r.bound,
                if r.pass { "yes" } else { "**NO**" }
            ));
        }
        let out = root().join("target/differential_report.md");
        std::fs::create_dir_all(out.parent().unwrap()).unwrap();
        std::fs::write(&out, &md).unwrap();
        eprintln!("{md}");
        let failed: Vec<&Row> = self.rows.iter().filter(|r| !r.pass).collect();
        assert!(
            failed.is_empty(),
            "{} comparisons exceeded tolerance (see target/differential_report.md)",
            failed.len()
        );
    }
}

// ---------- fixture helpers ----------

/// jsonlite's `auto_unbox` writes length-1 vectors as scalars; treat both uniformly.
fn items(v: &Value) -> Vec<&Value> {
    match v {
        Value::Array(a) => a.iter().collect(),
        Value::Null => Vec::new(),
        other => vec![other],
    }
}

fn f64s(v: &Value) -> Vec<f64> {
    items(v)
        .into_iter()
        .map(|x| x.as_f64().unwrap_or(f64::NAN))
        .collect()
}

fn strs(v: &Value) -> Vec<String> {
    items(v)
        .into_iter()
        .map(|x| {
            x.as_str()
                .unwrap_or_else(|| panic!("expected string, got {x}"))
                .to_string()
        })
        .collect()
}

fn u32s(v: &Value) -> Vec<u32> {
    items(v).into_iter().map(|x| x.as_u64().unwrap() as u32).collect()
}

fn build_recipe(task: &Value) -> Recipe {
    let mut rec = Recipe::new(task["outcome"].as_str().unwrap());
    for s in task["steps"].as_array().unwrap() {
        rec = match s.as_str().unwrap() {
            "impute_median_numeric" => rec.step(StepImputeMedian::new(Selector::AllNumericPredictors)),
            "impute_mean_numeric" => rec.step(StepImputeMean::new(Selector::AllNumericPredictors)),
            "normalize_numeric" => rec.step(StepNormalize::new(Selector::AllNumericPredictors)),
            "dummy_nominal" => rec.step(StepDummy::new(Selector::AllNominalPredictors)),
            other => panic!("unknown step {other}"),
        };
    }
    rec
}

fn task_table(task: &Value) -> Table {
    let full = read_csv(root().join(format!(
        "reference/fixtures/datasets/{}.csv",
        task["dataset"].as_str().unwrap()
    )))
    .unwrap();
    let outcome = task["outcome"].as_str().unwrap();
    let mut cols = vec![(outcome.to_string(), full.column(outcome).unwrap().clone())];
    for p in strs(&task["predictors"]) {
        cols.push((p.clone(), full.column(&p).unwrap().clone()));
    }
    Table::new(cols).unwrap()
}

fn model_for(task: &Value, engine: Engine) -> Box<dyn ModelSpec> {
    if task["type"] == "regression" {
        Box::new(LinearRegSpec::new().engine(engine))
    } else {
        Box::new(LogisticRegSpec::new().engine(engine))
    }
}

fn matrix_columns(x: &DesignMatrix) -> Vec<(String, Vec<f64>)> {
    (0..x.ncols())
        .map(|j| (x.names()[j].clone(), x.column(j).to_vec()))
        .collect()
}

// ---------- the suite ----------

#[test]
fn differential_against_r_fixtures() {
    let tol = Tolerances::load();
    let tasks = load_json("reference/tasks.json");
    let mut report = Report { rows: Vec::new() };

    for task in tasks["tasks"].as_array().unwrap() {
        let id = task["id"].as_str().unwrap();
        let fx = load_json(&format!("reference/fixtures/r/{id}.json"));
        let split = load_json(&format!("reference/fixtures/splits/{id}.json"));
        let data = task_table(task);
        let train_ids: Vec<usize> = u32s(&split["initial"]["train"])
            .into_iter()
            .map(|v| v as usize)
            .collect();
        let test_ids: Vec<usize> = u32s(&split["initial"]["test"])
            .into_iter()
            .map(|v| v as usize)
            .collect();
        let train = data.select_rows(&train_ids).unwrap();
        let test = data.select_rows(&test_ids).unwrap();
        assert_eq!(train.nrows() as u64, fx["n_train"].as_u64().unwrap());
        let is_reg = task["type"] == "regression";
        let outcome = task["outcome"].as_str().unwrap();

        // ---- Level 1: preprocessing ----
        let recipe = build_recipe(task);
        let prepped = recipe.prep(&train).unwrap();
        let r_steps = items(&fx["prep"]["steps"]);
        assert_eq!(prepped.steps().len(), r_steps.len());
        for (ours, theirs) in prepped.steps().iter().zip(r_steps) {
            let state = ours.state().unwrap();
            let tidy = &theirs["tidy"];
            match theirs["step"].as_str().unwrap() {
                "step_normalize" => {
                    let terms = strs(&tidy["terms"]);
                    let stat = strs(&tidy["statistic"]);
                    let vals = f64s(&tidy["value"]);
                    let mut r_means = Vec::new();
                    let mut r_sds = Vec::new();
                    let our_stats = state["stats"].as_array().unwrap();
                    for (name, _, _) in our_stats.iter().map(|s| {
                        (
                            s[0].as_str().unwrap(),
                            s[1].as_f64().unwrap(),
                            s[2].as_f64().unwrap(),
                        )
                    }) {
                        let find = |which: &str| {
                            terms
                                .iter()
                                .zip(&stat)
                                .zip(&vals)
                                .find(|((t, s), _)| *t == name && *s == which)
                                .map(|(_, v)| *v)
                                .unwrap()
                        };
                        r_means.push(find("mean"));
                        r_sds.push(find("sd"));
                    }
                    let our_means: Vec<f64> = our_stats.iter().map(|s| s[1].as_f64().unwrap()).collect();
                    let our_sds: Vec<f64> = our_stats.iter().map(|s| s[2].as_f64().unwrap()).collect();
                    let b = tol.pair(id, "prep_stats");
                    report.compare(id, "-", "L1", "normalize means", &our_means, &r_means, b);
                    report.compare(id, "-", "L1", "normalize sds", &our_sds, &r_sds, b);
                }
                "step_impute_median" | "step_impute_mean" => {
                    let terms = strs(&tidy["terms"]);
                    let vals = f64s(&tidy["value"]);
                    let fills = state["fills"].as_array().unwrap();
                    let ours: Vec<f64> = fills.iter().map(|f| f[1].as_f64().unwrap()).collect();
                    let theirs: Vec<f64> = fills
                        .iter()
                        .map(|f| {
                            let n = f[0].as_str().unwrap();
                            terms
                                .iter()
                                .zip(&vals)
                                .find(|(t, _)| *t == n)
                                .map(|(_, v)| *v)
                                .unwrap()
                        })
                        .collect();
                    report.compare(
                        id,
                        "-",
                        "L1",
                        "impute fills",
                        &ours,
                        &theirs,
                        tol.pair(id, "prep_stats"),
                    );
                }
                "step_dummy" => {
                    let r_cols = strs(&tidy["columns"]);
                    let ours: Vec<String> = state["encodings"]
                        .as_array()
                        .unwrap()
                        .iter()
                        .flat_map(|e| {
                            e[2].as_array()
                                .unwrap()
                                .iter()
                                .map(|v| v.as_str().unwrap().to_string())
                        })
                        .collect();
                    // tidy(step_dummy) lists the *levels* used per term; names are var_level.
                    let terms = strs(&tidy["terms"]);
                    let r_names: Vec<String> = terms
                        .iter()
                        .zip(&r_cols)
                        .map(|(t, c)| format!("{t}_{c}"))
                        .collect();
                    report.exact(id, "-", "L1", "dummy column names", &ours, &r_names);
                }
                other => panic!("unexpected R step {other}"),
            }
        }
        let (x_train, y_train) = prepped.bake_matrix(&train).unwrap();
        let (x_test, y_test) = prepped.bake_matrix(&test).unwrap();
        let (y_train, y_test) = (y_train.unwrap(), y_test.unwrap());
        report.exact(
            id,
            "-",
            "L1",
            "predictor names/order",
            &x_train.names().to_vec(),
            &strs(&fx["prep"]["predictor_names"]),
        );
        for (name, col) in matrix_columns(&x_train) {
            report.compare(
                id,
                "-",
                "L1",
                &format!("baked train `{name}`"),
                &col,
                &f64s(&fx["prep"]["baked_train"][&name]),
                tol.pair(id, "baked"),
            );
        }
        for (name, col) in matrix_columns(&x_test) {
            report.compare(
                id,
                "-",
                "L1",
                &format!("baked test `{name}`"),
                &col,
                &f64s(&fx["prep"]["baked_test"][&name]),
                tol.pair(id, "baked"),
            );
        }

        // ---- Level 3 + 4 per engine ----
        let metrics = MetricSet::from_names(&strs(&task["metrics"])).unwrap();
        for engine in [Engine::Linfa, Engine::Smartcore] {
            let eng = engine.as_str();
            let spec = model_for(task, engine);
            let fitted = spec.fit(&x_train, &y_train).unwrap();
            let coefs = fitted.coefficients().unwrap();
            let r_names = strs(&fx["fit"]["coefficient_names"]);
            let our_names: Vec<String> = std::iter::once("(Intercept)".to_string())
                .chain(coefs.terms.iter().map(|t| t.0.clone()))
                .collect();
            report.exact(id, eng, "L3", "coefficient names", &our_names, &r_names);
            let all_ours: Vec<f64> = std::iter::once(coefs.intercept.unwrap())
                .chain(coefs.terms.iter().map(|t| t.1))
                .collect();
            let mut our_vals = Vec::new();
            let mut r_vals = Vec::new();
            let aliased = fitted.diagnostics().aliased;
            for (n, v) in r_names.iter().zip(&all_ours) {
                match fx["fit"]["coefficients"][n].as_f64() {
                    Some(r) => {
                        our_vals.push(*v);
                        r_vals.push(r);
                    }
                    // R reports NA for an aliased coefficient; we must have flagged the same column.
                    None => report.exact(
                        id,
                        eng,
                        "L3",
                        &format!("aliased `{n}` (R: NA)"),
                        &true,
                        &aliased.contains(n),
                    ),
                }
            }
            let coef_tol = tol.pair(id, if is_reg { "ols_coef" } else { "logistic_coef" });
            report.compare(id, eng, "L3", "coefficients", &our_vals, &r_vals, coef_tol);

            let pred_test = fitted.predict(&x_test).unwrap();
            if is_reg {
                let ours = match &pred_test {
                    Prediction::Numeric(p) => p.clone(),
                    _ => unreachable!(),
                };
                report.compare(
                    id,
                    eng,
                    "L3",
                    "test predictions",
                    &ours,
                    &f64s(&fx["fit"]["predicted_test"]),
                    tol.pair(id, "ols_pred"),
                );
            } else {
                let (codes, levels, probs) = match &pred_test {
                    Prediction::Class {
                        codes,
                        levels,
                        probabilities: Some(p),
                    } => (codes, levels, p),
                    _ => unreachable!(),
                };
                report.exact(
                    id,
                    eng,
                    "L3",
                    "outcome levels",
                    levels,
                    &strs(&fx["fit"]["outcome_levels"]),
                );
                for (j, lv) in levels.iter().enumerate() {
                    let theirs = f64s(&fx["fit"]["predicted_test"][format!(".pred_{lv}")]);
                    report.compare(
                        id,
                        eng,
                        "L3",
                        &format!("test prob `{lv}`"),
                        probs.column(j),
                        &theirs,
                        (0.0, tol.get(id, "logistic_prob_abs")),
                    );
                }
                let our_classes: Vec<String> = codes.iter().map(|c| levels[*c as usize].clone()).collect();
                report.exact(
                    id,
                    eng,
                    "L3",
                    "test classes",
                    &our_classes,
                    &strs(&fx["fit"]["class_test"]),
                );
                let d = fitted.diagnostics();
                report.exact(
                    id,
                    eng,
                    "L3",
                    "converged",
                    &d.converged,
                    &Some(fx["fit"]["converged"].as_bool().unwrap()),
                );
            }
            // test-set metrics (level 2 on real predictions)
            let ours = metrics.evaluate(&y_test, &pred_test).unwrap();
            for m in &ours {
                let theirs = fx["test_metrics"][&m.name].as_f64();
                match (m.value, theirs) {
                    (Some(a), Some(b)) => report.compare(
                        id,
                        eng,
                        "L2",
                        &format!("test metric {}", m.name),
                        &[a],
                        &[b],
                        tol.pair(id, "resample_metric"),
                    ),
                    (a, b) => report.exact(id, eng, "L2", &format!("test metric {} (NA)", m.name), &a, &b),
                }
            }

            // ---- Level 4: resampling with shared folds ----
            let folds = Folds::from_assignments(&u32s(&split["vfold"]["fold_of_row"])).unwrap();
            let wf = Workflow::from_boxed(build_recipe(task), spec);
            let res = wf
                .fit_resamples(&data, &folds, &metrics, Parallelism::threads(4))
                .unwrap();
            for fold in &res.per_fold {
                for m in &fold.metrics {
                    let theirs = fx["resample"]["per_fold"][&fold.id][&m.name].as_f64();
                    match (m.value, theirs) {
                        (Some(a), Some(b)) => report.compare(
                            id,
                            eng,
                            "L4",
                            &format!("{} {}", fold.id, m.name),
                            &[a],
                            &[b],
                            tol.pair(id, "resample_metric"),
                        ),
                        (a, b) => {
                            report.exact(id, eng, "L4", &format!("{} {} (NA)", fold.id, m.name), &a, &b)
                        }
                    }
                }
            }
            for s in &res.summary {
                let theirs = *items(&fx["resample"]["summary"])
                    .iter()
                    .find(|r| r["metric"] == s.name)
                    .unwrap();
                report.compare(
                    id,
                    eng,
                    "L4",
                    &format!("summary mean {}", s.name),
                    &[s.mean.unwrap()],
                    &[theirs["mean"].as_f64().unwrap()],
                    tol.pair(id, "summary"),
                );
                report.compare(
                    id,
                    eng,
                    "L4",
                    &format!("summary std_err {}", s.name),
                    &[s.std_err.unwrap()],
                    &[theirs["std_err"].as_f64().unwrap()],
                    tol.pair(id, "summary"),
                );
                report.exact(
                    id,
                    eng,
                    "L4",
                    &format!("summary n {}", s.name),
                    &(s.n as u64),
                    &theirs["n"].as_u64().unwrap(),
                );
            }
        }
        let _ = outcome;
    }

    // ---- Level 2: metrics on fixed arrays ----
    let fx = load_json("reference/fixtures/r/metrics_fixed.json");
    let b = tol.pair("-", "metrics_fixed");
    let t = f64s(&fx["regression"]["truth"]);
    let e = f64s(&fx["regression"]["estimate"]);
    for (name, f) in [
        ("rmse", rmse_vec as fn(&[f64], &[f64]) -> f64),
        ("mae", mae_vec),
        ("rsq", rsq_vec),
        ("rsq_trad", rsq_trad_vec),
    ] {
        report.compare(
            "metrics_fixed",
            "-",
            "L2",
            name,
            &[f(&t, &e)],
            &[fx["regression"][name].as_f64().unwrap()],
            b,
        );
    }
    let levels = strs(&fx["classification"]["levels"]);
    let code = |v: &Value| -> Vec<u32> {
        strs(v)
            .iter()
            .map(|s| levels.iter().position(|l| l == s).unwrap() as u32)
            .collect()
    };
    let (tc, ec, p) = (
        code(&fx["classification"]["truth"]),
        code(&fx["classification"]["estimate"]),
        f64s(&fx["classification"]["prob_event"]),
    );
    use tidyrustmodels::metrics as ym;
    report.compare(
        "metrics_fixed",
        "-",
        "L2",
        "accuracy",
        &[ym::accuracy(&tc, &ec).unwrap().unwrap()],
        &[fx["classification"]["accuracy"].as_f64().unwrap()],
        b,
    );
    report.compare(
        "metrics_fixed",
        "-",
        "L2",
        "precision",
        &[ym::precision(&tc, &ec, 0).unwrap().unwrap()],
        &[fx["classification"]["precision"].as_f64().unwrap()],
        b,
    );
    report.compare(
        "metrics_fixed",
        "-",
        "L2",
        "recall",
        &[ym::recall(&tc, &ec, 0).unwrap().unwrap()],
        &[fx["classification"]["recall"].as_f64().unwrap()],
        b,
    );
    report.compare(
        "metrics_fixed",
        "-",
        "L2",
        "mn_log_loss",
        &[ym::mn_log_loss(&tc, &p, 0).unwrap().unwrap()],
        &[fx["classification"]["mn_log_loss"].as_f64().unwrap()],
        b,
    );
    report.exact(
        "metrics_fixed",
        "-",
        "L2",
        "rsq constant truth is NA",
        &ym::rsq(&[1.0, 1.0, 1.0], &[1.0, 2.0, 3.0]).unwrap(),
        &fx["undefined"]["rsq_constant_truth"].as_f64(),
    );
    report.exact(
        "metrics_fixed",
        "-",
        "L2",
        "precision undefined is NA",
        &ym::precision(&[1, 1], &[1, 1], 0).unwrap(),
        &fx["undefined"]["precision_no_predicted_events"].as_f64(),
    );
    report.compare(
        "metrics_fixed",
        "-",
        "L2",
        "log loss eps clamp",
        &[ym::mn_log_loss(&[0, 1], &[0.0, 0.5], 0).unwrap().unwrap()],
        &[fx["undefined"]["log_loss_clamped"].as_f64().unwrap()],
        b,
    );

    report.write_and_assert();
}

fn rmse_vec(t: &[f64], e: &[f64]) -> f64 {
    tidyrustmodels::metrics::rmse(t, e).unwrap().unwrap()
}
fn mae_vec(t: &[f64], e: &[f64]) -> f64 {
    tidyrustmodels::metrics::mae(t, e).unwrap().unwrap()
}
fn rsq_vec(t: &[f64], e: &[f64]) -> f64 {
    tidyrustmodels::metrics::rsq(t, e).unwrap().unwrap()
}
fn rsq_trad_vec(t: &[f64], e: &[f64]) -> f64 {
    tidyrustmodels::metrics::rsq_trad(t, e).unwrap().unwrap()
}
