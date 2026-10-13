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

    fn lookup(&self, task: &str, key: &str) -> Option<f64> {
        self.tasks
            .get(task)
            .and_then(|t| t.get(key))
            .or_else(|| self.defaults.get(key))
            .copied()
    }

    /// Engine-specific bound (`<engine>_<key>`) when present, else the shared one.
    fn get_eng(&self, task: &str, key: &str, engine: &str) -> f64 {
        self.lookup(task, &format!("{engine}_{key}"))
            .unwrap_or_else(|| self.get(task, key))
    }

    fn pair_eng(&self, task: &str, base: &str, engine: &str) -> (f64, f64) {
        let rel = self.get_eng(task, &format!("{base}_rel"), engine);
        let abs = self
            .lookup(task, &format!("{engine}_{base}_abs"))
            .or_else(|| self.lookup(task, &format!("{base}_abs")))
            .unwrap_or(0.0);
        (rel, abs)
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
            "zv" => rec.step(StepZv::new(Selector::AllPredictors)),
            "nzv" => rec.step(StepNzv::new(Selector::AllPredictors)),
            "corr_numeric" => rec.step(StepCorr::new(Selector::AllNumericPredictors)),
            "other_nominal" => rec.step(StepOther::new(Selector::AllNominalPredictors)),
            "novel_nominal" => rec.step(StepNovel::new(Selector::AllNominalPredictors)),
            "unknown_nominal" => rec.step(StepUnknown::new(Selector::AllNominalPredictors)),
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

fn model_for(task: &Value, engine: Engine, y: &Outcome) -> Box<dyn ModelSpec> {
    model_for_seed(task, engine, y, 0)
}

fn model_for_seed(task: &Value, engine: Engine, y: &Outcome, seed: u64) -> Box<dyn ModelSpec> {
    if task["model"] == "rand_forest" {
        let a = &task["model_args"];
        let mut s = RandForestSpec::new().engine(engine).seed(Seed(seed));
        if let Some(v) = a["mtry"].as_u64() {
            s = s.mtry(v as usize);
        }
        if let Some(v) = a["trees"].as_u64() {
            s = s.trees(v as usize);
        }
        if let Some(v) = a["min_n"].as_u64() {
            s = s.min_n(v as usize);
        }
        return Box::new(s);
    }
    if task["model"] == "decision_tree" {
        let a = &task["model_args"];
        let mut s = DecisionTreeSpec::new().engine(engine);
        if let Some(v) = a["cost_complexity"].as_f64() {
            s = s.cost_complexity(v);
        }
        if let Some(v) = a["tree_depth"].as_u64() {
            s = s.tree_depth(v as usize);
        }
        if let Some(v) = a["min_n"].as_u64() {
            s = s.min_n(v as usize);
        }
        return Box::new(s);
    }
    if task["type"] == "regression" {
        Box::new(LinearRegSpec::new().engine(engine))
    } else if y.as_class().map(|(_, l)| l.len()).unwrap_or(0) > 2 {
        Box::new(MultinomRegSpec::new().engine(engine))
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
            let theirs_step = theirs["step"].as_str().unwrap();
            match theirs_step {
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
                "step_zv" | "step_nzv" | "step_corr" => {
                    let ours: Vec<String> = state["removals"]
                        .as_array()
                        .unwrap()
                        .iter()
                        .map(|v| v.as_str().unwrap().to_string())
                        .collect();
                    let mut theirs = strs(&tidy["terms"]);
                    let mut ours_sorted = ours.clone();
                    // Order of removals is not part of the contract; membership is.
                    ours_sorted.sort();
                    theirs.sort();
                    report.exact(
                        id,
                        "-",
                        "L1",
                        &format!("{} removals", theirs_step),
                        &ours_sorted,
                        &theirs,
                    );
                }
                "step_other" => {
                    let terms = strs(&tidy["terms"]);
                    let retained = strs(&tidy["retained"]);
                    for o in state["objects"].as_array().unwrap() {
                        let col = o["column"].as_str().unwrap();
                        let ours: Vec<String> = o["keep"]
                            .as_array()
                            .unwrap()
                            .iter()
                            .map(|v| v.as_str().unwrap().to_string())
                            .collect();
                        let theirs: Vec<String> = terms
                            .iter()
                            .zip(&retained)
                            .filter(|(t, _)| *t == col)
                            .map(|(_, r)| r.clone())
                            .collect();
                        report.exact(id, "-", "L1", &format!("other retained `{col}`"), &ours, &theirs);
                    }
                }
                "step_novel" | "step_unknown" => {
                    let terms = strs(&tidy["terms"]);
                    let values = strs(&tidy["value"]);
                    let ours: Vec<String> = state["objects"]
                        .as_array()
                        .unwrap()
                        .iter()
                        .map(|o| o[0].as_str().unwrap().to_string())
                        .collect();
                    report.exact(id, "-", "L1", &format!("{} columns", theirs_step), &ours, &terms);
                    let our_level = state["new_level"].as_str().unwrap().to_string();
                    report.exact(
                        id,
                        "-",
                        "L1",
                        &format!("{} level", theirs_step),
                        &our_level,
                        &values[0],
                    );
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
        let is_tree = task["model"] == "decision_tree";
        let is_rf = task["model"] == "rand_forest";
        let engines: Vec<Engine> = if is_tree || is_rf {
            vec![Engine::Native]
        } else {
            vec![Engine::Linfa, Engine::Smartcore, Engine::Native]
        };
        for engine in engines {
            let eng = engine.as_str();
            if is_rf {
                compare_forest(
                    &mut report,
                    &tol,
                    id,
                    eng,
                    task,
                    &x_train,
                    &y_train,
                    &x_test,
                    &y_test,
                    &metrics,
                    &fx,
                );
                continue;
            }
            let spec = model_for(task, engine, &y_train);
            let is_multi = spec.name() == "multinom_reg";
            let fitted = spec.fit(&x_train, &y_train).unwrap();
            let r_names = strs(&fx["fit"]["coefficient_names"]);
            let aliased = fitted.diagnostics().aliased;
            if is_tree {
                compare_tree(&mut report, &tol, id, eng, fitted.as_ref(), &fx["fit"]["tree"]);
            }
            // One coefficient set for lm/glm; one per non-baseline level for multinom.
            let sets: Vec<(Option<String>, Coefficients)> = if is_tree {
                Vec::new()
            } else if is_multi {
                let cc = fitted.class_coefficients().unwrap();
                report.exact(
                    id,
                    eng,
                    "L3",
                    "coefficient levels",
                    &cc.iter().map(|(l, _)| l.clone()).collect::<Vec<_>>(),
                    &strs(&fx["fit"]["coefficient_levels"]),
                );
                cc.into_iter().map(|(l, c)| (Some(l), c)).collect()
            } else {
                vec![(None, fitted.coefficients().unwrap())]
            };
            let coef_key = if is_reg {
                "ols_coef"
            } else if is_multi {
                "multinom_coef"
            } else {
                "logistic_coef"
            };
            let mut class_dump = serde_json::Map::new();
            for (level, coefs) in &sets {
                let label = level.as_deref().map(|l| format!(" `{l}`")).unwrap_or_default();
                let our_names: Vec<String> = std::iter::once("(Intercept)".to_string())
                    .chain(coefs.terms.iter().map(|t| t.0.clone()))
                    .collect();
                report.exact(
                    id,
                    eng,
                    "L3",
                    &format!("coefficient names{label}"),
                    &our_names,
                    &r_names,
                );
                let all_ours: Vec<f64> = std::iter::once(coefs.intercept.unwrap())
                    .chain(coefs.terms.iter().map(|t| t.1))
                    .collect();
                let theirs = match level {
                    Some(l) => &fx["fit"]["coefficients"][l],
                    None => &fx["fit"]["coefficients"],
                };
                let mut our_vals = Vec::new();
                let mut r_vals = Vec::new();
                for (n, v) in r_names.iter().zip(&all_ours) {
                    match theirs[n].as_f64() {
                        Some(r) => {
                            our_vals.push(*v);
                            r_vals.push(r);
                        }
                        // R reports NA for an aliased coefficient; we must have flagged the same column.
                        None => report.exact(
                            id,
                            eng,
                            "L3",
                            &format!("aliased `{n}`{label} (R: NA)"),
                            &true,
                            &aliased.contains(n),
                        ),
                    }
                }
                report.compare(
                    id,
                    eng,
                    "L3",
                    &format!("coefficients{label}"),
                    &our_vals,
                    &r_vals,
                    tol.pair_eng(id, coef_key, eng),
                );
                if let Some(l) = level {
                    let m: serde_json::Map<String, Value> = r_names
                        .iter()
                        .zip(&all_ours)
                        .map(|(n, v)| (n.clone(), serde_json::json!(v)))
                        .collect();
                    class_dump.insert(l.clone(), Value::Object(m));
                }
            }
            let coef_names: Option<Vec<String>> = sets.first().map(|(_, c)| {
                std::iter::once("(Intercept)".to_string())
                    .chain(c.terms.iter().map(|t| t.0.clone()))
                    .collect()
            });
            let coef_values: Option<Vec<f64>> = sets.first().map(|(_, c)| {
                std::iter::once(c.intercept.unwrap())
                    .chain(c.terms.iter().map(|t| t.1))
                    .collect()
            });
            if is_multi {
                // Both solvers minimise the same deviance; ours must not end above nnet's.
                // Deviance recomputed by the harness from the fitted probabilities on the
                // training rows (the L-BFGS engines report none; the native one must agree).
                let d = match fitted.predict(&x_train).unwrap() {
                    Prediction::Class {
                        probabilities: Some(p),
                        ..
                    } => {
                        let (codes, _) = y_train.as_class().unwrap();
                        -2.0 * codes
                            .iter()
                            .enumerate()
                            .map(|(i, c)| p.get(i, *c as usize).ln())
                            .sum::<f64>()
                    }
                    _ => unreachable!(),
                };
                if let Some(dd) = fitted.diagnostics().deviance {
                    report.compare(
                        id,
                        eng,
                        "L3",
                        "reported deviance vs recomputed",
                        &[dd],
                        &[d],
                        (1e-12, 0.0),
                    );
                }
                let r_dev = fx["fit"]["deviance"].as_f64().unwrap();
                report.exact(
                    id,
                    eng,
                    "L3",
                    "deviance <= nnet's (within 1e-9 rel)",
                    &true,
                    &(d <= r_dev * (1.0 + 1e-9)),
                );
                report.compare(
                    id,
                    eng,
                    "L3",
                    "deviance",
                    &[d],
                    &[r_dev],
                    tol.pair_eng(id, "multinom_deviance", eng),
                );
            }

            let pred_test = fitted.predict(&x_test).unwrap();
            // Level 5 support: dump this engine's outputs for the Python parity test.
            let mut dump = serde_json::json!({
                "task": id, "engine": eng,
                "coefficient_names": coef_names,
                "class_coefficients": if is_multi { Value::Object(class_dump.clone()) } else { Value::Null },
                "coefficients": coef_values,
            });
            match &pred_test {
                Prediction::Numeric(p) => dump["predicted_test"] = serde_json::json!(p),
                Prediction::Class {
                    codes,
                    levels,
                    probabilities,
                } => {
                    dump["prob_event_test"] = serde_json::json!(probabilities.as_ref().unwrap().column(0));
                    dump["prob_test"] = serde_json::json!(probabilities.as_ref().unwrap().to_row_major());
                    dump["class_test"] = serde_json::json!(codes
                        .iter()
                        .map(|c| levels[*c as usize].clone())
                        .collect::<Vec<_>>());
                }
            }
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
                    tol.pair_eng(id, if is_tree { "tree_pred" } else { "ols_pred" }, eng),
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
                        (
                            0.0,
                            tol.get_eng(
                                id,
                                if is_tree {
                                    "tree_prob_abs"
                                } else if is_multi {
                                    "multinom_prob_abs"
                                } else {
                                    "logistic_prob_abs"
                                },
                                eng,
                            ),
                        ),
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
                if !is_tree {
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
                        tol.pair_eng(id, "resample_metric", eng),
                    ),
                    (a, b) => report.exact(id, eng, "L2", &format!("test metric {} (NA)", m.name), &a, &b),
                }
            }

            // ---- Level 4: resampling with shared folds ----
            let folds = Folds::from_assignments(&u32s(&split["vfold"]["fold_of_row"])).unwrap();
            let wf = Workflow::from_boxed(build_recipe(task), spec);
            if is_tree {
                let fold_of_row = u32s(&split["vfold"]["fold_of_row"]);
                let v = split["vfold"]["v"].as_u64().unwrap() as u32;
                for k in 0..v {
                    let analysis: Vec<usize> = fold_of_row
                        .iter()
                        .enumerate()
                        .filter(|(_, f)| **f != k)
                        .map(|(i, _)| i)
                        .collect();
                    let fw = wf.fit(&data.select_rows(&analysis).unwrap()).unwrap();
                    let fxk = &fx["resample"]["fold_trees"][k as usize];
                    let label = format!("Fold{:02}", k + 1);
                    compare_tree_structure(&mut report, &tol, id, eng, &label, fw.model(), fxk);
                    // assessment predictions: ours vs R's, and the metrics recomputed on
                    // R's own predictions (isolates the metric port from the model)
                    let assess: Vec<usize> = fold_of_row
                        .iter()
                        .enumerate()
                        .filter(|(_, f)| **f == k)
                        .map(|(i, _)| i)
                        .collect();
                    let assess_data = data.select_rows(&assess).unwrap();
                    let ours = fw.predict(&assess_data).unwrap();
                    let truth =
                        Outcome::from_column(assess_data.column(task["outcome"].as_str().unwrap()).unwrap())
                            .unwrap();
                    let r_pred = match &ours {
                        Prediction::Numeric(p) => {
                            let r = f64s(&fxk["assess_pred"]);
                            report.compare(
                                id,
                                eng,
                                "L4",
                                &format!("{label} assessment predictions"),
                                p,
                                &r,
                                tol.pair_eng(id, "tree_pred", eng),
                            );
                            Prediction::Numeric(r)
                        }
                        Prediction::Class {
                            codes,
                            levels,
                            probabilities: Some(p),
                        } => {
                            let r_rows: Vec<Vec<f64>> =
                                items(&fxk["assess_prob"]).into_iter().map(f64s).collect();
                            let flat_ours = p.to_row_major();
                            let flat_r: Vec<f64> = r_rows.iter().flatten().copied().collect();
                            report.compare(
                                id,
                                eng,
                                "L4",
                                &format!("{label} assessment probabilities"),
                                &flat_ours,
                                &flat_r,
                                (0.0, tol.get_eng(id, "tree_prob_abs", eng)),
                            );
                            let r_class: Vec<String> = strs(&fxk["assess_class"]);
                            report.exact(
                                id,
                                eng,
                                "L4",
                                &format!("{label} assessment classes"),
                                &codes
                                    .iter()
                                    .map(|c| levels[*c as usize].clone())
                                    .collect::<Vec<_>>(),
                                &r_class,
                            );
                            let r_codes: Vec<u32> = r_class
                                .iter()
                                .map(|s| levels.iter().position(|l| l == s).unwrap() as u32)
                                .collect();
                            Prediction::Class {
                                codes: r_codes,
                                levels: levels.clone(),
                                probabilities: Some(
                                    DesignMatrix::from_row_major(
                                        &flat_r,
                                        r_rows.len(),
                                        levels.len(),
                                        levels.clone(),
                                    )
                                    .unwrap(),
                                ),
                            }
                        }
                        _ => unreachable!(),
                    };
                    for m in metrics.evaluate(&truth, &r_pred).unwrap() {
                        let theirs = fx["resample"]["per_fold"][&label][&m.name].as_f64();
                        match (m.value, theirs) {
                            (Some(a), Some(b)) => report.compare(
                                id,
                                eng,
                                "L4",
                                &format!("{label} {} on R's predictions", m.name),
                                &[a],
                                &[b],
                                tol.pair_eng(id, "resample_metric", eng),
                            ),
                            (a, b) => report.exact(
                                id,
                                eng,
                                "L4",
                                &format!("{label} {} on R's predictions (NA)", m.name),
                                &a,
                                &b,
                            ),
                        }
                    }
                }
            }
            let res = wf
                .fit_resamples(&data, &folds, &metrics, Parallelism::threads(4))
                .unwrap();
            dump["per_fold"] = serde_json::json!(res
                .per_fold
                .iter()
                .map(|f| {
                    let mut m = serde_json::Map::new();
                    m.insert("id".into(), serde_json::json!(f.id));
                    for v in &f.metrics {
                        m.insert(v.name.clone(), serde_json::json!(v.value));
                    }
                    serde_json::Value::Object(m)
                })
                .collect::<Vec<_>>());
            let dump_dir = root().join("target/rust_outputs");
            std::fs::create_dir_all(&dump_dir).unwrap();
            std::fs::write(
                dump_dir.join(format!("{id}_{eng}.json")),
                serde_json::to_string(&dump).unwrap(),
            )
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
                            tol.pair_eng(id, "resample_metric", eng),
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
                    tol.pair_eng(id, "summary", eng),
                );
                report.compare(
                    id,
                    eng,
                    "L4",
                    &format!("summary std_err {}", s.name),
                    &[s.std_err.unwrap()],
                    &[theirs["std_err"].as_f64().unwrap()],
                    tol.pair_eng(id, "summary", eng),
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

    // ---- Level 2 (Phase 8b): class and probability metrics on fixed arrays ----
    fn opt_cmp(
        report: &mut Report,
        section: &str,
        name: &str,
        ours: Option<f64>,
        theirs: Option<f64>,
        b: (f64, f64),
    ) {
        match (ours, theirs) {
            (Some(a), Some(t)) => report.compare(section, "-", "L2", name, &[a], &[t], b),
            (a, t) => report.exact(section, "-", "L2", &format!("{name} (NA)"), &a, &t),
        }
    }
    let table_of = |cm: &ym::ConfusionMatrix| -> Vec<Vec<f64>> {
        let k = cm.n_levels();
        (0..k).map(|r| (0..k).map(|c| cm.count(r, c)).collect()).collect()
    };
    let json_table = |v: &Value| -> Vec<Vec<f64>> { items(v).into_iter().map(f64s).collect() };
    for section in ["binary_full", "binary_ties"] {
        let fs = &fx[section];
        let levels = strs(&fs["levels"]);
        let code = |v: &Value| -> Vec<u32> {
            strs(v)
                .iter()
                .map(|s| levels.iter().position(|l| l == s).unwrap() as u32)
                .collect()
        };
        let (t, e, p) = (code(&fs["truth"]), code(&fs["estimate"]), f64s(&fs["prob_event"]));
        let cm = ym::ConfusionMatrix::from_codes(&t, &e, &levels).unwrap();
        let bin = ym::Estimator::Binary;
        report.exact(
            section,
            "-",
            "L2",
            "conf_mat",
            &table_of(&cm),
            &json_table(&fs["conf_mat"]),
        );
        let pairs: Vec<(&str, Option<f64>)> = vec![
            ("sens", cm.sens(bin).unwrap()),
            ("spec", cm.spec(bin).unwrap()),
            ("precision", cm.precision(bin).unwrap()),
            ("recall", cm.recall(bin).unwrap()),
            ("f_meas", cm.f_meas(bin, 1.0).unwrap()),
            ("f_meas_beta2", cm.f_meas(bin, 2.0).unwrap()),
            ("bal_accuracy", cm.bal_accuracy(bin).unwrap()),
            ("j_index", cm.j_index(bin).unwrap()),
            ("kap", cm.kap(ym::KapWeighting::None)),
            ("mcc", cm.mcc()),
            ("accuracy", cm.accuracy()),
            ("roc_auc", ym::roc_auc(&t, &p, 0).unwrap()),
            ("pr_auc", ym::pr_auc(&t, &p, 0).unwrap()),
            ("average_precision", ym::average_precision(&t, &p, 0).unwrap()),
            ("brier_class", ym::brier_class(&t, &p, 0).unwrap()),
            ("mn_log_loss", ym::mn_log_loss(&t, &p, 0).unwrap()),
        ];
        for (name, ours) in pairs {
            opt_cmp(&mut report, section, name, ours, fs[name].as_f64(), b);
        }
        let rc = ym::roc_curve(&t, &p, 0).unwrap();
        let finite: Vec<f64> = rc.threshold.iter().copied().filter(|x| x.is_finite()).collect();
        report.compare(
            section,
            "-",
            "L2",
            "roc_curve threshold",
            &finite,
            &f64s(&fs["roc_curve"]["threshold"]),
            b,
        );
        report.compare(
            section,
            "-",
            "L2",
            "roc_curve specificity",
            &rc.specificity,
            &f64s(&fs["roc_curve"]["specificity"]),
            b,
        );
        report.compare(
            section,
            "-",
            "L2",
            "roc_curve sensitivity",
            &rc.sensitivity,
            &f64s(&fs["roc_curve"]["sensitivity"]),
            b,
        );
        let pc = ym::pr_curve(&t, &p, 0).unwrap();
        let finite: Vec<f64> = pc.threshold.iter().copied().filter(|x| x.is_finite()).collect();
        report.compare(
            section,
            "-",
            "L2",
            "pr_curve threshold",
            &finite,
            &f64s(&fs["pr_curve"]["threshold"]),
            b,
        );
        report.compare(
            section,
            "-",
            "L2",
            "pr_curve recall",
            &pc.recall,
            &f64s(&fs["pr_curve"]["recall"]),
            b,
        );
        report.compare(
            section,
            "-",
            "L2",
            "pr_curve precision",
            &pc.precision,
            &f64s(&fs["pr_curve"]["precision"]),
            b,
        );
    }
    for section in ["multiclass", "multiclass_absent_level"] {
        let fs = &fx[section];
        let levels = strs(&fs["levels"]);
        let k = levels.len();
        let code = |v: &Value| -> Vec<u32> {
            strs(v)
                .iter()
                .map(|s| levels.iter().position(|l| l == s).unwrap() as u32)
                .collect()
        };
        let (t, e) = (code(&fs["truth"]), code(&fs["estimate"]));
        let rows = json_table(&fs["probs"]);
        let flat: Vec<f64> = rows.iter().flatten().copied().collect();
        let probs = DesignMatrix::from_row_major(&flat, rows.len(), k, levels.clone()).unwrap();
        let cm = ym::ConfusionMatrix::from_codes(&t, &e, &levels).unwrap();
        report.exact(
            section,
            "-",
            "L2",
            "conf_mat",
            &table_of(&cm),
            &json_table(&fs["conf_mat"]),
        );
        for (en, est) in [
            ("macro", ym::Estimator::Macro),
            ("macro_weighted", ym::Estimator::MacroWeighted),
            ("micro", ym::Estimator::Micro),
        ] {
            let pairs: Vec<(&str, Option<f64>)> = vec![
                ("sens", cm.sens(est).unwrap()),
                ("spec", cm.spec(est).unwrap()),
                ("precision", cm.precision(est).unwrap()),
                ("recall", cm.recall(est).unwrap()),
                ("f_meas", cm.f_meas(est, 1.0).unwrap()),
                ("bal_accuracy", cm.bal_accuracy(est).unwrap()),
                ("j_index", cm.j_index(est).unwrap()),
            ];
            for (name, ours) in pairs {
                opt_cmp(
                    &mut report,
                    section,
                    &format!("{name} {en}"),
                    ours,
                    fs[name][en].as_f64(),
                    b,
                );
            }
            if est != ym::Estimator::Micro {
                opt_cmp(
                    &mut report,
                    section,
                    &format!("roc_auc {en}"),
                    ym::roc_auc_multiclass(&t, &probs, est).unwrap(),
                    fs["roc_auc"][en].as_f64(),
                    b,
                );
                opt_cmp(
                    &mut report,
                    section,
                    &format!("pr_auc {en}"),
                    ym::pr_auc_multiclass(&t, &probs, est).unwrap(),
                    fs["pr_auc"][en].as_f64(),
                    b,
                );
                opt_cmp(
                    &mut report,
                    section,
                    &format!("average_precision {en}"),
                    ym::average_precision_multiclass(&t, &probs, est).unwrap(),
                    fs["average_precision"][en].as_f64(),
                    b,
                );
            }
        }
        let pairs: Vec<(&str, Option<f64>)> = vec![
            ("kap", cm.kap(ym::KapWeighting::None)),
            ("kap_linear", cm.kap(ym::KapWeighting::Linear)),
            ("kap_quadratic", cm.kap(ym::KapWeighting::Quadratic)),
            ("mcc", cm.mcc()),
            ("accuracy", cm.accuracy()),
            ("brier_class", ym::brier_class_multiclass(&t, &probs).unwrap()),
            ("mn_log_loss", ym::mn_log_loss_multiclass(&t, &probs).unwrap()),
        ];
        for (name, ours) in pairs {
            opt_cmp(&mut report, section, name, ours, fs[name].as_f64(), b);
        }
        opt_cmp(
            &mut report,
            section,
            "roc_auc hand_till",
            ym::roc_auc_multiclass(&t, &probs, ym::Estimator::HandTill).unwrap(),
            fs["roc_auc"]["hand_till"].as_f64(),
            b,
        );
    }
    {
        let lv2: Vec<String> = vec!["a".into(), "b".into()];
        let lv3: Vec<String> = vec!["x".into(), "y".into(), "z".into()];
        let cmx = |t: &[u32], e: &[u32], lv: &[String]| ym::ConfusionMatrix::from_codes(t, e, lv).unwrap();
        let u = &fx["undefined"];
        let bin = ym::Estimator::Binary;
        let pairs: Vec<(&str, Option<f64>)> = vec![
            ("mcc_binary_degenerate", cmx(&[0, 1], &[0, 0], &lv2).mcc()),
            (
                "mcc_multiclass_degenerate",
                cmx(&[0, 1, 2], &[0, 0, 0], &lv3).mcc(),
            ),
            (
                "kap_no_chance_disagreement",
                cmx(&[0, 0], &[0, 0], &lv2).kap(ym::KapWeighting::None),
            ),
            ("roc_auc_no_events", ym::roc_auc(&[1, 1], &[0.2, 0.3], 0).unwrap()),
            ("pr_auc_no_events", ym::pr_auc(&[1, 1], &[0.2, 0.3], 0).unwrap()),
            ("sens_no_events", cmx(&[1, 1], &[0, 1], &lv2).sens(bin).unwrap()),
            (
                "f_meas_both_zero",
                cmx(&[0, 1], &[1, 0], &lv2).f_meas(bin, 1.0).unwrap(),
            ),
        ];
        for (name, ours) in pairs {
            opt_cmp(&mut report, "metrics_fixed", name, ours, u[name].as_f64(), b);
        }
    }

    report.write_and_assert();
}

/// Compare an rpart-port tree with `fit$frame` / `fit$splits` from R, node by node in
/// pre-order: structure and counts exactly, floating-point quantities to tolerance.
fn compare_tree(
    report: &mut Report,
    tol: &Tolerances,
    id: &str,
    eng: &str,
    fitted: &dyn FittedModel,
    fx: &Value,
) {
    let state = fitted.state().unwrap();
    let tree: tidyrustmodels::models::FittedDecisionTree = serde_json::from_value(state).unwrap();
    let nodes = tree.tree.root.preorder();
    let b = tol.pair_eng(id, "tree_frame", eng);
    let r_nodes: Vec<u64> = items(&fx["node"]).iter().map(|v| v.as_u64().unwrap()).collect();
    report.exact(
        id,
        eng,
        "L3",
        "tree node numbers",
        &nodes.iter().map(|n| u64::from(n.number)).collect::<Vec<_>>(),
        &r_nodes,
    );
    let r_vars = strs(&fx["var"]);
    let our_vars: Vec<String> = nodes
        .iter()
        .map(|n| {
            n.split
                .as_ref()
                .map(|s| tree.var_name(s.var).to_string())
                .unwrap_or_else(|| "<leaf>".into())
        })
        .collect();
    report.exact(id, eng, "L3", "tree split variables", &our_vars, &r_vars);
    report.exact(
        id,
        eng,
        "L3",
        "tree node sizes",
        &nodes.iter().map(|n| n.n as u64).collect::<Vec<_>>(),
        &items(&fx["n"])
            .iter()
            .map(|v| v.as_u64().unwrap())
            .collect::<Vec<_>>(),
    );
    report.compare(
        id,
        eng,
        "L3",
        "tree node weights",
        &nodes.iter().map(|n| n.sum_wt).collect::<Vec<_>>(),
        &f64s(&fx["wt"]),
        b,
    );
    report.compare(
        id,
        eng,
        "L3",
        "tree node deviance",
        &nodes.iter().map(|n| n.risk).collect::<Vec<_>>(),
        &f64s(&fx["dev"]),
        b,
    );
    report.compare(
        id,
        eng,
        "L3",
        "tree node complexity",
        &nodes.iter().map(|n| n.complexity).collect::<Vec<_>>(),
        &f64s(&fx["complexity"]),
        b,
    );
    let yval: Vec<f64> = nodes
        .iter()
        .map(|n| match &n.value {
            tidyrustmodels::models::native::rpart::NodeValue::Mean(m) => *m,
            tidyrustmodels::models::native::rpart::NodeValue::Class { class, .. } => *class as f64,
        })
        .collect();
    report.compare(id, eng, "L3", "tree node yval", &yval, &f64s(&fx["yval"]), b);
    let splits: Vec<&tidyrustmodels::models::native::rpart::Split> =
        nodes.iter().filter_map(|n| n.split.as_ref()).collect();
    let r_points: Vec<f64> = f64s(&fx["split_point"])
        .into_iter()
        .filter(|v| !v.is_nan())
        .collect();
    report.compare(
        id,
        eng,
        "L3",
        "tree split points",
        &splits.iter().map(|s| s.point).collect::<Vec<_>>(),
        &r_points,
        b,
    );
    let r_dirs: Vec<i64> = items(&fx["split_dir"])
        .iter()
        .filter_map(|v| v.as_i64())
        .collect();
    report.exact(
        id,
        eng,
        "L3",
        "tree split directions",
        &splits.iter().map(|s| i64::from(s.direction)).collect::<Vec<_>>(),
        &r_dirs,
    );
    let r_counts: Vec<u64> = items(&fx["split_count"])
        .iter()
        .filter_map(|v| v.as_u64())
        .collect();
    report.exact(
        id,
        eng,
        "L3",
        "tree split counts",
        &splits.iter().map(|s| s.count as u64).collect::<Vec<_>>(),
        &r_counts,
    );
    let r_improve: Vec<f64> = f64s(&fx["split_improve"])
        .into_iter()
        .filter(|v| !v.is_nan())
        .collect();
    report.compare(
        id,
        eng,
        "L3",
        "tree split improvements",
        &splits.iter().map(|s| s.improve).collect::<Vec<_>>(),
        &r_improve,
        b,
    );
    if !fx["counts"].is_null() {
        let r_counts: Vec<f64> = items(&fx["counts"]).into_iter().flat_map(f64s).collect();
        let ours: Vec<f64> = nodes
            .iter()
            .flat_map(|n| match &n.value {
                tidyrustmodels::models::native::rpart::NodeValue::Class { counts, .. } => counts.clone(),
                _ => unreachable!(),
            })
            .collect();
        report.compare(id, eng, "L3", "tree class counts", &ours, &r_counts, b);
    }
}

/// Level 5: statistical comparison of a stochastic model. R fitted `S` seeds (fixture
/// `fit.rf.per_seed`); we fit `S` seeds of our own and compare the seed means of every
/// metric and the seed-averaged test predictions within `rf_z` standard errors.
#[allow(clippy::too_many_arguments)]
fn compare_forest(
    report: &mut Report,
    tol: &Tolerances,
    id: &str,
    eng: &str,
    task: &Value,
    x_train: &DesignMatrix,
    y_train: &Outcome,
    x_test: &DesignMatrix,
    y_test: &Outcome,
    metrics: &MetricSet,
    fx: &Value,
) {
    let rf = &fx["fit"]["rf"];
    let r_seeds = items(&rf["per_seed"]);
    let s_n = r_seeds.len();
    let z = tol.get("-", "rf_z");
    let row_fraction = tol.get("-", "rf_row_fraction");
    // resolved defaults must agree with ranger's
    let probe = model_for_seed(task, Engine::Native, y_train, 0)
        .fit(x_train, y_train)
        .unwrap();
    let note = probe.diagnostics().note.unwrap();
    let our_mtry: u64 = note
        .split("mtry ")
        .nth(1)
        .unwrap()
        .split(',')
        .next()
        .unwrap()
        .parse()
        .unwrap();
    let our_min: u64 = note
        .split("min.node.size ")
        .nth(1)
        .unwrap()
        .trim()
        .parse()
        .unwrap();
    report.exact(
        id,
        eng,
        "L5",
        "ranger mtry default",
        &our_mtry,
        &rf["mtry"].as_u64().unwrap(),
    );
    report.exact(
        id,
        eng,
        "L5",
        "ranger min.node.size default",
        &our_min,
        &rf["min_node_size"].as_u64().unwrap(),
    );
    report.exact(
        id,
        eng,
        "L5",
        "ranger tree type",
        &matches!(y_train, Outcome::Class { .. }),
        &(rf["treetype"].as_str().unwrap() == "Probability estimation"),
    );
    // our S seeds
    let mut ours: Vec<(Vec<MetricValue>, Vec<f64>)> = Vec::with_capacity(s_n);
    for s in 0..s_n as u64 {
        let fitted = model_for_seed(task, Engine::Native, y_train, s)
            .fit(x_train, y_train)
            .unwrap();
        let pred = fitted.predict(x_test).unwrap();
        let flat = match &pred {
            Prediction::Numeric(p) => p.clone(),
            Prediction::Class {
                probabilities: Some(p),
                ..
            } => p.to_row_major(),
            _ => unreachable!(),
        };
        ours.push((metrics.evaluate(y_test, &pred).unwrap(), flat));
    }
    // per-seed dump for diagnosis (target/rust_outputs/<task>_rf_seeds.json)
    {
        let dump: Vec<Value> = ours
            .iter()
            .map(|(mv, flat)| {
                let m: serde_json::Map<String, Value> = metrics
                    .metrics()
                    .iter()
                    .zip(mv)
                    .map(|(m, v)| (m.name().to_string(), serde_json::json!(v.value)))
                    .collect();
                serde_json::json!({ "metrics": m, "pred": flat })
            })
            .collect();
        let dir = root().join("target/rust_outputs");
        std::fs::create_dir_all(&dir).unwrap();
        std::fs::write(
            dir.join(format!("{id}_rf_seeds.json")),
            serde_json::to_string(&dump).unwrap(),
        )
        .unwrap();
    }
    let mean = |v: &[f64]| v.iter().sum::<f64>() / v.len() as f64;
    let var = |v: &[f64]| {
        let m = mean(v);
        v.iter().map(|x| (x - m) * (x - m)).sum::<f64>() / (v.len() as f64 - 1.0)
    };
    for (mi, m) in metrics.metrics().iter().enumerate() {
        let r: Vec<f64> = r_seeds
            .iter()
            .map(|s| s["metrics"][m.name()].as_f64().unwrap())
            .collect();
        let o: Vec<f64> = ours.iter().map(|(mv, _)| mv[mi].value.unwrap()).collect();
        let se = (var(&r) / s_n as f64 + var(&o) / s_n as f64).sqrt();
        report.compare(
            id,
            eng,
            "L5",
            &format!(
                "seed-mean {} (band {z} SE = {:.2e}; R sd {:.2e})",
                m.name(),
                z * se,
                var(&r).sqrt()
            ),
            &[mean(&o)],
            &[mean(&r)],
            (0.0, z * se),
        );
    }
    // seed-averaged predictions, row-wise bands
    let key = if matches!(y_train, Outcome::Numeric(_)) {
        "predicted_test"
    } else {
        "prob_test"
    };
    let r_rows: Vec<Vec<f64>> = r_seeds
        .iter()
        .map(|s| {
            if key == "prob_test" {
                items(&s[key]).into_iter().flat_map(f64s).collect()
            } else {
                f64s(&s[key])
            }
        })
        .collect();
    let width = r_rows[0].len();
    let mut within = 0usize;
    let mut max_std = 0.0f64;
    for j in 0..width {
        let r: Vec<f64> = r_rows.iter().map(|v| v[j]).collect();
        let o: Vec<f64> = ours.iter().map(|(_, v)| v[j]).collect();
        let se = (var(&r) / s_n as f64 + var(&o) / s_n as f64).sqrt();
        let d = (mean(&r) - mean(&o)).abs();
        if se > 0.0 {
            max_std = max_std.max(d / se);
        }
        if d <= z * se || (se == 0.0 && d == 0.0) {
            within += 1;
        }
    }
    let frac = within as f64 / width as f64;
    report.exact(
        id,
        eng,
        "L5",
        &format!("seed-averaged test predictions within {z} SE on >= {row_fraction} of {width} values (observed {frac:.3}, max {max_std:.2} SE)"),
        &true,
        &(frac >= row_fraction),
    );
}

/// Per-fold tree check (L4): structure exact, split points / improvements / deviances
/// to the tree tolerance.
fn compare_tree_structure(
    report: &mut Report,
    tol: &Tolerances,
    id: &str,
    eng: &str,
    label: &str,
    fitted: &dyn FittedModel,
    fx: &Value,
) {
    let state = fitted.state().unwrap();
    let tree: tidyrustmodels::models::FittedDecisionTree = serde_json::from_value(state).unwrap();
    let nodes = tree.tree.root.preorder();
    let b = tol.pair_eng(id, "tree_frame", eng);
    let r_nodes: Vec<u64> = items(&fx["node"]).iter().map(|v| v.as_u64().unwrap()).collect();
    report.exact(
        id,
        eng,
        "L4",
        &format!("{label} tree node numbers"),
        &nodes.iter().map(|n| u64::from(n.number)).collect::<Vec<_>>(),
        &r_nodes,
    );
    let our_vars: Vec<String> = nodes
        .iter()
        .map(|n| {
            n.split
                .as_ref()
                .map(|s| tree.var_name(s.var).to_string())
                .unwrap_or_else(|| "<leaf>".into())
        })
        .collect();
    report.exact(
        id,
        eng,
        "L4",
        &format!("{label} tree split variables"),
        &our_vars,
        &strs(&fx["var"]),
    );
    report.exact(
        id,
        eng,
        "L4",
        &format!("{label} tree node sizes"),
        &nodes.iter().map(|n| n.n as u64).collect::<Vec<_>>(),
        &items(&fx["n"])
            .iter()
            .map(|v| v.as_u64().unwrap())
            .collect::<Vec<_>>(),
    );
    report.compare(
        id,
        eng,
        "L4",
        &format!("{label} tree node deviance"),
        &nodes.iter().map(|n| n.risk).collect::<Vec<_>>(),
        &f64s(&fx["dev"]),
        b,
    );
    let splits: Vec<&tidyrustmodels::models::native::rpart::Split> =
        nodes.iter().filter_map(|n| n.split.as_ref()).collect();
    let r_points: Vec<f64> = f64s(&fx["split_point"])
        .into_iter()
        .filter(|v| !v.is_nan())
        .collect();
    report.compare(
        id,
        eng,
        "L4",
        &format!("{label} tree split points"),
        &splits.iter().map(|s| s.point).collect::<Vec<_>>(),
        &r_points,
        b,
    );
    let r_improve: Vec<f64> = f64s(&fx["split_improve"])
        .into_iter()
        .filter(|v| !v.is_nan())
        .collect();
    report.compare(
        id,
        eng,
        "L4",
        &format!("{label} tree split improvements"),
        &splits.iter().map(|s| s.improve).collect::<Vec<_>>(),
        &r_improve,
        b,
    );
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
