//! One benchmark trial of tidyrustmodels in a fresh process.
//!
//! Prints a single line `BENCH_RESULT {json}` on stdout. The Python driver records the
//! process wall-clock time and peak RSS externally; this binary only reports per-phase
//! timings measured with `Instant`.
//!
//! Usage (all flags required unless noted):
//!   trm-bench --dataset <csv> --outcome <col> --type regression|classification
//!             --predictors a,b,c --steps normalize_numeric,dummy_nominal
//!             --metrics rmse,rsq --engine linfa|smartcore --threads <k>
//!             [--warm <n>] [--v 5] [--tune-levels 5] [--seed 1]

use std::time::Instant;
use tidyrustmodels::io::read_csv;
use tidyrustmodels::prelude::*;

fn arg(args: &[String], name: &str) -> Option<String> {
    args.iter().position(|a| a == name).and_then(|i| args.get(i + 1).cloned())
}

fn secs(t: Instant) -> f64 {
    t.elapsed().as_secs_f64()
}

fn build_recipe(outcome: &str, steps: &[String]) -> Result<Recipe> {
    let mut rec = Recipe::new(outcome);
    for s in steps {
        rec = match s.as_str() {
            "impute_median_numeric" => rec.step(StepImputeMedian::new(Selector::AllNumericPredictors)),
            "impute_mean_numeric" => rec.step(StepImputeMean::new(Selector::AllNumericPredictors)),
            "normalize_numeric" => rec.step(StepNormalize::new(Selector::AllNumericPredictors)),
            "dummy_nominal" => rec.step(StepDummy::new(Selector::AllNominalPredictors)),
            "" => rec,
            other => return Err(Error::InvalidArgument(format!("unknown step {other}"))),
        };
    }
    Ok(rec)
}

fn median(v: &mut [f64]) -> f64 {
    v.sort_by(|a, b| a.total_cmp(b));
    let n = v.len();
    if n == 0 {
        f64::NAN
    } else if n % 2 == 1 {
        v[n / 2]
    } else {
        (v[n / 2 - 1] + v[n / 2]) / 2.0
    }
}

fn run() -> Result<()> {
    let args: Vec<String> = std::env::args().collect();
    let need = |n: &str| arg(&args, n).ok_or_else(|| Error::InvalidArgument(format!("missing {n}")));
    let dataset = need("--dataset")?;
    let outcome = need("--outcome")?;
    let task_type = need("--type")?;
    let predictors: Vec<String> = need("--predictors")?.split(',').map(String::from).collect();
    let steps: Vec<String> = need("--steps")?.split(',').map(String::from).collect();
    let metrics: Vec<String> = need("--metrics")?.split(',').map(String::from).collect();
    let engine = match need("--engine")?.as_str() {
        "linfa" => Engine::Linfa,
        "smartcore" => Engine::Smartcore,
        "native" => Engine::Native,
        e => return Err(Error::InvalidArgument(format!("unknown engine {e}"))),
    };
    let threads: usize = need("--threads")?.parse().map_err(|_| Error::InvalidArgument("--threads".into()))?;
    let warm: usize = arg(&args, "--warm").map(|s| s.parse().unwrap_or(0)).unwrap_or(0);
    let v: usize = arg(&args, "--v").map(|s| s.parse().unwrap_or(5)).unwrap_or(5);
    let tune_levels: usize = arg(&args, "--tune-levels").map(|s| s.parse().unwrap_or(5)).unwrap_or(5);
    let seed: u64 = arg(&args, "--seed").map(|s| s.parse().unwrap_or(1)).unwrap_or(1);
    let par = if threads <= 1 { Parallelism::SERIAL } else { Parallelism::threads(threads) };

    // ---- load (CSV -> Table, select task columns) ----
    let t0 = Instant::now();
    let full = read_csv(&dataset)?;
    let mut cols = vec![(outcome.clone(), full.column(&outcome).cloned().ok_or_else(|| Error::SchemaMismatch(outcome.clone()))?)];
    for p in &predictors {
        cols.push((p.clone(), full.column(p).cloned().ok_or_else(|| Error::SchemaMismatch(p.clone()))?));
    }
    let data = Table::new(cols)?;
    let load_s = secs(t0);
    let n = data.nrows();

    // ---- split (not timed as a phase; both systems use their own RNG) ----
    let strata: Option<Vec<u32>> = if task_type == "classification" {
        data.column(&outcome).and_then(|c| c.as_categorical()).map(|(codes, _)| codes.to_vec())
    } else {
        None
    };
    let split = initial_split(n, 0.75, strata.as_deref(), Seed(seed))?;
    let train = data.select_rows(&split.train)?;
    let test = data.select_rows(&split.test)?;

    // ---- prep + bake ----
    let recipe = build_recipe(&outcome, &steps)?;
    let t0 = Instant::now();
    let prepped = recipe.prep(&train)?;
    let prep_s = secs(t0);
    let t0 = Instant::now();
    let (x_train, y_train) = prepped.bake_matrix(&train)?;
    let (x_test, _) = prepped.bake_matrix(&test)?;
    let bake_s = secs(t0);
    let y_train = y_train.ok_or_else(|| Error::SchemaMismatch("no outcome".into()))?;

    // ---- fit + predict ----
    let spec: Box<dyn ModelSpec> = if task_type == "regression" {
        Box::new(LinearRegSpec::new().engine(engine))
    } else {
        Box::new(LogisticRegSpec::new().engine(engine))
    };
    let t0 = Instant::now();
    let fitted = spec.fit(&x_train, &y_train)?;
    let fit_s = secs(t0);
    let t0 = Instant::now();
    let pred = fitted.predict(&x_test)?;
    let predict_s = secs(t0);
    std::hint::black_box(&pred);

    // ---- workflow fit from the raw training table (prep + bake + fit), like R's fit(wf) ----
    let wf = Workflow::from_boxed(recipe.clone(), spec.clone_box());
    let t0 = Instant::now();
    let wf_fitted = wf.fit(&train)?;
    let wf_fit_s = secs(t0);
    std::hint::black_box(&wf_fitted);

    // ---- cross-validation ----
    let set = MetricSet::from_names(&metrics)?;
    let folds = vfold_cv(n, v, strata.as_deref(), Seed(seed + 1))?;
    let t0 = Instant::now();
    let cv = wf.fit_resamples(&data, &folds, &set, par)?;
    let cv_s = secs(t0);
    let cv_summary: Vec<serde_json::Value> = cv
        .summary
        .iter()
        .map(|s| serde_json::json!({"metric": s.name, "mean": s.mean, "std_err": s.std_err}))
        .collect();

    // ---- tuning (classification only: penalty is the tunable parameter) ----
    let (tune_s, tune_candidates) = if task_type == "classification" {
        let grid = grid_regular(&[ParamRange::log10("penalty", -4.0, 0.0)], tune_levels)?;
        let t0 = Instant::now();
        let res = tune_grid(&wf, &data, &folds, &grid, &set, par)?;
        (Some(secs(t0)), Some(res.candidates.len()))
    } else {
        (None, None)
    };

    // ---- warm repetitions (steady state, same process) ----
    let (mut w_prep, mut w_fit, mut w_wffit, mut w_predict) = (Vec::new(), Vec::new(), Vec::new(), Vec::new());
    for _ in 0..warm {
        let t0 = Instant::now();
        let wff = wf.fit(&train)?;
        w_wffit.push(secs(t0));
        std::hint::black_box(&wff);
        let t0 = Instant::now();
        let p = recipe.prep(&train)?;
        let (xt, yt) = p.bake_matrix(&train)?;
        w_prep.push(secs(t0));
        let yt = yt.expect("outcome present");
        let t0 = Instant::now();
        let f = spec.fit(&xt, &yt)?;
        w_fit.push(secs(t0));
        let t0 = Instant::now();
        let pr = f.predict(&x_test)?;
        w_predict.push(secs(t0));
        std::hint::black_box(&pr);
    }

    let out = serde_json::json!({
        "system": "rust",
        "engine": engine.as_str(),
        "threads": threads,
        "n_rows": n,
        "n_train": train.nrows(),
        "n_features": x_train.ncols(),
        "phases": {
            "load_s": load_s, "prep_s": prep_s, "bake_s": bake_s, "fit_s": fit_s, "wf_fit_s": wf_fit_s,
            "predict_s": predict_s, "cv_s": cv_s, "tune_s": tune_s,
        },
        "cv_summary": cv_summary,
        "tune_candidates": tune_candidates,
        "warm": {
            "iters": warm,
            "prep_bake_s_median": median(&mut w_prep),
            "fit_s_median": median(&mut w_fit),
            "wf_fit_s_median": median(&mut w_wffit),
            "predict_s_median": median(&mut w_predict),
        },
        "versions": { "tidyrustmodels": env!("CARGO_PKG_VERSION"), "profile": "release, lto=fat, codegen-units=1" },
    });
    println!("BENCH_RESULT {}", serde_json::to_string(&out).expect("json"));
    Ok(())
}

fn main() {
    if let Err(e) = run() {
        eprintln!("trm-bench error: {e}");
        std::process::exit(1);
    }
}
