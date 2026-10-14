//! One benchmark trial of tidyrustmodels in a fresh process.
//!
//! Prints a single line `BENCH_RESULT {json}` on stdout. The Python orchestrator records
//! the process wall-clock time and peak memory externally; this binary reports per-phase
//! timings measured with `Instant` and allocation counts from a counting global
//! allocator (every `alloc` / `realloc` call and its bytes; frees are not counted).
//!
//! Usage (shared CLI contract with benchmarks/r/bench_task.R and benchmarks/python/bench_task.py):
//!   trm-bench --dataset <csv> --outcome <col> --type regression|classification
//!             --predictors a,b,c --steps normalize_numeric,dummy_nominal --metrics rmse,rsq
//!             --engine native|linfa|smartcore --threads <k>
//!             [--model linear_reg|logistic_reg|decision_tree|rand_forest|boost_tree]
//!             [--model-args trees=100,min_n=5] [--case-weights col:frequency|importance]
//!             [--resampling vfold|repeated|bootstraps] [--v 10] [--repeats 3] [--times 25]
//!             [--tune none|grid] [--tune-levels 20] [--select] [--wfs] [--warm <n>] [--seed 1]
//!
//! Phases: load, prep, bake, fit (model only, baked design), wf_fit (prep + bake + fit
//! from the raw training table), predict, cv (resampling over the chosen scheme), tune
//! (grid x resamples), select (select_best + finalize + last_fit on the initial split),
//! wfs (a three-workflow set: base model, tuned tree, boosted tree; map + rank + fit_best).

use std::alloc::{GlobalAlloc, Layout, System};
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::Instant;
use tidyrustmodels::io::read_csv;
use tidyrustmodels::prelude::*;

// ---- allocation counting (A1) ----
struct Counting;
static ALLOC_COUNT: AtomicU64 = AtomicU64::new(0);
static ALLOC_BYTES: AtomicU64 = AtomicU64::new(0);

unsafe impl GlobalAlloc for Counting {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        ALLOC_COUNT.fetch_add(1, Ordering::Relaxed);
        ALLOC_BYTES.fetch_add(layout.size() as u64, Ordering::Relaxed);
        System.alloc(layout)
    }
    unsafe fn alloc_zeroed(&self, layout: Layout) -> *mut u8 {
        // Forwarded explicitly: the trait's default is `alloc` + a manual memset, which
        // would charge the Rust driver for zeroing that the system allocator gets from
        // fresh pages for free (runs G..J-pre measured that cost; see BENCHMARKS.md).
        ALLOC_COUNT.fetch_add(1, Ordering::Relaxed);
        ALLOC_BYTES.fetch_add(layout.size() as u64, Ordering::Relaxed);
        System.alloc_zeroed(layout)
    }
    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        System.dealloc(ptr, layout)
    }
    unsafe fn realloc(&self, ptr: *mut u8, layout: Layout, new_size: usize) -> *mut u8 {
        ALLOC_COUNT.fetch_add(1, Ordering::Relaxed);
        ALLOC_BYTES.fetch_add(new_size as u64, Ordering::Relaxed);
        System.realloc(ptr, layout, new_size)
    }
}

#[global_allocator]
static GLOBAL: Counting = Counting;

#[derive(Clone, Copy)]
struct AllocMark(u64, u64);

fn mark() -> AllocMark {
    AllocMark(
        ALLOC_COUNT.load(Ordering::Relaxed),
        ALLOC_BYTES.load(Ordering::Relaxed),
    )
}

fn since(m: AllocMark) -> (u64, u64) {
    let now = mark();
    (now.0 - m.0, now.1 - m.1)
}

fn arg(args: &[String], name: &str) -> Option<String> {
    args.iter()
        .position(|a| a == name)
        .and_then(|i| args.get(i + 1).cloned())
}

fn flag(args: &[String], name: &str) -> bool {
    args.iter().any(|a| a == name)
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
            "zv" => rec.step(StepZv::new(Selector::AllPredictors)),
            "" => rec,
            other => return Err(Error::InvalidArgument(format!("unknown step {other}"))),
        };
    }
    Ok(rec)
}

fn model_args(spec: &str) -> Vec<(String, f64)> {
    spec.split(',')
        .filter(|s| !s.is_empty())
        .filter_map(|kv| {
            let (k, v) = kv.split_once('=')?;
            Some((k.to_string(), v.parse::<f64>().ok()?))
        })
        .collect()
}

fn build_model(model: &str, args: &[(String, f64)], engine: Engine, seed: u64) -> Result<Box<dyn ModelSpec>> {
    let get = |k: &str| args.iter().find(|(n, _)| n == k).map(|(_, v)| *v);
    Ok(match model {
        "linear_reg" => Box::new(LinearRegSpec::new().engine(engine)),
        "logistic_reg" => Box::new(LogisticRegSpec::new().engine(engine)),
        "decision_tree" => {
            let mut s = DecisionTreeSpec::new().engine(engine);
            if let Some(v) = get("cost_complexity") {
                s = s.cost_complexity(v);
            }
            if let Some(v) = get("tree_depth") {
                s = s.tree_depth(v as usize);
            }
            if let Some(v) = get("min_n") {
                s = s.min_n(v as usize);
            }
            Box::new(s)
        }
        "rand_forest" => {
            let mut s = RandForestSpec::new().engine(engine).seed(Seed(seed));
            if let Some(v) = get("trees") {
                s = s.trees(v as usize);
            }
            if let Some(v) = get("mtry") {
                s = s.mtry(v as usize);
            }
            if let Some(v) = get("min_n") {
                s = s.min_n(v as usize);
            }
            Box::new(s)
        }
        "boost_tree" => {
            let mut s = BoostTreeSpec::new().engine(engine);
            if let Some(v) = get("trees") {
                s = s.trees(v as usize);
            }
            if let Some(v) = get("tree_depth") {
                s = s.tree_depth(v as usize);
            }
            if let Some(v) = get("learn_rate") {
                s = s.learn_rate(v);
            }
            if let Some(v) = get("min_n") {
                s = s.min_n(v);
            }
            Box::new(s)
        }
        other => return Err(Error::InvalidArgument(format!("unknown model {other}"))),
    })
}

/// The tuning grid of a model: the same explicit values as the R driver.
/// `levels` = points of the 1-D penalty grid; the 2-D tree / boosting grids have
/// 4 x 5 = 20 points regardless (explicit values, so dials' rounding plays no part).
fn tune_grid_for(model: &str, levels: usize) -> Result<Option<Grid>> {
    Ok(match model {
        "logistic_reg" => Some(grid_regular(&[ParamRange::log10("penalty", -4.0, 0.0)], levels)?),
        "decision_tree" => {
            let mut points = Vec::new();
            for cp in [1e-4, 1e-3, 1e-2, 1e-1] {
                for mn in [2.0, 5.0, 10.0, 15.0, 20.0] {
                    points.push(vec![
                        ("cost_complexity".to_string(), cp),
                        ("min_n".to_string(), mn),
                    ]);
                }
            }
            Some(Grid { points })
        }
        "boost_tree" => {
            let mut points = Vec::new();
            for depth in [2.0, 4.0, 6.0, 8.0] {
                for lr in [0.05, 0.1, 0.2, 0.3, 0.5] {
                    points.push(vec![
                        ("tree_depth".to_string(), depth),
                        ("learn_rate".to_string(), lr),
                    ]);
                }
            }
            Some(Grid { points })
        }
        _ => None,
    })
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
    let threads: usize = need("--threads")?
        .parse()
        .map_err(|_| Error::InvalidArgument("--threads".into()))?;
    let model_name = arg(&args, "--model").unwrap_or_else(|| {
        if task_type == "regression" {
            "linear_reg".into()
        } else {
            "logistic_reg".into()
        }
    });
    let margs = model_args(&arg(&args, "--model-args").unwrap_or_default());
    let case_weights: Option<(String, WeightKind)> = match arg(&args, "--case-weights") {
        Some(s) => {
            let (col, kind) = s.split_once(':').unwrap_or((s.as_str(), "frequency"));
            let kind = if kind == "importance" {
                WeightKind::Importance
            } else {
                WeightKind::Frequency
            };
            Some((col.to_string(), kind))
        }
        None => None,
    };
    let resampling = arg(&args, "--resampling").unwrap_or_else(|| "vfold".into());
    let warm: usize = arg(&args, "--warm").map(|s| s.parse().unwrap_or(0)).unwrap_or(0);
    let v: usize = arg(&args, "--v").map(|s| s.parse().unwrap_or(5)).unwrap_or(5);
    let repeats: usize = arg(&args, "--repeats")
        .map(|s| s.parse().unwrap_or(1))
        .unwrap_or(1);
    let times: usize = arg(&args, "--times")
        .map(|s| s.parse().unwrap_or(25))
        .unwrap_or(25);
    let tune_mode = arg(&args, "--tune").unwrap_or_else(|| "grid".into());
    let tune_levels: usize = arg(&args, "--tune-levels")
        .map(|s| s.parse().unwrap_or(5))
        .unwrap_or(5);
    let do_select = flag(&args, "--select");
    let do_wfs = flag(&args, "--wfs");
    let seed: u64 = arg(&args, "--seed").map(|s| s.parse().unwrap_or(1)).unwrap_or(1);
    let par = if threads <= 1 {
        Parallelism::SERIAL
    } else {
        Parallelism::threads(threads)
    };

    // ---- load (CSV -> Table, select task columns) ----
    let t0 = Instant::now();
    let m0 = mark();
    let full = read_csv(&dataset)?;
    let mut cols = vec![(
        outcome.clone(),
        full.column(&outcome)
            .cloned()
            .ok_or_else(|| Error::SchemaMismatch(outcome.clone()))?,
    )];
    for p in &predictors {
        cols.push((
            p.clone(),
            full.column(p)
                .cloned()
                .ok_or_else(|| Error::SchemaMismatch(p.clone()))?,
        ));
    }
    if let Some((w, _)) = &case_weights {
        cols.push((
            w.clone(),
            full.column(w)
                .cloned()
                .ok_or_else(|| Error::SchemaMismatch(w.clone()))?,
        ));
    }
    let data = Table::new(cols)?;
    drop(full);
    let load_s = secs(t0);
    let load_alloc = since(m0);
    let n = data.nrows();

    // ---- split (not timed as a phase; both systems use their own RNG) ----
    let strata: Option<Vec<u32>> = if task_type == "classification" {
        data.column(&outcome)
            .and_then(|c| c.as_categorical())
            .map(|(codes, _)| codes.to_vec())
    } else {
        None
    };
    let split = initial_split(n, 0.75, strata.as_deref(), Seed(seed))?;
    let train = data.select_rows(&split.train)?;
    let test = data.select_rows(&split.test)?;

    // ---- prep + bake ----
    let recipe = build_recipe(&outcome, &steps)?;
    let cw_spec = case_weights.as_ref().map(|(c, k)| (c.as_str(), *k));
    let t0 = Instant::now();
    let prepped = recipe.prep_with_weights(&train, cw_spec)?;
    let prep_s = secs(t0);
    let t0 = Instant::now();
    let (x_train, y_train) = prepped.bake_matrix(&train)?;
    let (x_test, _) = prepped.bake_matrix(&test)?;
    let bake_s = secs(t0);
    let y_train = y_train.ok_or_else(|| Error::SchemaMismatch("no outcome".into()))?;

    // ---- fit + predict (model only, baked design) ----
    let spec = build_model(&model_name, &margs, engine, seed)?;
    let t0 = Instant::now();
    let fitted = spec.fit(&x_train, &y_train)?;
    let fit_s = secs(t0);
    let t0 = Instant::now();
    let pred = fitted.predict(&x_test)?;
    let predict_s = secs(t0);
    std::hint::black_box(&pred);

    // ---- workflow fit from the raw training table (prep + bake + fit), like R's fit(wf) ----
    let mut wf = Workflow::from_boxed(recipe.clone(), spec.clone_box());
    if let Some((c, k)) = &case_weights {
        wf = wf.case_weights(c.clone(), *k);
    }
    let m0 = mark();
    let t0 = Instant::now();
    let wf_fitted = wf.fit(&train)?;
    let wf_fit_s = secs(t0);
    let wf_fit_alloc = since(m0);
    std::hint::black_box(&wf_fitted);

    // ---- resampling ----
    let set = MetricSet::from_names(&metrics)?;
    let folds = match resampling.as_str() {
        "vfold" => vfold_cv(n, v, strata.as_deref(), Seed(seed + 1))?,
        "repeated" => vfold_cv_repeated(n, v, repeats, strata.as_deref(), Seed(seed + 1))?,
        "bootstraps" => bootstraps(n, times, strata.as_deref(), Seed(seed + 1))?,
        other => return Err(Error::InvalidArgument(format!("unknown resampling {other}"))),
    };
    let m0 = mark();
    let t0 = Instant::now();
    let cv = wf.fit_resamples(&data, &folds, &set, par)?;
    let cv_s = secs(t0);
    let cv_alloc = since(m0);
    let cv_summary: Vec<serde_json::Value> = cv
        .summary
        .iter()
        .map(|s| serde_json::json!({"metric": s.name, "mean": s.mean, "std_err": s.std_err}))
        .collect();

    // ---- tuning + selection ----
    let mut tune_s = None;
    let mut tune_candidates = None;
    let mut select_s = None;
    let mut select_params = None;
    if tune_mode == "grid" {
        if let Some(grid) = tune_grid_for(&model_name, tune_levels)? {
            let t0 = Instant::now();
            let res = tune_grid(&wf, &data, &folds, &grid, &set, par)?;
            tune_s = Some(secs(t0));
            tune_candidates = Some(res.candidates.len());
            if do_select {
                let metric = set.metrics()[0];
                let t0 = Instant::now();
                let sel = res.select_best(metric)?;
                let lf = wf.finalize(&sel.params)?.last_fit(&data, &split, &set)?;
                select_s = Some(secs(t0));
                std::hint::black_box(&lf);
                select_params = Some(serde_json::json!(sel.params));
            }
        }
    }

    // ---- workflow set: base model, a tuned tree (3 cost_complexity values), xgboost ----
    let mut wfs_s = None;
    let mut wfs_best = None;
    if do_wfs {
        let mut set_ws = workflow_set(
            vec![(Some("rec".into()), recipe.clone())],
            vec![
                (Some("base".into()), spec.clone_box()),
                (
                    Some("tree".into()),
                    Box::new(DecisionTreeSpec::new().engine(engine)),
                ),
                (
                    Some("xgb".into()),
                    Box::new(BoostTreeSpec::new().engine(engine).trees(20)),
                ),
            ],
            true,
        )?;
        set_ws.option_grid(
            "rec_tree",
            Grid {
                points: [1e-3, 1e-2, 1e-1]
                    .iter()
                    .map(|cp| vec![("cost_complexity".to_string(), *cp)])
                    .collect(),
            },
        )?;
        let t0 = Instant::now();
        set_ws.workflow_map(&data, &folds, &set, par)?;
        let ranked = set_ws.rank_results(None, true)?;
        let best = set_ws.fit_best(&data, None)?;
        wfs_s = Some(secs(t0));
        std::hint::black_box(&best);
        wfs_best = ranked.first().map(|r| r.wflow_id.clone());
    }

    // ---- warm repetitions (steady state, same process) ----
    let (mut w_prep, mut w_fit, mut w_wffit, mut w_predict) =
        (Vec::new(), Vec::new(), Vec::new(), Vec::new());
    for _ in 0..warm {
        let t0 = Instant::now();
        let wff = wf.fit(&train)?;
        w_wffit.push(secs(t0));
        std::hint::black_box(&wff);
        let t0 = Instant::now();
        let p = recipe.prep_with_weights(&train, cw_spec)?;
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
        "model": model_name,
        "threads": threads,
        "n_rows": n,
        "n_train": train.nrows(),
        "n_features": x_train.ncols(),
        "n_resamples": folds.len(),
        "phases": {
            "load_s": load_s, "prep_s": prep_s, "bake_s": bake_s, "fit_s": fit_s, "wf_fit_s": wf_fit_s,
            "predict_s": predict_s, "cv_s": cv_s, "tune_s": tune_s, "select_s": select_s, "wfs_s": wfs_s,
        },
        "allocs": {
            "load": {"count": load_alloc.0, "bytes": load_alloc.1},
            "wf_fit": {"count": wf_fit_alloc.0, "bytes": wf_fit_alloc.1},
            "cv": {"count": cv_alloc.0, "bytes": cv_alloc.1},
        },
        "cv_summary": cv_summary,
        "tune_candidates": tune_candidates,
        "select_params": select_params,
        "wfs_best": wfs_best,
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
