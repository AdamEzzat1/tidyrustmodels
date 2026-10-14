//! Tuning a decision tree on the Ames housing data with tune's selection rules and the
//! `last_fit` / `fit_best` finishing steps (ADR-0014).
//!
//! Run from the repository root:
//!     cargo run --release --example ames_tuning

use tidyrustmodels::io::read_csv;
use tidyrustmodels::prelude::*;

fn main() -> Result<()> {
    // --- data: five predictors of the sale price (modeldata::ames export, see reference/) ---
    let full = read_csv("reference/fixtures/datasets/ames.csv")?;
    let keep = [
        "Sale_Price",
        "Gr_Liv_Area",
        "Year_Built",
        "Total_Bsmt_SF",
        "Bldg_Type",
        "Neighborhood",
    ];
    let data = Table::new(
        keep.iter()
            .map(|n| (n.to_string(), full.column(n).unwrap().clone()))
            .collect(),
    )?;
    println!("{} rows x {} columns", data.nrows(), data.ncols());

    // --- initial split and 5-fold CV repeated twice (rsample semantics, ADR-0013) ---
    let split = initial_split(data.nrows(), 0.75, None, Seed(2024))?;
    let folds = vfold_cv_repeated(data.nrows(), 5, 2, None, Seed(11))?;
    println!("resamples: {:?} ...", &folds.ids[..3]);

    // --- recipe + tree model; the tuned parameters are set by the grid ---
    let recipe = Recipe::new("Sale_Price")
        .step(StepOther::new(Selector::AllNominalPredictors).threshold(0.02))
        .step(StepDummy::new(Selector::AllNominalPredictors));
    let wf = Workflow::new(recipe, DecisionTreeSpec::new());

    // --- an explicit grid: cost_complexity x min_n (dials' grid_regular would do the same) ---
    let mut points = Vec::new();
    for cp in [1e-4, 1e-3, 1e-2] {
        for min_n in [5.0, 20.0, 50.0] {
            points.push(vec![
                ("cost_complexity".to_string(), cp),
                ("min_n".to_string(), min_n),
            ]);
        }
    }
    let grid = Grid { points };
    let metrics = MetricSet::from_names(&["rmse", "rsq"])?;
    let tuned = tune_grid(&wf, &data, &folds, &grid, &metrics, Parallelism::threads(4))?;

    println!("show_best(rmse):");
    for row in tuned.show_best(Metric::Rmse, 3) {
        println!(
            "  config {:>2}  cp {:.0e}  min_n {:>2}  rmse {:.0} (se {:.0}, n={})",
            row.config,
            row.params[0].1,
            row.params[1].1,
            row.mean.unwrap_or(f64::NAN),
            row.std_err.unwrap_or(f64::NAN),
            row.n
        );
    }

    // --- tune's selection rules: the numerically best, and the simplest within one SE ---
    let best = tuned.select_best(Metric::Rmse)?;
    let simplest = tuned.select_by_one_std_err(
        Metric::Rmse,
        &[SortKey::desc("cost_complexity"), SortKey::desc("min_n")],
    )?;
    println!("select_best -> {:?}", best.params);
    println!("select_by_one_std_err -> {:?}", simplest.params);

    // --- finalize + last_fit: refit on the training rows, score the held-out rows once ---
    let last = wf.finalize(&simplest.params)?.last_fit(&data, &split, &metrics)?;
    for m in &last.metrics {
        println!("last_fit {:<6} {:.4}", m.name, m.value.unwrap_or(f64::NAN));
    }
    println!(
        "last_fit predictions for {} assessment rows; tree note: {}",
        last.predictions.len(),
        last.workflow.model().diagnostics().note.unwrap_or_default()
    );

    // --- fit_best: the numerically best candidate refitted on every row ---
    let final_fit = fit_best(&tuned, &wf, &data, Metric::Rmse)?;
    let json = final_fit.to_json()?;
    let back = FittedWorkflow::from_json(&json)?;
    assert_eq!(back.predict(&data)?, final_fit.predict(&data)?);
    println!(
        "fit_best model serialized ({} bytes) and round-tripped",
        json.to_string().len()
    );
    Ok(())
}
