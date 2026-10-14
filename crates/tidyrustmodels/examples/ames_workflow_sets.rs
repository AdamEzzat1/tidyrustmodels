//! Comparing preprocessors x models on one set of resamples with a workflow set
//! (workflowsets semantics, ADR-0016), then refitting the winner.
//!
//! Run from the repository root:
//!     cargo run --release --example ames_workflow_sets

use tidyrustmodels::io::read_csv;
use tidyrustmodels::prelude::*;

fn main() -> Result<()> {
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
    let folds = vfold_cv(data.nrows(), 5, None, Seed(3))?;

    // --- two preprocessors (named) x three models (named) = six workflows ---
    let plain = Recipe::new("Sale_Price").step(StepDummy::new(Selector::AllNominalPredictors));
    let pooled = Recipe::new("Sale_Price")
        .step(StepOther::new(Selector::AllNominalPredictors).threshold(0.05))
        .step(StepNormalize::new(Selector::AllNumericPredictors))
        .step(StepDummy::new(Selector::AllNominalPredictors));
    let mut set = workflow_set(
        vec![(Some("plain".into()), plain), (Some("pooled".into()), pooled)],
        vec![
            (Some("lm".into()), Box::new(LinearRegSpec::new())),
            (Some("tree".into()), Box::new(DecisionTreeSpec::new())),
            (Some("xgb".into()), Box::new(BoostTreeSpec::new().trees(50))),
        ],
        true,
    )?;
    println!("workflows: {:?}", set.ids());

    // --- the tree workflows get a grid (option_add(grid = ..)); the others are resampled ---
    let grid = Grid {
        points: [1e-4, 1e-3, 1e-2]
            .iter()
            .map(|cp| vec![("cost_complexity".to_string(), *cp)])
            .collect(),
    };
    for id in ["plain_tree", "pooled_tree"] {
        set.option_grid(id, grid.clone())?;
    }
    let metrics = MetricSet::from_names(&["rmse", "rsq"])?;
    set.workflow_map(&data, &folds, &metrics, Parallelism::threads(4))?;

    // --- rank_results(select_best = TRUE): each workflow's best candidate, ranked on rmse ---
    println!(
        "{:<12} {:<17} {:>8} {:>8}  rank",
        "wflow_id", ".config", "rmse", "rsq"
    );
    let ranked = set.rank_results(Some(Metric::Rmse), true)?;
    for pair in ranked.chunks(2) {
        let (rmse, rsq) = (&pair[0], &pair[1]);
        println!(
            "{:<12} {:<17} {:>8.0} {:>8.4}  {}",
            rmse.wflow_id,
            rmse.config,
            rmse.mean.unwrap_or(f64::NAN),
            rsq.mean.unwrap_or(f64::NAN),
            rmse.rank.map_or("NA".to_string(), |r| r.to_string())
        );
    }

    // --- fit_best: the top workflow, finalized with its best parameters, on every row ---
    let (id, params) = set.best(Some(Metric::Rmse))?;
    println!("winner: {id} with {params:?}");
    let fitted = set.fit_best(&data, Some(Metric::Rmse))?;
    let first = match fitted.predict(&data)? {
        Prediction::Numeric(v) => v[0],
        _ => unreachable!("regression"),
    };
    println!("refitted on {} rows; first prediction {first:.0}", data.nrows());
    Ok(())
}
