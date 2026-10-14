//! End-to-end example on the Palmer penguins data: recipe, model, resampling, tuning.
//!
//! Run from the repository root:
//!     cargo run --release --example penguins_workflow

use tidyrustmodels::io::read_csv;
use tidyrustmodels::prelude::*;

fn main() -> Result<()> {
    // --- data (CSV exported from R's modeldata::penguins; see reference/) ---
    let full = read_csv("reference/fixtures/datasets/penguins_sex.csv")?;
    let keep = [
        "sex",
        "species",
        "bill_length_mm",
        "bill_depth_mm",
        "flipper_length_mm",
        "body_mass_g",
    ];
    let data = Table::new(
        keep.iter()
            .map(|n| (n.to_string(), full.column(n).unwrap().clone()))
            .collect(),
    )?;
    println!(
        "{} rows, outcome `sex` with levels {:?}",
        data.nrows(),
        data.column("sex").unwrap().as_categorical().unwrap().1
    );

    // --- resampling: stratified 75/25 split and 5-fold CV, reproducible from seeds ---
    let strata = data.column("sex").unwrap().as_categorical().unwrap().0.to_vec();
    let split = initial_split(data.nrows(), 0.75, Some(&strata), Seed(42))?;
    let train = data.select_rows(&split.train)?;
    let test = data.select_rows(&split.test)?;

    // --- recipe: prep on training rows only, bake anywhere ---
    let recipe = Recipe::new("sex")
        .step(StepImputeMedian::new(Selector::AllNumericPredictors))
        .step(StepNormalize::new(Selector::AllNumericPredictors))
        .step(StepDummy::new(Selector::AllNominalPredictors));

    // --- model spec + workflow ---
    let wf = Workflow::new(recipe, LogisticRegSpec::new().engine(Engine::Linfa));
    let fitted = wf.fit(&train)?;
    let coefs = fitted.model().coefficients().unwrap();
    println!("intercept {:.4}; slopes:", coefs.intercept.unwrap());
    for (name, b) in &coefs.terms {
        println!("  {name:<22} {b:>9.4}");
    }
    println!("diagnostics: {:?}", fitted.model().diagnostics());

    // --- evaluate on the held-out split ---
    let pred = fitted.predict(&test)?;
    let truth = Outcome::from_column(test.column("sex").unwrap())?;
    for m in
        MetricSet::from_names(&["accuracy", "mn_log_loss", "precision", "recall"])?.evaluate(&truth, &pred)?
    {
        println!("test {:<12} {:?}", m.name, m.value);
    }

    // --- cross-validation with the recipe re-prepped inside every fold ---
    // (the same two metrics as examples/penguins_workflow.py; the tuning step below
    // selects on `mn_log_loss`, so it must be in the set)
    let metrics = MetricSet::from_names(&["accuracy", "mn_log_loss"])?;
    let folds = vfold_cv(data.nrows(), 5, Some(&strata), Seed(7))?;
    let cv = wf.fit_resamples(&data, &folds, &metrics, Parallelism::threads(4))?;
    for s in &cv.summary {
        println!(
            "cv {:<12} mean {:.4} std_err {:.4} (n={})",
            s.name,
            s.mean.unwrap(),
            s.std_err.unwrap(),
            s.n
        );
    }

    // --- tune the L2 penalty over a log10 grid ---
    let grid = grid_regular(&[ParamRange::log10("penalty", -4.0, 0.0)], 5)?;
    let tuned = tune_grid(&wf, &data, &folds, &grid, &metrics, Parallelism::threads(4))?;
    println!("best by mn_log_loss:");
    for row in tuned.show_best(Metric::MnLogLoss, 3) {
        println!(
            "  penalty {:.2e} -> {:.4} (se {:.4})",
            row.params[0].1,
            row.mean.unwrap_or(f64::NAN),
            row.std_err.unwrap_or(f64::NAN)
        );
    }
    // tune's selection rules and the final fit (ADR-0014)
    let best = tuned.select_best(Metric::MnLogLoss)?;
    let simplest = tuned.select_by_one_std_err(Metric::MnLogLoss, &[SortKey::desc("penalty")])?;
    println!(
        "select_best: candidate {}; one-SE rule: candidate {}",
        best.index, simplest.index
    );
    let final_fit = fit_best(&tuned, &wf, &data, Metric::MnLogLoss)?;
    println!("fit_best refitted on {} rows", final_fit.predict(&data)?.len());

    // --- serialize the fitted workflow and predict from the deserialized copy ---
    let json = fitted.to_json()?;
    let back = FittedWorkflow::from_json(&json)?;
    assert_eq!(back.predict(&test)?, pred);
    println!(
        "serialized workflow: {} bytes; round-trip predictions identical",
        json.to_string().len()
    );
    Ok(())
}
