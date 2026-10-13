//! tidyrustmodels: a tidymodels-inspired machine learning framework for Rust.
//!
//! ```
//! use tidyrustmodels::prelude::*;
//!
//! # fn main() -> tidyrustmodels::Result<()> {
//! let data = Table::new(vec![
//!     ("mpg", Column::numeric(vec![21.0, 22.8, 18.7, 18.1, 14.3, 24.4])?),
//!     ("wt", Column::numeric(vec![2.62, 2.32, 3.44, 3.46, 3.57, 3.19])?),
//!     ("am", Column::categorical(&["manual", "manual", "auto", "auto", "auto", "auto"])?),
//! ])?;
//! let recipe = Recipe::new("mpg")
//!     .step(StepNormalize::new(Selector::AllNumericPredictors))
//!     .step(StepDummy::new(Selector::AllNominalPredictors));
//! let wf = Workflow::new(recipe, LinearRegSpec::new());
//! let fitted = wf.fit(&data)?;
//! let pred = fitted.predict(&data)?;
//! assert_eq!(pred.len(), 6);
//! # Ok(()) }
//! ```

#![forbid(unsafe_code)]
#![warn(missing_docs, rust_2018_idioms)]

pub mod io;

/// Arrow interop (`RecordBatch` <-> `Table`), enabled by the `arrow` feature (ADR-0003).
#[cfg(feature = "arrow")]
pub mod arrow;

pub use tidyrustmodels_core as core;
pub use tidyrustmodels_metrics as metrics;
pub use tidyrustmodels_models as models;
pub use tidyrustmodels_recipes as recipes;
pub use tidyrustmodels_resample as resample;
pub use tidyrustmodels_tune as tune;
pub use tidyrustmodels_workflows as workflows;

pub use tidyrustmodels_core::{Error, Result};

/// Everything needed for a typical workflow.
pub mod prelude {
    pub use tidyrustmodels_core::{
        Capability, CaseWeights, Coefficients, Column, ColumnKind, DesignMatrix, Error, FitDiagnostics,
        FittedModel, ModelSpec, Outcome, Parallelism, Prediction, Result, Role, Schema, Seed, Table,
        WeightKind,
    };
    pub use tidyrustmodels_metrics::{
        ConfusionMatrix, Estimator, KapWeighting, Metric, MetricSet, MetricValue,
    };
    pub use tidyrustmodels_models::{
        BoostTreeSpec, DecisionTreeSpec, Engine, LinearRegSpec, LogisticRegSpec, MultinomRegSpec,
        RandForestSpec,
    };
    pub use tidyrustmodels_recipes::{
        Recipe, Selector, SparseOutput, StepCorr, StepDummy, StepImputeMean, StepImputeMedian, StepNormalize,
        StepNovel, StepNzv, StepOther, StepSelect, StepUnknown, StepZv,
    };
    pub use tidyrustmodels_resample::{
        balance_observations, balance_prop, bootstraps, freq_table, group_bootstraps, group_initial_split,
        group_mc_cv, group_vfold_cv, initial_split, initial_time_split, initial_validation_split, loo_cv,
        make_strata_categorical, make_strata_numeric, mc_cv, rolling_origin, sliding_index, sliding_window,
        strata_from_numeric, vfold_cv, vfold_cv_repeated, Folds, GroupBalance, Split, ThreeWaySplit,
    };
    pub use tidyrustmodels_tune::{
        fit_best, grid_random, grid_regular, tune_grid, BestRow, Grid, ParamRange, Selection, SortKey,
        TuneResults,
    };
    pub use tidyrustmodels_workflows::{FittedWorkflow, LastFit, ResampleResults, Workflow};
}
