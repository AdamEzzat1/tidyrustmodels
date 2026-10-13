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
        Capability, Coefficients, Column, ColumnKind, DesignMatrix, Error, FitDiagnostics, FittedModel,
        ModelSpec, Outcome, Parallelism, Prediction, Result, Role, Schema, Seed, Table,
    };
    pub use tidyrustmodels_metrics::{
        ConfusionMatrix, Estimator, KapWeighting, Metric, MetricSet, MetricValue,
    };
    pub use tidyrustmodels_models::{
        DecisionTreeSpec, Engine, LinearRegSpec, LogisticRegSpec, MultinomRegSpec, RandForestSpec,
    };
    pub use tidyrustmodels_recipes::{
        Recipe, Selector, StepCorr, StepDummy, StepImputeMean, StepImputeMedian, StepNormalize, StepNovel,
        StepNzv, StepOther, StepSelect, StepUnknown, StepZv,
    };
    pub use tidyrustmodels_resample::{initial_split, strata_from_numeric, vfold_cv, Folds, Split};
    pub use tidyrustmodels_tune::{grid_random, grid_regular, tune_grid, Grid, ParamRange, TuneResults};
    pub use tidyrustmodels_workflows::{FittedWorkflow, ResampleResults, Workflow};
}
