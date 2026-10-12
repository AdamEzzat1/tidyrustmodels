//! Preprocessing with recipes semantics: `prep` estimates on training data, `bake`
//! applies to any data, and nothing is ever re-estimated at bake time.
//!
//! Semantics mirror the R `recipes` package (1.3.2) where the two overlap; every
//! deliberate difference is listed in `COMPATIBILITY.md`.

#![forbid(unsafe_code)]
#![warn(missing_docs, rust_2018_idioms)]

pub mod names;
pub mod recipe;
pub mod selector;
pub mod steps;

pub use recipe::{FittedRecipe, Recipe};
pub use selector::Selector;
pub use steps::{StepDummy, StepImputeMean, StepImputeMedian, StepNormalize, StepSelect};
