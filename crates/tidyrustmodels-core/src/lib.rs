//! Core contracts for tidyrustmodels.
//!
//! This crate defines the data types every other crate agrees on (`Table`, `Column`,
//! `Schema`, `DesignMatrix`, `Outcome`, `Prediction`), the structured `Error` type,
//! the lifecycle traits (`Step`/`FittedStep`, `ModelSpec`/`FittedModel`), and the
//! reproducibility policy (`Seed`, `Parallelism`).
//!
//! It deliberately has no numerical-library dependency: backends convert at their own
//! boundary (see ADR-0002).

#![forbid(unsafe_code)]
#![warn(missing_docs, rust_2018_idioms)]

pub mod data;
pub mod error;
pub mod matrix;
pub mod outcome;
pub mod rng;
pub mod traits;

pub use data::{Column, ColumnKind, Field, Role, Schema, Table};
pub use error::{Error, Result};
pub use matrix::DesignMatrix;
pub use outcome::{Coefficients, FitDiagnostics, Outcome, Prediction};
pub use rng::{Parallelism, Seed};
pub use traits::{BackendInfo, Capability, FittedModel, FittedStep, ModelSpec, Step};
