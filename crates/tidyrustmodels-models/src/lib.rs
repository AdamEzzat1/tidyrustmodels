//! Estimator specifications (`linear_reg`, `logistic_reg`) and the backend adapters
//! that fit them.
//!
//! Backends are used for fitting only. Every prediction is computed by this crate
//! from the fitted coefficients, so prediction semantics are identical across engines
//! and serialized models carry no backend-specific state (see ADR-0005).

#![forbid(unsafe_code)]
#![warn(missing_docs, rust_2018_idioms)]

pub mod alias;
pub mod backends;
pub mod diag;
pub mod fitted;
pub mod linear;
pub mod logistic;

pub use backends::Engine;
pub use fitted::{fitted_model_from_state, FittedLinear, FittedLogistic};
pub use linear::LinearRegSpec;
pub use logistic::LogisticRegSpec;
