//! Structured error type shared by every crate.

use thiserror::Error;

/// Every fallible tidyrustmodels operation returns this error.
#[derive(Debug, Error)]
#[non_exhaustive]
pub enum Error {
    /// Input data does not match the schema a fitted object was trained on.
    #[error("schema mismatch: {0}")]
    SchemaMismatch(String),

    /// A value violates a documented precondition (non-finite, missing where not
    /// allowed, unseen categorical level, ...).
    #[error("invalid value: {0}")]
    InvalidValue(String),

    /// An argument is out of range or inconsistent with other arguments.
    #[error("invalid argument: {0}")]
    InvalidArgument(String),

    /// The requested capability is not provided by this model or engine.
    #[error("unsupported: {0}")]
    Unsupported(String),

    /// A backend crate reported a failure. The engine name is recorded so the user
    /// knows which implementation failed.
    #[error("backend `{engine}` failed: {message}")]
    Backend {
        /// Engine identifier (e.g. `linfa`, `smartcore`).
        engine: String,
        /// Backend-supplied message.
        message: String,
    },

    /// Serialization or deserialization failed, including version mismatches.
    #[error("serialization: {0}")]
    Serialization(String),
}

/// Convenience alias used throughout the workspace.
pub type Result<T> = std::result::Result<T, Error>;

impl Error {
    /// Build a backend error from any displayable source.
    pub fn backend(engine: impl Into<String>, source: impl std::fmt::Display) -> Self {
        Error::Backend {
            engine: engine.into(),
            message: source.to_string(),
        }
    }
}
