//! Backend adapters. Each adapter converts the framework's design matrix into the
//! backend's array type, runs the fit, and returns plain coefficients.

#[cfg(feature = "linfa")]
pub mod linfa;
#[cfg(feature = "smartcore")]
pub mod smartcore;

use crate::logistic::LogisticOptions;
use serde::{Deserialize, Serialize};
use tidyrustmodels_core::{BackendInfo, DesignMatrix, Error, FitDiagnostics, Result};

/// Available fitting engines. Availability depends on compile-time features.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize, Default)]
#[serde(rename_all = "lowercase")]
pub enum Engine {
    /// `linfa-linear` / `linfa-logistic` 0.8.1.
    Linfa,
    /// `smartcore` 0.6.16.
    Smartcore,
    /// Pure-Rust ports of R's own algorithms (`lm.fit` pivoted QR, `glm.fit` IRLS);
    /// always compiled in and the default since ADR-0009 (exact R parity, 3–4× faster
    /// logistic fits than the L-BFGS engines in BENCHMARKS.md run `…-f`).
    #[default]
    Native,
}

impl Engine {
    /// Stable identifier.
    pub fn as_str(self) -> &'static str {
        match self {
            Engine::Linfa => "linfa",
            Engine::Smartcore => "smartcore",
            Engine::Native => "native",
        }
    }

    /// Parse a stable identifier.
    pub fn from_name(name: &str) -> Option<Engine> {
        match name {
            "linfa" => Some(Engine::Linfa),
            "smartcore" => Some(Engine::Smartcore),
            "native" => Some(Engine::Native),
            _ => None,
        }
    }

    /// Whether this engine was compiled in.
    pub fn is_available(self) -> bool {
        match self {
            Engine::Linfa => cfg!(feature = "linfa"),
            Engine::Smartcore => cfg!(feature = "smartcore"),
            Engine::Native => true,
        }
    }
}

/// What every adapter returns.
#[derive(Clone, Debug)]
pub struct FitOutput {
    /// Intercept, if requested.
    pub intercept: Option<f64>,
    /// Slope per design-matrix column.
    pub coefs: Vec<f64>,
    /// Adapter-supplied diagnostics (the framework fills in the gradient check).
    pub diagnostics: FitDiagnostics,
}

/// Backend facts for a model/engine pair.
pub fn info(engine: Engine, model: &str) -> BackendInfo {
    match (engine, model) {
        (Engine::Linfa, "linear_reg") => BackendInfo {
            engine: "linfa".into(),
            crate_version: "0.8.1".into(),
            solver: "least squares via linfa-linalg (pure Rust, no LAPACK)".into(),
        },
        (Engine::Linfa, _) => BackendInfo {
            engine: "linfa".into(),
            crate_version: "0.8.1".into(),
            solver: "L-BFGS (argmin 0.11, m=10, More-Thuente line search)".into(),
        },
        (Engine::Smartcore, "linear_reg") => BackendInfo {
            engine: "smartcore".into(),
            crate_version: "0.6.16".into(),
            solver: "SVD least squares (smartcore default)".into(),
        },
        (Engine::Smartcore, _) => BackendInfo {
            engine: "smartcore".into(),
            crate_version: "0.6.16".into(),
            solver: "L-BFGS (smartcore, g_atol=1e-8)".into(),
        },
        (Engine::Native, "linear_reg") => BackendInfo {
            engine: "native".into(),
            crate_version: env!("CARGO_PKG_VERSION").into(),
            solver: "Householder QR with dqrdc2 pivoting (lm.fit port, tol 1e-7)".into(),
        },
        (Engine::Native, _) => BackendInfo {
            engine: "native".into(),
            crate_version: env!("CARGO_PKG_VERSION").into(),
            solver: "IRLS with weighted pivoted QR (glm.fit port)".into(),
        },
    }
}

fn unavailable(engine: Engine) -> Error {
    Error::Unsupported(format!(
        "engine `{}` was not compiled in (enable the `{}` feature of tidyrustmodels-models)",
        engine.as_str(),
        engine.as_str()
    ))
}

/// Dispatch an OLS fit.
pub(crate) fn fit_linear(engine: Engine, x: &DesignMatrix, y: &[f64], intercept: bool) -> Result<FitOutput> {
    match engine {
        #[cfg(feature = "linfa")]
        Engine::Linfa => linfa::fit_linear(x, y, intercept),
        #[cfg(feature = "smartcore")]
        Engine::Smartcore => smartcore::fit_linear(x, y, intercept),
        Engine::Native => crate::native::fit_linear(x, y, intercept),
        #[allow(unreachable_patterns)]
        other => Err(unavailable(other)),
    }
}

/// Dispatch a logistic fit. `y` holds 0/1 codes; coefficients are for class 1.
pub(crate) fn fit_logistic(
    engine: Engine,
    x: &DesignMatrix,
    y: &[u32],
    opts: LogisticOptions,
) -> Result<FitOutput> {
    match engine {
        #[cfg(feature = "linfa")]
        Engine::Linfa => linfa::fit_logistic(x, y, opts),
        #[cfg(feature = "smartcore")]
        Engine::Smartcore => smartcore::fit_logistic(x, y, opts),
        Engine::Native => crate::native::fit_logistic(
            x,
            y,
            crate::native::NativeLogisticOptions {
                intercept: opts.intercept,
                penalty: opts.penalty,
                epsilon: opts.irls_epsilon,
                maxit: opts.irls_max_iterations,
            },
        ),
        #[allow(unreachable_patterns)]
        other => Err(unavailable(other)),
    }
}
