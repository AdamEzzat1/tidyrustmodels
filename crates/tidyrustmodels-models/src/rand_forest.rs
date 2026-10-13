//! `rand_forest`: random forests with ranger's growing rules (`parsnip::rand_forest()`
//! with engine `ranger`); see `native::forest` and ADR-0011.

use crate::backends::Engine;
use crate::native::forest::{self, Forest, ForestOptions, Kind};
use serde::{Deserialize, Serialize};
use tidyrustmodels_core::traits::check_features;
use tidyrustmodels_core::{
    BackendInfo, Capability, CaseWeights, DesignMatrix, Error, FitDiagnostics, FittedModel, ModelSpec,
    Outcome, Prediction, Result, Seed,
};

/// A random forest with ranger's defaults as parsnip applies them: `trees = 500`,
/// `mtry = floor(sqrt(p))` (at least 1), `min_n = 5` for regression and `10` for
/// classification (parsnip fits ranger probability forests), bootstrap with
/// replacement, no depth limit. Predictors must be numeric and complete.
///
/// Forests are random: the `seed` fixes the per-tree ChaCha8 streams, which differ
/// from ranger's generator, so two forests are compared statistically (VERIFICATION.md
/// Phase 8e), never tree by tree.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct RandForestSpec {
    engine: Engine,
    mtry: Option<usize>,
    trees: usize,
    min_n: Option<usize>,
    seed: Seed,
}

impl Default for RandForestSpec {
    fn default() -> Self {
        RandForestSpec {
            engine: Engine::Native,
            mtry: None,
            trees: 500,
            min_n: None,
            seed: Seed(0),
        }
    }
}

impl RandForestSpec {
    /// ranger defaults, seed 0.
    pub fn new() -> Self {
        Self::default()
    }

    /// Engine; only `native` implements this model.
    pub fn engine(mut self, engine: Engine) -> Self {
        self.engine = engine;
        self
    }

    /// `mtry`: variables tried at each split (default `floor(sqrt(p))`).
    pub fn mtry(mut self, mtry: usize) -> Self {
        self.mtry = Some(mtry);
        self
    }

    /// `trees` (`num.trees`, default 500).
    pub fn trees(mut self, trees: usize) -> Self {
        self.trees = trees;
        self
    }

    /// `min_n` (`min.node.size`; default 5 regression / 10 classification).
    pub fn min_n(mut self, min_n: usize) -> Self {
        self.min_n = Some(min_n);
        self
    }

    /// Seed of the per-tree random streams.
    pub fn seed(mut self, seed: Seed) -> Self {
        self.seed = seed;
        self
    }

    fn validate(&self) -> Result<()> {
        if self.engine != Engine::Native {
            return Err(Error::Unsupported(format!(
                "rand_forest is implemented on the `native` engine only (got `{}`)",
                self.engine.as_str()
            )));
        }
        if self.trees == 0 {
            return Err(Error::InvalidArgument("trees must be >= 1".into()));
        }
        if self.mtry == Some(0) {
            return Err(Error::InvalidArgument("mtry must be >= 1".into()));
        }
        if self.min_n == Some(0) {
            return Err(Error::InvalidArgument("min_n must be >= 1".into()));
        }
        Ok(())
    }
}

impl ModelSpec for RandForestSpec {
    /// ranger accepts sparse matrices, so workflows runs its toggle (whose rule
    /// always says "no" for ranger); the fit is the same either way.
    fn sparse_engine(&self) -> Option<&'static str> {
        Some("ranger")
    }

    fn name(&self) -> &'static str {
        "rand_forest"
    }

    fn capabilities(&self) -> &'static [Capability] {
        &[
            Capability::Regression,
            Capability::BinaryClassification,
            Capability::MulticlassClassification,
            Capability::ClassProbabilities,
            Capability::CaseWeights,
        ]
    }

    fn backend(&self) -> BackendInfo {
        BackendInfo {
            engine: "native".into(),
            crate_version: env!("CARGO_PKG_VERSION").into(),
            solver: "random forest with ranger 0.18.0 growing rules (variance / gini, bootstrap, mtry)"
                .into(),
        }
    }

    fn fit(&self, x: &DesignMatrix, y: &Outcome) -> Result<Box<dyn FittedModel>> {
        self.fit_impl(x, y, None)
    }

    /// ranger's `case.weights`: sampling weights for the bootstrap of every tree
    /// (rows with larger weights are drawn more often; zero-weight rows never).
    fn fit_weighted(
        &self,
        x: &DesignMatrix,
        y: &Outcome,
        weights: &CaseWeights,
    ) -> Result<Box<dyn FittedModel>> {
        self.fit_impl(x, y, Some(weights))
    }

    fn tunable(&self) -> &'static [&'static str] {
        &["mtry", "trees", "min_n"]
    }

    fn with_param(&self, name: &str, value: f64) -> Result<Box<dyn ModelSpec>> {
        let v = value.round();
        if !v.is_finite() || v < 0.0 {
            return Err(Error::InvalidArgument(format!(
                "`{name}` must be a non-negative number, got {value}"
            )));
        }
        let s = match name {
            "mtry" => self.clone().mtry(v as usize),
            "trees" => self.clone().trees(v as usize),
            "min_n" => self.clone().min_n(v as usize),
            other => {
                return Err(Error::Unsupported(format!(
                    "rand_forest has no tunable parameter `{other}`"
                )))
            }
        };
        s.validate()?;
        Ok(Box::new(s))
    }

    fn clone_box(&self) -> Box<dyn ModelSpec> {
        Box::new(self.clone())
    }
}

impl RandForestSpec {
    fn fit_impl(
        &self,
        x: &DesignMatrix,
        y: &Outcome,
        weights: Option<&CaseWeights>,
    ) -> Result<Box<dyn FittedModel>> {
        self.validate()?;
        let (n, p) = (x.nrows(), x.ncols());

        if n != y.len() {
            return Err(Error::InvalidArgument(format!(
                "design matrix has {n} rows but the outcome has {}",
                y.len()
            )));
        }
        if n == 0 || p == 0 {
            return Err(Error::InvalidArgument(
                "cannot fit a forest on zero rows or zero predictors".into(),
            ));
        }
        if let Some(i) = x.data().iter().position(|v| !v.is_finite()) {
            return Err(Error::InvalidValue(format!(
                "rand_forest needs finite predictors (non-finite at element {i}); impute first"
            )));
        }
        let (kind, yv, levels, default_min_n) = match y {
            Outcome::Numeric(v) => (Kind::Regression, v.clone(), None, 5),
            Outcome::Class { codes, levels } => (
                Kind::Probability { k: levels.len() },
                codes.iter().map(|c| f64::from(*c)).collect(),
                Some(levels.clone()),
                10,
            ),
        };
        let mtry = self.mtry.unwrap_or(((p as f64).sqrt() as usize).max(1));
        if mtry > p {
            return Err(Error::InvalidArgument(format!(
                "mtry ({mtry}) exceeds the number of predictors ({p})"
            )));
        }
        let opts = ForestOptions {
            trees: self.trees,
            mtry,
            min_node_size: self.min_n.unwrap_or(default_min_n),
            min_bucket: 1,
            seed: self.seed,
        };
        crate::linear::check_weights(weights, n)?;
        if let Some(w) = weights {
            if w.values().iter().all(|v| *v == 0.0) {
                return Err(Error::InvalidValue("all case weights are zero".into()));
            }
        }
        let forest = forest::fit_weighted(x.data(), n, p, &yv, kind, opts, weights.map(|w| w.values()));
        Ok(Box::new(FittedRandForest {
            model: "rand_forest".into(),
            format_version: crate::fitted::MODEL_FORMAT_VERSION,
            forest,
            names: x.names().to_vec(),
            levels,
            backend: self.backend(),
            diagnostics: FitDiagnostics {
                note: Some(format!(
                    "{} trees, mtry {mtry}, min.node.size {}{}",
                    self.trees,
                    opts.min_node_size,
                    match weights {
                        Some(w) => format!("; case weights ({}): weighted bootstrap", w.kind().as_str()),
                        None => String::new(),
                    }
                )),
                ..Default::default()
            },
        }))
    }
}

/// A fitted random forest.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct FittedRandForest {
    model: String,
    format_version: u32,
    /// The forest.
    pub forest: Forest,
    names: Vec<String>,
    /// Outcome levels for classification, `None` for regression.
    pub levels: Option<Vec<String>>,
    backend: BackendInfo,
    diagnostics: FitDiagnostics,
}

impl FittedModel for FittedRandForest {
    fn predict(&self, x: &DesignMatrix) -> Result<Prediction> {
        check_features(&self.names, x)?;
        let n = x.nrows();
        match &self.levels {
            None => Ok(Prediction::Numeric(
                (0..n).map(|i| self.forest.predict(|v| x.get(i, v))[0]).collect(),
            )),
            Some(levels) => {
                let k = levels.len();
                let mut codes = Vec::with_capacity(n);
                let mut probs = vec![0.0; n * k];
                for i in 0..n {
                    let p = self.forest.predict(|v| x.get(i, v));
                    let mut best = 0;
                    for (j, pj) in p.iter().enumerate() {
                        probs[j * n + i] = *pj;
                        if *pj > p[best] {
                            best = j;
                        }
                    }
                    codes.push(best as u32);
                }
                Ok(Prediction::Class {
                    codes,
                    levels: levels.clone(),
                    probabilities: Some(DesignMatrix::from_column_major(probs, n, k, levels.clone())?),
                })
            }
        }
    }

    fn feature_names(&self) -> &[String] {
        &self.names
    }

    fn diagnostics(&self) -> FitDiagnostics {
        self.diagnostics.clone()
    }

    fn backend(&self) -> BackendInfo {
        self.backend.clone()
    }

    fn state(&self) -> Result<serde_json::Value> {
        serde_json::to_value(self).map_err(|e| Error::Serialization(e.to_string()))
    }
}
