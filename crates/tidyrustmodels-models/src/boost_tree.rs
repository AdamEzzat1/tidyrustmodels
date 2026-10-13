//! `boost_tree`: gradient-boosted trees by xgboost's `hist` algorithm
//! (`parsnip::boost_tree()` with engine `xgboost`), ported natively; see `native::xgb`
//! and ADR-0011 Amendment B.

use crate::backends::Engine;
use crate::native::xgb::{self, Booster, Objective, Params};
use serde::{Deserialize, Serialize};
use tidyrustmodels_core::sparsity::should_use_sparsity;
use tidyrustmodels_core::traits::check_features;
use tidyrustmodels_core::{
    BackendInfo, Capability, CaseWeights, DesignMatrix, Error, FitDiagnostics, FittedModel, ModelSpec,
    Outcome, Prediction, Result,
};

/// Boosted trees with parsnip's xgboost defaults: `trees = 15` (`nrounds`),
/// `tree_depth = 6` (`max_depth`), `learn_rate = 0.3` (`eta`), `min_n = 1`
/// (`min_child_weight`), `loss_reduction = 0` (`gamma`), L2 penalty `lambda = 1`, L1
/// penalty `alpha = 0`, 256 histogram bins, squared error / logistic / softprob by
/// outcome, base score from the training labels. Predictors must be numeric and
/// complete. `mtry` (`colsample_bynode`), `sample_size` (`subsample`) and
/// `stop_iter` are not offered: the first two draw random numbers that cannot be
/// matched to xgboost's and the third needs a validation set.
///
/// Zeros as missing values: inside a workflow, tidymodels passes xgboost a sparse
/// matrix when `workflows` judges the baked design sparse enough, and xgboost then
/// treats every exact zero as missing (ADR-0011 Amendment B). The default policy
/// (`Auto`) reproduces that decision from the recipe's sparsity estimate; a
/// standalone `fit` is dense, as `parsnip::fit_xy()` on a data frame is;
/// [`zeros_as_missing`](Self::zeros_as_missing) forces either behaviour.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct BoostTreeSpec {
    engine: Engine,
    trees: usize,
    tree_depth: usize,
    learn_rate: f64,
    min_n: f64,
    loss_reduction: f64,
    lambda: f64,
    alpha: f64,
    sparse: SparsePolicy,
}

/// How exact zeros in the design matrix are treated.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum SparsePolicy {
    /// Decide as `workflows::toggle_sparsity` does when fitted through a workflow;
    /// dense otherwise.
    Auto,
    /// Zeros are values.
    Dense,
    /// Zeros are missing values with learned default directions.
    ZerosMissing,
}

impl Default for BoostTreeSpec {
    fn default() -> Self {
        BoostTreeSpec {
            engine: Engine::Native,
            trees: 15,
            tree_depth: 6,
            learn_rate: 0.3,
            min_n: 1.0,
            loss_reduction: 0.0,
            lambda: 1.0,
            alpha: 0.0,
            sparse: SparsePolicy::Auto,
        }
    }
}

impl BoostTreeSpec {
    /// parsnip's xgboost defaults on the native engine.
    pub fn new() -> Self {
        Self::default()
    }

    /// Engine; only `native` implements this model.
    pub fn engine(mut self, engine: Engine) -> Self {
        self.engine = engine;
        self
    }

    /// `trees` → `nrounds` (default 15).
    pub fn trees(mut self, trees: usize) -> Self {
        self.trees = trees;
        self
    }

    /// `tree_depth` → `max_depth` (default 6).
    pub fn tree_depth(mut self, depth: usize) -> Self {
        self.tree_depth = depth;
        self
    }

    /// `learn_rate` → `eta` (default 0.3).
    pub fn learn_rate(mut self, eta: f64) -> Self {
        self.learn_rate = eta;
        self
    }

    /// `min_n` → `min_child_weight` (default 1; capped at the number of rows as parsnip
    /// does).
    pub fn min_n(mut self, min_n: f64) -> Self {
        self.min_n = min_n;
        self
    }

    /// `loss_reduction` → `gamma` (default 0).
    pub fn loss_reduction(mut self, gamma: f64) -> Self {
        self.loss_reduction = gamma;
        self
    }

    /// xgboost `lambda` (L2 on leaf weights, default 1).
    pub fn lambda(mut self, lambda: f64) -> Self {
        self.lambda = lambda;
        self
    }

    /// xgboost `alpha` (L1 on leaf weights, default 0).
    pub fn alpha(mut self, alpha: f64) -> Self {
        self.alpha = alpha;
        self
    }

    /// Force zeros to be missing values (`true`) or ordinary values (`false`) instead
    /// of the workflow decision.
    pub fn zeros_as_missing(mut self, yes: bool) -> Self {
        self.sparse = if yes {
            SparsePolicy::ZerosMissing
        } else {
            SparsePolicy::Dense
        };
        self
    }

    /// The sparse-data policy.
    pub fn sparse_policy(&self) -> SparsePolicy {
        self.sparse
    }

    fn validate(&self) -> Result<()> {
        if self.engine != Engine::Native {
            return Err(Error::Unsupported(format!(
                "boost_tree is implemented on the `native` engine only (got `{}`)",
                self.engine.as_str()
            )));
        }
        if self.trees == 0 {
            return Err(Error::InvalidArgument("trees must be >= 1".into()));
        }
        if self.tree_depth == 0 {
            return Err(Error::InvalidArgument("tree_depth must be >= 1".into()));
        }
        for (name, v) in [
            ("learn_rate", self.learn_rate),
            ("min_n", self.min_n),
            ("loss_reduction", self.loss_reduction),
            ("lambda", self.lambda),
            ("alpha", self.alpha),
        ] {
            if !v.is_finite() || v < 0.0 {
                return Err(Error::InvalidArgument(format!(
                    "{name} must be finite and >= 0 (got {v})"
                )));
            }
        }
        if self.learn_rate == 0.0 {
            return Err(Error::InvalidArgument("learn_rate must be > 0".into()));
        }
        Ok(())
    }
}

impl ModelSpec for BoostTreeSpec {
    fn name(&self) -> &'static str {
        "boost_tree"
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
            solver: "xgboost 3.2.1 hist booster port (depth-wise, single thread, float32)".into(),
        }
    }

    fn fit(&self, x: &DesignMatrix, y: &Outcome) -> Result<Box<dyn FittedModel>> {
        self.fit_impl(x, y, None)
    }

    /// xgboost's DMatrix `weight`: scales every row's gradient pair, weights the
    /// quantile sketch and the base score (parsnip passes case weights this way).
    fn fit_weighted(
        &self,
        x: &DesignMatrix,
        y: &Outcome,
        weights: &CaseWeights,
    ) -> Result<Box<dyn FittedModel>> {
        self.fit_impl(x, y, Some(weights))
    }

    /// In parsnip's argument order (`boost_tree(mtry, trees, min_n, tree_depth,
    /// learn_rate, loss_reduction, ...)`), which tune uses to number its `.config`s.
    fn tunable(&self) -> &'static [&'static str] {
        &["trees", "min_n", "tree_depth", "learn_rate", "loss_reduction"]
    }

    fn with_param(&self, name: &str, value: f64) -> Result<Box<dyn ModelSpec>> {
        if !value.is_finite() || value < 0.0 {
            return Err(Error::InvalidArgument(format!(
                "`{name}` must be a non-negative number, got {value}"
            )));
        }
        let s = match name {
            "trees" => self.clone().trees(value.round() as usize),
            "tree_depth" => self.clone().tree_depth(value.round() as usize),
            "learn_rate" => self.clone().learn_rate(value),
            "min_n" => self.clone().min_n(value),
            "loss_reduction" => self.clone().loss_reduction(value),
            other => {
                return Err(Error::Unsupported(format!(
                    "boost_tree has no tunable parameter `{other}`"
                )))
            }
        };
        s.validate()?;
        Ok(Box::new(s))
    }

    fn resolve_sparsity(&self, estimated_sparsity: f64, n_rows: usize) -> Option<Box<dyn ModelSpec>> {
        if self.sparse != SparsePolicy::Auto {
            return None;
        }
        let mut s = self.clone();
        s.sparse = if should_use_sparsity(estimated_sparsity, "xgboost", n_rows) {
            SparsePolicy::ZerosMissing
        } else {
            SparsePolicy::Dense
        };
        Some(Box::new(s))
    }

    fn clone_box(&self) -> Box<dyn ModelSpec> {
        Box::new(self.clone())
    }
}

impl BoostTreeSpec {
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
                "cannot boost on zero rows or zero predictors".into(),
            ));
        }

        if let Some(i) = x.data().iter().position(|v| !v.is_finite()) {
            return Err(Error::InvalidValue(format!(
                "boost_tree needs finite predictors (non-finite at element {i}); impute first"
            )));
        }
        let (objective, labels, levels): (Objective, Vec<f64>, Option<Vec<String>>) = match y {
            Outcome::Numeric(v) => {
                if let Some(i) = v.iter().position(|t| !t.is_finite()) {
                    return Err(Error::InvalidValue(format!(
                        "boost_tree needs a finite outcome (non-finite at row {i})"
                    )));
                }
                (Objective::SquaredError, v.clone(), None)
            }
            Outcome::Class { codes, levels } => {
                if levels.len() < 2 {
                    return Err(Error::InvalidArgument(
                        "classification needs at least two outcome levels".into(),
                    ));
                }
                if levels.len() == 2 {
                    // parsnip event_level = "first": the first level is the event
                    (
                        Objective::Logistic,
                        codes.iter().map(|c| if *c == 0 { 1.0 } else { 0.0 }).collect(),
                        Some(levels.clone()),
                    )
                } else {
                    (
                        Objective::Softprob { k: levels.len() },
                        codes.iter().map(|c| f64::from(*c)).collect(),
                        Some(levels.clone()),
                    )
                }
            }
        };
        let params = Params {
            rounds: self.trees,
            max_depth: self.tree_depth as u32,
            eta: self.learn_rate as f32,
            gamma: self.loss_reduction as f32,
            min_child_weight: self.min_n.min(n as f64) as f32,
            lambda: self.lambda as f32,
            alpha: self.alpha as f32,
            max_bin: 256,
        };
        let zero_as_missing = self.sparse == SparsePolicy::ZerosMissing;
        crate::linear::check_weights(weights, n)?;
        let booster = xgb::fit_weighted(
            x.data(),
            n,
            p,
            &labels,
            objective,
            params,
            zero_as_missing,
            weights.map(|w| w.values()),
        );
        Ok(Box::new(FittedBoostTree {
            model: "boost_tree".into(),
            format_version: crate::fitted::MODEL_FORMAT_VERSION,
            booster,
            names: x.names().to_vec(),
            levels,
            backend: self.backend(),
            diagnostics: FitDiagnostics {
                note: Some(format!(
                    "{} rounds, max_depth {}, eta {}, min_child_weight {}, gamma {}, zeros as missing: {}{}",
                    self.trees,
                    self.tree_depth,
                    params.eta,
                    params.min_child_weight,
                    params.gamma,
                    zero_as_missing,
                    match weights {
                        Some(w) => format!("; case weights ({})", w.kind().as_str()),
                        None => String::new(),
                    }
                )),
                ..Default::default()
            },
        }))
    }
}

/// A fitted boosted-tree model.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct FittedBoostTree {
    model: String,
    format_version: u32,
    /// The booster.
    pub booster: Booster,
    names: Vec<String>,
    /// Outcome levels for classification, `None` for regression.
    pub levels: Option<Vec<String>>,
    backend: BackendInfo,
    diagnostics: FitDiagnostics,
}

impl FittedModel for FittedBoostTree {
    fn predict(&self, x: &DesignMatrix) -> Result<Prediction> {
        check_features(&self.names, x)?;
        let n = x.nrows();
        // xgb.DMatrix stores the predictors as float
        let row = |i: usize| move |f: usize| x.get(i, f) as f32;
        match &self.levels {
            None => Ok(Prediction::Numeric(
                (0..n)
                    .map(|i| f64::from(self.booster.predict(row(i))[0]))
                    .collect(),
            )),
            Some(levels) => {
                let k = levels.len();
                let mut codes = Vec::with_capacity(n);
                let mut probs = vec![0.0; n * k];
                for i in 0..n {
                    let out = self.booster.predict(row(i));
                    if k == 2 {
                        // parsnip: .pred_<first> = p, .pred_<second> = 1 - p (in double);
                        // class = first level when p >= 0.5
                        let p = f64::from(out[0]);
                        probs[i] = p;
                        probs[n + i] = 1.0 - p;
                        codes.push(if p >= 0.5 { 0 } else { 1 });
                    } else {
                        let mut best = 0;
                        for (j, pj) in out.iter().enumerate() {
                            probs[j * n + i] = f64::from(*pj);
                            if *pj > out[best] {
                                best = j;
                            }
                        }
                        codes.push(best as u32);
                    }
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
