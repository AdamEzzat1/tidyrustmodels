//! `boost_tree`: gradient-boosted trees by xgboost's `hist` algorithm
//! (`parsnip::boost_tree()` with engine `xgboost`), ported natively; see `native::xgb`
//! and ADR-0011 Amendment B.

use crate::backends::Engine;
use crate::native::xgb::{self, Booster, Objective, Params};
use serde::{Deserialize, Serialize};
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

/// How exact zeros in the design matrix are treated (ADR-0011 B, ADR-0015). Whatever
/// the policy, a *sparse* design (any sparse column, `DesignMatrix::is_sparse`) is a
/// `dgCMatrix` to xgboost and its zeros are missing, at fit and at predict.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum SparsePolicy {
    /// tidymodels' behaviour: a `Workflow::fit` applies `workflows:::toggle_sparsity`
    /// (the recipe's `step_dummy` emits sparse indicators when the decision says so);
    /// a standalone fit or a resampling fit reads the design as given.
    Auto,
    /// Never toggle the recipe: zeros are values unless the design is sparse.
    Dense,
    /// Zeros are always missing values with learned default directions, at fit and
    /// at predict, whatever the design.
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

    /// Force zeros to be missing values (`true`, [`SparsePolicy::ZerosMissing`]) or
    /// skip the workflow decision (`false`, [`SparsePolicy::Dense`]); a sparse design
    /// still reads zeros as missing in both cases.
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

    /// `Auto` lets a workflow toggle the recipe (`workflows:::toggle_sparsity` with
    /// engine `"xgboost"`); a forced policy does not.
    fn sparse_engine(&self) -> Option<&'static str> {
        (self.sparse == SparsePolicy::Auto).then_some("xgboost")
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
        // zeros are missing when the design is sparse (parsnip's maybe_sparse_matrix
        // hands xgboost a dgCMatrix: sparse input columns, or sparse step_dummy
        // indicators after workflows' toggle; ADR-0015) or when the policy forces it
        let zero_as_missing = self.sparse == SparsePolicy::ZerosMissing || x.is_sparse();
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
            forced_zeros_missing: Some(self.sparse == SparsePolicy::ZerosMissing),
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
    /// `Some(true)` when the policy forced zeros to be missing: prediction then reads
    /// zeros as missing on any design. `Some(false)`: the new data decides. `None`
    /// (states written before ADR-0015): the fitted rule applies to dense new data.
    #[serde(default)]
    forced_zeros_missing: Option<bool>,
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
        // at prediction the new data decides (parsnip bakes it and calls
        // maybe_sparse_matrix again): a sparse design is a dgCMatrix whatever the fit
        // saw, and a dense design reads zeros as values even after a sparse fit
        // (measured in R, VERIFICATION.md Phase 8j), unless the policy forced the rule
        let forced = self.forced_zeros_missing.unwrap_or(self.booster.zero_as_missing);
        let zam = forced || x.is_sparse();
        match &self.levels {
            None => Ok(Prediction::Numeric(
                (0..n)
                    .map(|i| f64::from(self.booster.predict_with(row(i), zam)[0]))
                    .collect(),
            )),
            Some(levels) => {
                let k = levels.len();
                let mut codes = Vec::with_capacity(n);
                let mut probs = vec![0.0; n * k];
                for i in 0..n {
                    let out = self.booster.predict_with(row(i), zam);
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
