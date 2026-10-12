//! `decision_tree`: CART by the rpart algorithm (`parsnip::decision_tree()` with engine
//! `rpart`), ported natively; see `native::rpart` and ADR-0011.

use crate::backends::Engine;
use crate::native::rpart::{self, Control, Method, NodeValue, Tree};
use serde::{Deserialize, Serialize};
use tidyrustmodels_core::traits::check_features;
use tidyrustmodels_core::{
    BackendInfo, Capability, DesignMatrix, Error, FitDiagnostics, FittedModel, ModelSpec, Outcome,
    Prediction, Result,
};

/// A single CART tree with rpart's defaults: `cost_complexity = 0.01`,
/// `tree_depth = 30`, `min_n = 20` (`minsplit`; `minbucket = round(min_n / 3)`).
/// Regression splits by the weighted sum of squares (`method = "anova"`),
/// classification by Gini impurity with priors equal to the class frequencies.
/// Predictors must be numeric and complete (bake nominals to dummies and impute
/// first); rpart's categorical splits and surrogates are not supported.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct DecisionTreeSpec {
    engine: Engine,
    cost_complexity: f64,
    tree_depth: usize,
    min_n: usize,
}

impl Default for DecisionTreeSpec {
    fn default() -> Self {
        DecisionTreeSpec {
            engine: Engine::Native,
            cost_complexity: 0.01,
            tree_depth: 30,
            min_n: 20,
        }
    }
}

impl DecisionTreeSpec {
    /// rpart defaults on the native engine.
    pub fn new() -> Self {
        Self::default()
    }

    /// Engine; only `native` implements this model.
    pub fn engine(mut self, engine: Engine) -> Self {
        self.engine = engine;
        self
    }

    /// rpart `cp` (default 0.01).
    pub fn cost_complexity(mut self, cp: f64) -> Self {
        self.cost_complexity = cp;
        self
    }

    /// rpart `maxdepth` (default 30, the maximum rpart allows).
    pub fn tree_depth(mut self, depth: usize) -> Self {
        self.tree_depth = depth;
        self
    }

    /// rpart `minsplit` (default 20); `minbucket` follows as `round(min_n / 3)`.
    pub fn min_n(mut self, n: usize) -> Self {
        self.min_n = n;
        self
    }

    fn control(&self) -> Control {
        Control {
            minsplit: self.min_n,
            minbucket: ((self.min_n as f64) / 3.0).round_ties_even() as usize,
            cp: self.cost_complexity,
            maxdepth: self.tree_depth,
        }
    }

    fn validate(&self) -> Result<()> {
        if self.engine != Engine::Native {
            return Err(Error::Unsupported(format!(
                "decision_tree is implemented on the `native` engine only (got `{}`)",
                self.engine.as_str()
            )));
        }
        if !self.cost_complexity.is_finite() || self.cost_complexity < 0.0 {
            return Err(Error::InvalidArgument(
                "cost_complexity must be finite and >= 0".into(),
            ));
        }
        if self.tree_depth < 1 || self.tree_depth > 30 {
            return Err(Error::InvalidArgument(
                "tree_depth must be between 1 and 30 (rpart's maximum)".into(),
            ));
        }
        if self.min_n < 1 {
            return Err(Error::InvalidArgument("min_n must be >= 1".into()));
        }
        Ok(())
    }
}

impl ModelSpec for DecisionTreeSpec {
    fn name(&self) -> &'static str {
        "decision_tree"
    }

    fn capabilities(&self) -> &'static [Capability] {
        &[
            Capability::Regression,
            Capability::BinaryClassification,
            Capability::MulticlassClassification,
            Capability::ClassProbabilities,
        ]
    }

    fn backend(&self) -> BackendInfo {
        BackendInfo {
            engine: "native".into(),
            crate_version: env!("CARGO_PKG_VERSION").into(),
            solver: "rpart 4.1.23 recursive partitioning port (anova / gini, continuous splits)".into(),
        }
    }

    fn fit(&self, x: &DesignMatrix, y: &Outcome) -> Result<Box<dyn FittedModel>> {
        self.validate()?;
        let n = x.nrows();
        if n != y.len() {
            return Err(Error::InvalidArgument(format!(
                "design matrix has {n} rows but the outcome has {}",
                y.len()
            )));
        }
        if n == 0 {
            return Err(Error::InvalidArgument("cannot fit a tree on zero rows".into()));
        }
        if let Some(i) = x.data().iter().position(|v| !v.is_finite()) {
            return Err(Error::InvalidValue(format!(
                "decision_tree needs finite predictors (non-finite at element {i}); impute first"
            )));
        }
        let wt = vec![1.0; n];
        let (method, yv, levels) = match y {
            Outcome::Numeric(v) => (Method::Anova, v.clone(), None),
            Outcome::Class { codes, levels } => (
                Method::Gini { k: levels.len() },
                codes.iter().map(|c| f64::from(*c + 1)).collect(),
                Some(levels.clone()),
            ),
        };
        let tree = rpart::fit(x.data(), n, x.ncols(), &yv, &wt, method, self.control());
        let n_splits = tree.n_splits();
        Ok(Box::new(FittedDecisionTree {
            model: "decision_tree".into(),
            format_version: crate::fitted::MODEL_FORMAT_VERSION,
            tree,
            names: x.names().to_vec(),
            levels,
            backend: self.backend(),
            diagnostics: FitDiagnostics {
                note: Some(format!("{n_splits} splits")),
                ..Default::default()
            },
        }))
    }

    fn tunable(&self) -> &'static [&'static str] {
        &["cost_complexity", "tree_depth", "min_n"]
    }

    fn with_param(&self, name: &str, value: f64) -> Result<Box<dyn ModelSpec>> {
        let s = match name {
            "cost_complexity" => self.clone().cost_complexity(value),
            "tree_depth" => self.clone().tree_depth(value.round() as usize),
            "min_n" => self.clone().min_n(value.round() as usize),
            other => {
                return Err(Error::Unsupported(format!(
                    "decision_tree has no tunable parameter `{other}`"
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

/// A fitted rpart-style tree.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct FittedDecisionTree {
    model: String,
    format_version: u32,
    /// The tree.
    pub tree: Tree,
    names: Vec<String>,
    /// Outcome levels for classification, `None` for regression.
    pub levels: Option<Vec<String>>,
    backend: BackendInfo,
    diagnostics: FitDiagnostics,
}

impl FittedDecisionTree {
    /// Feature name of a split variable.
    pub fn var_name(&self, var: usize) -> &str {
        &self.names[var]
    }

    /// rpart's `yprob` (rpart.R, method "class"), in R's arithmetic: with
    /// `prior = counts / sum(counts)` over the weighted root counts,
    /// `v = prior / pmax(1, counts)`, `t = node_counts * v` and `yprob = t / rowSums(t)`.
    /// Mathematically this is the node's class share, but R's rounding gives the same
    /// fraction different last bits in different nodes, which matters for tie-sensitive
    /// metrics (ROC/PR curves) compared against R.
    fn yprob(&self, node_counts: &[f64]) -> Vec<f64> {
        let root = match &self.tree.root.value {
            NodeValue::Class { counts, .. } => counts,
            _ => unreachable!(),
        };
        let total: f64 = root.iter().sum();
        let t: Vec<f64> = node_counts
            .iter()
            .zip(root)
            .map(|(n, c)| {
                let prior = c / total;
                let v = prior / c.max(1.0);
                n * v
            })
            .collect();
        // rowSums accumulates in extended precision and rounds once; an error-free
        // transformation of the double sum gives the same result except in
        // double-rounding corner cases.
        let (mut hi, mut lo) = (0.0f64, 0.0f64);
        for x in &t {
            let s = hi + x;
            let bb = s - hi;
            let err = (hi - (s - bb)) + (x - bb);
            hi = s;
            lo += err;
        }
        let rs = hi + lo;
        t.iter().map(|x| x / rs).collect()
    }
}

impl FittedModel for FittedDecisionTree {
    fn predict(&self, x: &DesignMatrix) -> Result<Prediction> {
        check_features(&self.names, x)?;
        let n = x.nrows();
        match &self.levels {
            None => Ok(Prediction::Numeric(
                (0..n)
                    .map(|i| match self.tree.leaf_for(|v| x.get(i, v)).value {
                        NodeValue::Mean(m) => m,
                        _ => unreachable!(),
                    })
                    .collect(),
            )),
            Some(levels) => {
                let k = levels.len();
                let mut codes = Vec::with_capacity(n);
                let mut probs = vec![0.0; n * k];
                for i in 0..n {
                    match &self.tree.leaf_for(|v| x.get(i, v)).value {
                        NodeValue::Class { class, counts } => {
                            codes.push((*class - 1) as u32);
                            for (j, p) in self.yprob(counts).into_iter().enumerate() {
                                probs[j * n + i] = p;
                            }
                        }
                        _ => unreachable!(),
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
