//! Workflow sets: workflowsets 1.1.1 semantics (ADR-0016).
//!
//! [`workflow_set`] crosses (or pairs) preprocessors with models into one workflow per
//! combination, [`WorkflowSet::workflow_map`] evaluates every workflow on the same
//! resamples (`tune_grid` for workflows that carry a grid, `fit_resamples` otherwise),
//! and [`WorkflowSet::collect_metrics`], [`WorkflowSet::rank_results`] and
//! [`WorkflowSet::fit_best`] summarise, rank and refit as workflowsets does.
//!
//! Ported from the package sources (`workflow_set`, `cross_objects`, `fuse_objects`,
//! `fix_list_names`, `workflow_map`, `collect_metrics.workflow_set`, `rank_results`,
//! `pick_metric`, `fit_best.workflow_set`); where R draws random numbers (tie ranks
//! under `withr::with_seed(1)`) this crate is deterministic and documents the
//! difference.

#![forbid(unsafe_code)]
#![warn(missing_docs, rust_2018_idioms)]

use serde::{Deserialize, Serialize};
use tidyrustmodels_core::{Error, ModelSpec, Parallelism, Result, Table};
use tidyrustmodels_metrics::{Direction, Metric, MetricSet};
use tidyrustmodels_recipes::Recipe;
use tidyrustmodels_resample::Folds;
use tidyrustmodels_tune::{fit_best as tune_fit_best, tune_grid, Grid, ParamSet, TuneResults};
use tidyrustmodels_workflows::{FittedWorkflow, ResampleResults, Workflow};

/// What `workflow_map` stored for one workflow.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub enum WorkflowResult {
    /// `fit_resamples()` (tune's `resample_results`: one candidate, `.config`
    /// `pre0_mod0_post0`).
    Resamples(ResampleResults),
    /// `tune_grid()` over the workflow's grid.
    Tuned(TuneResults),
}

impl WorkflowResult {
    /// `(.config label, parameters, results)` per distinct candidate in `.config`
    /// order (tune's `collect_metrics()` collapses identical grid rows).
    fn candidates(&self) -> Vec<(String, ParamSet, &ResampleResults)> {
        match self {
            WorkflowResult::Resamples(r) => vec![("pre0_mod0_post0".to_string(), Vec::new(), r)],
            WorkflowResult::Tuned(t) => {
                let labels = t.config_labels();
                let mut rows: Vec<(String, ParamSet, &ResampleResults)> = Vec::new();
                for ((p, r), label) in t.candidates.iter().zip(labels) {
                    if !rows.iter().any(|(l, _, _)| *l == label) {
                        rows.push((label, p.clone(), r));
                    }
                }
                // labels are zero-padded, so lexical order is numeric order
                rows.sort_by(|a, b| a.0.cmp(&b.0));
                rows
            }
        }
    }
}

/// One workflow of the set (one row of the workflowsets tibble).
#[derive(Clone, Debug)]
pub struct WorkflowSetEntry {
    /// `wflow_id`: `"{preproc name}_{model name}"`.
    pub id: String,
    /// `info$preproc`: the preprocessor type (always `"recipe"` here).
    pub preproc: String,
    /// `info$model`: the model class (`"linear_reg"`, `"decision_tree"`, ...).
    pub model: String,
    /// The workflow.
    pub workflow: Workflow,
    /// `option_add(grid = ...)`: the grid `workflow_map` tunes over, if any.
    pub grid: Option<Grid>,
    /// What `workflow_map` produced.
    pub result: Option<WorkflowResult>,
}

/// A workflow set.
#[derive(Clone, Debug)]
pub struct WorkflowSet {
    /// The workflows in workflowsets' order.
    pub entries: Vec<WorkflowSetEntry>,
    /// The metric set `workflow_map` evaluated (`None` before mapping).
    pub metrics: Option<MetricSet>,
}

/// One row of `collect_metrics(workflow_set)`.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct CollectedMetric {
    /// `wflow_id`.
    pub wflow_id: String,
    /// `.config` (`pre0_mod0_post0` for a resampled workflow, tune's labels otherwise).
    pub config: String,
    /// `preproc` type.
    pub preproc: String,
    /// `model` class.
    pub model: String,
    /// `.metric`.
    pub metric: String,
    /// `mean` over the resamples where the metric was defined.
    pub mean: Option<f64>,
    /// `std_err`.
    pub std_err: Option<f64>,
    /// `n`.
    pub n: usize,
}

/// One row of `rank_results()`.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct RankRow {
    /// `wflow_id`.
    pub wflow_id: String,
    /// `.config`.
    pub config: String,
    /// `.metric`.
    pub metric: String,
    /// `mean`.
    pub mean: Option<f64>,
    /// `std_err`.
    pub std_err: Option<f64>,
    /// `n`.
    pub n: usize,
    /// `preprocessor` type.
    pub preprocessor: String,
    /// `model` class.
    pub model: String,
    /// `rank` on the ranking metric (`None` when that metric is undefined, as R's
    /// `rank()` keeps `NA`).
    pub rank: Option<usize>,
}

/// workflowsets' `fix_list_names()`: unnamed elements get their class name, made
/// unique as `vctrs::vec_as_names(repair = "unique")` does (every element of a
/// duplicated class gets `_{position}`), e.g. two unnamed recipes become `recipe_1`
/// and `recipe_2` while a single unnamed `linear_reg` stays `linear_reg`. Given
/// names are kept as they are.
pub fn fix_list_names(classes: &[&str], given: &[Option<String>]) -> Vec<String> {
    let repaired: Vec<String> = classes
        .iter()
        .enumerate()
        .map(|(i, c)| {
            if classes.iter().filter(|d| d == &c).count() > 1 {
                format!("{c}_{}", i + 1)
            } else {
                (*c).to_string()
            }
        })
        .collect();
    repaired
        .into_iter()
        .zip(given)
        .map(|(r, g)| match g {
            Some(name) if !name.is_empty() => name.clone(),
            _ => r,
        })
        .collect()
}

/// `workflowsets::workflow_set(preproc, models, cross)`: with `cross = true` every
/// preprocessor is combined with every model (preprocessors in the given order
/// outermost, models innermost, as `tidyr::crossing` lays them out); with
/// `cross = false` the lists are paired element-wise (a list of length one is still
/// crossed, `fuse_objects`). Ids are `"{preproc}_{model}"` with names from
/// [`fix_list_names`] (`None` = unnamed) and must be unique.
///
/// Not reproduced: `tidyr::crossing` de-duplicates *identical* preprocessor or model
/// objects; here every element is kept.
pub fn workflow_set(
    preproc: Vec<(Option<String>, Recipe)>,
    models: Vec<(Option<String>, Box<dyn ModelSpec>)>,
    cross: bool,
) -> Result<WorkflowSet> {
    if preproc.is_empty() || models.is_empty() {
        return Err(Error::InvalidArgument(
            "a workflow set needs at least one preprocessor and one model".into(),
        ));
    }
    if preproc.len() != models.len() && preproc.len() != 1 && models.len() != 1 && !cross {
        return Err(Error::InvalidArgument(
            "the lengths of `preproc` and `models` are different and `cross = false`".into(),
        ));
    }
    let pp_names = fix_list_names(
        &vec!["recipe"; preproc.len()],
        &preproc.iter().map(|(n, _)| n.clone()).collect::<Vec<_>>(),
    );
    let model_classes: Vec<&str> = models.iter().map(|(_, m)| m.name()).collect();
    let mod_names = fix_list_names(
        &model_classes,
        &models.iter().map(|(n, _)| n.clone()).collect::<Vec<_>>(),
    );
    let pairs: Vec<(usize, usize)> = if cross || preproc.len() == 1 || models.len() == 1 {
        (0..preproc.len())
            .flat_map(|p| (0..models.len()).map(move |m| (p, m)))
            .collect()
    } else {
        (0..preproc.len()).map(|i| (i, i)).collect()
    };
    let mut entries = Vec::with_capacity(pairs.len());
    for (p, m) in pairs {
        let id = format!("{}_{}", pp_names[p], mod_names[m]);
        if entries.iter().any(|e: &WorkflowSetEntry| e.id == id) {
            return Err(Error::InvalidArgument(format!(
                "workflow ids must be unique (`{id}` appears twice)"
            )));
        }
        entries.push(WorkflowSetEntry {
            id,
            preproc: "recipe".to_string(),
            model: models[m].1.name().to_string(),
            workflow: Workflow::from_boxed(preproc[p].1.clone(), models[m].1.clone_box()),
            grid: None,
            result: None,
        });
    }
    Ok(WorkflowSet {
        entries,
        metrics: None,
    })
}

impl WorkflowSet {
    /// The `wflow_id`s in order.
    pub fn ids(&self) -> Vec<&str> {
        self.entries.iter().map(|e| e.id.as_str()).collect()
    }

    /// The entry with this id.
    pub fn entry(&self, id: &str) -> Result<&WorkflowSetEntry> {
        self.entries
            .iter()
            .find(|e| e.id == id)
            .ok_or_else(|| Error::InvalidArgument(format!("no workflow with id `{id}` in the set")))
    }

    /// `extract_workflow(x, id)`.
    pub fn extract_workflow(&self, id: &str) -> Result<&Workflow> {
        Ok(&self.entry(id)?.workflow)
    }

    /// `extract_workflow_set_result(x, id)`: an error before `workflow_map`.
    pub fn extract_result(&self, id: &str) -> Result<&WorkflowResult> {
        self.entry(id)?.result.as_ref().ok_or_else(|| {
            Error::InvalidArgument(format!(
                "workflow `{id}` has no result yet; run `workflow_map` first"
            ))
        })
    }

    /// `option_add(x, grid = grid, id = id)`: the grid `workflow_map` tunes this
    /// workflow over. Any earlier result of the workflow is discarded.
    pub fn option_grid(&mut self, id: &str, grid: Grid) -> Result<()> {
        let e = self
            .entries
            .iter_mut()
            .find(|e| e.id == id)
            .ok_or_else(|| Error::InvalidArgument(format!("no workflow with id `{id}` in the set")))?;
        e.grid = Some(grid);
        e.result = None;
        Ok(())
    }

    /// `workflow_map(x, "tune_grid", resamples, metrics)`: every workflow with a grid
    /// is tuned over it (`tune_grid`), every other one is resampled
    /// (`fit_resamples`, which is what tune does for a workflow without tuning
    /// parameters, with a warning). All workflows see the same `folds`, as
    /// workflowsets requires.
    pub fn workflow_map(
        &mut self,
        data: &Table,
        folds: &Folds,
        metrics: &MetricSet,
        par: Parallelism,
    ) -> Result<()> {
        for e in &mut self.entries {
            e.result = Some(match &e.grid {
                Some(grid) => WorkflowResult::Tuned(tune_grid(&e.workflow, data, folds, grid, metrics, par)?),
                None => WorkflowResult::Resamples(e.workflow.fit_resamples(data, folds, metrics, par)?),
            });
        }
        self.metrics = Some(metrics.clone());
        Ok(())
    }

    fn check_complete(&self) -> Result<&MetricSet> {
        if let Some(e) = self.entries.iter().find(|e| e.result.is_none()) {
            return Err(Error::InvalidArgument(format!(
                "workflow `{}` has no result; run `workflow_map` on the whole set first",
                e.id
            )));
        }
        self.metrics
            .as_ref()
            .ok_or_else(|| Error::InvalidArgument("run `workflow_map` first".into()))
    }

    /// `collect_metrics(x)`: every workflow's summarised metrics, bound with
    /// `wflow_id`, `preproc` and `model`; rows in set order, then `.config` order,
    /// then metric name order (tune's `collect_metrics()` groups by metric name).
    pub fn collect_metrics(&self) -> Result<Vec<CollectedMetric>> {
        self.check_complete()?;
        let mut rows = Vec::new();
        for e in &self.entries {
            for (config, _, r) in e.result.as_ref().expect("complete").candidates() {
                let mut summaries: Vec<_> = r.summary.iter().collect();
                summaries.sort_by(|a, b| a.name.cmp(&b.name));
                for s in summaries {
                    rows.push(CollectedMetric {
                        wflow_id: e.id.clone(),
                        config: config.clone(),
                        preproc: e.preproc.clone(),
                        model: e.model.clone(),
                        metric: s.name.clone(),
                        mean: s.mean,
                        std_err: s.std_err,
                        n: s.n,
                    });
                }
            }
        }
        Ok(rows)
    }

    /// `pick_metric()` with `rank_metric = NULL`: the first metric of the set.
    fn pick_metric(&self, metric: Option<Metric>) -> Result<Metric> {
        let set = self.check_complete()?;
        match metric {
            Some(m) => {
                if !set.metrics().contains(&m) {
                    return Err(Error::InvalidArgument(format!(
                        "metric `{}` was not in the results",
                        m.name()
                    )));
                }
                Ok(m)
            }
            None => set
                .metrics()
                .first()
                .copied()
                .ok_or_else(|| Error::InvalidArgument("the metric set is empty".into())),
        }
    }

    /// `rank_results(x, rank_metric, select_best)`: every `(wflow_id, .config)` ranked
    /// on `metric` (the set's first metric when `None`; larger-is-better metrics are
    /// negated first), all metrics of each candidate reported, rows arranged by rank.
    /// With `select_best` only each workflow's best candidate is kept
    /// (`slice_min(with_ties = FALSE)`: the first smallest). An undefined mean ranks
    /// `None` and sorts last (R's `rank()` keeps `NA`).
    ///
    /// Difference from R: workflowsets breaks ties with `rank(ties.method =
    /// "random")` under `withr::with_seed(1)`; here tied candidates keep their order
    /// of appearance (set order, then `.config` order).
    pub fn rank_results(&self, metric: Option<Metric>, select_best: bool) -> Result<Vec<RankRow>> {
        let metric = self.pick_metric(metric)?;
        let results = self.collect_metrics()?;
        let maximize = metric.direction() == Direction::Maximize;
        // (wflow_id, config, mean to minimise)
        let mut ranked: Vec<(String, String, Option<f64>)> = results
            .iter()
            .filter(|r| r.metric == metric.name())
            .map(|r| {
                (
                    r.wflow_id.clone(),
                    r.config.clone(),
                    r.mean.map(|m| if maximize { -m } else { m }),
                )
            })
            .collect();
        if select_best {
            let mut kept: Vec<(String, String, Option<f64>)> = Vec::new();
            for e in &self.entries {
                let rows: Vec<&(String, String, Option<f64>)> =
                    ranked.iter().filter(|r| r.0 == e.id).collect();
                // the first row with the smallest defined mean; all undefined keeps the first
                let best = rows
                    .iter()
                    .filter(|r| r.2.is_some())
                    .min_by(|a, b| a.2.partial_cmp(&b.2).expect("defined means"))
                    .or(rows.first())
                    .copied();
                if let Some(b) = best {
                    kept.push(b.clone());
                }
            }
            ranked = kept;
        }
        // rank(mean): defined means in increasing order, ties by appearance, NA kept
        let mut order: Vec<usize> = (0..ranked.len()).filter(|&i| ranked[i].2.is_some()).collect();
        order.sort_by(|&a, &b| ranked[a].2.partial_cmp(&ranked[b].2).expect("defined means"));
        let mut rank: Vec<Option<usize>> = vec![None; ranked.len()];
        for (pos, &i) in order.iter().enumerate() {
            rank[i] = Some(pos + 1);
        }
        let mut rows: Vec<RankRow> = results
            .iter()
            .filter_map(|r| {
                let k = ranked
                    .iter()
                    .position(|(w, c, _)| *w == r.wflow_id && *c == r.config)?;
                Some(RankRow {
                    wflow_id: r.wflow_id.clone(),
                    config: r.config.clone(),
                    metric: r.metric.clone(),
                    mean: r.mean,
                    std_err: r.std_err,
                    n: r.n,
                    preprocessor: r.preproc.clone(),
                    model: r.model.clone(),
                    rank: rank[k],
                })
            })
            .collect();
        // arrange(rank): stable, NA last
        rows.sort_by_key(|r| r.rank.unwrap_or(usize::MAX));
        Ok(rows)
    }

    /// The workflow `fit_best` refits and the parameters it uses: the top workflow of
    /// `rank_results(select_best = true)` and `select_best()` of its result (empty for
    /// a resampled workflow).
    pub fn best(&self, metric: Option<Metric>) -> Result<(String, ParamSet)> {
        let metric = self.pick_metric(metric)?;
        let rankings = self.rank_results(Some(metric), true)?;
        let top = rankings
            .first()
            .ok_or_else(|| Error::InvalidArgument("no results to rank".into()))?;
        let params = match self.extract_result(&top.wflow_id)? {
            WorkflowResult::Resamples(_) => Vec::new(),
            WorkflowResult::Tuned(t) => t.select_best(metric)?.params,
        };
        Ok((top.wflow_id.clone(), params))
    }

    /// `fit_best(x, metric)`: the best workflow by `rank_results(select_best = TRUE)`,
    /// finalized with its `select_best()` parameters when it was tuned, fitted on all
    /// of `data` with `Workflow::fit` (`parsnip::fit()`, so workflows' sparse-data
    /// decision applies).
    pub fn fit_best(&self, data: &Table, metric: Option<Metric>) -> Result<FittedWorkflow> {
        let metric = self.pick_metric(metric)?;
        let (id, _) = self.best(Some(metric))?;
        let e = self.entry(&id)?;
        match e.result.as_ref().expect("complete") {
            WorkflowResult::Resamples(_) => e.workflow.fit(data),
            WorkflowResult::Tuned(t) => tune_fit_best(t, &e.workflow, data, metric),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tidyrustmodels_core::Column;
    use tidyrustmodels_models::{DecisionTreeSpec, LinearRegSpec};
    use tidyrustmodels_resample::vfold_cv;
    use tidyrustmodels_tune::ParamRange;

    fn data() -> Table {
        let n = 40;
        let x: Vec<f64> = (0..n).map(|i| i as f64 / 4.0).collect();
        let z: Vec<f64> = (0..n).map(|i| ((i * 7) % 11) as f64).collect();
        let y: Vec<f64> = x.iter().zip(&z).map(|(a, b)| 1.0 + 2.0 * a - 0.5 * b).collect();
        Table::new(vec![
            ("y", Column::numeric(y).unwrap()),
            ("x", Column::numeric(x).unwrap()),
            ("z", Column::numeric(z).unwrap()),
        ])
        .unwrap()
    }

    fn lm() -> Box<dyn ModelSpec> {
        Box::new(LinearRegSpec::new())
    }

    fn tree() -> Box<dyn ModelSpec> {
        Box::new(DecisionTreeSpec::new())
    }

    #[test]
    fn ids_follow_workflowsets_order_and_default_names() {
        // named: preprocessors outermost in the given order, models innermost
        let s = workflow_set(
            vec![
                (Some("zeta".into()), Recipe::new("y")),
                (Some("alpha".into()), Recipe::new("y")),
            ],
            vec![(Some("lm".into()), lm()), (Some("tree".into()), tree())],
            true,
        )
        .unwrap();
        assert_eq!(s.ids(), ["zeta_lm", "zeta_tree", "alpha_lm", "alpha_tree"]);
        assert_eq!(s.entries[1].model, "decision_tree");
        assert_eq!(s.entries[1].preproc, "recipe");
        // unnamed: class names, duplicated classes numbered by position
        let s = workflow_set(
            vec![(None, Recipe::new("y")), (None, Recipe::new("y"))],
            vec![(None, lm()), (None, tree()), (None, lm())],
            true,
        )
        .unwrap();
        assert_eq!(
            s.ids(),
            [
                "recipe_1_linear_reg_1",
                "recipe_1_decision_tree",
                "recipe_1_linear_reg_3",
                "recipe_2_linear_reg_1",
                "recipe_2_decision_tree",
                "recipe_2_linear_reg_3"
            ]
        );
        // cross = false pairs element-wise, unless one side has length one
        let s = workflow_set(
            vec![
                (Some("a".into()), Recipe::new("y")),
                (Some("b".into()), Recipe::new("y")),
            ],
            vec![(Some("lm".into()), lm()), (Some("tree".into()), tree())],
            false,
        )
        .unwrap();
        assert_eq!(s.ids(), ["a_lm", "b_tree"]);
        let s = workflow_set(
            vec![(Some("a".into()), Recipe::new("y"))],
            vec![(Some("lm".into()), lm()), (Some("tree".into()), tree())],
            false,
        )
        .unwrap();
        assert_eq!(s.ids(), ["a_lm", "a_tree"]);
        assert!(workflow_set(
            vec![(Some("a".into()), Recipe::new("y")); 2],
            vec![(Some("lm".into()), lm()); 3],
            false
        )
        .is_err());
        assert!(workflow_set(
            vec![(Some("a".into()), Recipe::new("y")); 2],
            vec![(Some("lm".into()), lm())],
            true
        )
        .is_err());
    }

    #[test]
    fn map_collect_rank_and_fit_best_follow_workflowsets() {
        let d = data();
        let folds = vfold_cv(d.nrows(), 4, None, tidyrustmodels_core::Seed(3)).unwrap();
        let metrics = MetricSet::from_names(&["rmse", "rsq"]).unwrap();
        let mut s = workflow_set(
            vec![(Some("rec".into()), Recipe::new("y"))],
            vec![(Some("lm".into()), lm()), (Some("tree".into()), tree())],
            true,
        )
        .unwrap();
        assert!(s.collect_metrics().is_err());
        let grid = tidyrustmodels_tune::grid_regular(&[ParamRange::linear("min_n", 2.0, 10.0)], 3).unwrap();
        s.option_grid("rec_tree", grid).unwrap();
        assert!(s.option_grid("nope", Grid { points: vec![] }).is_err());
        s.workflow_map(&d, &folds, &metrics, Parallelism::SERIAL).unwrap();
        let cm = s.collect_metrics().unwrap();
        // lm: one config with two metrics; tree: three configs with two metrics each
        assert_eq!(cm.len(), 2 + 6);
        assert_eq!(cm[0].config, "pre0_mod0_post0");
        assert_eq!(cm[0].metric, "rmse");
        assert_eq!(cm[1].metric, "rsq");
        assert_eq!(cm[2].config, "pre0_mod1_post0");
        assert_eq!(cm[6].config, "pre0_mod3_post0");
        assert!(cm.iter().all(|r| r.n == 4));
        // the linear model is exact on linear data: rank 1 on rmse (minimised) and on
        // rsq (maximised: negated means), every metric row carries the rank
        let rr = s.rank_results(Some(Metric::Rmse), false).unwrap();
        assert_eq!(rr.len(), 8);
        assert_eq!((rr[0].wflow_id.as_str(), rr[0].rank), ("rec_lm", Some(1)));
        assert_eq!(rr[1].rank, Some(1));
        assert!(rr[2..].iter().all(|r| r.wflow_id == "rec_tree"));
        let ranks: Vec<usize> = rr.iter().filter_map(|r| r.rank).collect();
        assert_eq!(ranks, [1, 1, 2, 2, 3, 3, 4, 4]);
        let rr_rsq = s.rank_results(Some(Metric::Rsq), false).unwrap();
        assert_eq!(rr_rsq[0].wflow_id, "rec_lm");
        // default metric = the first of the set
        assert_eq!(s.rank_results(None, false).unwrap(), rr);
        // select_best keeps one config per workflow
        let best = s.rank_results(None, true).unwrap();
        assert_eq!(best.len(), 4);
        assert_eq!(best[2].wflow_id, "rec_tree");
        assert_eq!(best[2].rank, Some(2));
        let tree_best_config = best[2].config.clone();
        let tree_rows: Vec<&RankRow> = rr
            .iter()
            .filter(|r| r.wflow_id == "rec_tree" && r.metric == "rmse")
            .collect();
        assert_eq!(tree_rows[0].config, tree_best_config);
        // fit_best refits the top workflow on all the data
        let (id, params) = s.best(None).unwrap();
        assert_eq!(id, "rec_lm");
        assert!(params.is_empty());
        let f = s.fit_best(&d, None).unwrap();
        let p = f.predict(&d).unwrap();
        let y = d.column("y").unwrap().as_numeric().unwrap();
        if let tidyrustmodels_core::Prediction::Numeric(p) = p {
            assert!(p.iter().zip(y).all(|(a, b)| (a - b).abs() < 1e-8));
        } else {
            panic!("numeric");
        }
        assert!(s.rank_results(Some(Metric::Mae), false).is_err());
        assert_eq!(s.extract_workflow("rec_lm").unwrap().model().name(), "linear_reg");
        assert!(matches!(
            s.extract_result("rec_tree").unwrap(),
            WorkflowResult::Tuned(_)
        ));
    }

    #[test]
    fn ties_keep_order_of_appearance() {
        let d = data();
        let folds = vfold_cv(d.nrows(), 3, None, tidyrustmodels_core::Seed(1)).unwrap();
        let metrics = MetricSet::from_names(&["rmse"]).unwrap();
        let mut s = workflow_set(
            vec![
                (Some("b".into()), Recipe::new("y")),
                (Some("a".into()), Recipe::new("y")),
            ],
            vec![(Some("lm".into()), lm())],
            true,
        )
        .unwrap();
        s.workflow_map(&d, &folds, &metrics, Parallelism::SERIAL).unwrap();
        let rr = s.rank_results(None, false).unwrap();
        assert_eq!(rr[0].wflow_id, "b_lm");
        assert_eq!(rr[0].rank, Some(1));
        assert_eq!(rr[1].rank, Some(2));
        assert_eq!(rr[0].mean, rr[1].mean);
    }
}
