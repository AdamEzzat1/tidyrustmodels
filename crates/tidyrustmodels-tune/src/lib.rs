//! Hyperparameter tuning over resamples: regular and random grids, `tune_grid`.

#![forbid(unsafe_code)]
#![warn(missing_docs, rust_2018_idioms)]

use rand::RngExt;
use serde::{Deserialize, Serialize};
use tidyrustmodels_core::{Error, Parallelism, Result, Seed, Table};
use tidyrustmodels_metrics::{Direction, Metric, MetricSet};
use tidyrustmodels_resample::Folds;
use tidyrustmodels_workflows::{parallel_map, FittedWorkflow, FoldMetrics, ResampleResults, Workflow};

/// One candidate: `(parameter name, value)` pairs.
pub type ParamSet = Vec<(String, f64)>;

/// A parameter range, optionally on a log10 scale like dials' `penalty()`.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct ParamRange {
    /// Parameter name understood by `ModelSpec::with_param`.
    pub name: String,
    /// Lower bound (in log10 units when `log10` is set).
    pub lower: f64,
    /// Upper bound (in log10 units when `log10` is set).
    pub upper: f64,
    /// Whether bounds and grid spacing are on the log10 scale.
    pub log10: bool,
}

impl ParamRange {
    /// A linear range.
    pub fn linear(name: &str, lower: f64, upper: f64) -> Self {
        ParamRange {
            name: name.into(),
            lower,
            upper,
            log10: false,
        }
    }

    /// A log10 range (`dials::penalty()` uses `(-10, 0)`).
    pub fn log10(name: &str, lower: f64, upper: f64) -> Self {
        ParamRange {
            name: name.into(),
            lower,
            upper,
            log10: true,
        }
    }

    fn value(&self, t: f64) -> f64 {
        let raw = self.lower + t * (self.upper - self.lower);
        if self.log10 {
            10f64.powf(raw)
        } else {
            raw
        }
    }

    fn validate(&self) -> Result<()> {
        if !(self.lower.is_finite() && self.upper.is_finite()) || self.lower > self.upper {
            return Err(Error::InvalidArgument(format!(
                "invalid range for `{}`",
                self.name
            )));
        }
        Ok(())
    }
}

/// A set of candidates.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Grid {
    /// Candidates in evaluation order.
    pub points: Vec<ParamSet>,
}

/// `dials::grid_regular`: `levels` evenly spaced values per parameter, full
/// Cartesian product, first parameter varying fastest.
pub fn grid_regular(ranges: &[ParamRange], levels: usize) -> Result<Grid> {
    if ranges.is_empty() || levels == 0 {
        return Err(Error::InvalidArgument(
            "grid_regular needs ranges and levels >= 1".into(),
        ));
    }
    for r in ranges {
        r.validate()?;
    }
    let per_param: Vec<Vec<f64>> = ranges
        .iter()
        .map(|r| {
            (0..levels)
                .map(|i| {
                    if levels == 1 {
                        0.0
                    } else {
                        i as f64 / (levels as f64 - 1.0)
                    }
                })
                .map(|t| r.value(t))
                .collect()
        })
        .collect();
    let total: usize = per_param.iter().map(Vec::len).product();
    let mut points = Vec::with_capacity(total);
    for idx in 0..total {
        let mut rem = idx;
        let mut p = Vec::with_capacity(ranges.len());
        for (r, vals) in ranges.iter().zip(&per_param) {
            p.push((r.name.clone(), vals[rem % vals.len()]));
            rem /= vals.len();
        }
        points.push(p);
    }
    Ok(Grid { points })
}

/// `dials::grid_random`: `size` candidates drawn uniformly (on each range's scale).
pub fn grid_random(ranges: &[ParamRange], size: usize, seed: Seed) -> Result<Grid> {
    if ranges.is_empty() || size == 0 {
        return Err(Error::InvalidArgument(
            "grid_random needs ranges and size >= 1".into(),
        ));
    }
    for r in ranges {
        r.validate()?;
    }
    let mut rng = seed.rng();
    let points = (0..size)
        .map(|_| {
            ranges
                .iter()
                .map(|r| (r.name.clone(), r.value(rng.random::<f64>())))
                .collect()
        })
        .collect();
    Ok(Grid { points })
}

/// Results for every candidate.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct TuneResults {
    /// `(candidate, resampling results)` in grid order.
    pub candidates: Vec<(ParamSet, ResampleResults)>,
    /// The model's tunable parameters in parsnip's argument order
    /// (`ModelSpec::tunable`), which fixes tune's `.config` numbering.
    pub param_order: Vec<String>,
}

/// One row of `show_best()`: a candidate (`index` into the grid, `config` as tune
/// numbers it) with its summary for the chosen metric.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct BestRow {
    /// Position of the candidate in the grid.
    pub index: usize,
    /// tune's `.config` number (1-based; `pre0_mod3_post0` is 3): the rank of the
    /// candidate's parameters sorted in `param_order`, identical candidates sharing it.
    pub config: usize,
    /// The candidate's parameters.
    pub params: ParamSet,
    /// Mean over the resamples where the metric was defined (`None` when it never was).
    pub mean: Option<f64>,
    /// Standard error over those resamples (`None` with fewer than two).
    pub std_err: Option<f64>,
    /// Number of resamples where the metric was defined.
    pub n: usize,
}

/// A selected candidate (`select_best()` and friends return the parameters and the
/// `.config`; here the grid position stands for the config).
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Selection {
    /// Position of the candidate in the grid.
    pub index: usize,
    /// tune's `.config` number (see [`BestRow::config`]).
    pub config: usize,
    /// Its parameters.
    pub params: ParamSet,
}

/// One `dplyr::arrange()` key for the `...` of `select_by_one_std_err()` /
/// `select_by_pct_loss()`: a parameter name, ascending or `desc()`. The keys order the
/// candidates from simplest to most complex before the first one is taken.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct SortKey {
    /// Parameter name.
    pub param: String,
    /// `desc(param)` when true.
    pub descending: bool,
}

impl SortKey {
    /// `arrange(param)`.
    pub fn asc(param: &str) -> Self {
        SortKey {
            param: param.into(),
            descending: false,
        }
    }

    /// `arrange(desc(param))`.
    pub fn desc(param: &str) -> Self {
        SortKey {
            param: param.into(),
            descending: true,
        }
    }

    /// Parse `"param"` or `"desc(param)"`.
    pub fn parse(text: &str) -> Result<Self> {
        let t = text.trim();
        if let Some(inner) = t.strip_prefix("desc(").and_then(|r| r.strip_suffix(')')) {
            return Ok(SortKey::desc(inner.trim()));
        }
        if t.is_empty() || t.contains('(') {
            return Err(Error::InvalidArgument(format!(
                "sort key must be `param` or `desc(param)`, got {text:?}"
            )));
        }
        Ok(SortKey::asc(t))
    }
}

impl TuneResults {
    /// tune's `.config` number of every candidate (`make_config_labs`: the distinct
    /// parameter rows arranged by the parameters in `param_order`, numbered from 1).
    pub fn configs(&self) -> Vec<usize> {
        let key = |p: &ParamSet| -> Vec<f64> {
            let mut k: Vec<f64> = self
                .param_order
                .iter()
                .filter_map(|name| p.iter().find(|(n, _)| n == name).map(|(_, v)| *v))
                .collect();
            // parameters outside the known order (none for a validated grid) follow,
            // in name order, so the key is still total
            let mut rest: Vec<(&String, &f64)> = p
                .iter()
                .filter(|(n, _)| !self.param_order.contains(n))
                .map(|(n, v)| (n, v))
                .collect();
            rest.sort_by(|a, b| a.0.cmp(b.0));
            k.extend(rest.into_iter().map(|(_, v)| *v));
            k
        };
        let mut distinct: Vec<Vec<f64>> = self.candidates.iter().map(|(p, _)| key(p)).collect();
        distinct.sort_by(|a, b| {
            a.iter()
                .zip(b)
                .map(|(x, y)| x.total_cmp(y))
                .find(|o| o.is_ne())
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        distinct.dedup();
        self.candidates
            .iter()
            .map(|(p, _)| distinct.iter().position(|d| *d == key(p)).expect("present") + 1)
            .collect()
    }

    /// `tune::collect_metrics()` for one metric: every candidate's summary in
    /// `.config` order (ties in grid order), as [`BestRow`]s.
    fn rows_for(&self, metric: Metric) -> Vec<BestRow> {
        let configs = self.configs();
        let mut rows: Vec<BestRow> = self
            .candidates
            .iter()
            .enumerate()
            .map(|(index, (p, r))| {
                let s = r.metric(metric.name());
                BestRow {
                    index,
                    config: configs[index],
                    params: p.clone(),
                    mean: s.and_then(|s| s.mean),
                    std_err: s.and_then(|s| s.std_err),
                    n: s.map_or(0, |s| s.n),
                }
            })
            .collect();
        rows.sort_by_key(|r| r.config);
        rows
    }

    fn check_metric(&self, metric: Metric) -> Result<()> {
        let known = self
            .candidates
            .first()
            .map(|(_, r)| r.summary.iter().any(|s| s.name == metric.name()))
            .unwrap_or(false);
        if !known {
            return Err(Error::InvalidArgument(format!(
                "`{}` was not in the metric set",
                metric.name()
            )));
        }
        Ok(())
    }

    /// `tune::show_best()`: the top `n` candidates for `metric`, best first; ties keep
    /// grid order (`dplyr::arrange` is stable) and candidates whose metric is undefined
    /// on every resample come last (`NA` sorts last).
    pub fn show_best(&self, metric: Metric, n: usize) -> Vec<BestRow> {
        let mut rows = self.rows_for(metric);
        rows.sort_by(|a, b| match (a.mean, b.mean) {
            (Some(x), Some(y)) => match metric.direction() {
                Direction::Maximize => y.total_cmp(&x),
                Direction::Minimize => x.total_cmp(&y),
            },
            (Some(_), None) => std::cmp::Ordering::Less,
            (None, Some(_)) => std::cmp::Ordering::Greater,
            (None, None) => std::cmp::Ordering::Equal,
        });
        rows.truncate(n);
        rows
    }

    /// `tune::select_best()`: the first row of [`TuneResults::show_best`].
    pub fn select_best(&self, metric: Metric) -> Result<Selection> {
        self.check_metric(metric)?;
        let best = self
            .show_best(metric, 1)
            .into_iter()
            .next()
            .filter(|r| r.mean.is_some())
            .ok_or_else(|| {
                Error::InvalidArgument(format!("no candidate has a value for `{}`", metric.name()))
            })?;
        Ok(Selection {
            index: best.index,
            config: best.config,
            params: best.params,
        })
    }

    /// `dplyr::arrange(!!!sort)` on the candidates' parameters: a stable sort by each
    /// key in turn, so unsorted ties keep grid order.
    fn arrange(rows: &mut [BestRow], sort: &[SortKey]) -> Result<()> {
        for key in sort.iter().rev() {
            let value_of = |r: &BestRow| -> Result<f64> {
                r.params
                    .iter()
                    .find(|(n, _)| n == &key.param)
                    .map(|(_, v)| *v)
                    .ok_or_else(|| {
                        Error::InvalidArgument(format!("Could not sort results by `{}`", key.param))
                    })
            };
            for r in rows.iter() {
                value_of(r)?;
            }
            rows.sort_by(|a, b| {
                let (x, y) = (value_of(a).expect("checked"), value_of(b).expect("checked"));
                if key.descending {
                    y.total_cmp(&x)
                } else {
                    x.total_cmp(&y)
                }
            });
        }
        Ok(())
    }

    /// `tune::select_by_one_std_err(x, ..., metric)`: every candidate whose mean is
    /// within one standard error of the best (`which.max` / `which.min` picks the
    /// first best; the bound is `best -/+ std_err[best]`), arranged by `sort`
    /// (simplest first); the first of them is selected.
    pub fn select_by_one_std_err(&self, metric: Metric, sort: &[SortKey]) -> Result<Selection> {
        self.check_metric(metric)?;
        let rows = self.rows_for(metric);
        let defined: Vec<&BestRow> = rows.iter().filter(|r| r.mean.is_some()).collect();
        let best = match metric.direction() {
            Direction::Maximize => defined
                .iter()
                .copied()
                .reduce(|a, b| if b.mean > a.mean { b } else { a }),
            Direction::Minimize => defined
                .iter()
                .copied()
                .reduce(|a, b| if b.mean < a.mean { b } else { a }),
        }
        .ok_or_else(|| Error::InvalidArgument(format!("no candidate has a value for `{}`", metric.name())))?;
        let se = best.std_err.ok_or_else(|| {
            Error::InvalidArgument(
                "the standard error of the best candidate is undefined (one resample)".into(),
            )
        })?;
        let best_mean = best.mean.expect("defined");
        let mut kept: Vec<BestRow> = match metric.direction() {
            Direction::Maximize => {
                let bound = best_mean - se;
                rows.into_iter()
                    .filter(|r| r.mean.is_some_and(|m| m >= bound))
                    .collect()
            }
            Direction::Minimize => {
                let bound = best_mean + se;
                rows.into_iter()
                    .filter(|r| r.mean.is_some_and(|m| m <= bound))
                    .collect()
            }
        };
        Self::arrange(&mut kept, sort)?;
        let first = kept.into_iter().next().expect("the best itself is kept");
        Ok(Selection {
            index: first.index,
            config: first.config,
            params: first.params,
        })
    }

    /// `tune::select_by_pct_loss(x, ..., metric, limit)`: candidates arranged by
    /// `sort` (simplest first); among those up to and including the best, the first
    /// whose percent loss `|(|mean| - |best|) / best| * 100` is below `limit`.
    pub fn select_by_pct_loss(&self, metric: Metric, sort: &[SortKey], limit: f64) -> Result<Selection> {
        self.check_metric(metric)?;
        let rows = self.rows_for(metric);
        let means = rows.iter().filter_map(|r| r.mean);
        let best = match metric.direction() {
            Direction::Maximize => means.fold(f64::NEG_INFINITY, f64::max),
            Direction::Minimize => means.fold(f64::INFINITY, f64::min),
        };
        if !best.is_finite() {
            return Err(Error::InvalidArgument(format!(
                "no candidate has a value for `{}`",
                metric.name()
            )));
        }
        let mut sorted: Vec<BestRow> = rows.into_iter().filter(|r| r.mean.is_some()).collect();
        Self::arrange(&mut sorted, sort)?;
        let loss = |m: f64| ((m.abs() - best.abs()) / best).abs() * 100.0;
        // `which(.loss == 0)[1]`: the best (or its first exact tie) in sorted order
        let best_pos = sorted
            .iter()
            .position(|r| loss(r.mean.expect("defined")) == 0.0)
            .expect("the best has zero loss");
        let pick = sorted[..=best_pos]
            .iter()
            .find(|r| loss(r.mean.expect("defined")) < limit)
            .ok_or_else(|| Error::InvalidArgument("no candidate is within the percent-loss limit".into()))?;
        Ok(Selection {
            index: pick.index,
            config: pick.config,
            params: pick.params.clone(),
        })
    }
}

/// `tune::fit_best()`: finalize `workflow` with [`TuneResults::select_best`] for
/// `metric` and fit it on all of `data` with `Workflow::fit` (parsnip's `fit()`, so
/// workflows' sparse-data decision applies, unlike inside resampling).
pub fn fit_best(
    results: &TuneResults,
    workflow: &Workflow,
    data: &Table,
    metric: Metric,
) -> Result<FittedWorkflow> {
    let sel = results.select_best(metric)?;
    workflow.finalize(&sel.params)?.fit(data)
}

/// Evaluate every grid point with `fit_resamples`. Work is parallelised over
/// (candidate, fold) pairs; results are deterministic regardless of thread count.
pub fn tune_grid(
    workflow: &Workflow,
    data: &Table,
    folds: &Folds,
    grid: &Grid,
    metrics: &MetricSet,
    par: Parallelism,
) -> Result<TuneResults> {
    if grid.points.is_empty() {
        return Err(Error::InvalidArgument("grid is empty".into()));
    }
    // Build every candidate workflow up front so parameter errors surface before work.
    let candidates: Vec<Workflow> = grid
        .points
        .iter()
        .map(|p| workflow.finalize(p))
        .collect::<Result<_>>()?;
    let jobs: Vec<(usize, usize)> = (0..candidates.len())
        .flat_map(|c| (0..folds.len()).map(move |k| (c, k)))
        .collect();
    let fold_results = parallel_map(jobs, par, |(c, k)| {
        let split = &folds.splits[k];
        let train = data.select_rows(&split.train)?;
        let test = data.select_rows(&split.test)?;
        // tune 2.1.0: fold fits skip workflows' sparsity toggle and use frequency
        // case weights in the metrics (Workflow::fit_fold)
        Ok(FoldMetrics {
            id: folds.ids[k].clone(),
            metrics: candidates[c].fit_fold(&train, &test, metrics)?,
        })
    })?;
    let mut iter = fold_results.into_iter();
    let results = grid
        .points
        .iter()
        .map(|p| {
            let per_fold: Vec<FoldMetrics> = iter.by_ref().take(folds.len()).collect();
            (p.clone(), ResampleResults::from_folds(per_fold))
        })
        .collect();
    Ok(TuneResults {
        candidates: results,
        param_order: workflow.model().tunable().iter().map(|s| s.to_string()).collect(),
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;
    use tidyrustmodels_metrics::MetricValue;
    use tidyrustmodels_workflows::MetricSummary;

    /// Hand-built results: `penalty` candidates with rmse means and standard errors.
    fn fake(rows: &[(f64, Option<f64>, Option<f64>)]) -> TuneResults {
        TuneResults {
            candidates: rows
                .iter()
                .map(|(p, mean, se)| {
                    (
                        vec![("penalty".to_string(), *p)],
                        ResampleResults {
                            per_fold: vec![FoldMetrics {
                                id: "Fold1".into(),
                                metrics: vec![MetricValue {
                                    name: "rmse".into(),
                                    value: *mean,
                                }],
                            }],
                            summary: vec![MetricSummary {
                                name: "rmse".into(),
                                mean: *mean,
                                std_err: *se,
                                n: usize::from(mean.is_some()),
                            }],
                        },
                    )
                })
                .collect(),
            param_order: vec!["penalty".into()],
        }
    }

    #[test]
    fn configs_follow_tune_numbering() {
        // grid rows (trees, learn_rate) in grid order; parsnip order is trees first
        let mut r = fake(&[(1.0, Some(1.0), None), (2.0, Some(1.0), None)]);
        r.param_order = vec!["trees".into(), "learn_rate".into()];
        r.candidates[0].0 = vec![("learn_rate".into(), 0.1), ("trees".into(), 25.0)];
        r.candidates[1].0 = vec![("learn_rate".into(), 0.3), ("trees".into(), 10.0)];
        assert_eq!(r.configs(), [2, 1]);
        // identical candidates share a config; show_best lists config order on ties
        r.candidates.push((
            vec![("learn_rate".into(), 0.3), ("trees".into(), 10.0)],
            r.candidates[1].1.clone(),
        ));
        assert_eq!(r.configs(), [2, 1, 1]);
        let order: Vec<usize> = r.show_best(Metric::Rmse, 5).iter().map(|b| b.index).collect();
        assert_eq!(order, [1, 2, 0]);
    }

    #[test]
    fn selection_rules_follow_tune() {
        // penalties 0.001, 0.01, 0.1, 1 with rmse 1.00, 1.02, 1.05, 1.30 (se 0.04)
        let r = fake(&[
            (0.001, Some(1.00), Some(0.04)),
            (0.01, Some(1.02), Some(0.04)),
            (0.1, Some(1.05), Some(0.04)),
            (1.0, Some(1.30), Some(0.04)),
        ]);
        assert_eq!(r.select_best(Metric::Rmse).unwrap().index, 0);
        let sb = r.show_best(Metric::Rmse, 10);
        assert_eq!(sb.iter().map(|b| b.index).collect::<Vec<_>>(), [0, 1, 2, 3]);
        // one SE: bound 1.04 keeps 0.001 and 0.01; the simplest (largest penalty) wins
        let one = r
            .select_by_one_std_err(Metric::Rmse, &[SortKey::desc("penalty")])
            .unwrap();
        assert_eq!(one.index, 1);
        // pct loss in desc(penalty) order: 1 (30 %), 0.1 (5 %), 0.01 (2 %), 0.001 (0)
        assert_eq!(
            r.select_by_pct_loss(Metric::Rmse, &[SortKey::desc("penalty")], 2.0)
                .unwrap()
                .index,
            0
        );
        assert_eq!(
            r.select_by_pct_loss(Metric::Rmse, &[SortKey::desc("penalty")], 3.0)
                .unwrap()
                .index,
            1
        );
        assert_eq!(
            r.select_by_pct_loss(Metric::Rmse, &[SortKey::desc("penalty")], 6.0)
                .unwrap()
                .index,
            2
        );
        // ties keep grid order (stable), undefined means sort last
        let t = fake(&[
            (0.1, Some(1.0), Some(0.1)),
            (0.2, Some(1.0), Some(0.1)),
            (0.3, None, None),
        ]);
        assert_eq!(t.select_best(Metric::Rmse).unwrap().index, 0);
        assert_eq!(t.show_best(Metric::Rmse, 5).last().unwrap().index, 2);
        assert!(t.select_best(Metric::Mae).is_err());
        assert!(SortKey::parse("desc(penalty)").unwrap().descending);
        assert!(!SortKey::parse(" penalty ").unwrap().descending);
        assert!(SortKey::parse("abs(penalty)").is_err());
    }
    use tidyrustmodels_core::Column;
    use tidyrustmodels_models::LogisticRegSpec;
    use tidyrustmodels_recipes::{Recipe, Selector, StepNormalize};
    use tidyrustmodels_resample::vfold_cv;

    #[test]
    fn regular_grid_is_cartesian_and_log_scaled() {
        let g = grid_regular(
            &[
                ParamRange::log10("penalty", -2.0, 0.0),
                ParamRange::linear("b", 0.0, 1.0),
            ],
            3,
        )
        .unwrap();
        assert_eq!(g.points.len(), 9);
        assert_abs_diff_eq!(g.points[0][0].1, 0.01);
        assert_abs_diff_eq!(g.points[1][0].1, 0.1, epsilon = 1e-12);
        assert_abs_diff_eq!(g.points[2][0].1, 1.0);
        assert_abs_diff_eq!(g.points[3][1].1, 0.5);
        let r = grid_random(&[ParamRange::log10("penalty", -3.0, 0.0)], 5, Seed(1)).unwrap();
        assert_eq!(
            r,
            grid_random(&[ParamRange::log10("penalty", -3.0, 0.0)], 5, Seed(1)).unwrap()
        );
        assert!(r.points.iter().all(|p| p[0].1 >= 1e-3 && p[0].1 <= 1.0));
    }

    #[test]
    fn tune_grid_runs_and_ranks() {
        let n = 60;
        let x: Vec<f64> = (0..n).map(|i| (i as f64 * 0.7).sin()).collect();
        let z: Vec<f64> = (0..n).map(|i| (i as f64 * 1.9).cos()).collect();
        let cls: Vec<&str> = (0..n)
            .map(|i| {
                if x[i] + 0.5 * z[i] + 0.3 * ((i * 7 % 11) as f64 / 11.0 - 0.5) > 0.0 {
                    "y"
                } else {
                    "n"
                }
            })
            .collect();
        let t = Table::new(vec![
            ("cls", Column::categorical(&cls).unwrap()),
            ("x", Column::numeric(x).unwrap()),
            ("z", Column::numeric(z).unwrap()),
        ])
        .unwrap();
        let wf = Workflow::new(
            Recipe::new("cls").step(StepNormalize::new(Selector::AllNumericPredictors)),
            LogisticRegSpec::new(),
        );
        let folds = vfold_cv(n, 4, None, Seed(2)).unwrap();
        let grid = grid_regular(&[ParamRange::log10("penalty", -3.0, 1.0)], 3).unwrap();
        let metrics = MetricSet::classification_defaults();
        let serial = tune_grid(&wf, &t, &folds, &grid, &metrics, Parallelism::SERIAL).unwrap();
        let parallel = tune_grid(&wf, &t, &folds, &grid, &metrics, Parallelism::threads(3)).unwrap();
        assert_eq!(serial, parallel);
        assert_eq!(serial.candidates.len(), 3);
        let best = serial.show_best(Metric::BrierClass, 3);
        assert_eq!(best.len(), 3);
        assert!(best[0].mean <= best[1].mean && best[1].mean <= best[2].mean);
        assert_eq!(
            serial.select_best(Metric::BrierClass).unwrap().index,
            best[0].index
        );
        let fitted = fit_best(&serial, &wf, &t, Metric::BrierClass).unwrap();
        assert_eq!(fitted.predict(&t).unwrap().len(), n);
        assert!(tune_grid(
            &wf,
            &t,
            &folds,
            &Grid {
                points: vec![vec![("nope".into(), 1.0)]]
            },
            &metrics,
            Parallelism::SERIAL
        )
        .is_err());
    }
}
