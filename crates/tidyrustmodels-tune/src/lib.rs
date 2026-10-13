//! Hyperparameter tuning over resamples: regular and random grids, `tune_grid`.

#![forbid(unsafe_code)]
#![warn(missing_docs, rust_2018_idioms)]

use rand::RngExt;
use serde::{Deserialize, Serialize};
use tidyrustmodels_core::{Error, Parallelism, Result, Seed, Table};
use tidyrustmodels_metrics::{Direction, Metric, MetricSet};
use tidyrustmodels_resample::Folds;
use tidyrustmodels_workflows::{parallel_map, FoldMetrics, ResampleResults, Workflow};

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
}

impl TuneResults {
    /// `tune::show_best`: the top `n` candidates for `metric` (mean, std_err).
    pub fn show_best(&self, metric: Metric, n: usize) -> Vec<(ParamSet, f64, Option<f64>)> {
        let mut rows: Vec<(ParamSet, f64, Option<f64>)> = self
            .candidates
            .iter()
            .filter_map(|(p, r)| {
                let s = r.metric(metric.name())?;
                Some((p.clone(), s.mean?, s.std_err))
            })
            .collect();
        match metric.direction() {
            Direction::Maximize => rows.sort_by(|a, b| b.1.total_cmp(&a.1)),
            Direction::Minimize => rows.sort_by(|a, b| a.1.total_cmp(&b.1)),
        }
        rows.truncate(n);
        rows
    }
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
        .map(|p| {
            let mut spec = workflow.model().clone_box();
            for (name, value) in p {
                spec = spec.with_param(name, *value)?;
            }
            let mut wf = Workflow::from_boxed(workflow.recipe().clone(), spec);
            if let Some((col, kind)) = workflow.case_weights_spec() {
                wf = wf.case_weights(col, kind);
            }
            Ok(wf)
        })
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
    Ok(TuneResults { candidates: results })
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;
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
        assert!(best[0].1 <= best[1].1 && best[1].1 <= best[2].1);
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
