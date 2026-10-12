//! Reproducible resampling with rsample semantics.
//!
//! Membership is never random beyond the explicit [`Seed`]. R's RNG and ChaCha cannot
//! agree, so parity tests load shared row ids through [`Folds::from_assignments`]
//! (ADR-0006); the generators here reproduce rsample's *algorithms*.

#![forbid(unsafe_code)]
#![warn(missing_docs, rust_2018_idioms)]

use rand::seq::SliceRandom;
use serde::{Deserialize, Serialize};
use tidyrustmodels_core::{Error, Result, Seed};

/// One analysis/assessment partition of row indices.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct Split {
    /// Analysis (training) rows, ascending.
    pub train: Vec<usize>,
    /// Assessment (test) rows, ascending.
    pub test: Vec<usize>,
}

impl Split {
    /// Build from explicit ids and verify they partition `0..n`.
    pub fn from_ids(mut train: Vec<usize>, mut test: Vec<usize>, n: usize) -> Result<Split> {
        train.sort_unstable();
        test.sort_unstable();
        let mut seen = vec![false; n];
        for (label, ids) in [("train", &train), ("test", &test)] {
            for &i in ids {
                if i >= n {
                    return Err(Error::InvalidArgument(format!(
                        "{label} row {i} out of range for {n} rows"
                    )));
                }
                if seen[i] {
                    return Err(Error::InvalidArgument(format!(
                        "row {i} appears twice across train/test"
                    )));
                }
                seen[i] = true;
            }
        }
        if let Some(missing) = seen.iter().position(|s| !s) {
            return Err(Error::InvalidArgument(format!(
                "row {missing} is in neither train nor test"
            )));
        }
        if train.is_empty() || test.is_empty() {
            return Err(Error::InvalidArgument(
                "train and test must both be non-empty".into(),
            ));
        }
        Ok(Split { train, test })
    }
}

/// A set of cross-validation splits.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct Folds {
    /// One split per fold.
    pub splits: Vec<Split>,
    /// rsample-style ids (`Fold01`, `Fold02`, ...).
    pub ids: Vec<String>,
    /// Number of rows the folds partition.
    pub n: usize,
}

impl Folds {
    /// Number of folds.
    pub fn len(&self) -> usize {
        self.splits.len()
    }

    /// Whether there are no folds.
    pub fn is_empty(&self) -> bool {
        self.splits.is_empty()
    }

    /// Build folds from a fold index per row (0-based). This is how shared fixture
    /// assignments are loaded so R and Rust evaluate on identical rows.
    pub fn from_assignments(fold_of_row: &[u32]) -> Result<Folds> {
        let n = fold_of_row.len();
        if n == 0 {
            return Err(Error::InvalidArgument("no rows".into()));
        }
        let v = *fold_of_row.iter().max().expect("non-empty") as usize + 1;
        if v < 2 {
            return Err(Error::InvalidArgument(
                "assignments must define at least 2 folds".into(),
            ));
        }
        let mut members: Vec<Vec<usize>> = vec![Vec::new(); v];
        for (i, f) in fold_of_row.iter().enumerate() {
            members[*f as usize].push(i);
        }
        if let Some(k) = members.iter().position(|m| m.is_empty()) {
            return Err(Error::InvalidArgument(format!("fold {k} has no rows")));
        }
        let splits = members
            .iter()
            .map(|test| {
                let train: Vec<usize> = (0..n).filter(|i| !test.contains(i)).collect();
                Split {
                    train,
                    test: test.clone(),
                }
            })
            .collect();
        Ok(Folds {
            splits,
            ids: fold_ids(v),
            n,
        })
    }

    /// Fold index per row (inverse of [`Folds::from_assignments`]).
    pub fn assignments(&self) -> Vec<u32> {
        let mut out = vec![0u32; self.n];
        for (k, s) in self.splits.iter().enumerate() {
            for &i in &s.test {
                out[i] = k as u32;
            }
        }
        out
    }
}

fn fold_ids(v: usize) -> Vec<String> {
    let width = v.to_string().len().max(2);
    (1..=v).map(|k| format!("Fold{k:0width$}")).collect()
}

fn check_strata(strata: Option<&[u32]>, n: usize) -> Result<Vec<Vec<usize>>> {
    match strata {
        None => Ok(vec![(0..n).collect()]),
        Some(s) => {
            if s.len() != n {
                return Err(Error::InvalidArgument(format!(
                    "strata has {} entries for {n} rows",
                    s.len()
                )));
            }
            let k = *s.iter().max().unwrap_or(&0) as usize + 1;
            let mut groups = vec![Vec::new(); k];
            for (i, g) in s.iter().enumerate() {
                groups[*g as usize].push(i);
            }
            groups.retain(|g| !g.is_empty());
            Ok(groups)
        }
    }
}

/// rsample's `vfold_cv`: fold labels `rep(1:v, length.out = n)` randomly permuted,
/// independently within each stratum when `strata` is given. Fold sizes differ by at
/// most one overall, and by at most one within each stratum.
pub fn vfold_cv(n: usize, v: usize, strata: Option<&[u32]>, seed: Seed) -> Result<Folds> {
    if v < 2 {
        return Err(Error::InvalidArgument(format!("v must be at least 2, got {v}")));
    }
    if v > n {
        return Err(Error::InvalidArgument(format!(
            "v = {v} exceeds the number of rows {n}"
        )));
    }
    let groups = check_strata(strata, n)?;
    let mut rng = seed.rng();
    let mut fold_of_row = vec![0u32; n];
    for g in &groups {
        let mut labels: Vec<u32> = (0..g.len()).map(|i| (i % v) as u32).collect();
        labels.shuffle(&mut rng);
        for (row, f) in g.iter().zip(labels) {
            fold_of_row[*row] = f;
        }
    }
    // With tiny strata some folds could be empty; rsample would then also produce
    // uneven folds. We require every fold to have at least one assessment row.
    let mut counts = vec![0usize; v];
    for f in &fold_of_row {
        counts[*f as usize] += 1;
    }
    if counts.contains(&0) {
        return Err(Error::InvalidArgument(
            "stratification left a fold with no assessment rows; reduce v or drop strata".into(),
        ));
    }
    Folds::from_assignments(&fold_of_row)
}

/// rsample's `initial_split` / `mc_cv` with `times = 1`: `floor(n * prop)` analysis
/// rows drawn without replacement, per stratum when `strata` is given.
pub fn initial_split(n: usize, prop: f64, strata: Option<&[u32]>, seed: Seed) -> Result<Split> {
    if !(prop > 0.0 && prop < 1.0) {
        return Err(Error::InvalidArgument(format!(
            "prop must be in (0, 1), got {prop}"
        )));
    }
    let groups = check_strata(strata, n)?;
    let mut rng = seed.rng();
    let mut train = Vec::with_capacity((n as f64 * prop) as usize);
    for g in &groups {
        let k = (g.len() as f64 * prop).floor() as usize;
        let mut rows = g.clone();
        rows.shuffle(&mut rng);
        train.extend_from_slice(&rows[..k]);
    }
    let in_train = {
        let mut m = vec![false; n];
        for &i in &train {
            m[i] = true;
        }
        m
    };
    let test: Vec<usize> = (0..n).filter(|i| !in_train[*i]).collect();
    Split::from_ids(train, test, n)
}

/// Bin a numeric variable into `breaks` quantile groups for stratification, as
/// rsample's `make_strata` does for numeric columns (type-7 quantiles,
/// `cut(include.lowest = TRUE)`). Returns a group code per row.
pub fn strata_from_numeric(values: &[f64], breaks: usize) -> Result<Vec<u32>> {
    if breaks < 2 {
        return Err(Error::InvalidArgument("breaks must be at least 2".into()));
    }
    if values.iter().any(|v| !v.is_finite()) {
        return Err(Error::InvalidValue("strata values must be finite".into()));
    }
    let mut sorted = values.to_vec();
    sorted.sort_by(|a, b| a.partial_cmp(b).expect("finite"));
    let n = sorted.len();
    if n == 0 {
        return Err(Error::InvalidArgument("no rows".into()));
    }
    // type-7 quantiles at probs k/breaks
    let mut cuts: Vec<f64> = (0..=breaks)
        .map(|k| {
            let h = (n as f64 - 1.0) * (k as f64 / breaks as f64);
            let lo = h.floor() as usize;
            let hi = (lo + 1).min(n - 1);
            sorted[lo] + (h - lo as f64) * (sorted[hi] - sorted[lo])
        })
        .collect();
    cuts.dedup();
    let groups = cuts.len() - 1;
    if groups < 1 {
        return Ok(vec![0; n]);
    }
    Ok(values
        .iter()
        .map(|v| {
            // intervals (c_k, c_{k+1}], first interval closed on the left
            let mut g = 0usize;
            for k in 0..groups {
                let lo_ok = if k == 0 { *v >= cuts[0] } else { *v > cuts[k] };
                if lo_ok && *v <= cuts[k + 1] {
                    g = k;
                    break;
                }
            }
            g as u32
        })
        .collect())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn vfold_is_balanced_and_reproducible() {
        let a = vfold_cv(23, 5, None, Seed(1)).unwrap();
        let b = vfold_cv(23, 5, None, Seed(1)).unwrap();
        assert_eq!(a, b);
        assert_ne!(a, vfold_cv(23, 5, None, Seed(2)).unwrap());
        let sizes: Vec<usize> = a.splits.iter().map(|s| s.test.len()).collect();
        assert_eq!(sizes.iter().sum::<usize>(), 23);
        assert!(sizes.iter().max().unwrap() - sizes.iter().min().unwrap() <= 1);
        assert_eq!(a.ids[0], "Fold01");
        assert_eq!(Folds::from_assignments(&a.assignments()).unwrap(), a);
    }

    #[test]
    fn stratified_vfold_balances_within_strata() {
        let strata: Vec<u32> = (0..40).map(|i| u32::from(i % 4 == 0)).collect(); // 10 ones, 30 zeros
        let f = vfold_cv(40, 5, Some(&strata), Seed(3)).unwrap();
        for s in &f.splits {
            let ones = s.test.iter().filter(|i| strata[**i] == 1).count();
            assert_eq!(ones, 2);
            assert_eq!(s.test.len(), 8);
        }
    }

    #[test]
    fn initial_split_sizes_and_strata() {
        let s = initial_split(100, 0.75, None, Seed(9)).unwrap();
        assert_eq!(s.train.len(), 75);
        assert_eq!(s.test.len(), 25);
        let strata: Vec<u32> = (0..100).map(|i| u32::from(i < 20)).collect();
        let s = initial_split(100, 0.75, Some(&strata), Seed(9)).unwrap();
        assert_eq!(s.train.iter().filter(|i| strata[**i] == 1).count(), 15);
        assert!(initial_split(10, 1.0, None, Seed(0)).is_err());
    }

    #[test]
    fn invalid_inputs() {
        assert!(vfold_cv(5, 1, None, Seed(0)).is_err());
        assert!(vfold_cv(5, 6, None, Seed(0)).is_err());
        assert!(Folds::from_assignments(&[0, 0, 2]).is_err());
        assert!(Split::from_ids(vec![0, 1], vec![1, 2], 3).is_err());
        assert!(Split::from_ids(vec![0], vec![2], 3).is_err());
    }

    #[test]
    fn numeric_strata_quartiles() {
        let v: Vec<f64> = (1..=20).map(|i| i as f64).collect();
        let s = strata_from_numeric(&v, 4).unwrap();
        let mut counts = [0; 4];
        for g in &s {
            counts[*g as usize] += 1;
        }
        assert_eq!(counts, [5, 5, 5, 5]);
        assert_eq!(s[0], 0);
        assert_eq!(s[19], 3);
    }
}
