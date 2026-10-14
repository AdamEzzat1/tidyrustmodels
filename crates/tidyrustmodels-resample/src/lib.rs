//! Reproducible resampling with rsample 1.3.2 semantics.
//!
//! Membership is never random beyond the explicit [`Seed`](tidyrustmodels_core::Seed). R's RNG and ChaCha cannot
//! agree, so parity tests load shared row ids through [`Folds::from_assignments`] /
//! [`Folds::from_splits`] (ADR-0006); the generators here reproduce rsample's
//! *algorithms* (ADR-0013): `vfold_cv` (with repeats), `bootstraps`, `mc_cv`, `loo_cv`,
//! `group_vfold_cv` / `group_bootstraps` / `group_mc_cv` / `group_initial_split`,
//! `rolling_origin`, `sliding_window`, `sliding_index`, `initial_split`,
//! `initial_time_split`, `initial_validation_split` and `make_strata`.
//!
//! An rsplit keeps its analysis rows in the order rsample draws them (bootstrap and
//! Monte-Carlo draws are unsorted; v-fold complements ascend), because row order
//! changes the last bits of a fit.

#![forbid(unsafe_code)]
#![warn(missing_docs, rust_2018_idioms)]

mod groups;
mod random;
mod strata;
mod time;

pub use groups::{
    balance_observations, balance_prop, freq_table, group_bootstraps, group_initial_split, group_mc_cv,
    group_vfold_cv, GroupBalance,
};
pub use random::{
    bootstraps, initial_split, initial_validation_split, loo_cv, mc_cv, vfold_cv, vfold_cv_repeated,
    ThreeWaySplit,
};
pub use strata::{make_strata_categorical, make_strata_numeric, strata_from_numeric};
pub use time::{initial_time_split, rolling_origin, sliding_index, sliding_window};

use serde::{Deserialize, Serialize};
use tidyrustmodels_core::{Error, Result};

/// One analysis/assessment pair of row indices (an rsample `rsplit`).
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct Split {
    /// Analysis (training) rows in rsample's order; a bootstrap repeats rows.
    pub train: Vec<usize>,
    /// Assessment (test) rows, ascending; may be empty (rsample warns, `fit_resamples`
    /// errors).
    pub test: Vec<usize>,
}

impl Split {
    /// Build a *partition* from explicit ids: every row of `0..n` in exactly one of the
    /// two sets, both non-empty, both sorted.
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

    /// Build an rsplit as `rsample::make_splits(list(analysis, assessment))` does: the
    /// analysis rows keep their order and may repeat, the assessment rows are sorted;
    /// both must be in range and the analysis set non-empty. Rows may belong to
    /// neither set (time-series windows) or to both (`apparent`).
    pub fn new(train: Vec<usize>, mut test: Vec<usize>, n: usize) -> Result<Split> {
        if train.is_empty() {
            return Err(Error::InvalidArgument(
                "at least one row should be selected for the analysis set".into(),
            ));
        }
        for (label, ids) in [("analysis", &train), ("assessment", &test)] {
            if let Some(&i) = ids.iter().find(|i| **i >= n) {
                return Err(Error::InvalidArgument(format!(
                    "{label} row {i} out of range for {n} rows"
                )));
            }
        }
        test.sort_unstable();
        Ok(Split { train, test })
    }

    /// Whether the two sets partition `0..n` with no repeats.
    pub fn is_partition(&self, n: usize) -> bool {
        if self.train.len() + self.test.len() != n {
            return false;
        }
        let mut seen = vec![false; n];
        for &i in self.train.iter().chain(&self.test) {
            if i >= n || seen[i] {
                return false;
            }
            seen[i] = true;
        }
        true
    }
}

/// A set of resamples (an rsample `rset`).
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct Folds {
    /// One split per resample.
    pub splits: Vec<Split>,
    /// rsample ids (`Fold1`, `Bootstrap01`, `Resample3`, `Slice2`; repeated v-fold
    /// joins rsample's `id` and `id2` as `Repeat1.Fold1`).
    pub ids: Vec<String>,
    /// Number of rows the resamples index.
    pub n: usize,
}

impl Folds {
    /// Number of resamples.
    pub fn len(&self) -> usize {
        self.splits.len()
    }

    /// Whether there are no resamples.
    pub fn is_empty(&self) -> bool {
        self.splits.is_empty()
    }

    /// Build folds from a fold index per row (0-based). This is how shared fixture
    /// assignments are loaded so R and Rust evaluate on identical rows. Analysis rows
    /// ascend (`setdiff(1:n, assessment)`), as rsample's v-fold complement.
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
            ids: names0(v, "Fold"),
            n,
        })
    }

    /// `rsample::manual_rset`: explicit splits with their ids.
    pub fn from_splits(splits: Vec<Split>, ids: Vec<String>, n: usize) -> Result<Folds> {
        if splits.is_empty() {
            return Err(Error::InvalidArgument("no resamples".into()));
        }
        if splits.len() != ids.len() {
            return Err(Error::InvalidArgument(format!(
                "{} splits but {} ids",
                splits.len(),
                ids.len()
            )));
        }
        for s in &splits {
            if let Some(&i) = s.train.iter().chain(&s.test).find(|i| **i >= n) {
                return Err(Error::InvalidArgument(format!(
                    "row {i} out of range for {n} rows"
                )));
            }
            if s.train.is_empty() {
                return Err(Error::InvalidArgument(
                    "at least one row should be selected for the analysis set".into(),
                ));
            }
        }
        Ok(Folds { splits, ids, n })
    }

    /// Fold index per row (inverse of [`Folds::from_assignments`]); only meaningful
    /// when the assessment sets partition the rows (v-fold).
    pub fn assignments(&self) -> Vec<u32> {
        let mut out = vec![0u32; self.n];
        for (k, s) in self.splits.iter().enumerate() {
            for &i in &s.test {
                out[i] = k as u32;
            }
        }
        out
    }

    /// Whether every split partitions the rows (v-fold style).
    pub fn is_partition(&self) -> bool {
        self.splits.iter().all(|s| s.is_partition(self.n))
    }
}

/// `rsample:::names0`: `prefix` followed by the number, zero-padded to the width of
/// the largest (`Fold1`…`Fold5`, `Fold01`…`Fold10`).
pub fn names0(num: usize, prefix: &str) -> Vec<String> {
    let width = num.to_string().len();
    (1..=num).map(|k| format!("{prefix}{k:0width$}")).collect()
}

/// Rows grouped by stratum code (`None` = one stratum), in code order.
pub(crate) fn strata_groups(strata: Option<&[u32]>, n: usize) -> Result<Vec<Vec<usize>>> {
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

#[cfg(test)]
mod tests {
    use super::*;
    use tidyrustmodels_core::Seed;

    #[test]
    fn ids_follow_names0() {
        assert_eq!(names0(5, "Fold"), ["Fold1", "Fold2", "Fold3", "Fold4", "Fold5"]);
        assert_eq!(names0(10, "Fold")[0], "Fold01");
        assert_eq!(names0(10, "Fold")[9], "Fold10");
        assert_eq!(names0(25, "Bootstrap")[0], "Bootstrap01");
        assert!(names0(0, "x").is_empty());
    }

    #[test]
    fn vfold_is_balanced_and_reproducible() {
        let a = vfold_cv(23, 5, None, Seed(1)).unwrap();
        let b = vfold_cv(23, 5, None, Seed(1)).unwrap();
        assert_eq!(a, b);
        assert_ne!(a, vfold_cv(23, 5, None, Seed(2)).unwrap());
        let sizes: Vec<usize> = a.splits.iter().map(|s| s.test.len()).collect();
        assert_eq!(sizes.iter().sum::<usize>(), 23);
        assert!(sizes.iter().max().unwrap() - sizes.iter().min().unwrap() <= 1);
        assert_eq!(a.ids[0], "Fold1");
        assert!(a.is_partition());
        assert_eq!(Folds::from_assignments(&a.assignments()).unwrap(), a);
    }

    #[test]
    fn splits_validate() {
        assert!(Folds::from_assignments(&[0, 0, 2]).is_err());
        assert!(Split::from_ids(vec![0, 1], vec![1, 2], 3).is_err());
        assert!(Split::from_ids(vec![0], vec![2], 3).is_err());
        let s = Split::new(vec![2, 2, 0], vec![1], 3).unwrap();
        assert_eq!(s.train, vec![2, 2, 0]);
        assert!(!s.is_partition(3));
        assert!(Split::new(vec![], vec![1], 3).is_err());
        assert!(Split::new(vec![3], vec![1], 3).is_err());
        assert!(Split::new(vec![0], vec![], 3).is_ok());
        assert!(Folds::from_splits(vec![s.clone()], vec!["A".into(), "B".into()], 3).is_err());
        let f = Folds::from_splits(vec![s], vec!["A".into()], 3).unwrap();
        assert!(!f.is_partition());
    }
}
