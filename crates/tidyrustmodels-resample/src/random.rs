//! Random resampling schemes: `vfold_cv` (with repeats), `bootstraps`, `mc_cv`,
//! `loo_cv`, `initial_split`, `initial_validation_split`.

use crate::{names0, strata_groups, Folds, Split};
use rand::seq::SliceRandom;
use rand::{Rng, RngExt};
use serde::{Deserialize, Serialize};
use tidyrustmodels_core::{Error, Result, Seed};

/// `rsample:::check_v` for rows.
fn check_v(v: usize, max_v: usize, what: &str, prevent_loo: bool) -> Result<()> {
    if v < 2 {
        return Err(Error::InvalidArgument(format!("v must be at least 2, got {v}")));
    }
    if v > max_v {
        return Err(Error::InvalidArgument(format!(
            "the number of {what} ({max_v}) is less than v = {v}"
        )));
    }
    if prevent_loo && v == max_v {
        return Err(Error::InvalidArgument(
            "leave-one-out cross-validation is not supported by this function; use loo_cv".into(),
        ));
    }
    Ok(())
}

/// One v-fold assignment: `sample(rep(1:v, length.out = n))`, independently within
/// each stratum (`add_vfolds`).
fn vfold_assignment(groups: &[Vec<usize>], n: usize, v: usize, rng: &mut impl Rng) -> Vec<u32> {
    let mut fold_of_row = vec![0u32; n];
    for g in groups {
        let mut labels: Vec<u32> = (0..g.len()).map(|i| (i % v) as u32).collect();
        labels.shuffle(rng);
        for (row, f) in g.iter().zip(labels) {
            fold_of_row[*row] = f;
        }
    }
    fold_of_row
}

fn folds_from_assignment(fold_of_row: &[u32], v: usize) -> Result<Folds> {
    let mut counts = vec![0usize; v];
    for f in fold_of_row {
        counts[*f as usize] += 1;
    }
    if counts.contains(&0) {
        return Err(Error::InvalidArgument(
            "stratification left a fold with no assessment rows; reduce v or drop strata".into(),
        ));
    }
    Folds::from_assignments(fold_of_row)
}

/// rsample's `vfold_cv`: fold labels `rep(1:v, length.out = n)` randomly permuted,
/// independently within each stratum when `strata` is given. Fold sizes differ by at
/// most one overall, and by at most one within each stratum. `v == n` is refused
/// (use [`loo_cv`]). Ids `Fold1`… (`names0`).
pub fn vfold_cv(n: usize, v: usize, strata: Option<&[u32]>, seed: Seed) -> Result<Folds> {
    check_v(v, n, "rows", true)?;
    let groups = strata_groups(strata, n)?;
    let mut rng = seed.rng();
    folds_from_assignment(&vfold_assignment(&groups, n, v, &mut rng), v)
}

/// `vfold_cv(repeats = )`: `repeats` independent v-fold assignments; ids join
/// rsample's `id` and `id2` as `Repeat1.Fold1`. With `repeats == 1` this is
/// [`vfold_cv`].
pub fn vfold_cv_repeated(
    n: usize,
    v: usize,
    repeats: usize,
    strata: Option<&[u32]>,
    seed: Seed,
) -> Result<Folds> {
    if repeats < 1 {
        return Err(Error::InvalidArgument("repeats must be at least 1".into()));
    }
    if repeats == 1 {
        return vfold_cv(n, v, strata, seed);
    }
    check_v(v, n, "rows", true)?;
    let groups = strata_groups(strata, n)?;
    let mut rng = seed.rng();
    let repeat_ids = names0(repeats, "Repeat");
    let mut splits = Vec::with_capacity(v * repeats);
    let mut ids = Vec::with_capacity(v * repeats);
    for r in repeat_ids {
        let f = folds_from_assignment(&vfold_assignment(&groups, n, v, &mut rng), v)?;
        for (s, id2) in f.splits.into_iter().zip(f.ids) {
            splits.push(s);
            ids.push(format!("{r}.{id2}"));
        }
    }
    Ok(Folds { splits, ids, n })
}

/// `rsample::bootstraps`: `times` draws of `n` rows with replacement (per stratum:
/// each stratum resampled to its own size, rows then sorted within the stratum as
/// `strat_sample` does); the assessment set is the out-of-bag rows (`complement`).
/// An empty assessment set is kept, as rsample does with a warning; `fit_resamples`
/// then fails on that resample. Ids `Bootstrap01`….
pub fn bootstraps(n: usize, times: usize, strata: Option<&[u32]>, seed: Seed) -> Result<Folds> {
    if n == 0 || times == 0 {
        return Err(Error::InvalidArgument("need rows and times >= 1".into()));
    }
    let groups = strata_groups(strata, n)?;
    let mut rng = seed.rng();
    let mut splits = Vec::with_capacity(times);
    let mut draws: Vec<Vec<usize>> = vec![Vec::with_capacity(n); times];
    // rsample draws all `times` samples of one stratum before the next
    for g in &groups {
        for d in draws.iter_mut() {
            let mut ind: Vec<usize> = (0..g.len()).map(|_| g[rng.random_range(0..g.len())]).collect();
            if strata.is_some() {
                ind.sort_unstable();
            }
            d.extend(ind);
        }
    }
    for train in draws {
        let mut in_bag = vec![false; n];
        for &i in &train {
            in_bag[i] = true;
        }
        let test: Vec<usize> = (0..n).filter(|i| !in_bag[*i]).collect();
        splits.push(Split::new(train, test, n)?);
    }
    Ok(Folds {
        splits,
        ids: names0(times, "Bootstrap"),
        n,
    })
}

/// `rsample::mc_cv`: `times` Monte-Carlo splits with `floor(n * prop)` analysis rows
/// drawn without replacement (per stratum when stratified, sorted within the stratum
/// as `strat_sample` does); the rest assess. Ids `Resample1`….
pub fn mc_cv(n: usize, prop: f64, times: usize, strata: Option<&[u32]>, seed: Seed) -> Result<Folds> {
    if !(prop > 0.0 && prop < 1.0) {
        return Err(Error::InvalidArgument(format!(
            "prop must be in (0, 1), got {prop}"
        )));
    }
    if n == 0 || times == 0 {
        return Err(Error::InvalidArgument("need rows and times >= 1".into()));
    }
    let groups = strata_groups(strata, n)?;
    let mut rng = seed.rng();
    let mut draws: Vec<Vec<usize>> = vec![Vec::new(); times];
    for g in &groups {
        let k = (g.len() as f64 * prop).floor() as usize;
        for d in draws.iter_mut() {
            let mut rows = g.clone();
            rows.shuffle(&mut rng);
            let mut ind: Vec<usize> = rows[..k].to_vec();
            if strata.is_some() {
                ind.sort_unstable();
            }
            d.extend(ind);
        }
    }
    let mut splits = Vec::with_capacity(times);
    for train in draws {
        let mut in_train = vec![false; n];
        for &i in &train {
            in_train[i] = true;
        }
        let test: Vec<usize> = (0..n).filter(|i| !in_train[*i]).collect();
        splits.push(Split::new(train, test, n)?);
    }
    Ok(Folds {
        splits,
        ids: names0(times, "Resample"),
        n,
    })
}

/// `rsample::loo_cv`: one resample per row, in a random order (rsample permutes the
/// fold labels). Ids `Resample1`…`Resample<n>`. tune does not resample with it.
pub fn loo_cv(n: usize, seed: Seed) -> Result<Folds> {
    if n < 2 {
        return Err(Error::InvalidArgument("loo_cv needs at least 2 rows".into()));
    }
    let mut rng = seed.rng();
    let fold_of_row = vfold_assignment(&[(0..n).collect()], n, n, &mut rng);
    let mut f = Folds::from_assignments(&fold_of_row)?;
    f.ids = (1..=n).map(|k| format!("Resample{k}")).collect();
    Ok(f)
}

/// rsample's `initial_split` / `mc_cv` with `times = 1`: `floor(n * prop)` analysis
/// rows drawn without replacement, per stratum when `strata` is given. The analysis
/// rows are returned sorted.
pub fn initial_split(n: usize, prop: f64, strata: Option<&[u32]>, seed: Seed) -> Result<Split> {
    let f = mc_cv(n, prop, 1, strata, seed)?;
    let s = f.splits.into_iter().next().expect("one split");
    Split::from_ids(s.train, s.test, n)
}

/// `rsample::initial_validation_split`: training, validation and test rows.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct ThreeWaySplit {
    /// Training rows (sorted).
    pub train: Vec<usize>,
    /// Validation rows (sorted).
    pub validation: Vec<usize>,
    /// Test rows (sorted).
    pub test: Vec<usize>,
    /// Number of rows.
    pub n: usize,
}

impl ThreeWaySplit {
    /// `rsample::validation_set`: one resample whose analysis set is the training
    /// rows and whose assessment set is the validation rows (id `validation`).
    pub fn validation_set(&self) -> Folds {
        Folds {
            splits: vec![Split {
                train: self.train.clone(),
                test: self.validation.clone(),
            }],
            ids: vec!["validation".into()],
            n: self.n,
        }
    }
}

/// `rsample::initial_validation_split(prop = c(train, validation))`: an
/// `mc_cv(times = 1)` draw of `floor(n * prop_train)` training rows, then a draw of
/// `floor(m * prop_val / (1 - prop_train))` validation rows from the `m` remaining
/// rows (stratified with `pool / (1 - prop_train)` in rsample; the strata codes are
/// reused here); the rest test.
pub fn initial_validation_split(
    n: usize,
    prop: (f64, f64),
    strata: Option<&[u32]>,
    seed: Seed,
) -> Result<ThreeWaySplit> {
    let (p_train, p_val) = prop;
    if !(p_train > 0.0 && p_train < 1.0 && p_val > 0.0 && p_val < 1.0) {
        return Err(Error::InvalidArgument(
            "elements of prop need to be in (0, 1)".into(),
        ));
    }
    if !(p_train + p_val > 0.0 && p_train + p_val < 1.0) {
        return Err(Error::InvalidArgument(
            "the sum of the proportions in prop needs to be in (0, 1)".into(),
        ));
    }
    let mut rng = seed.rng();
    let first = mc_cv(n, p_train, 1, strata, Seed(rng.random()))?;
    let first = first.splits.into_iter().next().expect("one");
    let mut train = first.train;
    train.sort_unstable();
    let rest = first.test; // ascending
    let m = rest.len();
    let sub_strata: Option<Vec<u32>> = strata.map(|s| rest.iter().map(|&i| s[i]).collect());
    let second = mc_cv(
        m,
        p_val / (1.0 - p_train),
        1,
        sub_strata.as_deref(),
        Seed(rng.random()),
    )?;
    let second = second.splits.into_iter().next().expect("one");
    let mut validation: Vec<usize> = second.train.iter().map(|&j| rest[j]).collect();
    validation.sort_unstable();
    let test: Vec<usize> = second.test.iter().map(|&j| rest[j]).collect();
    if train.is_empty() || validation.is_empty() || test.is_empty() {
        return Err(Error::InvalidArgument(
            "too few rows for a three-way split with these proportions".into(),
        ));
    }
    Ok(ThreeWaySplit {
        train,
        validation,
        test,
        n,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn stratified_vfold_balances_within_strata() {
        let strata: Vec<u32> = (0..40).map(|i| u32::from(i % 4 == 0)).collect(); // 10 ones, 30 zeros
        let f = vfold_cv(40, 5, Some(&strata), Seed(3)).unwrap();
        for s in &f.splits {
            let ones = s.test.iter().filter(|i| strata[**i] == 1).count();
            assert_eq!(ones, 2);
            assert_eq!(s.test.len(), 8);
        }
        assert!(vfold_cv(5, 1, None, Seed(0)).is_err());
        assert!(vfold_cv(5, 6, None, Seed(0)).is_err());
        assert!(vfold_cv(5, 5, None, Seed(0)).is_err(), "loo refused");
    }

    #[test]
    fn repeated_vfold_has_independent_partitions() {
        let f = vfold_cv_repeated(30, 3, 2, None, Seed(4)).unwrap();
        assert_eq!(f.len(), 6);
        assert_eq!(f.ids[0], "Repeat1.Fold1");
        assert_eq!(f.ids[5], "Repeat2.Fold3");
        for r in 0..2 {
            let mut all: Vec<usize> = f.splits[r * 3..(r + 1) * 3]
                .iter()
                .flat_map(|s| s.test.clone())
                .collect();
            all.sort_unstable();
            assert_eq!(all, (0..30).collect::<Vec<_>>());
        }
        assert_ne!(f.splits[0].test, f.splits[3].test);
        assert_eq!(
            vfold_cv_repeated(30, 3, 1, None, Seed(4)).unwrap(),
            vfold_cv(30, 3, None, Seed(4)).unwrap()
        );
    }

    #[test]
    fn bootstraps_draw_with_replacement_and_assess_out_of_bag() {
        let f = bootstraps(50, 4, None, Seed(5)).unwrap();
        assert_eq!(f.ids, ["Bootstrap1", "Bootstrap2", "Bootstrap3", "Bootstrap4"]);
        for s in &f.splits {
            assert_eq!(s.train.len(), 50);
            let mut sorted = s.train.clone();
            sorted.sort_unstable();
            sorted.dedup();
            assert!(sorted.len() < 50, "a bootstrap repeats rows");
            for t in &s.test {
                assert!(!s.train.contains(t));
            }
            assert_eq!(sorted.len() + s.test.len(), 50);
            assert!(!s.is_partition(50));
        }
        // stratified: each stratum resampled to its own size
        let strata: Vec<u32> = (0..40).map(|i| u32::from(i >= 30)).collect();
        let f = bootstraps(40, 2, Some(&strata), Seed(6)).unwrap();
        for s in &f.splits {
            assert_eq!(s.train.iter().filter(|i| strata[**i] == 1).count(), 10);
            assert_eq!(s.train.len(), 40);
        }
        assert_eq!(bootstraps(25, 25, None, Seed(0)).unwrap().ids[0], "Bootstrap01");
    }

    #[test]
    fn mc_cv_and_loo() {
        let f = mc_cv(20, 0.7, 3, None, Seed(7)).unwrap();
        assert_eq!(f.ids[2], "Resample3");
        for s in &f.splits {
            assert_eq!(s.train.len(), 14);
            assert_eq!(s.test.len(), 6);
            assert!(s.is_partition(20));
        }
        let l = loo_cv(7, Seed(1)).unwrap();
        assert_eq!(l.len(), 7);
        assert!(l.splits.iter().all(|s| s.test.len() == 1 && s.train.len() == 6));
        let s = initial_split(100, 0.75, None, Seed(9)).unwrap();
        assert_eq!((s.train.len(), s.test.len()), (75, 25));
        let strata: Vec<u32> = (0..100).map(|i| u32::from(i < 20)).collect();
        let s = initial_split(100, 0.75, Some(&strata), Seed(9)).unwrap();
        assert_eq!(s.train.iter().filter(|i| strata[**i] == 1).count(), 15);
        assert!(initial_split(10, 1.0, None, Seed(0)).is_err());
    }

    #[test]
    fn three_way_split_sizes_follow_rsample() {
        // rsample: n = 8, prop = c(0.5, 0.25) -> 4 / 2 / 2
        let t = initial_validation_split(8, (0.5, 0.25), None, Seed(2)).unwrap();
        assert_eq!((t.train.len(), t.validation.len(), t.test.len()), (4, 2, 2));
        let mut all: Vec<usize> = t
            .train
            .iter()
            .chain(&t.validation)
            .chain(&t.test)
            .copied()
            .collect();
        all.sort_unstable();
        assert_eq!(all, (0..8).collect::<Vec<_>>());
        let vs = t.validation_set();
        assert_eq!(vs.ids, ["validation"]);
        assert_eq!(vs.splits[0].test, t.validation);
        assert!(initial_validation_split(8, (0.6, 0.5), None, Seed(0)).is_err());
    }
}
