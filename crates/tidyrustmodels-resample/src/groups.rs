//! Grouped resampling (`group_vfold_cv`, `group_bootstraps`, `group_mc_cv`,
//! `group_initial_split`): every row of a group lands in the same set. The balancing
//! rules are rsample 1.3.2's `make_groups` with `balance = "groups"` / `"observations"`
//! / `"prop"` (`balance_groups_normal`, `balance_observations_helper`,
//! `balance_prop_helper`, `collapse_groups`).

use crate::{names0, Folds, Split};
use rand::seq::SliceRandom;
use rand::RngExt;
use serde::{Deserialize, Serialize};
use tidyrustmodels_core::{Error, Result, Seed};

/// `group_vfold_cv(balance = )`.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum GroupBalance {
    /// Each fold gets (nearly) the same number of groups.
    Groups,
    /// Groups are assigned greedily so each fold gets (nearly) the same number of rows.
    Observations,
}

/// Groups in order of first appearance with their row counts (`vec_count(sort =
/// "location")`): the table rsample's balance helpers work on.
pub fn freq_table(groups: &[u32]) -> Vec<(u32, usize)> {
    let mut order: Vec<u32> = Vec::new();
    let mut counts: Vec<usize> = Vec::new();
    for g in groups {
        match order.iter().position(|o| o == g) {
            Some(p) => counts[p] += 1,
            None => {
                order.push(*g);
                counts.push(1);
            }
        }
    }
    order.into_iter().zip(counts).collect()
}

/// `balance_observations_helper`: groups in a random order; the first `v` open one
/// fold each, every later group joins the fold whose share of the rows improves the
/// most (`which.min` of `post_error - pre_error`, first on ties). Returns the fold
/// (0-based) of each entry of `table`, in `table` order. `perm` is the shuffled order
/// of the table rows (rsample's `sample.int(nrow(freq_table))`); it is a parameter so
/// the rule can be checked against R on the same permutation (Phase 8h).
pub fn balance_observations(table: &[(u32, usize)], perm: &[usize], v: usize, n_obs: usize) -> Vec<usize> {
    let target = 1.0 / v as f64;
    let mut assignment = vec![usize::MAX; table.len()];
    let mut fold_counts = vec![0usize; v];
    for (k, &row) in perm.iter().enumerate() {
        let size = table[row].1;
        let a = if k < v {
            k
        } else {
            let mut best = 0usize;
            let mut best_imp = f64::INFINITY;
            for (a, &c) in fold_counts.iter().enumerate() {
                let pre = (c as f64 / n_obs as f64 - target).abs();
                let post = ((c + size) as f64 / n_obs as f64 - target).abs();
                let imp = post - pre;
                if imp < best_imp {
                    best_imp = imp;
                    best = a;
                }
            }
            best
        };
        assignment[row] = a;
        fold_counts[a] += size;
    }
    assignment
}

/// `balance_prop_helper`: for one resample, groups are drawn (`n` draws, with or
/// without replacement) and taken in draw order until their cumulative share of the
/// rows crosses `prop`, keeping the shorter prefix when it is closer to `prop`.
/// Returns the drawn positions kept (repeats possible with replacement). `draws` are
/// rsample's `sample.int(nrow(freq_table), n, replace)` positions, a parameter so the
/// rule can be checked against R on the same draws (Phase 8h).
pub fn balance_prop(table: &[(u32, usize)], draws: &[usize], prop: f64) -> Vec<usize> {
    let total: usize = table.iter().map(|t| t.1).sum();
    let mut cum = Vec::with_capacity(draws.len());
    let mut acc = 0usize;
    for &d in draws {
        acc += table[d].1;
        cum.push(acc as f64 / total as f64);
    }
    // `which(cumulative_proportion > prop)[[1]]` (1-based)
    let crosses = cum
        .iter()
        .position(|c| *c > prop)
        .map(|p| p + 1)
        .unwrap_or(cum.len());
    let mut k = crosses;
    if crosses >= 2 {
        let here = (cum[crosses - 1] - prop).abs();
        let prev = (cum[crosses - 2] - prop).abs();
        // `which.min(abs(c(here, prev) - prop)) - 1`: prefer the previous prefix only
        // when it is strictly closer
        if prev < here {
            k = crosses - 1;
        }
    }
    draws[..k].to_vec()
}

fn check_groups(groups: &[u32]) -> Result<Vec<(u32, usize)>> {
    if groups.is_empty() {
        return Err(Error::InvalidArgument("no rows".into()));
    }
    Ok(freq_table(groups))
}

/// `rsample::group_vfold_cv(group, v, repeats, balance)`: whole groups are assigned
/// to `v` folds (`v = None` means one fold per group, i.e. leave-one-group-out). Ids
/// `Resample1`… (`Repeat1.Resample1`… with repeats).
pub fn group_vfold_cv(
    groups: &[u32],
    v: Option<usize>,
    repeats: usize,
    balance: GroupBalance,
    seed: Seed,
) -> Result<Folds> {
    let table = check_groups(groups)?;
    let n = groups.len();
    let n_groups = table.len();
    let v = v.unwrap_or(n_groups);
    if v < 2 {
        return Err(Error::InvalidArgument(format!("v must be at least 2, got {v}")));
    }
    if v > n_groups {
        return Err(Error::InvalidArgument(format!(
            "the number of groups ({n_groups}) is less than v = {v}"
        )));
    }
    if repeats < 1 {
        return Err(Error::InvalidArgument("repeats must be at least 1".into()));
    }
    if repeats > 1 && v == n_groups {
        return Err(Error::InvalidArgument(format!(
            "repeated resampling when v is {v} would create identical resamples"
        )));
    }
    let mut rng = seed.rng();
    let mut splits = Vec::with_capacity(v * repeats);
    let mut ids = Vec::with_capacity(v * repeats);
    let repeat_ids = names0(repeats, "Repeat");
    for repeat_id in &repeat_ids {
        // fold of each group (table order)
        let fold_of_group: Vec<usize> = match balance {
            GroupBalance::Groups => {
                // `sample(rep(seq_len(v), length.out = n_groups))` over the groups in
                // order of first appearance
                let mut labels: Vec<usize> = (0..n_groups).map(|i| i % v).collect();
                labels.shuffle(&mut rng);
                labels
            }
            GroupBalance::Observations => {
                let mut perm: Vec<usize> = (0..n_groups).collect();
                perm.shuffle(&mut rng);
                let assignment = balance_observations(&table, &perm, v, n);
                // collapse_groups: assignments are relabelled by a random permutation
                let mut relabel: Vec<usize> = (0..v).collect();
                relabel.shuffle(&mut rng);
                assignment.iter().map(|a| relabel[*a]).collect()
            }
        };
        let fold_of_row: Vec<usize> = groups
            .iter()
            .map(|g| fold_of_group[table.iter().position(|t| t.0 == *g).expect("group")])
            .collect();
        for k in 0..v {
            let test: Vec<usize> = (0..n).filter(|i| fold_of_row[*i] == k).collect();
            let train: Vec<usize> = (0..n).filter(|i| fold_of_row[*i] != k).collect();
            splits.push(Split::new(train, test, n)?);
            ids.push(if repeats == 1 {
                format!("Resample{}", k + 1)
            } else {
                format!("{}.Resample{}", repeat_id, k + 1)
            });
        }
    }
    // names0 padding for the plain ids
    if repeats == 1 {
        ids = names0(v, "Resample");
    }
    Ok(Folds { splits, ids, n })
}

/// Rows of the groups drawn for one resample (`collapse_groups` + `left_join`):
/// ascending row order, each row repeated once per draw of its group.
fn rows_of_draws(groups: &[u32], table: &[(u32, usize)], draws: &[usize]) -> Vec<usize> {
    let mut times_drawn = vec![0usize; table.len()];
    for &d in draws {
        times_drawn[d] += 1;
    }
    let mut out = Vec::new();
    for (i, g) in groups.iter().enumerate() {
        let p = table.iter().position(|t| t.0 == *g).expect("group");
        for _ in 0..times_drawn[p] {
            out.push(i);
        }
    }
    out
}

/// `rsample::group_bootstraps`: each resample draws groups with replacement until
/// their rows cover the whole data (`balance = "prop"`, `prop = 1`); the assessment
/// set is the rows of groups never drawn. Ids `Bootstrap01`….
pub fn group_bootstraps(groups: &[u32], times: usize, seed: Seed) -> Result<Folds> {
    let table = check_groups(groups)?;
    if times == 0 {
        return Err(Error::InvalidArgument("times must be at least 1".into()));
    }
    let n = groups.len();
    let n_groups = table.len();
    let min_count = table.iter().map(|t| t.1).min().expect("non-empty");
    // `n <- ceiling(n_groups * prop * sum(count) / min(count))` draws with replacement
    let n_draw = ((n_groups * n) as f64 / min_count as f64).ceil() as usize;
    let mut rng = seed.rng();
    let mut splits = Vec::with_capacity(times);
    for _ in 0..times {
        let draws: Vec<usize> = (0..n_draw).map(|_| rng.random_range(0..n_groups)).collect();
        let kept = balance_prop(&table, &draws, 1.0);
        let train = rows_of_draws(groups, &table, &kept);
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

/// `rsample::group_mc_cv`: each resample takes groups in a random order until their
/// rows cover about `prop` of the data; the other groups assess. Ids `Resample1`….
pub fn group_mc_cv(groups: &[u32], prop: f64, times: usize, seed: Seed) -> Result<Folds> {
    if !(prop > 0.0 && prop < 1.0) {
        return Err(Error::InvalidArgument(format!(
            "prop must be in (0, 1), got {prop}"
        )));
    }
    let table = check_groups(groups)?;
    if times == 0 {
        return Err(Error::InvalidArgument("times must be at least 1".into()));
    }
    let n = groups.len();
    let n_groups = table.len();
    let mut rng = seed.rng();
    let mut splits = Vec::with_capacity(times);
    for _ in 0..times {
        let mut draws: Vec<usize> = (0..n_groups).collect();
        draws.shuffle(&mut rng);
        let kept = balance_prop(&table, &draws, prop);
        let train = rows_of_draws(groups, &table, &kept);
        let mut in_train = vec![false; n];
        for &i in &train {
            in_train[i] = true;
        }
        let test: Vec<usize> = (0..n).filter(|i| !in_train[*i]).collect();
        if test.is_empty() {
            return Err(Error::InvalidArgument(
                "some assessment sets contained zero rows; consider a non-grouped resampling method".into(),
            ));
        }
        splits.push(Split::new(train, test, n)?);
    }
    Ok(Folds {
        splits,
        ids: names0(times, "Resample"),
        n,
    })
}

/// `rsample::group_initial_split`: [`group_mc_cv`] with `times = 1`.
pub fn group_initial_split(groups: &[u32], prop: f64, seed: Seed) -> Result<Split> {
    let f = group_mc_cv(groups, prop, 1, seed)?;
    Ok(f.splits.into_iter().next().expect("one split"))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn groups() -> Vec<u32> {
        // a×6, b×5, c×4, d×2, e×2, f×1 (rsample probe layout)
        let sizes = [6usize, 5, 4, 2, 2, 1];
        sizes
            .iter()
            .enumerate()
            .flat_map(|(g, s)| std::iter::repeat_n(g as u32, *s))
            .collect()
    }

    #[test]
    fn groups_never_straddle_folds_and_balance_rules_hold() {
        let g = groups();
        for balance in [GroupBalance::Groups, GroupBalance::Observations] {
            let f = group_vfold_cv(&g, Some(3), 1, balance, Seed(2)).unwrap();
            assert_eq!(f.ids, ["Resample1", "Resample2", "Resample3"]);
            assert!(f.is_partition());
            for s in &f.splits {
                let test_groups: std::collections::BTreeSet<u32> = s.test.iter().map(|i| g[*i]).collect();
                assert!(s.train.iter().all(|i| !test_groups.contains(&g[*i])));
                if balance == GroupBalance::Groups {
                    assert_eq!(test_groups.len(), 2);
                }
            }
            if balance == GroupBalance::Observations {
                // greedy fill: 20 rows over 3 folds; the exact layout depends on the
                // shuffle (checked on a fixed permutation below), but the folds never
                // differ by more than the largest group that is placed late
                let sizes: Vec<usize> = f.splits.iter().map(|s| s.test.len()).collect();
                assert_eq!(sizes.iter().sum::<usize>(), 20);
                assert!(
                    sizes.iter().max().unwrap() - sizes.iter().min().unwrap() <= 2,
                    "{sizes:?}"
                );
            }
        }
        // leave-one-group-out
        let f = group_vfold_cv(&g, None, 1, GroupBalance::Groups, Seed(0)).unwrap();
        assert_eq!(f.len(), 6);
        assert!(group_vfold_cv(&g, Some(7), 1, GroupBalance::Groups, Seed(0)).is_err());
        assert!(group_vfold_cv(&g, Some(6), 2, GroupBalance::Groups, Seed(0)).is_err());
        let r = group_vfold_cv(&g, Some(3), 2, GroupBalance::Groups, Seed(0)).unwrap();
        assert_eq!(r.ids[3], "Repeat2.Resample1");
    }

    #[test]
    fn observation_balance_matches_the_hand_computed_greedy() {
        // table a6 b5 c4 d2 e2 f1 in perm order f, a, b, c, d, e with v = 3, n = 20:
        // f->0, a->1, b->2; c (4): counts 1,6,5 -> errors pre |.05-.33|,.. -> fold 0;
        // d (2): counts 5,6,5 -> fold 0 (imp -0.1 vs +0.1); e (2): counts 7,6,5 -> fold 2
        let table = freq_table(&groups());
        let perm = [5, 0, 1, 2, 3, 4];
        assert_eq!(balance_observations(&table, &perm, 3, 20), vec![1, 2, 0, 0, 2, 0]);
    }

    #[test]
    fn prop_balance_crosses_target_or_stops_before() {
        let table = freq_table(&groups());
        // draws a(6) b(5) c(4): cumulative .3 .55 .75; prop .5 -> crosses at b (.55),
        // previous .3 is farther -> keep a, b
        assert_eq!(balance_prop(&table, &[0, 1, 2], 0.5), vec![0, 1]);
        // prop .35: crosses at b (.55, dist .2) vs a (.3, dist .05) -> keep a only
        assert_eq!(balance_prop(&table, &[0, 1, 2], 0.35), vec![0]);
        let g = groups();
        let b = group_bootstraps(&g, 3, Seed(1)).unwrap();
        for s in &b.splits {
            // every assessed row's group is absent from the analysis set
            let in_groups: std::collections::BTreeSet<u32> = s.train.iter().map(|i| g[*i]).collect();
            assert!(s.test.iter().all(|i| !in_groups.contains(&g[*i])));
            assert!(s.train.windows(2).all(|w| w[0] <= w[1]), "ascending with repeats");
        }
        let m = group_mc_cv(&g, 0.5, 2, Seed(1)).unwrap();
        assert!(m.splits.iter().all(|s| s.is_partition(20)));
        let s = group_initial_split(&g, 0.7, Seed(3)).unwrap();
        assert!(s.is_partition(20));
    }
}
