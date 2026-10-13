//! Time-series resampling: `rolling_origin`, `sliding_window`, `sliding_index` and
//! `initial_time_split` (rsample 1.3.2; the sliding functions follow slider's window
//! and completeness rules). Everything here is deterministic.

use crate::{names0, Folds, Split};
use tidyrustmodels_core::{Error, Result};

/// `rsample::rolling_origin(initial, assess, cumulative, skip, lag)`: analysis windows
/// end at `initial, initial + skip + 1, …, n - assess` (1-based), starting at row 1
/// when `cumulative` or `initial` rows before the end otherwise; the assessment window
/// is the next `assess` rows, extended back by `lag` rows. Ids `Slice1`….
pub fn rolling_origin(
    n: usize,
    initial: usize,
    assess: usize,
    cumulative: bool,
    skip: usize,
    lag: usize,
) -> Result<Folds> {
    if initial == 0 || assess == 0 {
        return Err(Error::InvalidArgument(
            "initial and assess must be positive".into(),
        ));
    }
    if n < initial + assess {
        return Err(Error::InvalidArgument(format!(
            "there should be at least {} rows in the data",
            initial + assess
        )));
    }
    if lag > initial {
        return Err(Error::InvalidArgument(
            "lag must be less than or equal to the number of training observations".into(),
        ));
    }
    let mut splits = Vec::new();
    let mut stop = initial; // 1-based inclusive end of the analysis window
    while stop <= n - assess {
        let start = if cumulative { 1 } else { stop - initial + 1 };
        let train: Vec<usize> = (start..=stop).map(|r| r - 1).collect();
        let test: Vec<usize> = (stop + 1 - lag..=stop + assess).map(|r| r - 1).collect();
        splits.push(Split::new(train, test, n)?);
        stop += skip + 1;
    }
    let ids = names0(splits.len(), "Slice");
    Ok(Folds { splits, ids, n })
}

/// Apply `slice_skip` then `slice_step`.
fn skip_and_step(
    mut windows: Vec<(Vec<usize>, Vec<usize>)>,
    skip: usize,
    step: usize,
) -> Vec<(Vec<usize>, Vec<usize>)> {
    if skip > 0 {
        windows = windows.into_iter().skip(skip).collect();
    }
    if step > 1 {
        windows = windows.into_iter().step_by(step).collect();
    }
    windows
}

/// `rsample::sliding_window(lookback, assess_start, assess_stop, complete, step,
/// skip)`: for every row `i`, the analysis window is `[i - lookback, i]` (`lookback =
/// None` means everything before, slider's `Inf`) and the assessment window
/// `[i + assess_start, i + assess_stop]`; windows that run past either end are
/// dropped (`complete = TRUE`; with `complete = FALSE` an analysis window may be
/// truncated at the start but the assessment window must still fit). Then the first
/// `skip` windows are dropped and every `step`-th of the rest kept. Ids `Slice1`….
pub fn sliding_window(
    n: usize,
    lookback: Option<usize>,
    assess_start: usize,
    assess_stop: usize,
    complete: bool,
    step: usize,
    skip: usize,
) -> Result<Folds> {
    if assess_start == 0 || assess_stop == 0 {
        return Err(Error::InvalidArgument(
            "assess_start and assess_stop must be positive".into(),
        ));
    }
    if assess_start > assess_stop {
        return Err(Error::InvalidArgument(
            "assess_start must be less than or equal to assess_stop".into(),
        ));
    }
    if step == 0 {
        return Err(Error::InvalidArgument("step must be positive".into()));
    }
    let mut windows = Vec::new();
    for i in 1..=n {
        // analysis window
        let start = match lookback {
            None => 1,
            Some(lb) => {
                if i <= lb {
                    if complete {
                        continue;
                    }
                    1
                } else {
                    i - lb
                }
            }
        };
        // assessment window must be complete
        if i + assess_stop > n {
            continue;
        }
        let train: Vec<usize> = (start..=i).map(|r| r - 1).collect();
        let test: Vec<usize> = (i + assess_start..=i + assess_stop).map(|r| r - 1).collect();
        windows.push((train, test));
    }
    let windows = skip_and_step(windows, skip, step);
    if windows.is_empty() {
        return Err(Error::InvalidArgument("no complete windows".into()));
    }
    let splits: Vec<Split> = windows
        .into_iter()
        .map(|(train, test)| Split::new(train, test, n))
        .collect::<Result<_>>()?;
    let ids = names0(splits.len(), "Slice");
    Ok(Folds { splits, ids, n })
}

/// `rsample::sliding_index(index, lookback, assess_start, assess_stop, complete,
/// step, skip)` for a numeric, non-decreasing index: windows are measured in index
/// units (`[x_i - lookback, x_i]` and `[x_i + assess_start, x_i + assess_stop]`), a
/// window is complete when its range lies within `[x_1, x_n]` (slider), and an
/// assessment window with no rows in its range is kept empty, as slider does. Ids
/// `Slice1`….
pub fn sliding_index(
    index: &[f64],
    lookback: Option<f64>,
    assess_start: f64,
    assess_stop: f64,
    complete: bool,
    step: usize,
    skip: usize,
) -> Result<Folds> {
    let n = index.len();
    if n == 0 {
        return Err(Error::InvalidArgument("no rows".into()));
    }
    if index.iter().any(|v| !v.is_finite()) {
        return Err(Error::InvalidValue("index must be finite".into()));
    }
    if index.windows(2).any(|w| w[0] > w[1]) {
        return Err(Error::InvalidArgument(
            "index must be sorted in ascending order".into(),
        ));
    }
    if !(assess_start > 0.0 && assess_stop > 0.0) {
        return Err(Error::InvalidArgument(
            "assess_start and assess_stop must be positive".into(),
        ));
    }
    if assess_start > assess_stop {
        return Err(Error::InvalidArgument(
            "assess_start must be less than or equal to assess_stop".into(),
        ));
    }
    if let Some(lb) = lookback {
        if lb < 0.0 {
            return Err(Error::InvalidArgument(
                "lookback must be positive, or zero".into(),
            ));
        }
    }
    if step == 0 {
        return Err(Error::InvalidArgument("step must be positive".into()));
    }
    let (lo, hi) = (index[0], index[n - 1]);
    let mut windows = Vec::new();
    for i in 0..n {
        let x = index[i];
        let in_lo = match lookback {
            None => lo,
            Some(lb) => {
                if complete && x - lb < lo {
                    continue;
                }
                x - lb
            }
        };
        if x + assess_stop > hi {
            continue;
        }
        let train: Vec<usize> = (0..n).filter(|j| index[*j] >= in_lo && index[*j] <= x).collect();
        let test: Vec<usize> = (0..n)
            .filter(|j| index[*j] >= x + assess_start && index[*j] <= x + assess_stop)
            .collect();
        windows.push((train, test));
    }
    let windows = skip_and_step(windows, skip, step);
    if windows.is_empty() {
        return Err(Error::InvalidArgument("no complete windows".into()));
    }
    let splits: Vec<Split> = windows
        .into_iter()
        .map(|(train, test)| Split::new(train, test, n))
        .collect::<Result<_>>()?;
    let ids = names0(splits.len(), "Slice");
    Ok(Folds { splits, ids, n })
}

/// `rsample::initial_time_split(prop)`: the first `floor(n * prop)` rows train, the
/// rest test.
pub fn initial_time_split(n: usize, prop: f64) -> Result<Split> {
    if !(prop > 0.0 && prop < 1.0) {
        return Err(Error::InvalidArgument(format!(
            "prop must be in (0, 1), got {prop}"
        )));
    }
    let n_train = (n as f64 * prop).floor() as usize;
    if n_train == 0 || n_train >= n {
        return Err(Error::InvalidArgument("too few rows for the split".into()));
    }
    Split::from_ids((0..n_train).collect(), (n_train..n).collect(), n)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn ids1(v: &[usize]) -> Vec<usize> {
        v.iter().map(|r| r + 1).collect()
    }

    #[test]
    fn rolling_origin_matches_rsample_probe() {
        // rsample: n = 8, initial 3, assess 2, cumulative FALSE, skip 1, lag 1
        let f = rolling_origin(8, 3, 2, false, 1, 1).unwrap();
        assert_eq!(f.len(), 2);
        assert_eq!(ids1(&f.splits[0].train), [1, 2, 3]);
        assert_eq!(ids1(&f.splits[0].test), [3, 4, 5]);
        assert_eq!(ids1(&f.splits[1].train), [3, 4, 5]);
        assert_eq!(ids1(&f.splits[1].test), [5, 6, 7]);
        assert_eq!(f.ids, ["Slice1", "Slice2"]);
        let c = rolling_origin(8, 3, 2, true, 0, 0).unwrap();
        assert_eq!(c.len(), 4);
        assert_eq!(ids1(&c.splits[3].train), [1, 2, 3, 4, 5, 6]);
        assert_eq!(ids1(&c.splits[3].test), [7, 8]);
        assert!(rolling_origin(4, 3, 2, true, 0, 0).is_err());
        assert!(rolling_origin(8, 3, 2, true, 0, 4).is_err());
    }

    #[test]
    fn sliding_window_matches_rsample_probe() {
        // lookback 2, assess 1..2, complete, step 2, skip 1 -> (2-4|5-6), (4-6|7-8)
        let f = sliding_window(8, Some(2), 1, 2, true, 2, 1).unwrap();
        assert_eq!(f.len(), 2);
        assert_eq!(ids1(&f.splits[0].train), [2, 3, 4]);
        assert_eq!(ids1(&f.splits[0].test), [5, 6]);
        assert_eq!(ids1(&f.splits[1].train), [4, 5, 6]);
        assert_eq!(ids1(&f.splits[1].test), [7, 8]);
        // complete = FALSE: truncated analysis windows kept
        let g = sliding_window(8, Some(2), 1, 1, false, 1, 0).unwrap();
        assert_eq!(g.len(), 7);
        assert_eq!(ids1(&g.splits[0].train), [1]);
        assert_eq!(ids1(&g.splits[0].test), [2]);
        assert_eq!(ids1(&g.splits[2].train), [1, 2, 3]);
        // lookback Inf (cumulative)
        let h = sliding_window(8, None, 1, 2, true, 1, 0).unwrap();
        assert_eq!(h.len(), 6);
        assert_eq!(ids1(&h.splits[5].train), [1, 2, 3, 4, 5, 6]);
        assert_eq!(ids1(&h.splits[5].test), [7, 8]);
        assert!(sliding_window(8, Some(2), 2, 1, true, 1, 0).is_err());
    }

    #[test]
    fn sliding_index_matches_rsample_probe() {
        // index 1 2 2 5 6 9 10 11, lookback 3, assess 1..2
        let idx = [1.0, 2.0, 2.0, 5.0, 6.0, 9.0, 10.0, 11.0];
        let f = sliding_index(&idx, Some(3.0), 1.0, 2.0, true, 1, 0).unwrap();
        assert_eq!(f.len(), 3);
        assert_eq!(ids1(&f.splits[0].train), [2, 3, 4]);
        assert_eq!(ids1(&f.splits[0].test), [5]);
        assert_eq!(ids1(&f.splits[1].train), [4, 5]);
        assert!(f.splits[1].test.is_empty(), "no rows in [7, 8]");
        assert_eq!(ids1(&f.splits[2].train), [5, 6]);
        assert_eq!(ids1(&f.splits[2].test), [7, 8]);
        // integer index equals sliding_window
        let seq: Vec<f64> = (1..=8).map(|i| i as f64).collect();
        let a = sliding_index(&seq, Some(2.0), 1.0, 1.0, true, 1, 0).unwrap();
        let b = sliding_window(8, Some(2), 1, 1, true, 1, 0).unwrap();
        assert_eq!(a.splits, b.splits);
        assert!(sliding_index(&[2.0, 1.0], Some(1.0), 1.0, 1.0, true, 1, 0).is_err());
    }

    #[test]
    fn initial_time_split_takes_the_first_rows() {
        let s = initial_time_split(8, 0.6).unwrap();
        assert_eq!(s.train, [0, 1, 2, 3]);
        assert_eq!(s.test, [4, 5, 6, 7]);
        assert!(initial_time_split(1, 0.5).is_err());
    }
}
