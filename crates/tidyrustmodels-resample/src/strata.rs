//! `rsample:::make_strata` (rsample 1.3.2): stratum codes for a categorical or numeric
//! variable, with rsample's pooling of rare classes and its quantile binning.

use rand::seq::IndexedRandom;
use tidyrustmodels_core::{Error, Result, Seed};

/// `make_strata` for a factor / character variable: classes whose share of the rows
/// is below `pool` (default 0.1) are pooled, which rsample implements by setting them
/// to `NA` and then assigning each such row to one of the remaining classes uniformly
/// at random (`sample(levels(out), size = num_miss, replace = TRUE)`); a missing
/// value is treated the same way. When every class is below `pool` the data is too
/// small to stratify and every row gets stratum 0. Returns a code per row in the order
/// of the surviving levels (`levels(factor(as.character(x)))`, i.e. sorted labels).
pub fn make_strata_categorical(
    codes: &[Option<u32>],
    levels: &[String],
    pool: f64,
    seed: Seed,
) -> Result<Vec<u32>> {
    let n = codes.len();
    if n == 0 {
        return Err(Error::InvalidArgument("no rows".into()));
    }
    if !(0.0..1.0).contains(&pool) {
        return Err(Error::InvalidArgument(format!(
            "pool must be in [0, 1), got {pool}"
        )));
    }
    let mut counts = vec![0usize; levels.len()];
    for c in codes.iter().flatten() {
        let c = *c as usize;
        if c >= levels.len() {
            return Err(Error::InvalidArgument(format!("stratum code {c} has no level")));
        }
        counts[c] += 1;
    }
    let nf = n as f64;
    // `xtab <- sort(table(x))` over the observed classes; `pcts <- xtab / n`
    let observed: Vec<usize> = (0..levels.len()).filter(|k| counts[*k] > 0).collect();
    if observed.iter().all(|&k| (counts[k] as f64) / nf < pool) {
        // "Too little data to stratify." -> unstratified
        return Ok(vec![0; n]);
    }
    let keep: Vec<usize> = observed
        .iter()
        .copied()
        .filter(|&k| (counts[k] as f64) / nf >= pool)
        .collect();
    // surviving levels in sorted-label order (`factor(as.character(x))`)
    let mut order: Vec<usize> = keep.clone();
    order.sort_by(|a, b| levels[*a].cmp(&levels[*b]));
    let code_of = |k: usize| order.iter().position(|o| *o == k).map(|p| p as u32);
    let mut rng = seed.rng();
    let mut out = Vec::with_capacity(n);
    for c in codes {
        match c.and_then(|c| code_of(c as usize)) {
            Some(code) => out.push(code),
            None => out.push(
                *(0..order.len() as u32)
                    .collect::<Vec<u32>>()
                    .choose(&mut rng)
                    .expect("levels"),
            ),
        }
    }
    Ok(out)
}

/// `make_strata` for a numeric variable: `breaks` is reduced to at most
/// `floor(n / depth)` (depth 20); below two breaks the data is unstratified (all 0);
/// otherwise the rows are cut at the type-7 quantiles `(0:breaks)/breaks` of the
/// non-missing values (`unique`d, `include.lowest = TRUE`), and missing values are
/// assigned to a bin uniformly at random.
pub fn make_strata_numeric(
    values: &[Option<f64>],
    breaks: usize,
    depth: usize,
    seed: Seed,
) -> Result<Vec<u32>> {
    let n = values.len();
    if n == 0 {
        return Err(Error::InvalidArgument("no rows".into()));
    }
    if values.iter().flatten().any(|v| !v.is_finite()) {
        return Err(Error::InvalidValue("strata values must be finite".into()));
    }
    if breaks < 2 {
        return Ok(vec![0; n]);
    }
    let breaks = n.checked_div(depth).map_or(breaks, |cap| breaks.min(cap));
    if breaks < 2 {
        return Ok(vec![0; n]);
    }
    let mut sorted: Vec<f64> = values.iter().flatten().copied().collect();
    if sorted.is_empty() {
        return Ok(vec![0; n]);
    }
    sorted.sort_by(|a, b| a.partial_cmp(b).expect("finite"));
    let m = sorted.len();
    // type-7 quantiles at probs k / breaks
    let mut cuts: Vec<f64> = (0..=breaks)
        .map(|k| {
            let h = (m as f64 - 1.0) * (k as f64 / breaks as f64);
            let lo = h.floor() as usize;
            let hi = (lo + 1).min(m - 1);
            sorted[lo] + (h - lo as f64) * (sorted[hi] - sorted[lo])
        })
        .collect();
    cuts.dedup();
    let n_bins = cuts.len() - 1;
    if n_bins < 1 {
        return Ok(vec![0; n]);
    }
    let mut rng = seed.rng();
    let bins: Vec<u32> = (0..n_bins as u32).collect();
    Ok(values
        .iter()
        .map(|v| match v {
            None => *bins.choose(&mut rng).expect("bins"),
            Some(v) => {
                // intervals (c_k, c_{k+1}], first interval closed on the left
                let mut g = 0usize;
                for k in 0..n_bins {
                    let lo_ok = if k == 0 { *v >= cuts[0] } else { *v > cuts[k] };
                    if lo_ok && *v <= cuts[k + 1] {
                        g = k;
                        break;
                    }
                }
                g as u32
            }
        })
        .collect())
}

/// Bin a complete numeric variable into `breaks` quantile groups for stratification
/// with rsample's rules ([`make_strata_numeric`] with depth 20).
pub fn strata_from_numeric(values: &[f64], breaks: usize) -> Result<Vec<u32>> {
    if breaks < 2 {
        return Err(Error::InvalidArgument("breaks must be at least 2".into()));
    }
    let opt: Vec<Option<f64>> = values.iter().map(|v| Some(*v)).collect();
    make_strata_numeric(&opt, breaks, 20, Seed(0))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn numeric_strata_quartiles_and_depth_rule() {
        let v: Vec<Option<f64>> = (1..=100).map(|i| Some(i as f64)).collect();
        let s = make_strata_numeric(&v, 4, 20, Seed(0)).unwrap();
        let mut counts = [0; 4];
        for g in &s {
            counts[*g as usize] += 1;
        }
        assert_eq!(counts, [25, 25, 25, 25]);
        // n = 40: at most floor(40 / 20) = 2 breaks (rsample warns and uses 2)
        let v: Vec<Option<f64>> = (1..=40).map(|i| Some(i as f64)).collect();
        let s = make_strata_numeric(&v, 4, 20, Seed(0)).unwrap();
        assert_eq!(*s.iter().max().unwrap(), 1);
        assert_eq!(s.iter().filter(|g| **g == 0).count(), 20);
        // n = 30: floor(30 / 20) = 1 -> unstratified
        let v: Vec<Option<f64>> = (1..=30).map(|i| Some(i as f64)).collect();
        assert!(make_strata_numeric(&v, 4, 20, Seed(0))
            .unwrap()
            .iter()
            .all(|g| *g == 0));
        // missing values go to a random bin
        let mut v: Vec<Option<f64>> = (1..=100).map(|i| Some(i as f64)).collect();
        v[3] = None;
        let s = make_strata_numeric(&v, 4, 20, Seed(1)).unwrap();
        assert!(s[3] < 4);
        assert_eq!(strata_from_numeric(&[1.0, 2.0, 3.0], 2).unwrap(), vec![0, 0, 0]);
    }

    #[test]
    fn rare_classes_are_pooled_at_random() {
        let levels = vec!["a".to_string(), "b".to_string(), "c".to_string()];
        let codes: Vec<Option<u32>> = (0..100)
            .map(|i| {
                Some(if i < 50 {
                    0
                } else if i < 95 {
                    1
                } else {
                    2
                })
            })
            .collect();
        let s = make_strata_categorical(&codes, &levels, 0.1, Seed(2)).unwrap();
        assert!(s.iter().all(|g| *g < 2));
        assert_eq!(s[0], 0);
        assert_eq!(s[60], 1);
        let rare: Vec<u32> = s[95..].to_vec();
        assert!(rare.iter().all(|g| *g < 2));
        // all classes below the pool: unstratified
        let codes: Vec<Option<u32>> = (0..30).map(|i| Some(i as u32 % 3)).collect();
        let lv: Vec<String> = (0..3).map(|i| i.to_string()).collect();
        assert!(make_strata_categorical(&codes, &lv, 0.5, Seed(0))
            .unwrap()
            .iter()
            .all(|g| *g == 0));
    }
}
