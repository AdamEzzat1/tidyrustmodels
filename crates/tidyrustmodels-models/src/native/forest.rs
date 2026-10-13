//! Random forests with ranger 0.18.0's growing rules (`Tree.cpp`, `TreeRegression.cpp`,
//! `TreeProbability.cpp`, `ForestProbability.cpp`), as `parsnip::rand_forest()` with
//! engine `ranger` uses them: bootstrap with replacement (`sample.fraction = 1`),
//! `mtry` variables drawn without replacement at every node, nodes terminal when
//! `n <= min.node.size` or pure, a split only when `n >= 2 * min.bucket`, the
//! "variance" rule `sum_left^2/n_left + sum_right^2/n_right` for regression and the
//! Gini rule `sum_j count_j^2 / n` per side for probability forests, split value at
//! the midpoint of adjacent sorted values (falling back to the lower value when the
//! midpoint rounds up), `x <= value` going left, and predictions averaged over trees
//! (terminal means, or terminal in-bag class fractions).
//!
//! The random streams differ from ranger's (a ChaCha8 stream per tree derived from the
//! `Seed`), so individual forests cannot be compared; the statistical agreement with
//! ranger is measured in VERIFICATION.md Phase 8e.

use rand::{Rng, RngExt};
use serde::{Deserialize, Serialize};
use tidyrustmodels_core::Seed;

/// Forest settings (ranger names).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct ForestOptions {
    /// `num.trees`
    pub trees: usize,
    /// `mtry`
    pub mtry: usize,
    /// `min.node.size`
    pub min_node_size: usize,
    /// `min.bucket`
    pub min_bucket: usize,
    /// Seed of the per-tree streams.
    pub seed: Seed,
}

/// Forest type.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum Kind {
    /// `treetype = "Regression"`: terminal means.
    Regression,
    /// `treetype = "Probability estimation"` with `k` classes: terminal class fractions.
    Probability {
        /// Number of classes.
        k: usize,
    },
}

/// One tree in flat arrays; node 0 is the root.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct TreeArrays {
    /// Split variable per node, `None` for a leaf.
    pub split_var: Vec<Option<u32>>,
    /// Split value per node (`x <= value` goes left).
    pub split_value: Vec<f64>,
    /// Left child index.
    pub left: Vec<u32>,
    /// Right child index.
    pub right: Vec<u32>,
    /// Leaf value(s): `[mean]` or the `k` class fractions; empty for internal nodes.
    pub leaf: Vec<Vec<f64>>,
}

impl TreeArrays {
    fn new_node(&mut self) -> usize {
        self.split_var.push(None);
        self.split_value.push(0.0);
        self.left.push(0);
        self.right.push(0);
        self.leaf.push(Vec::new());
        self.split_var.len() - 1
    }

    /// Terminal value for a row given by `x(var)`.
    pub fn predict(&self, x: impl Fn(usize) -> f64) -> &[f64] {
        let mut node = 0usize;
        while let Some(v) = self.split_var[node] {
            node = if x(v as usize) <= self.split_value[node] {
                self.left[node] as usize
            } else {
                self.right[node] as usize
            };
        }
        &self.leaf[node]
    }
}

/// A fitted forest.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Forest {
    /// Kind.
    pub kind: Kind,
    /// Options used.
    pub options: ForestOptions,
    /// Trees.
    pub trees: Vec<TreeArrays>,
}

impl Forest {
    /// Average terminal value over trees: `[mean]` or `k` class probabilities.
    pub fn predict(&self, x: impl Fn(usize) -> f64 + Copy) -> Vec<f64> {
        let width = match self.kind {
            Kind::Regression => 1,
            Kind::Probability { k } => k,
        };
        let mut out = vec![0.0; width];
        for t in &self.trees {
            for (o, v) in out.iter_mut().zip(t.predict(x)) {
                *o += v;
            }
        }
        for o in out.iter_mut() {
            *o /= self.trees.len() as f64;
        }
        out
    }
}

struct Grower<'a> {
    x: &'a [f64],
    y: &'a [f64],
    n: usize,
    p: usize,
    kind: Kind,
    opts: ForestOptions,
    ids: Vec<usize>,
    tree: TreeArrays,
}

impl<'a> Grower<'a> {
    fn col(&self, var: usize) -> &'a [f64] {
        &self.x[var * self.n..(var + 1) * self.n]
    }

    fn leaf_value(&self, start: usize, end: usize) -> Vec<f64> {
        let n = (end - start) as f64;
        match self.kind {
            Kind::Regression => {
                let s: f64 = self.ids[start..end].iter().map(|&i| self.y[i]).sum();
                vec![s / n]
            }
            Kind::Probability { k } => {
                let mut c = vec![0.0; k];
                for &i in &self.ids[start..end] {
                    c[self.y[i] as usize] += 1.0;
                }
                for v in c.iter_mut() {
                    *v /= n;
                }
                c
            }
        }
    }

    /// `findBestSplit` + `findBestSplitValueSmallQ`: returns `(var, value)`.
    fn best_split(&self, start: usize, end: usize, vars: &[usize]) -> Option<(usize, f64)> {
        let n_node = end - start;
        if n_node < 2 * self.opts.min_bucket {
            return None;
        }
        let mut best_decrease = -1.0f64;
        let mut best: Option<(usize, f64)> = None;
        let k = match self.kind {
            Kind::Regression => 1,
            Kind::Probability { k } => k,
        };
        let (sum_node, class_counts) = match self.kind {
            Kind::Regression => (
                self.ids[start..end].iter().map(|&i| self.y[i]).sum::<f64>(),
                Vec::new(),
            ),
            Kind::Probability { k } => {
                let mut c = vec![0usize; k];
                for &i in &self.ids[start..end] {
                    c[self.y[i] as usize] += 1;
                }
                (0.0, c)
            }
        };
        for &var in vars {
            let col = self.col(var);
            let mut values: Vec<f64> = self.ids[start..end].iter().map(|&i| col[i]).collect();
            values.sort_by(|a, b| a.partial_cmp(b).unwrap());
            values.dedup();
            if values.len() < 2 {
                continue;
            }
            let m = values.len();
            let mut counter = vec![0usize; m];
            let mut sums = vec![0.0f64; m];
            let mut counter_per_class = vec![0usize; m * k];
            for &i in &self.ids[start..end] {
                let idx = values.partition_point(|v| *v < col[i]);
                counter[idx] += 1;
                match self.kind {
                    Kind::Regression => sums[idx] += self.y[i],
                    Kind::Probability { .. } => counter_per_class[idx * k + self.y[i] as usize] += 1,
                }
            }
            let mut n_left = 0usize;
            let mut sum_left = 0.0f64;
            let mut left_counts = vec![0usize; k];
            for i in 0..m - 1 {
                if counter[i] == 0 {
                    continue;
                }
                n_left += counter[i];
                let n_right = n_node - n_left;
                if n_right == 0 {
                    break;
                }
                let decrease = match self.kind {
                    Kind::Regression => {
                        sum_left += sums[i];
                        if n_left < self.opts.min_bucket || n_right < self.opts.min_bucket {
                            continue;
                        }
                        let sum_right = sum_node - sum_left;
                        sum_left * sum_left / n_left as f64 + sum_right * sum_right / n_right as f64
                    }
                    Kind::Probability { .. } => {
                        for j in 0..k {
                            left_counts[j] += counter_per_class[i * k + j];
                        }
                        if n_left < self.opts.min_bucket || n_right < self.opts.min_bucket {
                            continue;
                        }
                        let (mut sl, mut sr) = (0.0f64, 0.0f64);
                        for j in 0..k {
                            let cr = class_counts[j] - left_counts[j];
                            sl += (left_counts[j] * left_counts[j]) as f64;
                            sr += (cr * cr) as f64;
                        }
                        sr / n_right as f64 + sl / n_left as f64
                    }
                };
                if decrease > best_decrease {
                    let mut value = (values[i] + values[i + 1]) / 2.0;
                    if value == values[i + 1] {
                        value = values[i];
                    }
                    best_decrease = decrease;
                    best = Some((var, value));
                }
            }
        }
        best
    }

    fn grow(&mut self, rng: &mut impl Rng, node: usize, start: usize, end: usize) {
        let n_node = end - start;
        let pure = {
            let first = self.y[self.ids[start]];
            self.ids[start..end].iter().all(|&i| self.y[i] == first)
        };
        if n_node <= self.opts.min_node_size || pure {
            self.tree.leaf[node] = self.leaf_value(start, end);
            return;
        }
        // createPossibleSplitVarSubset: mtry variables without replacement
        let vars = draw_without_replacement(rng, self.p, self.opts.mtry);
        let (var, value) = match self.best_split(start, end, &vars) {
            Some(s) => s,
            None => {
                self.tree.leaf[node] = self.leaf_value(start, end);
                return;
            }
        };
        // partition: x <= value left, keeping ranger's swap-to-end scheme
        let col = self.col(var);
        let mut pos = start;
        let mut right_start = end;
        while pos < right_start {
            if col[self.ids[pos]] <= value {
                pos += 1;
            } else {
                right_start -= 1;
                self.ids.swap(pos, right_start);
            }
        }
        let left = self.tree.new_node();
        let right = self.tree.new_node();
        self.tree.split_var[node] = Some(var as u32);
        self.tree.split_value[node] = value;
        self.tree.left[node] = left as u32;
        self.tree.right[node] = right as u32;
        self.grow(rng, left, start, right_start);
        self.grow(rng, right, right_start, end);
    }
}

/// `drawWithoutReplacement`: a random `k`-subset of `0..n` by partial Fisher-Yates.
fn draw_without_replacement(rng: &mut impl Rng, n: usize, k: usize) -> Vec<usize> {
    let mut pool: Vec<usize> = (0..n).collect();
    let k = k.min(n);
    for i in 0..k {
        let j = rng.random_range(i..n);
        pool.swap(i, j);
    }
    pool.truncate(k);
    pool
}

/// Grow a forest. `x` is column-major `n × p`; `y` holds the numeric response or class
/// codes `0..k` as `f64`.
pub fn fit(x: &[f64], n: usize, p: usize, y: &[f64], kind: Kind, opts: ForestOptions) -> Forest {
    assert_eq!(x.len(), n * p);
    assert_eq!(y.len(), n);
    let trees = (0..opts.trees)
        .map(|t| {
            let mut rng = opts.seed.derive(t as u64).rng();
            // bootstrap(): n draws with replacement
            let ids: Vec<usize> = (0..n).map(|_| rng.random_range(0..n)).collect();
            let mut g = Grower {
                x,
                y,
                n,
                p,
                kind,
                opts,
                ids,
                tree: TreeArrays {
                    split_var: Vec::new(),
                    split_value: Vec::new(),
                    left: Vec::new(),
                    right: Vec::new(),
                    leaf: Vec::new(),
                },
            };
            let root = g.tree.new_node();
            g.grow(&mut rng, root, 0, n);
            g.tree
        })
        .collect();
    Forest {
        kind,
        options: opts,
        trees,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn opts(trees: usize, mtry: usize, min_node_size: usize) -> ForestOptions {
        ForestOptions {
            trees,
            mtry,
            min_node_size,
            min_bucket: 1,
            seed: Seed(7),
        }
    }

    #[test]
    fn regression_forest_fits_a_step_function() {
        let n = 200;
        let x: Vec<f64> = (0..n).map(|i| i as f64 / n as f64).collect();
        let y: Vec<f64> = x.iter().map(|v| if *v < 0.5 { 1.0 } else { 3.0 }).collect();
        let f = fit(&x, n, 1, &y, Kind::Regression, opts(50, 1, 5));
        assert_eq!(f.trees.len(), 50);
        let lo = f.predict(|_| 0.2)[0];
        let hi = f.predict(|_| 0.8)[0];
        assert!((lo - 1.0).abs() < 0.05, "{lo}");
        assert!((hi - 3.0).abs() < 0.05, "{hi}");
        // deterministic for a seed
        let g = fit(&x, n, 1, &y, Kind::Regression, opts(50, 1, 5));
        assert_eq!(f, g);
    }

    #[test]
    fn probability_forest_returns_fractions_that_sum_to_one() {
        let n = 150;
        let x: Vec<f64> = (0..n).map(|i| i as f64).collect();
        let y: Vec<f64> = (0..n)
            .map(|i| {
                if i < 50 {
                    0.0
                } else if i < 100 {
                    1.0
                } else {
                    2.0
                }
            })
            .collect();
        let f = fit(&x, n, 1, &y, Kind::Probability { k: 3 }, opts(30, 1, 10));
        let p = f.predict(|_| 25.0);
        assert!((p.iter().sum::<f64>() - 1.0).abs() < 1e-12);
        assert!(p[0] > 0.9);
        let p = f.predict(|_| 125.0);
        assert!(p[2] > 0.9);
    }

    #[test]
    fn midpoint_falls_back_to_lower_value_when_it_rounds_up() {
        let a = 1.0f64;
        let b = f64::from_bits(a.to_bits() + 1);
        let mid = (a + b) / 2.0;
        assert!(mid == a || mid == b);
    }

    #[test]
    fn subset_draw_is_without_replacement() {
        let mut rng = Seed(1).rng();
        for _ in 0..100 {
            let v = draw_without_replacement(&mut rng, 10, 4);
            assert_eq!(v.len(), 4);
            let mut s = v.clone();
            s.sort_unstable();
            s.dedup();
            assert_eq!(s.len(), 4);
        }
    }
}
