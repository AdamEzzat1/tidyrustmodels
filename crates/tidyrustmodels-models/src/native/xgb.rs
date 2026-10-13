//! Port of xgboost 3.2.1's CPU `hist` tree booster for dense predictors without
//! missing values, single-threaded and depth-wise, with the parameters that
//! `parsnip::boost_tree()` exposes (ADR-0011 Amendment B).
//!
//! Sources transcribed (xgboost 3.2.1.1 as shipped on CRAN):
//! `src/common/quantile.{h,cc}` (weighted quantile sketch, pruning, merging and cut
//! points), `src/common/hist_util.{h,cc}` (bin search, histogram accumulation),
//! `src/tree/hist/evaluate_splits.h` and `src/tree/split_evaluator.h` (split scan,
//! gain), `src/tree/param.h` (weight, L1 threshold, split-entry replacement rule),
//! `src/tree/driver.h` and `src/tree/updater_quantile_hist.cc` (depth-wise
//! expansion, root statistics), `src/tree/hist/histogram.cc` (which child is built
//! and which is subtracted), `src/objective/{regression_obj.cu, multiclass_obj.cu,
//! init_estimation.cc}` and `src/common/stats.cc` (gradients and base score),
//! `src/common/math.h` (float sigmoid and softmax), `src/predictor/predict_fn.h`
//! (`x < split` goes left).
//!
//! Missing values: tidymodels hands xgboost a sparse matrix when workflows' sparsity
//! heuristic says so (`core::sparsity`), and then every exact zero of the design is a
//! missing value: it is left out of the sketch and the histograms, each split learns a
//! default direction (`default_left`) from the forward and backward scans, and
//! prediction follows that direction. `zero_as_missing` switches this on.
//!
//! Precision follows xgboost exactly: predictors, labels, gradients, split values,
//! leaf values, gains and predictions are `f32`; histogram and node sums are `f64`;
//! sketch ranks are `f32`. The order of every accumulation is the single-thread order
//! of the C++ code (rows ascending within a node, bins ascending within a feature).

use serde::{Deserialize, Serialize};

/// `kRtEps` in `include/xgboost/base.h`.
pub const K_RT_EPS: f32 = 1e-6;

/// Training parameters (xgboost names in the comments).
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct Params {
    /// `nrounds`.
    pub rounds: usize,
    /// `max_depth` (root at depth 0; leaves at this depth are never split).
    pub max_depth: u32,
    /// `eta` / `learning_rate`.
    pub eta: f32,
    /// `gamma` / `min_split_loss`.
    pub gamma: f32,
    /// `min_child_weight`.
    pub min_child_weight: f32,
    /// `lambda` / `reg_lambda`.
    pub lambda: f32,
    /// `alpha` / `reg_alpha`.
    pub alpha: f32,
    /// `max_bin`.
    pub max_bin: usize,
}

/// Objective (xgboost name): the gradient, the base score and the output transform.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum Objective {
    /// `reg:squarederror`.
    SquaredError,
    /// `binary:logistic`; labels are 1 for the event.
    Logistic,
    /// `multi:softprob` with `k` classes; labels are class codes `0..k`.
    Softprob {
        /// Number of classes.
        k: usize,
    },
}

impl Objective {
    /// Trees per round (`num_output_group`).
    pub fn groups(&self) -> usize {
        match self {
            Objective::Softprob { k } => *k,
            _ => 1,
        }
    }
}

/// One regression tree in xgboost's array layout (`learner.gradient_booster.model.trees`):
/// `split_conditions[i]` holds the split value of an internal node and the leaf value
/// (already scaled by `eta`) of a leaf; `left_children[i] == -1` marks a leaf.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize, Default)]
pub struct Tree {
    /// Split feature per node (0 for leaves).
    pub split_indices: Vec<u32>,
    /// Split value of an internal node, `eta`-scaled leaf value of a leaf.
    pub split_conditions: Vec<f32>,
    /// Left child or `-1`.
    pub left_children: Vec<i32>,
    /// Right child or `-1`.
    pub right_children: Vec<i32>,
    /// Unscaled weight of an internal node; the leaf value for a leaf.
    pub base_weights: Vec<f32>,
    /// Hessian sum of the node's rows (`float`, as `RTreeNodeStat::sum_hess`).
    pub sum_hessian: Vec<f32>,
    /// Gain of the node's split (0 for leaves).
    pub loss_changes: Vec<f32>,
    /// Direction taken by missing values (`default_left`).
    pub default_left: Vec<bool>,
}

impl Tree {
    fn alloc(&mut self) -> usize {
        self.split_indices.push(0);
        self.split_conditions.push(0.0);
        self.left_children.push(-1);
        self.right_children.push(-1);
        self.base_weights.push(0.0);
        self.sum_hessian.push(0.0);
        self.loss_changes.push(0.0);
        self.default_left.push(false);
        self.split_indices.len() - 1
    }

    /// Number of nodes.
    pub fn len(&self) -> usize {
        self.split_indices.len()
    }

    /// Whether the tree has no nodes (never true for a grown tree).
    pub fn is_empty(&self) -> bool {
        self.split_indices.is_empty()
    }

    /// Leaf reached by a row (`GetNextNode`: missing values take the default child,
    /// otherwise `x < split` goes left). With `zero_missing`, a zero is missing.
    pub fn leaf_for(&self, x: impl Fn(usize) -> f32, zero_missing: bool) -> usize {
        let mut nid = 0usize;
        while self.left_children[nid] != -1 {
            let f = self.split_indices[nid] as usize;
            let v = x(f);
            let go_left = if zero_missing && v == 0.0 {
                self.default_left[nid]
            } else {
                v < self.split_conditions[nid]
            };
            nid = if go_left {
                self.left_children[nid] as usize
            } else {
                self.right_children[nid] as usize
            };
        }
        nid
    }

    /// Leaf value of a row.
    pub fn predict(&self, x: impl Fn(usize) -> f32, zero_missing: bool) -> f32 {
        self.split_conditions[self.leaf_for(x, zero_missing)]
    }
}

/// A fitted booster: `trees[r * groups + g]` is the tree of round `r` for group `g`.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Booster {
    /// Objective the booster was trained with.
    pub objective: Objective,
    /// Training parameters.
    pub params: Params,
    /// `base_score` per group as xgboost stores it: the label mean for squared
    /// error, the event rate for logistic, centred log frequencies for softprob.
    pub base_score: Vec<f32>,
    /// Trees in training order.
    pub trees: Vec<Tree>,
    /// Number of predictors.
    pub n_features: usize,
    /// Whether exact zeros were (and are, at prediction) treated as missing.
    pub zero_as_missing: bool,
}

impl Booster {
    /// Raw margins (`output_margin = TRUE`): base score plus the leaves, accumulated in
    /// `f32` tree by tree as the predictor does.
    pub fn margin(&self, x: impl Fn(usize) -> f32) -> Vec<f32> {
        let k = self.objective.groups();
        let mut m = base_margin(self.objective, &self.base_score);
        for (t, tree) in self.trees.iter().enumerate() {
            m[t % k] += tree.predict(&x, self.zero_as_missing);
        }
        m
    }

    /// Transformed predictions: the value, the event probability, or the class
    /// probabilities (`PredTransform`).
    pub fn predict(&self, x: impl Fn(usize) -> f32) -> Vec<f32> {
        let mut m = self.margin(x);
        match self.objective {
            Objective::SquaredError => {}
            Objective::Logistic => m[0] = sigmoid(m[0]),
            Objective::Softprob { .. } => softmax(&mut m),
        }
        m
    }
}

/// `common::Sigmoid(float)`.
pub fn sigmoid(x: f32) -> f32 {
    let x = (-x).min(88.7f32);
    let denom = x.exp() + 1.0f32 + 1e-16f32;
    1.0f32 / denom
}

/// `common::Softmax` on floats (double accumulator for the normaliser).
pub fn softmax(v: &mut [f32]) {
    let mut wmax = v[0];
    for &x in &v[1..] {
        wmax = x.max(wmax);
    }
    let mut wsum = 0.0f64;
    for x in v.iter_mut() {
        *x = (*x - wmax).exp();
        wsum += *x as f64;
    }
    for x in v.iter_mut() {
        *x /= wsum as f32;
    }
}

// ---------------------------------------------------------------------------------
// Weighted quantile sketch (WQuantileSketch<float, float>)
// ---------------------------------------------------------------------------------

#[derive(Clone, Copy, Debug, Default, PartialEq)]
struct Entry {
    rmin: f32,
    rmax: f32,
    wmin: f32,
    value: f32,
}

impl Entry {
    fn rmin_next(&self) -> f32 {
        self.rmin + self.wmin
    }
    fn rmax_prev(&self) -> f32 {
        self.rmax - self.wmin
    }
}

#[derive(Clone, Debug, Default)]
struct Summary {
    data: Vec<Entry>,
}

impl Summary {
    fn len(&self) -> usize {
        self.data.len()
    }

    /// `WQSummary::SetPrune`.
    fn set_prune(&mut self, src: &Summary, maxsize: usize) {
        if src.len() <= maxsize {
            self.data = src.data.clone();
            return;
        }
        let begin = src.data[0].rmax;
        let range = src.data[src.len() - 1].rmin - src.data[0].rmax;
        let n = maxsize - 1;
        let mut out = Vec::with_capacity(maxsize);
        out.push(src.data[0]);
        let (mut i, mut lastidx) = (1usize, 0usize);
        for k in 1..n {
            let dx2 = 2.0f32 * ((k as f32 * range) / n as f32 + begin);
            while i < src.len() - 1 && dx2 >= src.data[i + 1].rmax + src.data[i + 1].rmin {
                i += 1;
            }
            if i == src.len() - 1 {
                break;
            }
            if dx2 < src.data[i].rmin_next() + src.data[i + 1].rmax_prev() {
                if i != lastidx {
                    out.push(src.data[i]);
                    lastidx = i;
                }
            } else if i + 1 != lastidx {
                out.push(src.data[i + 1]);
                lastidx = i + 1;
            }
        }
        if lastidx != src.len() - 1 {
            out.push(src.data[src.len() - 1]);
        }
        self.data = out;
    }

    /// `WQSummary::SetCombine`.
    fn set_combine(&mut self, sa: &Summary, sb: &Summary) {
        if sa.len() == 0 {
            self.data = sb.data.clone();
            return;
        }
        if sb.len() == 0 {
            self.data = sa.data.clone();
            return;
        }
        let (a, b) = (&sa.data, &sb.data);
        let (mut ia, mut ib) = (0usize, 0usize);
        let (mut aprev_rmin, mut bprev_rmin) = (0.0f32, 0.0f32);
        let mut out = Vec::with_capacity(a.len() + b.len());
        while ia < a.len() && ib < b.len() {
            let (ea, eb) = (a[ia], b[ib]);
            if ea.value == eb.value {
                out.push(Entry {
                    rmin: ea.rmin + eb.rmin,
                    rmax: ea.rmax + eb.rmax,
                    wmin: ea.wmin + eb.wmin,
                    value: ea.value,
                });
                aprev_rmin = ea.rmin_next();
                bprev_rmin = eb.rmin_next();
                ia += 1;
                ib += 1;
            } else if ea.value < eb.value {
                out.push(Entry {
                    rmin: ea.rmin + bprev_rmin,
                    rmax: ea.rmax + eb.rmax_prev(),
                    wmin: ea.wmin,
                    value: ea.value,
                });
                aprev_rmin = ea.rmin_next();
                ia += 1;
            } else {
                out.push(Entry {
                    rmin: eb.rmin + aprev_rmin,
                    rmax: eb.rmax + ea.rmax_prev(),
                    wmin: eb.wmin,
                    value: eb.value,
                });
                bprev_rmin = eb.rmin_next();
                ib += 1;
            }
        }
        if ia < a.len() {
            let brmax = b[b.len() - 1].rmax;
            while ia < a.len() {
                let ea = a[ia];
                out.push(Entry {
                    rmin: ea.rmin + bprev_rmin,
                    rmax: ea.rmax + brmax,
                    wmin: ea.wmin,
                    value: ea.value,
                });
                ia += 1;
            }
        }
        if ib < b.len() {
            let armax = a[a.len() - 1].rmax;
            while ib < b.len() {
                let eb = b[ib];
                out.push(Entry {
                    rmin: eb.rmin + aprev_rmin,
                    rmax: eb.rmax + armax,
                    wmin: eb.wmin,
                    value: eb.value,
                });
                ib += 1;
            }
        }
        self.data = out;
    }
}

/// `Queue::MakeSummary`: sort, merge equal values, exact ranks.
fn make_summary(queue: &[(f32, f32)]) -> Summary {
    let mut q = queue.to_vec();
    q.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());
    let mut data = Vec::with_capacity(q.len());
    let mut wsum = 0.0f32;
    let mut i = 0;
    while i < q.len() {
        let mut j = i + 1;
        let mut w = q[i].1;
        while j < q.len() && q[j].0 == q[i].0 {
            w += q[j].1;
            j += 1;
        }
        data.push(Entry {
            rmin: wsum,
            rmax: wsum + w,
            wmin: w,
            value: q[i].0,
        });
        wsum += w;
        i = j;
    }
    Summary { data }
}

/// `QuantileSketchTemplate` with `WQSummary` levels.
struct Sketch {
    limit_size: usize,
    queue: Vec<(f32, f32)>,
    qtail: usize,
    /// `level[0]` is scratch space; `levels` stays empty until the first `push_temp`.
    levels: Vec<Summary>,
    temp: Summary,
}

impl Sketch {
    /// `Init(maxn, eps)` followed by the host container's queue resize.
    fn new(maxn: usize, eps: f64) -> Self {
        let mut nlevel = 1usize;
        let mut limit_size;
        loop {
            limit_size = (nlevel as f64 / eps).ceil() as usize + 1;
            limit_size = limit_size.min(maxn);
            let n = 1usize << nlevel;
            if n * limit_size >= maxn {
                break;
            }
            nlevel += 1;
        }
        Sketch {
            limit_size,
            queue: vec![(0.0, 0.0); limit_size * 2],
            qtail: 0,
            levels: Vec::new(),
            temp: Summary::default(),
        }
    }

    /// `Push(x, w)`.
    fn push(&mut self, x: f32, w: f32) {
        if w == 0.0 {
            return;
        }
        if self.qtail == self.queue.len() && self.queue[self.qtail - 1].0 != x {
            self.temp = make_summary(&self.queue[..self.qtail]);
            self.qtail = 0;
            self.push_temp();
        }
        if self.qtail == 0 || self.queue[self.qtail - 1].0 != x {
            self.queue[self.qtail] = (x, w);
            self.qtail += 1;
        } else {
            self.queue[self.qtail - 1].1 += w;
        }
    }

    /// `PushTemp`.
    fn push_temp(&mut self) {
        let limit = self.limit_size;
        let mut l = 1usize;
        loop {
            if self.levels.len() < l + 1 {
                self.levels.resize(l + 1, Summary::default());
            }
            if self.levels[l].len() == 0 {
                let mut s = Summary::default();
                s.set_prune(&self.temp, limit);
                self.levels[l] = s;
                break;
            }
            let mut l0 = Summary::default();
            l0.set_prune(&self.temp, limit);
            self.levels[0] = l0;
            let mut t = Summary::default();
            t.set_combine(&self.levels[0], &self.levels[l]);
            self.temp = t;
            if self.temp.len() > limit {
                self.levels[l].data.clear();
            } else {
                self.levels[l] = self.temp.clone();
                break;
            }
            l += 1;
        }
    }

    /// `GetSummary`.
    fn summary(&self) -> Summary {
        let limit = self.limit_size;
        let mut out = make_summary(&self.queue[..self.qtail]);
        if !self.levels.is_empty() {
            let mut l0 = Summary::default();
            l0.set_prune(&out, limit);
            for l in 1..self.levels.len() {
                if self.levels[l].len() == 0 {
                    continue;
                }
                if l0.len() == 0 {
                    l0 = self.levels[l].clone();
                } else {
                    let mut t = Summary::default();
                    t.set_combine(&l0, &self.levels[l]);
                    out = t;
                    let mut p = Summary::default();
                    p.set_prune(&out, limit);
                    l0 = p;
                }
            }
            l0
        } else if out.len() > limit {
            let mut t = Summary::default();
            t.set_prune(&out, limit);
            t
        } else {
            out
        }
    }
}

/// `HistogramCuts`: per-feature cut values (`ptrs[f]..ptrs[f + 1]`), the last one
/// above every observed value.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Cuts {
    /// Offsets of each feature's cuts in `values`.
    pub ptrs: Vec<u32>,
    /// Cut values, ascending within a feature.
    pub values: Vec<f32>,
    /// A value below each feature's minimum (`min_vals_`).
    pub min_vals: Vec<f32>,
}

impl Cuts {
    /// `SketchOnDMatrix` + `MakeCuts` for unweighted data; `x` is column-major
    /// `n × p` in `f32`. With `zero_missing`, zeros are absent from the sketch.
    pub fn make(x: &[f32], n: usize, p: usize, max_bin: usize, zero_missing: bool) -> Cuts {
        Self::make_weighted(x, n, p, max_bin, zero_missing, None)
    }

    /// [`Cuts::make`] with DMatrix row weights: `PushRowPage` pushes every present
    /// value with its row's weight (`info.weights_`), which the weighted quantile
    /// sketch turns into weighted ranks; the column size (hence `eps`) still counts
    /// the present values.
    pub fn make_weighted(
        x: &[f32],
        n: usize,
        p: usize,
        max_bin: usize,
        zero_missing: bool,
        weights: Option<&[f32]>,
    ) -> Cuts {
        let mut cuts = Cuts {
            ptrs: vec![0],
            values: Vec::new(),
            min_vals: Vec::with_capacity(p),
        };
        for j in 0..p {
            let present: Vec<(f32, f32)> = x[j * n..(j + 1) * n]
                .iter()
                .enumerate()
                .filter(|(_, v)| !(zero_missing && **v == 0.0))
                .map(|(i, v)| (*v, weights.map_or(1.0f32, |w| w[i])))
                .collect();
            let csize = present.len();
            // HostSketchContainer: eps from min(max_bin, column size), at least 1 bin
            let n_bins = max_bin.min(csize).max(1);
            let eps = 1.0f64 / ((n_bins as f32) * 8.0f32) as f64;
            let mut sk = Sketch::new(csize, eps);
            for &(v, w) in &present {
                sk.push(v, w);
            }
            let out = sk.summary();
            // AllReduce: intermediate_num_cuts = min(column size, max_bin * kFactor)
            let intermediate = csize.min(max_bin * 8);
            let mut reduced = Summary::default();
            reduced.set_prune(&out, intermediate);
            // MakeCuts
            let max_num_bins = intermediate.min(max_bin);
            let mut a = Summary::default();
            a.set_prune(&reduced, max_num_bins + 1);
            let min_val = if intermediate != 0 {
                let mval = a.data[0].value;
                mval - mval.abs() - 1e-5f32
            } else {
                1e-5f32
            };
            cuts.min_vals.push(min_val);
            // AddCutPoint: the minimum is implicit, start from the second entry
            let required = a.len().min(max_num_bins);
            for i in 1..required {
                let cpt = a.data[i].value;
                if i == 1 || cpt > *cuts.values.last().unwrap() {
                    cuts.values.push(cpt);
                }
            }
            let cpt = if a.len() > 0 {
                a.data[a.len() - 1].value
            } else {
                min_val
            };
            let last = cpt + (cpt.abs() + 1e-5f32);
            cuts.values.push(last);
            cuts.ptrs.push(cuts.values.len() as u32);
        }
        cuts
    }

    /// Total number of bins.
    pub fn total_bins(&self) -> usize {
        self.values.len()
    }

    /// `SearchBin`: the first cut above the value (global bin index).
    pub fn search_bin(&self, value: f32, feature: usize) -> u32 {
        let beg = self.ptrs[feature] as usize;
        let end = self.ptrs[feature + 1] as usize;
        let mut idx = beg + self.values[beg..end].partition_point(|c| *c <= value);
        if idx == end {
            idx -= 1;
        }
        idx as u32
    }
}

// ---------------------------------------------------------------------------------
// Gradients and base score
// ---------------------------------------------------------------------------------

/// `GetGradient` for the objective; `preds` and `out` are `n × groups` row-major.
/// With DMatrix weights the pairs are scaled by the row weight: `(g * w, h * w)` for
/// the regression losses (`RegLossObj`, the hessian floor applied before the weight)
/// and, for softprob, `(g * w, max(2 p (1 - p) * w, eps))` (`SoftmaxMultiClassObj`).
fn gradients(obj: Objective, preds: &[f32], labels: &[f32], weights: Option<&[f32]>, out: &mut [(f32, f32)]) {
    let wt = |i: usize| weights.map_or(1.0f32, |w| w[i]);
    match obj {
        Objective::SquaredError => {
            for (i, &y) in labels.iter().enumerate() {
                let w = wt(i);
                out[i] = ((preds[i] - y) * w, 1.0f32 * w);
            }
        }
        Objective::Logistic => {
            for (i, &y) in labels.iter().enumerate() {
                let p = sigmoid(preds[i]);
                let w = wt(i);
                out[i] = ((p - y) * w, (p * (1.0f32 - p)).max(1e-16f32) * w);
            }
        }
        Objective::Softprob { k } => {
            for (i, &y) in labels.iter().enumerate() {
                let point = &preds[i * k..(i + 1) * k];
                let mut wmax = f32::MIN_POSITIVE;
                for &v in point {
                    wmax = v.max(wmax);
                }
                let mut wsum = 0.0f64;
                for &v in point {
                    wsum += (v - wmax).exp() as f64;
                }
                let label = y as usize;
                let w = wt(i);
                for (c, &v) in point.iter().enumerate() {
                    let p = (v - wmax).exp() / (wsum as f32);
                    let h = (2.0f32 * p * (1.0f32 - p) * w).max(1e-16f32);
                    let g = if label == c { p - 1.0f32 } else { p };
                    out[i * k + c] = (g * w, h);
                }
            }
        }
    }
}

/// `ProbToMargin`: the stored base score on the margin scale (logit for logistic,
/// identity otherwise).
fn base_margin(obj: Objective, base: &[f32]) -> Vec<f32> {
    match obj {
        Objective::Logistic => base
            .iter()
            .map(|b| {
                let b = b.clamp(K_RT_EPS, 1.0f32 - K_RT_EPS);
                -(1.0f32 / b - 1.0f32).ln()
            })
            .collect(),
        _ => base.to_vec(),
    }
}

/// `InitEstimation` (`boost_from_average`) on the scale xgboost stores: the label
/// mean (squared error and logistic) or the centred log class frequencies. With
/// DMatrix weights: `common::WeightedSampleMean` (`sum_i y_i / sum_w * w_i` in
/// double) and, for softprob, the weighted class histogram (`SmallHistogram`, float)
/// over the double weight sum.
fn base_score(obj: Objective, labels: &[f32], weights: Option<&[f32]>) -> Vec<f32> {
    let n = labels.len();
    match obj {
        Objective::SquaredError | Objective::Logistic => {
            let mut m = 0.0f64;
            match weights {
                None => {
                    // common::SampleMean: double accumulation of y / n
                    let nf = n as f64;
                    for &y in labels {
                        m += y as f64 / nf;
                    }
                }
                Some(w) => {
                    let sum_w: f64 = w.iter().fold(0.0f64, |acc, v| acc + *v as f64);
                    for (&y, &wi) in labels.iter().zip(w) {
                        m += y as f64 / sum_w * wi as f64;
                    }
                }
            }
            vec![m as f32]
        }
        Objective::Softprob { k } => {
            let mut counts = vec![0.0f32; k];
            for (i, &y) in labels.iter().enumerate() {
                counts[y as usize] += weights.map_or(1.0f32, |w| w[i]);
            }
            let sum_weight = match weights {
                None => n as f64,
                Some(w) => w.iter().fold(0.0f64, |acc, v| acc + *v as f64),
            };
            let mut v: Vec<f32> = counts
                .iter()
                .map(|c| (*c as f64 * (1.0f64 / sum_weight)) as f32)
                .collect();
            for x in v.iter_mut() {
                *x = (*x + K_RT_EPS).ln();
            }
            let nf = k as f32;
            let mut mean = 0.0f32;
            for &x in &v {
                mean += x / nf;
            }
            for x in v.iter_mut() {
                *x -= mean;
            }
            v
        }
    }
}

// ---------------------------------------------------------------------------------
// Tree growing
// ---------------------------------------------------------------------------------

/// `ThresholdL1`.
fn threshold_l1(w: f64, alpha: f32) -> f64 {
    let a = alpha as f64;
    if w > a {
        w - a
    } else if w < -a {
        w + a
    } else {
        0.0
    }
}

/// `tree::CalcWeight` on double sums, narrowed to `float` by the evaluator.
fn calc_weight(p: &Params, g: f64, h: f64) -> f32 {
    if h < p.min_child_weight as f64 || h <= 0.0 {
        return 0.0;
    }
    let dw = -threshold_l1(g, p.alpha) / (h + p.lambda as f64);
    dw as f32
}

/// `SplitEvaluator::CalcGainGivenWeight` without constraints or `max_delta_step`:
/// `Divide(Sqr(ThresholdL1(G)), H + lambda)` with both operands narrowed to `float`.
fn calc_gain(p: &Params, g: f64, h: f64) -> f32 {
    if h <= 0.0 {
        return 0.0;
    }
    let t = threshold_l1(g, p.alpha);
    let num = (t * t) as f32;
    let den = (h + p.lambda as f64) as f32;
    num / den
}

#[derive(Clone, Copy, Debug)]
struct NodeStats {
    g: f64,
    h: f64,
    root_gain: f32,
}

#[derive(Clone, Copy, Debug, Default)]
struct Split {
    loss_chg: f32,
    sindex: u32,
    value: f32,
    default_left: bool,
    left: (f64, f64),
    right: (f64, f64),
}

impl Split {
    /// `SplitEntryContainer::NeedReplace` (kept literally, including its NaN behaviour).
    #[allow(clippy::neg_cmp_op_on_partial_ord)]
    fn need_replace(&self, new_loss: f32, split_index: u32) -> bool {
        if new_loss.is_infinite() {
            false
        } else if self.sindex <= split_index {
            new_loss > self.loss_chg
        } else {
            !(self.loss_chg > new_loss)
        }
    }

    fn update(&mut self, other: &Split) -> bool {
        if self.need_replace(other.loss_chg, other.sindex) {
            *self = *other;
            true
        } else {
            false
        }
    }
}

#[derive(Clone, Copy, Debug)]
struct Candidate {
    nid: usize,
    depth: u32,
    split: Split,
}

impl Candidate {
    /// `CPUExpandEntry::IsValidImpl`.
    fn is_valid(&self, p: &Params) -> bool {
        if self.split.loss_chg <= K_RT_EPS {
            return false;
        }
        if self.split.left.1 == 0.0 || self.split.right.1 == 0.0 {
            return false;
        }
        if self.split.loss_chg < p.gamma {
            return false;
        }
        if p.max_depth > 0 && self.depth == p.max_depth {
            return false;
        }
        true
    }
}

/// Bin index of a missing entry.
const MISSING: u32 = u32::MAX;

struct Grower<'a> {
    gidx: &'a [u32],
    dense: bool,
    cuts: &'a Cuts,
    gpair: &'a [(f32, f32)],
    p: usize,
    params: Params,
    tree: Tree,
    rows: Vec<Vec<usize>>,
    hists: Vec<Vec<(f64, f64)>>,
    stats: Vec<NodeStats>,
    depth: Vec<u32>,
}

impl<'a> Grower<'a> {
    /// `BuildHist`: double sums of the float pairs, rows then features in order.
    fn build_hist(&self, rows: &[usize]) -> Vec<(f64, f64)> {
        let mut hist = vec![(0.0f64, 0.0f64); self.cuts.total_bins()];
        for &r in rows {
            let (g, h) = self.gpair[r];
            let base = r * self.p;
            for j in 0..self.p {
                let bin = self.gidx[base + j];
                if bin == MISSING {
                    continue;
                }
                hist[bin as usize].0 += g as f64;
                hist[bin as usize].1 += h as f64;
            }
        }
        hist
    }

    /// `EvaluateSplits` for one node: the forward scan of every feature, and the
    /// backward scan (missing values on the left) whenever the feature's histogram sum
    /// differs from the node's statistics (`SplitContainsMissingValues`, an exact
    /// comparison that round-off alone can trip).
    #[allow(clippy::needless_range_loop)]
    fn evaluate(&self, nid: usize) -> Candidate {
        let parent = self.stats[nid];
        let hist = &self.hists[nid];
        let mcw = self.params.min_child_weight as f64;
        let mut best = Split::default();
        for f in 0..self.p {
            let beg = self.cuts.ptrs[f] as usize;
            let end = self.cuts.ptrs[f + 1] as usize;
            // forward: split at the right bound of each bin, missing go right
            let mut fbest = Split::default();
            let (mut lg, mut lh) = (0.0f64, 0.0f64);
            for i in beg..end {
                lg += hist[i].0;
                lh += hist[i].1;
                let (rg, rh) = (parent.g - lg, parent.h - lh);
                if lh >= mcw && rh >= mcw {
                    let gain = calc_gain(&self.params, lg, lh) + calc_gain(&self.params, rg, rh);
                    let loss_chg = gain - parent.root_gain;
                    let cand = Split {
                        loss_chg,
                        sindex: f as u32,
                        value: self.cuts.values[i],
                        default_left: false,
                        left: (lg, lh),
                        right: (rg, rh),
                    };
                    fbest.update(&cand);
                }
            }
            best.update(&fbest);
            if lg == parent.g && lh == parent.h {
                continue;
            }
            // backward: split at the left bound of each bin, missing go left
            let mut bbest = Split::default();
            let (mut tg, mut th) = (0.0f64, 0.0f64);
            let mut i = end;
            while i > beg {
                i -= 1;
                tg += hist[i].0;
                th += hist[i].1;
                let (rg, rh) = (parent.g - tg, parent.h - th);
                if th >= mcw && rh >= mcw {
                    let gain = calc_gain(&self.params, rg, rh) + calc_gain(&self.params, tg, th);
                    let loss_chg = gain - parent.root_gain;
                    let value = if i == beg {
                        self.cuts.min_vals[f]
                    } else {
                        self.cuts.values[i - 1]
                    };
                    let cand = Split {
                        loss_chg,
                        sindex: f as u32,
                        value,
                        default_left: true,
                        left: (rg, rh),
                        right: (tg, th),
                    };
                    bbest.update(&cand);
                }
            }
            best.update(&bbest);
        }
        Candidate {
            nid,
            depth: self.depth[nid],
            split: best,
        }
    }

    /// `ApplyTreeSplit` + `RegTree::ExpandNode`.
    fn apply(&mut self, c: &Candidate) {
        let p = self.params;
        let parent = self.stats[c.nid];
        let base_weight = calc_weight(&p, parent.g, parent.h);
        let lw = calc_weight(&p, c.split.left.0, c.split.left.1);
        let rw = calc_weight(&p, c.split.right.0, c.split.right.1);
        let left = self.tree.alloc();
        let right = self.tree.alloc();
        let t = &mut self.tree;
        t.split_indices[c.nid] = c.split.sindex;
        t.split_conditions[c.nid] = c.split.value;
        t.left_children[c.nid] = left as i32;
        t.right_children[c.nid] = right as i32;
        t.loss_changes[c.nid] = c.split.loss_chg;
        t.default_left[c.nid] = c.split.default_left;
        t.sum_hessian[c.nid] = parent.h as f32;
        t.base_weights[c.nid] = base_weight;
        let (ll, rl) = (lw * p.eta, rw * p.eta);
        t.split_conditions[left] = ll;
        t.split_conditions[right] = rl;
        t.sum_hessian[left] = c.split.left.1 as f32;
        t.sum_hessian[right] = c.split.right.1 as f32;
        t.base_weights[left] = ll;
        t.base_weights[right] = rl;
        self.stats.push(NodeStats {
            g: c.split.left.0,
            h: c.split.left.1,
            root_gain: calc_gain(&p, c.split.left.0, c.split.left.1),
        });
        self.stats.push(NodeStats {
            g: c.split.right.0,
            h: c.split.right.1,
            root_gain: calc_gain(&p, c.split.right.0, c.split.right.1),
        });
        self.depth.push(c.depth + 1);
        self.depth.push(c.depth + 1);
        self.rows.push(Vec::new());
        self.rows.push(Vec::new());
        self.hists.push(Vec::new());
        self.hists.push(Vec::new());
    }

    /// `UpdatePosition`: `bin <= split bin` goes left, order preserved.
    fn partition(&mut self, c: &Candidate) {
        let f = c.split.sindex as usize;
        let beg = self.cuts.ptrs[f] as usize;
        let end = self.cuts.ptrs[f + 1] as usize;
        let mut split_bin: i64 = -1;
        for b in beg..end {
            if self.cuts.values[b] == c.split.value {
                split_bin = b as i64;
            }
        }
        let rows = std::mem::take(&mut self.rows[c.nid]);
        let (mut l, mut r) = (Vec::new(), Vec::new());
        for row in rows {
            let bin = self.gidx[row * self.p + f];
            let go_left = if bin == MISSING {
                c.split.default_left
            } else {
                (bin as i64) <= split_bin
            };
            if go_left {
                l.push(row);
            } else {
                r.push(row);
            }
        }
        let left = self.tree.left_children[c.nid] as usize;
        let right = self.tree.right_children[c.nid] as usize;
        self.rows[left] = l;
        self.rows[right] = r;
        self.rows[c.nid] = Vec::new();
    }

    /// `BuildHistLeftRight` with `AssignNodes`: build the child with the smaller hessian
    /// sum (the left on ties), subtract for the other.
    fn child_hists(&mut self, c: &Candidate) {
        let left = self.tree.left_children[c.nid] as usize;
        let right = self.tree.right_children[c.nid] as usize;
        let fewer_right = c.split.right.1 < c.split.left.1;
        let (build, sub) = if fewer_right { (right, left) } else { (left, right) };
        let built = self.build_hist(&self.rows[build]);
        let subtracted: Vec<(f64, f64)> = self.hists[c.nid]
            .iter()
            .zip(&built)
            .map(|(a, b)| (a.0 - b.0, a.1 - b.1))
            .collect();
        self.hists[build] = built;
        self.hists[sub] = subtracted;
    }

    /// `Driver::Pop` for depth-wise growth: the smallest node ids first, one depth at
    /// a time, invalid entries dropped.
    fn pop(&self, queue: &mut Vec<Candidate>) -> Vec<Candidate> {
        if queue.is_empty() {
            return Vec::new();
        }
        queue.sort_by_key(|c| c.nid);
        let level = queue[0].depth;
        let mut out = Vec::new();
        while !queue.is_empty() && queue[0].depth == level {
            let c = queue.remove(0);
            if c.is_valid(&self.params) {
                out.push(c);
            }
        }
        out
    }

    fn grow(mut self, n: usize) -> Tree {
        let p = self.params;
        let root = self.tree.alloc();
        self.rows.push((0..n).collect());
        let hist = self.build_hist(&self.rows[root]);
        // dense data: the root sum is the sum of the first feature's bins; otherwise
        // the gradient pairs are summed row by row
        let (mut g, mut h) = (0.0f64, 0.0f64);
        if self.dense {
            for bin in &hist[self.cuts.ptrs[0] as usize..self.cuts.ptrs[1] as usize] {
                g += bin.0;
                h += bin.1;
            }
        } else {
            for &(gg, hh) in self.gpair {
                g += gg as f64;
                h += hh as f64;
            }
        }
        self.hists.push(hist);
        let weight = calc_weight(&p, g, h);
        self.stats.push(NodeStats {
            g,
            h,
            root_gain: calc_gain(&p, g, h),
        });
        self.depth.push(0);
        self.tree.sum_hessian[root] = h as f32;
        self.tree.base_weights[root] = weight;
        self.tree.split_conditions[root] = p.eta * weight;
        let mut queue = vec![self.evaluate(root)];
        let mut expand = self.pop(&mut queue);
        while !expand.is_empty() {
            let mut valid = Vec::new();
            for c in &expand {
                self.apply(c);
                let child_valid = !(p.max_depth > 0 && c.depth + 1 >= p.max_depth);
                if child_valid {
                    valid.push(*c);
                }
            }
            for c in &expand {
                self.partition(c);
            }
            let mut best = Vec::new();
            for c in &valid {
                self.child_hists(c);
                let left = self.tree.left_children[c.nid] as usize;
                let right = self.tree.right_children[c.nid] as usize;
                best.push(self.evaluate(left));
                best.push(self.evaluate(right));
            }
            for b in best {
                if b.split.loss_chg > K_RT_EPS {
                    queue.push(b);
                }
            }
            expand = self.pop(&mut queue);
        }
        self.tree
    }
}

/// Train a booster. `x` is column-major `n × p` (converted to `f32` as `xgb.DMatrix`
/// does), `labels` the numeric response, the 0/1 event indicator, or the class codes.
/// With `zero_as_missing`, every exact zero of `x` is a missing value (the sparse
/// matrix tidymodels passes to xgboost).
pub fn fit(
    x: &[f64],
    n: usize,
    p: usize,
    labels: &[f64],
    obj: Objective,
    params: Params,
    zero_as_missing: bool,
) -> Booster {
    fit_weighted(x, n, p, labels, obj, params, zero_as_missing, None)
}

/// [`fit`] with DMatrix row weights (`xgb.DMatrix(weight = )`): the quantile sketch,
/// the gradient pairs (hence `min_child_weight` and every leaf) and the base score are
/// weighted as xgboost does.
#[allow(clippy::too_many_arguments)]
pub fn fit_weighted(
    x: &[f64],
    n: usize,
    p: usize,
    labels: &[f64],
    obj: Objective,
    params: Params,
    zero_as_missing: bool,
    weights: Option<&[f64]>,
) -> Booster {
    assert_eq!(x.len(), n * p);
    assert_eq!(labels.len(), n);
    let xf: Vec<f32> = x.iter().map(|v| *v as f32).collect();
    let yf: Vec<f32> = labels.iter().map(|v| *v as f32).collect();
    let wf: Option<Vec<f32>> = weights.map(|w| {
        assert_eq!(w.len(), n);
        w.iter().map(|v| *v as f32).collect()
    });
    let cuts = Cuts::make_weighted(&xf, n, p, params.max_bin, zero_as_missing, wf.as_deref());
    let mut gidx = vec![0u32; n * p];
    let mut any_missing = false;
    for i in 0..n {
        for j in 0..p {
            let v = xf[j * n + i];
            gidx[i * p + j] = if zero_as_missing && v == 0.0 {
                any_missing = true;
                MISSING
            } else {
                cuts.search_bin(v, j)
            };
        }
    }
    let k = obj.groups();
    let base = base_score(obj, &yf, wf.as_deref());
    let margin0 = base_margin(obj, &base);
    let mut preds = vec![0.0f32; n * k];
    for i in 0..n {
        for g in 0..k {
            preds[i * k + g] = margin0[g];
        }
    }
    let mut gpair = vec![(0.0f32, 0.0f32); n * k];
    let mut group_pairs = vec![(0.0f32, 0.0f32); n];
    let mut trees = Vec::with_capacity(params.rounds * k);
    for _ in 0..params.rounds {
        gradients(obj, &preds, &yf, wf.as_deref(), &mut gpair);
        for g in 0..k {
            for i in 0..n {
                group_pairs[i] = gpair[i * k + g];
            }
            let grower = Grower {
                gidx: &gidx,
                dense: !any_missing,
                cuts: &cuts,
                gpair: &group_pairs,
                p,
                params,
                tree: Tree::default(),
                rows: Vec::new(),
                hists: Vec::new(),
                stats: Vec::new(),
                depth: Vec::new(),
            };
            let tree = grower.grow(n);
            // UpdatePredictionCache: add the new leaves in float
            for i in 0..n {
                preds[i * k + g] += tree.predict(|f| xf[f * n + i], zero_as_missing);
            }
            trees.push(tree);
        }
    }
    Booster {
        objective: obj,
        params,
        base_score: base,
        trees,
        n_features: p,
        zero_as_missing,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn params(rounds: usize, depth: u32) -> Params {
        Params {
            rounds,
            max_depth: depth,
            eta: 0.3,
            gamma: 0.0,
            min_child_weight: 1.0,
            lambda: 1.0,
            alpha: 0.0,
            max_bin: 256,
        }
    }

    #[test]
    fn integer_weights_equal_replicated_rows_for_one_tree() {
        // x: 12 rows; weights 0..3; a weighted single-round fit must equal the fit on
        // the replicated rows: same cuts (weighted ranks), same histograms, same
        // leaves and base score (the sums are exact in float for these small integers)
        let n = 12;
        let x: Vec<f64> = (0..n).map(|i| (i as f64 * 0.7).sin() * 3.0).collect();
        let y: Vec<f64> = x.iter().map(|v| v * v).collect();
        let w: Vec<f64> = (0..n).map(|i| (i % 4) as f64).collect();
        let (mut xr, mut yr) = (Vec::new(), Vec::new());
        for i in 0..n {
            for _ in 0..w[i] as usize {
                xr.push(x[i]);
                yr.push(y[i]);
            }
        }
        let a = fit_weighted(
            &x,
            n,
            1,
            &y,
            Objective::SquaredError,
            params(1, 2),
            false,
            Some(&w),
        );
        let b = fit(
            &xr,
            xr.len(),
            1,
            &yr,
            Objective::SquaredError,
            params(1, 2),
            false,
        );
        assert_eq!(a.base_score, b.base_score);
        assert_eq!(a.trees[0].split_conditions, b.trees[0].split_conditions);
        assert_eq!(a.trees[0].sum_hessian, b.trees[0].sum_hessian);
        assert_eq!(a.trees[0].left_children, b.trees[0].left_children);
    }

    #[test]
    fn cuts_are_the_sorted_unique_values_above_the_minimum_plus_a_sentinel() {
        // 5 rows, values with a tie: cuts skip the minimum and end above the maximum
        let x = [3.0f32, 1.0, 2.0, 2.0, 5.0];
        let c = Cuts::make(&x, 5, 1, 256, false);
        assert_eq!(c.ptrs, vec![0, 4]);
        assert_eq!(&c.values[..3], &[2.0, 3.0, 5.0]);
        assert!(c.values[3] > 5.0);
        assert_eq!(c.min_vals[0], 1.0 - 1.0 - 1e-5);
        // bins: 1 -> 0, 2 -> 1, 3 -> 2, 5 -> 3
        assert_eq!(c.search_bin(1.0, 0), 0);
        assert_eq!(c.search_bin(2.0, 0), 1);
        assert_eq!(c.search_bin(2.5, 0), 1);
        assert_eq!(c.search_bin(5.0, 0), 3);
        assert_eq!(c.search_bin(9.0, 0), 3);
    }

    #[test]
    fn sketch_prunes_to_max_bin_when_there_are_more_unique_values() {
        let n = 3000usize;
        let x: Vec<f32> = (0..n).map(|i| i as f32 * 0.5).collect();
        let c = Cuts::make(&x, n, 1, 256, false);
        let m = c.values.len();
        assert!(m <= 256, "{m}");
        assert!(m >= 200, "{m}");
        for w in c.values.windows(2) {
            assert!(w[0] < w[1]);
        }
    }

    #[test]
    fn one_round_depth_one_reproduces_the_closed_form_stump() {
        // y = 0 for x < 5, 10 for x >= 5; base = mean = 5; residuals -5 / +5 (float)
        let n = 10usize;
        let x: Vec<f64> = (0..n).map(|i| i as f64).collect();
        let y: Vec<f64> = (0..n).map(|i| if i < 5 { 0.0 } else { 10.0 }).collect();
        let b = fit(&x, n, 1, &y, Objective::SquaredError, params(1, 1), false);
        assert_eq!(b.base_score, vec![5.0]);
        let t = &b.trees[0];
        assert_eq!(t.len(), 3);
        assert_eq!(t.split_indices[0], 0);
        assert_eq!(t.split_conditions[0], 5.0); // x < 5 goes left
                                                // leaf = eta * (-G / (H + lambda)) = 0.3 * (25 / 6) on the right
        let expect = 0.3f32 * (25.0f32 / 6.0f32);
        assert!(
            (t.split_conditions[2] - expect).abs() < 1e-6,
            "{}",
            t.split_conditions[2]
        );
        assert!((t.split_conditions[1] + expect).abs() < 1e-6);
        assert_eq!(t.sum_hessian[0], 10.0);
        // gain = G_L^2/(H_L+1) + G_R^2/(H_R+1) - G^2/(H+1) = 2 * 625/6 - 0
        assert!((t.loss_changes[0] - 2.0 * 625.0 / 6.0).abs() < 1e-3);
        let pred = b.predict(|_| 7.0)[0];
        assert!((pred - (5.0 + expect)).abs() < 1e-6);
    }

    #[test]
    fn logistic_and_softprob_outputs_are_probabilities() {
        let n = 40usize;
        let x: Vec<f64> = (0..n).map(|i| i as f64).collect();
        let y: Vec<f64> = (0..n).map(|i| if i < 20 { 1.0 } else { 0.0 }).collect();
        let b = fit(&x, n, 1, &y, Objective::Logistic, params(5, 2), false);
        let p_low = b.predict(|_| 3.0)[0];
        let p_high = b.predict(|_| 30.0)[0];
        assert!(p_low > 0.8 && p_high < 0.2, "{p_low} {p_high}");
        let y3: Vec<f64> = (0..n).map(|i| (i / 14) as f64).collect();
        let b3 = fit(&x, n, 1, &y3, Objective::Softprob { k: 3 }, params(5, 2), false);
        assert_eq!(b3.trees.len(), 15);
        let pr = b3.predict(|_| 35.0);
        let s: f32 = pr.iter().sum();
        assert!((s - 1.0).abs() < 1e-6);
        assert!(pr[2] > 0.6, "{pr:?}");
    }

    #[test]
    fn gamma_and_depth_stop_growth() {
        let n = 10usize;
        let x: Vec<f64> = (0..n).map(|i| i as f64).collect();
        let y: Vec<f64> = (0..n).map(|i| if i < 5 { 0.0 } else { 10.0 }).collect();
        let mut pr = params(1, 6);
        pr.gamma = 1e9;
        let b = fit(&x, n, 1, &y, Objective::SquaredError, pr, false);
        assert_eq!(b.trees[0].len(), 1);
        let b2 = fit(&x, n, 1, &y, Objective::SquaredError, params(1, 2), false);
        assert!(b2.trees[0].len() <= 7);
    }

    #[test]
    fn zeros_as_missing_use_the_sentinel_cut_and_a_default_direction() {
        // a 0/1 indicator: with zeros missing, the only cut is the sentinel above 1 and
        // the zero rows follow the learned default direction (right here)
        let n = 10usize;
        let x: Vec<f64> = (0..n).map(|i| if i < 5 { 0.0 } else { 1.0 }).collect();
        let y: Vec<f64> = (0..n).map(|i| if i < 5 { 0.0 } else { 10.0 }).collect();
        let b = fit(&x, n, 1, &y, Objective::SquaredError, params(1, 1), true);
        let t = &b.trees[0];
        assert_eq!(t.len(), 3);
        assert_eq!(t.split_conditions[0], 1.0f32 + (1.0f32 + 1e-5f32));
        assert!(!t.default_left[0]);
        let leaf = 0.3f32 * (25.0f32 / 6.0f32);
        assert!(
            (t.split_conditions[1] - leaf).abs() < 1e-6,
            "{}",
            t.split_conditions[1]
        );
        assert!(
            (t.split_conditions[2] + leaf).abs() < 1e-6,
            "{}",
            t.split_conditions[2]
        );
        assert_eq!(b.predict(|_| 1.0)[0], 5.0f32 + leaf);
        assert_eq!(b.predict(|_| 0.0)[0], 5.0f32 - leaf);
        // dense treatment of the same data splits at 1.0 with the zeros on the left
        let d = fit(&x, n, 1, &y, Objective::SquaredError, params(1, 1), false);
        assert_eq!(d.trees[0].split_conditions[0], 1.0);
        assert_eq!(d.predict(|_| 0.0)[0], 5.0f32 - leaf);
    }
}
