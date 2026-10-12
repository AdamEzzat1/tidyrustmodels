//! A port of rpart 4.1.23's recursive partitioning (`rpart.c`, `partition.c`,
//! `bsplit.c`, `nodesplit.c`, `anova.c`, `gini.c`, `rpmatrix.c`, `pred_rpart.c`) for
//! **continuous predictors without missing values**, which is what a baked recipe
//! produces here (nominals are dummy-encoded, missing values imputed). Categorical
//! splits (`graycode.c`), surrogates and cross-validated cp tables are therefore not
//! ported; the fitted tree is the one `rpart()` returns, since the C code already
//! collapses every node whose complexity is at or below `cp * root risk`.
//!
//! Classification uses rpart's default parameters: priors equal to the (weighted)
//! class frequencies and a 0/1 loss, under which `gini.c`'s altered priors are 1 up to
//! round-off (they are reproduced exactly, see `altered_priors`), the split criterion
//! is the weighted Gini impurity and a node's risk is its weighted misclassification
//! count. Case weights enter every sum as in rpart.
//!
//! Tie rules preserved from the C source: among equal improvements the lowest variable
//! index wins (`insert_split`), among equal split points within a variable the first in
//! sorted order wins (`temp > best` / `temp < best`), the split value is the midpoint of
//! the adjacent sorted values, and the side with the smaller mean (anova) or smaller
//! mean class index (gini) is sent left.

use serde::{Deserialize, Serialize};

const LEFT: i8 = -1;
const RIGHT: i8 = 1;

/// `rpart.control` subset used by the port.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct Control {
    /// `minsplit`: minimum observations for a node to be considered for splitting.
    pub minsplit: usize,
    /// `minbucket`: minimum observations in a child.
    pub minbucket: usize,
    /// `cp`: complexity parameter relative to the root risk.
    pub cp: f64,
    /// `maxdepth`: maximum depth, root = depth 0; rpart caps it at 30.
    pub maxdepth: usize,
}

/// Splitting method.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum Method {
    /// `method = "anova"`: weighted sum of squares.
    Anova,
    /// `method = "class"` with Gini splitting and `k` classes (codes `1..=k` in `y`).
    Gini {
        /// Number of classes.
        k: usize,
    },
}

/// A node's response estimate.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub enum NodeValue {
    /// Weighted mean (`anova`).
    Mean(f64),
    /// Predicted class (1-based) and weighted class counts (`class`).
    Class {
        /// 1-based class index with the smallest loss (first on ties).
        class: usize,
        /// Weighted count per class.
        counts: Vec<f64>,
    },
}

/// The primary split of a node.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Split {
    /// 0-based predictor index.
    pub var: usize,
    /// Split point: observations with `x < point` form one side.
    pub point: f64,
    /// Which side `x < point` goes to: `-1` left, `1` right (rpart's `ncat` column).
    pub direction: i8,
    /// rpart's `improve` as stored in `fit$splits`.
    pub improve: f64,
    /// Observations with a non-missing, positive-weight value of the variable.
    pub count: usize,
}

/// A tree node, numbered as rpart does (`1`, children `2k` and `2k + 1`).
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Node {
    /// rpart node number.
    pub number: u32,
    /// Observations.
    pub n: usize,
    /// Sum of case weights.
    pub sum_wt: f64,
    /// Risk (`dev` in `frame`).
    pub risk: f64,
    /// Complexity after `make_cp_list`'s fix-up, scaled by the root risk (`frame$complexity`).
    pub complexity: f64,
    /// Response estimate.
    pub value: NodeValue,
    /// Primary split, if the node has children.
    pub split: Option<Split>,
    /// Left child.
    pub left: Option<Box<Node>>,
    /// Right child.
    pub right: Option<Box<Node>>,
}

impl Node {
    fn leaf(number: u32) -> Node {
        Node {
            number,
            n: 0,
            sum_wt: 0.0,
            risk: 0.0,
            complexity: 0.0,
            value: NodeValue::Mean(0.0),
            split: None,
            left: None,
            right: None,
        }
    }

    /// Whether the node has no children.
    pub fn is_leaf(&self) -> bool {
        self.left.is_none()
    }

    /// Pre-order traversal (rpart's `frame` row order).
    pub fn preorder(&self) -> Vec<&Node> {
        let mut out = Vec::new();
        self.walk(&mut out);
        out
    }

    fn walk<'a>(&'a self, out: &mut Vec<&'a Node>) {
        out.push(self);
        if let Some(l) = &self.left {
            l.walk(out);
        }
        if let Some(r) = &self.right {
            r.walk(out);
        }
    }
}

/// A fitted rpart-style tree.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Tree {
    /// Method used.
    pub method: Method,
    /// Control used.
    pub control: Control,
    /// Root node.
    pub root: Node,
}

impl Tree {
    /// Walk `pred_rpart`'s rule down to the leaf for a row given by `x(var)`.
    pub fn leaf_for(&self, x: impl Fn(usize) -> f64) -> &Node {
        let mut node = &self.root;
        while let Some(s) = &node.split {
            let dir = if x(s.var) < s.point {
                s.direction
            } else {
                -s.direction
            };
            node = if dir == LEFT {
                node.left.as_ref().unwrap()
            } else {
                node.right.as_ref().unwrap()
            };
        }
        node
    }

    /// Number of splits (internal nodes).
    pub fn n_splits(&self) -> usize {
        self.root.preorder().iter().filter(|n| !n.is_leaf()).count()
    }
}

struct Rp<'a> {
    x: &'a [f64],
    y: &'a [f64],
    wt: &'a [f64],
    n: usize,
    p: usize,
    method: Method,
    control: Control,
    aprior: Vec<f64>,
    alpha: f64,
    maxnode: f64,
    iscale: f64,
    /// `rp.sorts`: per variable, observation indices sorted by that variable, reordered
    /// in place as nodes are split so that `[n1, n2)` holds a node's observations.
    sorts: Vec<Vec<usize>>,
    which: Vec<u32>,
    // scratch
    xtemp: Vec<f64>,
    ytemp: Vec<f64>,
    wtemp: Vec<f64>,
}

fn eval(method: Method, y: &[f64], wt: &[f64]) -> (NodeValue, f64) {
    match method {
        Method::Anova => {
            // anovass
            let (mut temp, mut twt) = (0.0, 0.0);
            for (yi, w) in y.iter().zip(wt) {
                temp += yi * w;
                twt += w;
            }
            let mean = temp / twt;
            let mut ss = 0.0;
            for (yi, w) in y.iter().zip(wt) {
                let t = yi - mean;
                ss += t * t * w;
            }
            (NodeValue::Mean(mean), ss)
        }
        Method::Gini { k } => {
            // ginidev with prior = 1 and 0/1 loss
            let mut freq = vec![0.0; k];
            for (yi, w) in y.iter().zip(wt) {
                freq[*yi as usize - 1] += w;
            }
            let (mut max, mut dev) = (0usize, 0.0);
            for i in 0..k {
                let mut temp = 0.0;
                for (j, f) in freq.iter().enumerate() {
                    if j != i {
                        temp += f;
                    }
                }
                if i == 0 || temp < dev {
                    max = i;
                    dev = temp;
                }
            }
            (
                NodeValue::Class {
                    class: max + 1,
                    counts: freq,
                },
                dev,
            )
        }
    }
}

/// `anova()` for a continuous predictor: returns `(improve, split, direction)`.
fn choose_anova(y: &[f64], x: &[f64], wt: &[f64], edge: usize, myrisk: f64) -> Option<(f64, f64, i8)> {
    let n = y.len();
    let (mut right_wt, mut right_sum) = (0.0, 0.0);
    for (yi, w) in y.iter().zip(wt) {
        right_sum += yi * w;
        right_wt += w;
    }
    let grandmean = right_sum / right_wt;
    let (mut left_sum, mut left_wt, mut left_n) = (0.0, 0.0, 0usize);
    let mut right_n = n;
    right_sum = 0.0;
    let mut best = 0.0;
    let mut where_ = 0usize;
    let mut direction = LEFT;
    let mut i = 0;
    while right_n > edge {
        left_wt += wt[i];
        right_wt -= wt[i];
        left_n += 1;
        right_n -= 1;
        let temp = (y[i] - grandmean) * wt[i];
        left_sum += temp;
        right_sum -= temp;
        if x[i + 1] != x[i] && left_n >= edge {
            let temp = left_sum * left_sum / left_wt + right_sum * right_sum / right_wt;
            if temp > best {
                best = temp;
                where_ = i;
                direction = if left_sum < right_sum { LEFT } else { RIGHT };
            }
        }
        i += 1;
    }
    if best > 0.0 {
        Some((best / myrisk, (x[where_] + x[where_ + 1]) / 2.0, direction))
    } else {
        None
    }
}

/// `giniinit`'s altered priors for the default `parms` (priors = weighted class
/// frequencies, 0/1 loss). Mathematically every entry is 1, but R computes them as
/// `(2 * prior_i) / (sum_of_2K_terms * freq_i)` and the sum of the class proportions
/// need not round to exactly 1, so the factors carry an ulp that decides exact ties
/// between candidate splits; they are reproduced operation for operation.
fn altered_priors(k: usize, y: &[f64], wt: &[f64]) -> Vec<f64> {
    let mut freq = vec![0.0; k];
    let mut total = 0.0;
    for (yi, w) in y.iter().zip(wt) {
        freq[*yi as usize - 1] += w;
        total += w;
    }
    // rpart.class: prior = counts / sum(counts)
    let counts = freq.clone();
    let sum_counts: f64 = counts.iter().sum();
    let mut prior: Vec<f64> = counts.iter().map(|c| c / sum_counts).collect();
    for f in freq.iter_mut() {
        *f /= total;
    }
    let mut aprior = vec![0.0; k];
    let mut temp = 0.0;
    for i in 0..k {
        for j in 0..k {
            let loss = if i == j { 0.0 } else { 1.0 };
            temp += loss * prior[i];
            aprior[i] += loss * prior[i];
        }
    }
    for i in 0..k {
        if freq[i] > 0.0 {
            prior[i] /= freq[i];
            aprior[i] /= temp * freq[i];
        }
    }
    aprior
}

/// `gini()` for a continuous predictor with the altered priors of `giniinit`.
fn choose_gini(
    k: usize,
    aprior: &[f64],
    y: &[f64],
    x: &[f64],
    wt: &[f64],
    edge: usize,
) -> Option<(f64, f64, i8)> {
    let mut left = vec![0.0; k];
    let mut right = vec![0.0; k];
    let (mut lwt, mut rwt) = (0.0, 0.0);
    let (mut ltot, mut rtot) = (0usize, 0usize);
    for (yi, w) in y.iter().zip(wt) {
        let j = *yi as usize - 1;
        rwt += aprior[j] * w;
        right[j] += w;
        rtot += 1;
    }
    let mut total_ss = 0.0;
    for (i, r) in right.iter().enumerate() {
        let temp = aprior[i] * r / rwt;
        total_ss += rwt * (temp * (1.0 - temp));
    }
    let mut best = total_ss;
    let mut where_ = 0usize;
    let mut direction = LEFT;
    let mut i = 0;
    while rtot > edge {
        let j = y[i] as usize - 1;
        rwt -= aprior[j] * wt[i];
        lwt += aprior[j] * wt[i];
        rtot -= 1;
        ltot += 1;
        right[j] -= wt[i];
        left[j] += wt[i];
        if ltot >= edge && x[i + 1] != x[i] {
            let (mut temp, mut lmean, mut rmean) = (0.0, 0.0, 0.0);
            for j in 0..k {
                let p = aprior[j] * left[j] / lwt;
                temp += lwt * (p * (1.0 - p));
                lmean += p * j as f64;
                let p = aprior[j] * right[j] / rwt;
                temp += rwt * (p * (1.0 - p));
                rmean += p * j as f64;
            }
            if temp < best {
                best = temp;
                where_ = i;
                direction = if lmean < rmean { LEFT } else { RIGHT };
            }
        }
        i += 1;
    }
    let improve = total_ss - best;
    if improve > 0.0 {
        Some((improve, (x[where_] + x[where_ + 1]) / 2.0, direction))
    } else {
        None
    }
}

impl<'a> Rp<'a> {
    fn col(&self, var: usize) -> &'a [f64] {
        &self.x[var * self.n..(var + 1) * self.n]
    }

    /// `bsplit`: best primary split over all variables.
    fn bsplit(&mut self, risk: f64, n1: usize, n2: usize) -> Option<Split> {
        let mut primary: Option<Split> = None;
        for var in 0..self.p {
            let col = self.col(var);
            let mut k = 0;
            for &obs in &self.sorts[var][n1..n2] {
                if self.wt[obs] > 0.0 {
                    self.xtemp[k] = col[obs];
                    self.ytemp[k] = self.y[obs];
                    self.wtemp[k] = self.wt[obs];
                    k += 1;
                }
            }
            if k == 0 || self.xtemp[0] == self.xtemp[k - 1] {
                continue;
            }
            let edge = self.control.minbucket;
            let found = match self.method {
                Method::Anova => {
                    choose_anova(&self.ytemp[..k], &self.xtemp[..k], &self.wtemp[..k], edge, risk)
                }
                Method::Gini { k: nc } => choose_gini(
                    nc,
                    &self.aprior,
                    &self.ytemp[..k],
                    &self.xtemp[..k],
                    &self.wtemp[..k],
                    edge,
                ),
            };
            let (improve, point, direction) = match found {
                Some(f) => f,
                None => (0.0, 0.0, LEFT),
            };
            if improve > self.iscale {
                self.iscale = improve;
            }
            if improve > self.iscale * 1e-10 {
                // insert_split: a new split precedes an existing one only if strictly better
                let better = match &primary {
                    Some(s) => improve > s.improve,
                    None => true,
                };
                if better {
                    primary = Some(Split {
                        var,
                        point,
                        direction,
                        improve,
                        count: k,
                    });
                }
            }
        }
        primary
    }

    /// `nodesplit`: assign `which` and reorder every variable's sort index so that the
    /// left child's observations come first, preserving order.
    fn nodesplit(&mut self, nodenum: u32, split: &Split, n1: usize, n2: usize) -> (usize, usize) {
        let leftson = 2 * nodenum;
        let rightson = leftson + 1;
        let col = self.col(split.var);
        let (mut nleft, mut nright) = (0usize, 0usize);
        for &j in &self.sorts[split.var][n1..n2] {
            let k = if col[j] < split.point {
                split.direction
            } else {
                -split.direction
            };
            if k == LEFT {
                self.which[j] = leftson;
                nleft += 1;
            } else {
                self.which[j] = rightson;
                nright += 1;
            }
        }
        let mut tempvec = Vec::with_capacity(n2 - n1);
        for var in 0..self.p {
            let sindex = &mut self.sorts[var];
            let mut i1 = n1;
            tempvec.clear();
            for i in n1..n2 {
                let j = sindex[i];
                if self.which[j] == leftson {
                    sindex[i1] = j;
                    i1 += 1;
                } else {
                    tempvec.push(j);
                }
            }
            sindex[n1 + nleft..n2].copy_from_slice(&tempvec);
        }
        (nleft, nright)
    }

    /// `partition`: returns `(number of splits below, sum of leaf risks)`.
    fn partition(&mut self, nodenum: u32, me: &mut Node, n1: usize, n2: usize) -> (usize, f64) {
        let n = n2 - n1;
        me.number = nodenum;
        let tempcp;
        if nodenum > 1 {
            let mut twt = 0.0;
            for (k, &j) in self.sorts[0][n1..n2].iter().enumerate() {
                self.wtemp[k] = self.wt[j];
                self.ytemp[k] = self.y[j];
                twt += self.wt[j];
            }
            let (value, risk) = eval(self.method, &self.ytemp[..n], &self.wtemp[..n]);
            me.value = value;
            me.risk = risk;
            me.n = n;
            me.sum_wt = twt;
            tempcp = me.risk.min(me.complexity);
        } else {
            tempcp = me.risk;
        }
        if me.n < self.control.minsplit || tempcp <= self.alpha || f64::from(nodenum) > self.maxnode {
            me.complexity = self.alpha;
            me.split = None;
            return (0, me.risk);
        }
        let split = match self.bsplit(me.risk, n1, n2) {
            Some(s) => s,
            None => {
                me.complexity = self.alpha;
                me.split = None;
                return (0, me.risk);
            }
        };
        let (nleft, nright) = self.nodesplit(nodenum, &split, n1, n2);
        me.split = Some(split);
        let mut left = Node::leaf(2 * nodenum);
        left.complexity = tempcp - self.alpha;
        let (mut left_split, mut left_risk) = self.partition(2 * nodenum, &mut left, n1, n1 + nleft);
        let mut tempcp = (me.risk - left_risk) / (left_split as f64 + 1.0);
        let tempcp2 = me.risk - left.risk;
        if tempcp < tempcp2 {
            tempcp = tempcp2;
        }
        if tempcp > me.complexity {
            tempcp = me.complexity;
        }
        let mut right = Node::leaf(2 * nodenum + 1);
        right.complexity = tempcp - self.alpha;
        let (mut right_split, mut right_risk) =
            self.partition(2 * nodenum + 1, &mut right, n1 + nleft, n1 + nleft + nright);
        let mut tempcp = (me.risk - (left_risk + right_risk)) / ((left_split + right_split) as f64 + 1.0);
        if right.complexity > left.complexity {
            if tempcp > left.complexity {
                left_risk = left.risk;
                left_split = 0;
                tempcp = (me.risk - (left_risk + right_risk)) / ((left_split + right_split) as f64 + 1.0);
                if tempcp > right.complexity {
                    right_risk = right.risk;
                    right_split = 0;
                }
            }
        } else if tempcp > right.complexity {
            right_split = 0;
            right_risk = right.risk;
            tempcp = (me.risk - (left_risk + right_risk)) / ((left_split + right_split) as f64 + 1.0);
            if tempcp > left.complexity {
                left_risk = left.risk;
                left_split = 0;
            }
        }
        me.complexity = (me.risk - (left_risk + right_risk)) / ((left_split + right_split) as f64 + 1.0);
        if me.complexity <= self.alpha {
            // All was in vain: this node does not split after all.
            me.split = None;
            for &j in &self.sorts[0][n1..n2] {
                self.which[j] = nodenum;
            }
            (0, me.risk)
        } else {
            me.left = Some(Box::new(left));
            me.right = Some(Box::new(right));
            (left_split + right_split + 1, left_risk + right_risk)
        }
    }
}

/// `make_cp_list`'s fix-up (`complexity = min(mine, parent's)`) followed by
/// `rpmatrix`'s scaling by the root risk.
fn fix_cp(node: &mut Node, parent: f64, scale: f64) {
    if node.complexity > parent {
        node.complexity = parent;
    }
    let raw = node.complexity;
    if let Some(l) = node.left.as_mut() {
        fix_cp(l, raw, scale);
    }
    if let Some(r) = node.right.as_mut() {
        fix_cp(r, raw, scale);
    }
    node.complexity = raw * scale;
}

/// `mysort.c`: rpart's quicksort (insertion sort below 12 elements, median-of-three
/// pivot), ported literally because it is not stable and the order of tied values
/// decides the floating-point accumulation order of the split sums, which in turn
/// breaks exact ties between candidate splits.
fn mysort(mut start: isize, mut stop: isize, x: &mut [f64], cvec: &mut [usize]) {
    while start < stop {
        if stop - start < 11 {
            for i in (start + 1)..=stop {
                let temp = x[i as usize];
                let tempd = cvec[i as usize];
                let mut j = i - 1;
                while j >= start && x[j as usize] > temp {
                    x[(j + 1) as usize] = x[j as usize];
                    cvec[(j + 1) as usize] = cvec[j as usize];
                    j -= 1;
                }
                x[(j + 1) as usize] = temp;
                cvec[(j + 1) as usize] = tempd;
            }
            return;
        }
        let mut i = start;
        let mut j = stop;
        let k = (start + stop) / 2;
        let (xi, xj, xk) = (x[i as usize], x[j as usize], x[k as usize]);
        let mut median = xk;
        if xi >= xk {
            if xj > xk {
                median = if xi > xj { xj } else { xi };
            }
        } else if xj < xk {
            median = if xi > xj { xi } else { xj };
        }
        while i < j {
            while x[i as usize] < median {
                i += 1;
            }
            while x[j as usize] > median {
                j -= 1;
            }
            if i < j {
                if x[i as usize] > x[j as usize] {
                    x.swap(i as usize, j as usize);
                    cvec.swap(i as usize, j as usize);
                }
                i += 1;
                j -= 1;
            }
        }
        while x[i as usize] >= median && i > start {
            i -= 1;
        }
        while x[j as usize] <= median && j < stop {
            j += 1;
        }
        if (i - start) < (stop - j) {
            if i - start > 0 {
                mysort(start, i, x, cvec);
            }
            start = j;
        } else {
            if stop - j > 0 {
                mysort(j, stop, x, cvec);
            }
            stop = i;
        }
    }
}

/// Fit a tree. `x` is column-major `n × p` with finite values; `y` holds the numeric
/// response (`Anova`) or 1-based class codes as `f64` (`Gini`); `wt` are case weights.
pub fn fit(x: &[f64], n: usize, p: usize, y: &[f64], wt: &[f64], method: Method, control: Control) -> Tree {
    assert_eq!(x.len(), n * p);
    assert_eq!(y.len(), n);
    assert_eq!(wt.len(), n);
    // rp.sorts, built with rpart's own sort so that tied values end up in the same
    // order as in R (see `mysort`).
    let sorts: Vec<Vec<usize>> = (0..p)
        .map(|var| {
            let mut xt: Vec<f64> = x[var * n..(var + 1) * n].to_vec();
            let mut idx: Vec<usize> = (0..n).collect();
            if n > 0 {
                mysort(0, n as isize - 1, &mut xt, &mut idx);
            }
            idx
        })
        .collect();
    let mut rp = Rp {
        x,
        y,
        wt,
        n,
        p,
        method,
        control,
        aprior: match method {
            Method::Anova => Vec::new(),
            Method::Gini { k } => altered_priors(k, y, wt),
        },
        alpha: 0.0,
        maxnode: 2f64.powi(control.maxdepth as i32) - 1.0,
        iscale: 0.0,
        sorts,
        which: vec![1; n],
        xtemp: vec![0.0; n],
        ytemp: vec![0.0; n],
        wtemp: vec![0.0; n],
    };
    let mut root = Node::leaf(1);
    let (value, risk) = eval(method, y, wt);
    root.n = n;
    root.sum_wt = wt.iter().sum();
    root.value = value;
    root.risk = risk;
    root.complexity = risk;
    rp.alpha = control.cp * risk;
    rp.partition(1, &mut root, 0, n);
    // rpart scales complexities by 1 / root risk; a constant outcome (risk 0) would
    // give NaN there, which no caller can use, so report unscaled zeros instead.
    let scale = if risk > 0.0 { 1.0 / risk } else { 0.0 };
    // make_cp_list fix-up starts at the root with its own complexity as the parent.
    let root_cp = root.complexity;
    fix_cp(&mut root, root_cp, scale);
    Tree {
        method,
        control,
        root,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;

    fn ctrl(minsplit: usize, cp: f64, maxdepth: usize) -> Control {
        Control {
            minsplit,
            minbucket: ((minsplit as f64) / 3.0).round_ties_even() as usize,
            cp,
            maxdepth,
        }
    }

    #[test]
    fn anova_single_split_at_midpoint_and_lower_mean_goes_left() {
        // y jumps at x between 3 and 4; the higher-mean side is x >= 3.5
        let x: Vec<f64> = (1..=8).map(|v| v as f64).collect();
        let y = vec![1.0, 1.2, 0.9, 1.1, 5.0, 5.2, 4.9, 5.1];
        let wt = vec![1.0; 8];
        let t = fit(&x, 8, 1, &y, &wt, Method::Anova, ctrl(2, 0.01, 30));
        let s = t.root.split.as_ref().unwrap();
        assert_eq!(s.var, 0);
        assert_abs_diff_eq!(s.point, 4.5);
        assert_eq!(s.direction, LEFT); // left_sum < right_sum -> x < 4.5 goes left
        let l = t.root.left.as_ref().unwrap();
        assert_eq!(l.number, 2);
        assert_eq!(l.n, 4);
        assert_abs_diff_eq!(
            match l.value {
                NodeValue::Mean(m) => m,
                _ => panic!(),
            },
            1.05
        );
        assert_eq!(t.leaf_for(|_| 2.0).number, 2);
        assert_eq!(t.leaf_for(|_| 7.0).number, 3);
        // children are leaves: minbucket 1, but cp stops further splits (tiny improvements)
        assert!(l.is_leaf());
        // root cp = (risk - children's risk) / root risk, i.e. the relative drop
        let r = t.root.right.as_ref().unwrap();
        assert_abs_diff_eq!(
            t.root.complexity,
            (t.root.risk - l.risk - r.risk) / t.root.risk,
            epsilon = 1e-15
        );
        assert_abs_diff_eq!(l.complexity, 0.01); // leaves carry cp
    }

    #[test]
    fn minsplit_and_maxdepth_stop_growth() {
        let x: Vec<f64> = (1..=8).map(|v| v as f64).collect();
        let y = vec![1.0, 1.2, 0.9, 1.1, 5.0, 5.2, 4.9, 5.1];
        let wt = vec![1.0; 8];
        let t = fit(&x, 8, 1, &y, &wt, Method::Anova, ctrl(20, 0.01, 30));
        assert!(t.root.is_leaf());
        let t = fit(&x, 8, 1, &y, &wt, Method::Anova, ctrl(2, 0.0, 1));
        assert_eq!(t.n_splits(), 1);
        let t = fit(&x, 8, 1, &y, &wt, Method::Anova, ctrl(2, 0.0, 30));
        assert!(t.n_splits() > 1);
    }

    #[test]
    fn gini_split_and_class_counts() {
        // class 1 for x <= 4, class 2 otherwise, one impurity
        let x: Vec<f64> = (1..=10).map(|v| v as f64).collect();
        let y = vec![1.0, 1.0, 1.0, 1.0, 2.0, 2.0, 1.0, 2.0, 2.0, 2.0];
        let wt = vec![1.0; 10];
        let t = fit(&x, 10, 1, &y, &wt, Method::Gini { k: 2 }, ctrl(2, 0.01, 30));
        let s = t.root.split.as_ref().unwrap();
        assert_abs_diff_eq!(s.point, 4.5);
        match &t.root.value {
            NodeValue::Class { class, counts } => {
                assert_eq!(*class, 1); // tie 5/5 -> first class
                assert_eq!(counts, &vec![5.0, 5.0]);
            }
            _ => panic!(),
        }
        assert_abs_diff_eq!(t.root.risk, 5.0);
        let l = t.leaf_for(|_| 1.0);
        assert!(matches!(&l.value, NodeValue::Class { class: 1, .. }));
    }

    #[test]
    fn mysort_orders_values_and_carries_indices() {
        let mut x = vec![
            5.0, 3.0, 9.0, 1.0, 3.0, 7.0, 2.0, 8.0, 6.0, 4.0, 0.0, 3.0, 11.0, 10.0,
        ];
        let mut idx: Vec<usize> = (0..x.len()).collect();
        let orig = x.clone();
        mysort(0, x.len() as isize - 1, &mut x, &mut idx);
        assert!(x.windows(2).all(|w| w[0] <= w[1]));
        for (i, v) in idx.iter().enumerate() {
            assert_eq!(orig[*v], x[i]);
        }
    }

    #[test]
    fn ties_in_improvement_pick_the_first_variable() {
        let x1: Vec<f64> = (1..=8).map(|v| v as f64).collect();
        let x2 = x1.clone();
        let mut x = x1.clone();
        x.extend(x2);
        let y = vec![1.0, 1.2, 0.9, 1.1, 5.0, 5.2, 4.9, 5.1];
        let wt = vec![1.0; 8];
        let t = fit(&x, 8, 2, &y, &wt, Method::Anova, ctrl(2, 0.01, 30));
        assert_eq!(t.root.split.as_ref().unwrap().var, 0);
    }
}
