"""Generate the small synthetic datasets used by the differential suite.

Deterministic given the seed and CPython's `random` module (outputs are committed, so
regeneration is only needed when the design changes). Run from the repo root:

    python -I reference/fixtures/datasets/make_synthetic.py
"""
import csv
import math
import os
import random

HERE = os.path.dirname(os.path.abspath(__file__))


def add_weights(rows, seed):
    """Append two case-weight columns, drawn from their own generator so the existing
    columns are unchanged: `w_freq`, frequency weights in 0..4 (10% zeros), and
    `w_imp`, importance weights in [0.2, 2.5] with 5% zeros."""
    rng = random.Random(seed)
    for r in rows:
        u = rng.random()
        w_freq = 0 if u < 0.10 else 1 if u < 0.45 else 2 if u < 0.75 else 3 if u < 0.90 else 4
        w_imp = 0.0 if rng.random() < 0.05 else round(rng.uniform(0.2, 2.5), 3)
        r.extend([w_freq, w_imp])
    return rows


def write_csv(name, header, rows):
    path = os.path.join(HERE, name + ".csv")
    with open(path, "w", newline="") as fh:
        w = csv.writer(fh, lineterminator="\n")
        w.writerow(header)
        for r in rows:
            w.writerow(["" if v is None else v for v in r])
    print("wrote", path, len(rows), "rows")


def synthetic_reg_small():
    rng = random.Random(20261012)
    rows = []
    levels = ["a", "b", "c"]
    effect = {"a": 0.0, "b": 1.0, "c": -1.0}
    for i in range(40):
        x1 = rng.gauss(0.0, 1.0)
        x2 = rng.uniform(-2.0, 2.0)
        x3 = rng.gauss(5.0, 2.0)
        g = levels[rng.randrange(3)]
        y = 1.5 + 2.0 * x1 - 1.0 * x2 + 0.5 * x3 + effect[g] + rng.gauss(0.0, 0.5)
        rows.append([round(x1, 6), round(x2, 6), round(x3, 6), g, round(y, 6)])
    # mask 4 x3 values as missing (after y was generated from the full value)
    for i in [3, 11, 19, 30]:
        rows[i][2] = None
    add_weights(rows, 20261101)
    write_csv("synthetic_reg_small", ["x1", "x2", "x3", "g", "y", "w_freq", "w_imp"], rows)


def synthetic_clf_small():
    rng = random.Random(20261013)
    rows = []
    while True:
        rows = []
        for i in range(60):
            x1 = rng.gauss(0.0, 1.0)
            x2 = rng.gauss(0.0, 1.0)
            g = "b" if rng.random() < 0.5 else "a"
            logit = -0.3 + 1.2 * x1 - 0.8 * x2 + (0.7 if g == "b" else 0.0)
            p = 1.0 / (1.0 + math.exp(-logit))
            y = "yes" if rng.random() < p else "no"
            rows.append([round(x1, 6), round(x2, 6), g, y])
        n_yes = sum(r[3] == "yes" for r in rows)
        if 20 <= n_yes <= 40:
            break
    add_weights(rows, 20261102)
    write_csv("synthetic_clf_small", ["x1", "x2", "g", "y", "w_freq", "w_imp"], rows)


def synthetic_multi_small():
    """Three overlapping classes (a 40%, b 35%, c 25%) with class-conditional means in
    (x1, x2), a noise predictor x3 and a nominal predictor g whose frequency depends on
    the class. The overlap keeps the multinomial MLE finite (no separation)."""
    rng = random.Random(20261015)
    means = {"a": (0.0, 0.0), "b": (1.5, 0.5), "c": (0.5, 1.8)}
    p_q = {"a": 0.3, "b": 0.6, "c": 0.45}
    rows = []
    for i in range(180):
        u = rng.random()
        y = "a" if u < 0.40 else ("b" if u < 0.75 else "c")
        mx, my = means[y]
        x1 = rng.gauss(mx, 1.0)
        x2 = rng.gauss(my, 1.0)
        x3 = rng.uniform(-1.0, 1.0)
        g = "q" if rng.random() < p_q[y] else "p"
        rows.append([round(x1, 6), round(x2, 6), round(x3, 6), g, y])
    add_weights(rows, 20261103)
    write_csv("synthetic_multi_small", ["x1", "x2", "x3", "g", "y", "w_freq", "w_imp"], rows)


def synthetic_filter_reg():
    """Designed so every filter/level step has something to do: x2 is ~0.98 correlated
    with x1 (step_corr), x3 is 96% zeros (step_nzv), x4 is constant and x5 all missing
    (step_zv), g has two rare levels (step_other), h has missing values (step_unknown)."""
    rng = random.Random(20261014)
    rows = []
    g_levels = ["p", "q", "r", "s", "t", "u"]
    g_probs = [0.40, 0.30, 0.15, 0.10, 0.03, 0.02]
    for i in range(150):
        x1 = rng.gauss(0.0, 1.0)
        x2 = x1 + rng.gauss(0.0, 0.2)
        x3 = 0.0 if rng.random() < 0.96 else round(rng.uniform(1.0, 3.0), 3)
        x4 = 2.5
        x5 = None
        u = rng.random()
        acc = 0.0
        g = g_levels[-1]
        for lv, pr in zip(g_levels, g_probs):
            acc += pr
            if u < acc:
                g = lv
                break
        h = None if rng.random() < 0.08 else ("m" if rng.random() < 0.6 else "n")
        geff = {"p": 0.0, "q": 0.5, "r": -0.5, "s": 1.0, "t": 2.0, "u": -2.0}[g]
        heff = 0.0 if h is None else (0.3 if h == "m" else -0.3)
        y = 2.0 + 1.5 * x1 - 0.5 * x2 + 0.8 * x3 + geff + heff + rng.gauss(0.0, 0.4)
        rows.append([round(x1, 6), round(x2, 6), x3, x4, x5, g, h, round(y, 6)])
    # a few missing x1 values for the imputation step
    for i in [5, 40, 77, 120]:
        rows[i][0] = None
    add_weights(rows, 20261104)
    write_csv("synthetic_filter_reg", ["x1", "x2", "x3", "x4", "x5", "g", "h", "y", "w_freq", "w_imp"], rows)


if __name__ == "__main__":
    synthetic_reg_small()
    synthetic_clf_small()
    synthetic_filter_reg()
    synthetic_multi_small()
