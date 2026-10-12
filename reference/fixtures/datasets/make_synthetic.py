"""Generate the two small synthetic datasets used by the differential suite.

Deterministic given the seed and CPython's `random` module (outputs are committed, so
regeneration is only needed when the design changes). Run from the repo root:

    python -I reference/fixtures/datasets/make_synthetic.py
"""
import csv
import math
import os
import random

HERE = os.path.dirname(os.path.abspath(__file__))


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
    write_csv("synthetic_reg_small", ["x1", "x2", "x3", "g", "y"], rows)


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
    write_csv("synthetic_clf_small", ["x1", "x2", "g", "y"], rows)


if __name__ == "__main__":
    synthetic_reg_small()
    synthetic_clf_small()
