"""Generate synthetic scaling datasets for the benchmark suite (gitignored output).

    python -I benchmarks/make_scaling_data.py --sizes 1000,10000,100000

Writes benchmarks/data/synth_<n>_reg.csv and synth_<n>_clf.csv with columns
x1..x5 (numeric; x3 has ~2% missing), g (categorical, 10 levels) and y.
Deterministic given the seed; the generator (not the files) is committed.
"""
from __future__ import annotations

import argparse
import csv
import math
import os
import random

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "benchmarks", "data")


def gen(n: int, kind: str, seed: int) -> None:
    rng = random.Random(f"{seed}:{n}:{kind}")
    levels = [f"L{i}" for i in range(10)]
    effects = [rng.uniform(-1, 1) for _ in levels]
    path = os.path.join(OUT, f"synth_{n}_{kind}.csv")
    with open(path, "w", newline="") as fh:
        w = csv.writer(fh, lineterminator="\n")
        w.writerow(["x1", "x2", "x3", "x4", "x5", "g", "y"])
        for _ in range(n):
            x = [rng.gauss(0, 1) for _ in range(5)]
            gi = rng.randrange(10)
            eta = 0.5 + 1.5 * x[0] - 1.0 * x[1] + 0.7 * x[2] + 0.2 * x[3] - 0.4 * x[4] + effects[gi]
            if kind == "reg":
                y = f"{eta + rng.gauss(0, 1):.6f}"
            else:
                p = 1 / (1 + math.exp(-eta))
                y = "yes" if rng.random() < p else "no"
            row = [f"{v:.6f}" for v in x]
            if rng.random() < 0.02:
                row[2] = ""
            w.writerow([*row, levels[gi], y])
    print("wrote", path, n, "rows")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sizes", default="1000,10000,100000")
    ap.add_argument("--seed", type=int, default=20261012)
    a = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    for n in [int(s) for s in a.sizes.split(",")]:
        gen(n, "reg", a.seed)
        gen(n, "clf", a.seed)


if __name__ == "__main__":
    main()
