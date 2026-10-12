"""Generate shared split / fold assignments and dataset checksums (ADR-0006).

Reads reference/tasks.json and each task's dataset CSV, then writes
reference/fixtures/splits/<task_id>.json with 0-based row ids. Classification tasks
are stratified by the outcome; regression tasks are not. The algorithm mirrors
rsample: `floor(n * prop)` analysis rows per stratum for the initial split, and
`rep(1:v, length.out = n)` shuffled within each stratum for v-fold.

Run from the repository root AFTER the datasets exist:

    python -I reference/fixtures/generate_splits.py
"""
import csv
import hashlib
import json
import os
import random

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
TASKS = os.path.join(ROOT, "reference", "tasks.json")
DATASETS = os.path.join(ROOT, "reference", "fixtures", "datasets")
SPLITS = os.path.join(ROOT, "reference", "fixtures", "splits")

PROP = 0.75
V = 5
SEED = 20261012


def read_column(path, name):
    with open(path, newline="") as fh:
        r = csv.DictReader(fh)
        return [row[name] for row in r]


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 16), b""):
            h.update(chunk)
    return h.hexdigest()


def strata_groups(values):
    """Row indices grouped by stratum label, groups in sorted label order."""
    groups = {}
    for i, v in enumerate(values):
        groups.setdefault(v, []).append(i)
    return [groups[k] for k in sorted(groups)]


def initial_split(groups, rng):
    train = []
    for g in groups:
        k = int(len(g) * PROP)
        train.extend(rng.sample(g, k))
    train.sort()
    in_train = set(train)
    n = sum(len(g) for g in groups)
    test = [i for i in range(n) if i not in in_train]
    return train, test


def vfold(groups, rng):
    n = sum(len(g) for g in groups)
    fold_of_row = [None] * n
    for g in groups:
        labels = [i % V for i in range(len(g))]
        rng.shuffle(labels)
        for row, f in zip(g, labels):
            fold_of_row[row] = f
    return fold_of_row


def main():
    with open(TASKS) as fh:
        tasks = json.load(fh)["tasks"]
    os.makedirs(SPLITS, exist_ok=True)
    checksums = {}
    for t in tasks:
        path = os.path.join(DATASETS, t["dataset"] + ".csv")
        outcome = read_column(path, t["outcome"])
        n = len(outcome)
        if t["type"] == "classification":
            groups = strata_groups(outcome)
            strata = t["outcome"]
        else:
            groups = [list(range(n))]
            strata = None
        rng = random.Random(f"{SEED}:{t['id']}")
        train, test = initial_split(groups, rng)
        fold_of_row = vfold(groups, rng)
        out = {
            "task": t["id"],
            "dataset": t["dataset"],
            "n": n,
            "seed": SEED,
            "strata": strata,
            "initial": {"prop": PROP, "train": train, "test": test},
            "vfold": {"v": V, "fold_of_row": fold_of_row},
        }
        with open(os.path.join(SPLITS, t["id"] + ".json"), "w") as fh:
            json.dump(out, fh, separators=(",", ":"))
        checksums[t["dataset"]] = {"sha256": sha256(path), "rows": n}
        print(f"{t['id']}: n={n} train={len(train)} test={len(test)} folds={V} strata={strata}")
    with open(os.path.join(DATASETS, "checksums.json"), "w") as fh:
        json.dump(checksums, fh, indent=1, sort_keys=True)
    print("wrote checksums.json")


if __name__ == "__main__":
    main()
