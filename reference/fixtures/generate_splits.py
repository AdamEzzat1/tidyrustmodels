"""Generate shared split / resample memberships and dataset checksums (ADR-0006, ADR-0013).

Reads reference/tasks.json and each task's dataset CSV, then writes
reference/fixtures/splits/<task_id>.json with 0-based row ids:

* `initial`: the 75/25 split (stratified by the outcome for classification; the first
  75 % of the rows for time-series tasks, as `initial_time_split`).
* `rset`: one resample per entry, `analysis` rows in the order rsample would draw them
  (bootstrap and Monte-Carlo draws are unsorted, v-fold complements ascend) and
  `assessment` rows ascending, with rsample's ids (`names0`: Fold1, Bootstrap01, ...).

The algorithms mirror rsample 1.3.2 so the *structure* of every resample is right
(balanced folds, sorted stratified bootstrap draws, grouped rows kept together,
deterministic time-series windows); the random draws themselves come from CPython's
generator, so R and Rust load these memberships instead of drawing their own.

Run from the repository root AFTER the datasets exist:

    python -I reference/fixtures/generate_splits.py
"""
import csv
import hashlib
import json
import math
import os
import random

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
TASKS = os.path.join(ROOT, "reference", "tasks.json")
DATASETS = os.path.join(ROOT, "reference", "fixtures", "datasets")
SPLITS = os.path.join(ROOT, "reference", "fixtures", "splits")

PROP = 0.75
V = 5
SEED = 20261012
TIME_SCHEMES = ("rolling_origin", "sliding_window", "sliding_index")


def read_columns(path, names):
    with open(path, newline="") as fh:
        rows = list(csv.DictReader(fh))
    return {name: [row[name] for row in rows] for name in names}


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 16), b""):
            h.update(chunk)
    return h.hexdigest()


def names0(num, prefix):
    """rsample:::names0: zero-padded to the width of the largest number."""
    width = len(str(num))
    return [f"{prefix}{k:0{width}d}" for k in range(1, num + 1)]


def strata_groups(values):
    """Row indices grouped by stratum label, groups in sorted label order."""
    groups = {}
    for i, v in enumerate(values):
        groups.setdefault(v, []).append(i)
    return [groups[k] for k in sorted(groups)]


# ---------------------------------------------------------------- random schemes


def initial_split(groups, rng, prop=PROP):
    train = []
    for g in groups:
        train.extend(rng.sample(g, int(len(g) * prop)))
    train.sort()
    in_train = set(train)
    n = sum(len(g) for g in groups)
    return train, [i for i in range(n) if i not in in_train]


def vfold_assignment(groups, rng, v):
    """`sample(rep(1:v, length.out = n))` within each stratum."""
    n = sum(len(g) for g in groups)
    fold_of_row = [None] * n
    for g in groups:
        labels = [i % v for i in range(len(g))]
        rng.shuffle(labels)
        for row, f in zip(g, labels):
            fold_of_row[row] = f
    return fold_of_row


def folds_of(fold_of_row, v):
    n = len(fold_of_row)
    out = []
    for k in range(v):
        assess = [i for i in range(n) if fold_of_row[i] == k]
        out.append(([i for i in range(n) if fold_of_row[i] != k], assess))
    return out


def vfold_cv(groups, rng, v, repeats):
    splits, ids = [], []
    for r in range(repeats):
        splits.extend(folds_of(vfold_assignment(groups, rng, v), v))
        fold_ids = names0(v, "Fold")
        ids.extend(fold_ids if repeats == 1 else [f"Repeat{r + 1}.{f}" for f in fold_ids])
    return splits, ids


def bootstraps(groups, rng, times):
    """Unstratified: `sample(n, replace = TRUE)` in draw order. Stratified: every
    stratum's draw is sorted (`strat_sample`), strata concatenated in label order.
    The assessment set is the out-of-bag rows, ascending."""
    n = sum(len(g) for g in groups)
    if len(groups) == 1:
        draws = [[rng.randrange(n) for _ in range(n)] for _ in range(times)]
    else:
        per_stratum = [[sorted(rng.choices(g, k=len(g))) for _ in range(times)] for g in groups]
        draws = [sum((per_stratum[s][k] for s in range(len(groups))), []) for k in range(times)]
    out = []
    for d in draws:
        in_bag = set(d)
        out.append((d, [i for i in range(n) if i not in in_bag]))
    return out, names0(times, "Bootstrap")


def mc_cv(groups, rng, prop, times):
    """Unstratified: `sample(n, size = floor(n * prop))` unsorted. Stratified: sorted
    within each stratum (`strat_sample`), strata concatenated. Assessment: the rest."""
    n = sum(len(g) for g in groups)
    out = []
    for _ in range(times):
        if len(groups) == 1:
            train = rng.sample(range(n), int(n * prop))
        else:
            train = sum((sorted(rng.sample(g, int(len(g) * prop))) for g in groups), [])
        in_train = set(train)
        out.append((train, [i for i in range(n) if i not in in_train]))
    return out, names0(times, "Resample")


# ---------------------------------------------------------------- grouped schemes


def freq_table(group_of_row):
    """Groups in first-appearance order with their counts (`vec_count(sort = "location")`)."""
    order, counts = [], {}
    for g in group_of_row:
        if g not in counts:
            order.append(g)
            counts[g] = 0
        counts[g] += 1
    return [(g, counts[g]) for g in order]


def balance_observations(table, perm, v, n_obs):
    """rsample:::balance_observations_helper on a given permutation of the table rows."""
    target = 1.0 / v
    assignment = [None] * len(table)
    fold_counts = [0] * v
    for k, row in enumerate(perm):
        size = table[row][1]
        if k < v:
            a = k
        else:
            best, best_imp = 0, math.inf
            for a_, c in enumerate(fold_counts):
                imp = abs((c + size) / n_obs - target) - abs(c / n_obs - target)
                if imp < best_imp:
                    best, best_imp = a_, imp
            a = best
        assignment[row] = a
        fold_counts[a] += size
    return assignment


def balance_prop(table, draws, prop):
    """rsample:::balance_prop_helper on given draws: keep the shortest prefix whose
    cumulative share crosses `prop`, or the one before it when that is closer."""
    total = sum(c for _, c in table)
    cum, acc = [], 0
    for d in draws:
        acc += table[d][1]
        cum.append(acc / total)
    crosses = next((k + 1 for k, c in enumerate(cum) if c > prop), len(cum))
    k = crosses
    if crosses >= 2 and abs(cum[crosses - 2] - prop) < abs(cum[crosses - 1] - prop):
        k = crosses - 1
    return draws[:k]


def group_vfold_cv(group_of_row, rng, v, repeats, balance):
    table = freq_table(group_of_row)
    n_groups = len(table)
    v = v or n_groups
    n = len(group_of_row)
    splits, ids = [], []
    for r in range(repeats):
        if balance == "groups":
            labels = [k % v for k in range(n_groups)]
            rng.shuffle(labels)
            fold_of_group = dict(zip((g for g, _ in table), labels))
        else:
            perm = rng.sample(range(n_groups), n_groups)
            assignment = balance_observations(table, perm, v, n)
            # collapse_groups: fold labels are shuffled afresh
            relabel = [k % v for k in range(v)]
            rng.shuffle(relabel)
            fold_of_group = {table[i][0]: relabel[assignment[i]] for i in range(n_groups)}
        fold_of_row = [fold_of_group[g] for g in group_of_row]
        splits.extend(folds_of(fold_of_row, v))
        fold_ids = names0(v, "Resample")
        ids.extend(fold_ids if repeats == 1 else [f"Repeat{r + 1}.{f}" for f in fold_ids])
    return splits, ids


def rows_of_draws(group_of_row, table, kept):
    """Rows of the kept groups in original row order, a row repeated once per draw of
    its group (`left_join(multiple = "all")` in collapse_groups)."""
    times = {}
    for d in kept:
        times[table[d][0]] = times.get(table[d][0], 0) + 1
    out = []
    for i, g in enumerate(group_of_row):
        out.extend([i] * times.get(g, 0))
    return out


def group_bootstraps(group_of_row, rng, times):
    table = freq_table(group_of_row)
    n, n_groups = len(group_of_row), len(table)
    n_draw = math.ceil(n_groups * n / min(c for _, c in table))
    out = []
    for _ in range(times):
        draws = [rng.randrange(n_groups) for _ in range(n_draw)]
        train = rows_of_draws(group_of_row, table, balance_prop(table, draws, 1.0))
        in_bag = set(train)
        out.append((train, [i for i in range(n) if i not in in_bag]))
    return out, names0(times, "Bootstrap")


def group_mc_cv(group_of_row, rng, prop, times):
    table = freq_table(group_of_row)
    n, n_groups = len(group_of_row), len(table)
    out = []
    for _ in range(times):
        draws = rng.sample(range(n_groups), n_groups)
        train = rows_of_draws(group_of_row, table, balance_prop(table, draws, prop))
        in_train = set(train)
        test = [i for i in range(n) if i not in in_train]
        if not test:
            raise SystemExit("group_mc_cv: an assessment set is empty; lower prop")
        out.append((train, test))
    return out, names0(times, "Resample")


# ---------------------------------------------------------------- time series


def rolling_origin(n, initial, assess, cumulative, skip, lag):
    out = []
    stop = initial
    while stop <= n - assess:
        start = 1 if cumulative else stop - initial + 1
        out.append(([r - 1 for r in range(start, stop + 1)], [r - 1 for r in range(stop + 1 - lag, stop + assess + 1)]))
        stop += skip + 1
    return out, names0(len(out), "Slice")


def skip_and_step(windows, skip, step):
    return windows[skip:][::step]


def sliding_window(n, lookback, assess_start, assess_stop, complete, step, skip):
    windows = []
    for i in range(1, n + 1):
        if lookback is None:
            start = 1
        elif i <= lookback:
            if complete:
                continue
            start = 1
        else:
            start = i - lookback
        if i + assess_stop > n:
            continue
        windows.append(([r - 1 for r in range(start, i + 1)], [r - 1 for r in range(i + assess_start, i + assess_stop + 1)]))
    out = skip_and_step(windows, skip, step)
    return out, names0(len(out), "Slice")


def sliding_index(index, lookback, assess_start, assess_stop, complete, step, skip):
    n = len(index)
    lo, hi = index[0], index[-1]
    windows = []
    for i in range(n):
        x = index[i]
        if lookback is None:
            in_lo = lo
        else:
            if complete and x - lookback < lo:
                continue
            in_lo = x - lookback
        if x + assess_stop > hi:
            continue
        train = [j for j in range(n) if in_lo <= index[j] <= x]
        test = [j for j in range(n) if x + assess_start <= index[j] <= x + assess_stop]
        windows.append((train, test))
    out = skip_and_step(windows, skip, step)
    return out, names0(len(out), "Slice")


# ---------------------------------------------------------------- driver


def build_rset(task, groups, columns, n, rng):
    rs = dict(task.get("resampling") or {"scheme": "vfold_cv", "v": V})
    scheme = rs["scheme"]
    if scheme == "vfold_cv":
        splits, ids = vfold_cv(groups, rng, rs["v"], rs.get("repeats", 1))
    elif scheme == "bootstraps":
        splits, ids = bootstraps(groups, rng, rs["times"])
    elif scheme == "mc_cv":
        splits, ids = mc_cv(groups, rng, rs["prop"], rs["times"])
    elif scheme == "group_vfold_cv":
        splits, ids = group_vfold_cv(columns[rs["group"]], rng, rs.get("v"), rs.get("repeats", 1), rs.get("balance", "groups"))
    elif scheme == "group_bootstraps":
        splits, ids = group_bootstraps(columns[rs["group"]], rng, rs["times"])
    elif scheme == "group_mc_cv":
        splits, ids = group_mc_cv(columns[rs["group"]], rng, rs["prop"], rs["times"])
    elif scheme == "rolling_origin":
        splits, ids = rolling_origin(n, rs["initial"], rs["assess"], rs.get("cumulative", True), rs.get("skip", 0), rs.get("lag", 0))
    elif scheme == "sliding_window":
        splits, ids = sliding_window(n, rs.get("lookback", 0), rs.get("assess_start", 1), rs.get("assess_stop", 1),
                                     rs.get("complete", True), rs.get("step", 1), rs.get("skip", 0))
    elif scheme == "sliding_index":
        index = [float(v) for v in columns[rs["index"]]]
        splits, ids = sliding_index(index, rs.get("lookback", 0), rs.get("assess_start", 1), rs.get("assess_stop", 1),
                                    rs.get("complete", True), rs.get("step", 1), rs.get("skip", 0))
    else:
        raise SystemExit(f"{task['id']}: unknown resampling scheme {scheme}")
    for a, s in splits:
        assert a and all(0 <= i < n for i in a) and all(0 <= i < n for i in s), task["id"]
        assert s == sorted(s), task["id"]
    return {
        "scheme": scheme,
        "params": {k: v for k, v in rs.items() if k != "scheme"},
        "ids": ids,
        "analysis": [a for a, _ in splits],
        "assessment": [s for _, s in splits],
    }


def main():
    with open(TASKS) as fh:
        tasks = json.load(fh)["tasks"]
    os.makedirs(SPLITS, exist_ok=True)
    checksums = {}
    for t in tasks:
        path = os.path.join(DATASETS, t["dataset"] + ".csv")
        rs = t.get("resampling") or {}
        aux = [c for c in (rs.get("group"), rs.get("index")) if c]
        columns = read_columns(path, [t["outcome"], *aux])
        outcome = columns[t["outcome"]]
        n = len(outcome)
        if t["type"] == "classification":
            groups = strata_groups(outcome)
            strata = t["outcome"]
        else:
            groups = [list(range(n))]
            strata = None
        rng = random.Random(f"{SEED}:{t['id']}")
        if rs.get("scheme") in TIME_SCHEMES:
            n_train = int(n * PROP)
            train, test = list(range(n_train)), list(range(n_train, n))
            initial = {"prop": PROP, "scheme": "initial_time_split", "train": train, "test": test}
        else:
            train, test = initial_split(groups, rng)
            initial = {"prop": PROP, "scheme": "initial_split", "train": train, "test": test}
        rset = build_rset(t, groups, columns, n, rng)
        out = {
            "task": t["id"],
            "dataset": t["dataset"],
            "n": n,
            "seed": SEED,
            "strata": strata,
            "initial": initial,
            "rset": rset,
        }
        with open(os.path.join(SPLITS, t["id"] + ".json"), "w") as fh:
            json.dump(out, fh, separators=(",", ":"))
        checksums[t["dataset"]] = {"sha256": sha256(path), "rows": n}
        sizes = sorted({(len(a), len(s)) for a, s in zip(rset["analysis"], rset["assessment"])})
        print(f"{t['id']}: n={n} train={len(train)} test={len(test)} {rset['scheme']} x{len(rset['ids'])} "
              f"(analysis, assessment) sizes {sizes[:3]}{'...' if len(sizes) > 3 else ''} strata={strata}")
    with open(os.path.join(DATASETS, "checksums.json"), "w") as fh:
        json.dump(checksums, fh, indent=1, sort_keys=True)
    print("wrote checksums.json")


if __name__ == "__main__":
    main()
