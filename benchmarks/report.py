"""Aggregate a benchmark run into Markdown.

    python -I benchmarks/report.py --run-id <id>

Writes benchmarks/results/<run_id>/report.md. Medians over cold trials with IQR, min and
n; ratios are R median / candidate median (>1 favours the candidate under the recorded
conditions). Candidates are `rust/<engine>` (the Rust binary) and `python/<engine>` (the
PyO3 wrapper driven from CPython). Rows whose category is "workflow-level" compare
different algorithms and must not be read as equivalent-algorithm results.

Phase semantics (all drivers): `wf_fit_s` = workflow fit from the raw training table
(prep + bake + model fit); `fit_s` = model fit on an already-baked design (Rust and R
only; the Python API has no separate bake); `cv_s` = v-fold fit_resamples; `tune_s` =
grid tuning (classification only; R uses glmnet ridge).
"""
from __future__ import annotations

import argparse
import csv
import json
import statistics
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PHASES = ["wall_s", "library_load_s", "load_s", "prep_s", "bake_s", "fit_s", "wf_fit_s", "predict_s", "cv_s", "tune_s",
          "warm_prep_bake_s", "warm_fit_s", "warm_wf_fit_s", "warm_predict_s", "peak_rss_tree_mb", "peak_wset_mb"]
VERIFIED = {"synth_reg", "synth_clf", "iris_clf", "penguins_reg", "penguins_clf", "ames_reg"}


def f(x):
    try:
        v = float(x)
        return v if v == v else None
    except (TypeError, ValueError):
        return None


def q(vals, p):
    vals = sorted(vals)
    k = (len(vals) - 1) * p
    lo, hi = int(k), min(int(k) + 1, len(vals) - 1)
    return vals[lo] + (vals[hi] - vals[lo]) * (k - lo)


def fmt(vals, unit=""):
    vals = [v for v in vals if v is not None]
    if not vals:
        return "not run"
    med = statistics.median(vals)
    iqr = q(vals, 0.75) - q(vals, 0.25)
    return f"{med:.4g}{unit} (IQR {iqr:.2g}, min {min(vals):.4g}, n={len(vals)})"


def med(rows, key):
    vals = [f(r.get(key)) for r in rows]
    vals = [v for v in vals if v is not None]
    return statistics.median(vals) if vals else None


def ratio(a_rows, b_rows, key):
    a_, b_ = med(a_rows, key), med(b_rows, key)
    return f"{a_ / b_:.2f}" if a_ and b_ else "n/a"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-id", required=True)
    a = ap.parse_args()
    d = ROOT / "benchmarks" / "results" / a.run_id
    rows = list(csv.DictReader(open(d / "results.csv", newline="")))
    env = json.loads((d / "env.json").read_text())
    ok = [r for r in rows if r["status"] == "ok"]

    groups = defaultdict(list)
    for r in rows:
        groups[(r["task"], int(r["threads"]), r["system"], r["engine"])].append(r)
    tasks = sorted({(r["task"], int(r["threads"])) for r in rows})
    candidates = sorted({(r["system"], r["engine"]) for r in rows if r["system"] != "r"})

    out = [f"# Benchmark report `{a.run_id}`", ""]
    out.append("Environment: " + ", ".join(f"{k}={v}" for k, v in env.items()))
    out.append("")
    out.append("Category: *framework-overhead* = same algorithm class and objective (OLS vs OLS); "
               "*workflow-level* = different optimizer or algorithm (R IRLS `glm` vs L-BFGS; "
               "R `glmnet` ridge vs unscaled-L2 L-BFGS for tuning). Medians over cold trials (fresh "
               "process each) with IQR, min and n. Ratio = R / candidate (>1 favours the candidate "
               "under these conditions). `wf_fit` = prep + bake + model fit from the raw training "
               "table, the like-for-like fit comparison; `fit` = model only on a baked design.")
    out.append("")
    out.append("| Dataset | Category | Threads | Candidate | R wall | cand. wall | ratio | R wf_fit | cand. wf_fit | ratio | R cv | cand. cv | ratio | R tune | cand. tune | ratio | R peak WS | cand. peak WS | ratio | Numerical parity |")
    out.append("|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|")
    for task, threads in tasks:
        r_rows = [r for r in ok if r["task"] == task and int(r["threads"]) == threads and r["system"] == "r"]
        ttype = next((r["task_type"] for r in rows if r["task"] == task), "?")
        category = "framework-overhead (OLS vs OLS)" if ttype == "regression" else "workflow-level (IRLS vs L-BFGS)"
        parity = "VERIFICATION.md Phase 4 (L3/L4 pass)" if task in VERIFIED else "unverified (synthetic scaling data)"
        for system, engine in candidates:
            c_rows = [r for r in ok if r["task"] == task and int(r["threads"]) == threads
                      and r["system"] == system and r["engine"] == engine]
            if not c_rows:
                continue
            out.append("| " + " | ".join([
                task, category, str(threads), f"{system}/{engine}",
                fmt([f(r.get("wall_s")) for r in r_rows], "s"), fmt([f(r.get("wall_s")) for r in c_rows], "s"), ratio(r_rows, c_rows, "wall_s"),
                fmt([f(r.get("wf_fit_s")) for r in r_rows], "s"), fmt([f(r.get("wf_fit_s")) for r in c_rows], "s"), ratio(r_rows, c_rows, "wf_fit_s"),
                fmt([f(r.get("cv_s")) for r in r_rows], "s"), fmt([f(r.get("cv_s")) for r in c_rows], "s"), ratio(r_rows, c_rows, "cv_s"),
                fmt([f(r.get("tune_s")) for r in r_rows], "s"), fmt([f(r.get("tune_s")) for r in c_rows], "s"), ratio(r_rows, c_rows, "tune_s"),
                fmt([f(r.get("peak_wset_mb")) for r in r_rows], " MB"), fmt([f(r.get("peak_wset_mb")) for r in c_rows], " MB"), ratio(r_rows, c_rows, "peak_wset_mb"),
                parity,
            ]) + " |")
    out.append("")
    out.append("## Per-phase medians (seconds; memory in MB)")
    out.append("")
    out.append("| task | threads | system | engine | n | " + " | ".join(PHASES) + " | status counts |")
    out.append("|---|---|---|---|---|" + "---|" * len(PHASES) + "---|")
    for (task, threads, system, engine), rs in sorted(groups.items()):
        good = [r for r in rs if r["status"] == "ok"]
        counts = defaultdict(int)
        for r in rs:
            counts[r["status"]] += 1
        cells = []
        for ph in PHASES:
            m = med(good, ph)
            cells.append(f"{m:.4g}" if m is not None else "not run")
        n = next((r["n_rows"] for r in good), "?")
        out.append(f"| {task} | {threads} | {system} | {engine} | {n} | " + " | ".join(cells) + f" | {dict(counts)} |")
    out.append("")
    out.append("Notes: `wall_s` includes process start, interpreter/library load (`library_load_s`: R "
               "tidymodels attach; Python import of tidyrustmodels+numpy+pandas) and CSV parsing. "
               "`tune_s` is R glmnet ridge vs L-BFGS with unscaled L2 (workflow-level only). "
               "`peak_wset_mb` is the OS-reported exact peak of the root process; `peak_rss_tree_mb` "
               "is the 1 ms-sampled maximum of the summed process tree (lower bound; `not run` when "
               "the process exited before the first sample).")
    (d / "report.md").write_text("\n".join(out) + "\n")
    print("\n".join(out))


if __name__ == "__main__":
    main()
