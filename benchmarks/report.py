"""Aggregate a benchmark run into the BENCHMARKS.md table format.

    python -I benchmarks/report.py --run-id <id>

Writes benchmarks/results/<run_id>/report.md. Medians and interquartile ranges over
trials; speedup = R median / Rust median (>1 favours tidyrustmodels); memory ratio
likewise. Rows whose category is "workflow-level" compare different algorithms and
must not be read as equivalent-algorithm results.
"""
from __future__ import annotations

import argparse
import csv
import json
import statistics
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def q(vals, p):
    vals = sorted(vals)
    if not vals:
        return float("nan")
    k = (len(vals) - 1) * p
    lo, hi = int(k), min(int(k) + 1, len(vals) - 1)
    return vals[lo] + (vals[hi] - vals[lo]) * (k - lo)


def fmt(vals, unit=""):
    vals = [v for v in vals if v is not None and v == v]
    if not vals:
        return "not run"
    med = statistics.median(vals)
    iqr = q(vals, 0.75) - q(vals, 0.25)
    return f"{med:.4g}{unit} (IQR {iqr:.2g}, n={len(vals)})"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-id", required=True)
    a = ap.parse_args()
    d = ROOT / "benchmarks" / "results" / a.run_id
    rows = list(csv.DictReader(open(d / "results.csv", newline="")))
    env = json.loads((d / "env.json").read_text())

    def f(x):
        try:
            return float(x)
        except (TypeError, ValueError):
            return None

    groups = defaultdict(list)
    for r in rows:
        groups[(r["task"], r["threads"], r["system"], r["engine"])].append(r)

    tasks = sorted({(r["task"], r["threads"]) for r in rows})
    phases = ["wall_s", "load_s", "prep_s", "bake_s", "fit_s", "predict_s", "cv_s", "tune_s", "warm_fit_s", "warm_prep_bake_s", "warm_predict_s"]

    out = [f"# Benchmark report `{a.run_id}`", ""]
    out.append("Environment: " + ", ".join(f"{k}={v}" for k, v in env.items()))
    out.append("")
    out.append("Category: *framework-overhead* = same algorithm class and objective (OLS vs OLS); "
               "*workflow-level* = different optimizer or algorithm (R IRLS `glm` vs L-BFGS; "
               "R `glmnet` ridge vs unscaled-L2 L-BFGS for tuning). Times are medians over cold "
               "trials (fresh process each), IQR in parentheses. Speedup = R / Rust (>1 favours "
               "tidyrustmodels under these conditions).")
    out.append("")
    out.append("| Dataset | Workflow | Category | Threads | Engine | R wall | Rust wall | Speedup (wall) | R fit | Rust fit | Speedup (fit) | R cv | Rust cv | Speedup (cv) | R peak RSS | Rust peak RSS | Memory ratio | Numerical parity |")
    out.append("|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|")
    for task, threads in tasks:
        r_rows = [r for r in rows if r["task"] == task and r["threads"] == threads and r["system"] == "r" and r["status"] == "ok"]
        ttype = next((r["task_type"] for r in rows if r["task"] == task), "?")
        category = "framework-overhead (OLS vs OLS)" if ttype == "regression" else "workflow-level (IRLS vs L-BFGS)"
        parity = "VERIFICATION.md Phase 4 (L3/L4 pass)" if task in {"synth_reg", "synth_clf", "iris_clf", "penguins_reg", "penguins_clf", "ames_reg"} else "unverified (synthetic scaling data)"
        for engine in sorted({r["engine"] for r in rows if r["system"] == "rust"}):
            rust_rows = [r for r in rows if r["task"] == task and r["threads"] == threads and r["system"] == "rust" and r["engine"] == engine and r["status"] == "ok"]

            def med(rs, k):
                vals = [f(r[k]) for r in rs if f(r[k]) is not None]
                return statistics.median(vals) if vals else None

            def ratio(k):
                a_, b_ = med(r_rows, k), med(rust_rows, k)
                return f"{a_ / b_:.2f}" if a_ and b_ else "n/a"

            out.append("| " + " | ".join([
                task, "recipe + " + ("linear_reg" if ttype == "regression" else "logistic_reg") + " + 5-fold CV", category, threads, engine,
                fmt([f(r["wall_s"]) for r in r_rows], "s"), fmt([f(r["wall_s"]) for r in rust_rows], "s"), ratio("wall_s"),
                fmt([f(r["fit_s"]) for r in r_rows], "s"), fmt([f(r["fit_s"]) for r in rust_rows], "s"), ratio("fit_s"),
                fmt([f(r["cv_s"]) for r in r_rows], "s"), fmt([f(r["cv_s"]) for r in rust_rows], "s"), ratio("cv_s"),
                fmt([f(r["peak_rss_tree_mb"]) for r in r_rows], " MB"), fmt([f(r["peak_rss_tree_mb"]) for r in rust_rows], " MB"), ratio("peak_rss_tree_mb"),
                parity,
            ]) + " |")
    out.append("")
    out.append("## Per-phase medians (seconds)")
    out.append("")
    out.append("| task | threads | system | engine | n | " + " | ".join(phases) + " | status counts |")
    out.append("|---|---|---|---|---|" + "---|" * len(phases) + "---|")
    for (task, threads, system, engine), rs in sorted(groups.items()):
        ok = [r for r in rs if r["status"] == "ok"]
        counts = defaultdict(int)
        for r in rs:
            counts[r["status"]] += 1
        cells = []
        for ph in phases:
            vals = [f(r[ph]) for r in ok if f(r[ph]) is not None]
            cells.append(f"{statistics.median(vals):.4g}" if vals else "not run")
        n = next((r["n_rows"] for r in ok), "?")
        out.append(f"| {task} | {threads} | {system} | {engine} | {n} | " + " | ".join(cells) + f" | {dict(counts)} |")
    out.append("")
    out.append("Notes: `wall_s` includes process start, library load (R: tidymodels attach, reported as "
               "`library_load_s` in results.csv) and CSV parsing. `tune_s` is R glmnet ridge vs Rust L-BFGS "
               "with unscaled L2 (workflow-level only). Peak RSS is the sampled maximum of the process tree.")
    (d / "report.md").write_text("\n".join(out) + "\n")
    print("\n".join(out))


if __name__ == "__main__":
    main()
