"""A/B report for a run made with `run_benchmarks.py --ab-baseline <binary> --systems rust_base,rust`.

    python -I benchmarks/ab_report.py --run-id <id> [--min-speedup 1.05] [--max-regression 0.97]

For every task x threads: baseline (`rust_base`) vs candidate (`rust`) medians per phase with
IQR and n, the candidate/baseline ratio (<1 is faster / smaller), and the drift verdict:
every fingerprint the driver emits (model state, predictions, workflow JSON, per-fold CV
metrics, tuning results, selection, workflow-set ranking) must be identical across all
trials of both systems. Writes benchmarks/results/<run_id>/ab_report.md.
"""
from __future__ import annotations

import argparse
import csv
import json
import statistics
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

PHASES = [
    ("workflow fit, cold", "wf_fit_s"),
    ("workflow fit, warm median", "warm_wf_fit_s"),
    ("prep + bake, warm median", "warm_prep_bake_s"),
    ("model fit, warm median", "warm_fit_s"),
    ("predict", "predict_s"),
    ("resampling (cv)", "cv_s"),
    ("tune", "tune_s"),
    ("select + last_fit", "select_s"),
    ("workflow set", "wfs_s"),
    ("wall (cold process)", "wall_s"),
    ("peak working set MB", "peak_wset_mb"),
    ("allocations wf_fit (count)", "alloc_wf_fit_count"),
    ("allocations wf_fit (MB)", "alloc_wf_fit_mb"),
    ("allocations cv (count)", "alloc_cv_count"),
    ("allocations cv (MB)", "alloc_cv_mb"),
]


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


def stats(rows, key):
    vals = [f(r.get(key)) for r in rows]
    vals = [v for v in vals if v is not None]
    if not vals:
        return None
    return {"median": statistics.median(vals), "iqr": q(vals, 0.75) - q(vals, 0.25), "min": min(vals), "n": len(vals)}


def show(s, key):
    if s is None:
        return "—"
    v = s["median"]
    if key.endswith("_s"):
        unit = "s"
        if v < 1e-3:
            txt = f"{v*1e6:.3g} µs"
        elif v < 1:
            txt = f"{v*1e3:.3g} ms"
        else:
            txt = f"{v:.3g} s"
        rel = (s["iqr"] / v * 100) if v else 0
        return f"{txt} (IQR {rel:.0f}%, n={s['n']})"
    if key.endswith("_mb"):
        return f"{v:.3g} MB (n={s['n']})"
    return f"{v:,.0f} (n={s['n']})"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-id", required=True)
    ap.add_argument("--min-speedup", type=float, default=1.05, help="ratio baseline/candidate that counts as a win")
    ap.add_argument("--max-regression", type=float, default=0.97, help="baseline/candidate below this is a regression")
    a = ap.parse_args()
    d = ROOT / "benchmarks" / "results" / a.run_id
    rows = list(csv.DictReader(open(d / "results.csv", newline="")))
    env = json.loads((d / "env.json").read_text())
    groups = defaultdict(list)
    for r in rows:
        groups[(r["task"], int(r["threads"]), r["system"])].append(r)
    keys = sorted({(k[0], k[1]) for k in groups})

    out = [f"# A/B report `{a.run_id}`", ""]
    out.append(f"Baseline binary: `{env.get('ab_baseline_binary')}`; candidate binary: `{env.get('rust_binary')}`; "
               f"git commit of the candidate checkout: `{env.get('git_commit')}`; trial order: A-B / B-A alternating; "
               f"protocol: {env.get('protocol')}.")
    out.append("")
    out.append("Ratio = candidate / baseline median (< 1: candidate faster or smaller). IQR is relative to the median. "
               "Drift = every fingerprint identical across all trials of both binaries (bitwise-equal values).")
    out.append("")
    summary = []
    for task, threads in keys:
        base = [r for r in groups[(task, threads, "rust_base")] if r["status"] == "ok"]
        cand = [r for r in groups[(task, threads, "rust")] if r["status"] == "ok"]
        n_base_all = len(groups[(task, threads, "rust_base")])
        n_cand_all = len(groups[(task, threads, "rust")])
        # drift: fingerprints
        fps = defaultdict(set)
        for r in base + cand:
            fp = r.get("fingerprints") or ""
            if fp:
                for k, v in json.loads(fp).items():
                    if v is not None:
                        fps[k].add(v)
        drift_keys = sorted(k for k, s in fps.items() if len(s) > 1)
        drift = "PASS" if fps and not drift_keys else ("FAIL: " + ", ".join(drift_keys) if drift_keys else "no fingerprints")
        n_rows = next((r["n_rows"] for r in base + cand), "?")
        model = next((r.get("model") for r in base + cand), "?")
        out.append(f"## {task} ({n_rows} rows, {model}, {threads} thread{'s' if threads != 1 else ''}) — "
                   f"ok trials baseline {len(base)}/{n_base_all}, candidate {len(cand)}/{n_cand_all}; drift: **{drift}**")
        out.append("")
        out.append("| phase | baseline | candidate | ratio | verdict |")
        out.append("|---|---|---|---|---|")
        for label, key in PHASES:
            sb, sc = stats(base, key), stats(cand, key)
            if sb is None and sc is None:
                continue
            if sb and sc and sb["median"]:
                ratio = sc["median"] / sb["median"]
                if ratio <= 1 / a.min_speedup:
                    verdict = "faster" if key.endswith("_s") else "smaller"
                elif ratio >= 1 / a.max_regression:
                    verdict = "**regression**" if key.endswith("_s") else "**larger**"
                else:
                    verdict = "within noise band"
                # a difference inside both IQRs is noise regardless of the ratio
                if key.endswith("_s") and abs(sc["median"] - sb["median"]) < max(sb["iqr"], sc["iqr"]):
                    verdict += " (inside IQR)"
                ratio_txt = f"{ratio:.3f}"
                summary.append((task, threads, label, ratio, verdict))
            else:
                ratio_txt, verdict = "—", "—"
            out.append(f"| {label} | {show(sb, key)} | {show(sc, key)} | {ratio_txt} | {verdict} |")
        out.append("")
    out.append("## Summary of phases with a decision")
    out.append("")
    out.append("| task | threads | phase | ratio | verdict |")
    out.append("|---|---|---|---|---|")
    for task, threads, label, ratio, verdict in summary:
        if "wall" in label or "allocations" in label:
            continue
        out.append(f"| {task} | {threads} | {label} | {ratio:.3f} | {verdict} |")
    (d / "ab_report.md").write_text("\n".join(out) + "\n", encoding="utf-8")
    print("\n".join(out))


if __name__ == "__main__":
    main()
