"""Benchmark orchestrator: runs R tidymodels and Rust tidyrustmodels on identical tasks.

Each trial is a fresh process (cold start). The orchestrator measures:
  * wall-clock time of the whole process (includes library/binary load and CSV parsing);
  * peak resident memory of the process tree, as the maximum over time of the summed
    RSS of the process and its children, sampled every ~1 ms, plus (on Windows) the
    OS-tracked exact peak working set of each process;
  * the per-phase timings the process itself reports on its `BENCH_RESULT` line.

Usage (from the repository root, inside the venv):
    python -I benchmarks/run_benchmarks.py --run-id 2026-10-12a --trials 5 --warm 10 \
        --tasks penguins_reg,penguins_clf,ames_reg,synth_1000,synth_10000,synth_100000 \
        --threads 1,4

Results: benchmarks/results/<run_id>/results.csv and env.json. Then run report.py.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import platform
import subprocess
import sys
import time
from pathlib import Path

import psutil

ROOT = Path(__file__).resolve().parents[1]
RSCRIPT = os.environ.get("RSCRIPT", r"C:\Program Files\R\R-4.4.0\bin\Rscript.exe")
BENCH_BIN = ROOT / "benchmarks" / "rust" / "target" / "release" / ("trm-bench.exe" if os.name == "nt" else "trm-bench")

REFERENCE_TASKS = {t["id"]: t for t in json.loads((ROOT / "reference" / "tasks.json").read_text())["tasks"]}


def task_spec(task_id: str) -> dict:
    if task_id in REFERENCE_TASKS:
        t = REFERENCE_TASKS[task_id]
        return {
            "id": task_id,
            "dataset": str(ROOT / "reference" / "fixtures" / "datasets" / f"{t['dataset']}.csv"),
            "outcome": t["outcome"],
            "type": t["type"],
            "predictors": t["predictors"],
            "steps": t["steps"],
            "metrics": t["metrics"],
        }
    if task_id.startswith("synth_"):
        n = int(task_id.split("_")[1])
        kind = task_id.split("_")[2] if task_id.count("_") == 2 else "reg"
        return {
            "id": task_id,
            "dataset": str(ROOT / "benchmarks" / "data" / f"synth_{n}_{kind}.csv"),
            "outcome": "y",
            "type": "regression" if kind == "reg" else "classification",
            "predictors": ["x1", "x2", "x3", "x4", "x5", "g"],
            "steps": ["impute_median_numeric", "normalize_numeric", "dummy_nominal"],
            "metrics": ["rmse", "rsq"] if kind == "reg" else ["accuracy", "mn_log_loss"],
        }
    raise SystemExit(f"unknown task {task_id}")


def command(system: str, spec: dict, engine: str, threads: int, warm: int) -> list[str]:
    common = [
        "--dataset", spec["dataset"], "--outcome", spec["outcome"], "--type", spec["type"],
        "--predictors", ",".join(spec["predictors"]), "--steps", ",".join(spec["steps"]),
        "--metrics", ",".join(spec["metrics"]), "--threads", str(threads), "--warm", str(warm),
    ]
    if system == "rust":
        return [str(BENCH_BIN), "--engine", engine, *common]
    return [RSCRIPT, str(ROOT / "benchmarks" / "r" / "bench_task.R"), *common]


def run_trial(cmd: list[str], timeout_s: float) -> dict:
    """Spawn, sample the process tree's RSS, parse the BENCH_RESULT line."""
    t0 = time.perf_counter()
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, cwd=ROOT)
    ps = psutil.Process(proc.pid)
    peak_sum = 0
    peak_wset = 0
    seen: dict[int, psutil.Process] = {}
    while proc.poll() is None:
        try:
            procs = [ps, *ps.children(recursive=True)]
        except psutil.NoSuchProcess:
            break
        total = 0
        for p in procs:
            try:
                mi = p.memory_info()
                total += mi.rss
                seen[p.pid] = p
                if hasattr(mi, "peak_wset"):
                    peak_wset = max(peak_wset, mi.peak_wset)
            except psutil.Error:
                pass
        peak_sum = max(peak_sum, total)
        if time.perf_counter() - t0 > timeout_s:
            proc.kill()
            return {"status": "timeout", "wall_s": time.perf_counter() - t0}
        time.sleep(0.001)
    wall = time.perf_counter() - t0
    out, err = proc.communicate()
    if proc.returncode != 0:
        return {"status": "failed", "wall_s": wall, "stderr": err[-2000:]}
    line = next((l for l in out.splitlines() if l.startswith("BENCH_RESULT ")), None)
    if line is None:
        return {"status": "no_result", "wall_s": wall, "stderr": err[-2000:]}
    payload = json.loads(line[len("BENCH_RESULT "):])
    return {
        "status": "ok",
        "wall_s": wall,
        "peak_rss_tree_mb": peak_sum / 2**20,
        "peak_wset_mb": peak_wset / 2**20 if peak_wset else None,
        "payload": payload,
    }


def environment() -> dict:
    env = {
        "platform": platform.platform(),
        "processor": platform.processor(),
        "cpu_count_logical": os.cpu_count(),
        "cpu_count_physical": psutil.cpu_count(logical=False),
        "ram_gb": round(psutil.virtual_memory().total / 2**30, 1),
        "python": sys.version.split()[0],
        "git_commit": subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, cwd=ROOT).stdout.strip(),
    }
    try:
        env["rustc"] = subprocess.run(["rustc", "--version"], capture_output=True, text=True).stdout.strip()
        env["r"] = subprocess.run([RSCRIPT, "--version"], capture_output=True, text=True).stdout.strip() or subprocess.run([RSCRIPT, "--version"], capture_output=True, text=True).stderr.strip()
    except OSError:
        pass
    return env


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-id", required=True)
    ap.add_argument("--tasks", required=True, help="comma-separated task ids")
    ap.add_argument("--systems", default="r,rust")
    ap.add_argument("--engines", default="linfa,smartcore", help="Rust engines")
    ap.add_argument("--threads", default="1")
    ap.add_argument("--trials", type=int, default=5)
    ap.add_argument("--warm", type=int, default=10)
    ap.add_argument("--timeout", type=float, default=600.0)
    a = ap.parse_args()

    out_dir = ROOT / "benchmarks" / "results" / a.run_id
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "env.json").write_text(json.dumps(environment(), indent=1))
    if "rust" in a.systems and not BENCH_BIN.exists():
        raise SystemExit(f"build the bench binary first: cargo build --release --manifest-path benchmarks/rust/Cargo.toml ({BENCH_BIN})")

    fields = [
        "run_id", "timestamp", "system", "engine", "task", "task_type", "threads", "trial", "status",
        "wall_s", "peak_rss_tree_mb", "peak_wset_mb", "n_rows", "n_features",
        "load_s", "prep_s", "bake_s", "fit_s", "predict_s", "cv_s", "tune_s", "library_load_s",
        "warm_iters", "warm_prep_bake_s", "warm_fit_s", "warm_predict_s", "cv_summary", "stderr",
    ]
    path = out_dir / "results.csv"
    new = not path.exists()
    with open(path, "a", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fields)
        if new:
            w.writeheader()
        for task_id in a.tasks.split(","):
            spec = task_spec(task_id)
            if not Path(spec["dataset"]).exists():
                print(f"[skip] {task_id}: dataset missing ({spec['dataset']})")
                continue
            for threads in [int(t) for t in a.threads.split(",")]:
                configs = []
                if "r" in a.systems.split(","):
                    configs.append(("r", "lm" if spec["type"] == "regression" else "glm"))
                if "rust" in a.systems.split(","):
                    configs += [("rust", e) for e in a.engines.split(",")]
                for system, engine in configs:
                    for trial in range(a.trials):
                        cmd = command(system, spec, engine, threads, a.warm)
                        res = run_trial(cmd, a.timeout)
                        row = {
                            "run_id": a.run_id, "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
                            "system": system, "engine": engine, "task": task_id, "task_type": spec["type"],
                            "threads": threads, "trial": trial, "status": res["status"],
                            "wall_s": round(res.get("wall_s", float("nan")), 4),
                            "peak_rss_tree_mb": res.get("peak_rss_tree_mb"), "peak_wset_mb": res.get("peak_wset_mb"),
                            "stderr": res.get("stderr", ""),
                        }
                        p = res.get("payload")
                        if p:
                            ph = p["phases"]
                            row.update({
                                "n_rows": p["n_rows"], "n_features": p["n_features"],
                                "load_s": ph.get("load_s"), "prep_s": ph.get("prep_s"), "bake_s": ph.get("bake_s"),
                                "fit_s": ph.get("fit_s"), "predict_s": ph.get("predict_s"), "cv_s": ph.get("cv_s"),
                                "tune_s": ph.get("tune_s"), "library_load_s": ph.get("library_load_s"),
                                "warm_iters": p["warm"]["iters"], "warm_prep_bake_s": p["warm"]["prep_bake_s_median"],
                                "warm_fit_s": p["warm"]["fit_s_median"], "warm_predict_s": p["warm"]["predict_s_median"],
                                "cv_summary": json.dumps(p.get("cv_summary")),
                            })
                        w.writerow(row)
                        fh.flush()
                        print(f"{task_id:<16} {system:<5} {engine:<9} t={threads} trial={trial} {res['status']:<8} wall={row['wall_s']}s fit={row.get('fit_s')} cv={row.get('cv_s')}")
    print("wrote", path)


if __name__ == "__main__":
    main()
