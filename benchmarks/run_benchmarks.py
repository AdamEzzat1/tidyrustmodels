"""Benchmark orchestrator: runs R tidymodels, Rust tidyrustmodels and the tidyrustmodels
Python wrapper on identical tasks.

Each trial is a fresh process (cold start). The orchestrator measures:
  * wall-clock time of the whole process (includes interpreter/library/binary load and
    CSV parsing);
  * peak memory, two ways:
      - `peak_rss_tree_mb`: maximum over time of the summed RSS of the process and its
        children, sampled every ~1 ms. Misses processes shorter than a few samples and is
        a lower bound in general.
      - `peak_wset_mb`: the OS-tracked exact peak of the largest single process in the
        tree, read *after* exit. Windows: `PeakWorkingSetSize` via GetProcessMemoryInfo on
        a handle retained for every process seen (Rscript.exe and venv python.exe are
        launchers; the work happens in a child). POSIX: `ru_maxrss` of RUSAGE_CHILDREN
        measured by a tiny wrapper (benchmarks/_rusage_wrap.py).
  * the per-phase timings the process itself reports on its `BENCH_RESULT` line.

Usage (from the repository root, inside the venv):
    python -I benchmarks/run_benchmarks.py --run-id 2026-10-12a --trials 5 --warm 10 \
        --tasks penguins_reg,penguins_clf,ames_reg,synth_1000_reg,synth_10000_reg \
        --threads 1,4 --systems r,rust,python

Results: benchmarks/results/<run_id>/results.csv and env.json. Then run report.py.
"""
from __future__ import annotations

import argparse
import csv
import ctypes
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
PY_DRIVER = ROOT / "benchmarks" / "python" / "bench_task.py"
RUSAGE_WRAP = ROOT / "benchmarks" / "_rusage_wrap.py"

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
    if system == "python":
        return [sys.executable, "-I", str(PY_DRIVER), "--engine", engine, *common]
    return [RSCRIPT, str(ROOT / "benchmarks" / "r" / "bench_task.R"), *common]


# ---------------------------------------------------------------- exact peak memory

class _PROCESS_MEMORY_COUNTERS(ctypes.Structure):
    _fields_ = [
        ("cb", ctypes.c_uint32),
        ("PageFaultCount", ctypes.c_uint32),
        ("PeakWorkingSetSize", ctypes.c_size_t),
        ("WorkingSetSize", ctypes.c_size_t),
        ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
        ("QuotaPagedPoolUsage", ctypes.c_size_t),
        ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
        ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
        ("PagefileUsage", ctypes.c_size_t),
        ("PeakPagefileUsage", ctypes.c_size_t),
    ]


PROCESS_QUERY_LIMITED_INFORMATION = 0x1000


def windows_open_handle(pid: int):
    """Open a query-only handle so the process object (and its peak counters) outlives exit."""
    h = ctypes.windll.kernel32.OpenProcess(PROCESS_QUERY_LIMITED_INFORMATION, False, int(pid))
    return h or None


def windows_peak_working_set(handle) -> int | None:
    """Exact peak working set of one process, readable after exit while a handle is open."""
    pmc = _PROCESS_MEMORY_COUNTERS()
    pmc.cb = ctypes.sizeof(pmc)
    fn = ctypes.windll.kernel32.K32GetProcessMemoryInfo
    fn.argtypes = [ctypes.c_void_p, ctypes.POINTER(_PROCESS_MEMORY_COUNTERS), ctypes.c_uint32]
    fn.restype = ctypes.c_int
    if not fn(int(handle), ctypes.byref(pmc), pmc.cb):
        return None
    return int(pmc.PeakWorkingSetSize)


def run_trial(cmd: list[str], timeout_s: float) -> dict:
    """Spawn, sample the process tree's RSS, read the exact peak after exit, parse BENCH_RESULT."""
    wrapped = os.name != "nt"
    if wrapped:
        cmd = [sys.executable, "-I", str(RUSAGE_WRAP), *cmd]
    t0 = time.perf_counter()
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, cwd=ROOT)
    ps = psutil.Process(proc.pid)
    peak_sum = 0
    # Windows: Rscript.exe and a venv python.exe are launchers whose real work happens in a
    # child, so every process ever seen in the tree gets its own retained handle.
    handles: dict[int, object] = {}
    if os.name == "nt":
        handles[proc.pid] = getattr(proc, "_handle", None)
    while proc.poll() is None:
        try:
            procs = [ps, *ps.children(recursive=True)]
        except psutil.NoSuchProcess:
            break
        total = 0
        for p in procs:
            try:
                total += p.memory_info().rss
            except psutil.Error:
                continue
            if os.name == "nt" and p.pid not in handles:
                handles[p.pid] = windows_open_handle(p.pid)
        peak_sum = max(peak_sum, total)
        if time.perf_counter() - t0 > timeout_s:
            proc.kill()
            return {"status": "timeout", "wall_s": time.perf_counter() - t0}
        time.sleep(0.001)
    wall = time.perf_counter() - t0
    peak_wset = None
    if os.name == "nt":
        peaks = [windows_peak_working_set(h) for h in handles.values() if h]
        peaks = [v for v in peaks if v]
        peak_wset = max(peaks) if peaks else None
        for pid, h in handles.items():
            if h and pid != proc.pid:
                ctypes.windll.kernel32.CloseHandle(h)
    out, err = proc.communicate()
    if wrapped:
        for line in err.splitlines():
            if line.startswith("BENCH_RUSAGE_MAXRSS_KB "):
                peak_wset = int(line.split()[1]) * 1024
        err = "\n".join(l for l in err.splitlines() if not l.startswith("BENCH_RUSAGE_MAXRSS_KB "))
    if proc.returncode != 0:
        return {"status": "failed", "wall_s": wall, "stderr": err[-2000:]}
    line = next((l for l in out.splitlines() if l.startswith("BENCH_RESULT ")), None)
    if line is None:
        return {"status": "no_result", "wall_s": wall, "stderr": err[-2000:]}
    payload = json.loads(line[len("BENCH_RESULT "):])
    return {
        "status": "ok",
        "wall_s": wall,
        "peak_rss_tree_mb": peak_sum / 2**20 if peak_sum else None,
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
        "peak_memory_method": "Windows PeakWorkingSetSize via retained handles, max over every process seen in the tree (exact per process)" if os.name == "nt"
                              else "POSIX ru_maxrss of RUSAGE_CHILDREN via wrapper (exact, largest descendant)",
    }
    try:
        env["rustc"] = subprocess.run(["rustc", "--version"], capture_output=True, text=True).stdout.strip()
        env["r"] = subprocess.run([RSCRIPT, "--version"], capture_output=True, text=True).stdout.strip() or subprocess.run([RSCRIPT, "--version"], capture_output=True, text=True).stderr.strip()
    except OSError:
        pass
    return env


FIELDS = [
    "run_id", "timestamp", "system", "engine", "task", "task_type", "threads", "trial", "status",
    "wall_s", "peak_rss_tree_mb", "peak_wset_mb", "n_rows", "n_features",
    "load_s", "prep_s", "bake_s", "fit_s", "wf_fit_s", "predict_s", "cv_s", "tune_s", "library_load_s",
    "warm_iters", "warm_prep_bake_s", "warm_fit_s", "warm_wf_fit_s", "warm_predict_s", "cv_summary", "stderr",
]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-id", required=True)
    ap.add_argument("--tasks", required=True, help="comma-separated task ids")
    ap.add_argument("--systems", default="r,rust,python")
    ap.add_argument("--engines", default="linfa,smartcore", help="Rust/Python engines")
    ap.add_argument("--threads", default="1")
    ap.add_argument("--trials", type=int, default=5)
    ap.add_argument("--warm", type=int, default=10)
    ap.add_argument("--timeout", type=float, default=600.0)
    a = ap.parse_args()
    systems = a.systems.split(",")

    out_dir = ROOT / "benchmarks" / "results" / a.run_id
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "env.json").write_text(json.dumps(environment(), indent=1))
    if "rust" in systems and not BENCH_BIN.exists():
        raise SystemExit(f"build the bench binary first: cargo build --release --manifest-path benchmarks/rust/Cargo.toml ({BENCH_BIN})")
    if "python" in systems:
        chk = subprocess.run([sys.executable, "-I", "-c", "import tidyrustmodels"], capture_output=True, text=True)
        if chk.returncode != 0:
            raise SystemExit("python system requested but `import tidyrustmodels` fails in this interpreter "
                             "(run `maturin develop --release` in the venv first):\n" + chk.stderr[-500:])

    path = out_dir / "results.csv"
    new = not path.exists()
    with open(path, "a", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=FIELDS)
        if new:
            w.writeheader()
        for task_id in a.tasks.split(","):
            spec = task_spec(task_id)
            if not Path(spec["dataset"]).exists():
                print(f"[skip] {task_id}: dataset missing ({spec['dataset']})")
                continue
            for threads in [int(t) for t in a.threads.split(",")]:
                configs = []
                if "r" in systems:
                    configs.append(("r", "lm" if spec["type"] == "regression" else "glm"))
                for system in ("rust", "python"):
                    if system in systems:
                        configs += [(system, e) for e in a.engines.split(",")]
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
                            wm = p["warm"]
                            row.update({
                                "n_rows": p["n_rows"], "n_features": p["n_features"],
                                "load_s": ph.get("load_s"), "prep_s": ph.get("prep_s"), "bake_s": ph.get("bake_s"),
                                "fit_s": ph.get("fit_s"), "wf_fit_s": ph.get("wf_fit_s"), "predict_s": ph.get("predict_s"),
                                "cv_s": ph.get("cv_s"), "tune_s": ph.get("tune_s"), "library_load_s": ph.get("library_load_s"),
                                "warm_iters": wm["iters"], "warm_prep_bake_s": wm.get("prep_bake_s_median"),
                                "warm_fit_s": wm.get("fit_s_median"), "warm_wf_fit_s": wm.get("wf_fit_s_median"),
                                "warm_predict_s": wm.get("predict_s_median"),
                                "cv_summary": json.dumps(p.get("cv_summary")),
                            })
                        w.writerow(row)
                        fh.flush()
                        print(f"{task_id:<16} {system:<6} {engine:<9} t={threads} trial={trial} {res['status']:<8} "
                              f"wall={row['wall_s']}s wf_fit={row.get('wf_fit_s')} cv={row.get('cv_s')} "
                              f"peak_ws={row.get('peak_wset_mb')}")
    print("wrote", path)


if __name__ == "__main__":
    main()
