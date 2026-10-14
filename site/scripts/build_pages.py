"""Render repository documents and benchmark data into `site/generated/`.

Run from the repository root before `quarto render site`:

    python -I site/scripts/build_pages.py

Everything the site says about the library comes from files in the repository; this
script copies or transforms them so the pages can include them. Nothing here is
hand-written content. Outputs (all under site/generated/, which is not committed):

    features.md                  docs/FEATURES.md without its H1
    install.md                   README.md "Installing" section
    quickstart-rust.md           README.md "Quick start (Rust)" code block
    quickstart-python.md         README.md "Quick start (Python)" code block
    examples/<name>.<lang>.md    full example source as a fenced block with `source=`
    examples/<name>.<lang>.out.md   its recorded output (site/data/example-outputs/)
    limitations.md               LIMITATIONS.md "Known limitations" section
    unsupported.md               COMPATIBILITY.md "Unsupported in v0.1" + the
                                 "not supported" rows of its contract tables
    roadmap.md, changelog.md     ROADMAP.md / CHANGELOG.md without their H1
    adr-index.md + ../adr/*.md   index table and copies of docs/adr/*.md
    verification-phase4.md       VERIFICATION.md Phase 4 summary (how the suite works)
    verification-platforms.md    VERIFICATION.md cross-platform section (ADR-0017)
    verification-inventory.md    VERIFICATION.md test inventory
    benchmarks-deviations.md     BENCHMARKS.md deviations + disturbance log (Phase 5)
    benchmarks-findings.md       BENCHMARKS.md per-run findings bullets (runs G–H)
    python-api.md                public API of python/tidyrustmodels (from `__all__`)
    ../data/bench.json           medians / quartiles per task × threads × system from
                                 benchmarks/results/<run>/results.csv (Phase 5 runs)
"""
from __future__ import annotations

import csv
import io
import json
import re
import statistics
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SITE = ROOT / "site"
GEN = SITE / "generated"
DATA = SITE / "data"


def read(p: Path) -> str:
    return p.read_text(encoding="utf-8")


def write(p: Path, text: str) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)
    with io.open(p, "w", encoding="utf-8", newline="\n") as f:
        f.write(text if text.endswith("\n") else text + "\n")


def strip_h1(text: str) -> str:
    lines = text.splitlines()
    if lines and lines[0].startswith("# "):
        lines = lines[1:]
    return "\n".join(lines).lstrip("\n")


def section(text: str, heading: str, level: str = "## ") -> str:
    """Body of the section whose heading line starts with `level + heading`, up to the
    next heading of the same or higher level."""
    lines = text.splitlines()
    start = next(i for i, l in enumerate(lines) if l.startswith(level + heading))
    depth = len(level.strip())
    end = len(lines)
    in_fence = False
    for j in range(start + 1, len(lines)):
        if lines[j].startswith("```"):
            in_fence = not in_fence  # a `# comment` inside a code block is not a heading
            continue
        if in_fence:
            continue
        m = re.match(r"^(#+) ", lines[j])
        if m and len(m.group(1)) <= depth:
            end = j
            break
    return "\n".join(lines[start + 1 : end]).strip("\n")


def fenced(lang: str, source: Path, body: str) -> str:
    rel = source.relative_to(ROOT).as_posix()
    return f'```{{.{lang} source="{rel}"}}\n{body.rstrip()}\n```\n'


def provenance(label: str, source: Path) -> str:
    rel = source.relative_to(ROOT).as_posix()
    return f'<p class="provenance">{label} <a href="{repo_url(rel)}"><code>{rel}</code></a></p>\n\n'


def repo_url(rel: str) -> str:
    return f"https://github.com/AdamEzzat1/tidyrustmodels/blob/main/{rel}"


# ---------- documents ----------

def build_docs() -> None:
    feats = ROOT / "docs" / "FEATURES.md"
    write(GEN / "features.md", provenance("Rendered from", feats) + strip_h1(read(feats)))

    readme = read(ROOT / "README.md")
    write(GEN / "install.md", provenance("From", ROOT / "README.md") + section(readme, "Installing"))
    for lang, heading in (("rust", "Quick start (Rust)"), ("python", "Quick start (Python)")):
        body = section(readme, heading)
        m = re.search(r"```(\w+)\n(.*?)```", body, re.S)
        if not m:
            raise SystemExit(f"README: no code block under '{heading}'")
        write(GEN / f"quickstart-{lang}.md", fenced(m.group(1), ROOT / "README.md", m.group(2)))

    lim = read(ROOT / "LIMITATIONS.md")
    write(GEN / "limitations.md", provenance("From", ROOT / "LIMITATIONS.md") + section(lim, "Known limitations"))

    compat = read(ROOT / "COMPATIBILITY.md")
    unsupported = section(compat, "Unsupported in v0.1")
    rows = [l for l in compat.splitlines() if l.startswith("| ") and re.search(r"\|\s*not supported\s*\|", l)]
    table = "\n".join(["| tidymodels | status | note | verified |", "|---|---|---|---|", *rows])
    write(
        GEN / "unsupported.md",
        provenance("From", ROOT / "COMPATIBILITY.md")
        + unsupported
        + "\n\nRows of the compatibility contract marked *not supported*:\n\n"
        + table,
    )

    for name in ("ROADMAP", "CHANGELOG"):
        p = ROOT / f"{name}.md"
        write(GEN / f"{name.lower()}.md", provenance("Rendered from", p) + strip_h1(read(p)))

    ver = read(ROOT / "VERIFICATION.md")
    p4 = section(ver, "Phase 4: differential verification")
    p4 = p4.split("\nInterpretation:")[0]
    write(GEN / "verification-phase4.md", provenance("From", ROOT / "VERIFICATION.md") + p4)
    write(
        GEN / "verification-platforms.md",
        provenance("From", ROOT / "VERIFICATION.md") + section(ver, "The differential suite on Linux and macOS"),
    )
    write(GEN / "verification-inventory.md", provenance("From", ROOT / "VERIFICATION.md") + section(ver, "Test inventory"))

    bench = read(ROOT / "BENCHMARKS.md")
    phase5 = section(bench, "Phase 5 protocol runs", "### ")
    dev_start = phase5.index("Deviations from the protocol")
    write(GEN / "benchmarks-deviations.md", provenance("From", ROOT / "BENCHMARKS.md") + phase5[dev_start:])
    write(GEN / "benchmarks-findings.md", provenance("From", ROOT / "BENCHMARKS.md") + findings(bench))

    build_adrs()
    build_examples()
    build_python_api()


def findings(bench: str) -> str:
    """The `- **...**` bullet blocks of each Phase 5 run section (G, K, I, J, H), and
    the "Python wrapper vs Rust binary" paragraph of run G."""
    out: list[str] = []
    run = None
    block: list[str] = []
    in_phase5 = False

    def flush() -> None:
        nonlocal block
        if block:
            out.append("\n".join(block))
            block = []

    for line in bench.splitlines():
        if line.startswith("### "):
            in_phase5 = line.startswith("### Phase 5 protocol runs")
            flush()
            run = None
            continue
        if not in_phase5:
            continue
        m = re.match(r"^#### Run ([A-Z]):(.*)", line)
        if m:
            flush()
            run = m.group(1)
            out.append(f"\n### Run {run}: {m.group(2).strip()}\n")
            continue
        if run is None:
            continue
        if line.startswith("- **") or line.startswith("**Python wrapper vs Rust binary.**"):
            flush()
            block = [line]
        elif block and (line.startswith("  ") or (line and not line.startswith(("|", "#", "- ", "Full per-phase", "wfs_best")) and not line.startswith("```"))):
            block.append(line)
        else:
            flush()
    flush()
    return "\n".join(out).strip() + "\n"


def build_adrs() -> None:
    adr_dir = ROOT / "docs" / "adr"
    rows = ["| ADR | title | date | status |", "|---|---|---|---|"]
    for p in sorted(adr_dir.glob("*.md")):
        text = read(p)
        lines = text.splitlines()
        title = lines[0].lstrip("# ").strip()
        num, _, rest = title.partition(":")
        meta = next((l for l in lines[1:6] if l.startswith("Date:")), "")
        date = re.search(r"Date:\s*([0-9-]+)", meta)
        status = re.search(r"Status:\s*([^.]+)", meta)
        slug = p.stem
        rows.append(
            f"| [{num.strip()}](adr/{slug}.html) | {rest.strip()} | {date.group(1) if date else ''} | {status.group(1).strip() if status else ''} |"
        )
        write(
            SITE / "adr" / f"{slug}.md",
            "---\ntitle: \"" + title.replace('"', "'") + "\"\n---\n\n"
            + provenance("Rendered from", p)
            + strip_h1(text),
        )
    write(GEN / "adr-index.md", provenance("Index of", adr_dir / "*.md") + "\n".join(rows))


def build_examples() -> None:
    names = ("penguins_workflow", "ames_tuning", "ames_workflow_sets")
    commit_file = DATA / "example-outputs" / "COMMIT"
    stamp = commit_file.read_text(encoding="utf-8").split() if commit_file.exists() else ["?", "?"]
    for name in names:
        for lang, src in (("rust", ROOT / "crates" / "tidyrustmodels" / "examples" / f"{name}.rs"), ("python", ROOT / "examples" / f"{name}.py")):
            write(GEN / "examples" / f"{name}.{lang}.md", fenced(lang, src, read(src)))
            out = DATA / "example-outputs" / f"{name}.{lang}.txt"
            if out.exists():
                rel = out.relative_to(ROOT).as_posix()
                write(
                    GEN / "examples" / f"{name}.{lang}.out.md",
                    f'<p class="provenance">Output recorded at commit <code>{stamp[0]}</code> on {stamp[1]} '
                    f'(<a href="{repo_url(rel)}"><code>{rel}</code></a>):</p>\n\n'
                    + f'```{{.text source="{rel}"}}\n{read(out).rstrip()}\n```\n',
                )


def build_python_api() -> None:
    init = ROOT / "python" / "tidyrustmodels" / "__init__.py"
    text = read(init)
    m = re.search(r"__all__\s*=\s*\[(.*?)\]", text, re.S)
    names = re.findall(r'"([^"]+)"', m.group(1)) if m else []
    sys.path.insert(0, str(ROOT / "python"))
    try:
        import tidyrustmodels as trm  # type: ignore

        live = True
    except Exception as e:  # noqa: BLE001 - the extension may not be built here
        trm = None
        live = False
        reason = f"{type(e).__name__}: {e}"
    head = provenance("Generated from", init)
    if not live:
        head += (
            "::: {.callout-note}\nThe compiled extension was not importable when this page was built "
            f"(`{reason}`), so only the exported names are listed; docstrings appear once the site is "
            "built with the extension installed (the deploy workflow does that).\n:::\n\n"
        )
    rows = ["| name | kind | summary |", "|---|---|---|"]
    details: list[str] = []
    for n in names:
        if n == "__version__":
            continue
        obj = getattr(trm, n, None) if live else None
        kind = "class" if isinstance(obj, type) else ("function" if callable(obj) else ("exception" if obj is None else type(obj).__name__))
        if live and isinstance(obj, type) and issubclass(obj, BaseException):
            kind = "exception"
        doc = (obj.__doc__ or "").strip().splitlines()[0] if (live and obj is not None and obj.__doc__) else ""
        rows.append(f"| `{n}` | {kind} | {doc} |")
        if live and isinstance(obj, type) and not issubclass(obj, BaseException):
            methods = []
            for attr in sorted(vars(obj)):
                if attr.startswith("_"):
                    continue
                member = getattr(obj, attr)
                d = (getattr(member, "__doc__", "") or "").strip().splitlines()
                methods.append(f"| `{attr}` | {d[0] if d else ''} |")
            if methods:
                details.append(f"\n#### `{n}`\n\n| member | summary |\n|---|---|\n" + "\n".join(methods) + "\n")
    version = getattr(trm, "__version__", None) if live else None
    body = head + (f"Package version `{version}`.\n\n" if version else "") + "\n".join(rows) + "\n" + "".join(details)
    write(GEN / "python-api.md", body)


# ---------- benchmark data ----------

PHASE5_RUNS = {
    "2026-10-13-win11-i7-11390H-g": "G",
    "2026-10-13-win11-i7-11390H-k": "K",
    "2026-10-13-win11-i7-11390H-i": "I",
    "2026-10-13-win11-i7-11390H-j": "J",
    "2026-10-13-win11-i7-11390H-h": "H",
}
# Configurations BENCHMARKS.md marks as disturbed or invalid (its disturbance log).
INVALID = {("G", "synth_100000_reg", 4), ("G", "synth_100000_clf", 1), ("G", "synth_100000_clf", 4)}
DISTURBED = {
    ("G", "synth_100000_reg", 1): "trials 5–6 mildly disturbed (BENCHMARKS.md run G)",
    ("H", "synth_1000000_reg", 1): "trials 7–9 ran 1.5–1.8× slower on all systems (BENCHMARKS.md run H)",
}
PHASES = ("wall_s", "wf_fit_s", "warm_wf_fit_s", "cv_s", "tune_s", "wfs_s", "peak_wset_mb")


def quartiles(xs: list[float]) -> dict:
    xs = sorted(xs)
    q = statistics.quantiles(xs, n=4, method="inclusive") if len(xs) >= 2 else [xs[0], xs[0], xs[0]]
    return {"median": statistics.median(xs), "q1": q[0], "q3": q[2], "n": len(xs)}


def build_bench() -> None:
    sys.path.insert(0, str(ROOT / "benchmarks"))
    try:
        import tasks as task_registry  # type: ignore
    except Exception:  # noqa: BLE001
        task_registry = None
    cells: dict[tuple, dict] = {}
    for run_id, letter in PHASE5_RUNS.items():
        p = ROOT / "benchmarks" / "results" / run_id / "results.csv"
        if not p.exists():
            continue
        with open(p, encoding="utf-8", newline="") as f:
            for row in csv.DictReader(f):
                if row["status"] != "ok":
                    continue
                task, threads, system = row["task"], int(row["threads"]), row["system"]
                if (letter, task, threads) in INVALID:
                    continue
                key = (letter, task, threads, system)
                c = cells.setdefault(
                    key,
                    {
                        "run": letter,
                        "run_id": run_id,
                        "task": task,
                        "threads": threads,
                        "system": system,
                        "engine": row["engine"],
                        "model": row["model"],
                        "task_type": row["task_type"],
                        "n_rows": int(float(row["n_rows"])),
                        "n_resamples": int(float(row["n_resamples"] or 0)),
                        "samples": {ph: [] for ph in PHASES},
                        "disturbed": DISTURBED.get((letter, task, threads)),
                    },
                )
                for ph in PHASES:
                    v = row.get(ph, "")
                    if v not in ("", None):
                        c["samples"][ph].append(float(v))
    out = []
    for c in cells.values():
        c["stats"] = {ph: quartiles(v) for ph, v in c["samples"].items() if v}
        del c["samples"]
        if task_registry is not None:
            try:
                spec = task_registry.task_spec(c["task"])
                c["category"] = task_registry.category(spec, c["engine"])
            except Exception:  # noqa: BLE001
                c["category"] = None
        out.append(c)
    out.sort(key=lambda c: (c["run"], c["n_rows"], c["task"], c["threads"], c["system"]))
    write(
        DATA / "bench.json",
        json.dumps(
            {
                "source": "benchmarks/results/<run>/results.csv, status == ok, medians and quartiles over cold trials; "
                "invalid configurations of run G excluded per BENCHMARKS.md; see `disturbed`",
                "runs": PHASE5_RUNS,
                "cells": out,
            },
            indent=1,
        ),
    )
    print(f"bench.json: {len(out)} cells from {len(PHASE5_RUNS)} runs")


if __name__ == "__main__":
    GEN.mkdir(parents=True, exist_ok=True)
    build_docs()
    build_bench()
    print("generated:", sorted(p.relative_to(SITE).as_posix() for p in GEN.rglob("*.md"))[:8], "...")
