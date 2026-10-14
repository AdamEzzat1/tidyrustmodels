"""Fail when the README's benchmark summary disagrees with the per-trial CSVs.

Recomputes the headline ranges (site/scripts/build_pages.py `headline()`, the same
function that renders the site's headline table) and checks that every range the
README paragraph "What is measured today" quotes lies within 4% of the computed one
(the paragraph rounds to two or three significant digits) and that the quoted string
is present in README.md. The quoted values live in CLAIMS below: when the README
changes, change them here, and this script says whether the data agrees.

Run from the repository root: python -I site/scripts/check_ratios.py
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import build_pages as bp  # noqa: E402

LIN = "linear / logistic (`lm` / `glm` vs the native ports)"
TREE = "rpart-style trees"
XGB = "xgboost-style boosting"
RF = "random forests vs ranger (different forests)"

# (headline key, README-quoted low, README-quoted high, README string that must appear)
CLAIMS = [
    ((LIN, "workflow fit, cold"), 3.4, 930, "3.4–930×"),
    ((LIN, "10-fold CV"), 3.7, 2200, "3.7–2200×"),
    ((LIN, "tuning grid"), 1.1, 360, "1.1–360×"),
    ((TREE, "workflow fit, cold"), 14, 270, "14–270×"),
    ((TREE, "workflow fit, warm"), 13, 450, "13–450×"),
    ((TREE, "10-fold CV"), 13, 470, "13–470×"),
    ((TREE, "tuning grid"), 13, 425, "13–425×"),
    ((XGB, "workflow fit, cold"), 1.9, 4.4, "1.9–4.4×"),
    ((XGB, "workflow fit, warm"), 2.0, 4.6, "2.0–4.6×"),
    ((XGB, "10-fold CV"), 2.6, 7.8, "2.6–7.8×"),
    ((XGB, "tuning grid"), 4.1, 4.1, "4.1×"),
    ((RF, "workflow fit, cold"), 0.87, 1.04, "0.86–1.04×"),
    ((RF, "10-fold CV"), 0.98, 1.13, "0.98–1.13×"),
    (("workflow sets", "workflow set, end to end"), 14, 270, "14–270×"),
    (("memory", "R / Rust peak working set, 1 thread"), 3.5, 70, "3.5–70×"),
    (("memory", "R / Rust peak working set, all cells"), 3.1, 70, "3.1–70×"),
    (("memory", "R / Python peak working set, all cells"), 2.4, 5.1, "2.4–5.1×"),
    (("python wrapper", "Python / Rust on the 1e6-row and forest phases"), 0.95, 1.03, "0.95–1.03×"),
    (("python wrapper", "Python / Rust on the other phases above 0.1 s"), 0.57, 1.2, "0.57–1.2×"),
]
# single points the paragraph names
POINTS = [
    ("OLS 930× at 342 rows", "penguins_reg @1", "wf_fit_s", 930),
    ("OLS 12× at 100k", "synth_100000_reg @1", "wf_fit_s", 12),
    ("OLS 6.1× at 1e6", "synth_1000000_reg @1", "wf_fit_s", 6.1),
    ("logistic 3.4× at 1e6", "synth_1000000_clf @1", "wf_fit_s", 3.4),
]
TOL = 0.04


def close(q: float, c: float) -> bool:
    return abs(q - c) <= TOL * c


def main() -> int:
    import contextlib
    import io

    readme = (bp.ROOT / "README.md").read_text(encoding="utf-8")
    with contextlib.redirect_stdout(io.StringIO()):  # build_bench prints a summary line
        cells = bp.build_bench()
    h = bp.headline(cells)
    failures: list[str] = []
    for key, qlo, qhi, text in CLAIMS:
        if key not in h:
            failures.append(f"no data for {key}")
            continue
        lo, lo_t, hi, hi_t, _ = h[key]
        if not (close(qlo, lo) and close(qhi, hi)):
            failures.append(f"{key}: README says {text}, data says {lo:.3g}× ({lo_t}) – {hi:.3g}× ({hi_t})")
        if text not in readme:
            failures.append(f"README.md does not contain `{text}` for {key}")
    by_task = {(task_thr): systems for task_thr, systems in bp.ratio_cells(cells, 1).items()}
    for label, task_thr, phase, q in POINTS:
        task, thr = task_thr.split(" @")
        systems = by_task.get((task, int(thr)))
        if not systems:
            failures.append(f"{label}: no cell {task_thr}")
            continue
        c = systems["r"]["stats"][phase]["median"] / systems["rust"]["stats"][phase]["median"]
        if not close(q, c):
            failures.append(f"{label}: data says {c:.3g}×")
    print(f"check_ratios: {len(CLAIMS)} ranges and {len(POINTS)} points recomputed from the CSVs, {len(failures)} disagreements")
    for f in failures:
        print("  -", f)
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
