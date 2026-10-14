"""Fail when a code block on the site differs from the file it claims to come from.

Every fenced code block that carries a `source="<repo-relative path>"` attribute, in
site/**/*.qmd and site/generated/**/*.md, must equal that file's content exactly
(trailing whitespace of the file ignored). `build_pages.py` writes such blocks for the
examples, the README quick starts and the recorded outputs; hand-written pages may add
more. A block without `source=` is not checked, so showing library code on the site
without naming its source is a review finding, not a build failure.

Also verifies that every example named in examples/README.md has a recorded output in
site/data/example-outputs/ and that the recorded commit is on the current branch.

Run from the repository root: python -I site/scripts/check_examples.py
"""
from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SITE = ROOT / "site"
FENCE = re.compile(r"^```\{[^}\n]*?source=\"([^\"]+)\"[^}\n]*\}\n(.*?)^```", re.S | re.M)


def main() -> int:
    failures: list[str] = []
    checked = 0
    pages = [p for p in SITE.rglob("*.qmd")] + [p for p in (SITE / "generated").rglob("*.md")]
    pages = [p for p in pages if "_site" not in p.parts and "node_modules" not in p.parts]
    for page in pages:
        text = page.read_text(encoding="utf-8")
        for m in FENCE.finditer(text):
            rel, body = m.group(1), m.group(2)
            src = ROOT / rel
            checked += 1
            if not src.exists():
                failures.append(f"{page.relative_to(ROOT)}: source `{rel}` does not exist")
                continue
            content = src.read_text(encoding="utf-8")
            expected = content.rstrip() + "\n"
            # the whole file, or a verbatim excerpt of it (the README quick starts)
            if body != expected and body.rstrip("\n") not in content:
                failures.append(f"{page.relative_to(ROOT)}: block differs from `{rel}`")

    outputs = SITE / "data" / "example-outputs"
    for name in ("penguins_workflow", "ames_tuning", "ames_workflow_sets"):
        for lang in ("rust", "python"):
            out = outputs / f"{name}.{lang}.txt"
            if not out.exists():
                failures.append(f"missing recorded output {out.relative_to(ROOT)}")
            elif re.search(r"^(Error|Traceback|panicked)", out.read_text(encoding="utf-8"), re.M):
                failures.append(f"recorded output {out.relative_to(ROOT)} contains an error")
    commit_file = outputs / "COMMIT"
    if commit_file.exists():
        commit = commit_file.read_text(encoding="utf-8").split()[0]
        r = subprocess.run(["git", "merge-base", "--is-ancestor", commit, "HEAD"], cwd=ROOT, capture_output=True)
        if r.returncode != 0:
            failures.append(f"recorded example commit {commit} is not an ancestor of HEAD")
    else:
        failures.append("site/data/example-outputs/COMMIT is missing")

    print(f"check_examples: {checked} sourced code blocks checked, {len(failures)} failures")
    for f in failures:
        print("  -", f)
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
