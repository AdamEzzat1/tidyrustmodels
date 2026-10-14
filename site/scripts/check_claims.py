"""Fail when a hand-written page states a number that no repository document states.

Scans site/*.qmd (not the generated includes, which are copies of repository files)
for ratio, percentage and tolerance tokens:

    12×   6–930×   ±5%   1e-12   2.9e-5   17 542 comparisons

and requires each token to appear verbatim in at least one of BENCHMARKS.md,
VERIFICATION.md, LIMITATIONS.md, README.md, COMPATIBILITY.md, docs/FEATURES.md,
docs/releases/0.1.0.md, docs/adr/*.md, docs/perf/*.md or reference/fixtures/tolerances.toml.
Tokens inside fenced code blocks and inside `{{< include >}}` shortcodes are skipped.

Run from the repository root: python -I site/scripts/check_claims.py
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SITE = ROOT / "site"
SOURCES = [
    ROOT / "BENCHMARKS.md",
    ROOT / "VERIFICATION.md",
    ROOT / "LIMITATIONS.md",
    ROOT / "README.md",
    ROOT / "COMPATIBILITY.md",
    ROOT / "docs" / "FEATURES.md",
    ROOT / "docs" / "releases" / "0.1.0.md",
    ROOT / "reference" / "fixtures" / "tolerances.toml",
    *sorted((ROOT / "docs" / "adr").glob("*.md")),
    *sorted((ROOT / "docs" / "perf").glob("*.md")),
]
TOKEN = re.compile(
    r"(?<![\w.])(?:"
    r"\d[\d.,]*(?:\s?[–-]\s?\d[\d.,]*)?\s?×"  # 12×, 6–930×
    r"|±\s?\d+(?:\.\d+)?\s?%"  # ±5%
    r"|\d+(?:\.\d+)?e-\d+"  # 1e-12, 2.9e-5
    r"|\d{1,3}(?: \d{3})+ comparisons"  # 17 542 comparisons
    r")"
)


def strip_code(text: str) -> str:
    text = re.sub(r"```.*?```", "", text, flags=re.S)
    text = re.sub(r"`[^`\n]*`", "", text)
    text = re.sub(r"\{\{<.*?>\}\}", "", text)
    return text


def main() -> int:
    corpus = "\n".join(p.read_text(encoding="utf-8") for p in SOURCES if p.exists())
    corpus_norm = corpus.replace(" ", " ")
    failures: list[str] = []
    total = 0
    for page in sorted(SITE.glob("*.qmd")):
        text = strip_code(page.read_text(encoding="utf-8"))
        for m in TOKEN.finditer(text):
            tok = m.group(0).replace(" ", " ").strip()
            total += 1
            variants = {tok, tok.replace(" ×", "×"), tok.replace("×", " ×"), tok.replace(" – ", "–"), tok.replace("–", "-")}
            if not any(v in corpus_norm for v in variants):
                failures.append(f"{page.name}: `{tok}` not found in any source document")
    print(f"check_claims: {total} numeric claims on hand-written pages, {len(failures)} unsourced")
    for f in failures:
        print("  -", f)
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
