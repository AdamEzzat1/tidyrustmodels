# The tidyrustmodels website

A Quarto website (`site/`) with three small TypeScript components, deployed to GitHub
Pages by `.github/workflows/site.yml`. Its rule: every claim, number and code block on
the site is traced to a file in the repository, and the build checks it.

## Build locally

From the repository root (Quarto ≥ 1.3, Node ≥ 20, the project's `.venv`):

```bash
cd site && npm ci && npm run typecheck && npm run build && npm test && cd ..
python -I site/scripts/build_pages.py       # repository documents -> site/generated/, data/bench.json
python -I site/scripts/check_examples.py    # every `source="..."` code block equals its file
python -I site/scripts/check_claims.py      # every ×, %, 1e-N token on hand-written pages exists in the documents
python -I site/scripts/check_ratios.py      # README's benchmark ranges recomputed from benchmarks/results/*/results.csv
quarto render site                          # -> site/_site
quarto preview site                         # local preview
```

The Python API page carries docstrings when the extension is importable (`maturin
develop --release` first); otherwise it lists the exported names and says so. The Rust
API docs are attached by the workflow from `cargo doc` under `/api/rust/`.

## Where the content comes from

| page | source |
|---|---|
| Home quick starts | README.md "Quick start" blocks (verbatim excerpts) |
| Get started | README.md "Installing"; `crates/tidyrustmodels/examples/*.rs`, `examples/*.py`; outputs recorded in `site/data/example-outputs/` with the commit in `COMMIT` |
| What it can do | `docs/FEATURES.md` |
| Gaps | hand-written table compiled from `docs/FEATURES.md`, `COMPATIBILITY.md`, `LIMITATIONS.md`; the limitations and unsupported sections are included verbatim |
| Verification | `VERIFICATION.md` (Phase 4 summary, cross-platform section, test inventory) |
| Benchmarks | README.md headline ranges, `BENCHMARKS.md` deviations / disturbance log / per-run findings; charts from `benchmarks/results/<run>/results.csv` |
| Design decisions | `docs/adr/*.md` (index + rendered copies) |
| Changelog, Roadmap | `CHANGELOG.md`, `ROADMAP.md` |
| API reference | `python/tidyrustmodels/__init__.py` (`__all__`, docstrings); rustdoc |

`site/NOTES.md` is the claims ledger: every hand-written statement with its source.

## Recording the example outputs

The outputs shown on Get started are files, not live runs (Quarto cannot run Rust).
After changing an example or the library, re-record them and the commit:

```bash
cargo build --release --examples -p tidyrustmodels
for ex in penguins_workflow ames_tuning ames_workflow_sets; do
  target/release/examples/$ex > site/data/example-outputs/$ex.rust.txt
  python examples/$ex.py > site/data/example-outputs/$ex.python.txt
done
git rev-parse --short HEAD > site/data/example-outputs/COMMIT; date +%F >> site/data/example-outputs/COMMIT
```

`check_examples.py` fails if an output contains an error or the recorded commit is not
an ancestor of `HEAD`. CI (`ci.yml`) runs all six examples on every push.

## Components (`site/ts/`)

- `code-tabs.ts`: remembers the Rust / Python tab choice across pages (localStorage in
  try/catch; the page works without it).
- `table-filter.ts`: a search box above every `.filterable` table.
- `bench-chart.ts`: SVG bar charts from `data/bench.json` for `[data-bench-chart]`
  elements (`data-phase`, `data-runs`, `data-threads`, `data-tasks`, `data-models`).

All three are compiled by esbuild (pinned in `package.json`) into `assets/*.js`, which
is not committed. `npm test` runs them against jsdom pages. The pages render fully
without JavaScript: tables and code are static HTML; only filtering, tab memory and the
charts need it (the chart numbers are also in the included BENCHMARKS.md text).

## Theme

`theme.scss` (light) and `theme-dark.scss` layered on Quarto's `cosmo`: deep ink base,
teal-grey secondary text, one raspberry-family accent, Rust terracotta and Python teal
only for the language markers. Fonts are system stacks; `assets/site.css` holds
commented `@font-face` templates for self-hosting Source Sans 3 / JetBrains Mono under
`assets/fonts/` (no font is fetched from a third party at runtime). The logo
(`assets/logo.svg`) is an original mark.

## Deploy

`site.yml` builds on every push to `main` that touches the site or the documents it
renders, and on manual dispatch. The repository's Pages source must be set to **GitHub
Actions** (Settings → Pages) once; until then the deploy job fails with a
configuration error while the build job still verifies everything.
