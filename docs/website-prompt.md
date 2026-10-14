# Prompt: the tidyrustmodels website (Quarto + TypeScript, GitHub Pages)

> Draft, 2026-10-14. Hand this to a fresh session in `C:\Users\adame\tidyrustmodels`.

You are a **stack role group** building the public website for `tidyrustmodels`, a
Rust-native machine-learning framework with tidymodels semantics and a Python wrapper.
Work as five roles that hand off to each other and review each other's output; name
the role whenever you switch.

| role | owns | must never |
|---|---|---|
| **Site architect** (Quarto) | `_quarto.yml`, page structure, navigation, GitHub Pages workflow | ship a page that is not reachable from the navbar or sidebar |
| **Front-end engineer** (TypeScript) | every interactive component, the build step that compiles TS into `site/assets/`, the lightbox-free code tabs | add a runtime dependency that is not pinned, or write JS by hand where TS is the rule |
| **Technical writer** (fluent in tidymodels: recipes / parsnip / rsample / yardstick / tune / workflows / workflowsets) | all prose and all code examples | write a sentence about what the library *does* that is not traceable to a file in the repo |
| **Verifier** | the evidence trail: every claim, number and code block mapped to its source file; the check scripts | accept "it should work" for anything that can be run |
| **Designer** | the theme (`theme.scss`, dark variant), typography, the logo mark, figures | copy tidymodels' or Posit's logos, hex stickers, or palette verbatim |

## What the site is for

Explain to two audiences, tidymodels users and Rust/Python ML engineers, what this
library is, what it can do **today**, how far it is from the tidymodels ecosystem, and
how to try it. Its difference from tidymodels is the point of the site and must be
stated on the landing page in one paragraph: it is written in Rust, exposes the same
mental model (recipe → model spec → workflow → resample / tune → metrics → workflow
sets), is verified **differentially against R's own output**, and has a Python wrapper
that calls the same Rust code. The site is for an **ongoing project**; it is not a
product launch, and it is **not affiliated with Posit or the tidymodels team**. Say so
in the footer of every page and on the landing page.

## Honesty rules (the Verifier enforces these before any page is merged)

1. **Every capability claim comes from a repository document**: `docs/FEATURES.md`
   (the feature matrix), `COMPATIBILITY.md` (what maps to which R function, and the
   "not supported" rows), `LIMITATIONS.md` (known limitations and the risk register),
   `VERIFICATION.md` (what was verified, how, and to what tolerance), `ROADMAP.md`
   (phases and open items), `docs/releases/0.1.0.md`, `CHANGELOG.md`. If a sentence on
   the site cannot be pointed at one of these, it does not go on the site.
2. **Every number comes from `BENCHMARKS.md` with its run id**, and the site repeats
   the caveats that stand next to it there (one Windows laptop, R single-threaded,
   disturbed trials labelled, no result transfers to other hardware, forests are *on
   par* with ranger, not faster). No rounded-up "up to 1000×" headlines; quote the
   range and the row count it belongs to.
3. **Every code block is copied from a tested example**, never typed into the page:
   Rust `crates/tidyrustmodels/examples/{penguins_workflow,ames_tuning,ames_workflow_sets}.rs`,
   Python `examples/{penguins_workflow,ames_tuning,ames_workflow_sets}.py`, and the
   README quick starts. Their printed outputs are recorded in VERIFICATION.md (Phase 6)
   and the site shows those recorded outputs, labelled with the date and commit. Quarto
   cannot execute Rust, so the Verifier writes `site/scripts/check_examples.py` which
   fails the build when a code block on the site differs from its source file.
4. **The gap page is a first-class page**, not a footnote: a table with one row per
   tidymodels package (recipes, parsnip, rsample, yardstick, tune, workflows,
   workflowsets, dials, broom, stacks, finetune, probably, censored, spatialsample,
   textrecipes, embed, themis, agua, bonsai, multilevelmod, poissonreg, rules,
   discrim, baguette) and three columns: *covered* (with the number of functions or
   steps, from `docs/FEATURES.md`), *partial* (what exactly is missing, from
   `COMPATIBILITY.md`'s "not supported" rows), *not started*. Where a package is not
   started, say "not started", never "coming soon".
5. **Verification is described as it is**: differential tests against pinned R
   fixtures with per-quantity tolerances that are never silently relaxed; exact
   matches where they are exact (rpart trees, rsample ids, tune selection), measured
   bounds where they are bounds (float32 for xgboost, 4-SE bands over 10 seeds for
   forests, round-off for lm/glm), and the platform note from ADR-0017: the xgboost
   port is bit-exact against R on Windows (where the fixtures were made) and within the
   C runtime's float32 `expf` ulp on Linux and macOS.
6. **Status language**: version is `0.1.0-dev` until a tag exists; the Rust crate and
   the wheels are **not published** to crates.io / PyPI until ROADMAP.md says they are;
   installation instructions are the ones in README.md ("Installing") and nothing else.
7. No testimonials, no "trusted by", no invented roadmap dates. Phase names and open
   items come from ROADMAP.md verbatim.

## Pages (minimum)

- **Home**: the one-paragraph difference; three cards (Rust core, Python wrapper,
  verified against R); one tabbed Rust/Python quick start (from the README quick
  starts, outputs recorded); a status strip ("0.1.0-dev, ongoing project, not
  affiliated with tidymodels"); links to the gap page and the benchmarks page.
- **Why Rust**: what the Rust core buys (one implementation for both languages, memory
  behaviour, the measured speedups with their caveats) and what it costs (no R
  interface, smaller ecosystem, a port is not the original: cite the forest and the
  `expf` findings as examples of the honesty the project keeps).
- **Get started** (Rust / Python tabs): install from README.md, the penguins workflow
  end to end, then one tuning example and one workflow-set example, each with its
  recorded output.
- **What it can do**: the feature matrix rendered from `docs/FEATURES.md` (parse the
  Markdown at build time; do not duplicate it), with a TypeScript filter by package
  and by engine.
- **Gaps against tidymodels**: the table from rule 4, plus the "known limitations"
  list from LIMITATIONS.md and the open ROADMAP items.
- **Verification**: how the differential suite works, the tolerance file, the
  platform note, and the test inventory table from VERIFICATION.md.
- **Benchmarks**: the headline ranges from BENCHMARKS.md with their run ids, one
  TypeScript chart per family built from the CSVs in `benchmarks/results/<run>/`
  (medians only, IQR shown, disturbed configurations marked as in the source), and
  the deviations / disturbance log reproduced, not summarised away.
- **Design decisions**: an index of `docs/adr/*.md` (title, date, status, one-line
  context) linking to the rendered ADRs.
- **Changelog** and **Roadmap**: rendered from the repository files at build time.
- **API reference**: link to the rustdoc output (`cargo doc --workspace --no-deps`)
  published under `/api/rust/`, and a generated page of the Python package's public
  API (`python/tidyrustmodels/__init__.py`, its `__all__`, docstrings) under
  `/api/python/`.

## Stack

- **Quarto** website project (`site/` directory in this repo, `_quarto.yml`, output to
  `site/_site`); `freeze: true`; no R or Python execution at render time except the
  build-time scripts below, which run before `quarto render`.
- **TypeScript** compiled with `esbuild` (pinned) from `site/ts/` into
  `site/assets/*.js`, loaded with `include-in-header`. Components: Rust/Python code
  tabs that remember the viewer's choice (localStorage, wrapped in try/catch), the
  feature-matrix filter, the benchmark charts (a small, pinned charting library or
  plain SVG; no framework), the gap table's package filter. Everything must render
  with JavaScript disabled (the tables and code blocks are static HTML; TS only adds
  filtering and tabs).
- **Build scripts** (`site/scripts/`, Python, run with `-I`): render
  FEATURES.md / CHANGELOG.md / ROADMAP.md / ADR index / Python API page into `site/`
  pages; `check_examples.py` (rule 3); `check_claims.py` that greps the site for
  numbers with `×` or `%` and verifies each appears in BENCHMARKS.md.
- **GitHub Pages** through `.github/workflows/site.yml`: on push to `main` and on
  `workflow_dispatch`; steps: checkout, `quarto-actions/setup-quarto` (pinned
  version), Node (pinned) + `npm ci` for esbuild, Python for the scripts, `cargo doc`
  for the Rust API, the check scripts (fail the build on any mismatch), `quarto
  render`, `actions/upload-pages-artifact`, `actions/deploy-pages`. Set `permissions`
  to the minimum (`pages: write`, `id-token: write`, `contents: read`). The first
  deploy needs the repo's Pages source set to "GitHub Actions"; say so in the PR.
- The site must not break the existing CI (`ci.yml`) and must not add anything to
  the Rust workspace.

## Theme (Designer)

Professional, recognisably in the tidymodels family without copying it. tidymodels'
own site pairs a deep navy with a raspberry accent and generous white space; use that
**mood**, not its assets: a deep navy/ink base (`#1b2a41`-ish) for the navbar and
headings, a teal-grey for secondary text, **one** warm accent in the raspberry family
for links and the active tab, and Rust's terracotta (`#b7410e`-ish) **only** for the
"Rust" badge and the Rust tab so the two languages are distinguishable at a glance.
Light and dark variants (`theme.scss` + `theme-dark.scss`, Quarto's `theme: light/dark`
toggle). Type: a humanist sans for text (Source Sans 3 or IBM Plex Sans, self-hosted,
no Google Fonts call at runtime), a monospaced face with clear `l/1/I` for code (JetBrains
Mono or IBM Plex Mono). Code blocks use Quarto's highlighting with a style that reads on
both variants. A simple geometric logo mark (an SVG you draw: e.g. a gear/crab-free
abstraction of the recipe → model → metrics pipeline); no crab, no hex sticker, no
Posit assets. Mobile first: 16px gutters, no horizontal scroll, tables collapse to
scrollable regions with a visible edge.

## Process

1. Read, in this order: README.md, docs/FEATURES.md, COMPATIBILITY.md,
   LIMITATIONS.md, VERIFICATION.md, BENCHMARKS.md (at least the headline tables and
   the deviations), ROADMAP.md, docs/releases/0.1.0.md, the three examples in both
   languages, `docs/adr/0017-*.md`. Write `site/NOTES.md` listing every claim you
   intend to make with its source line before writing a page.
2. Site architect: `_quarto.yml` and the page skeleton; Front-end: the TS build and
   the three components with a Playwright-free smoke test (`node` runs the compiled
   bundle against a jsdom page); Writer: pages in the order Home, Get started, Gaps,
   What it can do, Verification, Benchmarks, Why Rust, the rest; Verifier: the two
   check scripts and a review of every page against `site/NOTES.md`; Designer: theme
   after the first three pages exist, then a pass over all of them.
3. Render locally (`quarto render site`), run both check scripts, open the output in
   a browser at phone and desktop widths, and fix what you see.
4. Deliver: the `site/` tree, the workflow, a `site/README.md` (how to build, how the
   checks work, where numbers come from), and a short report: pages built, claims
   count and sources, checks passed, what is intentionally missing (and why), the two
   things you would do next. Commit with `git -c user.name="Adam Ezzat"
   -c user.email="adamezzat7@gmail.com"` and the trailer
   `Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>`, on a branch, as a PR.
   Do not change README.md, BENCHMARKS.md or the release notes from this task.

## Out of scope

Publishing to crates.io / PyPI, writing new library features, new benchmarks,
blog posts, analytics or cookie banners (none; the site sets no cookies), comments.
