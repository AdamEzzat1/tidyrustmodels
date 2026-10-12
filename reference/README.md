# Reference harness

Produces the pinned R tidymodels fixtures that the Rust differential tests consume
(docs/specs/verification-design.md). Run from the repository root, in this order:

```
python -I reference/fixtures/datasets/make_synthetic.py      # synthetic CSVs
Rscript reference/tidymodels-r/export_datasets.R              # iris / penguins / ames CSVs
python -I reference/fixtures/generate_splits.py               # shared split + fold ids, checksums
Rscript reference/tidymodels-r/run_reference.R                # fixtures in reference/fixtures/r/
cargo test -p tidyrustmodels --test differential              # Rust vs fixtures
```

`reference/tasks.json` is the single source of truth for which dataset, outcome,
predictors, steps and metrics every task uses; R, Python and Rust all read it.

Layout:

```
tasks.json                  task manifest
tidymodels-r/               R scripts (export, reference run)
independent/                baselines independent of R and Rust (exact OLS, ...)
fixtures/datasets/          CSVs + checksums.json + provenance_r.json
fixtures/splits/            <task>.json with 0-based train/test ids and fold_of_row
fixtures/r/                 <task>.json reference outputs, metrics_fixed.json
fixtures/tolerances.toml    per-level / per-task tolerances with justifications
```

Fixtures are committed so CI never needs R. Regeneration is a reviewed change: the
diff of the fixture files is the evidence that the reference moved.
