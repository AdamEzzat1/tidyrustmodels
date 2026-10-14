# Examples

| file | language | run |
|------|----------|-----|
| `../crates/tidyrustmodels/examples/penguins_workflow.rs` | Rust | `cargo run --release --example penguins_workflow` |
| `penguins_workflow.py` | Python | `python examples/penguins_workflow.py` (after `maturin develop --release`) |
| `../crates/tidyrustmodels/examples/ames_tuning.rs` | Rust | `cargo run --release --example ames_tuning` |
| `ames_tuning.py` | Python | `python examples/ames_tuning.py` |
| `../crates/tidyrustmodels/examples/ames_workflow_sets.rs` | Rust | `cargo run --release --example ames_workflow_sets` |
| `ames_workflow_sets.py` | Python | `python examples/ames_workflow_sets.py` |

`penguins_workflow` covers the same workflow in both languages: stratified split,
recipe (impute, normalize, dummy), logistic regression, held-out metrics, 5-fold
cross-validation with the recipe re-prepped inside every fold, penalty tuning on a log
grid, tune's selection rules, and a JSON round trip of the fitted workflow. It reads
the CSV exported from R's `modeldata::penguins` (`reference/fixtures/datasets/penguins_sex.csv`).

`ames_tuning` tunes an rpart-style decision tree on `modeldata::ames` over an explicit
`cost_complexity` x `min_n` grid with repeated cross-validation, applies `select_best`
and the one-standard-error rule, finishes with `finalize` + `last_fit` on the initial
split and `fit_best` on every row (ADR-0014).

`ames_workflow_sets` crosses two recipes with a linear model, a tuned tree and a
boosted tree, maps them over one set of folds, ranks the workflows and refits the
winner (ADR-0016).
