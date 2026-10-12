# Examples

| file | language | run |
|------|----------|-----|
| `../crates/tidyrustmodels/examples/penguins_workflow.rs` | Rust | `cargo run --release --example penguins_workflow` |
| `penguins_workflow.py` | Python | `python examples/penguins_workflow.py` (after `maturin develop --release`) |

Both examples cover the same workflow: stratified split, recipe (impute, normalize,
dummy), logistic regression, held-out metrics, 5-fold cross-validation with the recipe
re-prepped inside every fold, penalty tuning on a log grid, and a JSON round trip of
the fitted workflow. They read the CSV exported from R's `modeldata::penguins`
(`reference/fixtures/datasets/penguins_sex.csv`).
