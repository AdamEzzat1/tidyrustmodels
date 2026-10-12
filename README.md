# tidyrustmodels

A Rust-native machine learning framework with tidymodels semantics: recipes that prep on
training data and bake anywhere, estimator specifications that are independent of the
backend that fits them, leakage-safe resampling, yardstick-compatible metrics, and
tuning. Python bindings are an optional interface, not a requirement.

Status: **pre-release (0.1.0-dev).** See ROADMAP.md for what exists and what is gated.
Nothing in this repository claims parity with or performance relative to R tidymodels
unless VERIFICATION.md or BENCHMARKS.md records the measurement.

What is verified today (VERIFICATION.md):

- Preprocessing, OLS, metrics and resampling match R tidymodels 1.5.0 to floating-point
  round-off on six reference tasks; logistic regression matches to ~1e-8 relative
  (same objective, different optimizer).
- Both engines (`linfa`, `smartcore`) reproduce exact analytic baselines that are
  independent of R.
- The Python bindings produce bitwise-identical results to the Rust API.

What is measured today (BENCHMARKS.md, one Windows laptop, 5 trials): OLS workflows run
8–240× faster per workflow fit and 5–57× faster end-to-end than R tidymodels with 8–50× less
memory; logistic regression at 1e5 rows is *not* faster than R's IRLS and is slower
than glmnet for tuning. The Python wrapper adds nothing measurable over the Rust binary beyond
a ~0.6 s import; it is 10× faster than R for OLS at 100k rows but only 1.5× for
logistic regression, where tuning is equal to glmnet. Read the deviations list there
before quoting any number.

## Quick start (Rust)

```rust
use tidyrustmodels::prelude::*;

let data = tidyrustmodels::io::read_csv("reference/fixtures/datasets/penguins_sex.csv")?;
let recipe = Recipe::new("sex")
    .step(StepImputeMedian::new(Selector::AllNumericPredictors))
    .step(StepNormalize::new(Selector::AllNumericPredictors))
    .step(StepDummy::new(Selector::AllNominalPredictors));
let wf = Workflow::new(recipe, LogisticRegSpec::new());
let folds = vfold_cv(data.nrows(), 5, None, Seed(7))?;
let cv = wf.fit_resamples(&data, &folds, &MetricSet::classification_defaults(), Parallelism::threads(4))?;
```

See `crates/tidyrustmodels/examples/penguins_workflow.rs` for the full example
(`cargo run --release --example penguins_workflow`).

## Quick start (Python)

```python
import tidyrustmodels as trm
data = trm.from_pandas(df)
wf = trm.Workflow(trm.Recipe("sex").step_normalize("all_numeric_predictors")
                                    .step_dummy("all_nominal_predictors"),
                  trm.LogisticReg())
folds = trm.vfold_cv(data.nrows, 5, seed=7, strata=list(df["sex"]))
cv = wf.fit_resamples(data, folds, ["accuracy", "mn_log_loss"], threads=4)
```

## Building

```
cargo build --workspace --release
cargo test --workspace            # includes the differential suite against committed R fixtures
```

Python bindings (Python ≥ 3.11, inside a virtual environment):

```
python -m venv .venv && .venv/Scripts/activate      # or source .venv/bin/activate
pip install maturin numpy pandas pytest
maturin develop --release
pytest python/tests
```

## Documents

| file | purpose |
|------|---------|
| ARCHITECTURE.md | crate boundaries, contracts, lifecycle |
| DEPENDENCIES.md | verified dependency facts and decisions |
| COMPATIBILITY.md | tidymodels semantic/feature compatibility, tested vs intended |
| VERIFICATION.md | tests executed and numerical comparison records |
| BENCHMARKS.md | methodology and measured results |
| LIMITATIONS.md | unsupported features, findings, risk register |
| ROADMAP.md | milestones and gates |
| docs/adr/ | architecture decision records |
| docs/specs/ | API spec, verification harness design |
| docs/reviews/ | independent review records |
| reference/ | R reference harness, fixtures, independent baselines |
| benchmarks/ | benchmark drivers and results |

## Layout

```
crates/            Rust workspace crates (core, recipes, models, resample, metrics, workflows, tune, facade)
bindings/python/   PyO3 extension crate
python/            Python package and tests
reference/         R scripts, datasets, shared splits, fixtures, tolerances
benchmarks/        Rust/R drivers, orchestrator, results
examples/          Rust and Python examples
```

License: MIT OR Apache-2.0.
