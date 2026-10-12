# tidyrustmodels

A Rust-native machine learning framework with tidymodels semantics: recipes that prep on
training data and bake anywhere, estimator specifications that are independent of the
backend that fits them, leakage-safe resampling, yardstick-compatible metrics, and
tuning. Python bindings are an optional interface, not a requirement.

Status: **pre-release, under construction.** See ROADMAP.md for what exists and what
is gated. Nothing in this repository claims parity with or performance relative to R
tidymodels unless VERIFICATION.md or BENCHMARKS.md records the measurement.

## Documents

| file | purpose |
|------|---------|
| ARCHITECTURE.md | crate boundaries, contracts, lifecycle |
| DEPENDENCIES.md | verified dependency facts and decisions |
| COMPATIBILITY.md | tidymodels semantic/feature compatibility |
| VERIFICATION.md | tests executed and numerical comparison records |
| BENCHMARKS.md | methodology and measured results |
| LIMITATIONS.md | unsupported features, risk register |
| ROADMAP.md | milestones and gates |
| docs/adr/ | architecture decision records |
| docs/specs/ | API spec, verification harness design |

## Layout

```
crates/            Rust workspace crates
bindings/python/   PyO3 extension crate
python/            Python package
reference/         R reference scripts, fixtures, independent baselines
benchmarks/        benchmark drivers and runner
```

## Building

```
cargo build --workspace --release
cargo test --workspace
```

Python bindings (requires maturin):

```
cd bindings/python && maturin develop --release
```

License: MIT OR Apache-2.0.
