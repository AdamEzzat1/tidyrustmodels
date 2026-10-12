# Architecture

Status: Phase 0 decision (2026-10-12). Revised as milestones land; every structural change
is recorded as an ADR under `docs/adr/`.

## Goals that shape the design

1. Rust-native execution with no Python runtime dependency.
2. tidymodels *semantics* (prep/bake separation, leakage-safe resampling, explicit
   outcome/predictor roles, event-level conventions) rather than its syntax.
3. Backend independence: estimator specifications describe *what* to fit; adapters decide
   *how*, and the public API never leaks a backend's types.
4. Everything reproducible from a seed plus a recorded configuration.

## Workspace layout

```
crates/
  tidyrustmodels-core       data contracts, schemas, errors, RNG policy, traits
  tidyrustmodels-recipes    preprocessing steps (prep on training data only, then bake)
  tidyrustmodels-models     estimator specs + backend adapters (linfa, smartcore)
  tidyrustmodels-resample   splits, k-fold, stratified k-fold, fixture-loaded folds
  tidyrustmodels-metrics    yardstick-compatible metric definitions
  tidyrustmodels-workflows  recipe + model composition; fit_resamples
  tidyrustmodels-tune       grid / random search over resamples
  tidyrustmodels            facade crate re-exporting a curated public API
bindings/python             PyO3 crate (`tidyrustmodels._native`), maturin packaged
python/tidyrustmodels       pure-Python wrapper package
reference/tidymodels-r      R scripts producing pinned reference fixtures
reference/fixtures          datasets, shared split/fold ids, R outputs (JSON/CSV)
benchmarks                  Rust and R benchmark drivers plus the Python runner
```

Deviation from the proposed layout: there is no separate `tidyrustmodels-data` crate.
The data contract (`Table`, `Column`, `DesignMatrix`, `Outcome`) is the core's
reason to exist; splitting it out would create a crate with one consumer and a cyclic
temptation (core traits need the data types, data types need core errors).
See ADR-0001.

Dependency direction (acyclic, enforced by Cargo):

```
core <- recipes  <- workflows <- tune
core <- models   <- workflows
core <- resample <- workflows
core <- metrics  <- workflows
facade depends on all; bindings depend on the facade only.
```

## Core contracts

### Data

- `Table`: named, typed columns of equal length. Column kinds in v0.1:
  `Numeric` (`Vec<f64>` + optional validity mask) and `Categorical`
  (`Vec<u32>` codes + `Vec<String>` levels + optional validity mask).
  Missing values are explicit through the validity mask, never through sentinel values.
  `NaN` in a numeric column is an *error* at ingestion unless the caller opts in;
  the framework never silently treats `NaN` as missing.
- `Schema`: ordered column names, kinds, and roles (`Predictor`, `Outcome`, `Id`, `Ignored`).
  Every fitted recipe and fitted model carries the schema it was fitted on and
  rejects incompatible inputs at `bake`/`predict` time.
- `DesignMatrix`: dense column-major `f64` with feature names. Column-major because
  preprocessing is column-wise, R is column-major, and it converts to `ndarray`'s
  Fortran-order layout without a transpose.
- `Outcome`: `Numeric(Vec<f64>)` or `Class { codes: Vec<u32>, levels: Vec<String> }`.
  For binary classification the **event level is `levels[0]`** (tidymodels
  `event_level = "first"`). Adapters that model a different class internally must map.

### Lifecycle traits

```
Step        (unfitted preprocessing)  --prep(training)-->  FittedStep  --bake(any)-->
ModelSpec   (unfitted estimator)      --fit(design,y)-->   FittedModel --predict-->
Workflow    = Recipe + ModelSpec      --fit(training)-->   FittedWorkflow
```

Fitted objects are immutable. `prep` sees training rows only. `bake` and `predict`
never update state. Resampling re-runs `prep` inside every fold (leakage safety).

### Capabilities

`ModelSpec::capabilities()` returns a declared set (`Regression`, `BinaryClassification`,
`ClassProbabilities`, `Coefficients`, `Serializable`). Workflows check capabilities
before running rather than discovering failures at predict time. Backend adapters
declare only what they genuinely implement; e.g. the smartcore logistic adapter
derives probabilities itself and declares that fact in its `BackendInfo`.

### Backend adapters

`tidyrustmodels-models` exposes `LinearRegSpec` and `LogisticRegSpec` with an
`engine` field. Engines are feature-gated adapters:

| engine      | crate (version)                          | notes                                                  |
|-------------|------------------------------------------|--------------------------------------------------------|
| `linfa`     | linfa-linear 0.8.1, linfa-logistic 0.8.1 | default; pure-Rust path (no LAPACK)                    |
| `smartcore` | smartcore 0.6.16                         | second backend; no native `predict_proba` for logistic |

ndarray is **not** part of any public interface: linfa pins ndarray 0.16 and smartcore
pins 0.17, which are distinct types. Adapters copy the design matrix into the backend's
array type (O(n·p), measured in benchmarks; zero-copy deferred to Phase 5 and only with
evidence).

### Errors

One `Error` enum in core (`thiserror`), with variants for schema mismatch, invalid
value (non-finite), unsupported capability, backend failure (boxed source), and
serialization. No library-facing `panic!`/`unwrap` on user input; adapters validate
shapes *before* calling backends that assert internally (linfa-linear asserts on shape).

### Randomness and parallelism

- All randomness flows from an explicit `u64` seed through `rand_chacha::ChaCha8Rng`.
  No thread-local or global RNG in library code.
- Parallelism is opt-in via a `Parallelism { threads: Option<usize> }` config passed to
  `fit_resamples`/`tune_grid`, executed with a scoped Rayon pool. Serial results and
  parallel results must be bit-identical (tested).

### Precision policy

v0.1 computes in `f64` only. `f32` support is a declared non-goal until a measured
need appears.

### Serialization

Fitted recipes, models, and workflows serialize through `serde` to a versioned JSON
envelope `{ "format_version": 1, ... }`. Deserialization validates the version and the
schema. Backend-internal fitted state is stored as plain coefficients so a serialized
model does not depend on the backend crate's serde layout.

## Python layer

`bindings/python` wraps the facade crate. NumPy arrays are read through
`PyReadonlyArray2<f64>`; C-contiguous input is copied once into the column-major
`DesignMatrix` (documented, measured). pandas/Arrow input converts column-wise.
Rust errors map to a small exception hierarchy rooted at `TidyRustModelsError`.
Python never participates in the Rust test suite; parity between the two interfaces
is tested from Python.

## What is deliberately not here

- No custom linear algebra engine. Backends own their solvers.
- No global registries. Engines are enum variants, selected explicitly.
- No DataFrame library in core. Polars/Arrow interop is an optional feature of the
  facade and bindings (ADR-0003).
