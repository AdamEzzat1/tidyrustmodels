# ADR-0002: ndarray is an adapter detail, not a public type

Date: 2026-10-12 · Status: accepted

## Context
Verified from crates.io metadata on 2026-10-12: linfa 0.8.1 depends on ndarray ^0.16,
smartcore 0.6.16 (`ndarray-bindings`) on ^0.17, numpy 0.29 accepts 0.15 to 0.17, polars
0.55 on ^0.17. Distinct ndarray major versions are distinct Rust types.

## Decision
Public types are `DesignMatrix` (column-major `Vec<f64>` + shape + names) and plain
slices. Each backend adapter converts to its own ndarray version inside the models
crate. The core crate has no ndarray dependency.

## Consequences
- One O(n·p) copy per backend fit/predict call. Measured in Phase 5; a zero-copy
  path (`ArrayView2::from_shape_ptr`, unsafe) is only justified by benchmark evidence.
- Upgrading a backend's ndarray version touches one adapter module.
