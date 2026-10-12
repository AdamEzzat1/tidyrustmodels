# ADR-0003: No Polars/Arrow dependency in core

Date: 2026-10-12 · Status: accepted

## Context
Polars (Rust 0.55.2) and arrow-rs (60.0.0) are large, release breaking majors every
one to two months, and their Python bridges (pyo3-polars 0.28, pyo3-arrow 0.19) must be
version-matched on both sides. The framework needs typed columns with validity masks,
not a query engine.

## Decision
`core::Table` is a minimal owned columnar container. Arrow and Polars interop live
behind optional features of the facade/bindings and convert column-wise.

## Consequences
- Fast clean builds, stable core API.
- Users with large Polars frames pay one conversion; the cost is measured in benchmarks
  ("data conversion overhead" metric). If it dominates, revisit with evidence.
