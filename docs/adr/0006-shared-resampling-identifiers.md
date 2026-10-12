# ADR-0006: Resampling identifiers are shared through fixtures, not seeds

Date: 2026-10-12 · Status: accepted

## Context
R's `sample()` (Mersenne-Twister with rejection sampling) and Rust's ChaCha streams
cannot be made to agree. rsample's unstratified `vfold_cv` draws
`sample(rep(1:v, length.out = n))`; stratified folds do the same within each stratum
(verified by printing `rsample:::vfold_splits` and `rsample:::add_vfolds`, rsample 1.3.2).

## Decision
A Python generator writes `reference/fixtures/generated/<dataset>/<scheme>.json`
containing 0-based row ids per fold / split. R loads them through
`rsample::make_splits` + `rsample::manual_rset`; Rust loads them through
`resample::Folds::from_assignments`. Rust's own `vfold_cv` reproduces rsample's
*algorithm* (balanced assignment, per-stratum when stratified) from a ChaCha seed and is
tested for the structural invariants, not for identical membership.

## Consequences
Parity tests isolate modeling differences from RNG differences.
