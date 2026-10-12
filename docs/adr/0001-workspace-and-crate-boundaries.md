# ADR-0001: Workspace and crate boundaries

Date: 2026-10-12 · Status: accepted

## Context
The master prompt proposes nine crates. Crate boundaries are cheap to add and expensive
to remove; each one adds a semver surface and a build-graph node.

## Decision
Start with seven library crates plus a facade (`core`, `recipes`, `models`, `resample`,
`metrics`, `workflows`, `tune`, `tidyrustmodels`). Fold the proposed `-data` crate into
`core` because the trait contracts and the data types are mutually dependent.

## Consequences
- `core` is the only crate every other crate depends on; it must stay small and stable.
- If `core`'s data types grow (Arrow-backed columns, lazy tables), revisit splitting.
