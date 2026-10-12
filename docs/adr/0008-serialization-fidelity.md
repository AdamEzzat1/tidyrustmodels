# ADR-0008: Serialized fitted objects must round-trip bit-exactly

Date: 2026-10-12 · Status: accepted

## Context
The Python parity test found that a fitted workflow serialized to JSON text and read
back produced probabilities differing by one ulp in 14 of 80 rows. The Rust tests had
only round-tripped through `serde_json::Value`, never through text. Cause: serde_json
parses floats with best-effort precision unless its `float_roundtrip` feature is on;
the recipe's stored means and standard deviations changed in the last bit.

## Decision
- `serde_json` is compiled with `float_roundtrip` in every workspace (root, bindings,
  benchmarks).
- Round-trip tests go through *text*, and assert bitwise equality of predictions
  (`assert_eq!` on `Prediction`, `np.array_equal` in Python), not approximate equality.
- Format version numbers (`format_version`) are checked on deserialization; unknown
  versions are errors, never silently accepted.

## Consequences
- Float parsing is ~2x slower for JSON loads; fitted objects are small, so this is
  irrelevant in practice.
- A serialized model is a faithful artefact: predictions from the deserialized copy
  are identical to the original, which the differential and parity suites rely on.
