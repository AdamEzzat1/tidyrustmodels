# ADR-0004: Binary event level is the first level

Date: 2026-10-12 · Status: accepted

## Context
tidymodels/yardstick default `event_level = "first"`: precision, recall, and
`mn_log_loss` treat the first factor level as the event (verified by printing
`yardstick:::yardstick_event_level()` and `yardstick:::precision_binary`, yardstick 1.4.0).
R's `glm` models the probability of the *second* level; parsnip reports `.pred_<level>`
columns for both levels so the user never needs to know.
linfa-logistic reports the probability of the *larger* class under `Ord`
(verified in `linfa-logistic-0.8.1/src/lib.rs`, `predict_probabilities` doc comment).

## Decision
`Outcome::Class` stores levels in a user-visible order; `levels[0]` is the event.
Classification models always return a full probability matrix (one column per level,
ordered as `levels`). Metrics consume that matrix plus the event level. Adapters map
their internal convention to this contract and are tested for it.

## Consequences
- No ambiguity about which probability a metric uses.
- Parity tests compare `.pred_<levels[0]>` from R with column 0 from Rust.
