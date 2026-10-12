# Verification records

Each milestone appends a record. Records state what was executed, with the command
and the outcome. Nothing in this file is inferred.

## Phase 0 (2026-10-12)

No framework code exists yet, so no framework tests were run. Executed in Phase 0:

- `cargo info` / crates.io API queries for all candidate crates (see DEPENDENCIES.md).
- Source inspection of linfa-linear 0.8.1, linfa-logistic 0.8.1, smartcore 0.6.16,
  pyo3 0.29.3, numpy 0.29.0 from the local Cargo registry.
- `Rscript` probes printing the bodies of `yardstick:::rsq_impl`, `rsq_trad_impl`,
  `rmse_impl`, `mae_impl`, `precision_binary`, `recall_binary`, `mn_log_loss_binary`,
  `yardstick_cor`, `recipes:::prep.step_normalize`, `recipes:::variances`,
  `recipes:::averages`, `recipes:::medians`, `recipes::dummy_names`,
  `rsample:::vfold_splits`, `rsample:::add_vfolds`, `rsample:::make_strata`,
  `rsample:::mc_splits`, and `glm.control()` defaults, on tidymodels 1.5.0 / R 4.4.0.

## Test inventory (to be maintained)

| suite | location | runs in CI | needs R |
|-------|----------|------------|---------|
| unit | each crate `src/**/tests` | yes | no |
| property | each crate `tests/prop_*.rs` | yes | no |
| analytic baselines | `tests/baselines/` | yes | no |
| differential vs pinned R fixtures | `tests/differential/` | yes | no (fixtures committed) |
| fixture regeneration | `reference/tidymodels-r/` | scheduled job only | yes |
| Python parity | `python/tests/` | yes (bindings job) | no |
| benchmarks | `benchmarks/` | dedicated machine / manual | yes |
