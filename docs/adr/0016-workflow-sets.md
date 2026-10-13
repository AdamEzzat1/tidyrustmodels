# ADR-0016: Workflow sets

Date: 2026-10-13. Status: accepted.

## Context
workflowsets 1.1.1 turns lists of preprocessors and models into a tibble of
workflows (`workflow_set()`), runs one tuning or resampling function over all of
them on the same resamples (`workflow_map()`), and summarises, ranks and refits
(`collect_metrics()`, `rank_results()`, `fit_best()`). Its semantics were read from
the package sources and probed in R (`target/wfs_probe*.log`, 2026-10-13):

- `cross_objects()` lays the combinations out with the preprocessor list outermost
  and the model list innermost, in the given order (measured: `(zeta, alpha) x (lm,
  tree)` gives `zeta_lm, zeta_tree, alpha_lm, alpha_tree`); `fuse_objects()` pairs
  the lists element-wise when `cross = FALSE`, unless one list has length one.
  Ids are `paste(preproc, model, sep = "_")`; unnamed elements get their class name
  made unique by `vctrs::vec_as_names(repair = "unique")` with `...` replaced by `_`
  (`recipe_1`, `recipe_2`; a single `linear_reg` stays).
- `workflow_map("tune_grid")` on a workflow without `tune()` placeholders falls back
  to `fit_resamples()` with a warning; its single `.config` is `pre0_mod0_post0`
  while tuned candidates carry tune's zero-padded labels (`pre0_mod01_post0` ...).
- `collect_metrics()` binds every result's summary with `wflow_id`, `preproc` and
  `model`; rows are workflow-major, then `.config`, then metric name (tune groups
  by metric, so `rsq` follows `rmse` whatever the set order).
- `rank_results()` takes the ranking metric's mean per `(wflow_id, .config)`,
  negates it for larger-is-better metrics, optionally keeps each workflow's best
  (`slice_min(with_ties = FALSE)`), and ranks with `rank(ties.method = "random")`
  under `withr::with_seed(1)`; all metrics of a candidate are reported, arranged by
  rank, NA ranks last. The default metric is the first of the set.
- `fit_best()` takes the top workflow of `rank_results(select_best = TRUE)`, its
  `select_best()` parameters and tune's `fit_best()` (`finalize_workflow` +
  `fit()`, so workflows' sparse-data decision applies); results must have been
  produced with `save_workflow = TRUE`.
- `tidyr::crossing()` de-duplicates identical preprocessor or model objects.

## Decision
1. A new crate `tidyrustmodels-workflowsets` (re-exported as
   `tidyrustmodels::workflowsets` and in the prelude) with `workflow_set(preproc,
   models, cross)` over `(Option<String>, Recipe)` / `(Option<String>, Box<dyn
   ModelSpec>)` lists, `fix_list_names` (the naming rule), and `WorkflowSet` holding
   `WorkflowSetEntry { id, preproc, model, workflow, grid, result }` in
   workflowsets' order.
2. `WorkflowSet::option_grid(id, grid)` stands in for `option_add(grid = ...)` and
   tune's `tune()` placeholders: `workflow_map(data, folds, metrics, par)` tunes
   entries with a grid (`tune_grid`) and resamples the others (`fit_resamples`),
   storing `WorkflowResult::{Tuned, Resamples}`; the metric set is kept for the
   default ranking metric. Results are always refittable (no `save_workflow`).
3. `collect_metrics`, `rank_results(metric, select_best)`, `best(metric)` (the
   workflow and parameters `fit_best` uses) and `fit_best(data, metric)` follow the
   sources above. `TuneResults::config_labels()` reproduces tune's padded labels.
4. **Documented differences:** ties in `rank_results` keep their order of
   appearance instead of R's seed-1 random ranks (the verification treats tie groups
   as sets and requires the reference tasks to have no tie at the top); identical
   preprocessor or model objects are not de-duplicated; `workflow_map`'s `seed` is
   not modelled (every model here is deterministic given its own seed); `comment_*`,
   `option_*` beyond the grid, `collect_predictions`, `collect_extracts`,
   `autoplot`, `leave_var_out_formulas` and the race / Bayesian / simulated-
   annealing functions are not ported.
5. Verification (Phase 8k): reference tasks carry a `workflow_set` field
   (preprocessors as step lists with optional names, models with optional
   `model_args` and `grid`, `cross`, the ranking metric); the R runner builds the
   set, asserts workflowsets' ids against the naming rule re-derived from
   `fix_list_names`, maps with `tune_grid` on the shared rset, and records ids,
   info, result classes, `collect_metrics`, `rank_results` in three modes (metric,
   metric + `select_best`, default metric) and `fit_best` (workflow, parameters,
   model class, test predictions); the harness compares ids, info, classes and the
   metric rows exactly in order, values within the summary bounds, ranks exactly
   outside tie groups and as sets inside, and the refit's predictions within the
   chosen model's bounds.
6. Python: `workflow_set(preproc, models, cross)` and `WorkflowSet` with `ids`,
   `info`, `option_grid`, `workflow_map`, `collect_metrics`, `rank_results`,
   `best`, `fit_best`, `extract_workflow`, `extract_result`.

## Consequences
- Users get the tidymodels model-comparison loop end to end: cross recipes with
  models, evaluate on one set of resamples, rank, refit the winner.
- `Workflow` is now `Clone` (needed to hand workflows out of a set).
- The crate has no R-side randomness to reproduce except tie ranks, which the
  verification shows are the only place the two implementations can order rows
  differently.
