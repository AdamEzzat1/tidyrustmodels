# ADR-0014: Tuning selection rules, `finalize_workflow`, `last_fit` and `fit_best`

Date: 2026-10-13. Status: accepted.

## Context
`tune_grid` existed, but the steps tidymodels users take after it did not:
choosing a candidate (`show_best`, `select_best`, `select_by_one_std_err`,
`select_by_pct_loss`), writing it into the workflow (`finalize_workflow`), and the
final fit on the training split with an assessment on the test split (`last_fit`)
or on every row (`fit_best`). tune 2.1.0's rules have details that matter:

- `show_best()` sorts `collect_metrics()` by the mean with `dplyr::arrange()`, a
  **stable** sort, so ties keep `.config` order; candidates whose metric is `NA`
  on every resample sort last. `select_best()` is its first row. The `.config`
  labels (`pre0_mod3_post0`) are **not** grid order: `make_config_labs()` takes the
  distinct model-parameter rows, arranges them by the parameters in parsnip's
  argument order (`tune_args()` order, e.g. `cost_complexity, tree_depth, min_n`;
  `trees` before `learn_rate`) and numbers them, so that order decides every
  tie-break.
- For `boost_tree`, `trees` is a sub-model parameter: tune fits the largest
  `trees` once per setting of the other parameters and predicts the smaller
  candidates from it (`multi_predict()`). parsnip 1.5.0's `xgb_by_tree()` passes
  `iterationrange = c(1, trees + 1)` to xgboost ≥ 2.0, whose range is 1-based and
  **inclusive at both ends**, so every non-maximal candidate is scored with
  `trees + 1` trees (verified: tune's numbers equal this crate's `trees + 1`
  booster to round-off, Phase 8i).
- `select_by_one_std_err(x, ..., metric)` takes the **first** best (`which.max` /
  `which.min`), bounds at `best ∓ std_err[best]`, keeps the candidates within the
  bound, arranges them by the `...` expressions (parameter names, possibly inside
  `desc()`, ordering from the simplest model) and takes the first. With one
  resample the standard error is `NA`, the filter keeps nothing, and tune returns
  an empty tibble.
- `select_by_pct_loss(x, ..., metric, limit = 2)` computes
  `.loss = |(|mean| − |best|) / best| · 100`, arranges by the `...`, finds the
  best's position (`which(.loss == 0)[1]`) and takes the first candidate **at or
  before it** whose loss is below `limit`.
- `finalize_workflow()` calls `update()` on the model specification with the
  selected parameters (`tune()` placeholders are replaced); recipe parameters are
  finalized the same way.
- `last_fit()` wraps the split in `manual_rset(list(split), "train/test split")`
  and runs `resample_workflow()`: the fit goes through tune's `.fit_pre` /
  `.fit_model` path, i.e. **without** workflows' sparse-data toggle and **with**
  frequency case weights in the metrics; the fitted workflow is kept in
  `.workflow`, the assessment predictions in `.predictions` (with `.row`).
- `fit_best()` finalizes with `select_best()` and calls `parsnip::fit()` on
  `x$splits[[1]]$data` (every row), so the toggle **does** apply there.

## Decision
1. `TuneResults::show_best` returns `BestRow { index, config, params, mean, std_err,
   n }` with tune's ordering (candidates in `.config` order, then a stable sort by
   the mean, undefined means last); `TuneResults::configs()` numbers the candidates
   as `make_config_labs()` does, from `ModelSpec::tunable()`, which therefore lists
   parameters in parsnip's argument order. `select_best` returns a `Selection {
   index, config, params }` where `index` is the grid position. `select_by_one_std_err(metric, &[SortKey])` and
   `select_by_pct_loss(metric, &[SortKey], limit)` implement the rules above with
   `SortKey::{asc, desc, parse("desc(param)")}` standing for the `...`; the
   one-resample case is an error rather than an empty result.
2. `Workflow::finalize(&params)` is `finalize_workflow()` for model parameters
   (`ModelSpec::with_param`); `tune_grid` builds its candidates through it.
3. `Workflow::last_fit(data, &Split, metrics) -> LastFit { workflow, predictions,
   metrics }` runs the fold path (`fit_fold_full`), so a `last_fit` model can
   differ from `fit()` on a sparse design exactly as in tune; `fit_best(results,
   workflow, data, metric)` finalizes with `select_best` and calls `Workflow::fit`.
4. Verification (VERIFICATION.md Phase 8i): reference tasks carry an explicit
   `tuning.grid` shared verbatim by R and Rust (no `dials` grid generation on either
   side, so no round-off in the candidate values), run on the task's shared rset;
   R records every candidate's per-resample and summary metrics keyed by grid row
   (joined to tune's `.config`), the `show_best` order and the three selections as
   grid-row indices, and `last_fit` / `fit_best` metrics, predictions and models.
   Rust must reproduce the orders and selections exactly and the fits to the model's
   usual bounds.
5. `tune_grid` evaluates every candidate with its own fit: the sub-model shortcut
   is not reproduced, and neither is parsnip's off-by-one. A `trees` candidate
   therefore means exactly that many trees here, which differs from tune's number
   for every non-maximal `trees` candidate (COMPATIBILITY.md); the reference task
   `synth_reg_xgb_trees` records both R numbers so the difference is measured, not
   assumed.
6. The framework has no integer column type, and recipes casts imputed statistics
   to the column's type (`cast()`: `as.integer(round(x))` on integer columns). The
   R fixtures are generated with whole-number columns read as doubles, and the
   rounding is recorded as a difference for users with integer columns in R.
7. Not ported: recipe-parameter tuning, `tune_bayes`, `tune_race_*`,
   `control_grid(save_pred = TRUE)` prediction collection across resamples,
   `.config` strings themselves (the grid index stands for them), the `zero`
   metric direction, survival `eval_time`, and `last_fit(add_validation_set =)`.

## Consequences
- `show_best`'s return type changed (tuple → `BestRow`); the example and the
  Python `tune_grid` result changed with it: Python now returns a `TuneResults`
  object (indexable like the previous list) with `show_best`, `select_best`,
  `select_by_one_std_err`, `select_by_pct_loss` and `fit_best`;
  `Workflow.finalize(params)` and `Workflow.last_fit(data, train_rows, test_rows,
  metrics)` are new.
- The reference task schema gains `tuning` (`reference/tasks.json` notes).
