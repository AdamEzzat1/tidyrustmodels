# ADR-0012: Case weights

Date: 2026-10-13. Status: accepted.

## Context
tidymodels carries case weights through the whole pipeline, but not uniformly:
`hardhat` distinguishes **frequency weights** (non-negative integers that stand for
repeated rows) from **importance weights** (non-negative doubles), and every package
decides for itself which kind it honours.

- **recipes 1.3.2** gives a `hardhat` weights column the role `case_weights`
  automatically; the column is never a predictor and is retained by `bake()`. The
  unsupervised steps (`step_normalize`, `step_impute_mean`/`_median`, `step_nzv`,
  `step_corr`, `step_other`, `step_filter_missing`, …) use the weights only when they
  are frequency weights (`are_weights_used(unsupervised = TRUE)`); importance weights
  are ignored by every step. The weighted statistics are `stats::cov.wt` with the
  weights normalised to sum one (centre, unbiased variance, correlation on the
  listwise-complete rows), `weighted_median_impl` (first sorted value whose
  cumulative weight share exceeds one half) and `hardhat::weighted_table` (weighted
  level counts, divided by the weight of the non-missing rows).
- **parsnip 1.3.x** declares per engine whether weights are allowed: `lm`, `glm`,
  `rpart`, `ranger` and `xgboost` take them (frequency weights cast to integer,
  importance to double); `nnet` for `multinom_reg` does not and fails with
  "Case weights are not enabled by the underlying model implementation.".
  The engines then do different things: `lm.wfit` drops zero-weight rows and scales
  the rest by `sqrt(w)`; `glm.fit` runs IRLS with prior weights (`good <- weights > 0`,
  `mustart = (weights * y + 0.5) / (weights + 1)`, weighted deviance, a warning for
  non-integer `weights * y`); rpart weights every node statistic and split while
  `minsplit`/`minbucket` keep counting observations; ranger uses `case.weights` as
  *sampling* weights for each tree's bootstrap (`std::discrete_distribution`), and
  treats weights that are all equal as no weights; xgboost scales each row's
  gradient pair, weights the quantile sketch and the base score (`WeightedSampleMean`,
  weighted class histogram).
- **workflows 1.3.0** `add_case_weights(col)` validates the column class and hands the
  vector to parsnip; prediction data need not carry the column.
- **tune 2.1.0** passes case weights to yardstick only when they are frequency weights
  (`.use_case_weights_with_yardstick()`), and yardstick then computes every metric with
  `case_weights =`: weighted means and sums, a weighted confusion table, `cov.wt`
  correlation for `rsq`, threshold curves on the non-zero-weight rows with cumulative
  weights, softmax-normalised Brier weights, and `roc_auc` switching from `hand_till`
  to `macro` for multiclass outcomes (`hand_till` with weights is an error).

Two of these facts are easy to get wrong from memory and were established from the
installed sources (reference/tidymodels-r probes): that recipe steps ignore importance
weights, and that resampling metrics are weighted for frequency weights only.

## Decision
1. **Types.** `core::CaseWeights { kind: WeightKind, values }` with
   `WeightKind::{Frequency, Importance}`; the frequency constructor rejects
   non-integers, both reject negative or non-finite values, zeros are allowed.
   `used_by_steps()` and `used_by_metrics()` encode recipes' and tune's rule
   (frequency only). `Role::CaseWeights(kind)` is a schema role; `Capability::CaseWeights`
   advertises engines that take them.
2. **Models.** `ModelSpec::fit_weighted(x, y, &CaseWeights)` defaults to parsnip's
   error; `fit_maybe_weighted` dispatches. The native engine implements it for
   `linear_reg` (`lm.wfit`), `logistic_reg` (`glm.fit` prior weights, incl. the
   non-integer warning), `decision_tree` (rpart `weights`), `rand_forest` (ranger
   `case.weights` as weighted bootstrap sampling, equal weights → plain bootstrap) and
   `boost_tree` (xgboost DMatrix `weight`: gradients, sketch, base score). The linfa and
   smartcore engines refuse with `Unsupported`; `multinom_reg` refuses on every engine
   with parsnip's message. Each weighted fit records `case weights (<kind>)` in its
   diagnostics note.
3. **Recipes.** `Recipe::prep_with_weights(training, Some((column, kind)))` gives the
   column the `case_weights` role (`Table::schema_with_roles`); steps read it through
   `steps::step_weights`, which returns the values only for frequency weights. The
   weighted statistics are the ones named above; `bake` tolerates an absent weights
   column; `bake_matrix` never includes it in the design.
4. **Workflows and tuning.** `Workflow::case_weights(column, kind)` is
   `add_case_weights()`. `fit` preps with the role and fits the model with the weights;
   `fit_fold` (used by `fit_resamples` and `tune_grid`) evaluates the metrics with the
   assessment rows' weights when they are frequency weights and unweighted otherwise.
   `tune_grid` now also skips the sparse-data toggle, as tune does (it previously
   called `fit`).
5. **Metrics.** Every metric has a weighted form (`*_weighted`, `Metric::compute_weighted`,
   `MetricSet::evaluate_weighted`) following yardstick's arithmetic, including the
   `hand_till` + weights error and the macro default under weights. Python:
   `class_metrics(..., case_weights=)`, `Workflow(..., case_weights=, weight_kind=)`.
6. **Verification.** Eleven reference tasks with a `case_weights` field (frequency and
   importance; lm, glm, rpart, ranger, xgboost incl. multiclass; a filter task that
   exercises the weighted nzv/corr/other/median statistics) on datasets that gained
   `w_freq` (0–4, with zeros) and `w_imp` (0.2–2.5, with zeros) columns. R fixtures carry
   the weighted fit, weighted resampling metrics for frequency weights, and the test
   metrics both unweighted and with `case_weights =`. The suite also checks that the
   non-native engines refuse the weights.

## Consequences
- A weighted workflow is a different pipeline for the two kinds: frequency weights
  change the recipe statistics, the model and the resampling metrics; importance
  weights change the model only. Users who expect importance weights to affect
  `step_normalize` get tidymodels' behaviour, not their expectation.
- rpart-style trees with frequency weights are *not* the trees of the replicated data
  set: `minsplit`/`minbucket` count observations, not weight. lm and glm with
  frequency weights do reproduce the replicated data to solver tolerance (tested).
- recipes' `step_nzv()` with frequency weights errors on a column that has missing
  values (it subsets the data but not the weights before `hardhat::weighted_table`);
  this crate subsets both and proceeds. The reference task orders its steps so that R
  can run (impute before nzv); the divergence is recorded in COMPATIBILITY.md.
- yardstick's weighted `rsq` uses `cov.wt`'s unbiased correlation, so the weighted
  and unweighted `rsq` of equal weights differ only in round-off, but the weighted
  form is undefined with fewer than two non-zero-weight rows.
- Forests with weights remain statistically verified only (the sampler is the
  framework's), as in Amendment A of ADR-0011.
- `cov.wt(cor = TRUE)` builds its matrix elementwise (`Is * cov * rep(Is, each = p)`),
  so R's weighted correlation matrix can be asymmetric in the last bit and its diagonal
  is one only up to round-off; `step_corr`'s tie-break between columns with equal mean
  absolute correlation then depends on round-off. The port mirrors each pair and sets
  the diagonal to exactly one, which resolves exact ties as recipes does (drop the
  earlier column); a tie that R resolves by round-off is not reproducible.
