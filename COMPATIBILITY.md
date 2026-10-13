# tidymodels compatibility

Status column values: **tested** = covered by the differential suite against pinned R
fixtures (VERIFICATION.md Phase 4) or by analytic baselines; **intended** = implemented
to the documented semantics but not yet fixture-tested; **difference** = deliberate
deviation; **unsupported**.

Reference: R 4.4.0, tidymodels 1.5.0, parsnip 1.5.0, recipes 1.3.2, rsample 1.3.2,
yardstick 1.4.0, tune 2.1.0.

## Semantic contracts

| tidymodels feature | tidyrustmodels | semantic agreement | status |
|--------------------|----------------|--------------------|--------|
| `recipe()` + `prep()` + `bake()` | `Recipe::prep` / `FittedRecipe::bake` | prep on training only; bake never refits | tested (L1, L4) |
| `step_normalize()` | `StepNormalize` | mean via `colMeans(na.rm=TRUE)`, sd via `stats::var` (n-1), zero-variance sd → 1 | tested (L1, 1e-15) |
| `step_impute_mean()` | `StepImputeMean` | mean of non-missing training values (`trim = 0`) | intended |
| `step_impute_median()` | `StepImputeMedian` | `stats::median` (type 7) | tested (L1, exact) |
| `step_dummy()` | `StepDummy` | `one_hot = FALSE` drops first level; names `var_level` (`make.names`); indicator columns appended after remaining columns | tested (L1, names and order exact) |
| `step_dummy()` with unseen level at bake | error | recipes: warning + NA; use `StepNovel` before `StepDummy` to map unseen levels to `new`, as recipes documents | **difference** |
| `step_zv()` | `StepZv` | remove columns with < 2 distinct non-missing values (`one_unique`) | tested (L1, removals exact) |
| `step_nzv(freq_cut, unique_cut)` | `StepNzv` | recipes' `nzv`: most/second-most frequent ratio > `freq_cut` and % distinct (missing counted in the denominator) ≤ `unique_cut`, or single value / all missing | tested (L1, removals exact) |
| `step_corr(threshold)` | `StepCorr` | `corr_filter` (caret `findCorrelation(exact = FALSE)` port): Pearson, pairwise-complete, mean |cor| includes the diagonal, dense rank, drop the higher-ranked member of each pair above the cutoff; NA handling as recipes (warnings become `FittedCorr::notes`) | tested (L1, removals exact) |
| `step_other(threshold, other)` | `StepOther` | `keep_levels`: proportion of non-missing rows (`< 1`) or count (`>= 1`), strict `<`; all-rare keeps the modal level; no collapse leaves the column untouched; retained levels in original order; `other` last | tested (L1, retained levels exact on g/h and on Ames `Neighborhood` 28 → 8 + other) |
| `step_novel(new_level)` | `StepNovel` | adds the level; unseen labels → `new`; missing stays missing; prep errors if the label exists | tested (L1, columns and level exact; zero-count `new` dummy aliased like R's NA) |
| `step_unknown(new_level)` | `StepUnknown` | missing → `unknown`; unseen labels → missing (recipes warns; here silent) | tested (L1 exact); **difference**: no warning for unseen labels |
| `step_dummy()` with missing categorical value | missing indicator values (error at model time) | recipes: NA row | intended |
| `linear_reg() %>% set_engine("lm")` | `LinearRegSpec` (linfa / smartcore) | OLS; same objective, different solver (SVD / least squares) | tested (L3, 2.6e-13 rel) |
| `linear_reg() %>% set_engine("lm")` | `LinearRegSpec` engine `native` | **same algorithm**: Householder QR with `dqrdc2` pivoting, `tol = 1e-7` | tested (L3, 3.3e-13 rel; ADR-0009) |
| `lm` rank deficiency (`NA` coefficient) | aliased column dropped, coefficient 0, listed in `FitDiagnostics::aliased` | identical predictions; coefficient reported as 0 not `NA` | tested on one Ames fold (ADR-0007); **difference** in reporting |
| `lm` rank deficiency, general collinearity | engine `native`: the column `dqrdc2` pivots out (the *later* collinear column) is aliased, as R's `NA` | identical choice of aliased column and identical predictions | tested (unit, `tests/collinear.rs`, against R 4.4.0 output) |
| `logistic_reg() %>% set_engine("glm")` | `LogisticRegSpec { penalty: 0 }` (linfa / smartcore) | same objective (binomial MLE); **different optimizer** (IRLS vs L-BFGS) | tested (L3, 1.6e-8 rel) |
| `logistic_reg() %>% set_engine("glm")` | `LogisticRegSpec` engine `native` | **same algorithm**: `glm.fit` IRLS, clamped logit link, deviance criterion `epsilon = 1e-8`, `maxit = 25` | tested (L3, 3.3e-13 rel; probabilities 4e-15 abs; `converged` exact) |
| `glm` coefficient sign convention | log-odds of the second level | identical | tested |
| `logistic_reg(penalty = λ) %>% set_engine("glmnet")` | not supported | glmnet scales by 1/n and mixes L1/L2; `penalty` here is unscaled L2 | **not equivalent** |
| `initial_split(prop, strata)` | `initial_split` | `floor(n * prop)` per stratum; different RNG | semantic only |
| `initial_validation_split(prop = c(a, b))`, `validation_set()` | `initial_validation_split`, `ThreeWaySplit::validation_set` | train `floor(n a)`, then `b / (1 - a)` of the rest; one resample `validation` | semantic only |
| `initial_time_split(prop)` | `initial_time_split` | first `floor(n * prop)` rows | tested (exact, 4 tasks) |
| `vfold_cv(v, repeats, strata)` | `vfold_cv`, `vfold_cv_repeated` | `sample(rep(1:v, length.out = n))` per stratum; `v == n` refused as rsample; ids `names0` (`Fold1`, `Repeat1.Fold1`) | ids tested exactly; memberships semantic; metrics on shared rsets tested (L4) |
| `bootstraps(times, strata)` | `bootstraps` | draws kept in draw order, sorted within a stratum; assessment = out-of-bag; ids `Bootstrap01` | ids exact; metrics on shared rsets tested (L4, incl. xgboost) |
| `mc_cv(prop, times, strata)` | `mc_cv` | `floor(n * prop)` unsorted draws; ids `Resample1` | ids exact; metrics on shared rsets tested (L4) |
| `loo_cv()` | `loo_cv` | one row out, random order; ids `Resample1`. tune refuses `loo_cv` (`check_rset`); `fit_resamples`/`tune_grid` here accept it | ids per `names0` (unit); **difference**: not refused |
| `group_vfold_cv(group, v, repeats, balance)` | `group_vfold_cv(…, GroupBalance)` | `balance = "groups"` / `"observations"`; `v = NULL` leave-one-group-out; ids `Resample1` | balance rules exact on R's permutations (fixture); metrics on shared rsets tested (L4) |
| `group_bootstraps(group, times)`, `group_mc_cv(group, prop, times)`, `group_initial_split` | same names | `balance = "prop"` cut-off rule; repeated groups' rows adjacent in row order; empty assessment refused | cut-off rule exact on R's draws (fixture); metrics tested (L4) |
| `rolling_origin(initial, assess, cumulative, skip, lag)` | `rolling_origin` | identical windows; ids `Slice1` | tested (exact memberships, 2 tasks) |
| `sliding_window(lookback, assess_start, assess_stop, complete, step, skip)` | `sliding_window` | identical windows (slider rules, `Inf` = `None`) | tested (exact memberships) |
| `sliding_index(index, …)` on a numeric index | `sliding_index` | identical windows; an assessment window with no rows stays empty, as slider | tested (exact memberships) |
| `sliding_period()` | not supported | needs a date index | unsupported |
| `make_strata(x, breaks, pool, depth)` | `make_strata_categorical`, `make_strata_numeric`, `strata_from_numeric` | pooling below `pool` by random reassignment; `breaks <= floor(n / depth)`; type-7 quantile bins, `include.lowest`; NAs to a random bin | unit-tested rules |
| `rsample::make_splits()` + `manual_rset()` | `Split::new`, `Folds::from_splits` | analysis order kept, repeats allowed, assessment sorted, may be empty | tested (L4, every task) |
| `fit_resamples()` | `Workflow::fit_resamples` | prep inside each resample; empty assessment set is an error (rsample warns, tune fails) | tested (L4, 53 tasks) |
| `collect_metrics()` | `ResampleResults::summary` | mean, `sd/sqrt(n)`, n over defined resamples (repeats pooled) | tested (L4) |
| `tune_grid()` + `grid_regular()`/`grid_random()` | `tune_grid`, `grid_regular`, `grid_random` | same search semantics; different random points | tested with explicit shared grids (L4, 4 tasks, Phase 8i); dials grids intended |
| `show_best()`, `select_best()` | `TuneResults::show_best` (`BestRow`), `select_best` (`Selection` with grid `index` and tune's `config` number) | candidates in `.config` order (grid sorted on the parameters in parsnip's argument order, `make_config_labs()`), stable sort by mean, `NA` last; first row | tested (exact config numbers, order and index, 5 tasks) |
| `tune_grid()` sub-models (`boost_tree(trees)`) | every candidate fitted on its own | tune predicts smaller `trees` from the largest fit; parsnip 1.5.0 `xgb_by_tree()` + xgboost ≥ 2.0 inclusive `iterationrange` scores `trees + 1` trees | **difference** (R quirk; measured on `synth_reg_xgb_trees`: tune's value = this crate's `trees + 1`, Phase 8i) |
| `step_impute_mean()` / `step_impute_median()` on an **integer** column | statistic kept as a double | recipes `cast()`s the statistic to the column type: `as.integer(round(x))` | **difference** (this framework has no integer columns; fixtures use double columns) |
| sparse tibble input (`sparsevctrs::as_sparse_double`) | `Column::sparse_numeric` / `with_sparse` (flag, dense storage; ADR-0015) | the flag survives impute / zv / nzv / corr / dummy and is cleared by normalize, as recipes keeps or materialises sparse vectors | tested (`input_sparse` recorded by R, exact on 3 xgboost tasks) |
| `parsnip:::maybe_sparse_matrix()`: any sparse column → `dgCMatrix` for xgboost / ranger / glmnet, densified for lm / glm / rpart / nnet | `DesignMatrix::is_sparse`; the booster treats zeros as missing when the design it is given is sparse, at fit and again at predict (a sparse fit predicting dense new data reads zeros as values, as parsnip converts the baked new data again); other models ignore the flag | identical fits where R densifies or reads zeros as values (lm, rpart, ranger measured in R) | tested (9 tasks, Phase 8j: xgboost exact / to its float32 bounds, lm and rpart to round-off; densified-input predictions compared on the sparse-input tasks) |
| `step_dummy(sparse = "auto" / "yes" / "no")` | `StepDummy::sparse(SparseOutput::{Auto, Yes, No})`, Python `step_dummy(.., sparse=)` | `Yes` emits sparse indicator columns (the design is then sparse for xgboost); `Auto` is dense unless `Workflow::fit`'s toggle sets it | tested (`synth_vsparse_xgb_toggle_dummy`: toggled indicators make the design sparse, exact) |
| `arrow::Table` / `RecordBatch` input | `tidyrustmodels::arrow::table_from_record_batch` (feature `arrow`), Python `from_arrow` / `to_arrow` | column-wise copy: integers/floats → f64 (nulls missing), strings / dictionaries / booleans → categorical | unit-tested round trip |
| Polars DataFrame input | Python `from_polars` / `to_polars` (through Arrow); Rust: convert with `to_arrow()` first | | unit-tested round trip |
| `pd.SparseDtype` / `scipy.sparse` input | `from_pandas` flags fill-value-0 sparse columns; `from_scipy_sparse(matrix, names)` | sparse vectors as in R | unit-tested (booster fit equals the flagged-dense fit, differs from the dense fit) |
| `select_by_one_std_err(x, ..., metric)` | `select_by_one_std_err(metric, &[SortKey])` | first best, `best ∓ std_err`, arranged by the sort keys, first kept; one resample is an error (tune returns no rows) | tested (exact) |
| `select_by_pct_loss(x, ..., metric, limit)` | `select_by_pct_loss(metric, &[SortKey], limit)` | `|(|mean| - |best|) / best| * 100`, first candidate at or before the best in sort order with loss < limit | tested (exact) |
| `finalize_workflow(x, parameters)` | `Workflow::finalize(&params)` | model parameters only (`ModelSpec::with_param`); no recipe tuning | tested through `last_fit` |
| `last_fit(workflow, split, metrics)` | `Workflow::last_fit(data, &Split, metrics)` → `LastFit { workflow, predictions, metrics }` | tune's resample fit path: no sparse-data toggle, frequency weights in the metrics | tested (metrics, predictions, trees/boosters exact) |
| `fit_best(x, metric)` | `fit_best(results, workflow, data, metric)` | `finalize_workflow(select_best())` + `fit()` on every row (toggle applies) | tested (predictions, trees/boosters) |
| `rmse`, `mae` | `rmse`, `mae` | identical formulas | tested (L2) |
| `rsq` | `rsq` | **squared Pearson correlation** | tested (L2) |
| `rsq_trad` | `rsq_trad` | 1 - SSE/SST | tested (L2) |
| `accuracy` | `accuracy` | fraction equal | tested (L2) |
| `precision`, `recall` | `precision`, `recall` | `event_level = "first"`; undefined → `NA` ↔ `None` | tested (L2 incl. NA case) |
| `mn_log_loss` | `mn_log_loss` | `-mean(log p_truth)`, probabilities clamped to `[eps, 1-eps]`, eps = 2.2e-16; multiclass form on the probability matrix | tested (L2 incl. clamp case; multiclass L2 fixed arrays) |
| `conf_mat()` | `ConfusionMatrix::from_codes` | rows = prediction, columns = truth, counts as `f64` | tested (L2, exact) |
| `sens`/`sensitivity`, `spec`/`specificity`, `precision`, `recall` | `ConfusionMatrix::{sens, spec, precision, recall}` | `event_level = "first"`; estimators `binary`, `macro`, `macro_weighted`, `micro` with yardstick's `NA` rules (`weighted.mean(na.rm = TRUE)`, zero-weight levels dropped) | tested (L2 fixed arrays incl. ties, an unpredicted level and an absent level; L2/L4 on three tasks) |
| `f_meas(beta)` | `ConfusionMatrix::f_meas(est, beta)` | `0` when precision and recall are both 0, `NA` when either is undefined | tested (L2 incl. beta = 2 and both-zero case) |
| `bal_accuracy` | `ConfusionMatrix::bal_accuracy` | `(sens + spec)/2`; multiclass average **without** `na.rm` (one undefined level → `NA`), as yardstick | tested (L2 incl. absent-level case) |
| `j_index` | `ConfusionMatrix::j_index` | `sens + spec - 1`; multiclass average with `na.rm` | tested (L2) |
| `kap(weighting)` | `ConfusionMatrix::kap(KapWeighting)` | none / linear / quadratic; `NA` when chance disagreement is 0 (R: `NaN`) | tested (L2 incl. degenerate case) |
| `mcc` | `ConfusionMatrix::mcc` | binary: `NA` when any marginal is 0; multiclass: Gorodkin form, `NA` when the denominator is 0 | tested (L2 incl. both degenerate cases) |
| `roc_curve`, `roc_auc` | `roc_curve`, `roc_auc`, `roc_auc_multiclass` | `binary_threshold_curve` port (ties grouped at the last row), trapezoid; multiclass `hand_till` (default), `macro`, `macro_weighted`; no event or no control → `NA` | tested (L2 curves exact to 1e-15; L2/L4 on three tasks) |
| `pr_curve`, `pr_auc`, `average_precision` | `pr_curve`, `pr_auc`, `average_precision` (+ `_multiclass`) | same curve; `pr_auc` trapezoid, `average_precision` step sum; no events → `None` (R: `NaN`) | tested (L2) |
| `brier_class` | `brier_class`, `brier_class_multiclass` | `brier_ind` with unit case weights: `mean(rowSums((onehot - p)^2)) / 2` | tested (L2) |
| `metric_set(accuracy, roc_auc, brier_class)` (tune's classification default) | `MetricSet::classification_defaults()` | same three metrics | by construction |
| `.pred_<level>` probability columns | `Prediction::Class{probabilities}` ordered as `levels` | event = `levels[0]` | tested (L3) |
| `predict(type = "class")` | `Prediction::Class{codes}` | larger probability wins; ties → first level | tested (L3, exact) |
| `glm$converged`, `glm$iter`, `glm$deviance` | `FitDiagnostics { converged, iterations, deviance, final_gradient_norm }` | native: same meaning and same values; L-BFGS engines: convergence by gradient check, no iteration count | tested (`converged` exact on all engines) |
| `multinom_reg()` + `set_engine("nnet")` | `MultinomRegSpec` | first level is the baseline; `class_coefficients()` = `coef(multinom)` rows; native Newton-Raphson vs nnet's BFGS on the same objective (ADR-0010) | tested (L3 on 2 tasks × 3 engines: deviance 2.4e-15 rel, coefficients ≤ 3.5e-7 rel, probabilities ≤ 2e-8 abs — nnet's residual; L4 metrics) |
| `multinom_reg(penalty)` ↔ nnet `decay` | `MultinomRegSpec::penalty` | **different**: unscaled L2 on slopes only vs `decay` on all weights incl. intercepts; verified at `penalty = 0` only | **difference** |
| `predict(type = "class")` for `multinom` | arg-max | nnet's `max.col` breaks exact ties at random; here the first level wins | **difference** (ties only) |
| `multinom` with rank-deficient design | aliased columns → coefficient 0 + `aliased` (ADR-0007) | nnet keeps unidentified parameters (no `NA`) | **difference** |
| `multinom` with an outcome level absent from training | error | nnet fits (that level's probabilities collapse) | **difference** |
| `tidy(multinom_fit)` std.error / p.value | not provided | | unsupported |
| `decision_tree()` + `set_engine("rpart")` | `DecisionTreeSpec` (native port of rpart 4.1.23, ADR-0011) | `cost_complexity` → `cp`, `tree_depth` → `maxdepth`, `min_n` → `minsplit` (`minbucket = round(min_n/3)`); anova / gini with default priors; midpoint splits; rpart's quicksort tie order and altered-prior round-off reproduced | tested (L3 whole tree exact + round-off on 5 tasks, L4 25 fold trees and predictions, Phase 8d) |
| `predict(type = "prob")` for rpart | `yprob` in R's arithmetic (`counts × prior/freq`, `rowSums`) | mathematically the node class share; R's rounding gives different last bits for equal shares in different nodes, which ROC/PR ties see; reproduced | tested (fold ROC/PR exact) |
| `rand_forest()` + `set_engine("ranger")` | `RandForestSpec` (ranger 0.18.0 growing rules, ADR-0011 A) | `trees` → `num.trees` (500), `mtry` → `mtry` (`floor(sqrt(p))`), `min_n` → `min.node.size` (5 / 10, probability forests as parsnip); bootstrap with replacement, `min.bucket = 1`, midpoint splits, `x <= value` left; random streams are the framework's, not R's | tested statistically (L5: 10 seeds each side, metric means within 4 SE and ≥ 98 % of seed-averaged predictions within 4 SE on 5 tasks; Phase 8e) |
| ranger importance, OOB error, `max.depth`, `sample.fraction`, `replace = FALSE`, `splitrule` variants, class weights | not supported | | **unsupported** |
| `boost_tree()` + `set_engine("xgboost")` | `BoostTreeSpec` (port of xgboost 3.2.1's CPU `hist` booster, ADR-0011 B) | `trees` → `nrounds` (15), `tree_depth` → `max_depth` (6), `learn_rate` → `eta` (0.3), `min_n` → `min_child_weight` (1), `loss_reduction` → `gamma` (0), `lambda`/`alpha` engine args; squared error / `binary:logistic` (first level = event) / `multi:softprob`; base score from the labels; float32 throughout | tested (L3 trees, margins and predictions against xgboost's model JSON on 5 tasks, L4 25 fold boosters; Phase 8f) |
| workflows' sparse-data toggle (`toggle_sparsity`) for xgboost | `Recipe::estimate_sparsity` + `core::sparsity::should_use_sparsity` + `ModelSpec::sparse_engine` + `Recipe::toggle_sparse` (ADR-0015) | when the recipe's estimated sparsity is high enough, `Workflow::fit` sets `step_dummy`'s sparse output; if the recipe then bakes any sparse column the design goes to xgboost as a sparse matrix and **every exact zero is a missing value**; a recipe without a dummy step stays dense whatever the decision (measured in R); decision compared exactly (R samples 1000 rows for the estimate) | tested (Ames and `synth_vsparse_xgb_toggle_dummy`: sparse path; `synth_vsparse_xgb_toggle`: decision "yes" but dense; the rest dense) |
| xgboost `mtry` (`colsample_bynode`), `sample_size` (`subsample`), `stop_iter`, `tree_method = "exact"`/`approx`, GPU, categorical features, NA predictors, monotone/interaction constraints, `scale_pos_weight` | not supported | | **unsupported** |
| rpart categorical splits, surrogates, `xval` cp table, non-default `parms` (priors, loss), `cost` | not supported | nominals must be dummy-encoded and predictors complete (as a recipe does); predictors with missing values are an error | **unsupported** |
| `add_case_weights()` with `hardhat::frequency_weights()` / `importance_weights()` | `Workflow::case_weights(col, kind)`, `CaseWeights`, `WeightKind` (ADR-0012) | the column gets the `case_weights` role (never a predictor, not needed to predict); **frequency** weights are used by the unsupervised steps (`cov.wt` centre/variance/correlation, `weighted_median_impl`, `hardhat::weighted_table`), by the model and, in resampling, by yardstick (tune's `.use_case_weights_with_yardstick()`); **importance** weights reach the model only | tested (L1–L4 on 11 tasks; Phase 8g) |
| engines with case weights | native `lm.wfit` (zero-weight rows dropped, `sqrt(w)` scaling), `glm.fit` prior weights (incl. the non-integer-successes warning), rpart `weights` (statistics weighted, `minsplit`/`minbucket` count rows), ranger `case.weights` (weighted bootstrap sampling; equal weights = none), xgboost DMatrix `weight` (gradients, sketch, base score) | `multinom_reg` (nnet) refuses with parsnip's message; linfa/smartcore engines refuse with `Unsupported` | tested (L3 exact for lm/glm/rpart/xgboost, L5 for ranger; refusals checked) |
| yardstick `case_weights =` | `*_weighted` metric functions, `MetricSet::evaluate_weighted`; Python `class_metrics(case_weights=)` | weighted means/sums, weighted confusion table, `cov.wt` correlation for `rsq`, curves on non-zero-weight rows with cumulative weights, softmax Brier weights; multiclass `roc_auc` defaults to `macro` under weights and `hand_till` + weights is an error | tested (L2 weighted test metrics on frequency tasks; L4 weighted fold metrics) |
| `step_nzv()` with frequency weights on a column with missing values | the weights are subset with the data | recipes 1.3.2 errors (`hardhat::weighted_table()` size mismatch: `nzv()` drops the NAs from the data but not from the weights) | **difference** (R errors) |
| `step_corr()` with frequency weights: tie between columns of equal mean absolute correlation | symmetric matrix with an exact unit diagonal; exact ties drop the earlier column as recipes does | `cov.wt(cor = TRUE)` is elementwise, asymmetric in the last bit for ~35 % of inputs, so R may break such a tie by round-off | **difference** (round-off ties only; reference task agrees) |
| `workflow_set(preproc, models, cross)` | `workflow_set(Vec<(Option<String>, Recipe)>, Vec<(Option<String>, Box<dyn ModelSpec>)>, cross)` (crate `tidyrustmodels-workflowsets`, ADR-0016) | ids `"{preproc}_{model}"`, preprocessors outermost / models innermost, element-wise pairs with `cross = false`; unnamed elements named by class and position as `vctrs::vec_as_names(repair = "unique")` (`recipe_1`, `linear_reg`) | tested (ids exact on 3 tasks: named, unnamed, paired) |
| `workflow_map(x, "tune_grid", resamples, metrics)` | `WorkflowSet::option_grid(id, grid)` + `workflow_map(data, folds, metrics, par)` | a workflow with a grid is tuned, one without is resampled (tune's fallback for `tune_grid` without `tune()`), `.config` `pre0_mod0_post0` / `pre0_mod01_post0` ... | tested (result classes and `.config` labels exact) |
| `collect_metrics(workflow_set)` | `WorkflowSet::collect_metrics() -> Vec<CollectedMetric>` | rows workflow-major, then `.config`, then metric name; `preproc` / `model` columns | tested (row keys exact in order; means / std errors within the summary bounds) |
| `rank_results(x, rank_metric, select_best)` | `WorkflowSet::rank_results(Option<Metric>, bool) -> Vec<RankRow>` | larger-is-better means negated, per-workflow `slice_min(with_ties = FALSE)`, NA ranks last, default metric = first of the set; ties: R draws random ranks under seed 1, this crate keeps the order of appearance | tested (ranks exact outside tie groups, as sets inside; three modes per task) |
| `fit_best(workflow_set, metric)` | `WorkflowSet::fit_best(data, Option<Metric>)` (`best()` gives the workflow and parameters) | top of `rank_results(select_best = TRUE)`, `select_best()` of its result, `finalize` + `fit()` | tested (workflow, parameters, model class and test predictions; needs `save_workflow = TRUE` in R) |
| `comment_*`, `option_*` other than the grid, `collect_predictions()` / `collect_extracts()` on a set, `autoplot`, `leave_var_out_formulas`, `as_workflow_set`, race / Bayesian mapping | not supported | | unsupported |

## Unsupported in v0.1

`tune_bayes`, `tune_race_*`, recipe-parameter tuning, `collect_predictions()` across resamples (`save_pred`), `sliding_period`, `nested_cv`, `permutations`, `apparent`, case weights for
`multinom_reg` and for the linfa/smartcore engines, `step_filter_missing`,
glmnet-equivalent penalization, `tailor` post-processing,
`step_zv(group =)`, `step_corr(method = "spearman"/"kendall")`,
`roc_curve(thresholds =)`, `gain_curve`/`lift_curve`, `classification_cost`,
`roc_aunp`/`roc_aunu`, `detection_prevalence`, `npv`/`ppv`, `mn_log_loss(sum = TRUE)`.

## Engine-specific restrictions

| engine | restriction |
|---|---|
| smartcore | always fits an intercept (`intercept(false)` → `Unsupported`); ignores `max_iterations` / `gradient_tolerance` (fixed L-BFGS settings) |
| linfa | none beyond the framework's aliasing rule; errors on exact collinearity |
| native | ignores `max_iterations` / `gradient_tolerance` (L-BFGS settings); uses `irls_epsilon` / `irls_max_iterations` instead; `penalty` is handled by row augmentation (not glmnet) |
