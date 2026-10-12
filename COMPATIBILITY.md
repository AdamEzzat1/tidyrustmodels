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
| `step_dummy()` with unseen level at bake | error | recipes: warning + NA | **difference** |
| `step_dummy()` with missing categorical value | missing indicator values (error at model time) | recipes: NA row | intended |
| `linear_reg() %>% set_engine("lm")` | `LinearRegSpec` (linfa / smartcore) | OLS; same objective | tested (L3, 2.6e-13 rel) |
| `lm` rank deficiency (`NA` coefficient) | aliased column dropped, coefficient 0, listed in `FitDiagnostics::aliased` | identical predictions; coefficient reported as 0 not `NA` | tested on one Ames fold (ADR-0007); **difference** in reporting |
| `logistic_reg() %>% set_engine("glm")` | `LogisticRegSpec { penalty: 0 }` | same objective (binomial MLE); **different optimizer** (IRLS vs L-BFGS) | tested (L3, 1.6e-8 rel) |
| `glm` coefficient sign convention | log-odds of the second level | identical | tested |
| `logistic_reg(penalty = λ) %>% set_engine("glmnet")` | not supported | glmnet scales by 1/n and mixes L1/L2; `penalty` here is unscaled L2 | **not equivalent** |
| `initial_split(prop, strata)` | `initial_split` | same algorithm family; different RNG | semantic only |
| `vfold_cv(v, strata)` | `vfold_cv` | balanced assignment, per-stratum; different RNG | semantic only |
| `make_strata` on numeric outcome | `strata_from_numeric` | type-7 quantile bins, `include.lowest` | intended |
| `rsample::manual_rset` from ids | `Folds::from_assignments` | exact membership | tested (L4) |
| `fit_resamples()` | `Workflow::fit_resamples` | prep inside each fold | tested (L4, 210 fold metrics) |
| `collect_metrics()` | `ResampleResults::summary` | mean, `sd/sqrt(n)`, n over defined folds | tested (L4) |
| `tune_grid()` + `grid_regular()`/`grid_random()` | `tune_grid`, `grid_regular`, `grid_random` | same search semantics; different random points | intended |
| `rmse`, `mae` | `rmse`, `mae` | identical formulas | tested (L2) |
| `rsq` | `rsq` | **squared Pearson correlation** | tested (L2) |
| `rsq_trad` | `rsq_trad` | 1 - SSE/SST | tested (L2) |
| `accuracy` | `accuracy` | fraction equal | tested (L2) |
| `precision`, `recall` | `precision`, `recall` | `event_level = "first"`; undefined → `NA` ↔ `None` | tested (L2 incl. NA case) |
| `mn_log_loss` | `mn_log_loss` | `-mean(log p_truth)`, probabilities clamped to `[eps, 1-eps]`, eps = 2.2e-16 | tested (L2 incl. clamp case) |
| `.pred_<level>` probability columns | `Prediction::Class{probabilities}` ordered as `levels` | event = `levels[0]` | tested (L3) |
| `predict(type = "class")` | `Prediction::Class{codes}` | larger probability wins; ties → first level | tested (L3, exact) |
| `glm$converged`, `glm$iter` | `FitDiagnostics { converged, final_gradient_norm }` | both expose convergence; iteration counts not comparable (different algorithms) | tested (`converged`) |
| Case weights | not supported | | unsupported |
| `workflowsets` | not supported | | unsupported |

## Unsupported in v0.1

workflowsets, bootstraps, repeated CV, grouped resampling, case weights, multiclass,
glmnet-equivalent penalization, tree models, `tailor` post-processing, `roc_auc`.

## Engine-specific restrictions

| engine | restriction |
|---|---|
| smartcore | always fits an intercept (`intercept(false)` → `Unsupported`); ignores `max_iterations` / `gradient_tolerance` (fixed L-BFGS settings) |
| linfa | none beyond the framework's aliasing rule |
