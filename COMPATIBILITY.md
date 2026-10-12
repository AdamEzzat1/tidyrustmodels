# tidymodels compatibility

Status: semantic contracts defined in Phase 0. The "tested" column is filled only from
VERIFICATION.md records; "intended" means not yet verified by a fixture test.

Reference: R 4.4.0, tidymodels 1.5.0, parsnip 1.5.0, recipes 1.3.2, rsample 1.3.2,
yardstick 1.4.0, tune 2.1.0.

## Semantic contracts

| tidymodels feature | tidyrustmodels | semantic agreement | status |
|--------------------|----------------|--------------------|--------|
| `recipe()` + `prep()` + `bake()` | `Recipe::prep` / `FittedRecipe::bake` | prep on training only; bake never refits | intended |
| `step_normalize()` | `StepNormalize` | mean via `colMeans(na.rm=TRUE)`, sd via `stats::var` (n-1), zero-variance sd → 1 (`sd_check`) | intended |
| `step_impute_mean()` | `StepImputeMean` | mean of non-missing training values (`trim = 0`) | intended |
| `step_impute_median()` | `StepImputeMedian` | `stats::median` (type 7) | intended |
| `step_dummy()` | `StepDummy` | `one_hot = FALSE` drops first level; names `var_level` (`make.names` sanitised) | intended |
| `step_dummy()` with unseen level at bake | error | recipes: warning + NA | **intentional difference** |
| `linear_reg() %>% set_engine("lm")` | `LinearRegSpec` (linfa / smartcore) | OLS; same objective | intended |
| `logistic_reg() %>% set_engine("glm")` | `LogisticRegSpec { penalty: 0 }` | same objective (binomial MLE); **different optimizer** (IRLS vs L-BFGS) | intended; bounded by tolerance |
| `logistic_reg(penalty = λ) %>% set_engine("glmnet")` | not supported in v0.1 | glmnet's objective scales the penalty by 1/n and uses elastic-net mixture; linfa's `alpha` is unscaled L2 | **not equivalent** — documented |
| `initial_split(prop, strata)` | `initial_split` | same algorithm family; different RNG | semantic only |
| `vfold_cv(v, strata)` | `vfold_cv` | balanced assignment, per-stratum; different RNG | semantic only |
| `rsample::manual_rset` from ids | `Folds::from_assignments` | exact membership | intended |
| `fit_resamples()` | `Workflow::fit_resamples` | prep inside each fold | intended |
| `tune_grid()` + `grid_regular()`/`grid_random()` | `tune_grid`, `grid_regular`, `grid_random` | same search semantics; different random points | semantic only |
| `rmse`, `mae` | `rmse`, `mae` | identical formulas | intended |
| `rsq` | `rsq` | **squared Pearson correlation** (not 1 - SSE/SST) | intended |
| `rsq_trad` | `rsq_trad` | 1 - SSE/SST | intended |
| `accuracy` | `accuracy` | fraction equal | intended |
| `precision`, `recall` | `precision`, `recall` | `event_level = "first"`; undefined → `NA` ↔ `None` | intended |
| `mn_log_loss` | `mn_log_loss` | `-mean(log p_event)` with event = first level; yardstick clamps probabilities with `eps`: verified in fixture tests | intended |
| `.pred_<level>` probability columns | `Prediction::Class{probabilities}` columns ordered as `levels` | event = `levels[0]` | intended |
| `glm$converged`, `glm$iter` | `FitDiagnostics { converged, iterations }` | both expose convergence | intended |
| Case weights | not supported | | unsupported |
| `workflowsets` | not supported | | unsupported |

## Unsupported in v0.1

workflowsets, bootstraps, repeated CV, grouped resampling, case weights, multiclass,
glmnet-equivalent penalization, tree models, `tailor` post-processing.
