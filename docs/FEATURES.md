# Supported feature matrix (0.1.0 release candidate)

One line per feature with its verification status. Status values: **verified** =
compared against R tidymodels output in the differential suite (VERIFICATION.md, with
the phase) or against an analytic baseline; **unit-tested** = behaviour fixed by unit
or property tests only; **unsupported** = not in this release. COMPATIBILITY.md has
the semantic detail behind every row; LIMITATIONS.md the known gaps.

Reference versions: R 4.4.0, tidymodels 1.5.0, parsnip 1.5.0, recipes 1.3.2, rsample
1.3.2, yardstick 1.4.0, tune 2.1.0, workflows 1.3.0, workflowsets 1.1.1, rpart 4.1.23,
ranger 0.18.0, xgboost 3.2.1.1.

## Data

| feature | Rust | Python | status |
|---|---|---|---|
| columnar `Table` with explicit missing masks, categorical levels sorted bytewise | `Table`, `Column` | `Table.from_dict`, `from_numpy` | verified (every task) |
| CSV input | `tidyrustmodels::io::read_csv` | `pandas.read_csv` + `from_pandas` | verified (fixture datasets) |
| pandas | — | `from_pandas` / `to_pandas` (incl. `SparseDtype`) | unit-tested |
| Arrow | `tidyrustmodels::arrow` (feature `arrow`) | `from_arrow` / `to_arrow` | unit-tested round trip |
| Polars | through Arrow | `from_polars` / `to_polars` | unit-tested round trip |
| SciPy sparse | `Column::sparse_numeric` | `from_scipy_sparse`, `Table.with_sparse` | verified (sparse semantics, Phase 8j) |
| case-weight columns | `Workflow::case_weights` | `Workflow(case_weights=)` | verified (Phase 8g) |

## Preprocessing (recipes)

| step | Rust | Python | status |
|---|---|---|---|
| `step_select` | `StepSelect` | `step_select` | unit-tested |
| `step_impute_mean` / `step_impute_median` | `StepImputeMean` / `StepImputeMedian` | `step_impute_mean` / `step_impute_median` | verified (Phase 4; frequency weights Phase 8g) |
| `step_normalize` | `StepNormalize` | `step_normalize` | verified (Phase 4) |
| `step_dummy` (incl. `sparse = "yes"`) | `StepDummy` | `step_dummy(one_hot=, sparse=)` | verified (Phase 4, 8j) |
| `step_zv`, `step_nzv`, `step_corr` | `StepZv`, `StepNzv`, `StepCorr` | same names | verified (Phase 8a) |
| `step_other`, `step_novel`, `step_unknown` | `StepOther`, `StepNovel`, `StepUnknown` | same names | verified (Phase 8a) |
| prep on training rows only, bake anywhere, serialized fitted state | `Recipe::prep`, `FittedRecipe::bake`, JSON | inside `Workflow` | verified (leakage test, round trip) |
| `step_interact`, `step_pca`, splines, `step_filter_missing`, recipe-parameter tuning | — | — | unsupported |

## Models (parsnip) and engines

| model | engines | status |
|---|---|---|
| `linear_reg` / `lm` | `native` (pivoted-QR port of `lm.fit`, default), `linfa`, `smartcore` | verified (round-off vs R, Phase 4 / 7; analytic OLS baseline) |
| `logistic_reg` / `glm` | `native` (IRLS port of `glm.fit`, default), `linfa`, `smartcore` (L-BFGS) | verified (native round-off; L-BFGS 1e-8) |
| `multinom_reg` / `nnet` | `native`, `linfa`, `smartcore` | verified to nnet's convergence residual (Phase 8c) |
| `decision_tree` / `rpart` | `native` (rpart 4.1.23 port) | verified node for node (Phase 8d) |
| `rand_forest` / `ranger` | `native` (ranger growing rules, own RNG) | verified statistically, level 5 (Phase 8e) |
| `boost_tree` / `xgboost` | `native` (xgboost 3.2.1 hist port) | verified tree for tree, float32 bounds (Phase 8f) |
| `penalty` on logistic / multinomial | unscaled L2 on slopes | unit-tested; **not** glmnet-equivalent |
| glmnet, SVM, kNN, MARS, neural nets, `tailor` post-processing | — | unsupported |

## Resampling (rsample)

| scheme | status |
|---|---|
| `initial_split`, `initial_validation_split`, `initial_time_split` | verified ids / semantics (Phase 8h) |
| `vfold_cv` (+ `repeats`, `strata`), `bootstraps`, `mc_cv`, `loo_cv` | verified (ids exact; metrics on shared rsets) |
| `group_vfold_cv`, `group_bootstraps`, `group_mc_cv`, `group_initial_split` | verified (balance rules on R's permutations) |
| `rolling_origin`, `sliding_window`, `sliding_index` | verified (memberships exact) |
| `make_strata` | unit-tested |
| `sliding_period`, `nested_cv`, `permutations`, `apparent` | unsupported |

## Metrics (yardstick)

| family | status |
|---|---|
| `rmse`, `mae`, `rsq`, `rsq_trad` | verified (Phase 4) |
| `accuracy`, `mn_log_loss`, `precision`, `recall`, `sens`, `spec`, `f_meas`, `bal_accuracy`, `j_index`, `kap`, `mcc`, `brier_class`, `conf_mat` (binary, macro, macro_weighted, micro) | verified (Phase 8b) |
| `roc_auc`, `pr_auc`, `average_precision` and curves (binary; `hand_till`, macro, macro_weighted) | verified (Phase 8b) |
| case-weighted versions of all of the above | verified (Phase 8g) |
| `gain_curve`, `lift_curve`, `classification_cost`, `roc_aunp` / `roc_aunu`, `npv` / `ppv` | unsupported |

## Tuning (tune / dials)

| feature | status |
|---|---|
| `tune_grid` with regular, random or explicit grids; parallel over (candidate, fold) | verified (Phase 8i) |
| `show_best`, `select_best`, `select_by_one_std_err`, `select_by_pct_loss`, tune's `.config` numbering | verified exactly (Phase 8i) |
| `finalize_workflow`, `last_fit`, `fit_best` | verified (Phase 8i) |
| `tune_bayes`, racing, `collect_predictions(save_pred)` | unsupported |

## Workflow sets (workflowsets)

| feature | status |
|---|---|
| `workflow_set` (crossed / paired; default names), `workflow_map` (tune or resample), `collect_metrics`, `rank_results`, `fit_best`, `extract_*` | verified (Phase 8k; ties ordered rather than random) |
| comments, extracts, predictions on a set, racing / Bayesian mapping | unsupported |

## Framework properties

| property | status |
|---|---|
| serial == parallel results (`fit_resamples`, `tune_grid`) | unit-tested (`assert_eq!` on full results) |
| fitted workflow JSON round trip | unit-tested, bitwise |
| Python == Rust bitwise on shared inputs | verified (parity tests on every task) |
| no leakage: recipe re-prepped inside every resample | unit-tested against a manual computation |
| reproducibility from explicit `Seed`s | unit-tested |
