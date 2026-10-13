# Changelog

## 0.1.0-dev (unreleased)

First vertical slice. Everything listed here is backed by tests recorded in
VERIFICATION.md; nothing is claimed beyond those records.

### Added
- Core contracts: `Table`/`Column` with explicit missing-value masks, `Schema` with roles,
  column-major `DesignMatrix`, `Outcome`/`Prediction`, structured `Error`, `Step`/`ModelSpec`
  lifecycle traits, explicit `Seed` and bounded `Parallelism`.
- Recipes: `StepSelect`, `StepImputeMean`, `StepImputeMedian`, `StepNormalize`, `StepDummy`
  (R `make.names` port, recipes column order), prep-on-training/bake-anywhere, versioned JSON.
- Models: `LinearRegSpec` and `LogisticRegSpec` on two engines (`linfa` 0.8.1 default,
  `smartcore` 0.6.16); framework-side prediction from coefficients; aliased-column handling;
  backend-independent gradient convergence check and separation warning.
- Resampling: `initial_split`, `vfold_cv` (optionally stratified), `Folds::from_assignments`,
  numeric strata binning.
- Metrics: rmse, mae, rsq (squared correlation, as yardstick), rsq_trad, accuracy, precision,
  recall, mn_log_loss (eps-clamped, as yardstick); `MetricSet`.
- Classification metrics with yardstick 1.4.0 semantics: `ConfusionMatrix` (sens, spec,
  precision, recall, f_meas, bal_accuracy, j_index, kap, mcc, accuracy; binary / macro /
  macro_weighted / micro), `roc_curve`/`roc_auc` (Hand-Till for multiclass),
  `pr_curve`/`pr_auc`/`average_precision`, `brier_class`, multiclass `mn_log_loss`; all
  verified on fixed arrays with ties and degenerate cases and on three reference tasks
  (VERIFICATION.md Phase 8b). `MetricSet::classification_defaults()` is now tune's
  `accuracy, roc_auc, brier_class`. Python: `class_metrics`, `conf_mat`.
- `MultinomRegSpec` (`multinom_reg`, ADR-0010): multinomial regression with the first
  level as baseline on all three engines (native Newton-Raphson with the exact Hessian;
  linfa/smartcore softmax converted to baseline form), `FittedModel::class_coefficients`,
  `Capability::MulticlassClassification`; two reference tasks against `nnet` (deviance
  agrees to 2.4e-15, coefficients to nnet's 3.5e-7 residual). Python: `MultinomReg`.
- `DecisionTreeSpec` (`decision_tree`, engine rpart; ADR-0011): a native port of rpart
  4.1.23's partitioning for continuous predictors (anova and gini, rpart's quicksort tie
  order, altered-prior and `yprob` round-off reproduced). Five reference tasks: whole
  trees exact against R, 25 fold trees exact, predictions and metrics to round-off.
  Tasks can now name a `model` and `model_args`; fixtures carry 17 significant digits.
  Python: `DecisionTree`.
- `RandForestSpec` (`rand_forest`, engine ranger; ADR-0011 Amendment A): ranger 0.18.0's
  growing rules (bootstrap, `mtry`, variance / gini decrease, midpoint splits,
  probability forests for classification) under per-tree ChaCha8 streams from the
  framework's `Seed`. Verified statistically (differential level 5): on five tasks the
  metric means of 10 R and 10 Rust seeds agree within 4 SE (largest ratio 0.48) and at
  least 98 % of seed-averaged predictions row-wise; a 60-seed follow-up on penguins
  reproduced a bimodal log loss at the same rate (7/60) on both sides. Python:
  `RandForest`.
- `BoostTreeSpec` (`boost_tree`, engine xgboost; ADR-0011 Amendment B): a port of
  xgboost 3.2.1's CPU `hist` booster (quantile sketch, float32 gains and leaves,
  depth-wise growth, squared error / logistic / softprob). tidymodels' sparse-data
  decision is reproduced: `Recipe::estimate_sparsity`, `Step::sparse_adjustments`,
  `ModelSpec::resolve_sparsity` and `core::sparsity::should_use_sparsity`; on the
  sparse path every exact zero is a missing value, as xgboost sees it from a
  `dgCMatrix`. Five reference tasks: trees, margins and predictions match xgboost's
  model JSON (details in VERIFICATION.md Phase 8f). Python: `BoostTree`.
- `reference/tidymodels-r/run_reference.R` takes `TRM_TASKS=id1,id2` to regenerate a
  subset of fixtures.
- Workflows: `fit`, `predict`, `fit_resamples` with per-fold prep and serial-identical
  parallel execution; versioned JSON round trip.
- Tuning: `grid_regular`, `grid_random`, `tune_grid`, `show_best`.
- Python bindings (PyO3 0.29, maturin, abi3 ≥ 3.11) with NumPy/pandas input and an exception
  hierarchy; bitwise parity with the Rust API.
- Reference harness against R tidymodels 1.5.0 with committed fixtures, shared split ids,
  per-quantity tolerances, and a differential test (611 comparisons).
- `native` engine (ADR-0009), now the **default**: pure-Rust ports of R's `lm.fit`
  pivoted QR and `glm.fit` IRLS; matches R to round-off on every reference task,
  reproduces R's `NA` column choice under collinearity, and fits logistic models 3–4×
  faster than the L-BFGS engines. `FitDiagnostics` gains `deviance`; `LogisticRegSpec` gains
  `irls_epsilon` / `irls_max_iterations`.
- Recipe steps `StepZv`, `StepNzv`, `StepCorr`, `StepOther`, `StepNovel`, `StepUnknown`
  (recipes 1.3.2 semantics; two new differential tasks, all states exact against R).
  `read_csv` accepts entirely empty columns as numeric all-missing.
- Benchmark harness (Rust and R drivers, orchestrator with peak-RSS sampling, report) and
  four recorded runs (`benchmarks/results/2026-10-12-win11-i7-11390H` and `…-b`, `…-c`,
  `…-d`: exact peak memory, a like-for-like workflow-fit phase, the Python wrapper as a
  third system, interleaved trials; run C is labelled invalid for timing); see
  BENCHMARKS.md for what they show, including where tidyrustmodels is slower than R.

### Known limitations
See LIMITATIONS.md. Highlights: no penalised paths, trees or case weights;
general collinearity aliased by the native engine only; smartcore engine always fits an intercept and ignores optimizer settings;
unseen categorical levels at bake time are an error.
