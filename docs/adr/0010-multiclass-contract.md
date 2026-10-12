# ADR-0010: Multiclass outcomes — parametrisation, probabilities and estimators

Date: 2026-10-12 · Status: accepted

## Context
ADR-0004 fixed the binary contract: the first level is the event and `logistic_reg`
reports log-odds of the second level, as R's `glm` does. Phase 8 adds outcomes with
three or more levels. R users compare against `parsnip::multinom_reg()` (engine `nnet`)
and yardstick's multiclass estimators, so the framework needs one stated convention
for coefficients, the probability matrix, class prediction and metric averaging.

## Decision
1. **Model.** `MultinomRegSpec` (`multinom_reg`) fits the multinomial logit with the
   **first level as baseline** (`eta_0 = 0`) and one coefficient vector per other level,
   in level order. `FittedModel::class_coefficients()` returns `(level, Coefficients)`
   for levels `1..k`; this is exactly `coef(nnet::multinom)` (rows = non-baseline
   levels). `coefficients()` stays `None` for this model so callers cannot mistake one
   level's set for the whole model. Two-level outcomes are accepted and reduce to
   `logistic_reg` for the second level.
2. **Probabilities.** `Prediction::Class{probabilities}` keeps one column per level in
   `levels` order (unchanged), computed by the framework from the coefficients with a
   max-shifted softmax. Class prediction is the arg-max; ties go to the first level
   (`nnet`'s `max.col` breaks ties at random, which no fixture can reproduce).
3. **Engines.** `native` solves the objective by Newton-Raphson with the exact Hessian
   (Cholesky, step halving, relative-deviance criterion `epsilon = 1e-10`); `linfa` and
   `smartcore` fit their symmetric softmax parametrisations and the adapter converts to
   the baseline form (`theta_c - theta_0`), which is the identifiable quantity. Aliased
   columns follow ADR-0007 on every engine (the native engine also applies the `dqrdc2`
   rule); `nnet` instead keeps unidentified parameters, a documented difference.
4. **Penalty.** `penalty` is the framework's unscaled L2 term on the slopes of every
   level (intercepts free), the same meaning as in `logistic_reg`; it is not `nnet`'s
   `decay` (which penalises intercepts too) and not glmnet's `lambda`.
5. **Capability.** `Capability::MulticlassClassification` is required for outcomes with
   more than two levels; `check_capabilities` rejects `logistic_reg` on them.
6. **Metrics.** `Metric::compute` applies yardstick's defaults by level count: two
   levels → `binary` with the first level as event; more → `macro` for class metrics
   and `pr_auc`/`average_precision`, `hand_till` for `roc_auc`, and the single
   multiclass forms of `accuracy`, `kap`, `mcc`, `brier_class`, `mn_log_loss`. Explicit
   `macro_weighted` / `micro` estimators are available on `ConfusionMatrix` and the
   `*_multiclass` curve functions.
7. **Reference.** Differential fixtures use `multinom_reg() %>% set_engine("nnet",
   maxit = 10000, reltol = 1e-15)`; the default `reltol = 1e-8` leaves the parameters
   4e-4 (synthetic) to 1% (near-separable penguins) away from the optimum, which is
   recorded in VERIFICATION.md rather than hidden in a tolerance.

## Consequences
- Agreement with R is bounded by `nnet`'s convergence, not ours: on both reference
  tasks the deviances agree to 2.4e-15 relative with every Rust engine, ours is never
  above `nnet`'s, and coefficients differ by up to 3.5e-7 relative on every engine
  alike (tolerances.toml `multinom_*`).
- Metrics on multiclass models inherit that residual; the native engine's exact-arithmetic
  tolerance overrides are therefore replaced per task for the multiclass tasks.
- Separable or quasi-separable multiclass data (iris with all four predictors, penguins
  with `sex`) are excluded from the reference tasks: neither solver has a finite optimum
  to agree on, and `nnet`'s answer there depends on its stopping rule.
