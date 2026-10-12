# ADR-0011: Tree models — rpart port first, forests and boosting by their R semantics

Date: 2026-10-12 · Status: accepted (decision tree); forest and boosting sections are
amended as each lands

## Context
Phase 8 priority 5 asks for decision trees, random forests and boosted trees. The
framework's rule (master prompt, ADR-0009) is that a native algorithm needs a measured
justification, and every feature with an R equivalent needs differential verification.
The Rust ecosystem has tree learners (linfa-trees, smartcore), but none reproduces
`rpart`'s tree: they differ in the split criterion scaling, the minimum-bucket rule,
the midpoint convention, the cost-complexity collapse and the tie rules, so a
differential test against `parsnip::decision_tree()` (engine `rpart`) could only
compare metrics loosely. That is the measured justification: without a port there is
no exact R verification for trees at all.

## Decision
1. **`DecisionTreeSpec`** is a native port of rpart 4.1.23's C code
   (`partition.c`, `bsplit.c`, `nodesplit.c`, `anova.c`, `gini.c`, `rpmatrix.c`,
   `pred_rpart.c`), restricted to what a baked recipe provides: continuous predictors
   without missing values. Categorical splits (`graycode.c`), surrogates and the
   cross-validated cp table are not ported; nominals are dummy-encoded by the recipe,
   exactly as a tidymodels workflow with `step_dummy()` presents them to rpart.
2. **Parameters** map as parsnip does: `cost_complexity` → `cp` (0.01),
   `tree_depth` → `maxdepth` (30), `min_n` → `minsplit` (20) with
   `minbucket = round(min_n / 3)`. Classification uses the default priors (class
   frequencies) and 0/1 loss, under which `gini.c`'s altered priors are all 1.
3. **Prediction** follows `pred_rpart`: `x < point` takes the side recorded in the
   split's direction; class = the node's minimum-loss class (first on ties, as
   `ginidev`); probabilities = weighted class shares (`yprob` with default priors).
4. **Verification** compares the whole tree with `fit$frame` and `fit$splits`:
   node numbers, split variables, sizes, directions and counts exactly; split points,
   deviances, means/classes, class counts, cp and `improve` to round-off; then test
   predictions, test metrics and the 5-fold resampling metrics as for every model.
5. **Engines.** Only `native` implements `decision_tree`. linfa-trees / smartcore are
   not offered as engines for it: they would not be "the same model with another
   solver" (ADR-0005) but a different tree, and the framework does not want two
   things called `decision_tree` that disagree by design.

## Consequences
- The port is a transcription (~500 lines) with rpart's tie rules documented in the
  module header; it is reviewed against the cited C files rather than re-derived.
- Case weights (priority 6) enter the port through `wt` exactly as in rpart, so the
  weighted tree will be verifiable against `rpart(weights =)` without a redesign.
- Random forests and boosted trees cannot be verified this way (ranger's RNG and
  xgboost's float32 histogram training are not reproducible from outside); their
  sections of this ADR state the verification they get instead when they land.
