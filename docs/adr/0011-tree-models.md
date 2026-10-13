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
  sections of this ADR state the verification they get instead.

## Amendment A (2026-10-12): random forests

### Context
`parsnip::rand_forest()` with engine `ranger` fits ranger 0.18.0. A forest is a
function of its random streams (bootstrap draws and the `mtry` variable subsets per
node); ranger draws them from a Mersenne Twister seeded through R, and reproducing
those streams would mean porting the generator and ranger's exact call order, which
is not a model property. Two forests with the same rules and different streams are
two samples from one distribution, so that is what can be verified.

### Decision
1. **`RandForestSpec`** grows trees with ranger's rules (`Tree.cpp`,
   `TreeRegression.cpp`, `TreeProbability.cpp`): bootstrap of `n` rows with
   replacement; `mtry` variables drawn without replacement at every node; a node is
   terminal when `n <= min.node.size` or pure; the split maximises
   `sum_L^2/n_L + sum_R^2/n_R` (regression) or `sum_j (count_Lj^2)/n_L + sum_j
   (count_Rj^2)/n_R` (gini), any split being accepted (`best_decrease = -1`), subject
   to `min.bucket = 1`; the split value is the midpoint of adjacent sorted values,
   falling back to the lower value when the midpoint rounds to the upper one;
   `x <= value` goes left; leaves hold the in-bag mean or in-bag class fractions
   and the forest averages them. Defaults as parsnip resolves them: `trees = 500`,
   `mtry = max(1, floor(sqrt(p)))`, `min_n = 5` (regression) / `10` (classification,
   because parsnip fits ranger *probability* forests).
2. **Randomness** is the framework's: tree `t` of a forest with seed `s` uses the
   ChaCha8 stream `Seed(s).derive(t)`, so a fit is reproducible across platforms and
   thread counts, which ranger's is not across `num.threads`.
3. **Verification (level 5)** is statistical: R fits `S = 10` seeds, Rust fits 10
   seeds, and for every metric `|mean_R - mean_Rust| <= 4 * sqrt(var_R/S +
   var_Rust/S)`; the seed-averaged test predictions must satisfy the same band on at
   least 98 % of the values; `mtry`, `min.node.size` and the tree type must match
   ranger's resolved values exactly. A 4-SE band is a detector for a systematic
   difference between the two growers, not a round-off bound: a bias smaller than
   about `4 * sd / sqrt(5)` of a metric passes undetected, and that is stated in
   VERIFICATION.md with the measured sizes.
4. **Engines.** `native` only, for the reason in Decision 5 above.

### Consequences
- The forest cannot be compared tree by tree, and no level-4 (resampling) check is
  run for it: 5 folds x 10 seeds of forests would multiply the R run time for a
  check that the level-5 band already covers.
- Variable importance, OOB error, `max.depth`, `sample.fraction`, `replace = FALSE`,
  case weights, class weights, `splitrule` variants (extratrees, hellinger, maxstat)
  and categorical splits are not implemented; the recipe dummy-encodes nominals.

