# ADR-0015: Sparse input and Arrow / Polars / pandas interop

Date: 2026-10-13. Status: accepted.

## Context
tidymodels 1.5 accepts sparse data through `sparsevctrs`: a column can be a *sparse
vector* (default value 0, `NA` stored explicitly), a sparse tibble flows into
`recipe()` and `fit()`, and what reaches the engine depends on the storage type, not
only on the values (measured in R, Phase 8j):

- recipes 1.3.2 keeps a sparse vector sparse through `step_impute_mean` /
  `step_impute_median` (even when it fills values), `step_zv`, `step_nzv`, `step_corr`
  and `step_dummy` (other columns), and materialises it in `step_normalize`;
  `step_dummy(sparse = "yes")` (what workflows' toggle sets) emits sparse indicator
  columns.
- parsnip's `maybe_sparse_matrix()` turns a design with **any** sparse column into a
  `dgCMatrix` for engines with `allow_sparse_x` (xgboost, ranger, glmnet, LiblineaR)
  and densifies it with a warning for the others (lm, glm, rpart, nnet).
- xgboost reads a `dgCMatrix` with every exact zero missing, so the same values give
  a **different model** when a column is sparse (and different predictions when the
  new data's storage differs from the training data's); ranger and glmnet treat the
  zeros as values and give identical fits.
- `.recipes_estimate_sparsity()` and `should_use_sparsity()` are unaffected by the
  storage type (same estimate and decision for a sparse and a dense tibble).
- workflows' toggle acts on the **recipe**, not the model: `toggle_sparsity()` sets
  `sparse = "yes"` on every step whose `sparse` argument is `"auto"` (`step_dummy`
  among the steps ported here), and xgboost sees a `dgCMatrix` only if the baked
  predictors then hold sparse vectors. Measured (`VERIFICATION.md` Phase 8j, second
  pass): a numeric-only recipe whose decision is "yes" fits **dense** (equal to
  `fit_xy()` on the matrix; 0.038 max |Δpred| from the sparse fit), and a model
  fitted on sparse input predicts *dense* new data with zeros as **values**
  (0.87 max |Δpred| from its own sparse-input predictions).

This framework has no sparse storage (ADR-0003: a minimal owned columnar `Table`),
and ADR-0011 B modelled workflows' toggle as a policy on `BoostTreeSpec`, which gets
both measured cases wrong. Arrow and Polars input had been deferred to an optional
feature (ADR-0003).

## Decision
1. **A sparse flag, not sparse storage.** `Column::Numeric` carries `sparse: bool`
   (`Column::sparse_numeric`, `with_sparse`, `is_sparse`; serialized with a default
   so earlier recipe states load). Values stay dense: the flag records what R would
   store sparsely and is what changes the model. `DesignMatrix` carries the same flag,
   set by `bake_matrix` when any predictor column is sparse (`maybe_sparse_matrix`).
2. **Steps propagate the flag as recipes does:** imputation keeps it, normalisation
   clears it, filters keep the columns they keep, row selection keeps it, and
   `StepDummy` carries recipes' `sparse` argument (`SparseOutput::{Auto, Yes, No}`,
   `.sparse(..)`): with `Yes` its indicator columns are sparse vectors.
3. **The toggle is a recipe operation, as in workflows.** `ModelSpec::sparse_engine`
   (replacing ADR-0011 B's `resolve_sparsity`) names the R engine when it accepts
   sparse matrices and the policy is `Auto` (`"xgboost"`; `"ranger"`, whose rule is
   always "no"); `Workflow::fit` then computes `should_use_sparsity` on the recipe's
   estimate and preps `Recipe::toggle_sparse(choice)` (`.recipes_toggle_sparse_args`:
   `Step::toggle_sparse` sets `Auto` to `Yes` / `No`). The toggled recipe is what the
   fitted workflow keeps, so prediction bakes new data the same way. Resampling fits
   do not toggle (tune's `.fit_pre`, unchanged).
4. **The booster reads the design in front of it.** `BoostTreeSpec::fit` treats zeros
   as missing iff the design is sparse (sparse input surviving the recipe, or sparse
   indicators after the toggle) or the policy is `ZerosMissing`; `FittedBoostTree::
   predict` iff the *new* design is sparse or the policy forced it (`forced_zeros_
   missing`, serialized with a default so earlier states keep their fitted rule).
   `SparsePolicy::Dense` only skips the toggle. All other models ignore the flag (R
   densifies or reads zeros as values, with the same fit).
5. **Interop converts column-wise** (ADR-0003 unchanged): Rust `tidyrustmodels::arrow`
   (feature `arrow`; `arrow-array`/`arrow-schema` 60) maps a `RecordBatch` to a
   `Table` (integers and floats to `f64` with nulls missing; `Utf8`, `LargeUtf8`,
   dictionary strings and booleans to categoricals with sorted levels) and back
   (`Float64` / `Utf8` with nulls). Python adds `from_arrow` / `to_arrow`
   (pyarrow Table or RecordBatch), `from_polars` / `to_polars` (through Arrow),
   `to_pandas`, `from_scipy_sparse(matrix, names)` (every column a sparse vector) and
   `Table.with_sparse(names)` / `Table.sparse_columns`; `from_pandas` flags
   `SparseDtype` columns with fill value 0. The flag is not carried out of a Table
   (Arrow, Polars and pandas exports are dense).
6. **Verification** (Phase 8j): reference tasks carry `sparse_columns`, which the R
   runner turns into sparse vectors before anything else; fixtures record whether the
   baked predictors of the recipe as written hold sparse vectors
   (`fit.sparsity.input_sparse`), whether the recipe the fitted workflow keeps bakes a
   sparse design (`design_sparse`) and, for sparse-input tasks, the predictions on the
   test rows with the sparse columns materialised (`predicted_test_dense_input`); the
   harness checks the decision, both sparsity facts and the booster's missing rule
   exactly and compares the densified-input predictions; three very sparse tasks
   (`synth_vsparse_xgb_*`) exercise the decision "yes" without a dummy step (dense in
   R), with one (sparse indicators) and with sparse input.
7. Not ported: sparse storage and its memory savings, sparse output from `bake()` as
   storage, glmnet / LiblineaR engines, Polars in Rust (convert through Arrow),
   zero-copy Arrow.

## Consequences
- A sparse flag on a column changes xgboost fits on purpose; users converting from
  SciPy sparse matrices get tidymodels' behaviour, and `with_sparse([])` clears it.
- `ModelSpec::resolve_sparsity` is gone (breaking for implementors; `sparse_engine`
  replaces it) and the fitted booster no longer carries the toggle: the fitted
  workflow's recipe does, exactly as in R, so a `FittedBoostTree` used outside a
  workflow follows the design it is given.
- `Column::Numeric` literals gained a field; serialized states remain readable.
- Three synthetic datasets (`synthetic_sparse_reg`, `synthetic_sparse_clf`,
  `synthetic_vsparse_reg`) and nine tasks were added; `pyarrow`, `polars` and `scipy`
  are optional Python extras.
