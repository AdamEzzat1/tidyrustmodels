# One benchmark trial of R tidymodels in a fresh process. Mirrors benchmarks/rust.
# Prints a single line `BENCH_RESULT {json}` on stdout.
#
# Usage:
#   Rscript benchmarks/r/bench_task.R --dataset <csv> --outcome <col>
#       --type regression|classification --predictors a,b,c
#       --steps normalize_numeric,dummy_nominal --metrics rmse,rsq
#       --threads <k> [--warm <n>] [--v 5] [--tune-levels 5] [--seed 1]
#
# Threads > 1 uses future::plan(multisession, workers = k) for fit_resamples / tune_grid,
# which is tidymodels' documented way to parallelise. Model fitting itself (lm/glm) is
# single-threaded with R's reference BLAS on Windows.

t_start <- bench::hires_time()  # monotonic high-resolution clock; proc.time() resolves only ~10 ms on Windows
.libPaths(c(Sys.getenv("R_LIBS_USER"), .libPaths()))
suppressPackageStartupMessages(suppressWarnings({
  library(tidymodels)
  library(jsonlite)
}))
t_libs <- as.numeric(bench::hires_time() - t_start)
Sys.setlocale("LC_COLLATE", "C")

args <- commandArgs(trailingOnly = TRUE)
get <- function(name, default = NULL) {
  i <- match(name, args)
  if (is.na(i) || i == length(args)) return(default)
  args[[i + 1]]
}
dataset <- get("--dataset"); outcome <- get("--outcome"); type <- get("--type")
predictors <- strsplit(get("--predictors"), ",")[[1]]
steps <- strsplit(get("--steps", ""), ",")[[1]]
metrics <- strsplit(get("--metrics"), ",")[[1]]
threads <- as.integer(get("--threads", "1"))
warm <- as.integer(get("--warm", "0"))
v <- as.integer(get("--v", "5"))
tune_levels <- as.integer(get("--tune-levels", "5"))
seed <- as.integer(get("--seed", "1"))

elapsed <- function(expr) {
  t0 <- bench::hires_time()
  force(expr)
  as.numeric(bench::hires_time() - t0)
}

# ---- load ----
t0 <- bench::hires_time()
df <- read.csv(dataset, stringsAsFactors = FALSE, na.strings = "", check.names = FALSE)
df <- df[, c(outcome, predictors), drop = FALSE]
for (nm in names(df)) if (is.character(df[[nm]])) df[[nm]] <- factor(df[[nm]], levels = sort(unique(na.omit(df[[nm]]))))
load_s <- as.numeric(bench::hires_time() - t0)
n <- nrow(df)

# ---- split (own RNG) ----
set.seed(seed)
split <- if (type == "classification") initial_split(df, prop = 0.75, strata = all_of(outcome)) else initial_split(df, prop = 0.75)
train <- training(split); test <- testing(split)

add_steps <- function(rec) {
  for (s in steps) rec <- switch(s,
    impute_median_numeric = step_impute_median(rec, all_numeric_predictors()),
    impute_mean_numeric   = step_impute_mean(rec, all_numeric_predictors()),
    normalize_numeric     = step_normalize(rec, all_numeric_predictors()),
    dummy_nominal         = step_dummy(rec, all_nominal_predictors()),
    rec)
  rec
}
f <- as.formula(paste(outcome, "~ ."))
rec <- add_steps(recipe(f, data = train))
model <- if (type == "regression") linear_reg() %>% set_engine("lm") else logistic_reg() %>% set_engine("glm")
wf <- workflow() %>% add_recipe(rec) %>% add_model(model)

# ---- prep + bake ----
prep_s <- elapsed(prepped <- prep(rec, training = train))
bake_s <- elapsed({ baked_train <- bake(prepped, new_data = NULL); baked_test <- bake(prepped, new_data = test) })

# ---- fit: model only on the baked design (fit_s, like Rust ModelSpec::fit) and the
# ---- workflow fit from the raw training table (wf_fit_s = prep + bake + fit) ----
x_cols <- setdiff(names(baked_train), outcome)
fit_s <- elapsed(model_fit <- fit_xy(model, x = baked_train[, x_cols, drop = FALSE], y = baked_train[[outcome]]))
wf_fit_s <- elapsed(fit_obj <- fit(wf, data = train))
predict_s <- elapsed(pred <- predict(fit_obj, test))

# ---- cross-validation ----
metric_fns <- list(rmse = rmse, rsq = rsq, mae = mae, rsq_trad = rsq_trad,
                   accuracy = accuracy, mn_log_loss = mn_log_loss, precision = precision, recall = recall)
ms <- do.call(metric_set, unname(metric_fns[metrics]))
set.seed(seed + 1)
folds <- if (type == "classification") vfold_cv(df, v = v, strata = all_of(outcome)) else vfold_cv(df, v = v)
if (threads > 1) { suppressPackageStartupMessages(library(future)); plan(multisession, workers = threads) }
cv_s <- elapsed(cv <- fit_resamples(wf, resamples = folds, metrics = ms))
cvm <- collect_metrics(cv)
cv_summary <- lapply(seq_len(nrow(cvm)), function(i) list(metric = cvm$.metric[i], mean = cvm$mean[i], std_err = cvm$std_err[i]))

# ---- tuning (classification only) ----
tune_s <- NULL; tune_candidates <- NULL
if (type == "classification") {
  tune_wf <- workflow() %>% add_recipe(rec) %>%
    add_model(logistic_reg(penalty = tune(), mixture = 0) %>% set_engine("glmnet"))
  # NOTE: tidymodels has no penalised `glm`; the closest documented path is glmnet with
  # mixture = 0 (ridge). This is a *workflow-level* comparison (different algorithm and
  # penalty scaling), never an equivalent-algorithm one. See BENCHMARKS.md.
  if (requireNamespace("glmnet", quietly = TRUE)) {
    grid <- grid_regular(penalty(range = c(-4, 0)), levels = tune_levels)
    tune_s <- elapsed(tr <- tune_grid(tune_wf, resamples = folds, grid = grid, metrics = ms))
    tune_candidates <- nrow(grid)
  }
}
if (threads > 1) plan(sequential)

# ---- warm repetitions ----
w_prep <- c(); w_fit <- c(); w_wffit <- c(); w_predict <- c()
for (i in seq_len(warm)) {
  w_prep <- c(w_prep, elapsed({ p <- prep(rec, training = train); bt <- bake(p, new_data = NULL) }))
  w_fit <- c(w_fit, elapsed(fit_xy(model, x = bt[, x_cols, drop = FALSE], y = bt[[outcome]])))
  w_wffit <- c(w_wffit, elapsed(ff <- fit(wf, data = train)))
  w_predict <- c(w_predict, elapsed(predict(ff, test)))
}
med <- function(x) if (length(x)) median(x) else NA

out <- list(
  system = "r", engine = if (type == "regression") "lm" else "glm", threads = threads,
  n_rows = n, n_train = nrow(train), n_features = ncol(baked_train) - 1L,
  phases = list(load_s = load_s, prep_s = prep_s, bake_s = bake_s, fit_s = fit_s, wf_fit_s = wf_fit_s, predict_s = predict_s,
                cv_s = cv_s, tune_s = tune_s, library_load_s = t_libs),
  cv_summary = cv_summary, tune_candidates = tune_candidates,
  warm = list(iters = warm, prep_bake_s_median = med(w_prep), fit_s_median = med(w_fit), wf_fit_s_median = med(w_wffit), predict_s_median = med(w_predict)),
  versions = list(r = R.version.string, blas = extSoftVersion()[["BLAS"]], lapack = La_library(),
                  tidymodels = as.character(packageVersion("tidymodels")), parsnip = as.character(packageVersion("parsnip")),
                  recipes = as.character(packageVersion("recipes")), tune = as.character(packageVersion("tune")),
                  future = if (threads > 1) as.character(packageVersion("future")) else NULL)
)
cat("BENCH_RESULT", toJSON(out, auto_unbox = TRUE, digits = NA, null = "null"), "\n")
