# One benchmark trial of R tidymodels in a fresh process. Mirrors benchmarks/rust.
# Prints a single line `BENCH_RESULT {json}` on stdout.
#
# Usage (shared CLI contract with the Rust and Python drivers):
#   Rscript benchmarks/r/bench_task.R --dataset <csv> --outcome <col>
#       --type regression|classification --predictors a,b,c
#       --steps normalize_numeric,dummy_nominal --metrics rmse,rsq --threads <k>
#       [--model linear_reg|logistic_reg|decision_tree|rand_forest|boost_tree]
#       [--model-args trees=100,min_n=5] [--case-weights col:frequency|importance]
#       [--resampling vfold|repeated|bootstraps] [--v 10] [--repeats 3] [--times 25]
#       [--tune none|grid] [--tune-levels 20] [--select] [--wfs] [--warm <n>] [--seed 1]
#
# Threads > 1 uses future::plan(multisession, workers = k) for fit_resamples / tune_grid /
# workflow_map, tidymodels' documented way to parallelise. Engines run single-threaded
# (lm/glm with R's reference BLAS; rpart; ranger num.threads = 1; xgboost nthread = 1).
# Memory allocated by R-level code is measured once with bench::bench_memory() on the
# workflow fit (Rprofmem: R allocations only, not C-level mallocs inside engines).

t_start <- bench::hires_time()  # monotonic high-resolution clock; proc.time() resolves only ~10 ms on Windows
.libPaths(c(Sys.getenv("R_LIBS_USER"), .libPaths()))
suppressPackageStartupMessages(suppressWarnings({
  library(tidymodels)
  library(workflowsets)
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
has <- function(name) name %in% args
dataset <- get("--dataset"); outcome <- get("--outcome"); type <- get("--type")
predictors <- strsplit(get("--predictors"), ",")[[1]]
steps <- strsplit(get("--steps", ""), ",")[[1]]
metrics <- strsplit(get("--metrics"), ",")[[1]]
threads <- as.integer(get("--threads", "1"))
model_name <- get("--model", if (type == "regression") "linear_reg" else "logistic_reg")
margs_raw <- get("--model-args", "")
margs <- if (nzchar(margs_raw)) {
  kv <- strsplit(strsplit(margs_raw, ",")[[1]], "=")
  setNames(lapply(kv, function(p) as.numeric(p[2])), vapply(kv, `[`, "", 1))
} else list()
cw <- get("--case-weights")
resampling <- get("--resampling", "vfold")
warm <- as.integer(get("--warm", "0"))
v <- as.integer(get("--v", "5"))
repeats <- as.integer(get("--repeats", "1"))
times <- as.integer(get("--times", "25"))
tune_mode <- get("--tune", "grid")
tune_levels <- as.integer(get("--tune-levels", "5"))
do_select <- has("--select"); do_wfs <- has("--wfs")
seed <- as.integer(get("--seed", "1"))

elapsed <- function(expr) {
  t0 <- bench::hires_time()
  force(expr)
  as.numeric(bench::hires_time() - t0)
}

# ---- load ----
t0 <- bench::hires_time()
df <- read.csv(dataset, stringsAsFactors = FALSE, na.strings = "", check.names = FALSE)
cw_col <- NULL; cw_kind <- NULL
if (!is.null(cw)) { parts <- strsplit(cw, ":")[[1]]; cw_col <- parts[1]; cw_kind <- if (length(parts) > 1) parts[2] else "frequency" }
df <- df[, c(outcome, predictors, cw_col), drop = FALSE]
for (nm in names(df)) if (is.character(df[[nm]])) df[[nm]] <- factor(df[[nm]], levels = sort(unique(na.omit(df[[nm]]))))
if (!is.null(cw_col)) {
  df[[cw_col]] <- if (identical(cw_kind, "importance")) hardhat::importance_weights(as.double(df[[cw_col]])) else hardhat::frequency_weights(as.integer(df[[cw_col]]))
}
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
    zv                    = step_zv(rec, all_predictors()),
    rec)
  rec
}
f <- as.formula(paste(outcome, "~ ."))
rec <- add_steps(recipe(f, data = train))
make_model <- function(name, a) {
  a <- a[!vapply(a, is.null, TRUE)]
  switch(name,
    linear_reg = linear_reg() %>% set_engine("lm"),
    logistic_reg = logistic_reg() %>% set_engine("glm"),
    decision_tree = do.call(decision_tree, a[intersect(names(a), c("cost_complexity", "tree_depth", "min_n"))]) %>%
      set_engine("rpart") %>% set_mode(type),
    rand_forest = do.call(rand_forest, a[intersect(names(a), c("trees", "mtry", "min_n"))]) %>%
      set_engine("ranger", seed = seed, num.threads = 1) %>% set_mode(type),
    boost_tree = do.call(boost_tree, a[intersect(names(a), c("trees", "tree_depth", "learn_rate", "min_n"))]) %>%
      set_engine("xgboost", nthread = 1) %>% set_mode(type),
    stop("unknown model ", name))
}
model <- make_model(model_name, margs)
engine_name <- switch(model_name, linear_reg = "lm", logistic_reg = "glm", decision_tree = "rpart", rand_forest = "ranger", boost_tree = "xgboost")
make_wf <- function(m) {
  w <- workflow() %>% add_recipe(rec) %>% add_model(m)
  if (!is.null(cw_col)) w <- add_case_weights(w, !!rlang::sym(cw_col))
  w
}
wf <- make_wf(model)

# ---- prep + bake ----
prep_s <- elapsed(prepped <- prep(rec, training = train))
bake_s <- elapsed({ baked_train <- bake(prepped, new_data = NULL); baked_test <- bake(prepped, new_data = test) })

# ---- fit: model only on the baked design (fit_s, like Rust ModelSpec::fit) and the
# ---- workflow fit from the raw training table (wf_fit_s = prep + bake + fit) ----
x_cols <- setdiff(names(baked_train), c(outcome, cw_col))
fit_s <- elapsed(model_fit <- if (is.null(cw_col)) fit_xy(model, x = baked_train[, x_cols, drop = FALSE], y = baked_train[[outcome]]) else
  fit_xy(model, x = baked_train[, x_cols, drop = FALSE], y = baked_train[[outcome]], case_weights = baked_train[[cw_col]]))
wf_fit_s <- elapsed(fit_obj <- fit(wf, data = train))
predict_s <- elapsed(pred <- predict(fit_obj, test))
# R-level allocation of the workflow fit (bytes), measured outside the timed region
mem <- tryCatch(bench::bench_memory(fit(wf, data = train)), error = function(e) NULL)
wf_fit_mem_alloc_mb <- if (is.null(mem)) NULL else as.numeric(mem$mem_alloc) / 2^20

# ---- resampling ----
metric_fns <- list(rmse = rmse, rsq = rsq, mae = mae, rsq_trad = rsq_trad,
                   accuracy = accuracy, mn_log_loss = mn_log_loss, precision = precision, recall = recall,
                   roc_auc = roc_auc, brier_class = brier_class)
ms <- do.call(metric_set, metric_fns[metrics])
set.seed(seed + 1)
strata_arg <- if (type == "classification") rlang::quo(all_of(outcome)) else NULL
folds <- switch(resampling,
  vfold = if (type == "classification") vfold_cv(df, v = v, strata = all_of(outcome)) else vfold_cv(df, v = v),
  repeated = if (type == "classification") vfold_cv(df, v = v, repeats = repeats, strata = all_of(outcome)) else vfold_cv(df, v = v, repeats = repeats),
  bootstraps = if (type == "classification") bootstraps(df, times = times, strata = all_of(outcome)) else bootstraps(df, times = times),
  stop("unknown resampling ", resampling))
if (threads > 1) { suppressPackageStartupMessages(library(future)); plan(multisession, workers = threads) }
cv_s <- elapsed(cv <- fit_resamples(wf, resamples = folds, metrics = ms))
cvm <- collect_metrics(cv)
cv_summary <- lapply(seq_len(nrow(cvm)), function(i) list(metric = cvm$.metric[i], mean = cvm$mean[i], std_err = cvm$std_err[i]))

# ---- tuning + selection ----
tune_s <- NULL; tune_candidates <- NULL; select_s <- NULL; select_params <- NULL
tune_spec <- function(name) switch(name,
  logistic_reg = list(model = logistic_reg(penalty = tune(), mixture = 0) %>% set_engine("glmnet"),
                      # NOTE: tidymodels has no penalised `glm`; glmnet ridge is the closest documented path.
                      # Workflow-level comparison only (different algorithm and penalty scaling).
                      grid = grid_regular(penalty(range = c(-4, 0)), levels = tune_levels)),
  decision_tree = list(model = decision_tree(cost_complexity = tune(), min_n = tune()) %>% set_engine("rpart") %>% set_mode(type),
                       grid = tidyr::expand_grid(cost_complexity = c(1e-4, 1e-3, 1e-2, 1e-1), min_n = c(2L, 5L, 10L, 15L, 20L))),
  boost_tree = list(model = do.call(boost_tree, c(list(tree_depth = tune(), learn_rate = tune()), margs[intersect(names(margs), "trees")])) %>%
                      set_engine("xgboost", nthread = 1) %>% set_mode(type),
                    grid = tidyr::expand_grid(tree_depth = c(2L, 4L, 6L, 8L), learn_rate = c(0.05, 0.1, 0.2, 0.3, 0.5))),
  NULL)
ts <- if (tune_mode == "grid") tune_spec(model_name) else NULL
if (!is.null(ts) && (model_name != "logistic_reg" || requireNamespace("glmnet", quietly = TRUE))) {
  tune_wf <- make_wf(ts$model)
  tune_s <- elapsed(tr <- tune_grid(tune_wf, resamples = folds, grid = ts$grid, metrics = ms))
  tune_candidates <- nrow(ts$grid)
  if (do_select) {
    select_s <- elapsed({
      best <- select_best(tr, metric = metrics[1])
      lf <- last_fit(finalize_workflow(tune_wf, best), split = split, metrics = ms)
    })
    select_params <- as.list(best[setdiff(names(best), ".config")])
  }
}

# ---- workflow set: base model, a tuned tree (3 cost_complexity values), xgboost ----
wfs_s <- NULL; wfs_best <- NULL
if (do_wfs) {
  wfs_s <- elapsed({
    wfs <- workflow_set(list(rec = rec),
                        list(base = model,
                             tree = decision_tree(cost_complexity = tune()) %>% set_engine("rpart") %>% set_mode(type),
                             xgb = boost_tree(trees = 20) %>% set_engine("xgboost", nthread = 1) %>% set_mode(type)))
    wfs <- option_add(wfs, grid = tibble::tibble(cost_complexity = c(1e-3, 1e-2, 1e-1)), id = "rec_tree")
    wfs <- workflow_map(wfs, "tune_grid", resamples = folds, metrics = ms, seed = seed,
                        control = control_grid(save_workflow = TRUE), verbose = FALSE)
    ranked <- rank_results(wfs, select_best = TRUE)
    best_fit <- fit_best(wfs)
  })
  wfs_best <- ranked$wflow_id[1]
}
if (threads > 1) plan(sequential)

# ---- warm repetitions ----
w_prep <- c(); w_fit <- c(); w_wffit <- c(); w_predict <- c()
for (i in seq_len(warm)) {
  w_prep <- c(w_prep, elapsed({ p <- prep(rec, training = train); bt <- bake(p, new_data = NULL) }))
  w_fit <- c(w_fit, elapsed(if (is.null(cw_col)) fit_xy(model, x = bt[, x_cols, drop = FALSE], y = bt[[outcome]]) else
    fit_xy(model, x = bt[, x_cols, drop = FALSE], y = bt[[outcome]], case_weights = bt[[cw_col]])))
  w_wffit <- c(w_wffit, elapsed(ff <- fit(wf, data = train)))
  w_predict <- c(w_predict, elapsed(predict(ff, test)))
}
med <- function(x) if (length(x)) median(x) else NA

out <- list(
  system = "r", engine = engine_name, model = model_name, threads = threads,
  n_rows = n, n_train = nrow(train), n_features = length(x_cols), n_resamples = nrow(folds),
  phases = list(load_s = load_s, prep_s = prep_s, bake_s = bake_s, fit_s = fit_s, wf_fit_s = wf_fit_s, predict_s = predict_s,
                cv_s = cv_s, tune_s = tune_s, select_s = select_s, wfs_s = wfs_s, library_load_s = t_libs),
  allocs = list(wf_fit = list(count = NULL, bytes = if (is.null(wf_fit_mem_alloc_mb)) NULL else wf_fit_mem_alloc_mb * 2^20)),
  cv_summary = cv_summary, tune_candidates = tune_candidates, select_params = select_params, wfs_best = wfs_best,
  warm = list(iters = warm, prep_bake_s_median = med(w_prep), fit_s_median = med(w_fit), wf_fit_s_median = med(w_wffit), predict_s_median = med(w_predict)),
  versions = list(r = R.version.string, blas = extSoftVersion()[["BLAS"]], lapack = La_library(),
                  tidymodels = as.character(packageVersion("tidymodels")), parsnip = as.character(packageVersion("parsnip")),
                  recipes = as.character(packageVersion("recipes")), tune = as.character(packageVersion("tune")),
                  workflowsets = as.character(packageVersion("workflowsets")),
                  rpart = as.character(packageVersion("rpart")), ranger = as.character(packageVersion("ranger")),
                  xgboost = as.character(packageVersion("xgboost")),
                  future = if (threads > 1) as.character(packageVersion("future")) else NULL)
)
cat("BENCH_RESULT", toJSON(out, auto_unbox = TRUE, digits = NA, null = "null"), "\n")
