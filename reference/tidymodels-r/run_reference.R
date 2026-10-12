# Run the R tidymodels reference for every task in reference/tasks.json and write
# one JSON fixture per task to reference/fixtures/r/<task_id>.json, plus
# metrics_fixed.json (yardstick on fixed arrays, comparison level 2).
#
# Run from the repository root AFTER export_datasets.R, make_synthetic.py and
# generate_splits.py:
#   Rscript reference/tidymodels-r/run_reference.R
#
# Everything here is deterministic given the shared split ids: no RNG is used.

.libPaths(c(Sys.getenv("R_LIBS_USER"), .libPaths()))
suppressPackageStartupMessages(suppressWarnings({
  library(tidymodels)
  library(jsonlite)
}))
Sys.setlocale("LC_COLLATE", "C")
options(warn = 1)

root <- "."
tasks <- fromJSON(file.path(root, "reference", "tasks.json"), simplifyVector = FALSE)$tasks
out_dir <- file.path(root, "reference", "fixtures", "r")
dir.create(out_dir, recursive = TRUE, showWarnings = FALSE)

versions <- list(
  r = R.version.string,
  platform = R.version$platform,
  blas = extSoftVersion()[["BLAS"]],
  lapack = La_library(),
  packages = sapply(c("tidymodels", "parsnip", "recipes", "rsample", "yardstick", "tune", "workflows", "dials"),
                    function(p) as.character(packageVersion(p)), simplify = FALSE),
  generated_on = format(Sys.time(), "%Y-%m-%d %H:%M:%S")
)

read_dataset <- function(name, task) {
  df <- read.csv(file.path(root, "reference", "fixtures", "datasets", paste0(name, ".csv")),
                 stringsAsFactors = FALSE, na.strings = "", check.names = FALSE)
  keep <- c(task$outcome, unlist(task$predictors))
  df <- df[, keep, drop = FALSE]
  for (nm in names(df)) {
    if (is.character(df[[nm]])) {
      df[[nm]] <- factor(df[[nm]], levels = sort(unique(stats::na.omit(df[[nm]]))))
    }
  }
  df
}

add_steps <- function(rec, steps) {
  for (s in unlist(steps)) {
    rec <- switch(s,
      impute_median_numeric = step_impute_median(rec, all_numeric_predictors()),
      impute_mean_numeric   = step_impute_mean(rec, all_numeric_predictors()),
      normalize_numeric     = step_normalize(rec, all_numeric_predictors()),
      dummy_nominal         = step_dummy(rec, all_nominal_predictors()),
      stop("unknown step ", s))
  }
  rec
}

metric_fns <- list(rmse = rmse, rsq = rsq, mae = mae, rsq_trad = rsq_trad,
                   accuracy = accuracy, mn_log_loss = mn_log_loss, precision = precision, recall = recall)

columns_as_list <- function(df) {
  lapply(as.list(df), function(col) if (is.factor(col)) as.character(col) else col)
}

run_task <- function(task) {
  cat("== task", task$id, "\n")
  df <- read_dataset(task$dataset, task)
  split <- fromJSON(file.path(root, "reference", "fixtures", "splits", paste0(task$id, ".json")))
  train <- df[split$initial$train + 1, , drop = FALSE]
  test  <- df[split$initial$test + 1, , drop = FALSE]
  rownames(train) <- NULL; rownames(test) <- NULL

  f <- as.formula(paste(task$outcome, "~ ."))
  rec <- add_steps(recipe(f, data = train), task$steps)
  model <- if (task$type == "regression") linear_reg() %>% set_engine("lm") else logistic_reg() %>% set_engine("glm")
  wf <- workflow() %>% add_recipe(rec) %>% add_model(model)

  # ---- Level 1: preprocessing ----
  prepped <- prep(rec, training = train)
  step_states <- lapply(seq_along(prepped$steps), function(k) {
    td <- tidy(prepped, number = k)
    td$id <- NULL
    list(step = class(prepped$steps[[k]])[1], tidy = columns_as_list(td))
  })
  baked_train <- bake(prepped, new_data = NULL)
  baked_test  <- bake(prepped, new_data = test)
  pred_cols <- setdiff(names(baked_train), task$outcome)

  # ---- Level 3: estimator ----
  fit <- fit(wf, data = train)
  eng <- extract_fit_engine(fit)
  coefs <- coef(eng)
  fit_info <- list(
    coefficients = as.list(coefs),
    coefficient_names = names(coefs),
    fitted_train = if (task$type == "regression") as.numeric(predict(fit, train)$.pred)
                   else columns_as_list(predict(fit, train, type = "prob")),
    predicted_test = if (task$type == "regression") as.numeric(predict(fit, test)$.pred)
                     else columns_as_list(predict(fit, test, type = "prob")),
    class_test = if (task$type == "classification") as.character(predict(fit, test)$.pred_class) else NULL,
    converged = if (task$type == "classification") eng$converged else NULL,
    iterations = if (task$type == "classification") eng$iter else NULL,
    deviance = if (task$type == "classification") eng$deviance else NULL,
    outcome_levels = if (task$type == "classification") levels(train[[task$outcome]]) else NULL,
    xtx_condition = {
      X <- as.matrix(baked_train[, pred_cols, drop = FALSE]); X <- cbind(1, X)
      kappa(crossprod(X), exact = TRUE)
    }
  )
  test_metrics <- {
    ms <- do.call(metric_set, unname(metric_fns[unlist(task$metrics)]))
    if (task$type == "regression") {
      m <- ms(data.frame(truth = test[[task$outcome]], estimate = fit_info$predicted_test), truth = truth, estimate = estimate)
    } else {
      probs <- predict(fit, test, type = "prob"); cls <- predict(fit, test)$.pred_class
      d <- data.frame(truth = test[[task$outcome]], estimate = cls, prob = probs[[1]])
      m <- ms(d, truth = truth, estimate = estimate, prob)
    }
    setNames(as.list(m$.estimate), m$.metric)
  }

  # ---- Level 4: resampling with shared folds ----
  fold_of_row <- split$vfold$fold_of_row
  v <- split$vfold$v
  splits <- lapply(seq_len(v) - 1, function(k) {
    assess <- which(fold_of_row == k)
    make_splits(list(analysis = setdiff(seq_len(nrow(df)), assess), assessment = assess), data = df)
  })
  rset <- manual_rset(splits, sprintf("Fold%02d", seq_len(v)))
  ms <- do.call(metric_set, unname(metric_fns[unlist(task$metrics)]))
  res <- fit_resamples(wf, resamples = rset, metrics = ms)
  per_fold <- collect_metrics(res, summarize = FALSE)
  per_fold_list <- lapply(split(per_fold, per_fold$id), function(d) setNames(as.list(d$.estimate), d$.metric))
  summ <- collect_metrics(res)
  summary_list <- lapply(seq_len(nrow(summ)), function(i) list(metric = summ$.metric[i], mean = summ$mean[i], std_err = summ$std_err[i], n = summ$n[i]))

  out <- list(
    task = task,
    versions = versions,
    n_train = nrow(train), n_test = nrow(test),
    prep = list(steps = step_states,
                baked_train = columns_as_list(baked_train[, pred_cols, drop = FALSE]),
                baked_test = columns_as_list(baked_test[, pred_cols, drop = FALSE]),
                predictor_names = pred_cols),
    fit = fit_info,
    test_metrics = test_metrics,
    resample = list(ids = names(per_fold_list), per_fold = per_fold_list, summary = summary_list)
  )
  write_json(out, file.path(out_dir, paste0(task$id, ".json")), auto_unbox = TRUE, digits = NA, na = "null", null = "null", pretty = FALSE)
  cat("   wrote", file.path(out_dir, paste0(task$id, ".json")), "\n")
}

for (task in tasks) run_task(task)

# ---- Level 2: metrics on fixed arrays ----
truth_num <- c(1.0, 2.5, 3.0, 4.25, 5.0, 6.75)
est_num   <- c(1.2, 2.0, 3.5, 4.0, 5.5, 6.0)
lv <- c("a", "b")
truth_cls <- factor(c("a", "a", "b", "b", "a", "b"), levels = lv)
est_cls   <- factor(c("a", "b", "b", "b", "a", "a"), levels = lv)
prob_a    <- c(0.9, 0.4, 0.2, 0.1, 0.6, 0.55)
fixed <- list(
  versions = versions,
  regression = list(truth = truth_num, estimate = est_num,
                    rmse = rmse_vec(truth_num, est_num), mae = mae_vec(truth_num, est_num),
                    rsq = rsq_vec(truth_num, est_num), rsq_trad = rsq_trad_vec(truth_num, est_num)),
  classification = list(levels = lv, truth = as.character(truth_cls), estimate = as.character(est_cls), prob_event = prob_a,
                        accuracy = accuracy_vec(truth_cls, est_cls), precision = precision_vec(truth_cls, est_cls),
                        recall = recall_vec(truth_cls, est_cls), mn_log_loss = mn_log_loss_vec(truth_cls, prob_a)),
  undefined = list(
    rsq_constant_truth = rsq_vec(c(1, 1, 1), c(1, 2, 3)),
    precision_no_predicted_events = suppressWarnings(precision_vec(factor(c("b", "b"), levels = lv), factor(c("b", "b"), levels = lv))),
    log_loss_clamped = mn_log_loss_vec(factor(c("a", "b"), levels = lv), c(0, 0.5))
  )
)
write_json(fixed, file.path(out_dir, "metrics_fixed.json"), auto_unbox = TRUE, digits = NA, na = "null", null = "null", pretty = TRUE)
cat("wrote metrics_fixed.json\n")
