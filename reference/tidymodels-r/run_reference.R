# Run the R tidymodels reference for every task in reference/tasks.json and write
# one JSON fixture per task to reference/fixtures/r/<task_id>.json, plus
# metrics_fixed.json (yardstick on fixed arrays, comparison level 2).
# Numbers are written with 17 significant digits so every double round-trips exactly
# (15 digits merged distinct doubles, which breaks ties in curve metrics).
#
# Run from the repository root AFTER export_datasets.R, make_synthetic.py and
# generate_splits.py:
#   Rscript reference/tidymodels-r/run_reference.R
# (set TRM_TASKS=id1,id2 to regenerate a subset; metrics_fixed.json is always rewritten)
#
# Everything here is deterministic given the shared split ids; the only randomness is
# ranger's, fixed per forest through set_engine(seed =) (Phase 8e).

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
  keep <- c(task$outcome, unlist(task$predictors), task$case_weights$column)
  df <- df[, keep, drop = FALSE]
  for (nm in names(df)) {
    if (is.character(df[[nm]])) {
      df[[nm]] <- factor(df[[nm]], levels = sort(unique(stats::na.omit(df[[nm]]))))
    } else if (is.integer(df[[nm]])) {
      # the framework has only double columns; recipes casts imputed statistics to
      # the column type (`cast()`: integer columns get round()ed means / medians),
      # so whole-number columns are read as doubles on both sides (ADR-0014 finding)
      df[[nm]] <- as.double(df[[nm]])
    }
  }
  # case weights get hardhat's classes; recipe() then assigns the `case_weights` role
  if (!is.null(task$case_weights)) {
    cw <- task$case_weights$column
    df[[cw]] <- if (identical(task$case_weights$kind, "frequency")) hardhat::frequency_weights(as.integer(df[[cw]]))
                else hardhat::importance_weights(as.double(df[[cw]]))
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
      zv                    = step_zv(rec, all_predictors()),
      nzv                   = step_nzv(rec, all_predictors()),
      corr_numeric          = step_corr(rec, all_numeric_predictors()),
      other_nominal         = step_other(rec, all_nominal_predictors()),
      novel_nominal         = step_novel(rec, all_nominal_predictors()),
      unknown_nominal       = step_unknown(rec, all_nominal_predictors()),
      stop("unknown step ", s))
  }
  rec
}

metric_fns <- list(rmse = rmse, rsq = rsq, mae = mae, rsq_trad = rsq_trad,
                   accuracy = accuracy, mn_log_loss = mn_log_loss, precision = precision, recall = recall,
                   roc_auc = roc_auc, pr_auc = pr_auc, average_precision = average_precision,
                   f_meas = f_meas, sens = sens, spec = spec, bal_accuracy = bal_accuracy,
                   j_index = j_index, kap = kap, mcc = mcc, brier_class = brier_class)

columns_as_list <- function(df) {
  lapply(as.list(df), function(col) if (is.factor(col)) as.character(col) else col)
}

run_task <- function(task) {
  cat("== task", task$id, "\n")
  df <- read_dataset(task$dataset, task)
  # simplifyMatrix = FALSE keeps rset$analysis / rset$assessment as lists of integer
  # vectors even when every resample has the same size
  split <- fromJSON(file.path(root, "reference", "fixtures", "splits", paste0(task$id, ".json")), simplifyMatrix = FALSE)
  # the raw CSV for the columns the recipe never sees (grouping / index variables)
  raw <- read.csv(file.path(root, "reference", "fixtures", "datasets", paste0(task$dataset, ".csv")),
                  stringsAsFactors = FALSE, na.strings = "", check.names = FALSE)
  stopifnot(nrow(raw) == nrow(df))
  if (identical(split$initial$scheme, "initial_time_split")) {
    its <- initial_time_split(df, prop = split$initial$prop)
    stopifnot(identical(as.integer(its$in_id), as.integer(split$initial$train + 1L)))
  }
  train <- df[split$initial$train + 1, , drop = FALSE]
  test  <- df[split$initial$test + 1, , drop = FALSE]
  rownames(train) <- NULL; rownames(test) <- NULL

  f <- as.formula(paste(task$outcome, "~ ."))
  rec <- add_steps(recipe(f, data = train), task$steps)
  wcol <- task$case_weights$column
  # frequency weights reach yardstick (tune's .use_case_weights_with_yardstick()); importance weights do not
  metric_weights <- if (!is.null(wcol) && identical(task$case_weights$kind, "frequency")) wcol else NULL
  make_wf <- function(model) {
    w <- workflow() %>% add_recipe(rec) %>% add_model(model)
    if (!is.null(wcol)) w <- add_case_weights(w, !!rlang::sym(wcol))
    w
  }
  # metric set call with optional case weights taken from column `wcol` of `d`
  ms_eval <- function(ms, d, ..., weighted = FALSE) {
    if (weighted && !is.null(metric_weights)) ms(d, ..., case_weights = !!rlang::sym(metric_weights)) else ms(d, ...)
  }
  n_lv <- if (task$type == "classification") nlevels(train[[task$outcome]]) else 0L
  is_tree <- identical(task$model, "decision_tree")
  is_rf <- identical(task$model, "rand_forest")
  is_boost <- identical(task$model, "boost_tree")
  multi <- n_lv > 2 && !is_tree && !is_rf && !is_boost
  rf_seeds <- 1:10
  rf_model <- function(seed) {
    args <- if (is.null(task$model_args)) list() else task$model_args
    do.call(rand_forest, args) %>% set_engine("ranger", seed = seed, num.threads = 1) %>% set_mode(task$type)
  }
  model <- if (is_tree) {
    args <- if (is.null(task$model_args)) list() else task$model_args
    do.call(decision_tree, args) %>% set_engine("rpart") %>% set_mode(task$type)
  } else if (is_boost) {
    args <- if (is.null(task$model_args)) list() else task$model_args
    do.call(boost_tree, args) %>% set_engine("xgboost") %>% set_mode(task$type)
  } else if (is_rf) rf_model(rf_seeds[1]) else if (task$type == "regression") linear_reg() %>% set_engine("lm") else
           if (multi) multinom_reg() %>% set_engine("nnet", maxit = 10000, reltol = 1e-15, trace = FALSE) else
           logistic_reg() %>% set_engine("glm")
  wf <- make_wf(model)

  # ---- Level 1: preprocessing ----
  prepped <- prep(rec, training = train)
  step_states <- lapply(seq_along(prepped$steps), function(k) {
    td <- tidy(prepped, number = k)
    td$id <- NULL
    list(step = class(prepped$steps[[k]])[1], tidy = columns_as_list(td))
  })
  baked_train <- bake(prepped, new_data = NULL)
  baked_test  <- bake(prepped, new_data = test)
  pred_cols <- setdiff(names(baked_train), c(task$outcome, wcol))

  # ---- Level 3: estimator ----
  fit <- fit(wf, data = train)
  eng <- extract_fit_engine(fit)
  tree_info <- if (is_tree) {
    fr <- eng$frame
    nn <- nrow(fr)
    leaf <- fr$var == "<leaf>"
    idx <- 1L + c(0L, cumsum((!leaf) + fr$ncompete + fr$nsurrogate))[-(nn + 1L)]
    idx[leaf] <- NA
    sp <- eng$splits
    k <- n_lv
    list(node = as.integer(row.names(fr)), var = as.character(fr$var), n = fr$n, wt = fr$wt, dev = fr$dev,
         yval = fr$yval, complexity = fr$complexity,
         split_point = ifelse(is.na(idx), NA, sp[idx, "index"]),
         split_dir = ifelse(is.na(idx), NA, sp[idx, "ncat"]),
         split_improve = ifelse(is.na(idx), NA, sp[idx, "improve"]),
         split_count = ifelse(is.na(idx), NA, sp[idx, "count"]),
         counts = if (k > 0) unname(as.matrix(fr$yval2[, 1L + (1:k), drop = FALSE])) else NULL,
         yprob = if (k > 0) unname(as.matrix(fr$yval2[, 1L + k + (1:k), drop = FALSE])) else NULL,
         cptable = unname(eng$cptable[, c("CP", "nsplit", "rel error"), drop = FALSE]),
         control = eng$control[c("minsplit", "minbucket", "cp", "maxdepth")])
  } else NULL
  # xgboost: the model JSON (float32 split values, leaves, stats) and raw margins
  boost_dump <- function(b) {
    mj <- jsonlite::fromJSON(rawToChar(xgboost::xgb.save.raw(b, raw_format = "json")), simplifyVector = FALSE)
    lp <- mj$learner$learner_model_param
    trees <- lapply(mj$learner$gradient_booster$model$trees, function(t) {
      f <- function(nm) unlist(t[[nm]])
      list(split_indices = f("split_indices"), split_conditions = f("split_conditions"),
           left_children = f("left_children"), right_children = f("right_children"),
           base_weights = f("base_weights"), sum_hessian = f("sum_hessian"),
           loss_changes = f("loss_changes"), default_left = f("default_left"))
    })
    cfg <- xgboost::xgb.config(b); if (is.character(cfg)) cfg <- jsonlite::fromJSON(cfg)
    tp <- cfg$learner$gradient_booster$tree_train_param
    list(objective = mj$learner$objective$name, base_score = lp$base_score, num_class = lp$num_class,
         n_trees = length(trees), trees = trees,
         params = tp[c("eta", "max_depth", "gamma", "min_child_weight", "lambda", "alpha", "max_bin")],
         updater = cfg$learner$gradient_booster$gbtree_train_param$updater)
  }
  # raw margins through parsnip's own predict path (which bakes test data sparse or
  # dense exactly as the fit saw it); outputmargin skips the link transform
  boost_margin <- function(fitted_wf, newdata) {
    unname(as.matrix(predict(fitted_wf, newdata, type = "raw", opts = list(outputmargin = TRUE))))
  }
  boost_info <- if (is_boost) boost_dump(eng) else NULL
  rf_info <- if (is_rf) {
    ms <- do.call(metric_set, unname(metric_fns[unlist(task$metrics)]))
    per_seed <- lapply(rf_seeds, function(s) {
      fs <- fit(make_wf(rf_model(s)), data = train)
      if (task$type == "regression") {
        p <- as.numeric(predict(fs, test)$.pred)
        d <- data.frame(truth = test[[task$outcome]], estimate = p)
        if (!is.null(wcol)) d[[wcol]] <- test[[wcol]]
        m <- ms_eval(ms, d, truth = truth, estimate = estimate, weighted = TRUE)
        list(metrics = setNames(as.list(m$.estimate), m$.metric), predicted_test = p)
      } else {
        probs <- predict(fs, test, type = "prob"); cls <- predict(fs, test)$.pred_class
        d <- cbind(data.frame(truth = test[[task$outcome]], estimate = cls), as.data.frame(probs))
        if (!is.null(wcol)) d[[wcol]] <- test[[wcol]]
        m <- if (n_lv > 2) ms_eval(ms, d, truth = truth, estimate = estimate, tidyselect::starts_with(".pred_"), weighted = TRUE)
             else ms_eval(ms, d, truth = truth, estimate = estimate, !!rlang::sym(names(probs)[1]), weighted = TRUE)
        list(metrics = setNames(as.list(m$.estimate), m$.metric), prob_test = unname(as.matrix(probs)))
      }
    })
    list(seeds = rf_seeds, treetype = eng$treetype, mtry = eng$mtry, min_node_size = eng$min.node.size,
         num_trees = eng$num.trees, per_seed = per_seed)
  } else NULL
  coefs <- if (is_tree || is_rf || is_boost) NULL else coef(eng)
  coef_list <- function(cf) {
    if (is.null(cf)) return(NULL)
    if (is.matrix(cf)) lapply(setNames(seq_len(nrow(cf)), rownames(cf)), function(r) as.list(cf[r, ])) else as.list(cf)
  }
  nnet_default <- if (multi) {
    wf2 <- make_wf(multinom_reg() %>% set_engine("nnet", trace = FALSE))
    e2 <- extract_fit_engine(fit(wf2, data = train))
    list(value = e2$value, convergence = e2$convergence, coefficients = coef_list(coef(e2)))
  } else NULL
  fit_info <- list(
    tree = tree_info,
    rf = rf_info,
    boost = boost_info,
    sparsity = if (is_boost) {
      est <- recipes::.recipes_estimate_sparsity(rec)
      list(estimate = est, decision = workflows:::should_use_sparsity(est, "xgboost", nrow(train)))
    } else NULL,
    margin_test = if (is_boost) boost_margin(fit, test) else NULL,
    coefficients = coef_list(coefs),
    coefficient_names = if (is_tree || is_rf || is_boost) NULL else if (multi) colnames(coefs) else names(coefs),
    coefficient_levels = if (multi) rownames(coefs) else NULL,
    nnet_value = if (multi) eng$value else NULL,
    nnet_default = nnet_default,
    fitted_train = if (task$type == "regression") as.numeric(predict(fit, train)$.pred)
                   else columns_as_list(predict(fit, train, type = "prob")),
    predicted_test = if (task$type == "regression") as.numeric(predict(fit, test)$.pred)
                     else columns_as_list(predict(fit, test, type = "prob")),
    class_test = if (task$type == "classification") as.character(predict(fit, test)$.pred_class) else NULL,
    converged = if (task$type == "classification" && !is_tree && !is_rf && !is_boost) { if (multi) eng$convergence == 0 else eng$converged } else NULL,
    iterations = if (task$type == "classification" && !multi && !is_tree && !is_rf && !is_boost) eng$iter else NULL,
    deviance = if (task$type == "classification" && !is_tree && !is_rf && !is_boost) { if (multi) 2 * eng$value else eng$deviance } else NULL,
    outcome_levels = if (task$type == "classification") levels(train[[task$outcome]]) else NULL,
    xtx_condition = {
      X <- as.matrix(baked_train[, pred_cols, drop = FALSE]); X <- cbind(1, X)
      kappa(crossprod(X), exact = TRUE)
    }
  )
  test_metrics_fn <- function(weighted) {
    ms <- do.call(metric_set, unname(metric_fns[unlist(task$metrics)]))
    if (task$type == "regression") {
      d <- data.frame(truth = test[[task$outcome]], estimate = fit_info$predicted_test)
      if (!is.null(wcol)) d[[wcol]] <- test[[wcol]]
      m <- ms_eval(ms, d, truth = truth, estimate = estimate, weighted = weighted)
    } else {
      probs <- predict(fit, test, type = "prob"); cls <- predict(fit, test)$.pred_class
      if (n_lv > 2) {
        d <- cbind(data.frame(truth = test[[task$outcome]], estimate = cls), as.data.frame(probs))
        if (!is.null(wcol)) d[[wcol]] <- test[[wcol]]
        m <- ms_eval(ms, d, truth = truth, estimate = estimate, tidyselect::starts_with(".pred_"), weighted = weighted)
      } else {
        d <- data.frame(truth = test[[task$outcome]], estimate = cls, prob = probs[[1]])
        if (!is.null(wcol)) d[[wcol]] <- test[[wcol]]
        m <- ms_eval(ms, d, truth = truth, estimate = estimate, prob, weighted = weighted)
      }
    }
    setNames(as.list(m$.estimate), m$.metric)
  }
  test_metrics <- test_metrics_fn(FALSE)
  # the same metrics with `case_weights =` (frequency weights only; NULL otherwise)
  test_metrics_weighted <- if (!is.null(metric_weights)) test_metrics_fn(TRUE) else NULL

  # ---- Level 4: resampling with shared folds (skipped for stochastic forests) ----
  if (is_rf) {
    out <- list(task = task, versions = versions, n_train = nrow(train), n_test = nrow(test),
      prep = list(steps = step_states,
                  baked_train = columns_as_list(baked_train[, pred_cols, drop = FALSE]),
                  baked_test = columns_as_list(baked_test[, pred_cols, drop = FALSE]),
                  predictor_names = pred_cols),
      fit = fit_info, test_metrics = test_metrics, test_metrics_weighted = test_metrics_weighted, resample = NULL)
    write_json(out, file.path(out_dir, paste0(task$id, ".json")), auto_unbox = TRUE, digits = I(17), na = "null", null = "null", pretty = FALSE)
    cat("   wrote", file.path(out_dir, paste0(task$id, ".json")), "(forest: seed sweep, no resampling)\n")
    return(invisible(NULL))
  }
  # ---- the shared rset (ADR-0013): memberships from the split file, ids as rsample names them ----
  rs <- split$rset
  splits <- Map(function(a, s) make_splits(list(analysis = as.integer(unlist(a)) + 1L, assessment = as.integer(unlist(s)) + 1L), data = df),
                rs$analysis, rs$assessment)
  rset <- manual_rset(unname(splits), unlist(rs$ids))
  # what rsample's own function produces for the same scheme: the ids always (Rust
  # must name resamples identically) and, for the deterministic time-series schemes,
  # the memberships too (which must equal the shared file exactly)
  p <- rs$params
  rs_or <- function(x, d) if (is.null(x)) d else x
  # classification tasks stratify by the outcome, as the shared file does
  with_strata <- function(fn, ...) if (task$type == "classification") fn(df, ..., strata = all_of(task$outcome)) else fn(df, ...)
  native <- switch(rs$scheme,
    vfold_cv = with_strata(vfold_cv, v = p$v, repeats = rs_or(p$repeats, 1)),
    bootstraps = with_strata(bootstraps, times = p$times),
    mc_cv = with_strata(mc_cv, prop = p$prop, times = p$times),
    group_vfold_cv = group_vfold_cv(raw, group = p$group, v = p$v, repeats = rs_or(p$repeats, 1), balance = rs_or(p$balance, "groups")),
    group_bootstraps = group_bootstraps(raw, group = p$group, times = p$times),
    group_mc_cv = group_mc_cv(raw, group = p$group, prop = p$prop, times = p$times),
    rolling_origin = rolling_origin(df, initial = p$initial, assess = p$assess, cumulative = rs_or(p$cumulative, TRUE), skip = rs_or(p$skip, 0), lag = rs_or(p$lag, 0)),
    sliding_window = sliding_window(df, lookback = rs_or(p$lookback, 0), assess_start = rs_or(p$assess_start, 1), assess_stop = rs_or(p$assess_stop, 1),
                                    complete = rs_or(p$complete, TRUE), step = rs_or(p$step, 1), skip = rs_or(p$skip, 0)),
    sliding_index = sliding_index(raw, index = !!rlang::sym(p$index), lookback = rs_or(p$lookback, 0), assess_start = rs_or(p$assess_start, 1),
                                  assess_stop = rs_or(p$assess_stop, 1), complete = rs_or(p$complete, TRUE), step = rs_or(p$step, 1), skip = rs_or(p$skip, 0)),
    stop("unknown resampling scheme ", rs$scheme))
  native_ids <- if ("id2" %in% names(native)) paste(native$id, native$id2, sep = ".") else native$id
  stopifnot(identical(native_ids, unlist(rs$ids)))
  deterministic <- rs$scheme %in% c("rolling_origin", "sliding_window", "sliding_index")
  native_members <- if (deterministic) {
    an <- lapply(native$splits, function(x) as.integer(x$in_id) - 1L)
    as <- lapply(native$splits, function(x) as.integer(rsample::complement(x)) - 1L)
    stopifnot(identical(an, lapply(rs$analysis, function(a) as.integer(unlist(a)))),
              identical(as, lapply(rs$assessment, function(a) as.integer(unlist(a)))))
    list(analysis = an, assessment = as)
  } else NULL
  rsample_native <- list(scheme = rs$scheme, ids = native_ids, members = native_members)
  tree_frame <- function(e) {
    fr <- e$frame; nn <- nrow(fr); leaf <- fr$var == "<leaf>"
    idx <- 1L + c(0L, cumsum((!leaf) + fr$ncompete + fr$nsurrogate))[-(nn + 1L)]; idx[leaf] <- NA
    list(node = as.integer(row.names(fr)), var = as.character(fr$var), n = fr$n, dev = fr$dev,
         split_point = if (is.null(e$splits)) NULL else ifelse(is.na(idx), NA, e$splits[idx, "index"]),
         split_improve = if (is.null(e$splits)) NULL else ifelse(is.na(idx), NA, e$splits[idx, "improve"]))
  }
  # fit() applies workflows' sparsity toggle but tune's fit_resamples() does not
  # (tune 2.1.0), so a fold model from fit() can differ from the same fold inside
  # fit_resamples(); the fold boosters below come from fit() and carry their own metrics
  fold_boosts <- if (is_boost) lapply(splits, function(s) {
    f <- fit(wf, data = analysis(s)); a <- assessment(s)
    ms_f <- do.call(metric_set, unname(metric_fns[unlist(task$metrics)]))
    if (task$type == "regression") {
      p <- as.numeric(predict(f, a)$.pred)
      d <- data.frame(truth = a[[task$outcome]], estimate = p)
      if (!is.null(wcol)) d[[wcol]] <- a[[wcol]]
      m <- ms_eval(ms_f, d, truth = truth, estimate = estimate, weighted = TRUE)
      list(booster = boost_dump(extract_fit_engine(f)), assess_pred = p,
           assess_metrics = setNames(as.list(m$.estimate), m$.metric))
    } else {
      probs <- predict(f, a, type = "prob"); cls <- predict(f, a)$.pred_class
      d <- cbind(data.frame(truth = a[[task$outcome]], estimate = cls), as.data.frame(probs))
      if (!is.null(wcol)) d[[wcol]] <- a[[wcol]]
      m <- if (n_lv > 2) ms_eval(ms_f, d, truth = truth, estimate = estimate, tidyselect::starts_with(".pred_"), weighted = TRUE)
           else ms_eval(ms_f, d, truth = truth, estimate = estimate, !!rlang::sym(names(probs)[1]), weighted = TRUE)
      list(booster = boost_dump(extract_fit_engine(f)), assess_prob = unname(as.matrix(probs)),
           assess_class = as.character(cls), assess_metrics = setNames(as.list(m$.estimate), m$.metric))
    }
  }) else NULL
  fold_trees <- if (is_tree) lapply(splits, function(s) {
    f <- fit(wf, data = analysis(s)); a <- assessment(s)
    c(tree_frame(extract_fit_engine(f)),
      if (task$type == "regression") list(assess_pred = as.numeric(predict(f, a)$.pred))
      else list(assess_prob = unname(as.matrix(predict(f, a, type = "prob"))),
                assess_class = as.character(predict(f, a)$.pred_class)))
  }) else NULL
  ms <- do.call(metric_set, unname(metric_fns[unlist(task$metrics)]))
  res <- fit_resamples(wf, resamples = rset, metrics = ms)
  per_fold <- collect_metrics(res, summarize = FALSE)
  per_fold_list <- lapply(split(per_fold, per_fold$id), function(d) setNames(as.list(d$.estimate), d$.metric))
  summ <- collect_metrics(res)
  summary_list <- lapply(seq_len(nrow(summ)), function(i) list(metric = summ$.metric[i], mean = summ$mean[i], std_err = summ$std_err[i], n = summ$n[i]))

  # ---- tuning (ADR-0014): tune_grid on the shared rset with an explicit grid, the
  # selection rules, finalize_workflow + last_fit on the initial split, fit_best ----
  tuning_info <- if (!is.null(task$tuning)) {
    tg <- task$tuning
    pnames <- names(tg$grid[[1]])
    grid <- tibble::as_tibble(setNames(lapply(pnames, function(nm) sapply(tg$grid, function(g) as.numeric(g[[nm]]))), pnames))
    for (nm in intersect(pnames, c("min_n", "tree_depth", "trees", "mtry"))) grid[[nm]] <- as.integer(grid[[nm]])
    fixed <- if (is.null(task$model_args)) list() else task$model_args[setdiff(names(task$model_args), pnames)]
    targs <- c(setNames(lapply(pnames, function(x) tune()), pnames), fixed)
    model_t <- if (is_tree) do.call(decision_tree, targs) %>% set_engine("rpart") %>% set_mode(task$type) else
               if (is_boost) do.call(boost_tree, targs) %>% set_engine("xgboost") %>% set_mode(task$type) else
               stop("tuning tasks need a decision_tree or boost_tree model")
    wf_t <- make_wf(model_t)
    # a *named* metric set: show_best()/select_best() look the metric up by name
    ms_t <- do.call(metric_set, metric_fns[unlist(task$metrics)])
    tres <- tune_grid(wf_t, resamples = rset, grid = grid, metrics = ms_t, control = control_grid(save_workflow = TRUE))
    cm <- collect_metrics(tres)
    raw <- collect_metrics(tres, summarize = FALSE)
    key <- dplyr::distinct(cm[, c(pnames, ".config")])
    row_cfg <- dplyr::left_join(dplyr::mutate(grid, .row = dplyr::row_number()), key, by = pnames)
    stopifnot(!anyNA(row_cfg$.config), nrow(row_cfg) == nrow(grid), !anyDuplicated(row_cfg$.config))
    candidates <- lapply(seq_len(nrow(grid)), function(i) {
      cfg <- row_cfg$.config[i]
      sm <- cm[cm$.config == cfg, ]
      pf <- raw[raw$.config == cfg, ]
      list(config = cfg, params = as.list(grid[i, pnames]),
           summary = lapply(split(sm, sm$.metric), function(d) list(mean = d$mean, n = d$n, std_err = d$std_err)),
           per_fold = lapply(split(pf, pf$id), function(d) setNames(as.list(d$.estimate), d$.metric)))
    })
    # the same candidates evaluated one by one (fit_resamples on the finalized
    # workflow): what tune_grid computes unless it predicts sub-models from a larger
    # fit (boost_tree `trees`), where parsnip 1.5.0's xgb_by_tree() passes
    # iterationrange = c(1, trees + 1) to xgboost >= 2.0, whose range is inclusive at
    # both ends, i.e. one tree too many for every non-maximal `trees` candidate
    is_sub <- if (is_boost && "trees" %in% pnames) {
      others <- setdiff(pnames, "trees")
      key <- if (length(others)) do.call(paste, c(grid[others], sep = "|")) else rep("", nrow(grid))
      grid$trees < stats::ave(grid$trees, key, FUN = max)
    } else rep(FALSE, nrow(grid))
    direct <- lapply(seq_len(nrow(grid)), function(i) {
      wf_i <- finalize_workflow(wf_t, grid[i, pnames])
      r_i <- fit_resamples(wf_i, resamples = rset, metrics = ms_t)
      pf <- collect_metrics(r_i, summarize = FALSE); sm <- collect_metrics(r_i)
      list(summary = lapply(split(sm, sm$.metric), function(d) list(mean = d$mean, n = d$n, std_err = d$std_err)),
           per_fold = lapply(split(pf, pf$id), function(d) setNames(as.list(d$.estimate), d$.metric)))
    })
    for (i in seq_along(candidates)) { candidates[[i]]$direct <- direct[[i]]; candidates[[i]]$submodel <- is_sub[i] }
    cfg_index <- function(sel) match(sel$.config, row_cfg$.config) - 1L
    sorts <- lapply(unlist(tg$sort), function(sv) {
      m <- regmatches(sv, regexec("^desc[(](.*)[)]$", sv))[[1]]
      if (length(m)) rlang::quo(dplyr::desc(!!rlang::sym(m[2]))) else rlang::quo(!!rlang::sym(sv))
    })
    limit <- if (is.null(tg$limit)) 2 else tg$limit
    sb <- show_best(tres, metric = tg$metric, n = nrow(grid))
    best <- select_best(tres, metric = tg$metric)
    one_se <- select_by_one_std_err(tres, !!!sorts, metric = tg$metric)
    pct <- select_by_pct_loss(tres, !!!sorts, metric = tg$metric, limit = limit)
    final_wf <- finalize_workflow(wf_t, best)
    rs_init <- make_splits(list(analysis = as.integer(split$initial$train) + 1L, assessment = as.integer(split$initial$test) + 1L), data = df)
    lf <- last_fit(final_wf, split = rs_init, metrics = ms_t)
    lf_m <- collect_metrics(lf)
    lf_p <- collect_predictions(lf); lf_p <- lf_p[order(lf_p$.row), ]
    prob_cols <- function(d) setdiff(grep("^[.]pred_", names(d), value = TRUE), ".pred_class")
    pred_dump <- function(d) if (task$type == "regression") list(pred = as.numeric(d$.pred)) else
      list(class = as.character(d$.pred_class), prob = unname(as.matrix(d[, prob_cols(d), drop = FALSE])))
    model_dump <- function(f) if (is_tree) tree_frame(extract_fit_engine(f)) else boost_dump(extract_fit_engine(f))
    fb <- fit_best(tres, metric = tg$metric)
    fb_pred <- if (task$type == "regression") predict(fb, test) else cbind(predict(fb, test), predict(fb, test, type = "prob"))
    list(grid = lapply(seq_len(nrow(grid)), function(i) as.list(grid[i, pnames])), configs = row_cfg$.config, candidates = candidates,
         show_best = cfg_index(sb),
         select_best = list(index = cfg_index(best), params = as.list(best[pnames])),
         select_by_one_std_err = list(index = cfg_index(one_se), params = as.list(one_se[pnames])),
         select_by_pct_loss = list(index = cfg_index(pct), params = as.list(pct[pnames])),
         last_fit = list(metrics = setNames(as.list(lf_m$.estimate), lf_m$.metric), rows = as.integer(lf_p$.row) - 1L,
                         predictions = pred_dump(lf_p), model = model_dump(lf$.workflow[[1]])),
         fit_best = list(predictions = pred_dump(fb_pred), model = model_dump(fb)))
  } else NULL

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
    test_metrics_weighted = test_metrics_weighted,
    resample = list(ids = names(per_fold_list), per_fold = per_fold_list, summary = summary_list, fold_trees = fold_trees, fold_boosts = fold_boosts,
                    rsample_native = rsample_native),
    tuning = tuning_info
  )
  write_json(out, file.path(out_dir, paste0(task$id, ".json")), auto_unbox = TRUE, digits = I(17), na = "null", null = "null", pretty = FALSE)
  cat("   wrote", file.path(out_dir, paste0(task$id, ".json")), "\n")
}

only <- Sys.getenv("TRM_TASKS")  # optional comma-separated task ids to regenerate
if (nzchar(only)) tasks <- Filter(function(t) t$id %in% strsplit(only, ",")[[1]], tasks)
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
# ---- Level 2 (Phase 8b): class and probability metrics on fixed arrays ----
# Curves are stored without their infinite end-point thresholds (jsonlite cannot write Inf).
cls_metrics_binary <- function(truth, est, prob) {
  lv <- levels(truth)
  d <- data.frame(t = truth, e = est, p = prob)
  rc <- roc_curve(d, t, p); pc <- pr_curve(d, t, p)
  suppressWarnings(list(
    levels = lv, truth = as.character(truth), estimate = as.character(est), prob_event = prob,
    sens = sens_vec(truth, est), spec = spec_vec(truth, est), precision = precision_vec(truth, est),
    recall = recall_vec(truth, est), f_meas = f_meas_vec(truth, est), f_meas_beta2 = f_meas_vec(truth, est, beta = 2),
    bal_accuracy = bal_accuracy_vec(truth, est), j_index = j_index_vec(truth, est),
    kap = kap_vec(truth, est), mcc = mcc_vec(truth, est), accuracy = accuracy_vec(truth, est),
    roc_auc = roc_auc_vec(truth, prob), pr_auc = pr_auc_vec(truth, prob),
    average_precision = average_precision_vec(truth, prob),
    brier_class = brier_class_vec(truth, prob), mn_log_loss = mn_log_loss_vec(truth, prob),
    conf_mat = unclass(conf_mat(d, t, e)$table),
    roc_curve = list(threshold = rc$.threshold[is.finite(rc$.threshold)], specificity = rc$specificity, sensitivity = rc$sensitivity),
    pr_curve = list(threshold = pc$.threshold[is.finite(pc$.threshold)], recall = pc$recall, precision = pc$precision)
  ))
}
cls_metrics_multi <- function(truth, est, probs) {
  ests <- c("macro", "macro_weighted", "micro")
  per <- function(f) setNames(lapply(ests, function(e) f(truth, est, estimator = e)), ests)
  pm <- as.matrix(probs)
  d <- data.frame(t = truth, e = est)
  suppressWarnings(list(
    levels = levels(truth), truth = as.character(truth), estimate = as.character(est), probs = unname(pm),
    sens = per(sens_vec), spec = per(spec_vec), precision = per(precision_vec), recall = per(recall_vec),
    f_meas = per(f_meas_vec), bal_accuracy = per(bal_accuracy_vec), j_index = per(j_index_vec),
    kap = kap_vec(truth, est), kap_linear = kap_vec(truth, est, weighting = "linear"),
    kap_quadratic = kap_vec(truth, est, weighting = "quadratic"),
    mcc = mcc_vec(truth, est), accuracy = accuracy_vec(truth, est),
    roc_auc = list(hand_till = roc_auc_vec(truth, pm), macro = roc_auc_vec(truth, pm, estimator = "macro"),
                   macro_weighted = roc_auc_vec(truth, pm, estimator = "macro_weighted")),
    pr_auc = list(macro = pr_auc_vec(truth, pm), macro_weighted = pr_auc_vec(truth, pm, estimator = "macro_weighted")),
    average_precision = list(macro = average_precision_vec(truth, pm),
                             macro_weighted = average_precision_vec(truth, pm, estimator = "macro_weighted")),
    brier_class = brier_class_vec(truth, pm), mn_log_loss = mn_log_loss_vec(truth, pm),
    conf_mat = unclass(conf_mat(d, t, e)$table)
  ))
}
fixed$binary_full <- cls_metrics_binary(truth_cls, est_cls, prob_a)
# ties in the probabilities straddling both classes, and a tie group at the threshold
tt <- factor(c("a", "a", "b", "a", "b", "b", "a", "b", "a", "b"), levels = lv)
tp <- c(0.9, 0.7, 0.7, 0.5, 0.5, 0.5, 0.3, 0.3, 0.1, 0.8)
te <- factor(ifelse(tp >= 0.5, "a", "b"), levels = lv)
fixed$binary_ties <- cls_metrics_binary(tt, te, tp)
# three classes; `z` is never predicted (undefined precision for it) and the probability
# columns contain ties
lv3 <- c("x", "y", "z")
mt <- factor(c("x", "x", "x", "x", "y", "y", "y", "y", "z", "z", "z", "z"), levels = lv3)
me <- factor(c("x", "x", "y", "x", "y", "y", "x", "y", "y", "x", "y", "x"), levels = lv3)
mp <- data.frame(
  x = c(0.7, 0.6, 0.2, 0.5, 0.2, 0.1, 0.5, 0.3, 0.2, 0.5, 0.1, 0.4),
  y = c(0.2, 0.3, 0.6, 0.3, 0.6, 0.8, 0.3, 0.5, 0.5, 0.2, 0.6, 0.3),
  z = c(0.1, 0.1, 0.2, 0.2, 0.2, 0.1, 0.2, 0.2, 0.3, 0.3, 0.3, 0.3))
stopifnot(all(abs(rowSums(mp) - 1) < 1e-12))
fixed$multiclass <- cls_metrics_multi(mt, me, mp)
# level `z` absent from truth: Hand-Till drops it; macro recall is NA for it, so
# bal_accuracy (no na.rm) is NA while j_index (na.rm) is not
mt2 <- factor(c("x", "x", "y", "y", "x", "y"), levels = lv3)
me2 <- factor(c("x", "y", "y", "x", "x", "z"), levels = lv3)
mp2 <- data.frame(x = c(0.6, 0.4, 0.3, 0.5, 0.7, 0.2), y = c(0.3, 0.4, 0.5, 0.4, 0.2, 0.3), z = c(0.1, 0.2, 0.2, 0.1, 0.1, 0.5))
fixed$multiclass_absent_level <- cls_metrics_multi(mt2, me2, mp2)
fixed$undefined <- c(fixed$undefined, suppressWarnings(list(
  mcc_binary_degenerate = mcc_vec(factor(c("a", "b"), levels = lv), factor(c("a", "a"), levels = lv)),
  mcc_multiclass_degenerate = mcc_vec(factor(c("x", "y", "z"), levels = lv3), factor(c("x", "x", "x"), levels = lv3)),
  kap_no_chance_disagreement = kap_vec(factor(c("a", "a"), levels = lv), factor(c("a", "a"), levels = lv)),
  roc_auc_no_events = roc_auc_vec(factor(c("b", "b"), levels = lv), c(0.2, 0.3)),
  pr_auc_no_events = pr_auc_vec(factor(c("b", "b"), levels = lv), c(0.2, 0.3)),
  sens_no_events = sens_vec(factor(c("b", "b"), levels = lv), factor(c("a", "b"), levels = lv)),
  f_meas_both_zero = f_meas_vec(factor(c("a", "b"), levels = lv), factor(c("b", "a"), levels = lv))
)))
write_json(fixed, file.path(out_dir, "metrics_fixed.json"), auto_unbox = TRUE, digits = I(17), na = "null", null = "null", pretty = TRUE)
cat("wrote metrics_fixed.json\n")
