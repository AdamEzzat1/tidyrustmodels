# Fit one random-forest task with a range of ranger seeds and write the per-seed
# metrics and test predictions to target/<task>_more_seeds.json, in the same shape as
# fit$rf$per_seed of the regular fixture (VERIFICATION.md Phase 8e follow-ups):
#   Rscript reference/tidymodels-r/rf_more_seeds.R penguins_species_rf 101 160
lines <- readLines("reference/tidymodels-r/run_reference.R")
stop_at <- grep("^for \\(task in tasks\\) run_task\\(task\\)", lines)
eval(parse(text = lines[seq_len(stop_at - 1)]))

args <- commandArgs(trailingOnly = TRUE)
task_id <- args[1]
seeds <- seq.int(as.integer(args[2]), as.integer(args[3]))
task <- Filter(function(t) t$id == task_id, tasks)[[1]]

df <- read_dataset(task$dataset, task)
split <- fromJSON(file.path(root, "reference", "fixtures", "splits", paste0(task$id, ".json")))
train <- df[split$initial$train + 1, , drop = FALSE]
test  <- df[split$initial$test + 1, , drop = FALSE]
rownames(train) <- NULL; rownames(test) <- NULL
f <- as.formula(paste(task$outcome, "~ ."))
rec <- add_steps(recipe(f, data = train), task$steps)
n_lv <- if (task$type == "classification") nlevels(train[[task$outcome]]) else 0L
ms <- do.call(metric_set, unname(metric_fns[unlist(task$metrics)]))
rf_model <- function(seed) {
  a <- if (is.null(task$model_args)) list() else task$model_args
  do.call(rand_forest, a) %>% set_engine("ranger", seed = seed, num.threads = 1) %>% set_mode(task$type)
}
per_seed <- lapply(seeds, function(s) {
  fs <- fit(workflow() %>% add_recipe(rec) %>% add_model(rf_model(s)), data = train)
  if (task$type == "regression") {
    p <- as.numeric(predict(fs, test)$.pred)
    m <- ms(data.frame(truth = test[[task$outcome]], estimate = p), truth = truth, estimate = estimate)
    list(metrics = setNames(as.list(m$.estimate), m$.metric), predicted_test = p)
  } else {
    probs <- predict(fs, test, type = "prob"); cls <- predict(fs, test)$.pred_class
    d <- cbind(data.frame(truth = test[[task$outcome]], estimate = cls), as.data.frame(probs))
    m <- if (n_lv > 2) ms(d, truth = truth, estimate = estimate, tidyselect::starts_with(".pred_"))
         else ms(d, truth = truth, estimate = estimate, !!rlang::sym(names(probs)[1]))
    list(metrics = setNames(as.list(m$.estimate), m$.metric), prob_test = unname(as.matrix(probs)))
  }
})
out <- file.path("target", paste0(task_id, "_more_seeds.json"))
write_json(list(seeds = seeds, per_seed = per_seed), out, auto_unbox = TRUE, digits = I(17), na = "null", null = "null")
cat("wrote", out, "\n")
