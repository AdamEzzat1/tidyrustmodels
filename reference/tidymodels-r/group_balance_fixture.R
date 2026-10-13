# Fixture for rsample 1.3.2's grouped-resampling balance rules (ADR-0013, Phase 8h).
#
# rsample shuffles the groups with R's RNG, which Rust cannot reproduce, but the rules
# applied *after* the shuffle are deterministic: `balance_observations_helper` (greedy
# fold filling) and `balance_prop_helper` (cumulative-share cut-off). For each case this
# script records the permutation / draws that R's RNG produced under a seed (replayed
# here with the same `sample.int` call under the same seed, which is the helper's first
# and only RNG call) together with the assignment rsample derived from it, so the Rust
# port can be checked on the identical input.
#
# Run from the repository root: Rscript reference/tidymodels-r/group_balance_fixture.R
.libPaths(c(Sys.getenv("R_LIBS_USER"), .libPaths()))
suppressPackageStartupMessages({ library(rsample); library(jsonlite); library(vctrs) })
stopifnot(packageVersion("rsample") == "1.3.2")

cases <- list(
  list(name = "six_groups", sizes = c(6, 5, 4, 2, 2, 1), v = 3, prop = 0.5, seed = 11),
  list(name = "ten_groups", sizes = c(9, 1, 7, 3, 3, 8, 2, 5, 4, 6), v = 4, prop = 0.6, seed = 23),
  list(name = "equal_groups", sizes = rep(4, 8), v = 3, prop = 0.4, seed = 37),
  list(name = "skewed_groups", sizes = c(30, 1, 1, 2, 1, 3, 1, 1), v = 3, prop = 0.75, seed = 41)
)

out <- list(rsample = as.character(packageVersion("rsample")), cases = list())
for (cs in cases) {
  groups <- rep(seq_along(cs$sizes) - 1L, times = cs$sizes)   # 0-based codes, first-appearance order
  data_ind <- tibble::tibble(..index = seq_along(groups), ..group = as.character(groups))
  n_obs <- nrow(data_ind)
  freq_table <- vec_count(data_ind$..group, sort = "location")
  # balance_observations_helper: its only RNG call is sample.int(nrow(freq_table))
  set.seed(cs$seed); perm <- sample.int(nrow(freq_table))
  set.seed(cs$seed)
  obs <- rsample:::balance_observations_helper(data_ind, v = cs$v, target_per_fold = 1 / cs$v)
  obs_assign <- obs$assignment[match(freq_table$key, obs$key)] - 1L   # per table row, 0-based fold
  # balance_prop_helper without replacement: sample.int(n_groups, n_groups)
  set.seed(cs$seed); draws <- sample.int(nrow(freq_table), nrow(freq_table), replace = FALSE)
  set.seed(cs$seed)
  pr <- rsample:::balance_prop_helper(prop = cs$prop, data_ind = data_ind, v = 1, replace = FALSE)
  kept <- match(pr$key, freq_table$key) - 1L
  stopifnot(identical(kept, draws[seq_along(kept)] - 1L))
  # with replacement (group bootstraps, prop = 1): ceiling(n_groups * sum / min) draws
  n_draw <- ceiling(nrow(freq_table) * 1 * sum(freq_table$count) / min(freq_table$count))
  set.seed(cs$seed); draws_rep <- sample.int(nrow(freq_table), n_draw, replace = TRUE)
  set.seed(cs$seed)
  pb <- rsample:::balance_prop_helper(prop = 1, data_ind = data_ind, v = 1, replace = TRUE)
  kept_rep <- match(pb$key, freq_table$key) - 1L
  stopifnot(identical(kept_rep, draws_rep[seq_along(kept_rep)] - 1L))
  out$cases[[cs$name]] <- list(
    sizes = cs$sizes, v = cs$v, prop = cs$prop, n_obs = n_obs,
    observations = list(perm = perm - 1L, assignment = obs_assign),
    prop_without_replacement = list(draws = draws - 1L, kept = kept),
    prop_with_replacement = list(n_draw = n_draw, draws = draws_rep - 1L, kept = kept_rep)
  )
  cat(cs$name, ": obs folds", obs_assign, "| prop kept", kept, "| boot kept", length(kept_rep), "of", n_draw, "draws\n")
}
write_json(out, "reference/fixtures/r/group_balance.json", auto_unbox = TRUE, pretty = TRUE)
cat("wrote reference/fixtures/r/group_balance.json\n")
