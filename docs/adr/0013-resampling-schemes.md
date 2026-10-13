# ADR-0013: Resampling schemes with rsample semantics

Date: 2026-10-13. Status: accepted.

## Context
Until now the resample crate offered `initial_split` and `vfold_cv` only, and a
`Split` was by construction a partition of the rows with both sides sorted. rsample
1.3.2 has a much wider contract, and the differences are not cosmetic:

- An `rsplit` holds `in_id` (analysis rows, in the order they were drawn, possibly
  with repeats) and `out_id` (assessment rows, or `NA` meaning "the complement").
  Bootstrap and Monte-Carlo draws are **unsorted** (`sample(n, replace = TRUE)`,
  `sample(n, size)`), while a v-fold analysis set is the ascending complement of the
  fold. Row order changes the last bits of a least-squares fit, so a port that
  sorted everything would not reproduce R's metrics bit for bit.
- A resample may assess **nothing** (`sliding_index` with a gap in the index, a
  bootstrap or grouped bootstrap that drew every row): rsample warns, `tune` cannot
  score it.
- Time-series windows leave rows in neither set; `apparent()` puts them in both.
- Ids follow `names0()`: zero-padded to the width of the largest number (`Fold1`
  … `Fold5`, `Fold01` … `Fold10`, `Bootstrap01` … `Bootstrap25`), with repeats as
  `id = Repeat1`, `id2 = Fold1`.
- Grouped resampling assigns **whole groups** with three balancing rules
  (`balance_groups_normal`, `balance_observations_helper`, `balance_prop_helper`)
  followed by `collapse_groups`, which also relabels the folds at random; the rows
  of a repeatedly drawn group are repeated **adjacently in original row order**
  (`left_join(multiple = "all")`), not in draw order.
- `make_strata` pools classes below `pool = 0.1` by assigning their rows to the
  surviving classes **at random**, caps numeric breaks at `floor(n / depth)`
  (depth 20) and sends missing values to a random bin.
- `tune` refuses `loo_cv` and nested resamples; `collect_metrics()` pools repeats
  (the summary `n` is the number of resamples, e.g. `v * repeats`).

R's RNG cannot be reproduced from Rust (ADR-0006), so "same algorithm" can never
mean "same draw"; the question is what *can* be verified.

## Decision
1. **`Split` is an rsplit.** `train` keeps rsample's order and may repeat rows;
   `test` is sorted and may be empty; rows may be in neither set. `Split::new`
   mirrors `make_splits(list(analysis, assessment))` (analysis non-empty, both in
   range); `Split::from_ids` keeps the stricter partition contract for the initial
   split. `Folds` is an rset: `splits`, rsample `ids` (repeats joined as
   `Repeat1.Fold1`) and `n`; `Folds::from_splits` is `manual_rset()`,
   `Folds::from_assignments` builds a v-fold rset from a fold index per row.
   `Workflow::fit_fold` refuses an empty assessment set instead of returning empty
   metrics.
2. **Every rsample generator is ported as an algorithm under the framework's seeded
   RNG:** `vfold_cv` (+ `repeats`), `bootstraps`, `mc_cv`, `loo_cv`,
   `initial_split`, `initial_validation_split`, `group_vfold_cv` (balance
   `groups` / `observations`, `v = None` = leave-one-group-out, repeats),
   `group_bootstraps`, `group_mc_cv`, `group_initial_split`, `rolling_origin`,
   `sliding_window`, `sliding_index` (numeric index), `initial_time_split`,
   `make_strata` (categorical pooling and numeric binning). Where rsample's only
   randomness is a permutation or a set of draws, the rule applied afterwards is a
   separate function (`balance_observations`, `balance_prop`) that takes the
   permutation / draws as input.
3. **Ids use `names0` everywhere**, including the shared fixture files; the earlier
   `sprintf("Fold%02d")` convention of the harness was a deviation and is gone.
4. **Verification is split by what can be compared** (VERIFICATION.md Phase 8h):
   - *Deterministic schemes* (`rolling_origin`, `sliding_window`, `sliding_index`,
     `initial_time_split`): rsample's own memberships are recorded in the fixture and
     the Rust generator must reproduce them **exactly**, as must the shared file.
   - *Random schemes*: the shared file (Python, ADR-0006) carries the memberships
     under `rset`; R builds the rset with `make_splits()` + `manual_rset()` and runs
     `fit_resamples()`, Rust loads the same rows through `Folds::from_splits`, and
     the per-resample and summary metrics must agree to round-off. The *ids* that
     rsample's own function produces are recorded and must equal the Rust ids; the
     structure of the shared memberships (balanced folds, out-of-bag assessment,
     groups never straddling a split, `floor(n * prop)` analysis rows, ascending
     repeated rows for grouped bootstraps) is asserted on both draws.
   - *Balance rules*: R records the permutation / draws its RNG produced under a
     seed (replayed with the same `sample.int` call) and the assignment rsample
     derived from them (`group_balance_fixture.R`); the Rust rules must agree
     exactly on that input.
5. **Not ported** (recorded in LIMITATIONS.md): `sliding_period` (needs a date
   index and lubridate's period arithmetic), `nested_cv`, `permutations`,
   `apparent`, `collapse_groups`' random relabelling of the folds (a permutation of
   ids with no effect on the set of resamples), `make_strata`'s warnings, and
   `tune`'s `Apparent` row.

## Consequences
- `Folds::assignments()` is only meaningful for partitions; callers that need one
  fold index per row should check `is_partition()`.
- The shared split files changed shape (`rset` replaces `vfold`) and every id, so all
  R fixtures were regenerated; nothing but ids and the generation timestamp changed
  for the pre-existing tasks.
- The reference task schema gains `resampling` (see `reference/tasks.json` notes)
  and two datasets (`synthetic_group_reg`, `synthetic_ts`).
- Python exposes the same generators (`bootstraps`, `mc_cv`, `loo_cv`,
  `group_vfold_cv`, `group_bootstraps`, `group_mc_cv`, `group_initial_split`,
  `rolling_origin`, `sliding_window`, `sliding_index`, `initial_time_split`,
  `initial_validation_split`, `make_strata`, `vfold_cv(repeats=)`,
  `Folds.from_splits`).
