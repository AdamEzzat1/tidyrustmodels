"""Tuning a decision tree on the Ames housing data from Python: tune_grid, tune's
selection rules, finalize + last_fit, fit_best (ADR-0014).

Run from the repository root after `maturin develop --release`:
    python examples/ames_tuning.py
"""
import pandas as pd

import tidyrustmodels as trm

cols = ["Sale_Price", "Gr_Liv_Area", "Year_Built", "Total_Bsmt_SF", "Bldg_Type", "Neighborhood"]
df = pd.read_csv("reference/fixtures/datasets/ames.csv", keep_default_na=False, na_values=[""])[cols]
data = trm.from_pandas(df)
print(data)

train_rows, test_rows = trm.initial_split(data.nrows, 0.75, seed=2024)
folds = trm.vfold_cv(data.nrows, 5, seed=11, repeats=2)

recipe = trm.Recipe("Sale_Price").step_other("all_nominal_predictors", threshold=0.02).step_dummy("all_nominal_predictors")
wf = trm.Workflow(recipe, trm.DecisionTree())

grid = [{"cost_complexity": cp, "min_n": mn} for cp in (1e-4, 1e-3, 1e-2) for mn in (5, 20, 50)]
metrics = ["rmse", "rsq"]
tuned = trm.tune_grid(wf, data, folds, grid, metrics, threads=4)

print("show_best(rmse):")
for row in tuned.show_best("rmse", n=3):
    print(f"  config {row['config']:>2}  {row['params']}  rmse {row['mean']:.0f} (se {row['std_err']:.0f}, n={row['n']})")

best = tuned.select_best("rmse")
simplest = tuned.select_by_one_std_err("rmse", ["desc(cost_complexity)", "desc(min_n)"])
print("select_best ->", best["params"])
print("select_by_one_std_err ->", simplest["params"])

last = wf.finalize(simplest["params"]).last_fit(data, train_rows, test_rows, metrics)
print("last_fit metrics:", {k: round(v, 4) for k, v in last["metrics"].items()})
print("last_fit predictions:", len(last["predictions"]), "assessment rows")
print("tree note:", last["workflow"].diagnostics()["note"])

final = tuned.fit_best(wf, data, "rmse")
pred = final.predict(data)
print("fit_best refitted on", len(pred), "rows; first prediction", round(float(pred[0])))
again = trm.FittedWorkflow.from_json(final.to_json())
assert (again.predict(data) == pred).all()
