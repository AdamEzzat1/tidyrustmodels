"""Comparing preprocessors x models with a workflow set from Python (ADR-0016).

Run from the repository root after `maturin develop --release`:
    python examples/ames_workflow_sets.py
"""
import pandas as pd

import tidyrustmodels as trm

cols = ["Sale_Price", "Gr_Liv_Area", "Year_Built", "Total_Bsmt_SF", "Bldg_Type", "Neighborhood"]
df = pd.read_csv("reference/fixtures/datasets/ames.csv", keep_default_na=False, na_values=[""])[cols]
data = trm.from_pandas(df)
folds = trm.vfold_cv(data.nrows, 5, seed=3)

plain = trm.Recipe("Sale_Price").step_dummy("all_nominal_predictors")
pooled = (
    trm.Recipe("Sale_Price")
    .step_other("all_nominal_predictors", threshold=0.05)
    .step_normalize("all_numeric_predictors")
    .step_dummy("all_nominal_predictors")
)
ws = trm.workflow_set(
    [("plain", plain), ("pooled", pooled)],
    [("lm", trm.LinearReg()), ("tree", trm.DecisionTree()), ("xgb", trm.BoostTree(trees=50))],
)
print("workflows:", ws.ids)
for wid in ("plain_tree", "pooled_tree"):
    ws.option_grid(wid, [{"cost_complexity": cp} for cp in (1e-4, 1e-3, 1e-2)])
ws.workflow_map(data, folds, ["rmse", "rsq"], threads=4)

ranked = ws.rank_results("rmse", select_best=True)
print(pd.DataFrame(ranked)[["wflow_id", "config", "metric", "mean", "std_err", "rank"]].to_string(index=False))

print("winner:", ws.best("rmse"))
fitted = ws.fit_best(data, "rmse")
pred = fitted.predict(data)
print("refitted on", len(pred), "rows; first prediction", round(float(pred[0])))
