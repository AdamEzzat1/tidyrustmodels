"""End-to-end example from Python: recipe, model, resampling, tuning on Palmer penguins.

Run from the repository root after `maturin develop --release`:
    python examples/penguins_workflow.py
"""
import pandas as pd

import tidyrustmodels as trm

df = pd.read_csv("reference/fixtures/datasets/penguins_sex.csv", keep_default_na=False, na_values=[""])
cols = ["sex", "species", "bill_length_mm", "bill_depth_mm", "flipper_length_mm", "body_mass_g"]
data = trm.from_pandas(df[cols])
print(data)

# Stratified 75/25 split, reproducible from the seed.
train_rows, test_rows = trm.initial_split(data.nrows, 0.75, seed=42, strata=list(df["sex"]))
train, test = data.select_rows(train_rows), data.select_rows(test_rows)

recipe = (
    trm.Recipe("sex")
    .step_impute_median("all_numeric_predictors")
    .step_normalize("all_numeric_predictors")
    .step_dummy("all_nominal_predictors")
)
wf = trm.Workflow(recipe, trm.LogisticReg(engine="linfa"))
fitted = wf.fit(train)
print("features:", fitted.feature_names())
print("coefficients:", {k: round(v, 4) for k, v in fitted.coefficients().items()})
print("diagnostics:", fitted.diagnostics())
print("backend:", fitted.backend())

proba = fitted.predict_proba(test)          # columns ordered as levels: ["female", "male"]
classes = fitted.predict(test)
truth = list(df["sex"].iloc[test_rows])
print("test accuracy:", sum(c == t for c, t in zip(classes, truth)) / len(truth))
print("P(female) for first 5 test rows:", proba[:5, 0].round(3))

folds = trm.vfold_cv(data.nrows, 5, seed=7, strata=list(df["sex"]))
cv = wf.fit_resamples(data, folds, ["accuracy", "mn_log_loss"], threads=4)
for s in cv["summary"]:
    print(f"cv {s['metric']:<12} mean {s['mean']:.4f} std_err {s['std_err']:.4f}")

grid = trm.grid_regular([("penalty", -4.0, 0.0, True)], 5)
tuned = trm.tune_grid(wf, data, folds, grid, ["accuracy", "mn_log_loss"], threads=4)
best = min(tuned, key=lambda c: c["results"]["summary"][1]["mean"])
print("best penalty by mn_log_loss:", best["params"], round(best["results"]["summary"][1]["mean"], 4))

text = fitted.to_json()
again = trm.FittedWorkflow.from_json(text)
assert (again.predict_proba(test) == proba).all()
print("serialized workflow:", len(text), "bytes; round-trip identical")
