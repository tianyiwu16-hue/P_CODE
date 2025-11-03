# Re-run a lighter-weight analysis to complete within time limits.
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.inspection import permutation_importance
import matplotlib.pyplot as plt
import time, os

# Load data
file_path ="D:\数据要素大赛作品\expanded_student_dataset_500k.csv"
df = pd.read_csv(file_path)

features = [
    "Study_Hours_Per_Day",
    "Extracurricular_Hours_Per_Day",
    "Sleep_Hours_Per_Day",
    "Social_Hours_Per_Day",
    "Physical_Activity_Hours_Per_Day"
]
target = "GPA"
X = df[features].copy()
y = df[target].copy()

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Use fewer trees for speed
rf = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=1)
rf.fit(X_train, y_train)
y_pred = rf.predict(X_test)

rmse_test = mean_squared_error(y_test, y_pred, squared=False)
mae_test = mean_absolute_error(y_test, y_pred)
r2_test = r2_score(y_test, y_pred)

metrics_test = pd.DataFrame({
    "Metric": ["RMSE", "MAE", "R2"],
    "Value": [rmse_test, mae_test, r2_test]
})
print("\nTest set metrics:\n", metrics_test.to_string(index=False))

# CV (5-fold)
kf = KFold(n_splits=5, shuffle=True, random_state=42)
neg_mse_cv = cross_val_score(rf, X, y, scoring="neg_mean_squared_error", cv=kf, n_jobs=1)
r2_cv = cross_val_score(rf, X, y, scoring="r2", cv=kf, n_jobs=1)
rmse_cv = np.sqrt(-neg_mse_cv)

cv_summary = pd.DataFrame({
    "Fold": np.arange(1, len(rmse_cv)+1),
    "RMSE": rmse_cv,
    "R2": r2_cv
})
print("\nCross-validation per-fold results:\n", cv_summary.to_string(index=False))
print("\nCross-validation summary:\n", pd.DataFrame({
    "Metric": ["RMSE_mean", "RMSE_std", "R2_mean", "R2_std"],
    "Value": [rmse_cv.mean(), rmse_cv.std(), r2_cv.mean(), r2_cv.std()]
}).to_string(index=False))

# Permutation importance (test set) with n_repeats=20
perm_imp = permutation_importance(rf, X_test, y_test, n_repeats=20, random_state=42, n_jobs=1)
imp_means = perm_imp.importances_mean
imp_stds = perm_imp.importances_std

imp_df = pd.DataFrame({
    "feature": features,
    "perm_importance_mean": imp_means,
    "perm_importance_std": imp_stds
}).sort_values("perm_importance_mean", ascending=False)
print("\nPermutation importance (test set):\n", imp_df.to_string(index=False))

# Hypothesis testing via target permutation (reduced permutations to 100)
n_permutations = 100
rng = np.random.RandomState(0)
null_importances = {f: [] for f in features}

start_time = time.time()
for i in range(n_permutations):
    y_shuffled = y.sample(frac=1.0, random_state=rng.randint(0, 1_000_000)).reset_index(drop=True)
    rf_null = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=1)
    rf_null.fit(X, y_shuffled)
    pi = permutation_importance(rf_null, X, y_shuffled, n_repeats=5, random_state=42, n_jobs=1)
    for feat, val in zip(features, pi.importances_mean):
        null_importances[feat].append(val)
    if (i+1) % 20 == 0:
        print(f"Completed {i+1}/{n_permutations} permutations, elapsed {time.time()-start_time:.1f}s")

p_values = {}
observed_imp = dict(zip(imp_df["feature"], imp_df["perm_importance_mean"]))
for feat in features:
    null_vals = np.array(null_importances[feat])
    p = (np.sum(null_vals >= observed_imp[feat]) + 1) / (len(null_vals) + 1)
    p_values[feat] = p

pvals_df = pd.DataFrame({
    "feature": features,
    "observed_importance": [observed_imp[f] for f in features],
    "p_value": [p_values[f] for f in features]
}).sort_values("observed_importance", ascending=False)
print("\nPermutation test p-values for feature importance:\n", pvals_df.to_string(index=False))

# Stability across seeds (n_seeds=5 for speed)
n_seeds = 5
seed_metrics = []
seed_importances = []

for seed in range(n_seeds):
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=seed)
    rf_s = RandomForestRegressor(n_estimators=100, random_state=seed, n_jobs=1)
    rf_s.fit(X_tr, y_tr)
    y_pr = rf_s.predict(X_te)
    seed_metrics.append({
        "seed": seed,
        "RMSE": mean_squared_error(y_te, y_pr, squared=False),
        "MAE": mean_absolute_error(y_te, y_pr),
        "R2": r2_score(y_te, y_pr)
    })
    pi_s = permutation_importance(rf_s, X_te, y_te, n_repeats=10, random_state=seed, n_jobs=1)
    seed_importances.append(pi_s.importances_mean)

seed_metrics_df = pd.DataFrame(seed_metrics)
seed_importances_df = pd.DataFrame(seed_importances, columns=features)

print("\nMetrics across random seeds:\n", seed_metrics_df.to_string(index=False))
print("\nPermutation importances across seeds:\n", seed_importances_df.to_string(index=False))

stability_metrics = seed_metrics_df.describe().loc[["mean", "std"]].T.reset_index().rename(columns={"index":"metric"})
stability_importance_stats = pd.DataFrame({
    "feature": features,
    "importance_mean_across_seeds": seed_importances_df.mean().values,
    "importance_std_across_seeds": seed_importances_df.std().values
}).sort_values("importance_mean_across_seeds", ascending=False)

print("\nStability metrics summary:\n", stability_metrics.to_string(index=False))
print("\nStability of feature importances:\n", stability_importance_stats.to_string(index=False))

# Plot actual vs predicted
import matplotlib.pyplot as plt
plt.figure(figsize=(5,5))
plt.scatter(y_test, y_pred, alpha=0.7)
minv = min(y_test.min(), y_pred.min())
maxv = max(y_test.max(), y_pred.max())
plt.plot([minv, maxv], [minv, maxv], linestyle='--')
plt.xlabel("Actual GPA")
plt.ylabel("Predicted GPA")
plt.title("Actual vs Predicted on Test Set")
plt.grid(True, linestyle='--', alpha=0.4)
plt.tight_layout()
plt.show()

# Save outputs
out_dir = "/mnt/data/rf_analysis_outputs"
os.makedirs(out_dir, exist_ok=True)
metrics_test.to_csv(f"{out_dir}/test_metrics.csv", index=False)
cv_summary.to_csv(f"{out_dir}/cv_fold_results.csv", index=False)
imp_df.to_csv(f"{out_dir}/permutation_importance.csv", index=False)
pvals_df.to_csv(f"{out_dir}/permutation_test_pvalues.csv", index=False)
seed_metrics_df.to_csv(f"{out_dir}/seed_metrics.csv", index=False)
seed_importances_df.to_csv(f"{out_dir}/seed_importances.csv", index=False)

print(f"\nOutputs saved to {out_dir}. You can download the CSV files from that folder.")

















# Final lightweight run to produce complete results quickly.
import pandas as pd, numpy as np, time, os
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.inspection import permutation_importance
import matplotlib.pyplot as plt

file_path = "D:\数据要素大赛作品\expanded_student_dataset_500k.csv"
df = pd.read_csv(file_path)

features = [
    "Study_Hours_Per_Day",
    "Extracurricular_Hours_Per_Day",
    "Sleep_Hours_Per_Day",
    "Social_Hours_Per_Day",
    "Physical_Activity_Hours_Per_Day"
]
target = "GPA"
X = df[features]
y = df[target]

# split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# smaller forest
rf = RandomForestRegressor(n_estimators=50, random_state=42, n_jobs=1)
rf.fit(X_train, y_train)
y_pred = rf.predict(X_test)

rmse_test = mean_squared_error(y_test, y_pred, squared=False)
mae_test = mean_absolute_error(y_test, y_pred)
r2_test = r2_score(y_test, y_pred)

print(f"Test RMSE: {rmse_test:.4f}, MAE: {mae_test:.4f}, R2: {r2_test:.4f}")

# CV (5-fold) quick
kf = KFold(n_splits=5, shuffle=True, random_state=42)
neg_mse_cv = cross_val_score(rf, X, y, scoring="neg_mean_squared_error", cv=kf, n_jobs=1)
r2_cv = cross_val_score(rf, X, y, scoring="r2", cv=kf, n_jobs=1)
rmse_cv = np.sqrt(-neg_mse_cv)
print("CV RMSE mean/std: {:.4f} / {:.4f}".format(rmse_cv.mean(), rmse_cv.std()))
print("CV R2 mean/std: {:.4f} / {:.4f}".format(r2_cv.mean(), r2_cv.std()))

# Permutation importance on test with n_repeats=10
perm_imp = permutation_importance(rf, X_test, y_test, n_repeats=10, random_state=42, n_jobs=1)
imp_means = perm_imp.importances_mean
imp_stds = perm_imp.importances_std

imp_df = pd.DataFrame({
    "feature": features,
    "perm_importance_mean": imp_means,
    "perm_importance_std": imp_stds
}).sort_values("perm_importance_mean", ascending=False)
print("\nPermutation importance (test set):\n", imp_df.to_string(index=False))

# Approximate hypothesis testing: null distribution by shuffling target 50 times (light)
n_perm = 50
rng = np.random.RandomState(0)
null_importances = {f: [] for f in features}
start = time.time()
for i in range(n_perm):
    y_shuf = y.sample(frac=1.0, random_state=rng.randint(0,1_000_000)).reset_index(drop=True)
    rf_null = RandomForestRegressor(n_estimators=50, random_state=42, n_jobs=1)
    rf_null.fit(X, y_shuf)
    pi = permutation_importance(rf_null, X, y_shuf, n_repeats=5, random_state=42, n_jobs=1)
    for feat, val in zip(features, pi.importances_mean):
        null_importances[feat].append(val)
    if (i+1)%10==0:
        print(f"  {i+1}/{n_perm} null permutations done, elapsed {time.time()-start:.1f}s")
        
observed_imp = dict(zip(imp_df["feature"], imp_df["perm_importance_mean"]))
p_values = {}
for feat in features:
    null_vals = np.array(null_importances[feat])
    p = (np.sum(null_vals >= observed_imp[feat]) + 1) / (len(null_vals) + 1)
    p_values[feat] = p

pvals_df = pd.DataFrame({
    "feature": features,
    "observed_importance": [observed_imp[f] for f in features],
    "p_value": [p_values[f] for f in features]
}).sort_values("observed_importance", ascending=False)
print("\nApproximate permutation-test p-values for feature importance:\n", pvals_df.to_string(index=False))

# Stability: 5 seeds
seed_imp = []
seed_metrics = []
for seed in range(5):
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=seed)
    rf_s = RandomForestRegressor(n_estimators=50, random_state=seed, n_jobs=1)
    rf_s.fit(X_tr, y_tr)
    ypr = rf_s.predict(X_te)
    seed_metrics.append([seed, mean_squared_error(y_te, ypr, squared=False), mean_absolute_error(y_te, ypr), r2_score(y_te, ypr)])
    pi = permutation_importance(rf_s, X_te, y_te, n_repeats=10, random_state=seed, n_jobs=1)
    seed_imp.append(pi.importances_mean)

seed_metrics_df = pd.DataFrame(seed_metrics, columns=["seed","RMSE","MAE","R2"])
seed_imp_df = pd.DataFrame(seed_imp, columns=features)

print("\nStability metrics across seeds (mean/std):")
print(seed_metrics_df.describe().loc[["mean","std"]].to_string())

print("\nFeature importance stability (mean/std across seeds):")
stable_df = pd.DataFrame({
    "feature": features,
    "importance_mean": seed_imp_df.mean().values,
    "importance_std": seed_imp_df.std().values
}).sort_values("importance_mean", ascending=False)
print(stable_df.to_string(index=False))

# Save outputs
out_dir = "/mnt/data/rf_analysis_outputs"
os.makedirs(out_dir, exist_ok=True)
pd.DataFrame([{"RMSE":rmse_test,"MAE":mae_test,"R2":r2_test}]).to_csv(f"{out_dir}/test_metrics.csv", index=False)
imp_df.to_csv(f"{out_dir}/permutation_importance.csv", index=False)
pvals_df.to_csv(f"{out_dir}/permutation_test_pvalues_approx.csv", index=False)
seed_metrics_df.to_csv(f"{out_dir}/seed_metrics.csv", index=False)
seed_imp_df.to_csv(f"{out_dir}/seed_importances.csv", index=False)

print(f"\nLightweight analysis complete. Outputs saved to {out_dir}")








# Quick OLS linear regression (for statistical significance / p-values)
import pandas as pd, statsmodels.api as sm
file_path = "D:\数据要素大赛作品\expanded_student_dataset_500k.csv"
df = pd.read_csv(file_path)

features = [
    "Study_Hours_Per_Day",
    "Extracurricular_Hours_Per_Day",
    "Sleep_Hours_Per_Day",
    "Social_Hours_Per_Day",
    "Physical_Activity_Hours_Per_Day"
]
X = df[features]
y = df["GPA"]

X_const = sm.add_constant(X)
model = sm.OLS(y, X_const).fit()
summary_df = pd.DataFrame({
    "coef": model.params,
    "std_err": model.bse,
    "t": model.tvalues,
    "p_value": model.pvalues,
    "conf_low": model.conf_int().iloc[:,0],
    "conf_high": model.conf_int().iloc[:,1]
})
print(model.summary())
print("\nCoefficients table:\n", summary_df.to_string())

df = pd.read_excel(file_path)

features = [
    "Study_Hours_Per_Day",
    "Extracurricular_Hours_Per_Day",
    "Sleep_Hours_Per_Day",
    "Social_Hours_Per_Day",
    "Physical_Activity_Hours_Per_Day"
]
X = df[features]
y = df["GPA"]

X_const = sm.add_constant(X)
model = sm.OLS(y, X_const).fit()
summary_df = pd.DataFrame({
    "coef": model.params,
    "std_err": model.bse,
    "t": model.tvalues,
    "p_value": model.pvalues,
    "conf_low": model.conf_int().iloc[:,0],
    "conf_high": model.conf_int().iloc[:,1]
})
print(model.summary())
print("\nCoefficients table:\n", summary_df.to_string())







import shap
import matplotlib.pyplot as plt
from sklearn.inspection import PartialDependenceDisplay
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
import pandas as pd, os, numpy as np

# Patch for deprecated np.bool in SHAP
if not hasattr(np, "bool"):
    np.bool = np.bool_

# Load dataset
file_path ="D:\数据要素大赛作品\expanded_student_dataset_500k.csv"
df = pd.read_csv(file_path)
features = [
    "Study_Hours_Per_Day",
    "Extracurricular_Hours_Per_Day",
    "Sleep_Hours_Per_Day",
    "Social_Hours_Per_Day",
    "Physical_Activity_Hours_Per_Day"
]
X = df[features]
y = df["GPA"]

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train RF model
rf = RandomForestRegressor(n_estimators=200, random_state=42)
rf.fit(X_train, y_train)

# ==== SHAP analysis ====
explainer = shap.TreeExplainer(rf)
shap_values = explainer.shap_values(X_test)

out_dir = "/mnt/data/rf_analysis_outputs"
os.makedirs(out_dir, exist_ok=True)

# SHAP summary plot
plt.figure(figsize=(8,6))
shap.summary_plot(shap_values, X_test, show=False)
plt.tight_layout()
shap_summary_path = f"{out_dir}/shap_summary.png"
plt.savefig(shap_summary_path, dpi=300)
plt.close()

# SHAP bar plot (mean absolute shap values)
plt.figure(figsize=(8,6))
shap.summary_plot(shap_values, X_test, plot_type="bar", show=False)
plt.tight_layout()
shap_bar_path = f"{out_dir}/shap_bar.png"
plt.savefig(shap_bar_path, dpi=300)
plt.close()

# ==== Partial Dependence Plots ====
pdp_paths = {}
for feat in features:
    fig, ax = plt.subplots(figsize=(6,4))
    PartialDependenceDisplay.from_estimator(rf, X_test, [feat], ax=ax)
    plt.tight_layout()
    path = f"{out_dir}/pdp_{feat}.png"
    plt.savefig(path, dpi=300)
    plt.close(fig)
    pdp_paths[feat] = path

(shap_summary_path, shap_bar_path, pdp_paths)























from matplotlib import pyplot as plt
import matplotlib.image as mpimg

img = mpimg.imread('/mnt/data/rf_analysis_outputs/pdp_Study_Hours_Per_Day.png')
imgplot = plt.imshow(img)
plt.show()




from matplotlib import pyplot as plt
import matplotlib.image as mpimg

img = mpimg.imread('/mnt/data/rf_analysis_outputs/pdp_Extracurricular_Hours_Per_Day.png')
imgplot = plt.imshow(img)
plt.show()




from matplotlib import pyplot as plt
import matplotlib.image as mpimg

img = mpimg.imread('/mnt/data/rf_analysis_outputs/pdp_Sleep_Hours_Per_Day.png')
imgplot = plt.imshow(img)
plt.show()




from matplotlib import pyplot as plt
import matplotlib.image as mpimg

img = mpimg.imread('/mnt/data/rf_analysis_outputs/pdp_Social_Hours_Per_Day.png')
imgplot = plt.imshow(img)
plt.show()




from matplotlib import pyplot as plt
import matplotlib.image as mpimg

img = mpimg.imread('/mnt/data/rf_analysis_outputs/pdp_Physical_Activity_Hours_Per_Day.png')
imgplot = plt.imshow(img)
plt.show()




from matplotlib import pyplot as plt
import matplotlib.image as mpimg

img = mpimg.imread('/mnt/data/rf_analysis_outputs/shap_summary.png')
imgplot = plt.imshow(img)
plt.show()




from matplotlib import pyplot as plt
import matplotlib.image as mpimg

img = mpimg.imread('/mnt/data/rf_analysis_outputs/shap_bar.png')
imgplot = plt.imshow(img)
plt.show()









import pandas as pd

# 修复路径和读取方式
file_path = r"D:\数据要素大赛作品\expanded_student_dataset_500k.csv"
df = pd.read_csv(file_path)

print(df.shape)
print(df.head())

# 检查目标变量和特征是否存在
features = [
    "Study_Hours_Per_Day",
    "Extracurricular_Hours_Per_Day",
    "Sleep_Hours_Per_Day",
    "Social_Hours_Per_Day",
    "Physical_Activity_Hours_Per_Day"
]
target = "GPA"

print("Features in data:", df[features].columns.tolist())
print("Target:", df[target].name)











