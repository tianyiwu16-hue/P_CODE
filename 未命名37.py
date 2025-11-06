import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import joblib
import matplotlib.pyplot as plt

# 读取数据
file_path = "D:\桌面应用\student_lifestyle_dataset(3).xlsx"
df = pd.read_excel(file_path)

# 特征与目标
FEATURES = [
    "Study_Hours_Per_Day",
    "Extracurricular_Hours_Per_Day",
    "Sleep_Hours_Per_Day",
    "Social_Hours_Per_Day",
    "Physical_Activity_Hours_Per_Day"
]
TARGET = "GPA"

X = df[FEATURES]
y = df[TARGET]

# 划分训练/测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 小规模网格搜索参数
param_grid = {
    "n_estimators": [100, 200],
    "max_depth": [5, 10, None],
    "min_samples_split": [2, 5],
    "min_samples_leaf": [1, 2],
    "max_features": ["sqrt", "log2"]
}

rf = RandomForestRegressor(random_state=42)
grid_search = GridSearchCV(
    rf, param_grid=param_grid, cv=3,
    scoring="r2", n_jobs=-1
)
grid_search.fit(X_train, y_train)

# 最优模型
best_rf = grid_search.best_estimator_

# 测试集评估
y_pred = best_rf.predict(X_test)
r2 = r2_score(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
mae = mean_absolute_error(y_test, y_pred)

# 保存优化后的模型
optimized_model_path = "/mnt/data/rf_model_optimized.joblib"
joblib.dump(best_rf, optimized_model_path)

# 特征重要性图
importances = best_rf.feature_importances_
plt.figure(figsize=(8, 5))
plt.barh(FEATURES, importances, color="skyblue")
plt.xlabel("Feature Importance")
plt.title("Random Forest Feature Importances (Optimized)")
plt.tight_layout()
fig_path = "/mnt/data/feature_importance.png"
plt.savefig(fig_path)
plt.close()

optimized_model_path, fig_path, grid_search.best_params_, {"R2": r2, "RMSE": rmse, "MAE": mae}


