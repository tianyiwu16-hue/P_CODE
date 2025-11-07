import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
import joblib

# ================== 1. 读取原始数据 ==================
file_path = r"D:\桌面应用\student_lifestyle_dataset(3).xlsx"
df = pd.read_excel(file_path)

# 特征列和目标列
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

# ================== 2. 训练随机森林 ==================
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

rf_model = RandomForestRegressor(
    n_estimators=200,
    max_depth=5,
    min_samples_split=2,
    min_samples_leaf=2,
    max_features="sqrt",
    random_state=42
)

rf_model.fit(X_train, y_train)

# 保存模型
model_path = r"D:\数据要素大赛作品\rf_model_optimized.joblib"
joblib.dump(rf_model, model_path)
print(f"✅ 模型已保存到: {model_path}")

# ================== 3. 扩展 50 万条新数据 ==================
n_samples = 500_000
synthetic_data = pd.DataFrame()

for col in FEATURES:
    mu, sigma = df[col].mean(), df[col].std()
    synthetic_data[col] = np.random.normal(mu, sigma, n_samples)

# 时长不能为负
synthetic_data = synthetic_data.clip(lower=0)

# 用模型预测 GPA
synthetic_data[TARGET] = rf_model.predict(synthetic_data[FEATURES])

# 限制 GPA 范围 0 ~ 4
synthetic_data[TARGET] = synthetic_data[TARGET].clip(lower=0, upper=4)

# 保存为 CSV
output_path = r"D:\数据要素大赛作品\expanded_student_dataset_500k.csv"
synthetic_data.to_csv(output_path, index=False)
print(f"✅ 扩展数据已保存到: {output_path}")
