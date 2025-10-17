# -*- coding: utf-8 -*-
"""
Created on Thu Feb 27 14:43:29 2025

@author: 20997
"""
import pandas as pd

# 设置文件路径
csv_file_path = r"D:\原神文件\creditcard.csv"
xlsx_file_path = r"D:\原神文件\creditcard.xlsx"
# 假设 CSV 文件是主要数据源

creditcard = pd.read_csv(csv_file_path)

# 读取CSV文件
df_csv = pd.read_csv(csv_file_path)
print("CSV File Data:")
print(df_csv)

# 读取Excel文件
df_xlsx = pd.read_excel(xlsx_file_path, engine='openpyxl')
print("\nXLSX File Data:")
print(df_xlsx)

# 如果你想检查两个DataFrame是否相同，可以使用如下方法
# 注意: 这个操作有意义的前提是两个文件的数据结构一致
comparison = df_csv.equals(df_xlsx)
print(f"\nAre the CSV and XLSX files data the same? {comparison}")



# 选择用于KNN填充的相关列，并确保它们是数值型
df_for_knn = creditcard[['SEX', 'EDUCATION', 
                         'MARRIAGE', 'AGE', 'LIMIT_BAL']].copy()

# 标准化数据，因为KNNImputer对数值的尺度非常敏感
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
df_scaled = pd.DataFrame(scaler.fit_transform(df_for_knn), columns=df_for_knn.columns)
# 初始化KNNImputer
from sklearn.impute import KNNImputer
imputer = KNNImputer(n_neighbors=5)  # 可以根据实际情况调整n_neighbors参数
# 使用KNNImputer填充缺失值
df_filled_scaled = pd.DataFrame(imputer.fit_transform(df_scaled), columns=df_scaled.columns)
# 将填充后的数据反标准化回原始尺度
df_filled = pd.DataFrame(scaler.inverse_transform(df_filled_scaled), columns=df_scaled.columns)
# 更新原数据框中的'LIMIT_BAL'列
creditcard['LIMIT_BAL'] = df_filled['LIMIT_BAL']