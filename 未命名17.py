# -*- coding: utf-8 -*-
"""
Created on Thu Feb 27 14:48:44 2025

@author: 20997
"""
import pandas as pd
import numpy as np

# 假设我们有一个数据集
data = {'value': np.random.normal(loc=0, scale=1, size=100)}
df = pd.DataFrame(data)

def remove_outliers_3sigma(df, column):
    # 计算均值和标准差
    mean = df[column].mean()
    std = df[column].std()
    
    # 筛选出在3σ范围内的数据
    filtered = df[(df[column] > mean - 3 * std) & (df[column] < mean + 3 * std)]
    
    return filtered

filtered_df = remove_outliers_3sigma(df, 'value')
print(filtered_df)


def remove_outliers_iqr(df, column):
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    
    # 定义上下限
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    
    # 过滤掉异常值
    filtered = df[(df[column] > lower_bound) & (df[column] < upper_bound)]
    
    return filtered

filtered_df_iqr = remove_outliers_iqr(df, 'value')
print(filtered_df_iqr)




import pandas as pd
creditcard = pd.read_csv('path_to_creditcard_dataset.csv')
# Z值标准化
mean_limit_bal = creditcard['LIMIT_BAL'].mean()
std_limit_bal = creditcard['LIMIT_BAL'].std()
creditcard['LIMIT_BAL_ZScore'] = (creditcard['LIMIT_BAL'] - mean_limit_bal) / std_limit_bal

# 0-1归一化
min_ = creditcard['LIMIT_BAL'].min()
max_ = creditcard['LIMIT_BAL'].max()
creditcard['LIMIT_BAL_Normalized'] = (creditcard['LIMIT_BAL'] - min_) / (max_ - min_)

# 如果想要使用稳健缩放（针对可能存在的异常值）
Q1 = creditcard['LIMIT_BAL'].quantile(0.25)
Q3 = creditcard['LIMIT_BAL'].quantile(0.75)
IQR = Q3 - Q1
creditcard['LIMIT_BAL_RobustScaled'] = (creditcard['LIMIT_BAL'] - Q1) / IQR

import pandas as pd
import os

# 构建文件的绝对路径
file_path = os.path.join(os.path.expanduser("~"), "datasets", "creditcard.csv")

try:
    # 加载数据
    creditcard = pd.read_csv(file_path)
    
    # Z值标准化
    mean_limit_bal = creditcard['LIMIT_BAL'].mean()
    std_limit_bal = creditcard['LIMIT_BAL'].std()
    creditcard['LIMIT_BAL_ZScore'] = (creditcard['LIMIT_BAL'] - mean_limit_bal) / std_limit_bal
    
    # 0-1归一化
    min_ = creditcard['LIMIT_BAL'].min()
    max_ = creditcard['LIMIT_BAL'].max()
    creditcard['LIMIT_BAL_Normalized'] = (creditcard['LIMIT_BAL'] - min_) / (max_ - min_)
    
    # 打印处理后的结果以验证
    print(creditcard[['LIMIT_BAL', 'LIMIT_BAL_ZScore', 'LIMIT_BAL_Normalized']].head())
    
except FileNotFoundError:
    print(f"找不到文件: {file_path}. 请检查文件路径是否正确。")



import pandas as pd
import os

# 指定文件所在的目录和文件名
directory = "D:/原神文件"
file_name = "creditcard.csv"

# 构建完整路径
file_path = os.path.join(directory, file_name)

# 检查文件是否存在
if os.path.exists(file_path):
    try:
        # 加载数据
        creditcard = pd.read_csv(file_path)
        
        # Z值标准化
        mean_limit_bal = creditcard['LIMIT_BAL'].mean()
        std_limit_bal = creditcard['LIMIT_BAL'].std()
        creditcard['LIMIT_BAL_ZScore'] = (creditcard['LIMIT_BAL'] - mean_limit_bal) / std_limit_bal
        
        # 0-1归一化
        min_ = creditcard['LIMIT_BAL'].min()
        max_ = creditcard['LIMIT_BAL'].max()
        creditcard['LIMIT_BAL_Normalized'] = (creditcard['LIMIT_BAL'] - min_) / (max_ - min_)
        
        # 打印处理后的结果以验证
        print(creditcard[['LIMIT_BAL', 'LIMIT_BAL_ZScore', 'LIMIT_BAL_Normalized']].head())
    
    except Exception as e:
        print(f"读取或处理文件时出错: {e}")
else:
    print(f"找不到文件: {file_path}. 请检查文件路径是否正确。")




























