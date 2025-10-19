# -*- coding: utf-8 -*-
"""
Created on Mon Mar  3 10:07:37 2025

@author: 20997
"""

import pandas as pd
stockprice=pd.read_table('D:\原神文件\stockprice.txt',header=None)
stockprice.head(10)
# 定义列名
column_names = ['date', 'open', 'low', 'high', 'close', 'vol', 'amt']
# 将列名赋值给DataFrame
stockprice.columns = ['date', 'open', 'low', 
                      'high', 'close', 'vol', 'amt']
# 将 date 列转换为时间格式
stockprice['date'] = pd.to_datetime(stockprice['date'])
# 将 date 列设置为索引
stockprice.set_index('date', inplace=True)
# 检查结果
print(stockprice.head(10))

m2=pd.read_table('D:\原神文件\stockprice.txt',header=None,names=['m2'])
# 将列名赋值给DataFrame
m2.columns = ['m2']
# 生成从 2000 年 1 月开始的月度日期范围
date_range = pd.date_range(start='2000-01-01', periods=len(m2), freq='M')
# 将日期范围添加到数据集中
m2['date'] = date_range
m2['date'] = pd.to_datetime(m2['date'])
# 将 date 列设置为索引
m2.set_index('date', inplace=True)


import pandas as pd
import numpy as np
creditcard=pd.read_csv("D:\原神文件\creditcard.csv")
############方法1############
# 3. 随机选择100个索引
missing_index = creditcard.sample(n=100).index
# 4. 添加缺失值
creditcard.loc[missing_index, 'LIMIT_BAL'] = np.nan
# 5. 验证结果
missing_count = creditcard['LIMIT_BAL'].isna().sum()
print(f"成功添加缺失值数量: {missing_count} 条")
########方法2################
# 生成随机索引
n_missing = 100
random_indices = np.random.choice(creditcard.index, 
                                  size=n_missing, 
                                  replace=False)
# 添加缺失值
creditcard.loc[random_indices, 'LIMIT_BAL'] = np.nan
missing_count = creditcard['LIMIT_BAL'].isna().sum()
print(f"成功添加缺失值数量: {missing_count} 条")



import numpy as np
import statsmodels.api as sm
import matplotlib.pyplot as plt

# 设置随机种子以保证可重复性
np.random.seed(42)

# 生成数据
n = 1000
X = np.random.normal(0, 1, n)  # 自变量X ~ N(0,1)
u = np.random.normal(0, 1, n)  # 误差项
y = 1 + 0.8 * X + u           # 真实关系式

# 添加5个异常值
outlier_indices = np.random.choice(n, size=5, replace=False)
y_outliers = y.copy()
y_outliers[outlier_indices] += 20  # 显著增大这些点的y值

# 构建回归模型（含异常值）
X_sm = sm.add_constant(X)  # 添加常数项
model = sm.OLS(y_outliers, X_sm)
results = model.fit()

# 构建原始模型（无异常值）
model_original = sm.OLS(y, X_sm)
results_original = model_original.fit()

# 可视化结果
plt.figure(figsize=(10, 6))
plt.scatter(X, y_outliers, alpha=0.6, label='正常数据点')
plt.scatter(X[outlier_indices], y_outliers[outlier_indices], 
           color='red', marker='x', s=100, label='异常值')
plt.plot(X, results.fittedvalues, color='red', 
         label=f'含异常值回归: y = {results.params[0]:.2f} + {results.params[1]:.2f}x')
plt.plot(X, results_original.fittedvalues, color='blue', linestyle='--',
         label=f'原始回归: y = {results_original.params[0]:.2f} + {results_original.params[1]:.2f}x')
plt.xlabel('X')
plt.ylabel('y')
plt.title('异常值对线性回归的影响')
plt.legend()
plt.show()

# 打印回归结果对比
print("==================== 含异常值回归结果 ====================")
print(results.summary())
print("\n==================== 原始回归结果 ====================")
print(results_original.summary())