# -*- coding: utf-8 -*-
"""
Created on Mon Oct 14 16:07:02 2024

@author: 20997
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
#模拟生成一组工资数据
data=np.random.normal(1300,200,50).astype(int)
#将生成的工资数据转换成为数据框
df=pd.DataFrame({'values':data})
#定义分组间隔，并分组
bins=[700,800,900,1000,1100,1200,1300,1400,
                    1500,1600,1700,1800,1900]
df['bins'] = pd.cut(df['values'], bins=bins)
#统计每组频数
frequency = df['bins'].value_counts()
#按照分组标志排序并打印
ordered_frequency = frequency.sort_index()
print(ordered_frequency)
###############################################
#直方图
plt.hist(data,bins=[800,900,1000,1100,1200,1300,1400,
                    1500,1600,1700,1800,1900])
#累计直方图
plt.hist(data,bins=[800,900,1000,1100,1200,1300,1400,
                    1500,1600,1700,1800,1900],cumulative=True)
#频率密度直方图
plt.hist(data,bins=[800,900,1000,1100,1200,1300,1400,
                    1500,1600,1700,1800,1900],density=True)
#绘制分布曲线图
import seaborn as sns
sns.kdeplot(data)
############
# 模拟一组收入数据
np.random.seed(0)  # 设置随机种子以得到可重复的结果
incomes = np.random.normal(5000,1500,1000).astype(int)  # 使用正态分布生成收入数据
#incomes=(10000*np.random.beta(1,1,1000)).astype(int)  # 使用beta分布生成收入数据
# 对收入数据排序
sorted_incomes = np.sort(incomes)
# 计算累积收入百分比
cumulative_income = np.cumsum(sorted_incomes)
cumulative_income_percent = cumulative_income / cumulative_income[-1]
# 计算累积人口百分比
population_percent = np.arange(1, len(sorted_incomes) + 1) / len(sorted_incomes)
# 绘制洛伦兹曲线
plt.figure(figsize=(10, 6))
plt.plot(population_percent, cumulative_income_percent, marker='.', label='Lorenz Curve')
plt.plot([0, 1], [0, 1], color='red', linestyle='--', label='Line of Equality')  # 完全平等线
# 设置图表标题和标签
plt.title('Lorenz Curve')
plt.xlabel('Proportion of Population')
plt.ylabel('Proportion of Cumulative Income')
plt.legend()
plt.grid(True)
# 显示图表
plt.show()
# 计算基尼系数，np.trapz是numpy中数值积分的计算函数
lorenz_area = np.trapz(cumulative_income_percent, population_percent)
gini_coefficient=(0.5-lorenz_area)/0.5
print(f"Gini Coefficient: {gini_coefficient}")
#4.实际数据的分组统计
data=pd.read_csv("d:/pythonpath/creditcard.csv")
df=data['AGE']
bins=[18,25,35,45,55,65,75,85]
df['bins'] = pd.cut(df, bins=bins)
frequency = df['bins'].value_counts()
ordered_frequency = frequency.sort_index()
print(ordered_frequency)
plt.hist(data['AGE'],bins=bins)
##########
data1=pd.read_csv("D:\yuan1\creditcard.csv")
print(data1)