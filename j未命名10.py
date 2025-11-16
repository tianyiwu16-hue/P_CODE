import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
df=pd.read_csv("D:/学习工作文件夹/原神表格文件/housing.csv")  #读取数据
df.columns  ##数据集中都有哪些列
df.describe()  ##获取数据集整体的数据分布特征
df['sqft_living'].describe()  ##某个变量的数据分布特征
plt.hist(df['sqft_living'],bins=100)
plt.boxplot(df['sqft_living'])  ##绘制箱型图