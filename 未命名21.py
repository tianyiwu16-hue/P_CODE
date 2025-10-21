# -*- coding: utf-8 -*-
"""
Created on Mon Mar 24 11:```````01:02 2025

@author: 20997
"""
import pandas as pd
import numpy as np
# 假设 creditcard 数据集已加载为 DataFrame
# 示例数据集加载代码（替换为您的实际数据集）
creditcard = pd.read_csv("D:\原神文件\creditcard.csv")
# 计算偏度（默认是样本偏度，修正偏差）
age_skew = creditcard['AGE'].skew()

# 计算峰度（默认是样本超额峰度，即峰度 - 3）
age_kurtosis = creditcard['AGE'].kurtosis()

print(f"偏度 (Skewness): {age_skew:.3f}")
print(f"峰度 (Kurtosis): {age_kurtosis:.3f}")
###########
percentile_25 = creditcard['AGE'].quantile(0.25)
percentile_50 = creditcard['AGE'].quantile(0.50)  # 中位数
percentile_75 = creditcard['AGE'].quantile(0.75)

print(f"25% 百分位数: {percentile_25:.2f}")
print(f"50% 百分位数（中位数）: {percentile_50:.2f}")
print(f"75% 百分位数: {percentile_75:.2f}")
creditcard['AGE'].quantile([0,0.01,0.05,0.1,0.25,0.5])

percentiles = np.arange(10, 100, 10)  # 生成 [10, 20, ..., 90]
percentile_values = np.percentile(creditcard['AGE'], percentiles)

print("各个百分位数:")
for p, value in zip(percentiles, percentile_values):
    print(f"{p}% 百分位数: {value:.2f}")

from scipy.stats import jarque_bera

# 示例数据（假设 creditcard 是已加载的 DataFrame）
# 例如：
# creditcard = pd.read_csv("creditcard.csv")
# 进行 JB 检验
jb_stat, p_value = jarque_bera(creditcard['AGE'])

print(f"JB 统计量: {jb_stat:.3f}")
print(f"p 值: {p_value:.3f}")






import pandas as pd
from scipy.stats import pearsonr, spearmanr
wine=pd.read_csv("D:\原神文件\wine.csv")
# 计算 Pearson 相关系数及 p 值
pearson_coef, pearson_p = pearsonr(wine['X1'], wine['X2'])
# 计算 Spearman 相关系数及 p 值
spearman_coef, spearman_p = spearmanr(wine['X1'], wine['X2'])
print(f"Pearson 相关系数: {pearson_coef:.4f}, p 值: {pearson_p:.4f}")
print(f"Spearman 相关系数: {spearman_coef:.4f}, p 值: {spearman_p:.4f}")

wine_=wine.drop('Y',axis=1)
# 计算 Pearson 相关系数矩阵
pearson_corr = wine_.corr(method='pearson')['X1'].abs().sort_values(ascending=False)
# 计算 Spearman 相关系数矩阵
spearman_corr = wine_.corr(method='spearman')['X1'].abs().sort_values(ascending=False)
# 输出结果
print("Pearson 相关系数（绝对值排序）:")
print(pearson_corr)
print("\nSpearman 相关系数（绝对值排序）:")
print(spearman_corr)
# 找到 Pearson 和 Spearman 相关性最强的变量
pearson_top = pearson_corr.index[1]  # 跳过 X1 自身
spearman_top = spearman_corr.index[1]

print(f"\n与 X1 相关性最强的变量（Pearson）: {pearson_top}, 系数 = {pearson_corr[1]:.3f}")
print(f"与 X1 相关性最强的变量（Spearman）: {spearman_top}, 系数 = {spearman_corr[1]:.3f}")



import seaborn as sns
import matplotlib.pyplot as plt
# 计算 Pearson 相关系数矩阵
corr_matrix = wine_.corr()
plt.figure(figsize=(12, 10))
sns.heatmap(
    corr_matrix, 
    annot=True,         # 显示数值
    fmt=".2f",         # 数值格式（保留2位小数）
    cmap="coolwarm",   # 颜色映射：红（正相关）-蓝（负相关）
    vmin=-1,           # 颜色条最小值
    vmax=1,            # 颜色条最大值
    linewidths=0.5,    # 单元格边线宽度
    square=True,       # 使热力图单元格为正方形
    cbar_kws={"shrink": 0.8}  # 调整颜色条大小
)

plt.title("Wine 数据集特征 Pearson 相关性热力图", fontsize=14)
plt.xticks(rotation=45, ha="right")  # 调整 x 轴标签旋转
plt.tight_layout()  # 避免标签被截断
plt.show()




from sklearn.neighbors import NearestNeighbors
# 初始化最近邻模型（使用欧氏距离）
nbrs = NearestNeighbors(n_neighbors=6, metric='euclidean')  # 包含自己，所以选6
nbrs.fit(wine_)
# 获取索引200的样本（注意Python从0开始，实际是第201个样本）
query_index = 200  # 因为索引从0开始
distances, indices = nbrs.kneighbors([wine_.iloc[query_index]])
# 输出结果
print("最近的5个样本的索引（不包括自己）:", indices[0][1:])  # 跳过第一个（自己）
print("对应的距离:", distances[0][1:])

import numpy as np
from sklearn.metrics import pairwise_distances
wine_6=wine_[wine['Y']==8]
# 计算所有样本间的欧氏距离矩阵
dist_matrix = pairwise_distances(wine_6, metric='euclidean')
# 取上三角部分（避免重复计算和自身距离）
n_samples = wine_6.shape[0]
avg_distance = dist_matrix[np.triu_indices(n_samples, k=1)].mean()
print(f"类别6的样本间平均欧氏距离: {avg_distance:.4f}")

import numpy as np
from sklearn.metrics import pairwise_distances
wine_6=wine_[wine['Y']==7]
# 计算所有样本间的欧氏距离矩阵
dist_matrix = pairwise_distances(wine_6, metric='euclidean')
# 取上三角部分（避免重复计算和自身距离）
n_samples = wine_6.shape[0]
avg_distance = dist_matrix[np.triu_indices(n_samples, k=1)].mean()
print(f"类别6的样本间平均欧氏距离: {avg_distance:.4f}")


import numpy as np
from sklearn.metrics import pairwise_distances
wine_6=wine_[wine['Y']==6]
# 计算所有样本间的欧氏距离矩阵
dist_matrix = pairwise_distances(wine_6, metric='euclidean')
# 取上三角部分（避免重复计算和自身距离）
n_samples = wine_6.shape[0]
avg_distance = dist_matrix[np.triu_indices(n_samples, k=1)].mean()
print(f"类别6的样本间平均欧氏距离: {avg_distance:.4f}")



import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
house=pd.read_csv("D:\原神文件\housing.csv")

# 设置图形大小
plt.figure(figsize=(12, 5))

# 直方图
plt.subplot(1, 2, 1)  # 1行2列的第1个位置
sns.histplot(house['sqft_living'], bins=50, kde=True, color='skyblue')
plt.title('Distribution of Living Area (sqft)')
plt.xlabel('Living Area (square feet)')
plt.ylabel('Frequency')

# 箱线图
plt.subplot(1, 2, 2)  # 1行2列的第2个位置
sns.boxplot(y=house['sqft_living'], color='lightgreen')
plt.title('Boxplot of Living Area (sqft)')
plt.xlabel('Living Area (square feet)')

plt.tight_layout()  # 自动调整子图间距
plt.show()



# 统计各等级频数
grade_counts = house['grade'].value_counts().sort_index()  # 按等级排序
# 绘制饼图
plt.figure(figsize=(8, 8))
plt.pie(grade_counts, 
        labels=grade_counts.index, 
        autopct='%1.1f%%', 
        startangle=90,
        colors=plt.cm.Paired.colors)  # 使用彩虹色系
plt.title('Proportion of House Grades', fontsize=14, pad=20)
plt.axis('equal')  # 确保饼图为正圆形
plt.show()
# 设置中文字体（如果含中文标签）
plt.rcParams['font.sans-serif'] = ['SimHei']  # Windows系统
# 统计并过滤小类别（如占比<2%的合并为"其他"）
grade_counts = house['grade'].value_counts()
small_categories = grade_counts[grade_counts/grade_counts.sum() < 0.02]
if len(small_categories) > 0:
    grade_counts = grade_counts[grade_counts/grade_counts.sum() >= 0.02]
    grade_counts['Other'] = small_categories.sum()

# 绘制饼图
plt.figure(figsize=(10, 10))
explode = [0.1 if i == grade_counts.idxmax() else 0 for i in grade_counts.index]  # 突出最大值

wedges, texts, autotexts = plt.pie(
    grade_counts,
    labels=None,  # 隐藏标签，改用图例
    autopct='%1.1f%%',
    startangle=140,
    explode=explode,
    colors=sns.color_palette("pastel"),
    textprops={'fontsize': 20}
)

# 添加图例
plt.legend(wedges, 
          grade_counts.index,
          title="Grade Levels",
          loc="center left",
          bbox_to_anchor=(1, 0.5))

plt.title('房屋等级分布 (优化版)', fontsize=28, pad=20)
plt.tight_layout()
plt.show()


hs300=pd.read_csv("D:\原神文件\hs300_basic.csv")
hs300.set_index('date',inplace=True)
hs300.index=pd.to_datetime(hs300.index)
# 设置图形风格和大小
plt.figure(figsize=(12, 6))

# 绘制时序折线图
plt.plot(hs300.index, hs300['close'], 
         color='steelblue', 
         linewidth=1.5,
         label='Close Price')

# 添加标题和标签
plt.title('沪深300指数收盘价时序图', fontsize=14, pad=20)
plt.xlabel('日期', fontsize=12)
plt.ylabel('收盘价', fontsize=12)
plt.grid(True, linestyle='--', alpha=0.7)  # 添加网格线

# 自动旋转日期标签避免重叠
plt.gcf().autofmt_xdate()

plt.legend()
plt.tight_layout()
plt.show()






import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
house=pd.read_csv("D:\原神文件\housing.csv")
from scipy import stats
# 设置样式
sns.set(style="whitegrid")
plt.rcParams['font.sans-serif'] = ['SimHei']  # Windows系统
plt.figure(figsize=(10, 6), dpi=300)
# 计算皮尔逊相关系数和p值
corr, p_value = stats.pearsonr(house['price'], house['sqft_living'])
corr_text = f"Pearson r = {corr:.2f}\np-value = {p_value:.2e}"  # 科学计数法显示p值
# 绘制散点图
scatter = sns.scatterplot(
    data=house,
    x='sqft_living',
    y='price',
    alpha=0.6,
    color='steelblue',
    edgecolor='w',
    s=60  # 点大小
)
# 添加回归线
sns.regplot(
    data=house,
    x='sqft_living',
    y='price',
    scatter=False,  # 不重复绘制散点
    color='coral',
    line_kws={'lw': 2, 'ls': '--'}
)

# 标注相关系数
plt.text(
    x=0.05,
    y=0.95,
    s=corr_text,
    transform=plt.gca().transAxes,  # 使用相对坐标
    fontsize=12,
    bbox=dict(facecolor='white', alpha=0.8, edgecolor='gray')
)

# 优化图形
plt.title('House Price vs. Living Area', fontsize=15)
plt.xlabel('居住面积 (sqft)', fontsize=12)
plt.ylabel('房价($)', fontsize=12)
plt.tight_layout()
#plt.savefig('price_vs_sqft.png', dpi=300)  # 高清保存
plt.show()

# 计算对数变换后的相关系数（避免零值问题）
price_log = np.log(house['price'] + 1)  # +1防止log(0)
sqft_log = np.log(house['sqft_living'])
corr_log, p_value_log = stats.pearsonr(price_log, sqft_log)

# 绘制散点图（对数坐标轴）
scatter = sns.scatterplot(
    x=sqft_log,
    y=price_log,
    alpha=0.6,
    color='steelblue',
    edgecolor='w',
    s=60
)

# 添加回归线
sns.regplot(
    x=sqft_log,
    y=price_log,
    scatter=False,
    color='coral',
    line_kws={'lw': 2, 'ls': '--'}
)

# 标注相关系数
plt.text(
    x=0.05,
    y=0.95,
    s=f"Log-Log Pearson r = {corr_log:.2f}\np-value = {p_value_log:.2e}",
    transform=plt.gca().transAxes,
    fontsize=12,
    bbox=dict(facecolor='white', alpha=0.8)
)

# 优化图形
plt.title('House Price vs. Living Area (Log-Log Scale)', fontsize=15)
plt.xlabel('Living Area (sqft, log10)', fontsize=12)
plt.ylabel('Price (USD, log10)', fontsize=12)
plt.grid(True, which="both", ls="--", alpha=0.4)
plt.tight_layout()
plt.savefig('loglog_scatter.png', dpi=300)
plt.show()

#####单总体均值的假设检验##############
from scipy import stats
# 单样本t检验
t_statistic, p_value = stats.ttest_1samp(hs300['pct_chg'], popmean=0.02)
# 输出结果
print(f"t统计量: {t_statistic:.4f}")
print(f"p值: {p_value:.4f}")
# 单样本单侧t检验（右侧检验）
t_statistic, p_value_one_tail = stats.ttest_1samp(hs300['pct_chg'], 
                                                  popmean=0.02, 
                                                  alternative='less')
print(f"t统计量: {t_statistic:.4f}")
print(f"单侧检验p值: {p_value_one_tail:.4f}")



creditcard=pd.read_csv("D:\原神文件\creditcard.csv")
# 提取不同性别的 LIMIT_BAL
male_bal = creditcard[creditcard['SEX'] == 1]['LIMIT_BAL']
female_bal = creditcard[creditcard['SEX'] == 2]['LIMIT_BAL']
t_stat, p_value = stats.ttest_ind(male_bal, female_bal,alternative='greater')
print(f"独立样本t检验结果: t = {t_stat:.4f}, p = {p_value:.4f}")











