import pandas as pd
import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt
y=np.random.normal(100,14.14,10000)  #生成期望为100，标准差为14.14的正态分布随机数
df_y=pd.DataFrame(y) #转换成为pandas数据框
print(np.mean(y))  ##numpy中的平均数
print(df_y.mean())  ##pandas中的平均数
print(np.median(y)) ##numpy中的中位数
print(df_y.median()) ##pandas中的中位数
print(np.std(y)) ##numpy中的标准差
print(df_y.std()) ##pandas中的标准差
####计算分位数########
print("变量5%分位数是{}".format(np.quantile(y,0.05)))
print("变量95%分位数是{}".format(np.quantile(y,0.95)))
plt.hist(y,bins=100)
########换一个分布######
x=np.random.chisquare(100,10000)  #生成自由度为100的卡方分布随机数
df_x=pd.DataFrame(x) #转换成为pandas数据框
print(np.mean(x))  ##numpy中的平均数
print(df_x.mean())  ##pandas中的平均数
print(np.median(x)) ##numpy中的中位数
print(df_x.median()) ##pandas中的中位数
print(np.std(x)) ##numpy中的标准差
print(df_x.std()) ##pandas中的标准差
print("变量5%分位数是{}".format(np.quantile(x,0.05)))
print("变量95%分位数是{}".format(np.quantile(x,0.95)))
plt.hist(x,bins=100)




import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
df=pd.read_csv("D:/学习工作文件夹/原神表格文件/housing.csv")  #读取数据
df.columns  ##数据集中都有哪些列
df.describe()  ##获取数据集整体的数据分布特征
#df['sqft_living'].describe()  ##某个变量的数据分布特征
df['grade'].describe()  ##某个变量的数据分布特征
#plt.hist(df['sqft_living'],bins=100)
plt.hist(df['grade'],bins=100)
#plt.boxplot(df['sqft_living'])  ##绘制箱型图
plt.boxplot(df['grade'])  ##绘制箱型图


###自定义函数，承接第2小节代码######
def cv(series):
    return series.mean()/series.std()
df.apply(cv)  ##apply方法
df.agg(cv) ##agg方法
df['sqft_living'].agg(cv) ##验证计算结果
df['sqft_living'].mean()/df['sqft_living'].std()

#计算四分位据
import numpy as np

# 示例数据
data = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]

# 计算 Q1 和 Q3
Q1 = np.percentile(data, 25)
Q3 = np.percentile(data, 75)

# 计算 IQR
IQR = Q3 - Q1

print("Q1:", Q1)
print("Q3:", Q3)
print("IQR:", IQR)


###承接第2小节代码######
df['z_score']=(df['sqft_living']-df['sqft_living'].mean())/df['sqft_living'].std()
df['z_score'].describe()  ##验证z_score是否是0均值，标准差是1




import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
# 设置绘图尺寸
plt.figure(figsize=(10, 6))
# 设定x轴的范围
x = np.linspace(-10, 10, 1000)
# 不同的标准差
std_devs = [0.5, 1, 2, 4]
# 绘制每个标准差对应的正态分布
for std in std_devs:
    pdf = norm.pdf(x, loc=0, scale=std)  # 计算概率密度函数
    plt.plot(x, pdf, label=f'μ=0, σ={std}')  # 绘制曲线并添加标签
# 添加图表标题和图例
plt.title('Normal Distributions with Different Standard Deviations')
plt.xlabel('x')
plt.ylabel('Probability Density')
plt.legend()
# 显示图表
plt.grid(True)
plt.show()


from scipy.stats import norm
import scipy.integrate as integrate
norm.cdf(1.96)  ##服从标准正态分布的变量x小于等于1.96的概率
norm.cdf(1.96)-norm.cdf(-1.96)  #x介于-1.96至1.96的概率
# 使用 quad 函数计算pdf的积分
result, error = integrate.quad(lambda x: norm.pdf(x), -1.96, 1.96)

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import chi2

# 定义自由度
degrees_of_freedom = [4, 10, 20]

# 生成x轴的值
x = np.linspace(0, 20, 1000)

# 创建一个图形和一个子图
fig, ax = plt.subplots()

# 绘制不同自由度的卡方分布
for df in degrees_of_freedom:
    y = chi2.pdf(x, df)
    ax.plot(x, y, label=f'df = {df}')

# 设置标题和标签
ax.set_title('Chi-Squared Distribution')
ax.set_xlabel('x')
ax.set_ylabel('Probability Density')

# 显示图例
ax.legend()

# 显示图形
plt.show()






