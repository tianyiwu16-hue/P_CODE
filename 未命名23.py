import pandas as pd
from scipy.stats import ttest_ind  # 使用 scipy 的 ttest_ind 函数
#############双总体均值的假设检验###############
###############（1）############
creditcard=pd.read_csv("D:\原神文件\creditcard.csv")
# 提取不同性别的 LIMIT_BAL
male_bal = creditcard[creditcard['SEX'] == 1]['LIMIT_BAL']
female_bal = creditcard[creditcard['SEX'] == 2]['LIMIT_BAL']
t_stat, p_value = ttest_ind(male_bal, female_bal,alternative='greater')
print(f"独立样本t检验结果: t = {t_stat:.4f}, p = {p_value:.4f}")
#############（2）###############
wine=pd.read_csv("D:\原神文件\wine.csv")
y1 = wine[wine['Y'] == 6]['X1']
y2 = wine[wine['Y'] == 7]['X1']
t_stat, p_value = ttest_ind(y1, y2)
print(f"独立样本t检验结果: t = {t_stat:.4f}, p = {p_value:.4f}")
###########(3)###############
hs300=pd.read_csv("D:\原神文件\hs300_basic.csv")
hs300.set_index('date',inplace=True)
hs300.index=pd.to_datetime(hs300.index)
up = hs300[hs300['pct_chg'] >=0]['volume']
down = hs300[hs300['pct_chg'] <0]['volume']
t_stat, p_value = ttest_ind(up, down)
print(f"独立样本t检验结果: t = {t_stat:.4f}, p = {p_value:.4f}")






import pandas as pd
from scipy.stats import ttest_ind  # 使用 scipy 的 ttest_ind 函数

############# 双总体均值的假设检验 ###############

###############（1）############
creditcard = pd.read_csv("D:\\原神文件\\creditcard.csv")  # 注意路径中的双反斜杠
# 提取不同性别的 LIMIT_BAL
male_bal = creditcard[creditcard['SEX'] == 1]['LIMIT_BAL']
female_bal = creditcard[creditcard['SEX'] == 2]['LIMIT_BAL']

# 使用 scipy 的 ttest_ind 进行独立样本 t 检验
t_stat, p_value = ttest_ind(male_bal, female_bal, alternative='greater')
print(f"独立样本 t 检验结果: t = {t_stat:.4f}, p = {p_value:.4f}")

#############（2）###############
wine = pd.read_csv("D:\原神文件\wine.csv")  # 注意路径中的双反斜杠
y1 = wine[wine['Y'] == 6]['X1']
y2 = wine[wine['Y'] == 7]['X1']

# 使用 scipy 的 ttest_ind 进行独立样本 t 检验
t_stat, p_value = ttest_ind(y1, y2)
print(f"独立样本 t 检验结果: t = {t_stat:.4f}, p = {p_value:.4f}")

###########(3)###############
hs300 = pd.read_csv("D:\\原神文件\\hs300_basic.csv")  # 注意路径中的双反斜杠
hs300.set_index('date', inplace=True)
hs300.index = pd.to_datetime(hs300.index)

up = hs300[hs300['pct_chg'] >= 0]['volume']
down = hs300[hs300['pct_chg'] < 0]['volume']

# 使用 scipy 的 ttest_ind 进行独立样本 t 检验
t_stat, p_value = ttest_ind(up, down, equal_var=False)  # 假设方差不相等
print(f"独立样本 t 检验结果: t = {t_stat:.4f}, p = {p_value:.4f}")










import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import f_oneway, chi2_contingency  # 导入所需的统计方法
#################方差分析与列联表检验############
################(1)##################
edu_groups = creditcard.groupby('EDUCATION')['LIMIT_BAL']
f_stat, p_value = f_oneway(*[group for name, group in edu_groups])
print(f"F统计量={f_stat:.3f}, p值={p_value:.4f}")
##############(2)###############
house=pd.read_csv("D:\原神文件\housing.csv")
condition_groups = house.groupby('condition')['price']
f_stat, p_value = f_oneway(*[group for name, group in condition_groups])
print(f"F统计量={f_stat:.3f}, p值={p_value:.4f}")
###########(3)###############
# 假设数据已加载为creditcard
# 如果没有，可以使用: creditcard = pd.read_csv('your_file.csv')
# 1. 数据准备与探索
print("受教育水平分布:")
print(creditcard['EDUCATION'].value_counts())

print("\n婚姻状况分布:")
print(creditcard['MARRIAGE'].value_counts())

# 2. 创建列联表
contingency_table = pd.crosstab(creditcard['EDUCATION'], creditcard['MARRIAGE'])
print("\n列联表(观察频数):")
print(contingency_table)

# 3. 可视化
plt.figure(figsize=(10, 6))
sns.heatmap(contingency_table, annot=True, fmt='d', cmap='Blues')
plt.title('受教育水平与婚姻状况的列联表')
plt.xlabel('婚姻状况')
plt.ylabel('受教育水平')
plt.show()

# 堆叠条形图
contingency_table.plot(kind='bar', stacked=True, figsize=(10, 6))
plt.title('受教育水平与婚姻状况分布')
plt.xlabel('受教育水平')
plt.ylabel('频数')
plt.legend(title='婚姻状况')
plt.show()

# 4. 卡方独立性检验
print("\n卡方独立性检验:")
chi2, p, dof, expected = chi2_contingency(contingency_table)
print(f"卡方统计量 = {chi2:.3f}")
print(f"p值 = {p:.4f}")
print(f"自由度 = {dof}")
##############(4)##############
contingency_table = pd.crosstab(house['condition'], house['grade'])
print("\n列联表(观察频数):")
print(contingency_table)
chi2, p, dof, expected = chi2_contingency(contingency_table)
print(f"卡方统计量 = {chi2:.3f}")
print(f"p值 = {p:.4f}")
print(f"自由度 = {dof}")





import pandas as pd
import statsmodels.api as sm
from sklearn.model_selection import train_test_split

# 假设数据集为 housing，包含以下列：yr_built, sqft_living, 其他特征以及目标变量 price
# 示例数据集的列名可以根据实际情况调整
housing = pd.read_csv("D:\原神文件\housing.csv")  # 注意路径中的双反斜杠
# 1. 将定性变量进行虚拟变量编码，并将 yr_built 转换为 age
current_year = 2025
housing['age'] = current_year - housing['yr_built']
housing = housing.drop(columns=['yr_built'])  # 删除原始的 yr_built 列

# 对定性变量进行虚拟变量编码（假设有一个名为 'zipcode' 的定性变量）
#housing = pd.get_dummies(housing, columns=['zipcode'], drop_first=True)

# 2. 将数据集划分为训练集和测试集
X = housing.drop(columns=['price'])  # 特征变量，假设目标变量是 'price'
y = housing['price']  # 目标变量

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 3. 基于 statsmodels 包建立回归模型
# 添加常数项（截距）
X_train_sm = sm.add_constant(X_train)
model = sm.OLS(y_train, X_train_sm).fit()

# 打印回归结果
print(model.summary())

# 4. 对回归模型进行性能评价
# 在测试集上进行预测
X_test_sm = sm.add_constant(X_test)
y_pred = model.predict(X_test_sm)

# 计算均方误差 (MSE) 和 R^2
from sklearn.metrics import mean_squared_error, r2_score

mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"Mean Squared Error (MSE): {mse}")
print(f"R-squared (R2): {r2}")

# 5. 在 X 中增加 age 和 sqft_living 的平方项重新估计模型
# 添加平方项
X_train['age_sq'] = X_train['age'] ** 2
X_train['sqft_living_sq'] = X_train['sqft_living'] ** 2
X_test['age_sq'] = X_test['age'] ** 2
X_test['sqft_living_sq'] = X_test['sqft_living'] ** 2

# 重新添加常数项并拟合模型
X_train_sm_new = sm.add_constant(X_train)
model_new = sm.OLS(y_train, X_train_sm_new).fit()

# 打印新的回归结果
print(model_new.summary())

# 在测试集上进行预测并评价新模型
X_test_sm_new = sm.add_constant(X_test)
y_pred_new = model_new.predict(X_test_sm_new)

mse_new = mean_squared_error(y_test, y_pred_new)
r2_new = r2_score(y_test, y_pred_new)

print(f"New Mean Squared Error (MSE): {mse_new}")
print(f"New R-squared (R2): {r2_new}")



import pandas as pd
import statsmodels.api as sm
from sklearn.model_selection import train_test_split

house=pd.read_csv("D:\原神文件\housing.csv")
# 选择目标列
categorical_cols = ['grade', 'condition']
# 虚拟变量编码
house = pd.get_dummies(house, columns=categorical_cols, 
                              drop_first=True,dtype=int)
house['age']=2014-house['yr_built']
house.drop('yr_built',axis=1,inplace=True)

from sklearn.model_selection import train_test_split

# 1. 分离特征和目标变量
X = house.drop(columns=['price'])
y = house['price']
# 2. 划分数据集
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42)



import statsmodels.api as sm
from sklearn.metrics import r2_score, mean_squared_error

# 1. 添加常数项（截距项）
X_train_const = sm.add_constant(X_train)
X_test_const = sm.add_constant(X_test)

# 2. 创建并拟合模型
model = sm.OLS(y_train, X_train_const)
results = model.fit()

# 3. 输出模型摘要
print(results.summary())

# 4. 进行预测
y_train_pred = results.predict(X_train_const)
y_test_pred = results.predict(X_test_const)

# 5. 评估测试集表现
print("\n训练集评估：")
print(f"R²分数：{r2_score(y_train, y_train_pred):.4f}")
print(f"均方根误差：{mean_squared_error(y_train, y_train_pred)**0.5:.4f}")

print("\n测试集评估：")
print(f"R²分数：{r2_score(y_test, y_test_pred):.4f}")
print(f"均方根误差：{mean_squared_error(y_test, y_test_pred)**0.5:.4f}")



###############
insignificant_vars = ['grade_3','grade_7']

# 创建新特征矩阵（必须保留常数项）
selected_features = [col for col in X_train_const.columns if col not in insignificant_vars]
# 构建新数据集
X_train_new = X_train_const[selected_features]
X_test_new = X_test_const[selected_features]

# 重新估计模型 ------------------------------------------------
# 使用筛选后的特征拟合新模型
new_model = sm.OLS(y_train, X_train_new)
new_results = new_model.fit()

print("\n\n========== 优化后模型 ==========")
print(new_results.summary())
# 4. 进行预测
y_train_pred = new_results.predict(X_train_new)
y_test_pred = new_results.predict(X_test_new)

# 5. 评估测试集表现
print("\n训练集评估：")
print(f"R²分数：{r2_score(y_train, y_train_pred):.4f}")
print(f"均方根误差：{mean_squared_error(y_train, y_train_pred)**0.5:.4f}")

print("\n测试集评估：")
print(f"R²分数：{r2_score(y_test, y_test_pred):.4f}")
print(f"均方根误差：{mean_squared_error(y_test, y_test_pred)**0.5:.4f}")







############基于sklearn的回归#################
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn import linear_model
from sklearn.metrics import r2_score, mean_squared_error
house=pd.read_csv("D:\原神文件\housing.csv")
# 选择目标列
categorical_cols = ['grade', 'condition']
# 虚拟变量编码
house = pd.get_dummies(house, columns=categorical_cols, 
                              drop_first=True,dtype=int)
house['age']=2014-house['yr_built']
house.drop('yr_built',axis=1,inplace=True)
# 1. 分离特征和目标变量
X = house.drop(columns=['price'])
y = house['price']
# 2. 划分数据集
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42)
model2=linear_model.LinearRegression().fit(X_train,y_train)
model2.coef_
# 4. 进行预测
y_train_pred = model2.predict(X_train)
y_test_pred = model2.predict(X_test)

# 5. 评估测试集表现
print("\n训练集评估：")
print(f"R²分数：{r2_score(y_train, y_train_pred):.4f}")
print(f"均方根误差：{mean_squared_error(y_train, y_train_pred)**0.5:.4f}")

print("\n测试集评估：")
print(f"R²分数：{r2_score(y_test, y_test_pred):.4f}")
print(f"均方根误差：{mean_squared_error(y_test, y_test_pred)**0.5:.4f}")


import numpy as np
import pandas as pd
from sklearn.feature_selection import VarianceThreshold,SelectKBest
from sklearn.feature_selection import f_regression,mutual_info_classif,chi2
from sklearn import linear_model
BLOG=pd.read_csv(r"D:\原神文件\blogData_train.csv",header=None)
BLOG_test=pd.read_csv(r"D:\原神文件\blogData_test.csv",header=None)
ColName=['X'+str(k+1) for k in np.arange(280)]
ColName.append('Y')
BLOG.columns=ColName
BLOG_test.columns=ColName
y,X=BLOG['Y'],BLOG.drop('Y',axis=1)
y_test,X_test=BLOG_test['Y'],BLOG_test.drop('Y',axis=1)

####原始回归####
mdl_ols=linear_model.LinearRegression().fit(X,y)
print(mdl_ols.score(X,y))
print(mdl_ols.score(X_test,y_test))

####原始回归####
import time
s=time.time()
mdl_ols=linear_model.LinearRegression().fit(X,y)
e=time.time()
print(e-s)
print(mdl_ols.score(X,y))
print(mdl_ols.score(X_test,y_test))
##############使用方差过滤法进行特征选择############3
th=np.percentile(np.var(X),90)
selector = VarianceThreshold(threshold=th)
newX=selector.fit_transform(X)
indices=selector.get_support(indices=True)
#####特征选择后的回归############
s=time.time()
mdl_ols=linear_model.LinearRegression().fit(newX,y)
e=time.time()
print(e-s)
print(mdl_ols.score(newX,y))
newX_test=X_test.iloc[:,indices]
print(mdl_ols.score(newX_test,y_test))
#########使用F检验进行过滤##############
selector=SelectKBest(score_func=f_regression,k=28).fit(X,y)
indices=selector.get_support(indices=True)
newX2=X.iloc[:,indices]
mdl_ols=linear_model.LinearRegression().fit(newX2,y)
print(mdl_ols.score(newX2,y))
newX_test=X_test.iloc[:,indices]
print(mdl_ols.score(newX_test,y_test))


############根据spearman相关系数进行过滤############
from scipy.stats import spearmanr

def spearman_correlation(X, y):
    """
    计算每个特征与目标变量的斯皮尔曼相关系数绝对值
    返回值格式: (scores, pvalues)
    """
    scores = []
    pvalues = []
    for i in range(X.shape[1]):
        corr, pval = spearmanr(X[:, i], y)
        scores.append(np.abs(corr))  # 取绝对值，负相关也视为重要
        pvalues.append(pval)
    return np.array(scores), np.array(pvalues)
s=time.time()
selector=SelectKBest(score_func=spearman_correlation,k=28).fit(X,y)
e=time.time()
print(e-s)
indices=selector.get_support(indices=True)
newX2=X.iloc[:,indices]
mdl_ols=linear_model.LinearRegression().fit(newX2,y)
print(mdl_ols.score(newX2,y))
newX_test=X_test.iloc[:,indices]
print(mdl_ols.score(newX_test,y_test))




###################神经网络###############
from sklearn.neural_network import MLPClassifier
mlp = MLPClassifier(hidden_layer_sizes=(1000), activation='relu',
                    max_iter=500,solver='adam', verbose=5,  
                    learning_rate_init=0.001)
mlp.fit(X_train_scaled,y_train)
print(f'训练集准确率: {mlp.score(X_train_scaled,y_train):.4f}')
print(f'测试集准确率: {mlp.score(X_test_scaled,y_test):.4f}')


from sklearn.preprocessing import StandardScaler  # 添加这一行来导入StandardScaler
import pandas as pd
from sklearn import linear_model
from sklearn import tree
from ucimlrepo import fetch_ucirepo 
import numpy as np
from sklearn.model_selection import train_test_split
# fetch dataset 
spambase = fetch_ucirepo(id=94)   
# data (as pandas dataframes) 
X = spambase.data.features 
y = spambase.data.targets 
spam=pd.concat([X,y],axis=1)
# 2. 划分数据集
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42)


# 数据标准化（KNN对特征尺度敏感，建议始终标准化）
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)



#############决策树################
import matplotlib.pyplot as plt
from sklearn import tree
ctree=tree.DecisionTreeClassifier(max_depth=5,min_samples_split=50)
ctree.fit(X_train,y_train)
print(f'训练集准确率: {ctree.score(X_train,y_train):.4f}')
print(f'测试集准确率: {ctree.score(X_test,y_test):.4f}')
# 绘制决策树
plt.figure(figsize=(30, 20),dpi=300)
tree.plot_tree(
    ctree,
    feature_names=X_train.columns.tolist(),  # 特征名称（需替换为实际列名）
    class_names=['Class 0', 'Class 1'],      # 类别标签（按实际名称修改）
    filled=True,
    rounded=True
)
plt.title("决策树可视化")
plt.show()