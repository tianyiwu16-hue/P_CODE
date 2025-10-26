import surprise
print(surprise.__version__)


from surprise import Dataset
from surprise import SVD

# 加载movielens数据集
data = Dataset.load_builtin('ml-100k')

# 使用SVD算法
algo = SVD()

# 训练模型
trainset = data.build_full_trainset()
algo.fit(trainset)

# 获取预测
uid = str(196)  # 用户ID
iid = str(302)  # 项目ID

pred = algo.predict(uid, iid)
print(pred)







##############推荐系统##############
from surprise import dataset
from surprise import SVD,KNNBasic
import pandas as pd
from collections import defaultdict
ds=dataset.Dataset.load_builtin(name=u'ml-100k', prompt=True)
rawratings=ds.raw_ratings
rawratings=pd.DataFrame(rawratings)
rawratings.columns=['user_id','item_id','rating','imdb']
trainset = ds.build_full_trainset()
algo_knn = KNNBasic(k=40,min_k=3,sim_options={'user_based': True})
algo_svd=SVD()
algo_knn.fit(trainset)
pred = algo_knn.predict('186', '302', verbose=True)
algo_svd.fit(trainset)
pred = algo_svd.predict('186', '269', verbose=True)
temp=rawratings[rawratings['user_id']=='186']
testset = trainset.build_anti_testset()
predictions = algo_knn.test(testset)
def get_top_n(predictions, n=10):
    # First map the predictions to each user.
    top_n = defaultdict(list)
    for uid, iid, true_r, est, _ in predictions:
        top_n[uid].append((iid, est))

    # Then sort the predictions for each user and retrieve the k highest ones.
    for uid, user_ratings in top_n.items():
        user_ratings.sort(key=lambda x: x[1], reverse=True)
        top_n[uid] = user_ratings[:n]

    return top_n

testset = trainset.build_anti_testset()
predictions = algo_svd.test(testset)

top_n = get_top_n(predictions, n=10)



import surprise
print(surprise.__version__)




##############可解释性################
############线性回归模型的可解释性###########
from sklearn import linear_model
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
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

# 计算特征和目标的标准差（确保使用样本标准差，ddof=1）
std_X = np.std(X_train, axis=0, ddof=1)
std_y = np.std(y_train, ddof=1)

# 计算标准化回归系数
standardized_coef = model2.coef_ * (std_X / std_y)

# 创建带有特征名的Series
if isinstance(X_train, pd.DataFrame):
    coef_series = pd.Series(standardized_coef, index=X_train.columns)
else:
    # 若X_train是numpy数组，生成默认特征名称
    coef_series = pd.Series(standardized_coef, index=[f'Feature_{i}' for i in range(X_train.shape[1])])

# 按绝对值降序排序
sorted_coef = coef_series.iloc[np.argsort(-np.abs(coef_series))]

# 输出结果
print("标准化回归系数（按绝对值降序排序）：")
print(sorted_coef)

   

from sklearn import tree
from ucimlrepo import fetch_ucirepo 
import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import pandas as pd
# fetch dataset 
spambase = fetch_ucirepo(id=94)   
# data (as pandas dataframes) 
X = spambase.data.features 
y = spambase.data.targets 
spam=pd.concat([X,y],axis=1)
# 2. 划分数据集
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42)
ctree=tree.DecisionTreeClassifier(max_depth=5,min_samples_split=50)
ctree.fit(X_train,y_train)
print(f'训练集准确率: {ctree.score(X_train,y_train):.4f}')
print(f'测试集准确率: {ctree.score(X_test,y_test):.4f}')

# 提取特征重要性
feature_importances = ctree.feature_importances_

# 获取特征名称（适配 DataFrame 和 numpy 数组）
if isinstance(X_train, pd.DataFrame):
    feature_names = X_train.columns.tolist()
else:
    feature_names = [f"Feature_{i}" for i in range(X_train.shape[1])]

# 将特征重要性打包为 Series 并排序
importance_series = pd.Series(feature_importances, index=feature_names)
sorted_importance = importance_series.sort_values(ascending=False)

# 取前10个重要特征
top_10 = sorted_importance.head(10)

# 绘制水平条形图
plt.figure(figsize=(10, 6))
top_10.sort_values().plot(kind='barh', color='skyblue')
plt.title('Top 10 Important Features - Decision Tree (ctree)')
plt.xlabel('Feature Importance Score')
plt.ylabel('Features')

# 添加数值标签
for index, value in enumerate(top_10.sort_values()):
    plt.text(value, index, f'{value:.3f}', va='center')

plt.tight_layout()
plt.show()



#########置换特征重要性##############
from sklearn.neighbors import KNeighborsClassifier
from ucimlrepo import fetch_ucirepo 
import numpy as np

# fetch dataset 
spambase = fetch_ucirepo(id=94)   
# data (as pandas dataframes) 
X = spambase.data.features 
y = spambase.data.targets 
spam=pd.concat([X,y],axis=1)
# 2. 划分数据集
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42)
# 初始化KNN分类器（默认使用5个邻居）
knn = KNeighborsClassifier(n_neighbors=5)
# 训练模型
knn.fit(X_train, y_train)
from sklearn.inspection import permutation_importance

result = permutation_importance(knn, X_train, y_train, n_repeats=20)
sorted_idx = result.importances_mean.argsort()
fig, ax = plt.subplots()
ax.boxplot(result.importances[sorted_idx[-10:]].T,
           vert=False, labels=X.columns[sorted_idx[-10:]])
ax.set_title("Permutation Importances")
fig.tight_layout()
plt.show()


#######基于shap计算边际效应########
import shap
explainer = shap.KernelExplainer(knn.predict, shap.sample(X_train))  # 使用样本数据以减少计算时间
shap_values = explainer.shap_values(shap.sample(X_train,50))
shap.summary_plot(shap_values)
tree_explainer=shap.TreeExplainer(ctree,X_train)
shap_values = tree_explainer.shap_values(X_train)[:,:,0]
shap.summary_plot(shap_values)








