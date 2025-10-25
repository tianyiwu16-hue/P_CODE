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
# 导入所需库
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report
# 初始化KNN分类器（默认使用5个邻居）
knn = KNeighborsClassifier(n_neighbors=5)
# 训练模型
knn.fit(X_train, y_train)
# 预测
y_hat=knn.predict(X_train)
y_pred = knn.predict(X_test)
# 评估模型性能
print(f'训练集准确率: {accuracy_score(y_train, y_hat):.4f}')
print(f'测试集准确率: {accuracy_score(y_test, y_pred):.4f}')
print('\n分类报告:')
print(classification_report(y_test, y_pred))





import pandas as pd
from sklearn import linear_model
from sklearn import tree
from ucimlrepo import fetch_ucirepo 
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report

# 加载数据集
spambase = fetch_ucirepo(id=94)   
X = spambase.data.features 
y = spambase.data.targets 

# 划分数据集
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42)

# 数据标准化
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 初始化KNN分类器（默认使用5个邻居）
knn = KNeighborsClassifier(n_neighbors=5)

# 训练模型
knn.fit(X_train_scaled, y_train)

# 预测
y_hat = knn.predict(X_train_scaled)
y_pred = knn.predict(X_test_scaled)

# 评估模型性能
print(f'训练集准确率: {accuracy_score(y_train, y_hat):.4f}')
print(f'测试集准确率: {accuracy_score(y_test, y_pred):.4f}')

print('\n分类报告:')
report = classification_report(y_test, y_pred, target_names=['Not Spam', 'Spam'])
print(report)






# 数据标准化（KNN对特征尺度敏感，建议始终标准化）
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
# 初始化KNN分类器（默认使用5个邻居）
knn = KNeighborsClassifier(n_neighbors=7)
# 训练模型
knn.fit(X_train_scaled, y_train)
# 预测测试集
y_hat=knn.predict(X_train_scaled)
y_pred = knn.predict(X_test_scaled)
# 评估模型性能
print(f'训练集准确率: {accuracy_score(y_train, y_hat):.4f}')
print(f'测试集准确率: {accuracy_score(y_test, y_pred):.4f}')
print('\n分类报告:')
print(classification_report(y_test, y_pred))

knn = KNeighborsClassifier(n_neighbors=5,weights="distance",)
# 训练模型
knn.fit(X_train_scaled, y_train)
# 预测测试集
y_hat=knn.predict(X_train_scaled)
y_pred = knn.predict(X_test_scaled)
# 评估模型性能
print(f'训练集准确率: {accuracy_score(y_train, y_hat):.4f}')
print(f'测试集准确率: {accuracy_score(y_test, y_pred):.4f}')
print('\n分类报告:')
print(classification_report(y_test, y_pred))






# 使用交叉验证寻找最佳k值
from sklearn.model_selection import GridSearchCV

param_grid = {'n_neighbors': range(2, 20),'p':[1,2]}
grid_search = GridSearchCV(KNeighborsClassifier(weights="distance"), 
                           param_grid, cv=5)
grid_search.fit(X_train_scaled, y_train)

print(f'最佳k值: {grid_search.best_params_["n_neighbors"]}')
print(f'最佳p值: {grid_search.best_params_["p"]}')
print(f'最佳验证分数: {grid_search.best_score_:.4f}')

knn = KNeighborsClassifier(n_neighbors=10,weights="distance",p=1)
# 训练模型
knn.fit(X_train_scaled, y_train)
# 预测测试集
y_hat=knn.predict(X_train_scaled)
y_pred = knn.predict(X_test_scaled)
# 评估模型性能
print(f'训练集准确率: {accuracy_score(y_train, y_hat):.4f}')
print(f'测试集准确率: {accuracy_score(y_test, y_pred):.4f}')
print('\n分类报告:')
print(classification_report(y_test, y_pred))




##############聚类算法############
import os
import numpy as np
import pandas as pd
from sklearn import cluster
wine=pd.read_csv("D:\原神文件\wine.csv")
X=wine.iloc[:,0:11] #获得属性
kmeans = cluster.KMeans(n_clusters=8).fit(X)
labels=kmeans.labels_ #得到簇标签
np.bincount(labels)
kmeans.cluster_centers_ #得到簇中心
kmeans.inertia_ #得到距离平方

###
from sklearn import metrics
metrics.silhouette_score(X, labels)#计算轮廓系数
metrics.davies_bouldin_score(X, labels)#计算DBI
truelabel=wine['Y']
metrics.rand_score(truelabel, labels)


##############标准化后的聚类##############
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
kmeans = cluster.KMeans(n_clusters=6).fit(X_scaled)
labels=kmeans.labels_ #得到簇标签
metrics.silhouette_score(X_scaled, labels)#计算轮廓系数
metrics.davies_bouldin_score(X_scaled, labels)#计算DBI
truelabel=wine['Y']
metrics.rand_score(truelabel, labels)


import matplotlib.pyplot as plt
k_range = range(2, 11)
sse = []
# 计算每个 K 值对应的聚类误差（平方误差和）
for k in k_range:
    kmeans = cluster.KMeans(n_clusters=k, algorithm='elkan')
    kmeans.fit(X)
    sse.append(kmeans.inertia_)
plt.figure(figsize=(8, 6))
plt.plot(k_range, sse, marker='o')
plt.xlabel('Number of clusters (K)')
plt.ylabel('Sum of Squared Errors (SSE)')
plt.title('Elbow Method for Optimal K')
plt.xticks(k_range)
plt.grid(True)
plt.show()

############基于Kmeans删除异常值#############
kmeans = cluster.KMeans(n_clusters=10).fit(X)
# 计算每个样本到最近簇中心的距离
distances = np.min([np.linalg.norm(X - center, axis=1) for center in kmeans.cluster_centers_], axis=0)
# 设置距离阈值（如 95% 分位数）
threshold = np.percentile(distances, 99)
outliers = X[distances > threshold]
clean_data = X[distances <= threshold]































