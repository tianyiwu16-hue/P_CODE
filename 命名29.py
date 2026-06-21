import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# ==========================================
# 1. 使用 pandas 加载/模拟读取 CSV 数据
# ==========================================
# 加载 sklearn 自带的红酒数据集
wine_datasets = load_wine()

# 统一字段名称（对应题目要求的字段）
columns = [
    'Alcohol', 'Malic acid', 'Ash', 'Alcalinity of ash', 'Magnesium',
    'Total phenols', 'Flavanoids', 'Nonflavanoid phenols', 'Proanthocyanins',
    'Color intensity', 'Hue', 'OD280/OD315 of diluted wines', 'Proline'
]

# 构建 DataFrame
df = pd.DataFrame(wine_datasets.data, columns=columns)
# 将类别标签加入 DataFrame（Wine数据集默认标签为0, 1, 2，这里加1对应常规的1, 2, 3类）
df['class'] = wine_datasets.target + 1

# 注：如果您有实际的 CSV 文件，请使用下方这行代码替换上述加载过程：
# df = pd.read_csv('your_wine_data.csv')

print("--- 数据集前5行预览 ---")
print(df.head())
print("-" * 50)

# ==========================================
# 2. 将 class 作为标签 y，其余作为特征 X
# ==========================================
X = df.drop(columns=['class'])
y = df['class']

# ==========================================
# 3. 划分训练集和测试集
# ==========================================
# test_size=0.3 表示 30% 作为测试集，70% 作为训练集；random_state=42 确保结果可复现
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# ==========================================
# 4 & 5. 使用 sklearn 实现 ID3 决策树
# ==========================================
# sklearn的DecisionTreeClassifier通过设置 criterion='entropy' 来使用信息熵进行分裂（即ID3/C4.5的思想）
clf = DecisionTreeClassifier(criterion='entropy', random_state=42)
clf.fit(X_train, y_train)

# 使用测试集进行预测
y_pred = clf.predict(X_test)

# ==========================================
# 6. 输出模型评估结果
# ==========================================
# 计算模型准确率
accuracy = accuracy_score(y_test, y_pred)
print(f"模型准确率 Accuracy: {accuracy:.4f}\n")

# 输出分类报告
print("分类报告 classification_report:")
print(classification_report(y_test, y_pred))

# 输出混淆矩阵
cm = confusion_matrix(y_test, y_pred)
print("混淆矩阵 confusion_matrix:")
print(cm)
print("-" * 50)

# ==========================================
# 7. 可视化：决策树与混淆矩阵热力图
# ==========================================
# 设置 matplotlib 支持中文显示
plt.rcParams['font.sans-serif'] = ['SimHei']  # Windows系统常用黑体
plt.rcParams['axes.unicode_minus'] = False    # 正常显示负号

# 绘制决策树
plt.figure(figsize=(16, 10), dpi=100)
plot_tree(
    clf, 
    feature_names=X.columns, 
    class_names=['Class 1', 'Class 2', 'Class 3'], 
    filled=True, 
    rounded=True,
    fontsize=10
)
plt.title("基于信息熵(ID3思想)的决策树可视化", fontsize=16)
plt.show()

# 绘制混淆矩阵热力图
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=['Class 1', 'Class 2', 'Class 3'],
            yticklabels=['Class 1', 'Class 2', 'Class 3'])
plt.xlabel('预测类别 (Predicted)')
plt.ylabel('真实类别 (Actual)')
plt.title('混淆矩阵热力图')
plt.show()

# ==========================================
# 8. 给出每个特征的重要性排序
# ==========================================
importances = clf.feature_importances_
feature_imp_df = pd.DataFrame({
    'Feature': X.columns,
    'Importance': importances
}).sort_values(by='Importance', ascending=False) # 按重要性降序排列

print("特征重要性排序:")
print(feature_imp_df.to_string(index=False))















import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# ==========================================
# 1. 使用 pandas 加载/模拟读取 CSV 数据
# ==========================================
wine_datasets = load_wine()

# 统一字段名称（对应题目要求的字段）
columns = [
    'Alcohol', 'Malic acid', 'Ash', 'Alcalinity of ash', 'Magnesium',
    'Total phenols', 'Flavanoids', 'Nonflavanoid phenols', 'Proanthocyanins',
    'Color intensity', 'Hue', 'OD280/OD315 of diluted wines', 'Proline'
]

# 构建 DataFrame
df = pd.DataFrame(wine_datasets.data, columns=columns)
# 将类别标签加入 DataFrame（调整为 1, 2, 3 类）
df['class'] = wine_datasets.target + 1

# 注：如果您有实际的 CSV 文件，请使用下方这行代码替换上述加载过程：
# df = pd.read_csv('your_wine_data.csv')

print("--- 数据集前5行预览 ---")
print(df.head())
print("-" * 50)

# ==========================================
# 2. 将 class 列作为标签 y，其余作为特征 X
# ==========================================
X = df.drop(columns=['class'])
y = df['class']

# ==========================================
# 4. 划分训练集和测试集 (test_size=0.3, random_state=42)
# ==========================================
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# ==========================================
# 3. 对特征进行标准化（StandardScaler）
# ==========================================
# 逻辑回归对特征尺度敏感，标准化能加速收敛并使回归系数具有可比性
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test) # 测试集使用与训练集相同的特征缩放标准

# ==========================================
# 5 & 6. 使用 sklearn.linear_model.LogisticRegression
# ==========================================
# 指定多分类模式为 'multinomial'（Softmax回归），求解器为 'lbfgs'
# 删掉 multi_class 参数，新版 sklearn 会根据数据集的类别数自动启用多分类 multinomial 模式
lr_model = LogisticRegression(solver='lbfgs', random_state=42, max_iter=1000)
lr_model.fit(X_train_scaled, y_train)

# 使用标准化后的测试集进行预测
y_pred = lr_model.predict(X_test_scaled)

# ==========================================
# 7. 输出模型评估结果
# ==========================================
accuracy = accuracy_score(y_test, y_pred)
print(f"模型准确率 Accuracy: {accuracy:.4f}\n")

print("分类报告 classification_report:")
print(classification_report(y_test, y_pred))

cm = confusion_matrix(y_test, y_pred)
print("混淆矩阵 confusion_matrix:")
print(cm)
print("-" * 50)

# ==========================================
# 8 & 9. 输出与可视化：特征系数分析
# ==========================================
# 设置 matplotlib 支持中文显示
plt.rcParams['font.sans-serif'] = ['SimHei']  
plt.rcParams['axes.unicode_minus'] = False    

# 混淆矩阵热力图绘制
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Oranges',
            xticklabels=['Class 1', 'Class 2', 'Class 3'],
            yticklabels=['Class 1', 'Class 2', 'Class 3'])
plt.xlabel('预测类别 (Predicted)')
plt.ylabel('真实类别 (Actual)')
plt.title('逻辑回归 - 混淆矩阵热力图')
plt.show()

# 提取各分类的逻辑回归系数 (形状为 [n_classes, n_features])
coefficients = lr_model.coef_
class_names = ['Class 1', 'Class 2', 'Class 3']

# 创建系数 DataFrame 方便分析与绘图
coef_df = pd.DataFrame(coefficients, columns=X.columns, index=class_names).T

print("每个类别对应的特征系数（已排序输出）：")
for col in coef_df.columns:
    print(f"\n[{col}] 的特征系数排序 (从正向影响到负向影响):")
    # 按系数数值降序排列
    sorted_coef = coef_df[col].sort_values(ascending=False)
    for feature, coef_val in sorted_coef.items():
        print(f"  {feature:<30} : {coef_val:>8.4f}")
print("-" * 50)

# 特征系数可视化 (柱状图)
coef_df.plot(kind='barh', figsize=(14, 8), width=0.8)
plt.axvline(x=0, color='red', linestyle='--', linewidth=0.8) # 绘制0分界线
plt.title('多分类逻辑回归 - 各类别特征系数可视化', fontsize=16)
plt.xlabel('系数大小 (Coefficient Value)', fontsize=12)
plt.ylabel('特征名称 (Features)', fontsize=12)
plt.grid(axis='x', linestyle=':', alpha=0.6)
plt.legend(title='预测目标类别')
plt.tight_layout()
plt.show()

























