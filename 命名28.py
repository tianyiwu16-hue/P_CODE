import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# ====================== 1. 逻辑回归 (Softmax 版) ======================
class SoftmaxLogisticRegression:
    def __init__(self, learning_rate=0.01, epochs=2000, reg_lambda=0.1):
        self.lr = learning_rate
        self.epochs = epochs
        self.reg_lambda = reg_lambda
        self.weights = None
        self.bias = None
        self.loss_history = []

    def _softmax(self, z):
        exp_z = np.exp(z - np.max(z, axis=1, keepdims=True))
        return exp_z / np.sum(exp_z, axis=1, keepdims=True)

    def fit(self, X, y):
        n_samples, n_features = X.shape
        n_classes = len(np.unique(y))
        y_onehot = np.eye(n_classes)[y]
        self.weights = np.random.randn(n_features, n_classes) * 0.01
        self.bias = np.zeros(n_classes)
        
        for epoch in range(self.epochs):
            scores = X @ self.weights + self.bias
            probs = self._softmax(scores)
            loss = -np.mean(np.sum(y_onehot * np.log(probs + 1e-15), axis=1))
            self.loss_history.append(loss)
            dw = (1 / n_samples) * (X.T @ (probs - y_onehot)) + (self.reg_lambda / n_samples) * self.weights
            db = (1 / n_samples) * np.sum(probs - y_onehot, axis=0)
            self.weights -= self.lr * dw
            self.bias -= self.lr * db

    def predict(self, X):
        scores = X @ self.weights + self.bias
        return np.argmax(self._softmax(scores), axis=1)

# ====================== 2. 决策树 (ID3 改进版) ======================
class ID3DecisionTree:
    def __init__(self, max_depth=5, min_samples_split=10):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.tree = None

    def _entropy(self, y):
        if len(y) == 0: return 0
        probs = np.bincount(y) / len(y)
        return -np.sum([p * np.log2(p) for p in probs if p > 0])

    def _best_split(self, X, y):
        best_gain = -1
        split_idx, split_thresh = None, None
        parent_entropy = self._entropy(y)
        for feat_idx in range(X.shape[1]):
            vals = np.unique(X[:, feat_idx])
            thresholds = np.percentile(vals, np.linspace(5, 95, 10)) if len(vals) > 10 else vals
            for thresh in thresholds:
                left_mask = X[:, feat_idx] <= thresh
                y_l, y_r = y[left_mask], y[~left_mask]
                if len(y_l) < 2 or len(y_r) < 2: continue
                gain = parent_entropy - (len(y_l)/len(y)*self._entropy(y_l) + len(y_r)/len(y)*self._entropy(y_r))
                if gain > best_gain:
                    best_gain, split_idx, split_thresh = gain, feat_idx, thresh
        return split_idx, split_thresh

    def _build_tree(self, X, y, depth=0):
        if len(np.unique(y)) == 1 or len(y) < self.min_samples_split or depth >= self.max_depth:
            return np.bincount(y).argmax()
        feat, thresh = self._best_split(X, y)
        if feat is None: return np.bincount(y).argmax()
        left_idx = X[:, feat] <= thresh
        return {
            'feat': feat, 'thresh': thresh,
            'left': self._build_tree(X[left_idx], y[left_idx], depth + 1),
            'right': self._build_tree(X[~left_idx], y[~left_idx], depth + 1)
        }

    def fit(self, X, y):
        self.tree = self._build_tree(X, y)

    def _predict_one(self, x, tree):
        if not isinstance(tree, dict): return tree
        return self._predict_one(x, tree['left'] if x[tree['feat']] <= tree['thresh'] else tree['right'])

    def predict(self, X):
        return np.array([self._predict_one(x, self.tree) for x in X])

# ====================== 3. 针对你的 CSV 进行适配 ======================
if __name__ == "__main__":
    df = pd.read_csv(r"D:\桌面应用\cancer.csv")
    
    # 核心修改点 1: 提取标签 'diagnosis'
    # 你的 'diagnosis' 列包含 'M' (恶性) 和 'B' (良性)
    le = LabelEncoder()
    y = le.fit_transform(df['diagnosis'])
    target_names = le.classes_.astype(str) # ['B', 'M']

    # 核心修改点 2: 提取特征
    # 删掉 'id' (无用) 和 'diagnosis' (标签列)
    X = df.drop(columns=['id', 'diagnosis']).values

    print(f"特征形状: {X.shape}") # 应该是 (569, 30) 左右
    print(f"标签类别: {target_names} (B=0, M=1)")

    # 数据划分
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # 逻辑回归标准化
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # 训练
    print("\n--- 正在训练模型 ---")
    lr = SoftmaxLogisticRegression(learning_rate=0.1, epochs=1000)
    lr.fit(X_train_scaled, y_train)
    
    dt = ID3DecisionTree(max_depth=5)
    dt.fit(X_train, y_train)

    # 评估
    y_pred_lr = lr.predict(X_test_scaled)
    y_pred_dt = dt.predict(X_test)

    print("\n" + "="*40)
    print(f"逻辑回归准确率: {accuracy_score(y_test, y_pred_lr):.4f}")
    print(f"决策树准确率:   {accuracy_score(y_test, y_pred_dt):.4f}")
    print("="*40)

    # 可视化
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(lr.loss_history)
    plt.title("LR Training Loss")
    
    plt.subplot(1, 2, 2)
    sns.heatmap(confusion_matrix(y_test, y_pred_dt), annot=True, fmt='d', 
                cmap='Blues', xticklabels=target_names, yticklabels=target_names)
    plt.title("Decision Tree Confusion Matrix")
    plt.show()
    
    
    
    
    
    

    
    
    
    
    
    
    
    
    
    
    
# ====================== 修改后的主程序 ======================
if __name__ == "__main__":
    try:
        # 1. 加载数据
        df = pd.read_csv(r"D:\桌面应用\cancer.csv")
        print("成功加载数据集！")
        
        # --- 调试步骤：打印列名，帮你找到哪一列才是标签 ---
        print("\n你的 CSV 列名有：", df.columns.tolist())
        print("-" * 30)
        # ----------------------------------------------

        # 2. 确定特征 X 和 标签 y
        # 假设你的标签列名字叫 'target' 或者 'diagnosis'。
        # 请根据上面打印出的列名，把下面 ['target'] 改成你实际的标签列名。
        # 如果你确定最后一列就是标签，但报错了，说明那一列数据不对。
        
        # 方案 A：如果你知道标签列的名字（推荐）
        # label_col = 'target'  # <--- 修改这里为实际的列名，比如 'diagnosis'
        # X = df.drop(columns=[label_col]).values 
        # y_raw = df[label_col].values

        # 方案 B：假设最后一列是标签，但先剔除可能的 ID 列（比如第一列是 ID）
        # 这里演示如何剔除第一列 ID，并取最后一列为标签
        X = df.iloc[:, 1:-1].values  # 从第2列开始取到倒数第2列作为特征
        y_raw = df.iloc[:, -1].values # 取最后一列作为标签

        # 3. 标签编码
        le = LabelEncoder()
        y = le.fit_transform(y_raw)
        target_names = le.classes_.astype(str)
        
        # 检查一下分类数量，如果是几百个，说明你选错列了
        n_classes = len(np.unique(y))
        print(f"检测到类别数量: {n_classes} (通常应该是 2)")
        if n_classes > 10:
            print("警告：类别数量过多，你可能选错了列！请检查 CSV 文件。")

        # 4. 划分数据集
        # 如果还是报错，可以暂时去掉 stratify=y，但最好是修好上面的列选择
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y if n_classes < 10 else None
        )
        
        # 5. 预处理与训练 (后续逻辑不变...)
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        print("\n--- 正在训练 Softmax 逻辑回归 ---")
        lr_model = SoftmaxLogisticRegression(learning_rate=0.1, epochs=1500, reg_lambda=0.5)
        lr_model.fit(X_train_scaled, y_train)
        
        print("\n--- 正在训练 ID3 决策树 ---")
        dt_model = ID3DecisionTree(max_depth=5, min_samples_split=10)
        dt_model.fit(X_train, y_train)

        # 6. 输出结果
        print("\n" + "="*40)
        print(f"逻辑回归准确率: {accuracy_score(y_test, lr_model.predict(X_test_scaled)):.4f}")
        print(f"决策树准确率:   {accuracy_score(y_test, dt_model.predict(X_test)):.4f}")
        print("="*40)

    except Exception as e:
        print(f"发生错误: {e}")    
    
    
    
    
    
    
    
    
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)  
    
    
    
    
    
    
    