import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# ====================== 1. 改进版 Softmax Logistic Regression ======================
class SoftmaxLogisticRegression:
    def __init__(self, learning_rate=0.01, epochs=2000, reg_lambda=0.1, random_state=42):
        self.lr = learning_rate
        self.epochs = epochs
        self.reg_lambda = reg_lambda  # L2 正则化系数
        self.random_state = random_state
        self.weights = None
        self.bias = None
        self.loss_history = []

    def _softmax(self, z):
        # 减去最大值防止 exp(z) 溢出 (数值稳定性优化)
        exp_z = np.exp(z - np.max(z, axis=1, keepdims=True))
        return exp_z / np.sum(exp_z, axis=1, keepdims=True)

    def fit(self, X, y):
        np.random.seed(self.random_state)
        n_samples, n_features = X.shape
        n_classes = len(np.unique(y))
        
        # One-hot 编码
        y_onehot = np.eye(n_classes)[y]
        
        # He 初始化思想 (适用于线性层)
        self.weights = np.random.randn(n_features, n_classes) * 0.01
        self.bias = np.zeros(n_classes)
        
        for epoch in range(self.epochs):
            # 前向传播
            scores = X @ self.weights + self.bias
            probs = self._softmax(scores)
            
            # 计算交叉熵损失 + L2 正则化
            loss = -np.mean(np.sum(y_onehot * np.log(probs + 1e-15), axis=1))
            l2_penalty = (self.reg_lambda / (2 * n_samples)) * np.sum(np.square(self.weights))
            total_loss = loss + l2_penalty
            self.loss_history.append(total_loss)
            
            # 梯度计算 (含正则化梯度)
            dw = (1 / n_samples) * (X.T @ (probs - y_onehot)) + (self.reg_lambda / n_samples) * self.weights
            db = (1 / n_samples) * np.sum(probs - y_onehot, axis=0)
            
            # 参数更新
            self.weights -= self.lr * dw
            self.bias -= self.lr * db
            
            if epoch % 500 == 0:
                print(f"Epoch {epoch:4d} | Loss: {total_loss:.4f}")

    def predict(self, X):
        scores = X @ self.weights + self.bias
        return np.argmax(self._softmax(scores), axis=1)

# ====================== 2. 改进版 ID3 决策树 (支持连续值) ======================
class ID3DecisionTree:
    def __init__(self, max_depth=5, min_samples_split=10, min_impurity_decrease=1e-7):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_impurity_decrease = min_impurity_decrease
        self.tree = None

    def _entropy(self, y):
        if len(y) == 0: return 0
        probs = np.bincount(y) / len(y)
        return -np.sum([p * np.log2(p) for p in probs if p > 0])

    def _best_split(self, X, y):
        best_gain = -1
        split_idx, split_thresh = None, None
        parent_entropy = self._entropy(y)
        
        n_features = X.shape[1]
        for feat_idx in range(n_features):
            # 优化：对特征值排序，取中间值作为候选分裂点
            feat_values = X[:, feat_idx]
            unique_values = np.unique(feat_values)
            
            # 如果特征值太多，采样 20 个候选项以提高效率
            if len(unique_values) > 20:
                thresholds = np.percentile(unique_values, np.linspace(5, 95, 20))
            else:
                thresholds = (unique_values[:-1] + unique_values[1:]) / 2

            for thresh in thresholds:
                left_mask = feat_values <= thresh
                y_l, y_r = y[left_mask], y[~left_mask]
                
                if len(y_l) < 2 or len(y_r) < 2: continue
                
                # 计算信息增益
                n = len(y)
                child_entropy = (len(y_l)/n) * self._entropy(y_l) + (len(y_r)/n) * self._entropy(y_r)
                gain = parent_entropy - child_entropy
                
                if gain > best_gain:
                    best_gain, split_idx, split_thresh = gain, feat_idx, thresh
                    
        return split_idx, split_thresh, best_gain

    def _build_tree(self, X, y, depth=0):
        n_samples, n_labels = len(y), len(np.unique(y))
        
        # 停止条件
        if n_labels == 1 or n_samples < self.min_samples_split or depth >= self.max_depth:
            return np.bincount(y).argmax()
        
        feat, thresh, gain = self._best_split(X, y)
        
        # 如果增益太小，不再分裂
        if feat is None or gain < self.min_impurity_decrease:
            return np.bincount(y).argmax()
        
        left_idx = X[:, feat] <= thresh
        left_subtree = self._build_tree(X[left_idx], y[left_idx], depth + 1)
        right_subtree = self._build_tree(X[~left_idx], y[~left_idx], depth + 1)
        
        return {'feature': feat, 'threshold': thresh, 'left': left_subtree, 'right': right_subtree}

    def fit(self, X, y):
        self.tree = self._build_tree(X, y)

    def _predict_one(self, x, tree):
        if not isinstance(tree, dict): return tree
        if x[tree['feature']] <= tree['threshold']:
            return self._predict_one(x, tree['left'])
        return self._predict_one(x, tree['right'])

    def predict(self, X):
        return np.array([self._predict_one(x, self.tree) for x in X])

# ====================== 3. 运行与评估 ======================
if __name__ == "__main__":
    # 加载数据
    data = load_breast_cancer()
    X, y = data.data, data.target
    
    # 标准化 (对逻辑回归至关重要)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, random_state=42, stratify=y
    )

    # --- 训练逻辑回归 ---
    lr_model = SoftmaxLogisticRegression(learning_rate=0.1, epochs=1500, reg_lambda=0.5)
    lr_model.fit(X_train, y_train)
    y_pred_lr = lr_model.predict(X_test)

    # --- 训练决策树 ---
    dt_model = ID3DecisionTree(max_depth=5, min_samples_split=10)
    dt_model.fit(X_train, y_train)
    y_pred_dt = dt_model.predict(X_test)

    # --- 结果展示 ---
    fig, ax = plt.subplots(1, 2, figsize=(14, 5))

    # 1. Loss 曲线
    ax[0].plot(lr_model.loss_history)
    ax[0].set_title("Softmax LR Training Loss")
    ax[0].set_xlabel("Epochs")
    ax[0].set_ylabel("Cross Entropy Loss")

    # 2. 混淆矩阵 (以决策树为例)
    cm = confusion_matrix(y_test, y_pred_dt)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax[1],
                xticklabels=data.target_names, yticklabels=data.target_names)
    ax[1].set_title("Decision Tree Confusion Matrix")
    
    plt.tight_layout()
    plt.show()

    print("\n" + "="*30)
    print(f"Softmax LR Accuracy: {accuracy_score(y_test, y_pred_lr):.4f}")
    print(f"Decision Tree Accuracy: {accuracy_score(y_test, y_pred_dt):.4f}")
    print("="*30)
    print("\nDecision Tree Classification Report:\n", classification_report(y_test, y_pred_dt))