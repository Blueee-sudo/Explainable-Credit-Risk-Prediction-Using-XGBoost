import os
import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest, mutual_info_classif
from sklearn.preprocessing import FunctionTransformer
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
import joblib
import matplotlib.pyplot as plt

# 设置中文字体（解决中文显示问题）
plt.rcParams["font.family"] = ["SimHei", "Microsoft YaHei", "Arial Unicode MS"]
plt.rcParams["axes.unicode_minus"] = False

# 获取当前脚本所在目录
script_dir = os.path.dirname(os.path.abspath(__file__))
data_dir = os.path.join(script_dir, "..", "..", "..", "data", "processed")
models_dir = os.path.join(script_dir, "..", "..", "..", "models")
reports_dir = os.path.join(script_dir, "..", "..", "..", "reports", "figures")

# 构建文件路径
X_path = os.path.join(data_dir, "feature_engineered_data.csv")
y_path = os.path.join(data_dir, "target.csv")
model_path = os.path.join(models_dir, "feature_pyramid_xgboost.pkl")
feature_pyramid_path = os.path.join(models_dir, "feature_pyramid.pkl")
fig_path = os.path.join(reports_dir, "feature_pyramid_structure.png")

# 确保目录存在
for dir_path in [data_dir, models_dir, reports_dir]:
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)
        print(f"已创建目录: {dir_path}")

# 1. 加载数据（带异常处理）
try:
    X = pd.read_csv(X_path)
    y = pd.read_csv(y_path).squeeze()
    print("数据加载成功！")
except FileNotFoundError as e:
    print(f"错误：文件不存在 - {e}")
    print(f"请检查路径：{data_dir}")
    import sys
    sys.exit(1)

# 2. 定义特征金字塔网络
# 低级特征：原始特征的子集（选择部分数值特征）
low_level_features = [
    "RevolvingUtilizationOfUnsecuredLines",
    "age",
    "DebtRatio",
    "MonthlyIncome"
]

# 定义一个可序列化的函数替代lambda
def select_low_level_features(X):
    return X[low_level_features]

# 中级特征：通过互信息选择的Top K特征
mid_level_selector = SelectKBest(mutual_info_classif, k=8)

# 高级特征：通过PCA提取的低维表示
high_level_pca = PCA(n_components=5)

# 构建特征金字塔
feature_pyramid = FeatureUnion([
    ("low_level", FunctionTransformer(select_low_level_features)),  # 使用具名函数
    ("mid_level", mid_level_selector),
    ("high_level", high_level_pca)
])

# 3. 训练特征金字塔并转换数据
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

feature_pyramid.fit(X_train, y_train)
X_train_pyramid = feature_pyramid.transform(X_train)
X_test_pyramid = feature_pyramid.transform(X_test)

# 4. 训练XGBoost模型
xgb_params = {
    "objective": "binary:logistic",
    "eval_metric": "auc",
    "max_depth": 5,
    "learning_rate": 0.1,
    "n_estimators": 100,
    "random_state": 42
}

pyramid_xgb_model = xgb.XGBClassifier(**xgb_params)
pyramid_xgb_model.fit(X_train_pyramid, y_train)

# 5. 评估模型
y_pred_proba = pyramid_xgb_model.predict_proba(X_test_pyramid)[:, 1]
roc_auc = roc_auc_score(y_test, y_pred_proba)
print(f"特征金字塔XGBoost ROC-AUC: {roc_auc:.4f}")

# 6. 保存模型和特征处理器
joblib.dump(pyramid_xgb_model, model_path)
joblib.dump(feature_pyramid, feature_pyramid_path)
print(f"模型已保存至: {model_path}")
print(f"特征金字塔处理器已保存至: {feature_pyramid_path}")

# 7. 可视化特征金字塔结构
try:
    plt.figure(figsize=(10, 6))
    plt.bar(
        ["低级特征", "中级特征", "高级特征"],
        [len(low_level_features), mid_level_selector.k, high_level_pca.n_components]
    )
    plt.title("特征金字塔各层级特征数量")
    plt.xlabel("特征层级")
    plt.ylabel("特征数量")
    plt.tight_layout()
    plt.savefig(fig_path)
    plt.close()
    print(f"特征金字塔结构图已保存至: {fig_path}")
except Exception as e:
    print(f"可视化时出错: {e}")