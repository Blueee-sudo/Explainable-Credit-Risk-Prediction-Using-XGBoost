import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, classification_report, confusion_matrix
import joblib

# 加载特征工程后的数据
X = pd.read_csv("/data/processed/feature_engineered_data.csv")
y = pd.read_csv("/data/processed/target.csv").squeeze()

# 划分训练集/测试集（按时间或随机划分，此处用随机）
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42  # stratify保持类别分布
)

# 定义XGBoost参数（基础配置）
base_params = {
    "objective": "binary:logistic",
    "eval_metric": "auc",
    "max_depth": 5,
    "learning_rate": 0.1,
    "n_estimators": 100,
    "early_stopping_rounds": 20,  # 移到初始化参数中
    "random_state": 42
}

# 训练模型
base_model = xgb.XGBClassifier(**base_params)
base_model.fit(
    X_train,
    y_train,
    eval_set=[(X_test, y_test)],  # 保留验证集设置
    verbose=True  # 显示训练过程
)

# 评估性能
y_pred_proba = base_model.predict_proba(X_test)[:, 1]
roc_auc = roc_auc_score(y_test, y_pred_proba)
print(f"基准模型 ROC-AUC: {roc_auc:.4f}")
print("\n分类报告:")
print(classification_report(y_test, base_model.predict(X_test)))

# 保存模型与评估结果
joblib.dump(base_model, "/models/base_xgboost.pkl")
pd.DataFrame({"roc_auc": [roc_auc]}).to_csv("/reports/metrics/base_model_metrics.csv", index=False)