import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import xgboost as xgb
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
import joblib
import matplotlib.pyplot as plt
import os

# 设置中文字体
plt.rcParams["font.family"] = ["SimHei", "Microsoft YaHei", "Arial Unicode MS"]
plt.rcParams["axes.unicode_minus"] = False

# 获取当前脚本所在目录
script_dir = os.path.dirname(os.path.abspath(__file__))
data_dir = os.path.join(script_dir, "..", "..", "..", "data", "processed")
models_dir = os.path.join(script_dir, "..", "..", "..", "models")
reports_dir = os.path.join(script_dir, "..", "..", "..", "reports", "figures")

# 构建文件路径
X_train_path = os.path.join(data_dir, "feature_engineered_data.csv")
y_train_path = os.path.join(data_dir, "target.csv")
X_test_path = os.path.join(data_dir, "X_test.csv")
y_test_path = os.path.join(data_dir, "y_test.csv")
model_path = os.path.join(models_dir, "attention_xgboost.pkl")
attn_weights_path = os.path.join(models_dir, "attention_weights.pth")
fig_path = os.path.join(reports_dir, "attention_weights.png")

# 确保目录存在
for dir_path in [data_dir, models_dir, reports_dir]:
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)
        print(f"已创建目录: {dir_path}")

# 加载数据（带异常处理）
try:
    X_train = pd.read_csv(X_train_path).values
    y_train = pd.read_csv(y_train_path).values.ravel()
    X_test = pd.read_csv(X_test_path).values
    y_test = pd.read_csv(y_test_path).values.ravel()
    print("数据加载成功！")
except FileNotFoundError as e:
    print(f"错误：文件不存在 - {e}")
    print(f"请检查路径：{data_dir}")
    import sys
    sys.exit(1)

# 1. 定义注意力层（返回注意力权重）
class AttentionLayer(nn.Module):
    def __init__(self, input_dim):
        super(AttentionLayer, self).__init__()
        self.attention = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, input_dim),
            nn.Sigmoid()  # 确保权重在[0,1]范围内
        )

    def forward(self, x):
        attn_weights = self.attention(x)
        return attn_weights  # 直接返回注意力权重

# 2. 转换为PyTorch张量并归一化
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
X_train_norm = scaler.fit_transform(X_train)
X_test_norm = scaler.transform(X_test)
X_train_tensor = torch.FloatTensor(X_train_norm)
y_train_tensor = torch.FloatTensor(y_train)
X_test_tensor = torch.FloatTensor(X_test_norm)

# 创建数据加载器
train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)

# 3. 训练注意力层
input_dim = X_train.shape[1]
attention_model = AttentionLayer(input_dim)
criterion = nn.BCELoss()  # 归一化后输入在[0,1]，可使用BCELoss
optimizer = optim.Adam(attention_model.parameters(), lr=0.001)

# 训练循环
num_epochs = 50
for epoch in range(num_epochs):
    running_loss = 0.0
    for inputs, _ in train_loader:
        # 前向传播
        outputs = attention_model(inputs)

        # 计算损失
        loss = criterion(outputs, inputs)  # 现在计算权重与输入的差异

        # 反向传播与优化
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    if (epoch + 1) % 10 == 0:
        print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {running_loss / len(train_loader):.4f}")

# 4. 获取注意力加权后的特征
with torch.no_grad():
    X_train_attn = attention_model(X_train_tensor).numpy() * X_train_norm  # 应用权重到归一化后的特征
    X_test_attn = attention_model(X_test_tensor).numpy() * X_test_norm

# 5. 训练XGBoost模型
xgb_params = {
    "objective": "binary:logistic",
    "eval_metric": "auc",
    "max_depth": 5,
    "learning_rate": 0.1,
    "n_estimators": 100,
    "random_state": 42
}

attn_xgb_model = xgb.XGBClassifier(**xgb_params)
attn_xgb_model.fit(X_train_attn, y_train)

# 6. 评估模型
y_pred_proba = attn_xgb_model.predict_proba(X_test_attn)[:, 1]
roc_auc = roc_auc_score(y_test, y_pred_proba)
print(f"注意力XGBoost ROC-AUC: {roc_auc:.4f}")

# 7. 保存模型和注意力权重
joblib.dump(attn_xgb_model, model_path)
torch.save(attention_model.state_dict(), attn_weights_path)
print(f"模型已保存至: {model_path}")
print(f"注意力权重已保存至: {attn_weights_path}")

# 8. 可视化注意力权重
try:
    feature_names = pd.read_csv(X_train_path).columns
    # 获取所有样本的注意力权重并取平均
    with torch.no_grad():
        all_attn_weights = attention_model(X_train_tensor)
    avg_attn_weights = all_attn_weights.mean(dim=0).numpy()  # 形状为(input_dim,)

    plt.figure(figsize=(12, 6))
    plt.bar(feature_names, avg_attn_weights)
    plt.xticks(rotation=90)
    plt.title("特征注意力权重分布")
    plt.xlabel("特征名称")
    plt.ylabel("平均注意力权重")
    plt.tight_layout()
    plt.savefig(fig_path)
    plt.close()
    print(f"注意力权重分布图已保存至: {fig_path}")
except Exception as e:
    print(f"可视化注意力权重时出错: {e}")