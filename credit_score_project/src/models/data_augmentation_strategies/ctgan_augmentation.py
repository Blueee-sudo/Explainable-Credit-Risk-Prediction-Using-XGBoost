import pandas as pd
import joblib
from ctgan import CTGAN
from sklearn.model_selection import train_test_split

# 加载特征工程后的数据
X = pd.read_csv("/data/processed/feature_engineered_data.csv")
y = pd.read_csv("/data/processed/target.csv").squeeze()

# 划分训练集/测试集
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

# 合并特征与目标变量
train_data = pd.concat([X_train, y_train], axis=1)

# 分离少数类数据（目标=1）
minority_data = train_data[train_data["SeriousDlqin2yrs"] == 1]

# 训练CTGAN模型（仅针对少数类）
ctgan = CTGAN(
    epochs=100,                # 训练轮数
    batch_size=500,            # 批次大小
    log_frequency=True,        # 记录频率
    verbose=True,              # 显示训练进度
    generator_dim=(256, 256),  # 生成器网络维度
    discriminator_dim=(256, 256)  # 判别器网络维度
)
ctgan.fit(minority_data)

# 生成合成数据（数量为原始少数类的2倍）
synthetic_data = ctgan.sample(n=len(minority_data) * 2)

# 合并原始数据与合成数据
augmented_data = pd.concat([train_data, synthetic_data], ignore_index=True)

# 分离特征与目标变量
X_train_augmented = augmented_data.drop("SeriousDlqin2yrs", axis=1)
y_train_augmented = augmented_data["SeriousDlqin2yrs"]

# 保存增强后的数据
X_train_augmented.to_csv("D:/credit_score_project/data/processed/ctgan_X_train.csv", index=False)
y_train_augmented.to_csv("D:/credit_score_project/data/processed/ctgan_y_train.csv", index=False)
X_test.to_csv("D:/credit_score_project/data/processed/X_test.csv", index=False)
y_test.to_csv("D:/credit_score_project/data/processed/y_test.csv", index=False)

# 保存CTGAN模型
ctgan.save("D:/credit_score_project/models/ctgan_model.pkl")

# 分析增强前后的类别分布
print("增强前类别分布:")
print(y_train.value_counts())
print("\n增强后类别分布:")
print(y_train_augmented.value_counts())