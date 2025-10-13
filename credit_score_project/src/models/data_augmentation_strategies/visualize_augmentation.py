import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA

# 设置支持中文的字体
plt.rcParams["font.family"] = ["SimHei", "WenQuanYi Micro Hei", "Heiti TC", "sans-serif"]
plt.rcParams["axes.unicode_minus"] = False  # 确保负号正确显示

# 加载原始数据和增强后的数据
X_train = pd.read_csv("/data/processed/feature_engineered_data.csv")
y_train = pd.read_csv("/data/processed/target.csv").squeeze()

smoteenn_X = pd.read_csv("/data/processed/smoteenn_X_train.csv")
smoteenn_y = pd.read_csv("/data/processed/smoteenn_y_train.csv").squeeze()

ctgan_X = pd.read_csv("/data/processed/ctgan_X_train.csv")
ctgan_y = pd.read_csv("/data/processed/ctgan_y_train.csv").squeeze()

# 修复：使用同一个 PCA 实例进行所有转换
pca = PCA(n_components=2, random_state=42)
pca.fit(X_train)  # 只在原始数据上拟合 PCA

# 原始数据 PCA
X_train_pca = pca.transform(X_train)
train_pca_df = pd.DataFrame(X_train_pca, columns=["PC1", "PC2"])
train_pca_df["target"] = y_train

# SMOTE+ENN 增强数据 PCA
smoteenn_pca = pca.transform(smoteenn_X)
smoteenn_pca_df = pd.DataFrame(smoteenn_pca, columns=["PC1", "PC2"])
smoteenn_pca_df["target"] = smoteenn_y

# CTGAN 增强数据 PCA
ctgan_pca = pca.transform(ctgan_X)
ctgan_pca_df = pd.DataFrame(ctgan_pca, columns=["PC1", "PC2"])
ctgan_pca_df["target"] = ctgan_y

# 绘制对比图
plt.figure(figsize=(18, 5))

# 原始数据分布
plt.subplot(1, 3, 1)
sns.scatterplot(data=train_pca_df, x="PC1", y="PC2", hue="target", alpha=0.7, s=30)
plt.title("原始数据分布")

# SMOTE+ENN 增强后分布
plt.subplot(1, 3, 2)
sns.scatterplot(data=smoteenn_pca_df, x="PC1", y="PC2", hue="target", alpha=0.7, s=30)
plt.title("SMOTE+ENN 增强后分布")

# CTGAN 增强后分布
plt.subplot(1, 3, 3)
sns.scatterplot(data=ctgan_pca_df, x="PC1", y="PC2", hue="target", alpha=0.7, s=30)
plt.title("CTGAN 增强后分布")

plt.tight_layout()
plt.savefig("D:/credit_score_project/reports/figures/data_augmentation_comparison.png")
plt.close()