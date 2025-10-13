import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

plt.rcParams["font.family"] = ["SimHei", "WenQuanYi Micro Hei", "Heiti TC"]

# 加载原始数据
raw_data = pd.read_csv("/data/raw/cs-training.csv", index_col=0)

# 1. 基础信息统计
print("数据维度:", raw_data.shape)
print("\n缺失值统计（按列）:")
print(raw_data.isnull().sum())
print("\n目标变量分布:")
print(raw_data["SeriousDlqin2yrs"].value_counts(normalize=True))  # 计算类别比例

# 2. 目标变量分布可视化
plt.figure(figsize=(6, 4))
sns.countplot(x="SeriousDlqin2yrs", data=raw_data)
plt.title("目标变量类别分布")
plt.savefig("D:/credit_score_project/reports/figures/target_distribution.png")
plt.close()

# 3. 关键特征分布
plt.figure(figsize=(8, 4))
sns.histplot(raw_data["age"], bins=20, kde=True)
plt.title("年龄分布")
plt.savefig("D:/credit_score_project/reports/figures/age_distribution.png")
plt.close()