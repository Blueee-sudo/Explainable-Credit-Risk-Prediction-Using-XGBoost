import pandas as pd
import joblib
from imblearn.combine import SMOTEENN
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import EditedNearestNeighbours
from sklearn.model_selection import train_test_split

# 加载特征工程后的数据
X = pd.read_csv("/data/processed/feature_engineered_data.csv")
y = pd.read_csv("/data/processed/target.csv").squeeze()

# 划分训练集/测试集
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

# 创建SMOTE和ENN的独立实例，并设置参数
smote = SMOTE(random_state=42, k_neighbors=5)  # SMOTE参数
enn = EditedNearestNeighbours(n_neighbors=3)   # ENN参数

# 应用SMOTE+ENN混合采样
smote_enn = SMOTEENN(
    random_state=42,
    smote=smote,  # 传入SMOTE实例
    enn=enn       # 传入ENN实例
)

X_resampled, y_resampled = smote_enn.fit_resample(X_train, y_train)

# 保存增强后的数据
pd.DataFrame(X_resampled, columns=X.columns).to_csv(
    "/data/processed/smoteenn_X_train.csv", index=False
)
pd.DataFrame(y_resampled, columns=["SeriousDlqin2yrs"]).to_csv(
    "/data/processed/smoteenn_y_train.csv", index=False
)
pd.DataFrame(X_test, columns=X.columns).to_csv(
    "/data/processed/X_test.csv", index=False
)
pd.DataFrame(y_test, columns=["SeriousDlqin2yrs"]).to_csv(
    "/data/processed/y_test.csv", index=False
)

# 分析增强前后的类别分布
print("增强前类别分布:")
print(y_train.value_counts())
print("\n增强后类别分布:")
print(y_resampled.value_counts())