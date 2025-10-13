import pandas as pd
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
import warnings

warnings.filterwarnings("once")  # 仅显示一次警告，避免重复提示


def preprocess_data(raw_data):
    data = raw_data.copy()

    # 1. 缺失值填充（优化参数）
    features_for_impute = [col for col in data.columns if col != "MonthlyIncome"]
    imputer = IterativeImputer(
        estimator=RandomForestRegressor(n_estimators=8, random_state=42),
        max_iter=15,  # 增加迭代次数
        random_state=42,
        tol=1e-4  # 降低收敛阈值
    )
    print("开始填充缺失值...")
    data_imputed = imputer.fit_transform(data)
    data = pd.DataFrame(data_imputed, columns=data.columns)

    # 2. 异常值处理
    data = data[(data["age"] > 0) & (data["age"] < 120)]

    # 3. 特征标准化
    scaler = StandardScaler()
    numeric_features = [col for col in data.columns if col != "SeriousDlqin2yrs"]
    data[numeric_features] = scaler.fit_transform(data[numeric_features])

    return data


# 执行预处理
raw_data = pd.read_csv("/data/raw/cs-training.csv", index_col=0)
processed_data = preprocess_data(raw_data)
processed_data.to_csv("D:/credit_score_project/data/processed/processed_data.csv", index=False)
print("预处理完成，结果已保存")