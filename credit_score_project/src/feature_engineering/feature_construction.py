import pandas as pd
from sklearn.feature_selection import SelectKBest, mutual_info_classif

def create_new_features(data):
    data = data.copy()
    # 债务收入比（避免除零错误）
    data["DebtIncomeRatio"] = data["DebtRatio"] / data["MonthlyIncome"].replace(0, 1e-6)
    # 逾期频率（假设统计周期为12个月）
    data["LatePaymentFrequency"] = data["NumberOfTime30-59DaysPastDueNotWorse"] / 12
    return data

def select_features(X, y, k=15):
    # 基于互信息选择Top K特征
    selector = SelectKBest(mutual_info_classif, k=k)
    X_selected = selector.fit_transform(X, y)
    selected_features = X.columns[selector.get_support()]
    return X_selected, selected_features

# 执行特征工程
processed_data = pd.read_csv("/data/processed/processed_data.csv")
data_with_features = create_new_features(processed_data)
X = data_with_features.drop("SeriousDlqin2yrs", axis=1)
y = data_with_features["SeriousDlqin2yrs"]
X_selected, selected_features = select_features(X, y, k=15)

# 保存特征工程后的数据
pd.DataFrame(X_selected, columns=selected_features).to_csv(
    "/data/processed/feature_engineered_data.csv", index=False
)
y.to_csv("D:/credit_score_project/data/processed/target.csv", index=False)
