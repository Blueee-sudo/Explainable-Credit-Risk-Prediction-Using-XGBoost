import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from sklearn.metrics import roc_curve, auc, confusion_matrix, classification_report
import torch
import logging
from src.models.model_structure_improvements.attention_xgboost import AttentionLayer

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# 解决字体显示问题
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans', 'Arial']  # 多字体备选
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['font.family'] = 'sans-serif'

# 1. 加载数据
try:
    X_test = pd.read_csv("D:/credit_score_project/data/processed/X_test.csv")
    y_test = pd.read_csv("D:/credit_score_project/data/processed/y_test.csv").squeeze()
    logger.info("测试数据加载成功，特征数: %d, 样本数: %d", X_test.shape[1], X_test.shape[0])
except Exception as e:
    logger.error("加载测试数据失败: %s", str(e))
    exit(1)

# 2. 加载基准模型
try:
    base_model = joblib.load("D:/credit_score_project/models/base_xgboost.pkl")
    base_pred_proba = base_model.predict_proba(X_test)[:, 1]
    logger.info("基准模型加载成功")
except Exception as e:
    logger.error("加载基准模型失败: %s", str(e))
    base_model = None
    base_pred_proba = np.zeros(len(y_test))

# 3. 加载注意力增强模型
try:
    attention_model = AttentionLayer(X_test.shape[1])
    attention_model.load_state_dict(torch.load("D:/credit_score_project/models/attention_weights.pth"))
    attention_model.eval()

    X_test_tensor = torch.FloatTensor(X_test.values)
    with torch.no_grad():
        X_test_attn = attention_model(X_test_tensor).numpy()

    attn_xgb_model = joblib.load("D:/credit_score_project/models/attention_xgboost.pkl")
    attn_pred_proba = attn_xgb_model.predict_proba(X_test_attn)[:, 1]
    logger.info("注意力增强模型加载成功")
except Exception as e:
    logger.error("加载注意力增强模型失败: %s", str(e))
    attention_model = None
    attn_pred_proba = np.zeros(len(y_test))

# 4. 特征金字塔模型 - 跳过加载
logger.warning("特征金字塔模型由于特征工程不一致被跳过")

# 5. 绘制ROC曲线对比
plt.figure(figsize=(10, 8))

# 基准模型
if base_model:
    fpr_base, tpr_base, _ = roc_curve(y_test, base_pred_proba)
    roc_auc_base = auc(fpr_base, tpr_base)
    plt.plot(fpr_base, tpr_base, lw=2, label=f'Base Model (AUC = {roc_auc_base:.4f})')

# 注意力增强模型
if attention_model:
    fpr_attn, tpr_attn, _ = roc_curve(y_test, attn_pred_proba)
    roc_auc_attn = auc(fpr_attn, tpr_attn)
    plt.plot(fpr_attn, tpr_attn, lw=2, label=f'Attention Model (AUC = {roc_auc_attn:.4f})')

# 随机基线
plt.plot([0, 1], [0, 1], 'k--', lw=2)

plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Model ROC Curve Comparison')
plt.legend(loc="lower right")

try:
    plt.savefig("D:/credit_score_project/reports/figures/model_comparison_roc.png")
    logger.info("ROC曲线图保存成功")
except Exception as e:
    logger.error("保存ROC曲线图失败: %s", str(e))

plt.close()

# 6. 绘制混淆矩阵（以注意力模型为例）
if attention_model:
    try:
        y_pred_attn = attn_xgb_model.predict(X_test_attn)
        cm = confusion_matrix(y_test, y_pred_attn)

        # 计算百分比表示
        cm_percent = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis] * 100

        plt.figure(figsize=(10, 8))
        sns.heatmap(cm_percent, annot=True, fmt=".1f", cmap="Blues",
                    xticklabels=["Not Overdue", "Overdue"],
                    yticklabels=["Not Overdue", "Overdue"],
                    cbar_kws={'label': 'Percentage'})
        plt.xlabel('Predicted Label')
        plt.ylabel('True Label')
        plt.title('Confusion Matrix - Attention Model (Percentage)')

        try:
            plt.savefig("D:/credit_score_project/reports/figures/attention_model_confusion_matrix.png")
            logger.info("混淆矩阵图保存成功")
        except Exception as e:
            logger.error("保存混淆矩阵图失败: %s", str(e))

        plt.close()
    except Exception as e:
        logger.error("生成混淆矩阵失败: %s", str(e))

# 7. 输出分类报告
if base_model:
    logger.info("基准模型分类报告:")
    print(classification_report(y_test, base_model.predict(X_test)))

if attention_model:
    try:
        logger.info("\n注意力增强模型分类报告:")
        print(classification_report(y_test, y_pred_attn))
    except NameError:
        y_pred_attn = attn_xgb_model.predict(X_test_attn)
        print(classification_report(y_test, y_pred_attn))

# 8. 改进的特征重要性分析（注意力模型）
if attention_model:
    try:
        # 获取第一层线性权重（跳过激活函数层）
        if hasattr(attention_model, 'attention'):
            # 假设attention模块是一个Sequential: [Linear, Tanh, Linear]
            # 获取第一层线性权重
            weights = attention_model.attention[0].weight.detach().numpy()
            logger.info("第一层注意力权重形状: %s", weights.shape)

            # 计算特征重要性 - 对每个特征在所有隐藏单元上的权重取平均
            if weights.ndim == 2:
                # 权重形状应为 (hidden_units, input_features)
                feature_importance = np.abs(weights).mean(axis=0)

                # 确保特征重要性向量长度与特征数匹配
                if len(feature_importance) != len(X_test.columns):
                    logger.warning("特征重要性向量长度(%d)与特征数(%d)不匹配，使用平均权重",
                                   len(feature_importance), len(X_test.columns))
                    feature_importance = np.ones(len(X_test.columns)) / len(X_test.columns)

                # 排序特征重要性
                sorted_idx = np.argsort(feature_importance)[::-1]
                top_features = min(10, len(X_test.columns))  # 只显示前10个特征

                plt.figure(figsize=(12, 8))
                plt.bar(range(top_features), feature_importance[sorted_idx][:top_features])
                plt.xticks(range(top_features), X_test.columns[sorted_idx][:top_features], rotation=45, ha='right')
                plt.title("Top Feature Importance - Attention Model")
                plt.xlabel("Feature Names")
                plt.ylabel("Attention Weight")
                plt.tight_layout()

                try:
                    plt.savefig("D:/credit_score_project/reports/figures/attention_feature_importance.png")
                    logger.info("注意力特征重要性图保存成功")
                except Exception as e:
                    logger.error("保存注意力特征重要性图失败: %s", str(e))

                plt.close()
            else:
                logger.warning("无法处理非二维权重")
        else:
            logger.warning("模型中没有找到注意力层")
    except Exception as e:
        logger.error("生成注意力特征重要性图失败: %s", str(e))

# 9. 基准模型特征重要性（使用SHAP值）
if base_model:
    try:
        import shap

        logger.info("计算基准模型的SHAP值...")

        # 创建SHAP解释器
        explainer = shap.TreeExplainer(base_model)

        # 计算SHAP值（使用部分样本加快计算）
        sample_indices = np.random.choice(len(X_test), size=min(1000, len(X_test)), replace=False)
        X_sample = X_test.iloc[sample_indices]
        shap_values = explainer.shap_values(X_sample)

        # 绘制汇总图
        plt.figure(figsize=(12, 8))
        shap.summary_plot(shap_values, X_sample, plot_type="bar", show=False)
        plt.title("Feature Importance (SHAP) - Base Model")
        plt.tight_layout()

        try:
            plt.savefig("D:/credit_score_project/reports/figures/base_feature_importance_shap.png")
            logger.info("基准模型SHAP特征重要性图保存成功")
        except Exception as e:
            logger.error("保存SHAP特征重要性图失败: %s", str(e))

        plt.close()

        # 绘制详细SHAP值图
        plt.figure(figsize=(12, 8))
        shap.summary_plot(shap_values, X_sample, show=False)
        plt.title("SHAP Values - Base Model")
        plt.tight_layout()

        try:
            plt.savefig("D:/credit_score_project/reports/figures/base_shap_values.png")
            logger.info("基准模型SHAP值图保存成功")
        except Exception as e:
            logger.error("保存SHAP值图失败: %s", str(e))

        plt.close()

    except ImportError:
        logger.warning("SHAP库未安装，使用内置特征重要性")
        if hasattr(base_model, 'feature_importances_'):
            try:
                feature_importances = base_model.feature_importances_
                sorted_idx = np.argsort(feature_importances)[::-1]
                top_features = min(10, len(X_test.columns))

                plt.figure(figsize=(12, 8))
                plt.bar(range(top_features), feature_importances[sorted_idx][:top_features])
                plt.xticks(range(top_features), X_test.columns[sorted_idx][:top_features], rotation=45, ha='right')
                plt.title("Top Feature Importance - Base Model")
                plt.xlabel("Feature Names")
                plt.ylabel("Importance Score")
                plt.tight_layout()

                try:
                    plt.savefig("D:/credit_score_project/reports/figures/base_feature_importance.png")
                    logger.info("基准模型特征重要性图保存成功")
                except Exception as e:
                    logger.error("保存基准模型特征重要性图失败: %s", str(e))

                plt.close()
            except Exception as e:
                logger.error("生成基准模型特征重要性图失败: %s", str(e))
        else:
            logger.warning("基准模型没有特征重要性属性")

# 10. 模型性能对比表
if base_model and attention_model:
    try:
        from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

        # 计算基准模型指标
        base_pred = base_model.predict(X_test)
        base_metrics = {
            "Accuracy": accuracy_score(y_test, base_pred),
            "Precision": precision_score(y_test, base_pred),
            "Recall": recall_score(y_test, base_pred),
            "F1 Score": f1_score(y_test, base_pred),
            "AUC": roc_auc_score(y_test, base_pred_proba)
        }

        # 计算注意力模型指标
        attn_metrics = {
            "Accuracy": accuracy_score(y_test, y_pred_attn),
            "Precision": precision_score(y_test, y_pred_attn),
            "Recall": recall_score(y_test, y_pred_attn),
            "F1 Score": f1_score(y_test, y_pred_attn),
            "AUC": roc_auc_score(y_test, attn_pred_proba)
        }

        # 创建对比DataFrame
        metrics_df = pd.DataFrame({
            "Base Model": base_metrics,
            "Attention Model": attn_metrics
        })

        # 保存为CSV
        metrics_df.to_csv("D:/credit_score_project/reports/model_comparison_metrics.csv")
        logger.info("模型指标对比表保存成功")

        # 打印表格
        print("\n模型性能对比:")
        print(metrics_df)

    except Exception as e:
        logger.error("生成模型性能对比失败: %s", str(e))

logger.info("模型比较完成")