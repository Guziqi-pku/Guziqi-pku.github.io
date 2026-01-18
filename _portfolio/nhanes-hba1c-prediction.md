---
title: "基于 NHANES 数据的 HbA1c 水平预测与机器学习模型比较"
collection: portfolio
type: "Machine Learning"
permalink: /portfolio/hba1c-prediction
date: 2026-01-17
excerpt: "使用 NHANES 数据构建五种机器学习模型预测 HbA1c 水平，逻辑回归和梯度提升机表现最优（AUC 0.829/0.827）"
header:
  teaser: /images/portfolio/hba1c-prediction/model_roc_curves.png
tags:
  - 机器学习
  - 医疗预测
  - 糖尿病
  - 特征选择
  - 模型比较
tech_stack:
  - name: Python
  - name: Scikit-learn
  - name: SHAP
  - name: Pandas
  - name: Matplotlib
---
**项目背景**
糖尿病是全球性的公共卫生问题，糖化血红蛋白（HbA1c）是评估糖尿病控制情况的重要指标。本项目利用美国国家健康与营养调查（NHANES）2011-2018 年的数据，构建机器学习模型预测 HbA1c 水平，为糖尿病早期筛查和个性化管理提供数据支持。
**数据与方法**
数据来源
数据集：NHANES 2011-2018 四期数据
样本量：多期数据合并后的完整数据集
数据类型：人口学资料、体格测量、实验室检查、问卷调查
目标变量：HbA1c > 5.7% 定义为高 HbA1c 水平
数据处理流程
多源数据整合：合并 BMX（体格测量）、BPX（血压）、BPQ（血压问卷）、DEMO（人口学）、BIOPRO（生化指标）等表格
数据清洗：处理缺失值、异常值，去除胰岛素相关指标以避免数据泄漏
特征工程：计算平均血压、BMI 等衍生特征
数据集划分：按 7:3 比例划分训练集和验证集，保持类别平衡


核心实现
1. LASSO 特征选择
使用 LASSO 回归进行特征选择，从大量候选特征中筛选出最具预测价值的变量。
LASSO CV特征选择
lasso_cv = LogisticRegressionCV(
    Cs=np.logspace(-3, 3, 20),
    penalty='l1',
    solver='saga',
    cv=5,
    scoring='neg_log_loss',
    max_iter=5000,
    random_state=42,
    refit=True
)
lasso_cv.fit(X_train_scaled, y_train_lasso)

提取非零系数特征
coef = lasso_cv.coef_[0]
lasso_selected = lasso_results[lasso_results['Coefficient'] != 0]
top6_vars = lasso_selected.head(6)['Variable'].tolist()

2.多模型构建与训练
使用筛选出的前 6 个重要特征，构建五种机器学习模型（逻辑回归 、支持向量机、梯度提升机、贝叶斯算法和随机森林）进行对比分析。
定义五种机器学习模型
models = {
    'Logistic Regression': Pipeline([
        ('scaler', StandardScaler()), 
        ('model', LogisticRegression(max_iter=3000, random_state=42))
    ]),
    'SVM': Pipeline([
        ('scaler', StandardScaler()), 
        ('model', SVC(kernel='rbf', probability=True, random_state=42))
    ]),
    'Gradient Boosting': GradientBoostingClassifier(random_state=42),
    'Bayesian': GaussianNB(),
    'Random Forest': RandomForestClassifier(n_estimators=300, random_state=42, n_jobs=-1)
}

# 模型训练与评估
auc_results = {}
for name, model in models.items():
    model.fit(X_train_6, y_train)
    y_prob = model.predict_proba(X_val_6)[:, 1]
    auc = roc_auc_score(y_val, y_prob)
    auc_results[name] = auc

3. 模型性能评估
采用 13 种评估指标全面评估模型性能，包括准确率、AUC、F1 分数、MCC 等。
def evaluate_binary_model(y_true, y_prob, threshold=0.5):
    """综合评估二分类模型性能"""
    y_pred = (y_prob >= threshold).astype(int)
    
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    
    return {
        'Accuracy': accuracy_score(y_true, y_pred),
        'Balanced_Accuracy': balanced_accuracy_score(y_true, y_pred),
        'F1': f1_score(y_true, y_pred),
        'Youden_J': sensitivity + specificity - 1,
        'Kappa': cohen_kappa_score(y_true, y_pred),
        'MCC': matthews_corrcoef(y_true, y_pred),
        'PPV': tp / (tp + fp) if (tp + fp) > 0 else np.nan,
        'NPV': tn / (tn + fn) if (tn + fn) > 0 else np.nan,
        'AUC': roc_auc_score(y_true, y_prob),
        'Sensitivity': tp / (tp + fn) if (tp + fn) > 0 else np.nan,
        'Specificity': tn / (tn + fp) if (tn + fp) > 0 else np.nan
    }

4. SHAP 可解释性分析
使用 SHAP 值分析模型预测的驱动因素，增强模型的可解释性。
逻辑回归模型的SHAP分析
explainer_logit = shap.LinearExplainer(
    logit_model,
    X_train_logit_scaled,
    feature_names=top6_vars
)

梯度提升模型的SHAP分析
explainer_gb = shap.TreeExplainer(gb_model)
shap_values_gb_val = explainer_gb.shap_values(X_val_6)

![描述](/images/portfolio/hba1c-prediction/gb_shap_depend_protein.png)

