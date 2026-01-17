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
项目背景
本研究基于美国国家健康与营养调查（NHANES）2011-2018 年的数据，旨在构建一个预测高糖化血红蛋白（HbA1c > 5.7%）的机器学习模型。高 HbA1c 水平是糖尿病前期和糖尿病的重要标志，早期识别高风险人群对于预防和管理糖尿病具有重要意义。
研究使用了来自多个 NHANES 周期的数据，包括人口统计学、人体测量、血压、生物标志物等多维度信息，通过整合分析构建了一个全面的预测框架。
核心实现
1. 数据整合与预处理
# 合并多源数据
df_merged = df_bmx_all.merge(
    df_bpx_all[['SEQN','SBP_mean','DBP_mean']],
    on='SEQN', how='left'
)
df_merged = df_merged.merge(
    df_bpq_all[['SEQN','BPQ020','BPQ050A']],
    on='SEQN', how='left'
)
df_merged = df_merged.merge(
    df_biopro_all,
    on='SEQN', how='left'
)
df_merged = df_merged.merge(
    df_chol_all[['SEQN','LBXSCH']],
    on='SEQN', how='left'
)

# 创建标签
df_merged['high_hba1c'] = (df_merged['LBXGH'] > 5.7).astype(int)

2.特征选择（LASSO 回归）
# LASSO 特征选择
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

# 提取重要特征
lasso_selected = (
    lasso_results[lasso_results['Coefficient'] != 0]
    .assign(abs_coef=lambda x: x['Coefficient'].abs())
    .sort_values('abs_coef', ascending=False)
    .drop(columns='abs_coef')
)
top6_vars = lasso_selected.head(6)['Variable'].tolist()

3.模型构建与训练
# 定义五种机器学习模型
models = {
    'Logistic Regression': Pipeline([('scaler', StandardScaler()), ('model', LogisticRegression(max_iter=3000, random_state=42))]),
    'SVM': Pipeline([('scaler', StandardScaler()), ('model', SVC(kernel='rbf', probability=True, random_state=42))]),
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

4. SHAP 可解释性分析
# 逻辑回归 SHAP 分析
explainer_logit = shap.LinearExplainer(
    logit_model,
    X_train_logit_scaled,
    feature_names=top6_vars
)

# 梯度提升模型 SHAP 分析
explainer_gb = shap.TreeExplainer(gb_model)

# 单变量依赖分析
for feature in top6_vars:
    shap.dependence_plot(
        feature,
        shap_values_gb_val,
        X_val_6,
        feature_names=top6_vars,
        show=True)
分析结果
1. 人口学资料

image
基线特征表展示了研究人群的基本特征，包括年龄、性别、种族、教育水平等分布情况。
2. 特征选择结果

image
LASSO 回归选择了 6 个最重要的预测因子，避免了特征泄漏问题，确保模型的可解释性。
3. 模型性能比较

image
逻辑回归（AUC = 0.829）和梯度提升机（AUC = 0.827）在验证集上表现最优，显示出良好的预测能力。
4. 验证集模型性能热力图

image
多指标性能比较显示梯度提升机在验证集的综合表现最佳，各项指标均衡。
5. 训练集模型性能热力图

image
训练集性能分析显示模型未出现过拟合，泛化能力良好。
6. 逻辑回归模型 SHAP 分析

image

image

image

image
逻辑回归模型的特征重要性分析显示，BMI 和腰围是最重要的预测因子，其次是血压和胆固醇指标。
7. 梯度提升模型 SHAP 分析

image

image

image

image
非线性模型的 SHAP 分析揭示了更复杂的特征交互关系，BMI和腰围仍然是最重要的预测因子。
8. 单变量依赖关系分析

image

image

image

image

image

image
单变量依赖图显示，BMI 和腰围对预测的影响呈现非线性模式，当值超过一定阈值时，预测风险显著增加，这与临床观察一致。
结论
模型性能：逻辑回归和梯度提升机在预测 HbA1c 水平方面表现最佳，AUC 分别为 0.829 和 0.827。
重要特征：BMI、腰围、收缩压和舒张压是预测高 HbA1c 水平的最重要特征。
临床应用：该模型可用于识别糖尿病前期高风险人群，为早期干预提供依据。
方法学贡献：结合了 LASSO 特征选择、多种机器学习模型和 SHAP 可解释性分析，提供了一个完整的医疗预测建模框架。
技术要点
数据来源：NHANES 2011-2018 多周期数据
特征选择：LASSO 回归避免特征泄漏
模型对比：五种机器学习算法全面比较
可解释性：SHAP 分析揭示特征重要性
验证策略：7:3 分层抽样，确保模型泛化能力
