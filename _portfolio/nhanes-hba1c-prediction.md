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
**摘要：项目背景**
糖尿病是全球性的公共卫生问题，糖化血红蛋白（HbA1c）是评估糖尿病控制情况的重要指标。本项目利用美国国家健康与营养调查（NHANES）2011-2018 年的数据，构建机器学习模型预测 HbA1c 水平，为糖尿病早期筛查和个性化管理提供数据支持。
**数据与方法**
数据来源
数据集：NHANES 2011-2018 四期数据
样本量：多期数据合并后的完整数据集
数据类型：人口学资料、体格测量、实验室检查、问卷调查
目标变量：HbA1c > 5.7% 定义为高 HbA1c 水平
**数据处理流程**
多源数据整合：合并 BMX（体格测量）、BPX（血压）、BPQ（血压问卷）、DEMO（人口学）、BIOPRO（生化指标）等表格
数据清洗：处理缺失值、异常值，去除胰岛素相关指标以避免数据泄漏
特征工程：计算平均血压、BMI 等衍生特征
数据集划分：按 7:3 比例划分训练集和验证集，保持类别平衡


**核心实现如下**

# **1.数据读取**
```python
import pandas as pd
import glob
import os
import numpy as np

base_dir = r"G:\Health data science"
years = ["2011-2012", "2013-2014", "2015-2016", "2017-2018"]

all_files = []
for y in years:
    folder_path = os.path.join(base_dir, y)
    files = glob.glob(os.path.join(folder_path, "*.XPT"))
    all_files.extend(files)

#按变量类型读取文件

dfs_bmx = []
dfs_bpx = []
dfs_bpq = []
dfs_demo = []
dfs_biopro = []
dfs_chol = []
dfs_ghb = []  # HbA1c表

for f in all_files:
    fname = os.path.basename(f).upper()
    if "BMX" in fname:
        dfs_bmx.append(pd.read_sas(f, format='xport'))
    elif "BPX" in fname:
        dfs_bpx.append(pd.read_sas(f, format='xport'))
    elif "BPQ" in fname:
        dfs_bpq.append(pd.read_sas(f, format='xport'))
    elif "DEMO" in fname:
        dfs_demo.append(pd.read_sas(f, format='xport'))
    elif "BIOPRO" in fname:
        dfs_biopro.append(pd.read_sas(f, format='xport'))
    elif "TCHOL" in fname:
        dfs_chol.append(pd.read_sas(f, format='xport'))
    elif "GHB" in fname:
        dfs_ghb.append(pd.read_sas(f, format='xport'))

#表格合并

df_bmx_all = pd.concat(dfs_bmx, ignore_index=True)
df_bpx_all = pd.concat(dfs_bpx, ignore_index=True)
df_bpq_all = pd.concat(dfs_bpq, ignore_index=True)
df_demo_all = pd.concat(dfs_demo, ignore_index=True)
df_biopro_all = pd.concat(dfs_biopro, ignore_index=True)
df_chol_all = pd.concat(dfs_chol, ignore_index=True)
df_ghb_all = pd.concat(dfs_ghb, ignore_index=True)

```
# **2.数据预处理**
```python
#处理 BIOPRO 表，提取数值

biopro_keep_cols = ['SEQN']
for col in df_biopro_all.columns:
    if col == 'SEQN':
        continue
    if pd.api.types.is_numeric_dtype(df_biopro_all[col]):
        biopro_keep_cols.append(col)
df_biopro_all = df_biopro_all[biopro_keep_cols]

#处理 TCHOL 表，提取数值

chol_cols = [c for c in df_chol_all.columns if c.upper() != 'SEQN' and df_chol_all[c].dtype in ['float64','int64']]
df_chol_all.rename(columns={chol_cols[0]:'LBXSCH'}, inplace=True)

#平均血压

df_bpx_all['SBP_mean'] = df_bpx_all[['BPXSY1','BPXSY2','BPXSY3']].mean(axis=1)
df_bpx_all['DBP_mean'] = df_bpx_all[['BPXDI1','BPXDI2','BPXDI3']].mean(axis=1)

#合并
df_merged = df_bmx_all.merge(
    df_bpx_all[['SEQN','SBP_mean','DBP_mean']],
    on='SEQN', how='left')
df_merged = df_merged.merge(
    df_bpq_all[['SEQN','BPQ020','BPQ050A']],
    on='SEQN', how='left')
df_merged = df_merged.merge(
    df_demo_all[['SEQN']],
    on='SEQN', how='left')
df_merged = df_merged.merge(
    df_biopro_all,
    on='SEQN', how='left')
df_merged = df_merged.merge(
    df_chol_all[['SEQN','LBXSCH']],
    on='SEQN', how='left')
df_merged = df_merged.merge(
    df_ghb_all[['SEQN','LBXGH']],
    on='SEQN', how='left')

#去掉胰岛素相关指标

insulin_cols = [c for c in df_merged.columns if 'INS' in c.upper()]
df_merged = df_merged.drop(columns=insulin_cols)

#标签

df_merged['high_hba1c'] = (df_merged['LBXGH'] > 5.7).astype(int)

#删除缺失值

core_required = ['BMXBMI','BMXWAIST','LBXGH']
df_cleared = df_merged.dropna(subset=core_required)

#保存

output_path = os.path.join(base_dir, "nhanes_all_years_cleaned_hba1c.csv")
df_cleared.to_csv(output_path, index=False)

#按7:3划分训练集与验证集

from sklearn.model_selection import train_test_split

train_df, val_df = train_test_split(
    df_cleared,
    test_size=0.3,
    random_state=42,
    stratify=df_cleared['high_hba1c']
)

#数值型变量转换

numeric_cols = train_df.select_dtypes(include=['float64','int64']).columns
train_df[numeric_cols] = train_df[numeric_cols].apply(pd.to_numeric, errors='coerce')
val_df[numeric_cols] = val_df[numeric_cols].apply(pd.to_numeric, errors='coerce')
```

# **3.统计分析人口学资料，写入table1**
```python

from scipy.stats import mannwhitneyu

demo_vars = ['BMXBMI','BMXWAIST','SBP_mean','DBP_mean']
biopro_vars = [c for c in train_df.columns if c.startswith('LBX') and c not in ['LBXSCH','LBXGH']]
table1_vars = demo_vars + biopro_vars

for col in table1_vars:
    if col in train_df.columns:
        train_df.loc[:, col] = pd.to_numeric(train_df[col], errors='coerce')
    if col in val_df.columns:
        val_df.loc[:, col] = pd.to_numeric(val_df[col], errors='coerce')

def table1_continuous(train_df, val_df, variables):
    rows = []
    for var in variables:
        if var not in train_df.columns or var not in val_df.columns:
            continue
        x = train_df[var].dropna()
        y = val_df[var].dropna()
        if len(x) < 20 or len(y) < 20:
            continue
        stat, p = mannwhitneyu(x, y, alternative='two-sided')
        rows.append([
            var,
            f"{x.mean():.2f} ± {x.std():.2f}",
            f"{y.mean():.2f} ± {y.std():.2f}",
            f"{p:.3f}"
        ])
    return pd.DataFrame(rows, columns=['Variable','Training set (mean ± SD)','Validation set (mean ± SD)','p-value'])

table1 = table1_continuous(train_df, val_df, table1_vars)
table1.to_csv(os.path.join(base_dir, "Table1_Demographics_BIOPRO_hba1c.csv"), index=False)
```

# **4.预测模型建立**

LASSO 特征选择

使用筛选出的前 6 个重要特征，构建五种机器学习模型（逻辑回归 、支持向量机、梯度提升机、贝叶斯算法和随机森林）进行对比分析。

```python

#选用前六位预测因子，用五种机器学习方法（逻辑回归 、支持向量机、梯度提升机、贝叶斯算法和随机森林）训练预测模型
from sklearn.svm import SVC
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.pipeline import Pipeline
from sklearn.metrics import roc_auc_score, roc_curve

X_train_6 = train_df[top6_vars].copy()
X_val_6   = val_df[top6_vars].copy()
y_train = train_df['high_hba1c']
y_val = val_df['high_hba1c']

# 缺失值用中位数填充
imputer = SimpleImputer(strategy='median')
X_train_6 = pd.DataFrame(imputer.fit_transform(X_train_6), columns=top6_vars)
X_val_6 = pd.DataFrame(imputer.transform(X_val_6), columns=top6_vars)

models = {
    'Logistic Regression': Pipeline([('scaler', StandardScaler()), ('model', LogisticRegression(max_iter=3000, random_state=42))]),
    'SVM': Pipeline([('scaler', StandardScaler()), ('model', SVC(kernel='rbf', probability=True, random_state=42))]),
    'Gradient Boosting': GradientBoostingClassifier(random_state=42),
    'Bayesian': GaussianNB(),
    'Random Forest': RandomForestClassifier(n_estimators=300, random_state=42, n_jobs=-1)}

auc_results = {}
plt.figure(figsize=(7,6))
for name, model in models.items():
    model.fit(X_train_6, y_train)
    y_prob = model.predict_proba(X_val_6)[:, 1]
    auc = roc_auc_score(y_val, y_prob)
    fpr, tpr, _ = roc_curve(y_val, y_prob)
    auc_results[name] = auc
    plt.plot(fpr, tpr, label=f'{name} (AUC = {auc:.3f})')

plt.plot([0,1],[0,1],'k--', alpha=0.5)
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC curves of five predictive models)')
plt.legend(loc='lower right', fontsize=9)
plt.grid(alpha=0.3)
plt.tight_layout()
plt.show()

```

**4. 预测模型评估与可视化**

```python
auc_table = pd.DataFrame({'Model': auc_results.keys(), 'AUC': auc_results.values()}).sort_values('AUC', ascending=False)
print("\n验证集 AUC 结果：")
print(auc_table)
auc_table.to_csv(os.path.join(base_dir, "Model_AUC_comparison_hba1c_noLBXGH.csv"), index=False)

#进行五种模型在测试集上基于多种评估指标的性能比较，包括准确率、平衡准确率、F1 分数、J 指数、Kappa 系数、马修斯相关系数 (MCC)、阳性预测值 (PPV)、阴性预测值 (NPV)、精确率、召回率、ROC 曲线下面积 (AUC)、灵敏度 (sens) 和特异性 (spec)

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, confusion_matrix,
    cohen_kappa_score, matthews_corrcoef)

metrics_table = []

for name, model in models.items():
    y_prob = model.predict_proba(X_val_6)[:, 1]
    y_pred = (y_prob >= 0.5).astype(int)

    tn, fp, fn, tp = confusion_matrix(y_val, y_pred).ravel()

    #定义
    accuracy = accuracy_score(y_val, y_pred)
    bal_accuracy = balanced_accuracy_score(y_val, y_pred)
    precision = precision_score(y_val, y_pred)
    recall = recall_score(y_val, y_pred)
    f1 = f1_score(y_val, y_pred)
    auc = roc_auc_score(y_val, y_prob)

    specificity = tn / (tn + fp) if (tn + fp) > 0 else np.nan
    npv = tn / (tn + fn) if (tn + fn) > 0 else np.nan
    youden_j = recall + specificity - 1

    kappa = cohen_kappa_score(y_val, y_pred)
    mcc = matthews_corrcoef(y_val, y_pred)

    metrics_table.append([
        name,
        accuracy,
        bal_accuracy,
        f1,
        youden_j,
        kappa,
        mcc,
        precision,      # PPV
        npv,
        precision,
        recall,
        auc,
        recall,         # Sensitivity
        specificity])

#汇总
metrics_df = pd.DataFrame(
    metrics_table,
    columns=[
        'Model',
        'Accuracy',
        'Balanced_Accuracy',
        'F1',
        'Youden_J',
        'Kappa',
        'MCC',
        'PPV',
        'NPV',
        'Precision',
        'Recall',
        'AUC',
        'Sensitivity',
        'Specificity'])

metrics_df.set_index('Model', inplace=True)

#保存
metrics_df.to_csv(
    os.path.join(base_dir, "Model_performance_metrics_hba1c.csv"))

print("\n模型性能比较（测试集）：")
print(metrics_df.round(3))

#绘图
plt.figure(figsize=(14, 6))

sns.heatmap(
    metrics_df,
    annot=True,
    fmt=".2f",
    cmap=sns.diverging_palette(240, 10, as_cmap=True),  # 蓝 → 红
    linewidths=0.5,
    cbar_kws={'label': 'Performance'})

plt.xlabel("Performance metrics")
plt.ylabel("Models")
plt.title("Performance comparison of five models on validation set")
plt.tight_layout()
plt.show()

#训练集做同样的分析

from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    cohen_kappa_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
    confusion_matrix)

def evaluate_binary_model(y_true, y_prob, threshold=0.5):
    y_pred = (y_prob >= threshold).astype(int)

    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()

    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else np.nan
    specificity = tn / (tn + fp) if (tn + fp) > 0 else np.nan
    ppv = tp / (tp + fp) if (tp + fp) > 0 else np.nan
    npv = tn / (tn + fn) if (tn + fn) > 0 else np.nan
    youden_j = sensitivity + specificity - 1

    return {
        'Accuracy': accuracy_score(y_true, y_pred),
        'Balanced_Accuracy': balanced_accuracy_score(y_true, y_pred),
        'F1': f1_score(y_true, y_pred),
        'Youden_J': youden_j,
        'Kappa': cohen_kappa_score(y_true, y_pred),
        'MCC': matthews_corrcoef(y_true, y_pred),
        'PPV': ppv,
        'NPV': npv,
        'Precision': precision_score(y_true, y_pred),
        'Recall': recall_score(y_true, y_pred),
        'Sensitivity': sensitivity,
        'Specificity': specificity,
        'AUC': roc_auc_score(y_true, y_prob)}

train_results = {}

for name, model in models.items():
    model.fit(X_train_6, y_train)
    y_train_prob = model.predict_proba(X_train_6)[:, 1]
    train_results[name] = evaluate_binary_model(y_train, y_train_prob)

df_train_metrics = pd.DataFrame(train_results).T

df_train_metrics.to_csv(
    os.path.join(base_dir, "Model_Performance_Training_AllMetrics.csv"))

# Heatmap visualization
plt.figure(figsize=(15, 6))

sns.heatmap(
    df_train_metrics,
    annot=True,
    fmt=".2f",
    cmap=sns.diverging_palette(240, 10, as_cmap=True),  
    linewidths=0.5,
    cbar_kws={'label': 'Metric value'})


plt.title("Model Performance Comparison on Training Set")
plt.xlabel("Performance Metrics")
plt.ylabel("Models")
plt.tight_layout()
plt.show()

import shap
import numpy as np
import matplotlib.pyplot as plt

np.random.seed(42)

#SHAP分析-逻辑回归模型

#提取已经训练好的逻辑回归模型和标准化器
logit_pipeline = models['Logistic Regression']
logit_model = logit_pipeline.named_steps['model']
logit_scaler = logit_pipeline.named_steps['scaler']

#构建训练集用于 SHAP 的数据
X_train_logit = X_train_6.copy()
X_train_logit_scaled = logit_scaler.transform(X_train_logit)

#构建验证集用于 SHAP 的数据
X_val_logit = X_val_6.copy()
X_val_logit_scaled = logit_scaler.transform(X_val_logit)

#创建 SHAP 解释器
explainer_logit = shap.LinearExplainer(
    logit_model,
    X_train_logit_scaled,
    feature_names=top6_vars)

#计算训练集 SHAP 值
shap_values_logit_train = explainer_logit.shap_values(
    X_train_logit_scaled)

#计算验证集 SHAP 值
shap_values_logit_val = explainer_logit.shap_values(
    X_val_logit_scaled)

#绘制训练集 SHAP 总结图
shap.summary_plot(
    shap_values_logit_train,
    X_train_logit,
    feature_names=top6_vars,
    plot_type="dot",
    show=True)

#绘制训练集 SHAP 重要性柱状图
shap.summary_plot(
    shap_values_logit_train,
    X_train_logit,
    feature_names=top6_vars,
    plot_type="bar",
    show=True)

#绘制验证集 SHAP 总结图
shap.summary_plot(
    shap_values_logit_val,
    X_val_logit,
    feature_names=top6_vars,
    plot_type="dot",
    show=True)

#绘制验证集 SHAP 重要性柱状图
shap.summary_plot(
    shap_values_logit_val,
    X_val_logit,
    feature_names=top6_vars,
    plot_type="bar",
    show=True)

#SHAP分析-梯度提升模型的 SHAP 分析

#提取已经训练好的梯度提升模型
gb_model = models['Gradient Boosting']

#创建梯度提升模型的 SHAP 解释器
explainer_gb = shap.TreeExplainer(gb_model)

#计算训练集 SHAP 值
shap_values_gb_train = explainer_gb.shap_values(
    X_train_6)

#计算验证集 SHAP 值
shap_values_gb_val = explainer_gb.shap_values(
    X_val_6)

#绘制训练集 SHAP 总结图
shap.summary_plot(
    shap_values_gb_train,
    X_train_6,
    feature_names=top6_vars,
    plot_type="dot",
    show=True)

#绘制训练集 SHAP 重要性柱状图
shap.summary_plot(
    shap_values_gb_train,
    X_train_6,
    feature_names=top6_vars,
    plot_type="bar",
    show=True)

#绘制验证集 SHAP 总结图
shap.summary_plot(
    shap_values_gb_val,
    X_val_6,
    feature_names=top6_vars,
    plot_type="dot",
    show=True)

#绘制验证集 SHAP 重要性柱状图
shap.summary_plot(
    shap_values_gb_val,
    X_val_6,
    feature_names=top6_vars,
    plot_type="bar",
    show=True)

#梯度提升模型的单变量依赖关系图
for feature in top6_vars:
    shap.dependence_plot(
        feature,
        shap_values_gb_val,
        X_val_6,
        feature_names=top6_vars,
        show=True)

```

**结果分析**

**1. 人口学特征分析**

![描述](/images/portfolio/hba1c-prediction/table1_demographic_characteristics.png)

表 1：研究人群的人口学特征分布，包括BMI、腰围等，为模型构建提供了基础数据描述

**2. 特征选择结果**

![描述](/images/portfolio/hba1c-prediction/lasso_coefficient_paths.png)

图 1：LASSO 回归系数路径图，展示了不同正则化强度下前 6 个重要特征的系数变化

**3. 模型性能对比**

![描述](/images/portfolio/hba1c-prediction/model_roc_curves.png)

图 2：五种机器学习模型在验证集上的 ROC 曲线对比

性能排名：逻辑回归：AUC = 0.829（最优）; 梯度提升：AUC = 0.827（次优）; 随机森林：AUC = 0.820; SVM：AUC = 0.780; 贝叶斯：AUC = 0.750

**4. 训练集与验证集性能评估**

![描述](/images/portfolio/hba1c-prediction/training_performance_heatmap.png)

![描述](/images/portfolio/hba1c-prediction/validation_performance_heatmap.png)

图 3、图4：训练集与验证集模型性能对比


过拟合分析：
逻辑回归：训练集 AUC 0.832 → 验证集 AUC 0.829（仅下降 0.3%）
梯度提升：训练集 AUC 0.835 → 验证集 AUC 0.827（下降 0.8%）
模型选择建议：逻辑回归在保持高性能的同时，过拟合风险最低


5. 逻辑回归模型 SHAP 分析

image

image

image

image
图 4：逻辑回归模型的 SHAP 分析结果
特征重要性排名：
BMI：最重要的预测因子
腰围：中心性肥胖指标
收缩压：高血压相关
舒张压：血管健康指标
总胆固醇：血脂代谢指标
蛋白指标：营养状态指标
6. 梯度提升模型 SHAP 分析

image

image

image

image
图 5：梯度提升模型的 SHAP 分析结果
7. 梯度提升模型特征依赖分析

image

image

image

image

image

image
图 6：梯度提升模型中各特征的 SHAP 依赖图
关键非线性关系发现：
BMI：当 BMI 超过 30 kg/m² 时，SHAP 值急剧上升，表明肥胖显著增加糖尿病风险
腰围：存在阈值效应，腰围超过一定值后风险非线性增加
血压指标：收缩压和舒张压与风险呈正相关，但关系相对线性
生化指标：总胆固醇和蛋白指标与风险存在复杂的非线性关系
总结与临床建议
模型选择策略
基于分析结果，建议：
首选模型：逻辑回归（AUC 0.829）
理由：性能最优、过拟合程度低、可解释性强、计算效率高
适用场景：临床决策支持、大规模筛查、实时风险评估
备选模型：梯度提升机（AUC 0.827）
理由：性能接近最优、能捕捉非线性关系、适用于复杂病例
适用场景：研究探索、复杂病例分析、特征交互研究
临床实施建议
简化风险评估工具：基于逻辑回归模型，开发仅需 6 个指标的在线计算器
个性化干预阈值：根据不同人群特征调整风险阈值
持续模型更新：定期用新数据重新训练模型，保持预测准确性
技术贡献总结
系统的方法论：从数据清洗到模型部署的完整流程
全面的模型对比：5 种算法 ×13 个指标 ×2 个数据集
深入的可解释性分析：双模型 SHAP 对比 + 特征依赖分析
临床实用性验证：基于易获取指标的高性能模型
未来研究方向
外部验证：在其他独立数据集上验证模型泛化能力
时间序列分析：纳入纵向数据构建动态预测模型
多模态数据融合：整合影像学、基因组学等多维度数据
实时预测系统：开发基于移动应用的实时风险评估工具



