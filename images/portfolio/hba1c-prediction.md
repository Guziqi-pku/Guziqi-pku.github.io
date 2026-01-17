---
title: "基于常规体检指标的 HbA1c 风险预测模型"
collection: portfolio
type: "Machine Learning"
permalink: /portfolio/nhanes-hba1c-prediction
date: 2024-01-15
excerpt: "利用 LASSO 特征选择与集成学习，基于常规体检指标构建高精度 HbA1c 风险预测模型，AUC 达 0.87"
header:
  teaser: /images/portfolio/nhanes-hba1c-prediction/model_roc_curves.png
tags:
  - 机器学习
  - 医疗数据分析
  - 特征选择
  - 模型可解释性
tech_stack:
  - name: Python
  - name: Scikit-learn
  - name: XGBoost
  - name: SHAP
  - name: Pandas
  - name: Matplotlib
  - name: Seaborn
---

## 项目背景

糖化血红蛋白（HbA1c）是糖尿病诊断和管理的关键指标。传统的 HbA1c 检测需要静脉采血，成本较高且不便。本项目旨在利用美国国家健康与营养调查（NHANES）数据，基于常规体检指标（如 BMI、血压、腰围等）构建机器学习模型，预测个体 HbA1c 异常风险，为基层医疗提供低成本筛查工具。

## 数据与方法

### 研究人群特征
研究人群的基本特征如表1所示，包括训练集和验证集的人口统计学、体格测量和生化指标分布：

![人口学资料 Table 1](/images/portfolio/nhanes-hba1c-prediction/table1_demographic_characteristics.png)
*表1：研究人群基线特征（训练集 vs 验证集）*

**关键发现：**
1. **样本规模**：总样本量超过10,000人，具有统计学意义
2. **数据完整性**：关键变量缺失率低于5%，数据质量良好
3. **分组平衡**：训练集和验证集在人口学特征上分布均衡
4. **风险分布**：高HbA1c风险比例约30%，与人群流行病学数据一致
5. **特征分布**：连续变量（如BMI、血压）符合正态或近似正态分布

### 数据来源与预处理
项目整合了 NHANES 四个调查周期（2011-2012、2013-2014、2015-2016、2017-2018）的多个数据模块，包括人口学、体检、实验室和问卷数据。通过数据清洗、缺失值处理和特征工程，构建了包含50+个候选特征的数据集。

### 特征选择方法
采用 LASSO（Least Absolute Shrinkage and Selection Operator）回归进行特征选择，通过交叉验证确定最优正则化参数，从众多候选特征中筛选出最具预测价值的指标。

### 模型构建与评估
构建并比较了五种机器学习模型：
1. 逻辑回归（Logistic Regression）
2. 随机森林（Random Forest）
3. 梯度提升（Gradient Boosting）
4. 支持向量机（SVM）
5. 多层感知器（MLP）

使用13个评估指标进行综合性能评估，包括 AUC、准确率、召回率、F1分数等。

## 核心实现

### 1. LASSO 特征选择
```python
# LASSO 特征选择
from sklearn.linear_model import LassoCV

# 初始化 LASSO 交叉验证模型
lasso_cv = LassoCV(cv=5, random_state=42, max_iter=10000)

# 拟合模型
lasso_cv.fit(X_train_scaled, y_train)

# 获取选择的特征
selected_features = X_train.columns[lasso_cv.coef_ != 0]
print(f"LASSO 选择的特征数量: {len(selected_features)}")
print(f"选择的特征: {selected_features.tolist()}")
2. 模型训练与评估
# 定义模型列表
models = {
    'Logistic Regression': LogisticRegression(random_state=42),
    'Random Forest': RandomForestClassifier(random_state=42),
    'Gradient Boosting': GradientBoostingClassifier(random_state=42),
    'SVM': SVC(probability=True, random_state=42),
    'MLP': MLPClassifier(random_state=42, max_iter=1000)
}

# 训练和评估所有模型
results = {}
for name, model in models.items():
    # 训练模型
    model.fit(X_train_selected, y_train)
    
    # 预测
    y_pred = model.predict(X_val_selected)
    y_pred_proba = model.predict_proba(X_val_selected)[:, 1]
    
    # 计算评估指标
    results[name] = {
        'accuracy': accuracy_score(y_val, y_pred),
        'precision': precision_score(y_val, y_pred),
        'recall': recall_score(y_val, y_pred),
        'f1': f1_score(y_val, y_pred),
        'roc_auc': roc_auc_score(y_val, y_pred_proba)
    }
3. SHAP 可解释性分析
# 逻辑回归 SHAP 分析
import shap

# 创建 SHAP 解释器
explainer_logit = shap.LinearExplainer(logit_model, X_train_selected)
shap_values_logit = explainer_logit.shap_values(X_train_selected)

# 绘制 SHAP 总结图
shap.summary_plot(shap_values_logit, X_train_selected, 
                  feature_names=selected_features,
                  show=False)
plt.title('Logistic Regression SHAP Summary - Training Set')
plt.tight_layout()
plt.savefig('logit_train_shap_summary.png', dpi=300, bbox_inches='tight')
plt.close()

# 梯度提升模型 SHAP 分析
explainer_gb = shap.TreeExplainer(gb_model)
shap_values_gb = explainer_gb.shap_values(X_train_selected)

# 绘制单变量依赖图
for i, feature in enumerate(['Waist Circumference', 'BMI', 'Systolic BP']):
    shap.dependence_plot(feature, shap_values_gb, X_train_selected,
                        feature_names=selected_features,
                        show=False)
    plt.tight_layout()
    plt.savefig(f'gb_shap_depend_{feature.lower().replace(" ", "_")}.png', 
                dpi=300, bbox_inches='tight')
    plt.close()
分析结果
1. LASSO 特征选择结果
通过 LASSO 回归从众多候选特征中筛选出 6 个关键预测因子：腰围、BMI、收缩压、舒张压、总胆固醇和特定蛋白标志物。这些特征在临床实践中易于获取，具有良好的实用性。

image
图 1：LASSO 系数路径图，展示了不同正则化强度下特征系数的变化
2. 模型性能比较
五种模型在验证集上的 ROC 曲线显示，梯度提升模型表现最佳（AUC = 0.872），其次是逻辑回归（AUC = 0.865）。所有模型的 AUC 均超过 0.85，表明基于常规体检指标的预测具有较高准确性。

image
图 2：五种预测模型的 ROC 曲线比较
3. 综合性能评估
使用 13 个评估指标对模型进行全面评估，梯度提升模型在多数指标上表现最优，特别是在召回率（0.812）和 F1 分数（0.783）方面表现突出。

image
图 3：验证集上五种模型的综合性能热力图

image
图 4：训练集上模型的性能表现
4. 模型可解释性分析（SHAP 分析）
逻辑回归模型可解释性
逻辑回归模型因其良好的可解释性，特别适合临床应用。SHAP 分析显示各常规指标对 HbA1c 风险的贡献程度：
训练集分析：

image
图 5a：逻辑回归训练集 SHAP 特征重要性总结

image
图 5b：逻辑回归训练集特征重要性排序
验证集分析：

image
图 5c：逻辑回归验证集 SHAP 特征重要性总结

image
图 5d：逻辑回归验证集特征重要性排序
梯度提升模型可解释性
梯度提升模型虽然是非线性模型，但通过 SHAP 分析仍能提供良好的可解释性：
训练集分析：

image
图 6a：梯度提升训练集 SHAP 特征重要性总结

image
图 6b：梯度提升训练集特征重要性排序
验证集分析：

image
图 6c：梯度提升验证集 SHAP 特征重要性总结

image
图 6d：梯度提升验证集特征重要性排序
单变量依赖关系分析
梯度提升模型的单变量依赖关系图揭示了各特征与 HbA1c 风险之间的非线性关系：

image
图 7a：腰围对 HbA1c 风险的非线性影响模式

image
图 7b：BMI 对 HbA1c 风险的非线性影响模式

image
图 7c：收缩压对 HbA1c 风险的非线性影响模式

image
图 7d：舒张压对 HbA1c 风险的非线性影响模式

image
图 7e：总胆固醇对 HbA1c 风险的非线性影响模式

image
图 7f：特定蛋白标志物对 HbA1c 风险的非线性影响模式
结论与价值
主要发现
预测性能优异：基于 6 个常规体检指标的模型 AUC 达到 0.87，验证了常规指标对 HbA1c 风险的良好预测能力
特征重要性明确：腰围是最重要的预测因子，其次是 BMI 和血压指标，这与临床认知一致
模型可解释性强：通过 SHAP 分析提供了直观的特征贡献可视化，增强了模型的临床可接受性
实用价值高：所有特征均为常规体检项目，无需特殊检测，适合基层医疗机构推广
临床意义
低成本筛查：为 HbA1c 异常风险筛查提供了低成本替代方案
早期干预：有助于识别高风险个体，实现早期生活方式干预
资源优化：在资源有限地区可作为初步筛查工具，优化医疗资源配置
技术价值
特征工程示范：展示了如何从多源数据中提取有效特征
模型比较框架：提供了完整的机器学习模型比较评估流程
可解释性实践：结合了预测性能与模型可解释性的最佳实践
该项目不仅展示了机器学习在医疗数据分析中的应用，也为类似健康风险预测问题提供了可复用的技术框架。