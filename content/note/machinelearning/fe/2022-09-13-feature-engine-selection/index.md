---
title: 特征选择
author: 王哲峰
date: '2022-09-13'
slug: feature-engine-selection
categories:
  - feature engine
tags:
  - machinelearning
---

<style>
details {
    border: 1px solid #aaa;
    border-radius: 4px;
    padding: .5em .5em 0;
}
summary {
    font-weight: bold;
    margin: -.5em -.5em 0;
    padding: .5em;
}
details[open] {
    padding: .5em;
}
details[open] summary {
    border-bottom: 1px solid #aaa;
    margin-bottom: .5em;
}
</style>

<details><summary>目录</summary><p>

- [过滤](#过滤)
- [打包](#打包)
- [嵌入式](#嵌入式)
- [主成分分析](#主成分分析)
- [随机森林特征重要性评价](#随机森林特征重要性评价)
  - [参考](#参考)
</p></details><p></p>

特征选择技术可以精简掉无用的特征, 以降低最终模型的复杂性, 
它的最终目的是得到一个简约模型, 在不降低预测准确率或对预测准确率影响不大的情况下提高计算速度

# 过滤

对特征进行预处理, 除去那些不太可能对模型有用处的特征

* 计算特征与相应变量之间的相关性或互信息, 然后过滤掉那些在某个阈值之下的特征
* 没有考虑模型, 可能无法为模型选择出正确的特征

# 打包

* 试验特征的各个子集

# 嵌入式

将特征选择作为模型训练过程的一部分；

* 特征选择是决策书与生俱来的功能, 因为它在每个训练阶段都要选择一个特征来对树进行分割；
    - 决策树
    - GBM
    - XGBoost
    - LightGBM
    - CatBoost
    - RandomForest
* L1 正则化可以添加到任意线性模型的训练目标中, L1 正则化鼓励模型使用更少的特征, 所以也称为稀疏性约束；
    - LASSO

# 主成分分析


# 随机森林特征重要性评价

```python
import numpy as np
import pandas as pd
try:
    from sklearn.cross_validation import train_test_split
except:
    from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier


# data
data_url = 'http://archive.ics.uci.edu/ml/machine-learning-databases/wine/wine.data'
df = pd.read_csv(data_url, header = None)
df.columns = [
    'Class label', 'Alcohol', 'Malic acid', 'Ash', 
    'Alcalinity of ash', 'Magnesium', 'Total phenols', 
    'Flavanoids', 'Nonflavanoid phenols', 'Proanthocyanins', 
    'Color intensity', 'Hue', 'OD280/OD315 of diluted wines', 'Proline'
]

# data split
x, y = df.iloc[:, 1:].values, df.iloc[:, 0].values
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.3, random_state = 0)
feat_labels = df.columns[1:]

# model
rf_clf = RandomForestClassifier(
    n_estimators = 10000,
    random_state = 0,
    n_jobs = -1,
)
rf_clf.fit(x_train, y_train)

importances = rf_clf.feature_importances_
indices = np.argsort(importances)[::-1]
for f in range(x_train.shape[1]):
    print("%2d) %-*s %f" % (f + 1, 30, feat_labels[indices[f]], importances[indices[f]]))

# features selection
threshold = 0.15
x_selected = x_train[:, importances > threshold]
print(x_selected.shape)
```

## 参考

* https://mp.weixin.qq.com/s/2O9k0FSY15aHRSZ8D6B5Gg



