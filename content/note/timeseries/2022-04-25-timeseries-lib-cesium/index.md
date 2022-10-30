---
title: Lib cesium 时间序列特征工程
author: 王哲峰
date: '2022-04-25'
slug: timeseries-lib-cesium
categories:
  - timeseries
tags:
  - tool
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

- [使用示例](#使用示例)
  - [构建数据集](#构建数据集)
  - [特征工程](#特征工程)
    - [类别标签处理](#类别标签处理)
    - [特征构造](#特征构造)
    - [自定义特征函数](#自定义特征函数)
    - [多通道时间序列特征工程](#多通道时间序列特征工程)
  - [模型构建](#模型构建)
  - [预测](#预测)
</p></details><p></p>



# 使用示例

> EEG 数据上的癫痫分类检测

类别:

* Z: normal
* O: normal
* N: interictal
* F: interictal
* S: ictal

基本流程:

1. 时间序列数据特征构造
2. 构建分类模型
3. 预测

## 构建数据集

```python
import numpy as np
import matplotlib.pyplot as plt
import seaborn
seaborn.set()

from cesium import datasets

eeg = datasets.fetch_andrzejak()
```

## 特征工程

### 类别标签处理

```python
eeg["classes"] = eeg["classes"].astype("U16")
eeg["classes"][np.logical_or(eeg["classes"] == "Z", eeg["classes"] == "O")] = "Normal"
eeg["classes"][np.logical_or(eeg["classes"] == "N", eeg["classes"] == "F")] = "Interictal"
eeg["classes"][eeg["classes"] == "S"] = "Ictal"

fig, axs = plt.subplots(1, len(np.unique(eeg["classes"])), sharey = True)
for label, ax in zip(np.unique(eeg["classes"]), axs):
    i = np.where(eeg["classes"] == label)[0][0]
    ax.plot(eeg["times"][i], eeg["measurements"][i])
    ax.set(xlabel = "time (s)", ylabel = "signal", title = label)
```

### 特征构造

```python
from cesium import featurize

features_to_use = [
    "amplitude",
    "percent_beyond_1_std",
    "maximum",
    "max_slope",
    "median",
    "median_absolute_deviation",
    "percent_close_to_median",
    "minimum",
    "skew",
    "std",
    "weighted_average",
]

fset_cesium = featurize.featurize_time_series(
    times = eeg["time"],
    values = eeg["measurements"],
    errors = None,
    features_to_use = features_to_use,
    print(fset_cesium.head())
)
```

### 自定义特征函数

```python
import numpy as np
import scipy.stats

def mean_signal(t, m, e):
    return np.mean(m)

def std_signal(t, m, e):
    return np.std(m)

def mean_square_signal(t, m, e):
    return np.mean(m ** 2)

def abs_diffs_signal(t, m, e):
    return np.sum(np.abs(np.diff(m)))

def skew_signal(t, m, e):
    return scipy.stats.skew(m)


guo_features = {
    "mean": mean_signal,
    "std": std_signal,
    "mean2": mean_square_signal,
    "abs_diffs": abs_diffs_signals,
    "skew": skew_signal,
}
fset_guo = featurize.featurize_time_series(
    times = eeg["times"],
    values = eeg["measurements"],
    errors = None,
    features_to_use = list(guo_features.keys()),
    custom_functions = guo_features
)
```

### 多通道时间序列特征工程

```python
import pywt

n_channels = 5
eeg["dwts"] = [
    pywt.wavedec(m, pywt.Wavelet("db1"), level = n_channels -1)
    for m in eeg["measurements"]
]
fset_dwt = featurize.featurize_time_series(
    time = None,
    values = eeg["dwts"],
    errors = None,
    features_to_use = list(guo_features.keys()),
    custom_functions = guo_features,
)
```



## 模型构建

```python
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClsssifier
from sklearn.model_selection import train_test_split

train_idx, test_idx = train_test_split(
    np.arange(len(egg["classes"])), 
    random_state = 0
)
model_cesium = RandomForestClassifier(
    n_estimators = 128,
    max_features = "auto",
    random_state = 0
)
model_cesium.fit(
    fset_cesium.iloc[train_idx],
    eeg["classes"][train_idx],
)


model_guo = KNeighborsClassifier(3)
model_guo.fit(
    fset_guo.iloc[train_idx],
    eeg["classes"][train_idx],
)


model_dwt = KNeighborsClassifier(3)
model_dwt.fit(
    fset_dwt.iloc[train_idx],
    eeg["classes"][train_idx],
)
```



## 预测

```python
from sklearn.metrics import accuracy_score

preds_cesium = model_cesium.predict(fset_cesium)
preds_guo = model_guo.predict(fset_guo)
preds_dwt = model_dwt.predict(fset_dwt)

print("Built-in cesium features: training accuracy={:.2%}, test accuracy={:.2%}".format(
          accuracy_score(preds_cesium[train], eeg["classes"][train]),
          accuracy_score(preds_cesium[test], eeg["classes"][test])))
print("Guo et al. features: training accuracy={:.2%}, test accuracy={:.2%}".format(
          accuracy_score(preds_guo[train], eeg["classes"][train]),
          accuracy_score(preds_guo[test], eeg["classes"][test])))
print("Wavelet transform features: training accuracy={:.2%}, test accuracy={:.2%}".format(
          accuracy_score(preds_dwt[train], eeg["classes"][train]),
          accuracy_score(preds_dwt[test], eeg["classes"][test])))
```