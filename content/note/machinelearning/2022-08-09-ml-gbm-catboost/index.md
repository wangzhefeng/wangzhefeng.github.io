---
title: CatBoost
author: 王哲峰
date: '2022-08-09'
slug: ml-gbm-catboost
categories:
  - machinelearning
tags:
  - ml
  - model
---

<style>
h1 {
    background-color: #2B90B6;
    background-image: linear-gradient(45deg, #4EC5D4 10%, #146b8c 20%);
    background-size: 100%;
    -webkit-background-clip: text;
    -moz-background-clip: text;
    -webkit-text-fill-color: transparent;
    -moz-text-fill-color: transparent;
}
h2 {
    background-color: #2B90B6;
    background-image: linear-gradient(45deg, #4EC5D4 10%, #146b8c 20%);
    background-size: 100%;
    -webkit-background-clip: text;
    -moz-background-clip: text;
    -webkit-text-fill-color: transparent;
    -moz-text-fill-color: transparent;
}
h3 {
    background-color: #2B90B6;
    background-image: linear-gradient(45deg, #4EC5D4 10%, #146b8c 20%);
    background-size: 100%;
    -webkit-background-clip: text;
    -moz-background-clip: text;
    -webkit-text-fill-color: transparent;
    -moz-text-fill-color: transparent;
}
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

- [CatBoost 模型特点](#catboost-模型特点)
- [CatBoost 模型理论](#catboost-模型理论)
- [CatBoost 使用](#catboost-使用)
  - [下载依赖库](#下载依赖库)
  - [下载 CatBoost 库](#下载-catboost-库)
  - [快速开始](#快速开始)
  - [Parameter Config](#parameter-config)
  - [数据可视化](#数据可视化)
- [CatBoost API](#catboost-api)
</p></details><p></p>

# CatBoost 模型特点

CatBoost is a fast, scalabel, high performance open-scource gradient
boosting on decision trees library;

1. Greate quality without parameter tuning
    - Reduce time spent on parameter tuning, because CatBoost provides great results with default parameters;
2. Categorical features support(支持类别性特征, 不需要将类别型特征转换为数字类型)
    - Improve your training results with CastBoost that allows you to use non-numeric factors, 
      instead of having to pre-process your data or spend time and effort turning it to numbers.
3. Fast and scalable GPU version
    - Train your model on a fast implementation of gradient-boosting algorithm for GPU.
     Use a multi-card configuration for large datasets;
4. Imporved accuracy
    - Reduce overfitting when constructing your models with a novel gradient-boosting scheme;
5. Fast prediction
   - Apply your trained model quickly and efficiently even to latency-critical task using CatBoost's models applier;

# CatBoost 模型理论

# CatBoost 使用

## 下载依赖库

```bash
pip install numpy six
```

## 下载 CatBoost 库

```bash
# install catboost
$ pip install catboost
```

## 快速开始

- CatBoostClassifier

```python
import numpy as np
from catboost import CatBoostClassifier, Pool

# initialize data
train_data = np.random.randit(0, 100, size = (100, 10))
train_labels = np.random.randint(0, 2, size = (100))
test_data = catboost_pool = Pool(train_data, train_labels)

# build model
model = CatBoostClassifier(iterations = 2,
                            depth = 2,
                            learning_rate = 1,
                            loss_function = "Logloss",
                            verbose = True)

# train model
model.fit(train_data, train_labels)

# prediction using model
y_pred = model.predict(test_data)
y_pred_proba = model.predict_proba(test_data)
print("class = ", y_pred)
print("proba = ", y_pred_proba)
```

- CatBoostRegressor

```python
import numpy as np
from catboost import CatBoostRegressor, Pool

# initialize data
train_data = np.random.randint(0, 100, size = (100, 10))
train_labels = np.random.randint(0, 100, size = (100))
test_data = np.random.randint(0, 100, size = (50, 10))

# initialize Pool
train_pool = Pool(train_data, train_label, cat_features = [0, 2, 5])
test_pool = Pool(test_data, cat_features = [0, 2, 5])

# build model
model = CatBoostRegressor(iterations = 2, 
                            depth = 2,
                            learning_rate = 1, 
                            loss_function = "RMSE")

# train model
model.fit(train_pool)

# prediction
y_pred = model.predict(test_pool)
print(y_pred)
```

- CatBoost

```python
import numpy as np
from catboost import CatBoost, Pool

# read the dataset
train_data = np.random.randint(0, 100, size = (100, 10))
train_labels = np.random.randint(0, 2, size = (100))
test_data = np.random.randint(0, 100, size = (50, 10))

# init pool
train_pool = Pool(train_data, train_labels)
test_pool = Pool(test_data)

# build model
param = {
    "iterations": 5
}
model = CatBoost(param)

# train model
model.fit(train_pool)

# prediction
y_pred_class = model.predict(test_pool, prediction_type = "Class")
y_pred_proba = model.predict(test_pool, prediction_type = "Probability")
y_pred_raw_vals = model.predict(test_pool, prediction_type = "RawFormulaVal")
print("Class", y_pred_class)
print("Proba", y_pred_proba)
print("Raw", y_pred_raw_valss)
```

## Parameter Config

- Objectives and metrics
    - Regression
        - MAE
        - MAPE
        - Poisson
        - Quantile
        - RMSE
        - LogLinQuantile
        - Lq
        - Huber
        - Expectile
        - FairLoss
        - NumErrors
        - SMAPE
        - R2
        - MSLE
        - MedianAbsoluteError
    - Classification
        - Logloss
        - CrossEntropy
        - Precision
        - Recall
        - F1
        - BalancedAccuracy
        - BalancedErrorRate
        - MCC
        - Accuracy
        - CtrFactor
        - AUC
        - NormalizedGini
        - BriefScore
        - HingeLoss
        - HammingLoss
        - ZeroOneLoss
        - Kapp
        - WKappa
        - LogLikelihoodOfPrediction
    - Multiclassification
        - MultiClass
        - MultiClassOneVsAll
        - Precision
        - Recall
        - F1
        - TotalF1
        - MCC
        - Accuracy
        - HingeLoss
        - HammingLoss
        - ZeroOneLoss
        - Kappa
        - WKappa
    - Ranking

## 数据可视化

下载 `ipywidgets` 可视化库: 

```bash
# install visualization tools
$ pip install ipywidgets
$ jypyter nbextension enable --py widgetsnbextersion
```

CatBoost 数据可视化介绍: 

- [Data Visualization](https://catboost.ai/docs/features/visualization.html)

# CatBoost API

