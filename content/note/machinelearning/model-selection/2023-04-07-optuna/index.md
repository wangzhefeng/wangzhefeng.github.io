---
title: Optuna
subtitle: A hyperparameter optimization framework
author: 王哲峰
date: '2023-04-07'
slug: optuna
categories:
  - machinelearning
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

- [Optuna 简介](#optuna-简介)
- [Optuna 安装](#optuna-安装)
  - [Optuna](#optuna)
  - [Optuna Web Dashboard](#optuna-web-dashboard)
- [Optuna Demo](#optuna-demo)
  - [快速开始](#快速开始)
  - [Scikit-Learn](#scikit-learn)
  - [XGBoost](#xgboost)
  - [LightGBM](#lightgbm)
  - [Keras](#keras)
  - [TensorFlow](#tensorflow)
  - [PyTorch](#pytorch)
- [参考](#参考)
</p></details><p></p>

# Optuna 简介

Optuna是一种自动超参数优化软件框架，专为机器学习而设计。它具有命令式、define-by-run API。
得益于 define-by-run API，使用 Optuna 编写的代码享有高度模块化，
Optuna 的用户可以动态构建超参数的搜索空间

关键特征：

* 轻量级、多功能且与平台无关的架构
    - 通过几乎没有要求的简单安装来处理各种各样的任务
* Pythonic 搜索空间
    - 使用熟悉的 Python 语法（包括条件和循环）定义搜索空间
* 高效的优化算法
    - 采用最先进的算法对超参数进行采样并有效地修剪没有希望的试验
* 易于并行化
    - 将研究规模扩大到数十或数百人或工人，而无需对代码进行少量更改或更改
* 快速可视化
    - 检查来自各种绘图函数的优化历史

基本概念：

* 研究(Study)：基于目标函数的优化
* 试验(Trial)：目标函数的单次执行

# Optuna 安装 

## Optuna

```bash
# PyPI
$ pip install optuna
```

## Optuna Web Dashboard

```bash
$ pip install optuna-dashboard
$ optuna-dashboard sqllite:///db.sqlite3
```

# Optuna Demo

## 快速开始

```python
import optuna
from optuna.visualization import plot_intermediate_values

def objective(trial):
    x = trial.suggest_float("x", -10, 10)
    return (x - 2) ** 2

study = optuna.create_study()
study.optimize(objective, n_trials = 100)

print(study.best_params)  # E.g. {"x": 2.002108042}
plot_intermediate_values(study)
```

## Scikit-Learn

```python
from sklearn import datasets
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import optuna


def objective(trial):
    regressor_name = trial.suggest_categorical(
        'regressor', 
        ['SVR', 'RandomForest']
    )
    if regressor_name == 'SVR':
        svr_c = trial.suggest_float('svr_c', 1e-10, 1e10, log = True)
        regressor_obj = SVR(C = svr_c)
    else:
        rf_max_depth = trial.suggest_int('rf_max_depth', 2, 32)
        regressor_obj = RandomForestRegressor(max_depth = rf_max_depth, n_estimators = 10)
    
    # data
    X, y = datasets.fetch_california_housing(return_X_y = True)
    X_train, X_val, y_train, y_val = train_test_split(X, y, random_state = 0)

    # model training
    regressor_obj.fit(X_train, y_train)
    y_pred = regressor_obj.predict(X_val)

    # metric
    error = mean_squared_error(y_val, y_pred)

    return error


study = optuna.create_study(direction = "maximize")
study.optimize(objective, n_trials = 100)
print(study.best_params)
```

## XGBoost

```python
import xgboost as xgb
import optuna


def objective(trial):
    params = {
        "silent": 1,
        "objective": "binary:logistic",
        "booster": trial.suggest_categorical("booster", ["gbtree", "gblinear", "dart"]),
        "lambda": trial.suggest_float("lambda", 1e-8, 1.0, log = True),
        "alpha": trial.suggest_float("alpha", 1e-8, 1.0, log = True),
    }
    bst = xgb.train(params, dtrain)

    return accuracy

study = optuna.create_study(direction = "maximize")
study.optimize(objective, n_trials = 100)
print(study.best_params)
```

## LightGBM

```python
import lightgbm as lgb
import optuna

def objective(trial):
    param = {
        'objective': 'binary',
        'metric': 'binary_logloss',
        'verbosity': -1,
        'boosting_type': 'gbdt',
        'lambda_l1': trial.suggest_float('lambda_l1', 1e-8, 10.0, log=True),
        'lambda_l2': trial.suggest_float('lambda_l2', 1e-8, 10.0, log=True),
        'num_leaves': trial.suggest_int('num_leaves', 2, 256),
        'feature_fraction': trial.suggest_float('feature_fraction', 0.4, 1.0),
        'bagging_fraction': trial.suggest_float('bagging_fraction', 0.4, 1.0),
        'bagging_freq': trial.suggest_int('bagging_freq', 1, 7),
        'min_child_samples': trial.suggest_int('min_child_samples', 5, 100),
    }

    gbm = lgb.train(param, dtrain)

    return accuracy

study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=100)
```

## Keras

```python
import keras
import optuna

def objective(trial):
    # model
    model = Sequential()
    # model params
    model.add(
        Conv2D(
            filters = trial.suggest_categorical('filters', [32, 64]),
            kernel_size = trial.suggest_categorical('kernel_size', [3, 5]),
            strides = trial.suggest_categorical('strides', [1, 2]),
            activation = trial.suggest_categorical('activation', ['relu', 'linear']),
            input_shape = input_shape
        )
    )
    model.add(Flatten())
    model.add(Dense(CLASSES, activation='softmax'))
    # learning rate
    lr = trial.suggest_float('lr', 1e-5, 1e-1, log=True)
    # model compile
    model.compile(
        loss = 'sparse_categorical_crossentropy', 
        optimizer = RMSprop(lr = lr), 
        metrics=['accuracy']
    )

    return accuracy

study = optuna.create_study(direction = 'maximize')
study.optimize(objective, n_trials = 100)
```

## TensorFlow

```python
import tensorflow as tf
import optuna

def objective(trial):
    n_layers = trial.suggest_int('n_layers', 1, 3)
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Flatten())
    for i in range(n_layers):
        num_hidden = trial.suggest_int(f'n_units_l{i}', 4, 128, log = True)
        model.add(tf.keras.layers.Dense(num_hidden, activation = 'relu'))
    model.add(tf.keras.layers.Dense(CLASSES))

    return accuracy

study = optuna.create_study(direction = 'maximize')
study.optimize(objective, n_trials = 100)
```

## PyTorch

```python
import torch
import optuna

def objective(trial):
    n_layers = trial.suggest_int('n_layers', 1, 3)
    layers = []
    in_features = 28 * 28
    for i in range(n_layers):
        out_features = trial.suggest_int(f'n_units_l{i}', 4, 128)
        layers.append(torch.nn.Linear(in_features, out_features))
        layers.append(torch.nn.ReLU())
        in_features = out_features
    layers.append(torch.nn.Linear(in_features, 10))
    layers.append(torch.nn.LogSoftmax(dim=1))
    model = torch.nn.Sequential(*layers).to(torch.device('cpu'))

    return accuracy

study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=100)
```

# 参考

* [Optuna Doc](https://optuna.org/)
* [Optuna GitHub](https://github.com/optuna/optuna)
* [Optuna 可视化调参魔法指南](https://mp.weixin.qq.com/s/RUdYg6OBPIT5jR9ndHiDXg)
