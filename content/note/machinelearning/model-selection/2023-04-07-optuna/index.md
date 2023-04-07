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
  - [](#)
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

```python
import ...

# Define an objective function to be minimized.
def objective(trial):

    # Invoke suggest methods of a Trial object to generate hyperparameters.
    regressor_name = trial.suggest_categorical('regressor', ['SVR', 'RandomForest'])
    if regressor_name == 'SVR':
        svr_c = trial.suggest_float('svr_c', 1e-10, 1e10, log = True)
        regressor_obj = sklearn.svm.SVR(C = svr_c)
    else:
        rf_max_depth = trial.suggest_int('rf_max_depth', 2, 32)
        regressor_obj = sklearn.ensemble.RandomForestRegressor(max_depth = rf_max_depth)
    
    # data
    X, y = sklearn.datasets.fetch_california_housing(return_X_y = True)
    X_train, X_val, y_train, y_val = sklearn.model_selection.train_test_split(X, y, random_state = 0)

    # model training
    regressor_obj.fit(X_train, y_train)
    y_pred = regressor_obj.predict(X_val)

    error = sklearn.metrics.mean_squared_error(y_val, y_pred)

    return error  # An objective value linked with the Trial object.

study = optuna.create_study()  # Create a new study.
study.optimize(objective, n_trials = 100)  # Invoke optimization of the objective function.
```

## 快速开始

```python
import optuna

def objective(trial):
    x = trial.suggest_float("x", -10, 10)
    return (x - 2) ** 2

study = optuna.create_study()
study.optimize(objective, n_trials = 100)
study.best_params  # E.g. {"x": 2.002108042}
```

## 




# 参考

* [Optuna Doc](https://optuna.org/)
* [Optuna GitHub](https://github.com/optuna/optuna)
* [Optuna 可视化调参魔法指南](https://mp.weixin.qq.com/s/RUdYg6OBPIT5jR9ndHiDXg)
