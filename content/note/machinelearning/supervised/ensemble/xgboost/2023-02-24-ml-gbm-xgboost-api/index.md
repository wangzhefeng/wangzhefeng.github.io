---
title: XGBoost API
author: 王哲峰
date: '2023-02-24'
slug: ml-gbm-xgboost-api
categories:
  - machinelearning
tags:
  - model
  - api
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

- [XGBoost API](#xgboost-api)
  - [核心数据结构](#核心数据结构)
  - [Learning API](#learning-api)
  - [Scikit-Learn API](#scikit-learn-api)
  - [数据可视化 API](#数据可视化-api)
</p></details><p></p>

# XGBoost API

## 核心数据结构

```python
xgb.DMatrix(
    data,
    label = None,
    missing = None,
    weight = None,
    silent = False,
    feature_name = None,
    feature_type = None,
    nthread = None
)
```

```python
xgb.Booster(
    params = None, 
    cache = (),
    model_file = None
)
```

## Learning API

```python
xgb.train(
    params, 
    dtrain,
    num_boost_round = 10,
    evals = (),
    obj = None,
    feval = None,
    maximize = False,
    early_stopping_rounds = None,
    evals_result = None,
    verbose_eval = True,
    xgb_model = None,
    callbacks = None,
    learning_rate = None
)
```

```python
xgb.cv(
    params, 
    dtrain,
    num_boost_round = 10,
    # ---------
    # cv params
    nfold = 3,
    stratified = False,
    folds = None,
    metrics = (),
    # ---------
    obj = None,
    feval = None,
    maximize = False,
    early_stopping_rounds = None,
    fpreproc = None,
    as_pandas = True,
    verbose_eval = None,
    show_stdv = True,
    seed = 0,
    callbacks = None,
    shuffle = True
)
```

## Scikit-Learn API

```python
xgb.XGBRegressor(
    max_depth = 3,
    learning_rate = 0.1, 
    n_estimators = 100,
    verbosity = 1,
    silent = None,
    objective = "reg:squarederror",
    booster = "gbtree",
    n_jobs = 1,
    nthread = None,
    gamma = 0,
    min_child_weight = 1,
    max_delta_step = 0,
    subsample = 1,
    colsample_bytree = 1,
    colsample_bylevel = 1,
    colsample_bynode = 1,
    reg_alpha = 0,
    reg_lambda = 1,
    scale_pos_weight = 1,
    base_score = 0.5,
    random_state = 0,
    seed = None,
    missing = None,
    importance_type = "gain",
    **kwargs
)

xgbr.fit(
    X, y,
    sample_weight = None,
    eval_set = None,
    eval_metric = None, 
    early_stopping_rounds = None,
    verbose = True,
    xgb_model = None,
    sample_weight_eval_set = None,
    callbacks = None
)
```

```python
xgb.XGBClassifier(
    max_depth = 3,
    learning_rate = 0.1,
    n_estimators = 100,
    verbosity = 1,
    silent = None,
    objective = "binary:logistic",
    booster = "gbtree",
    n_jobs = 1,
    nthread = None,
    gamma = 0,
    min_child_weight = 1,
    max_delta_step = 0,
    subsample = 1,
    colsample_bytree = 1,
    colsample_bylevel = 1,
    colsample_bynode = 1,
    reg_alpha = 0,
    reg_lambda = 1,
    scale_pos_weight = 1,
    base_score = 0.5,
    random_state = 0,
    seed = None, 
    missing = None,
    **kwargs
)
xgbc.fit(
    X, 
    y,
    sample_weight = None,
    eval_set = None,
    eval_metric = None,
    early_stopping_rounds = None,
    verbose = True,
    xgb_model = None,
    sample_weight_eval_set = None,
    callbacks = None
)
```

## 数据可视化 API

```python
xgb.plot_importance(
    booster,
    ax = None,
    height = 0.2, 
    xlim = None,
    ylim = None,
    title = "Feature importance",
    xlabel = "F score",
    ylabel = "Features",
    importance_type = "weight"
)
```

