---
title: SVM 使用
author: 王哲峰
date: '2023-04-05'
slug: ml-svm-api
categories:
  - machinelearning
tags:
  - model
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

- [SVM API](#svm-api)
  - [分类](#分类)
    - [SVC](#svc)
    - [LinearSVC](#linearsvc)
    - [SGDClassifier](#sgdclassifier)
  - [回归](#回归)
    - [SVR](#svr)
    - [SGDRegressor](#sgdregressor)
    - [LinearSVR](#linearsvr)
    - [NuSVR](#nusvr)
  - [核函数](#核函数)
  - [方法](#方法)
- [参考](#参考)
</p></details><p></p>

# SVM API

## 分类

### SVC

```python
class sklearn.svm.SVC(
    *, 
    C = 1.0, 
    kernel = 'rbf',  # linear, poly, rbf(默认), sigmoid, precomputed
    degree = 3, 
    gamma = 'scale', 
    coef0 = 0.0, 
    shrinking = True, 
    probability = False, 
    tol = 0.001, 
    cache_size = 200, 
    class_weight = None, 
    verbose = False, 
    max_iter = -1, 
    decision_function_shape = 'ovr', 
    break_ties = False, 
    random_state = None
)
```

核心参数：

* C = 1.0
* kernel = "linear"
* degree = 3
* coef0 = 1

### LinearSVC

> * LinearSVC，Linear Support Vector Classification
> * 大数据集

```python
class sklearn.svm.LinearSVC(
    *,
    C = 1.0,
    loss = 'squared_hinge',
    penalty = 'l2',
    multi_class = 'ovr', 
    dual = True,
    tol = 0.0001,
    fit_intercept = True, 
    intercept_scaling = 1, 
    class_weight = None, 
    verbose = 0, 
    random_state = None, 
    max_iter = 1000
)
```

核心参数：

* C = 1
* loss = "hinge"

### SGDClassifier

> * SGDClassifier，Linear classifiers (SVM, logistic regression, etc.) with SGD training
> * 大数据集

```python
class sklearn.linear_model.SGDClassifier(
    loss = 'hinge',
    *,
    penalty = 'l2',
    alpha = 0.0001,
    l1_ratio = 0.15,
    fit_intercept = True,
    max_iter = 1000,
    tol = 0.001,
    shuffle = True,
    verbose = 0,
    epsilon = 0.1,
    n_jobs = None, 
    random_state = None, 
    learning_rate = 'optimal', 
    eta0 = 0.0, 
    power_t = 0.5, 
    early_stopping = False, 
    validation_fraction = 0.1, 
    n_iter_no_change = 5, 
    class_weight = None, 
    warm_start = False, 
    average = False
)
```

核心参数：

* loss = "hinge"
* alpha = 1 / (1 * 1)

## 回归

### SVR

> Epsilon-Support Vector Regression

```python
class sklearn.svm.SVR(
    *, 
    C = 1.0, 
    kernel = 'rbf', 
    degree = 3, 
    gamma = 'scale', 
    coef0 = 0.0, 
    tol = 0.001, 
    epsilon = 0.1, 
    shrinking = True, 
    cache_size = 200, 
    verbose = False, 
    max_iter = -1
)
```

核心参数：

* C
* kernel
* degree
* gamma
* coef0

### SGDRegressor

```python
class sklearn.linear_model.SGDRegressor(
    loss = 'squared_error', 
    *, 
    penalty = 'l2', 
    alpha = 0.0001, 
    l1_ratio = 0.15, 
    fit_intercept = True, 
    max_iter = 1000, 
    tol = 0.001, 
    shuffle = True, 
    verbose = 0, 
    epsilon = 0.1, 
    random_state = None, 
    learning_rate = 'invscaling', 
    eta0 = 0.01, 
    power_t = 0.25, 
    early_stopping = False, 
    validation_fraction = 0.1, 
    n_iter_no_change = 5, 
    warm_start = False, 
    average = False
)
```

核心参数：

* loss
* penalty
* alpha
* l1_ratio

### LinearSVR

```python
class sklearn.svm.LinearSVR(
    *, 
    epsilon = 0.0, 
    tol = 0.0001, 
    C = 1.0, 
    loss = 'epsilon_insensitive', 
    fit_intercept = True, 
    intercept_scaling = 1.0, 
    dual = True, 
    verbose = 0, 
    random_state = None, 
    max_iter = 1000
)
```

核心参数：

* C
* loss
* epsilon

### NuSVR

> Nu Support Vector Regression

```python
class sklearn.svm.NuSVR(
    *, 
    nu = 0.5, 
    C = 1.0, 
    kernel = 'rbf', 
    degree = 3, 
    gamma = 'scale', 
    coef0 = 0.0, 
    shrinking = True, 
    tol = 0.001, 
    cache_size = 200, 
    verbose = False, 
    max_iter = -1
)
```

核心参数：

* nu
* C
* kernel
* degree
* gamma
* coef0

## 核函数

* `linear`
* `poly`
* `rbf`
* `sigmoid`
* `precomputed`

## 方法

* `decision_function()`
* `densify()`
* `fit()`
* `get_params()`
* `set_params()`
* `partial_fit()`
* `predict()`
* `predict_log_proba()`
* `predict_proba()`
* `score()`
* `sparsify()`

# 参考

* [Scikit-Learn SVM Doc](https://scikit-learn.org/stable/modules/svm.html#svm-kernels)
