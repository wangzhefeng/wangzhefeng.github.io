---
title: LightGBM API
author: 王哲峰
date: '2023-02-24'
slug: ml-gbm-lightgbm-api
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

- [LightGBM 安装](#lightgbm-安装)
  - [pip 安装](#pip-安装)
  - [从源码安装](#从源码安装)
- [LightGBM 核心数据结构](#lightgbm-核心数据结构)
  - [数据接口](#数据接口)
    - [加载文本文件和二进制文件](#加载文本文件和二进制文件)
    - [加载 numpy 2 维数组](#加载-numpy-2-维数组)
    - [加载 pandas DataFrame](#加载-pandas-dataframe)
    - [加载 H2O DataTable Frame](#加载-h2o-datatable-frame)
    - [加载 scipy 稀疏矩阵](#加载-scipy-稀疏矩阵)
    - [保存数据为 LightGBM 二进制文件](#保存数据为-lightgbm-二进制文件)
    - [创建验证数据](#创建验证数据)
    - [在数据加载时标识特征名称和类别特征](#在数据加载时标识特征名称和类别特征)
    - [有效利用内存空间](#有效利用内存空间)
- [LightGBM 模型 API](#lightgbm-模型-api)
  - [Training API](#training-api)
  - [Scikit-learn API](#scikit-learn-api)
  - [Callbacks](#callbacks)
  - [Plotting](#plotting)
- [参考](#参考)
</p></details><p></p>

# LightGBM 安装

LightGBM 的安装非常简单，在 Linux 下很方便的就可以开启GPU训练。可以优先选用从 pip 安装，如果失败再从源码安装

## pip 安装

```bash
# 默认版本
$ pip install lightgbm

# MPI 版本
$ pip install lightgbm --install-option=--mpi

# GPU 版本
$ pip install lightgbm --install-option=--gpu
```

## 从源码安装

```bash
$ git clone --recursive https://github.com/microsoft/LighGBM
$ cd LightGBM
$ mkdir build
$ cd build
cmake ..
```

开启 MPI 通信机制，训练更快：

```bash
$ cmake -DUSE_MPI=ON ..
```

GPU 版本，训练更快：

```bash
$ cmake -DUSE_GPU=1 ..
$ make -j4
```

# LightGBM 核心数据结构

* `Dataset(data, label, reference, weight, ...)`
* `Booster(params, train_set, model_file, ...)`

## 数据接口

> 数据保存在 `lightgbm.Dataset` 对象中

* LibSVM(zero-based)、TSV、CSV、TXT 文本文件
* numpy 二维数组
* pandas DataFrame
* H2O DataTable’s Frame
* scipy sparse matrix
* LightGBM 二进制文件

### 加载文本文件和二进制文件

> LibSVM(zero-based) 文本文件、LightGBM 二进制文件

```python
import lightgbm as lgb

# csv
train_csv_data = lgb.Dataset('train.csv')

# tsv
train_tsv_data = lgb.Dataset('train.tsv')

# libsvm
train_svm_data = lgb.Dataset('train.svm')

# lightgbm bin
train_bin_data = lgb.Dataset('train.bin')
```

### 加载 numpy 2 维数组

```python
import liggtgbm as lgb

data = np.random.rand(500, 10)
label = np.random.randint(2, size = 500)
train_array = lgb.Dataset(data, label = label)
```

### 加载 pandas DataFrame

### 加载 H2O DataTable Frame

### 加载 scipy 稀疏矩阵

> scipy.sparse.csr_matrix 数组

```python
import lightgbm as lgb
import scipy

csr = scipy.sparse.csr_matirx((dat, (row, col)))
train_sparse = lgb.Dataset(csr)
```

### 保存数据为 LightGBM 二进制文件

```python
import lightgbm as lgb

train_data = lgb.Dataset("train.svm.txt")
train_data.save_binary('train.bin')
```

> 将数据保存为 LightGBM 二进制文件会使数据加载更快

### 创建验证数据

```python
import lightgbm as lgb

# 训练数据
train_data = lgb.Dataset("train.csv")

# 验证数据
validation_data = train_data.create_vaild('validation.svm')
# or
validation_data = lgb.Dataset('validation.svm', reference = train_data)
```

> 在 LightGBM 中, 验证数据应该与训练数据一致(格式)

### 在数据加载时标识特征名称和类别特征

```python
import numpy as np
import lightgbm as lgb

data = np.random.rand(500, 10)
label = np.random.randint(2, size = 500)
train_array = lgb.Dataset(data, label = label)
w = np.random.rand(500, 1)

train_data = lgb.Dataset(data, 
                        label = label, 
                        feature_name = ['c1', 'c2', 'c3'], 
                        categorical_feature = ['c3'],
                        weight = w,
                        free_raw_data = True)
# or
train_data.set_weight(w)

train_data.set_init_score()

train_data.set_group()
```

### 有效利用内存空间

The Dataset object in LightGBM is very memory-efficient, 
it only needs to save discrete bins. However, Numpy/Array/Pandas object is memory expensive. 
If you are concerned about your memory consumption, you can save memory by:

- 1.Set `free_raw_data=True` (default is `True`) when constructing the Dataset
- 2.Explicitly set `raw_data=None` after the Dataset has been constructed
- Call `gc`

# LightGBM 模型 API

## Training API

* `train(params, train_set, num_boost_round, ...)`
* `cv(params, train_ste, num_boost_round, ...)`

## Scikit-learn API

* `LGBMModel(boosting\ *type, num*\ leaves, ...)`
* `LGBMClassifier(boosting\ *type, num*\ leaves, ...)`
* `LGBMRegressor(boosting\ *type, num*\ leaves, ...)`
* `LGBMRanker(boosting\ *type, num*\ leaves, ...)`

```python
lightgbm.LGBMClassifier(
    boosting_type = "gbdt", # gbdt, dart, goss, rf
    num_leaves = 31, 
    max_depth = -1, 
    learning_rate = 0.1,
    n_estimators = 100,
    subsample_for_bin = 200000,
    objective = None, 
    class_weight = None,
    min_split_gain = 0.0,
    min_child_weight = 0.001, 
    min_child_samples = 20,
    subsample = 1.0,
    subsample_freq = 0,
    colsample_bytree = 1.0,
    reg_alpha = 0.0,
    reg_lambda = 0.0,
    random_state = None,
    n_jobs = -1, 
    silent = True,
    importance_type = "split",
    **kwargs
)

lgbc.fit(X, y,
    sample, 
    weight = None, 
    init_score = None,
    eval_set = None,
    eval_names = None, 
    eval_sample_weight = None,
    eval_class_weight = None,
    eval_init_score = None,
    eval_metric = None,
    early_stopping_rounds = None,
    verbose = True,
    feature_name = "auto",
    categorical_feature = "auto",
    callbacks = None
)

lgbc.predict(
    X, 
    raw_score = False,
    num_iteration = None,
    pred_leaf = False,
    pred_contrib = False,
    **kwargs
)

lgbc.predict_proba(
    X, 
    raw_score = False,
    num_iteration = None,
    pred_leaf = False,
    pred_contrib = False,
    **kwargs
)
```

```python
lightgbm.LGBMRegressor(boosting_type = "gbdt",
                      num_leaves = 31,
                      max_depth = -1,
                      learning_rate = 0.1,
                      n_estimators = 100,
                      subsample_for_bin = 200000,
                      objective = None,
                      class_weight = None,
                      min_split_gain = 0.0,
                      min_child_weight = 0.001,
                      min_child_samples = 20,
                      subsample = 1.0,
                      subsample_freq = 0,
                      colsample_bytree = 1.0,
                      reg_alpha = 0.0,
                      reg_lambda = 0.0,
                      random_state = None,
                      n_jobs = -1,
                      silent = True,
                      importance_type = "split",
                      **kwargs)

lgbr.fit(X, y, sample_weight = None,
        init_score = None, 
        eval_set = None,
        eval_names = None,
        eval_sample_weight = None,
        eval_init_score = None,
        eval_metric = None,
        early_stopping_rounds = None,
        verbose = True,
        feature_name = "auto",
        categorical_feature = "auto",
        callbacks = None)

lgbr.predict(X, 
            raw_score = False, 
            num_iteration = None, 
            pred_leaf = False,
            pred_contrib = False,
            **kwargs)
```

## Callbacks

- `early_stopping(stopping_round, ...)`
- `print_evaluation(period, show_stdv)`
- `record_evaluation(eval_result)`
- `reset_parameter(**kwargs)`

```python
early_stopping(stopping_round, ...)
print_evaluation(period, show_stdv)
record_evaluation(eval_result)
reset_parameter(**kwargs)
```

## Plotting

- `plot_importance(booster, ax, height, xlim, ...)`
- `plot_split_value_histogram(booster, feature)`
- `plot_metric(booster, metric, ...)`
- `plot_tree(booster, ax, tree_index, ...)`
- `create_tree_digraph(booster, tree_index, ...)`

```python
plot_importance(booster, ax, height, xlim, ...)
plot_split_value_histogram(booster, feature)
plot_metric(booster, ax, tree, index, ...)
plot_tree(booster, ax, tree_index, ...)
create_tree_digraph(booster, tree_index, ...)
```



# 参考

