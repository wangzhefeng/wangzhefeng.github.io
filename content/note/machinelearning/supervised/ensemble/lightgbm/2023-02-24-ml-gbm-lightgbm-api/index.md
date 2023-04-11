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
  - [加载文本文件和二进制文件](#加载文本文件和二进制文件)
  - [加载 numpy 2 维数组](#加载-numpy-2-维数组)
  - [加载 pandas DataFrame](#加载-pandas-dataframe)
  - [加载 H2O DataTable Frame](#加载-h2o-datatable-frame)
  - [加载 scipy 稀疏矩阵](#加载-scipy-稀疏矩阵)
  - [保存数据为 LightGBM 二进制文件](#保存数据为-lightgbm-二进制文件)
  - [创建验证数据](#创建验证数据)
  - [在数据加载时标识特征名称和类别特征](#在数据加载时标识特征名称和类别特征)
  - [有效利用内存空间](#有效利用内存空间)
- [LightGBM APIs](#lightgbm-apis)
  - [Training](#training)
  - [模型保存与加载](#模型保存与加载)
  - [交叉验证](#交叉验证)
  - [特征交互约束](#特征交互约束)
  - [自定义评价函数](#自定义评价函数)
  - [自定义损失函数](#自定义损失函数)
  - [提前停止训练](#提前停止训练)
  - [Callbacks](#callbacks)
  - [Plotting](#plotting)
- [LightGBM 调参参数](#lightgbm-调参参数)
  - [参数设置方式](#参数设置方式)
  - [参数类型](#参数类型)
    - [train 方法参数](#train-方法参数)
  - [调参技巧](#调参技巧)
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

> 数据保存在 `lightgbm.Dataset` 对象中

* LibSVM(zero-based)、TSV、CSV、TXT 文本文件
* numpy 二维数组
* pandas DataFrame
* H2O DataTable’s Frame
* scipy sparse matrix
* LightGBM 二进制文件

## 加载文本文件和二进制文件

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

## 加载 numpy 2 维数组

```python
import liggtgbm as lgb

data = np.random.rand(500, 10)
label = np.random.randint(2, size = 500)
train_array = lgb.Dataset(data, label = label)
```

## 加载 pandas DataFrame

## 加载 H2O DataTable Frame

## 加载 scipy 稀疏矩阵

> scipy.sparse.csr_matrix 数组

```python
import lightgbm as lgb
import scipy

csr = scipy.sparse.csr_matirx((dat, (row, col)))
train_sparse = lgb.Dataset(csr)
```

## 保存数据为 LightGBM 二进制文件

```python
import lightgbm as lgb

train_data = lgb.Dataset("train.svm.txt")
train_data.save_binary('train.bin')
```

> 将数据保存为 LightGBM 二进制文件会使数据加载更快

## 创建验证数据

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

## 在数据加载时标识特征名称和类别特征

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

## 有效利用内存空间

The Dataset object in LightGBM is very memory-efficient, 
it only needs to save discrete bins. However, Numpy/Array/Pandas object is memory expensive. 
If you are concerned about your memory consumption, you can save memory by:

- 1.Set `free_raw_data=True` (default is `True`) when constructing the Dataset
- 2.Explicitly set `raw_data=None` after the Dataset has been constructed
- Call `gc`

# LightGBM APIs

## Training

LightGBM API：

* `lightgbm..train(params, train_set, num_boost_round, ...)`
* `lightgbm.cv(params, train_ste, num_boost_round, ...)`

Scikit-learn API：

* `LGBMModel(boosting\ *type, num*\ leaves, ...)`
* `LGBMClassifier(boosting\ *type, num*\ leaves, ...)`
* `LGBMRegressor(boosting\ *type, num*\ leaves, ...)`
* `LGBMRanker(boosting\ *type, num*\ leaves, ...)`
* `.fit()`
* `.predict()`
* `.predict_proba()`

## 模型保存与加载

* save_model()
* model_to_string()
* lightgbm.Booster(model_file, model_str)

## 交叉验证

## 特征交互约束

## 自定义评价函数

## 自定义损失函数

## 提前停止训练

## Callbacks

* `early_stopping(stopping_round, ...)`
* `print_evaluation(period, show_stdv)`
* `record_evaluation(eval_result)`
* `reset_parameter(**kwargs)`

## Plotting

* `plot_importance(booster, ax, height, xlim, ...)`
* `plot_split_value_histogram(booster, feature)`
* `plot_metric(booster, metric, ...)`
* `plot_tree(booster, ax, tree_index, ...)`
* `create_tree_digraph(booster, tree_index, ...)`

# LightGBM 调参参数

## 参数设置方式

* 命令行参数
* 参数配置文件
* Python 参数字典

## 参数类型

* 核心参数
    - Booster 参数
* 学习控制参数
* IO 参数
* 目标参数
* 度量参数
* 网络参数
* GPU 参数
* 模型参数
* 其他参数

### train 方法参数

* objective
    - "regression"
    - "regression_l1"
    - "tweedie"
    - "binary"
    - "multiclass"
    - "multiclassova"
    - "cross_entropy"
* metric
    - "rmse"
    - "l2"
    - "l1"
    - "tweedie"
    - "binary_logloss"
    - "multi_logloss"
    - "auc"
    - "cross_entropy"
* boosting
    - "gbdt"
    - "rf"
    - "dart"
    - "goss"
* num_boost_round：默认 100
* learning_rate： 默认 0.1
* num_class
* num_leaves：默认 31
* num_threads 线程数
* seed
* max_depth：默认 -1
* min_data_in_leaf
* bagging_fraction
* feature_fraction
* extra_trees
* early_stopping_round
* monotone_constraints
* monotone_constraints_method
    - "basic"
    - "intermediate"
    - "advanced"
* interaction_constraints
* verbosity
    - <0
    - 0
    - 1
    - >1
* is_unbalance
* device_type
    - "cpu"
    - "gpu"
* force_col_wise：指定训练时是否强制按列构建直方图。如果数据有太多列，则将此参数设置为 True 将通过减少内存使用来提高训练过程速度
* force_row_wise：指定是否在训练时强制构建逐行直方图。如果数据行太多，则将此参数设置为 True 将通过减少内存使用来提高训练过程速度

## 调参技巧

<font size=0.01em>

|  参数 | 提高模型训练速度 | 提高准确率 | 加强拟合 | 处理过拟合(降低拟合) |
|------|---------------|----|----|----|
| 训练数据 | | 增加训练数据 | 增加训练数据 | 增加训练数据 |
| `dart` | 尝试使用 `dart`(带dropout的树) | 尝试使用 `dart`(带dropout的树) | |
| `bagging_fraction`</br>`bagging_freq` | 使用 bagging 样本采样，设置`bagging_fraction`</br>`bagging_freq` | | | 使用 bagging 样本采样，设置`bagging_fraction`</br>`bagging_freq` |
| `feature_fraction` | 使用子特征采样，</br>设置`feature_fraction` | | | 使用子特征采样，</br>设置</br>`feature_fraction` |
| `learning_rate` | | 使用较小的`learning_rate` 和较大的`num_iterations` | 使用较小的`learning_rate` 和较大的`num_iterations` | |
| `max_bin`</br>(会减慢训练速度) | 使用较小的</br>`max_bin` | 使用较大的</br>`max_bin` | 使用较大的</br>`max_bin` | 使用较小的</br>`max_bin` |
| `num_leaves`| 使用较小的</br>`num_leaves` | 使用较大的</br>`num_leaves` | 使用较大的</br>`num_leaves` | 使用较小的</br>`num_leaves` |
| `max_depth` | | | | 尝试`max_depth` 避免树生长太深 |
| `min_data_in_leaf`和`min_sum_`</br>`hessian_in_leaf`| | | | 使用`min_data_in_leaf` 和`min_sum_`</br>`hessian_in_leaf` |
| `lambda_l1`、`lambda_l2`、`min_gain_to_split` | | | | 尝试使用`lambda_l1`、`lambda_l2` 和`min_gain_to_split` 做正则化 |
| `path_smooth` | | | | 尝试提高</br>`path_smooth` |
| `extra_trees` | | | | 尝试使用`extra_trees` |
| `save_binary` | 使用`save_binary`加速数据加载速度 | | | |
| 并行           | 使用并行训练                     | | | |

</font>

# 参考

* [LightGBM各种操作](https://mp.weixin.qq.com/s/iLgVeAKXR3EJW2mHbEWiFQ)
