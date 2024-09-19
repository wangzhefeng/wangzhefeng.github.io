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
img {
    pointer-events: none;
}
</style>

<details><summary>目录</summary><p>

- [LightGBM 核心数据结构](#lightgbm-核心数据结构)
    - [数据格式](#数据格式)
        - [加载文本文件和二进制文件](#加载文本文件和二进制文件)
        - [加载 numpy 二维数组](#加载-numpy-二维数组)
        - [加载 pandas DataFrame](#加载-pandas-dataframe)
        - [加载 H2O DataTable Frame](#加载-h2o-datatable-frame)
        - [加载 scipy 稀疏矩阵](#加载-scipy-稀疏矩阵)
        - [保存数据为 LightGBM 二进制文件](#保存数据为-lightgbm-二进制文件)
        - [创建验证数据](#创建验证数据)
        - [在数据加载时标识特征名称和类别特征](#在数据加载时标识特征名称和类别特征)
        - [有效利用内存空间](#有效利用内存空间)
    - [Dataset](#dataset)
        - [API 及参数](#api-及参数)
        - [示例](#示例)
        - [注意事项](#注意事项)
- [LightGBM APIs](#lightgbm-apis)
    - [Booster](#booster)
    - [Training](#training)
        - [直接学习](#直接学习)
    - [Booster API 转换](#booster-api-转换)
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
- [Docker 安装和使用](#docker-安装和使用)
    - [CLI 模式](#cli-模式)
        - [安装](#安装)
        - [使用](#使用)
    - [Python 模式](#python-模式)
        - [安装](#安装-1)
        - [使用](#使用-1)
- [参考](#参考)
</p></details><p></p>

# LightGBM 核心数据结构

## 数据格式

> 数据保存在 `lightgbm.Dataset` 对象中

* LibSVM(zero-based)、TSV、CSV、TXT 文本文件
    - 可以包含标题
    - 可以指定 `label` 列、权重列、`query/group id` 列
    - 可以指定一个被忽略的列的列表
* numpy 二维数组
* pandas DataFrame
* H2O DataTable’s Frame
* scipy sparse matrix
* LightGBM 二进制文件

### 加载文本文件和二进制文件

> * LibSVM(zero-based) 、TSV、CSV、TXT 文本文件
> * LightGBM 二进制文件

```python
import lightgbm as lgb

# libsvm
train_svm_data = lgb.Dataset('train.svm')

# tsv
train_tsv_data = lgb.Dataset('train.tsv')

# csv
train_csv_data = lgb.Dataset('train.csv')

# txt
# TODO

# lightgbm bin
train_bin_data = lgb.Dataset('train.bin')
```

### 加载 numpy 二维数组

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

> 将数据保存为 LightGBM 二进制文件会使数据加载更快

```python
import lightgbm as lgb

train_data = lgb.Dataset("train.svm.txt")
train_data.save_binary('train.bin')
```

### 创建验证数据

> 在 LightGBM 中, 验证数据应该与训练数据格式一致

```python
import lightgbm as lgb

# 训练数据
train_data = lgb.Dataset("train.csv")

# 验证数据
validation_data = train_data.create_vaild('validation.svm')
# or
validation_data = lgb.Dataset('validation.svm', reference = train_data)
```

### 在数据加载时标识特征名称和类别特征

> 对于 `categorical_feature` 特征，首先需要将它转换为整数类型，并且只支持非负数。如果转换为连续的范围更佳

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

LightGBM 中的 `Dataset` 对象由于仅仅需要保存离散的数据桶，因此它具有很好的内存效率。
但是由于 numpy array/pandas 对象的内存开销较大，因此当使用它们来创建 `Dataset`` 时，
可以通过下面的方式来节省内存：

* 构造 Dataset 时，设置 `free_raw_data=True`
* 在构造 `Dataset` 之后，手动设置 `raw_data=True`
* 手动调用 `gc`

## Dataset

### API 及参数

Dataset：由 LightGBM 内部使用的数据结构，它存储了数据集。

```python
class lightgbm.Dataset(
    data, 
    label=None, 
    max_bin=None, 
    reference=None, 
    weight=None,
    group=None, 
    init_score=None, 
    silent=False, 
    feature_name='auto',
    categorical_feature='auto', 
    params=None, 
    free_raw_data=True
)
```

* 


### 示例

```python
import lightgbm as lgb

matrix_1 = lgb.Dataset("data/train.svm.txt")
matrix_2 = lgb.Dataset(
    data = np.arange(0, 12).reshape(4, 3),
    label = [1, 2, 3, 4],
    weight = [0.5, 0.4, 0.3, 0.2],
    silent = False,
    feature_name = ["a", "b", "c"],
)

matrix_2.get_ref_chain(ref_limit = 10)
matrix_2.subset(used_indices = [0, 1])
matrix_2.data
matrix_2.label
matrix_2.weight
matrix_2.init_score
matrix_2.group
```

### 注意事项

1. 要确保你的数据集的样本数足够大，从而满足一些限制条件（如：单个节点的最小样本数、单个桶的最小样本数等）。否则会直接报错。

# LightGBM APIs

## Booster

```python
class lightgbm.Booster(
    params = None,
    train_set = None,
    model_file = None,
    silent = False,
)
```




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

### 直接学习


## Booster API 转换

1. 从 LGBMModel 转换到 Booster：通过 `.booster_` 属性来获取底层的 `Booster`。
2. 使用 `Booster` 来预测分类的概率
    - 因为 `Booster` 仅仅提供了 `predict` 接口，而未提供 `predict_proba` 接口，因此需要使用这种转换
    - `LGBMClassifier` 的 `predict_proba` 方法中的源码如下：

    ```python
    class_probs = self.booster_.predict(
        X, 
        raw_score = raw_score,
        num_iteration = num_iteration,
    )

    if self._n_classes > 2:
        return class_probs
    else:
        return np.vstack((1. - class_probs, class_probs)).transpose()
    ```

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

# Docker 安装和使用

## CLI 模式

### 安装

```bash
$ mkdir lightgbm-docker
$ cd lightgbm-docker
$ wget https://raw.githubusercontent.com/Microsoft/LightGBM/master/docker/dockerfile-cli
$ docker build -t lightgbm-cli -f dockerfile-cli .
```

### 使用

```bash
$ docker run -rm -it \
--volume $HOME/lgbm.conf:/lgbm.conf \
--volume $HOME/model.txt:/model.txt \
--volume $HOME/tmp:/out \
lightgbm-cli \
config=lgbm.conf
```

## Python 模式

### 安装

```bash
$ mkdir lightgbm-docker
$ lightgbm-docker
$ wget https://raw.githubusercontent.com/Microsoft/LightGBM/master/docker/dockerfile-python
$ docker build -t lightgbm -f dockerfile-python .
```

### 使用

```bash
$ docker run --rm -it lightgbm
```

# 参考

* [LightGBM各种操作](https://mp.weixin.qq.com/s/iLgVeAKXR3EJW2mHbEWiFQ)
