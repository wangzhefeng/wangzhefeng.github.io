---
title: XGBoost
author: 王哲峰
date: '2022-07-31'
slug: ml-gbm-xgboost
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

- [XGBoost 模型特点](#xgboost-模型特点)
  - [XGBoost 特点](#xgboost-特点)
  - [XGBoost 优势](#xgboost-优势)
  - [XGBoost 缺点(LightGBM 的出发点)](#xgboost-缺点lightgbm-的出发点)
- [XGBoost 模型理论](#xgboost-模型理论)
  - [模型目标函数](#模型目标函数)
  - [模型目标](#模型目标)
  - [模型损失函数定义](#模型损失函数定义)
  - [模型正则项定义](#模型正则项定义)
  - [模型目标函数求解](#模型目标函数求解)
  - [模型中决策树节点分裂算法](#模型中决策树节点分裂算法)
  - [自定义损失函数](#自定义损失函数)
- [XGBoost 使用](#xgboost-使用)
  - [下载安装 xgboost 库](#下载安装-xgboost-库)
    - [Ubuntu/Debian](#ubuntudebian)
    - [OSX](#osx)
    - [Windows](#windows)
  - [导入 xgboost 库](#导入-xgboost-库)
  - [数据接口](#数据接口)
  - [参数设置](#参数设置)
  - [模型训练](#模型训练)
  - [模型预测](#模型预测)
  - [模型结果可视化](#模型结果可视化)
- [XGBoost API](#xgboost-api)
  - [Core Data Structure](#core-data-structure)
  - [Learning API](#learning-api)
  - [Scikit-Learn API](#scikit-learn-api)
  - [Plotting API](#plotting-api)
  - [Callback API](#callback-api)
  - [Dask API](#dask-api)
- [XGBoost调参](#xgboost调参)
  - [参数类型](#参数类型)
  - [参数调优的一般策略](#参数调优的一般策略)
  - [参数调优步骤](#参数调优步骤)
</p></details><p></p>


# XGBoost 模型特点

## XGBoost 特点

- 弱学习器
   - 传统的 GBDT 以 CART 作为基函数(base learner,b ase classifier, base
      function), 而 XGBoost 除了可以使用 CART, 还支持线性分类器 (linear
      classifier, linear regression, logistic regression)
   - 基于预排序(pre_sorted)方法构建决策树
      - 优点: 精确地找到分割点
      - 缺点: 内存消耗大, 时间消耗大, 对cache优化不友好
      - 首先, 对所有特征都按照特征的数值进行排序；
      - 其次, 在遍历分割点的时候用O(#data)的代价找到一个特征上的最好分割点；
      - 最后找到一个特征的分割点后, 将数据分裂成左右子节点；
- 梯度
   - 传统的 GBDT 在优化时只用到一阶导数信息(负梯度), XGBoost
      则对代价函数进行了二阶泰勒展开, 同时用到一阶和二阶导数. 且 XGBoost
      工具支持自定义代价函数, 只要函数可一阶和二阶求导

## XGBoost 优势

- 正则化的 GBM (Regularization)
   - 控制模型过拟合问题
- 并行化的 GBM (Parallel Processing)
- 高度灵活性 (High Flexibility)
- 能够处理缺失数据
- 自带树剪枝
- 内置交叉验证
- 能够继续使用当前训练的模型
- 可扩展性强
- 为稀疏数据设计的决策树训练方法
- 理论上得到验证的加权分位数粗略图法
- 分布式计算
- 设计高效核外计算, 进行 cache-aware 数据块处理

## XGBoost 缺点(LightGBM 的出发点)

- 每轮迭代时, 都需要遍历整个训练数据多次. 如果把整个训练数据装进内存则会限制训练数据的大小；
  如果不装进内存, 反复地读写训练数据又会消耗非常大的时间
- 预排序方法(pre-sorted): 首先, 空间消耗大. 这样的算法需要保存数据的特征值, 
  还保存了特征排序的结果(例如排序后的索引, 为了后续快速的计算分割点), 这里需要消耗训练数据两倍的内存. 其次时间上也有较大的开销, 在遍历每一个分割点的时候, 都需要进行分裂增益的计算, 消耗的代价大
- 对 cache 优化不友好. 在预排序后, 特征对梯度的访问是一种随机访问, 
  并且不同的特征访问的顺序不一样, 无法对 cache 进行优化. 同时, 在每一层长树的时候, 
  需要随机访问一个行索引到叶子索引的数组, 并且不同特征访问的顺序也不一样, 
  也会造成较大的 cache miss

# XGBoost 模型理论

## 模型目标函数

`$$L(\phi)=\sum_{i=1}^{n} l(y_i, \hat{y_i}) + \sum_{k=1}^{K} \Omega(f_k)$$`

其中: 

- `$L(\cdot)$` : 目标函数；
- `$l(\cdot)$` : 经验损失(误差)函数, 通常是凸函数. 
  用于刻画预测值`$\hat{y_i}$` 和真实值`$y_i$` 之间的差异, 即模型对训练数据的拟合程度
- `$\Omega(\cdot)$` : 模型的正则项. 用于降低模型的复杂度, 减轻过拟合问题
   - 决策树的叶子节点数量
   - 决策树的树深度
   - 决策树的叶节点权重得分的L1, L2正则
- `$f_k$` : 第 `$k$` 棵树
- `$y_i$` : 目标变量
- `$\hat{y_i}=\sum_{k=1}^{K}f_k(x_i), f_k \in F$`
   - 回归: 预测得分
   - 分类: 预测概率
   - 排序: 排序得分

## 模型目标

`$$\min L(\phi)$$`

## 模型损失函数定义

根据 Gradient Boosting 思想, 假设第 `$t$` 轮迭代的目标函数: 

`$$L^{(t)} = \sum_{i=1}^{n} l \bigg(y_i, \hat{y_i}^{(t-1)} + f_t(x_i) \bigg) + \Omega(f_t) + constant$$`

对目标函数 `$L^{(t)}$` 在 `$\hat{y_i}^{(t-1)}$`
处进行二阶泰勒展开, 可以加速优化过程, 得到目标函数的近似: 

`$$L^{(t)} \simeq \sum_{i=1}^{n} [l(y_i, \hat{y_i}^{(t-1)}) + g_i f_t (x_i) + \frac{1}{2} h_i f_t^2 (x_i)]+ \Omega(f_t) + constant$$`

- 函数 `$f(x)$` 在 `$x_0$` 处的二阶泰勒展开式: 

`$$f(x) = f(x_0) + f' (x_0)(x - x_0) + f''(x_0)(x - x_0)^2$$`

- 目标函数 `$l(y_i, x)$` 在 `$\hat{y_i}^{(t-1)}$`
  处的二阶泰勒展开式: 

`$$l(y_i, x) = l(y_i, \hat{y_i}^{(t - 1)}) + \frac{\partial l(y_i, \hat{y_i}^{(t - 1)})}{\partial \hat{y_i}^{(t - 1)}} (x - \hat{y_i}^{(t - 1)}) + \frac{1}{2} \frac{{\partial ^2} l(y_i, \hat{y_i}^{(t - 1)}) } {\partial \hat{y_i}^{(t - 1)}} (x - \hat{y_i}^{(t - 1)})^2$$`

- 令 `$x= \hat{y_i}^{(t-1)} + f_t (x_i)$`, 记一阶导数为

`$$g_i = \frac{\partial l(y_i, \hat{y_i}^{(t - 1)})}{\partial \hat{y_i}^{(t - 1)}}$$` , 

记二阶导数为

`$$h_i = \frac{{\partial ^2} l(y_i, \hat{y_i}^{(t - 1)})}{\partial \hat{y_i}^{(t - 1)}}$$` ；

可以得到

`$$l(y_i, \hat{y}^{(t-1)} + f_t(x_i)) = l(y_i, \hat{y}^{(t - 1)}) + g_i f_t(x_i) + \frac{1}{2} h_i f_t^2 (x_i)$$`

上面的目标函数 `$L^{(t)}$` 中, 第一项 `$l(y_i, \hat{y_i}^{(t-1)})$` 以及 `$constant$` 为常数项, 
在优化问题中可直接删除: 

`$${\tilde L}^{(t)} = \sum_{i=1}^{n} [g_{i}f_{t}(x_i) + \frac{1}{2}h_{i}f_{t}^{2}(x_i)]  + \Omega (f_t)$$`

## 模型正则项定义

通过 `叶子节点的权重得分` 和 `叶子节点的索引值(结构)` 定义一棵树: 

`$$f_t(x) = \omega_{q(x)}, \omega \in R^T, q: R^d \rightarrow \{1, 2, \ldots, T\}$$`

其中: 

- `$\omega$`: 叶子节点权重得分向量

- `$q(\cdot)$`: 树结构

定义正则项(可以使其他形式): 

`$\Omega(f_t)=\gamma T + \frac{1}{2}\lambda\sum_{j=1}^{T}\omega_j^2$`

其中: 

- `$T$` 是 叶节点的个数

- `$\omega_j^2$` 是叶子节点权重得分的L2范数

记决策树每个叶节点上的样本集合为: 

`${I_j} = \{ {i | q(x_i) = j} \}$`

将正则项 `$\Omega(f_t)$` 展开: 

`$\eqalign{
& {\tilde L}^{(t)} 
= \sum_{i=1}^{n}[g_i f_t(x_i) + \frac{1}{2}h_i f_{t}^{2}(X_I)] + \Omega(f_t) \cr 
& \;\;\;\;\;{\rm{ 
= }} \sum_{i=1}^{n}[g_i \omega_{q(x_i)} + \frac{1}{2} h_i \omega_{q(x_i)}^{2}] + \gamma T + \frac{1}{2} \lambda \sum_{j = 1}^{T} w_j^2   \cr 
& \;\;\;\;\;{\rm{ 
= }} \sum_{j = 1}^{T}[(\sum_{i \in I_j} g_i ) w_j + \frac{1}{2} (\sum_{i \in {I_j}} h_i  + \lambda ) w_j^2 ]  + \gamma T \cr}$`

记 `$G_j=\sum_{i\in I_j}g_i$` ,  `$H_j=\sum_{i\in I_j}h_i$`

`$\eqalign{
& {\tilde L}^(t) 
= \sum{j = 1}^{T} [(\sum_{i \in {I_j}} g_i) w_j + \frac{1}{2} (\sum_{i \in {I_j}} h_i  + \lambda ) w_j^2 ]  + \gamma T \cr
& \;\;\;\;\;{\rm{ 
= }} \sum_{j = 1}^{T} [G_j w_j + \frac{1}{2} (H_j  + \lambda ) w_j^2 ]  + \gamma T \cr
}$`

## 模型目标函数求解

对于固定的树结构 `$q(x)$` , 对 `$\omega_j$`
求导等于0, 得到目标函数的解析解 `$\omega_{j}^{\star}$` : 

`$w_{j}^{\star} = \frac{\sum\limits_{i \in I_j} g_i}{\sum\limits_{i \in I_j} h_i + \lambda}$`

即

`$w_{j}^{\star} = -\frac{G_j}{H_j+\lambda} $`

将上面得到的解析解带入目标函数: 

`${{\tilde L}^{( t )}}=-\frac{1}{2}\sum_{j=1}^{T}\frac{G_j^2}{H_j+\lambda} + \gamma T$`

   这里的 `${{\tilde L}^{( t )}}$`
   代表了当指定一个树结构时, 在目标函数上最多减少多少, 这里叫做 `结构分数(structure score)` ；这个分数越小, 代表这个树的结构越好；

## 模型中决策树节点分裂算法

(1)贪心算法(Exact Greedy Algorithm for split finding)

每次尝试对已有叶节点进行一次分割, 分割的规则: 

`$Gain = \frac{1}{2}\Bigg[\frac{G_{L}^{2}}{H_K + \lambda} + \frac{G_{R}^{2}}{H_R + \lambda} - \frac{(G_L + G_R)^{2}}{H_L + H_R + \lambda}\Bigg] - \gamma$`

其中: 

- `$\frac{G_{L}^{2}}{H_K + \lambda}$` : 左子树分数；

- `$\frac{G_{R}^{2}}{H_R + \lambda}$` : 右子树分数；

- `$\frac{(G_L + G_R)^{2}}{H_L + H_R + \lambda}$` : 
   不分割可以得到的分数；

- `$\gamma$` : 假如新的子节点引入的复杂度代价；

对树的每次扩展, 都要枚举所有可能的分割方案；并且对于某次分割, 都要计算每个特征值左边和右边的一阶和二阶导数和, 从而计算这次分割的增益
`$Gain$` ； 对于上面的分割增益
`$Gain$` , 都要判断分割每次分割对应的 `$Gain$`
的大小, 并且进行优化, 取最小的
`$Gain$` , 直到当新引入一个分割带来的增益小于某个阈值时, 就去掉这个分割. 这里的优化, 相当于对树进行剪枝. 

(2)近似算法(Approximate Algorithm for split finding)

   针对数据量大的情况, 不能直接计算

## 自定义损失函数



# XGBoost 使用

## 下载安装 xgboost 库

下载安装 xgboost Pre-build wheel Python 库

```bash
$ pip3 install xgboost
```

Building XGBoost from source

   1. 首先从 C++ 代码构建共享库(libxgboost.so 适用于 Linux OSX 和 xgboost.dll Windows)
   2. 然后安装语言包(例如Python Package)

1.从 C++ 代码构建共享库

- 目标: 
   - 在 Linux/OSX 上, 目标库是: libxgboost.so
   - 在 Windows 上, 目标库是: xgboost.dll

- 环境要求

   - 最新的支持 C++ 11 的 C++ 编译器(g++-4.8 or higher)

   - CMake 3.2 or higher

### Ubuntu/Debian

### OSX

(1) install with pip

```bash
# 使用 Homebrew 安装 gcc-8, 开启多线程(多个CUP线程训练)
brew install gcc@8

# 安装 XGBoost
pip3 install xgboost
# or
pip3 install --user xgboost
```

(2) build from the source code

```bash
# 使用 Homebrew 安装 gcc-8, 开启多线程(多个CUP线程训练)
brew install gcc@8

# Clone the xgboost repository
git clone --recursive https://github.com/dmlc/xgboost

# Create $`build/$` dir and invoker CMake
# Make sure to add CC=gcc-8 CXX-g++-8 so that Homebrew GCC is selected
# Build XGBoost with make
mkdir build
cd build
CC=gcc-8 CXX=g++-8 cmake ..
make -j4
```

### Windows


2.安装 Python 包

- Python 包位于 python-package/, 根据安装范围分以下情况安装: 

method 1: 在系统范围内安装, 需要root权限

```bash
# 依赖模块:distutils
# Ubuntu
sudo apt-get install python-setuptools
# MacOS
# 1.Download $`ez_setup.py$` module from "https://pypi.python.org/pypi/setuptools"
# 2. cd to the dir put the $`ez_setup.py$`
# 3. $`python ez_setup.py$`

cd python-package
sudo python setup.py install
```

```bash
# 设置环境变量 在: $`~/.zshrc$`
export PYTHONPATH=~/xgboost/python-package
```

method 2:  仅为当前用户安装

```bash
cd python-package
python setup.py develop --user
```

## 导入 xgboost 库

```python
import xgboost as xgb
```

## 数据接口

## 参数设置

```python
# dict
param = {
    "max_depth": 2,
    "eta": 1,
    "objective": "binary:logistic"
}
# list
param["nthread"] = 4
param["eval_metric"] = "auc"
param["eval_metric"] = ["auc", "ams@0"]

# 模型评估验证设置
evallist = [
    (dtest, "eval"),
    (dtrain, "train")
]
```

## 模型训练

```python
# 训练模型
# =======
num_round = 20
bst = xgb.train(param, 
                dtrain, 
                num_round, 
                evallist,
                evals = evals,
                early_stopping_rounds = 10)
bst.best_score
bst.best_iteration
bst.best_ntree_limit


# 保存模型
# =======
bst.save_model("0001.model")
# dump model
bst.dump_model("dump.raw.txt")
# dump model with feature map
bst.dump_model("dump.raw.txt", "featmap.txt")


# 加载训练好的模型
# ==============
bst = xgb.Booster({"nthread": 4})
bst.load_model("model.bin")
```

## 模型预测

```python
dtest = xgb.DMatrix(np.random.rand(7, 10))
y_pred = bst.predict(dtest)
# early stopping
y_pred = bst.predict(dtest, ntree_limit = bst.best_ntree_limit)
```

## 模型结果可视化

```python
import matplotlib.pyplot as plt
import graphvize
import xgboost as xgb

xgb.plot_importance(bst)
xgb.plot_tree(bst, num_trees = 2)
# IPython option(将目标树转换为graphviz实例)
xgb.to_graphviz(bst, num_trees = 2)
```


# XGBoost API

## Core Data Structure

```python
xgb.DMatrix(data,
            label = None,
            missing = None,
            weight = None,
            silent = False,
            feature_name = None,
            feature_type = None,
            nthread = None)
```

```python
xgb.Booster(params = None, 
            cache = (),
            model_file = None)
```


## Learning API

```python
xgb.train(params, 
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
            learning_rate = None)
```

```python
xgb.cv(params, 
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
        shuffle = True)
```


## Scikit-Learn API

```python
xgb.XGBRegressor(max_depth = 3,
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
                **kwargs)

xgbr.fit(X, y,
        sample_weight = None,
        eval_set = None,
        eval_metric = None, 
        early_stopping_rounds = None,
        verbose = True,
        xgb_model = None,
        sample_weight_eval_set = None,
        callbacks = None)
```

```python
xgb.XGBClassifier(max_depth = 3,
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
                    **kwargs)
xgbc.fit(X, y,
        sample_weight = None,
        eval_set = None,
        eval_metric = None,
        early_stopping_rounds = None,
        verbose = True,
        xgb_model = None,
        sample_weight_eval_set = None,
        callbacks = None)
```


## Plotting API

```python
xgb.plot_importance(booster,
                    ax = None,
                    height = 0.2, 
                    xlim = None,
                    ylim = None,
                    title = "Feature importance",
                    xlabel = "F score",
                    ylabel = "Features",
                    importance_type = "weight")
```


## Callback API



## Dask API



# XGBoost调参

## 参数类型

- 通用参数
   - 控制整个模型的通用性能；
   - `booster` : 基本学习器类型
      - `gbtree` : 基于树的模型
      - `gblinear` : 线性模型
   - `silent`
      - 0: 打印训练过程中的信息
      - 1: 不会打印训练过程中的信息
   - `nthread` : 模型并行化使用系统的核数
- Booster参数
   - 控制每步迭代中每个基学习器(树/线性模型)；
   - `eta` : learning rate
      - 0.3
      - 0.01 ` 0.2
      - 通过shrinking每步迭代的中基本学习器的权重, 使模型更加稳健
   - `min_child_weight`
      - 子节点中所有样本的权重和的最小值
      - 用于控制过拟合
   - `max_depth`
      - 树的最大深度
      - 用于控制过拟合
      - 3 ` 10
   - `max_leaf_nodes`
      - 树中叶节点的最大数量
   - `gamma`
      - 当分裂结果使得损失函数减少时, 才会进行分裂, 指定了节点进行分裂所需的最小损失函数量；
   - `max_delta_step`
      - 每棵树的权重估计, 不常用
   - `subsample`
      - 定义了构建每棵树使用的样本数量, 比较低的值会使模型保守, 防止过拟合；太低的值会导致模型欠拟合；
      - 0.5 ` 1
   - `colsample_bytree`
      - 类似于GBM中的 `max_features` , 定义了用来构建每棵树时使用的特征数
      - 0.5 ` 1
   - `colsample_bylevel`
      - 定义了树每次分裂时的特征比例, 不常用, 用 `colsample_bytree`
   - `lambda`
      - 叶节点权重得分的l2正则参数
      - 不常使用, 一般用来控制过拟合；
   - `alpha`
      - 叶节点权重得分的l1正则参数
      - 适用于高维数据中, 能使得算法加速
   - `scale_pos_weight`
      - 用在高度不平衡数据中, 能够使算法快速收敛；
- 学习任务参数
   - 控制模型优化的表现；
   - `objective`
      - `binary:logistic` : Logistic Regression(二分类), 返回分类概率
      - `multi:softmax` : 利用 `softmax` 目标函数的多分类, 返回分类标签
         - 需要设置 `num_class`
      - `multi:softprob` : 类似于 `multi:softmax` , 返回概率值
   - `eval_metric`
      - `rmse` : 平方根误差
      - `mae` : 绝对平均误差
      - `logloss` : 负对数似然
      - `error` : 二分类错误率
      - `merror` : 多分类错误率
      - `mlogloss` : 多分类负对数似然
      - `auc` :  Area Under the Curve

## 参数调优的一般策略

1. 首先, 选择一个相对较大的 `learning rate` , 比如:0.1 (一般分为在: 0.05-0.3). 根据这个选定的 `learning rate` 对树的数量 `number of tree` 进行CV调优；
2. 调节树参数:  `max_depth`, `min_child_weight`, `gamma`, `subsample`, `colsample_bytree` ；
3. 调节正则化参数 `lambda`, `alpha` ；
4. 减小 `learning rate` , 并且优化其他参数；

## 参数调优步骤

- 选择一个初始化的 `learning_rate` 和 `n_estimators` , 利用CV对 `n_estimators` 进行调优, 选择一个最优的 `n_estimators` ；
   - 对其他模型参数进行初始化(选择一个合理的值): 
      - `max_depth = 5`
         - 3 - 10
      - `min_child_weight = 1`
         - 类别不平衡数据选择一个较小值
      - `gamma = 0`
         - 0.1 - 0.2
      - `subsample = 0.8`
         - 0.5 - 0.9
      - `colsample_bytree = 0.8`
         - 0.5 - 0.9
      - `scale_pos_weight = 1`
         - 类别不平衡数据喜讯则一个较小值
- 调节对模型结果影响最大的参数
   - `max_depth`
   - `min_child_weight`


```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.pyplot import rcParams
rcParams['figure.figsize'] = 12, 4

import xgboost as xgb
from xgboost.sklearn import XGBClassifier
from sklearn.model_selection import cross_val_score
from sklearn import metrics
from sklearn.model_selection import GridSearchCV


# ==========================================
# data
# ==========================================
def data(train_path, test_path):
    train = pd.read_csv(train_path)
    test = pd.read_csv(test_path)
    target = "Disbursed"
    IDcol = 'ID'
    predictors = [x for x in train.columns if x not in [target, IDcol]]

    return train, test, predictors, target


# ==========================================
# XGBoost model and cross-validation
# ==========================================
def modelFit(alg, dtrain, predictors, target,
            scoring = 'auc', useTrainCV = True, cv_folds = 5, early_stopping_rounds = 50):
    if useTrainCV:
        xgb_param = alg.get_xgb_params()
        xgb_train = xgb.DMatrix(data = dtrain[predictors].values, label = dtrain[target].values)
        cv_result = xgb.cv(params = xgb_param,
                            dtrain = xgb_train,
                            num_boost_round = alg.get_params()['n_estimators'],
                            nfold = cv_folds,
                            stratified = False,
                            metrics = scoring,
                            early_stopping_rounds = early_stopping_rounds,
                            show_stdv = False)
        alg.set_params(n_estimators = cv_result.shape[0])

    alg.fit(dtrain[predictors], dtrain[target], eval_metric = scoring)
    dtrain_predictions = alg.predict(dtrain[predictors])
    dtrain_predprob = alg.predict_proba(dtrain[predictors])[:, 1]

    print("\nModel Report:")
    print("Accuracy: %.4f" % metrics.accuracy_score(dtrain[target].values, dtrain_predictions))
    print("AUC Score (Train): %f" % metrics.roc_auc_score(dtrain[target], dtrain_predprob))

    feat_imp = pd.Series(alg.booster().get_fscore()).sort_values(ascending = False)
    feat_imp.plot(kind = 'bar', title = "Feature Importances")
    plt.ylabel("Feature Importance Score")


# ==========================================
# parameter tuning
# ==========================================
def grid_search(train, predictors, target, param_xgb, param_grid, scoring, n_jobs, cv_method):
    grid_search = GridSearchCV(estimator = XGBClassifier(**param_xgb),
                                param_grid = param_grid,
                                scoring = scoring,
                                n_jobs = n_jobs,
                                iid = False,
                                cv = cv_method)
    grid_search.fit(train[predictors], train[target])
    print(grid_search.cv_results_)
    print(grid_search.best_params_)
    print(grid_search.best_score_)

    return grid_search


# -----------------------------------
# data
# -----------------------------------
train_path = "./data/GBM_XGBoost_data/Train_nyOWmfK.csv"
test_path = "./data/GBM_XGBoost_data/Test_bCtAN1w.csv"
train, test, predictors, target = data(train_path, test_path)


# -----------------------------------
# XGBoost 基于默认的learning rate 调节树的数量
# n_estimators
# -----------------------------------
param_xgb1 = {
    'learning_rate': 0.1,
    'n_estimators': 1000,
    'max_depth': 5,
    'min_child_weight': 1,
    'gamma': 0,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'objective': 'binary:logistic',
    'nthread': 4,
    'scale_pos_weight': 1,
    'seed': 27
}
scoring = "auc"
cv_method = 5
early_stopping_rounds = 50
xgb_model1 = XGBClassifier(**param_xgb1)

modelFit(alg = xgb_model1,
        dtrain = train,
        predictors = predictors,
        target = target,
        scoring = scoring,
        useTrainCV = True,
        cv_folds = cv_method,
        early_stopping_rounds = early_stopping_rounds)

# -----------------------------------
# 调节基于树的模型
# max_depth, min_child_weight
# -----------------------------------
param_xgb_tree1 = {
    'learning_rate': 0.1,
    'n_estimators': 140,
    'max_depth': 5,
    'min_child_weight': 1,
    'gamma': 0,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'objective': "binary:logistic",
    'nthread': 4,
    'scale_pos_weight': 1,
    'seed': 27
}
param_grid_tree1 = {
    'max_depth': range(3, 10, 2),
    'min_child_weight': range(1, 6, 2)
}
scoring = 'roc_auc'
n_jobs = 4
cv_method = 5

grid_search(train = train,
            predictors = predictors,
            target = target,
            param_xgb = param_xgb_tree1,
            param_grid = param_grid_tree1,
            scoring = scoring,
            n_jobs = n_jobs,
            cv_method = cv_method)


# -----------------------------------
param_xgb_tree2 = {
    'learning_rate': 0.1,
    'n_estimators': 140,
    'max_depth': 5,
    'min_child_weight': 2,
    'gamma': 0,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'objective': "binary:logistic",
    'nthread': 4,
    'scale_pos_weight': 1,
    'seed': 27
}
param_grid_tree2 = {
    'max_depth': [4, 5, 6],
    'min_child_weight': [4, 5, 6]
}
scoring = 'roc_auc'
n_jobs = 4
cv_method = 5

grid_search(train = train,
            predictors = predictors,
            target = target,
            param_xgb = param_xgb_tree2,
            param_grid = param_grid_tree2,
            scoring = scoring,
            n_jobs = n_jobs,
            cv_method = cv_method)

# -----------------------------------
param_xgb_tree3 = {
    'learning_rate': 0.1,
    'n_estimators': 140,
    'max_depth': 4,
    'min_child_weight': 2,
    'gamma': 0,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'objective': "binary:logistic",
    'nthread': 4,
    'scale_pos_weight': 1,
    'seed': 27
}
param_grid_tree3 = {
    'min_child_weight': [6, 8, 10, 12]
}
scoring = 'roc_auc'
n_jobs = 4
cv_method = 5

grid_xgb_tree1 = grid_search(train = train,
                            predictors = predictors,
                            target = target,
                            param_xgb = param_xgb_tree3,
                            param_grid = param_grid_tree3,
                            scoring = scoring,
                            n_jobs = n_jobs,
                            cv_method = cv_method)


scoring = "auc"
cv_method = 5
early_stopping_rounds = 50

modelFit(alg = grid_xgb_tree1.best_estimator_,
        dtrain = train,
        predictors = predictors,
        target = target,
        scoring = scoring,
        useTrainCV = True,
        cv_folds = cv_method,
        early_stopping_rounds = early_stopping_rounds)


# -----------------------------------
# 调节基于树的模型
# gamma
# -----------------------------------
param_xgb_tree4 = {
    'learning_rate': 0.1,
    'n_estimators': 140,
    'max_depth': 4,
    'min_child_weight': 6,
    'gamma': 0,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'objective': "binary:logistic",
    'nthread': 4,
    'scale_pos_weight': 1,
    'seed': 27
}
param_grid_tree4 = {
    'gamma': [i/10.0 for i in range(0, 5)]
}
scoring = 'roc_auc'
n_jobs = 4
cv_method = 5

grid_search(train = train,
            predictors = predictors,
            target = target,
            param_xgb = param_xgb_tree4,
            param_grid = param_grid_tree4,
            scoring = scoring,
            n_jobs = n_jobs,
            cv_method = cv_method)

# -----------------------------------
param_xgb2 = {
    'learning_rate': 0.1,
    'n_estimators': 1000,
    'max_depth': 4,
    'min_child_weight': 6,
    'gamma': 0,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'objective': 'binary:logistic',
    'nthread': 4,
    'scale_pos_weight': 1,
    'seed': 27
}
scoring = "auc"
cv_method = 5
early_stopping_rounds = 50
xgb_model2 = XGBClassifier(**param_xgb2)

modelFit(alg = xgb_model2,
        dtrain = train,
        predictors = predictors,
        target = target,
        scoring = scoring,
        useTrainCV = True,
        cv_folds = cv_method,
        early_stopping_rounds = early_stopping_rounds)




# -----------------------------------
# 调节基于树的模型
# subsample, colsample_bytree
# -----------------------------------
param_xgb_tree5 = {
    'learning_rate': 0.1,
    'n_estimators': 140,
    'max_depth': 4,
    'min_child_weight': 6,
    'gamma': 0,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'objective': "binary:logistic",
    'nthread': 4,
    'scale_pos_weight': 1,
    'seed': 27
}
param_grid_tree5 = {
    'subsample': [i/10.0 for i in range(6, 10)],
    'colsample_bytree': [i/10.0 for i in range(6, 10)]
}
scoring = 'roc_auc'
n_jobs = 4
cv_method = 5

grid_search(train = train,
            predictors = predictors,
            target = target,
            param_xgb = param_xgb_tree5,
            param_grid = param_grid_tree5,
            scoring = scoring,
            n_jobs = n_jobs,
            cv_method = cv_method)

# -----------------------------------
param_xgb_tree6 = {
    'learning_rate': 0.1,
    'n_estimators': 140,
    'max_depth': 4,
    'min_child_weight': 6,
    'gamma': 0,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'objective': "binary:logistic",
    'nthread': 4,
    'scale_pos_weight': 1,
    'seed': 27
}
param_grid_tree6 = {
    'subsample': [i/100.0 for i in range(75, 90, 5)],
    'colsample_bytree': [i/10.0 for i in range(75, 90, 5)]
}
scoring = 'roc_auc'
n_jobs = 4
cv_method = 5

grid_search(train = train,
            predictors = predictors,
            target = target,
            param_xgb = param_xgb_tree6,
            param_grid = param_grid_tree6,
            scoring = scoring,
            n_jobs = n_jobs,
            cv_method = cv_method)


# -----------------------------------
# 调节正则化参数
# reg_alpha, reg_lambda
# -----------------------------------
param_xgb_regu1 = {
    'learning_rate': 0.1,
    'n_estimators': 140,
    'max_depth': 4,
    'min_child_weight': 6,
    'gamma': 0,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'objective': "binary:logistic",
    'nthread': 4,
    'scale_pos_weight': 1,
    'seed': 27
}
param_grid_regu1 = {
    'reg_alpha': [1e-5, 1e-2, 0.1, 1, 100]
}
scoring = 'roc_auc'
n_jobs = 4
cv_method = 5

grid_search(train = train,
            predictors = predictors,
            target = target,
            param_xgb = param_xgb_regu1,
            param_grid = param_grid_regu1,
            scoring = scoring,
            n_jobs = n_jobs,
            cv_method = cv_method)

param_xgb_regu2 = {
    'learning_rate': 0.1,
    'n_estimators': 140,
    'max_depth': 4,
    'min_child_weight': 6,
    'gamma': 0,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'objective': "binary:logistic",
    'nthread': 4,
    'scale_pos_weight': 1,
    'seed': 27
}
param_grid_regu2 = {
    'reg_alpha': [0, 0.001, 0.005, 0.01, 0.05]
}
scoring = 'roc_auc'
n_jobs = 4
cv_method = 5

grid_search(train = train,
            predictors = predictors,
            target = target,
            param_xgb = param_xgb_regu2,
            param_grid = param_grid_regu2,
            scoring = scoring,
            n_jobs = n_jobs,
            cv_method = cv_method)



# -----------------------------------
param_xgb3 = {
    'learning_rate': 0.1,
    'n_estimators': 1000,
    'max_depth': 4,
    'min_child_weight': 6,
    'gamma': 0,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'reg_alpha': 0.005,
    'objective': 'binary:logistic',
    'nthread': 4,
    'scale_pos_weight': 1,
    'seed': 27
}
scoring = "auc"
cv_method = 5
early_stopping_rounds = 50
xgb_model3 = XGBClassifier(**param_xgb3)

modelFit(alg = xgb_model3,
        dtrain = train,
        predictors = predictors,
        target = target,
        scoring = scoring,
        useTrainCV = True,
        cv_folds = cv_method,
        early_stopping_rounds = early_stopping_rounds)



# -----------------------------------
# 降低learning rate
# 增加n_estimators
# -----------------------------------
param_xgb4 = {
    'learning_rate': 0.1,
    'n_estimators': 5000,
    'max_depth': 4,
    'min_child_weight': 6,
    'gamma': 0,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'reg_alpha': 0.005,
    'objective': 'binary:logistic',
    'nthread': 4,
    'scale_pos_weight': 1,
    'seed': 27
}
scoring = "auc"
cv_method = 5
early_stopping_rounds = 50
xgb_model4 = XGBClassifier(**param_xgb4)

modelFit(alg = xgb_model4,
        dtrain = train,
        predictors = predictors,
        target = target,
        scoring = scoring,
        useTrainCV = True,
        cv_folds = cv_method,
        early_stopping_rounds = early_stopping_rounds)
```

