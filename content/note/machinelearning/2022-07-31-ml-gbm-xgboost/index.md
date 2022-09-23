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

- [XGBoost 资源](#xgboost-资源)
- [XGBoost 简介](#xgboost-简介)
  - [XGBoost 特点](#xgboost-特点)
  - [XGBoost 优势](#xgboost-优势)
  - [XGBoost 缺点](#xgboost-缺点)
- [XGBoost 模型理论](#xgboost-模型理论)
  - [模型目标函数](#模型目标函数)
  - [模型目标](#模型目标)
  - [模型损失函数](#模型损失函数)
  - [模型正则项](#模型正则项)
  - [模型目标函数求解](#模型目标函数求解)
  - [树节点分裂策略](#树节点分裂策略)
  - [模型中决策树节点分裂算法](#模型中决策树节点分裂算法)
    - [贪心算法(Exact Greedy Algorithm for split finding)](#贪心算法exact-greedy-algorithm-for-split-finding)
    - [近似算法(Approximate Algorithm for split finding)](#近似算法approximate-algorithm-for-split-finding)
  - [自定义损失函数](#自定义损失函数)
- [XGBoost 使用](#xgboost-使用)
  - [下载安装 xgboost 库](#下载安装-xgboost-库)
    - [Ubuntu/Debian](#ubuntudebian)
    - [OSX](#osx)
    - [Windows](#windows)
  - [导入 xgboost 库](#导入-xgboost-库)
  - [数据接口](#数据接口)
  - [参数设置](#参数设置)
    - [通用参数](#通用参数)
  - [任务参数](#任务参数)
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

# XGBoost 资源

- [原始算法论文]()
- [GitHub-Python-Package]()
- [GitHub-R-Package]()
- [GitHub-Microsoft]()
- [Doc]()
- [Python 示例]()

# XGBoost 简介

## XGBoost 特点

弱学习器:

* 传统的 GBDT 以 CART 作为基函数(base learner, base classifier, base function), 
  而 XGBoost 除了可以使用 CART, 还支持线性分类器 (linear classifier, linear regression, logistic regression)
* 基于预排序(pre-sorted)方法构建决策树
    - 算法基本原理
        - 首先, 对所有特征都按照特征的数值进行排序
        - 其次, 在遍历分割点的时候用 `$O(\#data)$` 的代价找到一个特征上的最好分割点
        - 最后, 找到一个特征的分割点后, 将数据分裂成左右子节点
    - 优缺点
        - 优点: 精确地找到分割点
        - 缺点: 内存消耗大, 时间消耗大, 对 cache 优化不友好

梯度:

* 传统的 GBDT 在优化时只用到一阶导数信息(负梯度), 
  XGBoost 则对损失函数进行了二阶泰勒展开, 同时用到一阶和二阶导数. 
* 且 XGBoost 支持自定义损失函数, 只要函数可一阶和二阶求导

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
- 并行和分布式计算
- 设计高效核外计算, 进行 cache-aware 数据块处理

## XGBoost 缺点

> XGBoost 的缺点也是LightGBM 的出发点

每轮迭代时, 都需要遍历整个训练数据集多次:

* 如果把整个训练数据装进内存则会限制训练数据的大小
* 如果不装进内存, 反复地读写训练数据又会消耗非常大的时间

预排序方法(pre-sorted): 

* 首先, 空间消耗大. 这样的算法需要保存数据的特征值, 
  还保存了特征排序的结果(例如排序后的索引, 为了后续快速地计算分割点), 
  这里需要消耗训练数据两倍的内存 
* 其次时间上也有较大的开销, 在遍历每一个分割点的时候, 都需要进行分裂增益的计算, 消耗的代价大

对 cache 优化不友好:

* 在预排序后, 特征对梯度的访问是一种随机访问, 并且不同的特征访问的顺序不一样, 
  无法对 cache 进行优化. 同时, 在每一层长树的时候, 需要随机访问一个行索引到叶子索引的数组, 
  并且不同特征访问的顺序也不一样, 也会造成较大的 cache miss

# XGBoost 模型理论

## 模型目标函数

`$$L(\phi)=\sum_{i=1}^{n} l(y_i, \hat{y_i}) + \sum_{k=1}^{K} \Omega(f_k)$$`

其中: 

- `$L(\cdot)$` : 目标函数
- `$l(\cdot)$` : 经验损失(误差)函数, 通常是凸函数. 
  用于刻画预测值 `$\hat{y_i}$` 和真实值 `$y_i$` 之间的差异, 即模型对训练数据的拟合程度
- `$\Omega(\cdot)$` : 模型的正则化项. 用于降低模型的复杂度, 减轻过拟合
   - 决策树的叶子节点数量
   - 决策树的树深度
   - 决策树的叶节点权重得分的 L1, L2 正则
- `$y_i$` : 第 `$i$` 个样本的目标变量，即样本的真实观测值
- `$\hat{y_i}=\sum_{k=1}^{K}f_k(x_i), f_k \in F$`, 第 `$i$` 个样本的模型输出预测值:
   - 回归: 预测得分
   - 分类: 预测概率
   - 排序: 排序得分
- `$f_k$` : 第 `$k$` 棵树
- `$n$`: 样本数量
- `$K$`: 树的数量

## 模型目标

找到一组树，使得 `$L(\phi)$` 最小:

`$$\min L(\phi) = \min \bigg[\sum_{i=1}^{n} l(y_i, \hat{y_i}) + \sum_{k=1}^{K} \Omega(f_k)\bigg]$$`

这个模型目标函数是由经验(样本)损失和模型复杂度惩罚项(正则项)组成

## 模型损失函数

Additive Training Boosting 核心思想是，在已经训练好了 `$t-1$` 棵树后不再调整前 `$t-1$` 棵树，
即第 `$t$` 轮迭代的目标函数不对前 `$t-1$` 轮迭代的结果进行修改，
那么根据 GBM(Gradient Boosting Modeling) 思想, 假设第 `$t$` 棵树可以表示为: 

`$$\hat{y}_{i}^{(t)}=\sum_{k=1}^{t}f_{k}(x_{i})=\hat{y}_{i}^{(t-1)}+f_{t}(x_{i})$$`

那么，可以假设第 `$t$` 轮迭代的目标函数为: 

`$$\eqalign{
& {\tilde L}^{(t)} 
= \sum_{i=1}^{n} l(y_i, \hat{y_i}) + \sum_{k=1}^{K} \Omega(f_k) \cr
& \;\;\;\;\;{\rm{ 
= }} \sum_{i=1}^{n} l \bigg(y_i, \hat{y_i}^{(t-1)} + f_t(x_i) \bigg) + \Omega(f_t) + \sum_{i=1}^{t-1}\Omega(f_{i}) \cr
} $$`

其中:

* `$\sum_{i=1}^{t-1}\Omega(f_{i})$`: 这部分在前 `$t-1$` 棵树已知的情况下为常数

对目标函数 `$L^{(t)}$` 在 `$\hat{y_i}^{(t-1)}$`
处进行二阶泰勒展开, 可以加速优化过程, 得到目标函数的近似: 

`$$L^{(t)} \simeq \sum_{i=1}^{n} \bigg[l(y_i, \hat{y_i}^{(t-1)}) + g_i f_t (x_i) + \frac{1}{2} h_i f_t^2 (x_i)\bigg]+ \Omega(f_t) + constant$$`

具体推导过程如下:

> > 函数 `$f(x)$` 在 `$x_0$` 处的二阶泰勒展开式: 
> > 
> > `$$f(x) = f(x_0) + f' (x_0)(x - x_0) + f''(x_0)(x - x_0)^2$$`
> 
> 目标函数 `$l(y_i, x)$` 在 `$\hat{y_i}^{(t-1)}$` 处的二阶泰勒展开式: 
> 
> `$$l(y_i, x) = l(y_i, \hat{y_i}^{(t - 1)}) + \frac{\partial l(y_i, \hat{y_i}^{(t - 1)})}{\partial \hat{y_i}^{(t - 1)}} (x - \hat{y_i}^{(t - 1)}) + \frac{1}{2} \frac{{\partial ^{2}} l(y_i, \hat{y_i}^{(t - 1)}) } {\partial \hat{y_i}^{(t - 1)}} (x - \hat{y_i}^{(t - 1)})^2$$`
> 
> 令 `$x= \hat{y_i}^{(t-1)} + f_t (x_i)$`, 
> 
> 记一阶导数为
> 
> `$$g_i = \frac{\partial l(y_i, \hat{y_i}^{(t - 1)})}{\partial \hat{y_i}^{(t - 1)}}$$`
> 
> 记二阶导数为
> 
> `$$h_i = \frac{{\partial ^{2}} l(y_i, \hat{y_i}^{(t - 1)})}{\partial \hat{y_i}^{(t - 1)}}$$` 
> 
> 可以得到
> 
> `$$l(y_i, \hat{y}^{(t-1)} + f_t(x_i)) = l(y_i, \hat{y}^{(t - 1)}) + g_i f_t(x_i) + \frac{1}{2} h_i f_t^2 (x_i)$$`

上面的目标函数 `$L^{(t)}$` 中, 由于前 `$t-1$` 棵树已知，则第一项 `$l(y_i, \hat{y_i}^{(t-1)})$` 也为常数项，
以及 `$constant$` 为常数项, 在优化问题中可直接删除，因此模型损失函数为:

`$${\tilde L}^{(t)} = \sum_{i=1}^{n} \bigg[g_{i}f_{t}(x_i) + \frac{1}{2}h_{i}f_{t}^{2}(x_i) \bigg]  + \Omega (f_t)$$`

## 模型正则项

假设待训练的第 `$t$` 棵树有 `$T$` 个叶子节点，叶子节点的输出向量(叶子节点权重得分向量)表示为:

`$$[\omega_{1}, \omega_{2}, \ldots, \omega_{T}]$$`

假设样本 `$x \in R^{d}$` 到叶子节点的索引值 `$\{1, 2, \ldots, T\}$` 的映射表示为:

`$$q(x): R^d \rightarrow \{1, 2, \ldots, T\}$$`

通过叶子节点的权重得分和叶子节点的索引值(树结构)定义第 `$t$` 棵树:

`$$f_t(x) = \omega_{q(x)}, \omega \in R^{T}, q(x): R^d \rightarrow \{1, 2, \ldots, T\}$$`

其中: 

- `$\omega$`: 叶子节点权重得分向量
- `$q(\cdot)$`: 叶子节点的索引值，树结构
- `$T$`: 叶节点数量
- `$x$`: 样本向量
- `$d$`: 样本特征数量
- `$t$`: 模型的迭代轮数，构建的第 `$t$` 棵树

定义正则项(可以使其他形式): 

`$$\Omega(f_t)=\gamma T + \frac{1}{2}\lambda\sum_{j=1}^{T}\omega_j^2$$`

其中: 

- `$T$`: 叶节点数量
- `$\omega_j^2$`: 叶子节点权重得分向量的 L2 范数
- `$\gamma$`: 叶节点数量的正则化参数 
- `$\lambda$`: 叶子节点权重得分向量的正则化参数

记决策树每个叶节点上的样本集合为: 

`$${I_j} = \{ {i | q(x_i) = j} \}$$`

将正则项 `$\Omega(f_t)$` 展开: 

`$$\eqalign{
& {\tilde L}^{(t)} 
= \sum_{i=1}^{n} \bigg[g_i f_t(x_i) + \frac{1}{2}h_i f_{t}^{2}(X_I)\bigg] + \Omega(f_t) \cr 
& \;\;\;\;\;{\rm{ 
= }} \sum_{i=1}^{n} \bigg[g_i \omega_{q(x_i)} + \frac{1}{2} h_i \omega_{q(x_i)}^{2} \bigg] + \gamma T + \frac{1}{2} \lambda \sum_{j = 1}^{T} w_j^2 \cr 
& \;\;\;\;\;{\rm{ 
= }} \sum_{j = 1}^{T} \bigg[(\sum_{i \in I_j} g_i ) w_j + \frac{1}{2} (\sum_{i \in {I_j}} h_i  + \lambda ) w_j^2 \bigg]  + \gamma T \cr
}$$`

记 `$g_{j}=\sum_{i\in I_j}g_i$` ,  `$h_{j}=\sum_{i\in I_j}h_i$`

`$$\eqalign{
& {\tilde L}^{(t)} 
= \sum_{j = 1}^{T} \bigg[(\sum_{i \in {I_j}} g_i) w_j + \frac{1}{2} (\sum_{i \in {I_j}} h_i  + \lambda ) w_j^2 \bigg]  + \gamma T \cr
& \;\;\;\;\;{\rm{ 
= }} \sum_{j = 1}^{T} \bigg[g_{j} w_j + \frac{1}{2} (h_{j}  + \lambda ) w_j^2 \bigg]  + \gamma T \cr
}$$`

## 模型目标函数求解

对下面的目标函数进行优化求解:

`$$\underset{[\omega_{1}, \omega_{2}, \ldots, \omega_{T}]}{argmin} {\tilde L}^{(t)} = \underset{[\omega_{1}, \omega_{2}, \ldots, \omega_{T}]}{argmin} \bigg( \sum_{j = 1}^{T} \bigg[g_{j} w_j + \frac{1}{2} (h_{j}  + \lambda ) w_j^2 \bigg]  + \gamma T \bigg)$$`

易知，上述目标函数是一个累加的二次函数形式 `$f(x)=ax + bx^{2} +c$`，
因此根据二次函数最小值的解析解形式 `$x=-\frac{b}{2a}$`，
对于固定的树结构 `$q(x)$`, 对目标函数关于 `$\omega_j$` 求导等于 0, 
得到目标函数的解析解 `$\omega_{j}^{\star}$`:

`$$w_{j}^{\star} = \frac{\sum\limits_{i \in I_j} g_i}{\sum\limits_{i \in I_j} h_i + \lambda}$$`

即目标函数的最优参数:

`$$w_{j}^{\star} = -\frac{g_{j}}{h_{j}+\lambda}$$`

将上面得到的解析解带入目标函数: 

`$$\tilde{L}^{(t)}_{min}=-\frac{1}{2}\sum_{j=1}^{T}\frac{g_{j}^2}{h_{j}+\lambda} + \gamma T$$`

上式可以作为分裂节点的打分，形式上很像 CART 树纯度打分的计算，区别在于它是从目标函数中推导而得

这里的 `$\tilde{L}^{(t)}_{min}$` 代表了当指定一个树结构时, 
在目标函数上最多减少多少, 这里叫做 **结构分数(structure score)**，
这个分数越小, 代表这个树的结构越好

## 树节点分裂策略

实践中，很难去穷举每一棵树进行打分，再选出最好的。通常采用贪心的方式，逐层选择最佳的分裂节点。
假设 `$I_{L}$` 和 `$I_{R}$` 为分裂节点的左右节点，记 `$I=I_{L} \cup I_{R}$`。
每次尝试对已有叶节点进行一次分割, 分割的规则如下，即选择此节点分裂的增益为：

`$$Gain = \frac{1}{2}\Bigg[\frac{G_{L}^{2}}{H_K + \lambda} + \frac{G_{R}^{2}}{H_R + \lambda} - \frac{(G_L + G_R)^{2}}{H_L + H_R + \lambda}\Bigg] - \gamma$$`

其中: 

- `$\frac{G_{L}^{2}}{H_K + \lambda}$`: 左子树分数
- `$\frac{G_{R}^{2}}{H_R + \lambda}$`: 右子树分数
- `$\frac{(G_L + G_R)^{2}}{H_L + H_R + \lambda}$`: 不分割可以得到的分数
- `$\gamma$`: 假如新的子节点引入的复杂度代价

对树的每次扩展, 都要枚举所有可能的分割方案，并且对于某次分割, 
都要计算每个特征值左边和右边的一阶和二阶导数和, 从而计算这次分割的增益 `$Gain$`

对于上面的分割增益 `$Gain$`, 要判断分割每次分割对应的 `$Gain$` 的大小, 
并且进行优化, 取最小的 `$Gain$`, 直到当新引入一个分割带来的增益小于某个阈值时, 
就去掉这个分割. 这里的优化, 相当于对树进行剪枝

## 模型中决策树节点分裂算法

之前假设是已知前 `$t-1$` 棵树，因此现在来探讨怎么生成树。
根据决策树的生成策略，在每次分裂节点的时候需要考虑能使得损失函数减小最快的节点，
也就是分裂后损失函数的结构分数减去分裂前损失函数的结构分数，称之为增益(Gain)。
Gain 越大，越能说明分裂后目标函数值减少越多

### 贪心算法(Exact Greedy Algorithm for split finding)

在决策树(CART)里面，使用的是精确贪心算法(Basic Exact Greedy Algorithm), 
也就是将所有特征的所有取值排序(耗时耗内存巨大)，遍历所有特征中可能的分裂点位置，
根据节点分裂的增益公式计算增益，然后比较每一个点的 Gini 指数，找出变化最大的节点。

当特征是连续特征时，对连续值离散化，取两点的平均值为分割节点。
可以看到，这里的排序算法需要花费大量的时间，因为要遍历整个样本所有特征，而且还要排序

XGBoost 使用贪心算法:

> `$
  \eqalign{
    & Split\_Finding(): \cr
    & Input: I, instance \; set \; of \; current \; node  \cr 
    & Input: d, feature \; dimension  \cr 
    & gain \leftarrow 0  \cr 
    & G \leftarrow \sum \limits_{i \in I} g_{i}, H \leftarrow \sum \limits_{i \in I} h_{i}   \cr 
    & for \; k = 1 \; to \; m \; do  \cr 
    & \;\;\;\; G_{L} \leftarrow 0, H_{L} \leftarrow 0  \cr 
    & \;\;\;\;for\;j\;in\;sorted \left( {I, by \; {{\bf{x}}_{jk}}} \right)\;do  \cr 
    & \;\;\;\;\;\;\;\;G_{L} \leftarrow G_{L} + {g_{j}},H_{L} \leftarrow H_{L} + {h_{j}}  \cr 
    & \;\;\;\;\;\;\;\;{G_R} \leftarrow G - G_{L},H_{R} \leftarrow H - H_{L}  \cr 
    & \;\;\;\;\;\;\;\;score \leftarrow \max \left( {score,{{G_{L}^{2}} \over {H_{L} + \lambda }} + {G_{R}^{2} \over {H_{R} + \lambda }} - {{{G^2}} \over {H + \lambda }}} \right)  \cr 
    & \;\;\;\;end  \cr 
    & end \cr
    & Output:\; split\_value\; with\; max \; score}
  $`


### 近似算法(Approximate Algorithm for split finding)

如果数据不能一次读入内存，使用贪心算法效率较低。
在 XGBoost 里面使用的是近似算法(Approximate Algorithm)。

该算法首先根据特征分布的百分位数(percentiles)提出候选分裂点，
将连续特征映射到由这些候选点分割的桶中，
汇总统计信息并根据汇总的信息在提案中找到最佳解决方案。

对于某个特征 `$x_{k}$`，算法首先根据特征分布的分位数找到特征切割点的候选集合 `$S_{k}=\{S_{k_{1}}, S_{k_{2}}, \ldots, S_{k_{l}} \}$`,
然后将特征 `$x_{k}$` 的值根据集合 `$S_{k}$` 将样本进行分桶，
接着对每个桶内的样本统计值 `$G$`、`$H$` 进行累加，
记为分界点 `$v$` 的统计量，`$v$` 满足 `$\{{\bf x}_{kj}=s_{kv}\}$`。 
最后在分界点集合上调用上面的贪心算法进行贪心查找，得到的结果为最佳分裂点的近似。具体如下

> `$
  \eqalign{
  & for\;k = 1\;to\;m\;do  \cr 
  & \;\;\;\;Propose\;{S_k} = \left\{ {{s_{k1}},{s_{k2}},...,{s_{kl}}} \right\}\;by\;percentile\;on\;feature\;k  \cr 
  & \;\;\;\;Propose\;can\;be\;done\;per\;tree\left( {global} \right),or\;per\;split\left( {local} \right)  \cr 
  & end  \cr 
  & for\;k = 1\;to\;m\;do  \cr 
  & \;\;\;\;{G_{kv}} \leftarrow  = \sum\limits_{j \in \left\{ {j|{s_{k,v}} \ge {x_{jk}} > {s_{k,v - 1}}} \right\}} {{g_{j}}}   \cr 
  & \;\;\;\;{H_{kv}} \leftarrow  = \sum\limits_{j \in \left\{ {j|{s_{k,v}} \ge {x_{jk}} > {s_{k,v - 1}}} \right\}} {{h_{j}}}   \cr 
  & end  \cr 
  & call\;Split\_Finding\left( {} \right) \cr}
  $`

## 自定义损失函数

* TODO

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

### 通用参数

* booster:使用哪个弱学习器训练，默认gbtree，可选gbtree，gblinear 或dart
* nthread：用于运行XGBoost的并行线程数，默认为最大可用线程数
* verbosity：打印消息的详细程度。有效值为0（静默），1（警告），2（信息），3（调试）。
* Tree Booster的参数：
    - eta（learning_rate）：learning_rate，在更新中使用步长收缩以防止过度拟合，默认= 0.3，范围：[0,1]；典型值一般设置为：0.01-0.2
    - gamma（min_split_loss）：默认= 0，分裂节点时，损失函数减小值只有大于等于gamma节点才分裂，gamma值越大，算法越保守，越不容易过拟合，但性能就不一定能保证，需要平衡。范围：[0，∞]
    - max_depth：默认= 6，一棵树的最大深度。增加此值将使模型更复杂，并且更可能过度拟合。范围：[0，∞]
    - min_child_weight：默认值= 1，如果新分裂的节点的样本权重和小于min_child_weight则停止分裂 。这个可以用来减少过拟合，但是也不能太高，会导致欠拟合。范围：[0，∞]
    - max_delta_step：默认= 0，允许每个叶子输出的最大增量步长。如果将该值设置为0，则表示没有约束。如果将其设置为正值，
      则可以帮助使更新步骤更加保守。通常不需要此参数，但是当类极度不平衡时，它可能有助于逻辑回归。将其设置为1-10的值可能有助于控制更新。范围：[0，∞]
    - subsample：默认值= 1，构建每棵树对样本的采样率，如果设置成0.5，XGBoost会随机选择一半的样本作为训练集。范围：（0,1]
    - sampling_method：默认= uniform，用于对训练实例进行采样的方法。
        - uniform：每个训练实例的选择概率均等。通常将subsample> = 0.5 设置 为良好的效果。
        - gradient_based：每个训练实例的选择概率与规则化的梯度绝对值成正比，具体来说就是 图片 ，subsample可以设置为低至0.1，而不会损失模型精度。
    - colsample_bytree：默认= 1，列采样率，也就是特征采样率。范围为（0，1]
    - lambda（reg_lambda）：默认=1，L2正则化权重项。增加此值将使模型更加保守。
    - alpha（reg_alpha）：默认= 0，权重的L1正则化项。增加此值将使模型更加保守。
    - tree_method：默认=auto，XGBoost中使用的树构建算法。
        - auto：使用启发式选择最快的方法。
            - 对于小型数据集，exact将使用精确贪婪（）。
            - 对于较大的数据集，approx将选择近似算法（）。它建议尝试hist，gpu_hist，用大量的数据可能更高的性能。（gpu_hist）支持。external memory外部存储器。
        - exact：精确的贪婪算法。枚举所有拆分的候选点。
        - approx：使用分位数和梯度直方图的近似贪婪算法。
        - hist：更快的直方图优化的近似贪婪算法。（LightGBM也是使用直方图算法）
        - gpu_hist：GPU hist算法的实现。
    - scale_pos_weight:控制正负权重的平衡，这对于不平衡的类别很有用。Kaggle竞赛一般设置sum(negative instances) / sum(positive instances)，
      在类别高度不平衡的情况下，将参数设置大于0，可以加快收敛。
    - num_parallel_tree：默认=1，每次迭代期间构造的并行树的数量。此选项用于支持增强型随机森林。
    - monotone_constraints：可变单调性的约束，在某些情况下，如果有非常强烈的先验信念认为真实的关系具有一定的质量，则可以使用约束条件来提高模型的预测性能。
     （例如params_constrained['monotone_constraints'] = "(1,-1)"，(1,-1)我们告诉XGBoost对第一个预测变量施加增加的约束，对第二个预测变量施加减小的约束。）
* Linear Booster的参数：
    - lambda（reg_lambda）：默认= 0，L2正则化权重项。增加此值将使模型更加保守。归一化为训练示例数。
    - alpha（reg_alpha）：默认= 0，权重的L1正则化项。增加此值将使模型更加保守。归一化为训练示例数。
    - updater：默认= shotgun。
        - shotgun：基于shotgun算法的平行坐标下降算法。使用“ hogwild”并行性，因此每次运行都产生不确定的解决方案。
        - coord_descent：普通坐标下降算法。同样是多线程的，但仍会产生确定性的解决方案。
    - feature_selector：默认= cyclic。特征选择和排序方法
        - cyclic：通过每次循环一个特征来实现的。
        - shuffle：类似于cyclic，但是在每次更新之前都有随机的特征变换。
        - random：一个随机(有放回)特征选择器。
       - greedy：选择梯度最大的特征。（贪婪选择）
       - thrifty：近似贪婪特征选择（近似于greedy）
    - top_k：要选择的最重要特征数（在greedy和thrifty内）


## 任务参数

- objective：默认=reg:squarederror，表示最小平方误差
    - reg:squarederror,最小平方误差
    - reg:squaredlogerror,对数平方损失
    - reg:logistic,逻辑回归
    - reg:pseudohubererror,使用伪Huber损失进行回归，这是绝对损失的两倍可微选择
    - binary:logistic,二元分类的逻辑回归，输出概率
    - binary:logitraw：用于二进制分类的逻辑回归，逻辑转换之前的输出得分
    - binary:hinge：二进制分类的铰链损失。这使预测为0或1，而不是产生概率。（SVM就是铰链损失函数）
    - count:poisson –计数数据的泊松回归，泊松分布的输出平均值
    - survival:cox：针对正确的生存时间数据进行Cox回归（负值被视为正确的生存时间）
    - survival:aft：用于检查生存时间数据的加速故障时间模型
    - aft_loss_distribution：survival:aft和aft-nloglik度量标准使用的概率密度函数
    - multi:softmax：设置XGBoost以使用softmax目标进行多类分类，还需要设置num_class（类数）
    - multi:softprob：与softmax相同，但输出向量，可以进一步重整为矩阵。结果包含属于每个类别的每个数据点的预测概率
    - rank:pairwise：使用LambdaMART进行成对排名，从而使成对损失最小化
    - rank:ndcg：使用LambdaMART进行列表式排名，使标准化折让累积收益（NDCG）最大化
    - rank:map：使用LambdaMART进行列表平均排名，使平均平均精度（MAP）最大化
    - reg:gamma：使用对数链接进行伽马回归。输出是伽马分布的平均值
    - reg:tweedie：使用对数链接进行Tweedie回归
    - 自定义损失函数和评价指标
- eval_metric：验证数据的评估指标，将根据目标分配默认指标（回归均方根，分类误差，排名的平均平均精度），用户可以添加多个评估指标
    - rmse，均方根误差；rmsle：均方根对数误差；mae：平均绝对误差；mphe：平均伪Huber错误；logloss：负对数似然；error：二进制分类错误率
    - merror：多类分类错误率；mlogloss：多类logloss；auc：曲线下面积；aucpr：PR曲线下的面积；ndcg：归一化累计折扣；map：平均精度
- seed ：随机数种子，[默认= 0]

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

参数调优的一般步骤：

1. 确定（较大）学习速率和提升参数调优的初始值
2. max_depth 和 min_child_weight 参数调优
3. gamma参数调优
4. subsample 和 colsample_bytree 参数优
5. 正则化参数alpha调优
6. 降低学习速率和使用更多的决策树

## 参数类型

- 通用参数
   - 控制整个模型的通用性能
   - `booster` : 基本学习器类型
      - `gbtree` : 基于树的模型
      - `gblinear` : 线性模型
   - `silent`
      - 0: 打印训练过程中的信息
      - 1: 不会打印训练过程中的信息
   - `nthread` : 模型并行化使用系统的核数
- Booster参数
   - 控制每步迭代中每个基学习器(树/线性模型)
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
      - 当分裂结果使得损失函数减少时, 才会进行分裂, 指定了节点进行分裂所需的最小损失函数量
   - `max_delta_step`
      - 每棵树的权重估计, 不常用
   - `subsample`
      - 定义了构建每棵树使用的样本数量, 比较低的值会使模型保守, 防止过拟合太低的值会导致模型欠拟合
      - 0.5 ` 1
   - `colsample_bytree`
      - 类似于GBM中的 `max_features` , 定义了用来构建每棵树时使用的特征数
      - 0.5 ` 1
   - `colsample_bylevel`
      - 定义了树每次分裂时的特征比例, 不常用, 用 `colsample_bytree`
   - `lambda`
      - 叶节点权重得分的l2正则参数
      - 不常使用, 一般用来控制过拟合
   - `alpha`
      - 叶节点权重得分的l1正则参数
      - 适用于高维数据中, 能使得算法加速
   - `scale_pos_weight`
      - 用在高度不平衡数据中, 能够使算法快速收敛
- 学习任务参数
   - 控制模型优化的表现
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

1. 首先, 选择一个相对较大的 `learning rate` , 比如:0.1 (一般分为在: 0.05-0.3). 根据这个选定的 `learning rate` 对树的数量 `number of tree` 进行CV调优
2. 调节树参数:  `max_depth`, `min_child_weight`, `gamma`, `subsample`, `colsample_bytree` 
3. 调节正则化参数 `lambda`, `alpha` 
4. 减小 `learning rate` , 并且优化其他参数

## 参数调优步骤

- 选择一个初始化的 `learning_rate` 和 `n_estimators` , 利用CV对 `n_estimators` 进行调优, 选择一个最优的 `n_estimators` 
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

