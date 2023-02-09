---
title: XGBoost
author: 王哲峰
date: '2022-08-02'
slug: ml-gbm-xgboost
categories:
  - machinelearning
tags:
  - machinelearning
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
- [XGBoost 参数](#xgboost-参数)
  - [通用参数](#通用参数)
  - [Tree Booster 参数](#tree-booster-参数)
  - [Linear Booster 参数](#linear-booster-参数)
  - [学习任务参数](#学习任务参数)
  - [参数调节](#参数调节)
- [XGBoost API](#xgboost-api)
  - [核心数据结构](#核心数据结构)
  - [Learning API](#learning-api)
  - [Scikit-Learn API](#scikit-learn-api)
  - [数据可视化 API](#数据可视化-api)
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

`$$Gain = \frac{1}{2}\Bigg[\frac{G_{L}^{2}}{H_K + \lambda} + \frac{G_{R}^{2}}{H_R + \lambda} - \frac{(G_L + G_{R})^{2}}{H_L + H_R + \lambda}\Bigg] - \gamma$$`

其中: 

- `$\frac{G_{L}^{2}}{H_K + \lambda}$`: 左子树分数
- `$\frac{G_{R}^{2}}{H_R + \lambda}$`: 右子树分数
- `$\frac{(G_L + G_{R})^{2}}{H_L + H_R + \lambda}$`: 不分割可以得到的分数
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
    & \;\;\;\;\;\;\;\;{G_{R}} \leftarrow G - G_{L}, H_{R} \leftarrow H - H_{L}  \cr 
    & \;\;\;\;\;\;\;\;score \leftarrow \max \left( score, \frac{G_{L}^{2}}{H_{L} + \lambda} + \frac{G_{R}^{2}}{H_{R} + \lambda} - \frac{G^2}{H + \lambda} \right)  \cr 
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
记为分界点 `$v$` 的统计量，`$v$` 满足 `$\{{\bf{x}}_{kj}=s_{kv}\}$`。 
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

# XGBoost 参数

## 通用参数

> 控制整个模型的通用性能

* `booster`：基本学习器类型，默认 `gbtree`
    - `gbtree`：基于树的模型
    - `gblinear`：线性模型
    - `dart`：TODO
* `nthread`：用于运行的并行线程数，默认为最大可用线程数
* `verbosity`：打印消息的详细程度
    - `0`：静默
    - `1`：警告
    - `2`：信息
    - `3`：调试

## Tree Booster 参数

> 控制每步迭代中每个基学习器(树模型)

需要调参：

* `eta`(`learning_rate`)
    - 学习率，在更新中收缩每步迭代中基本学习期的权重，使模型更加稳健，防止过度拟合
    - 默认 0.3
    - 范围：`$[0, 1]$`
    - 典型值一般设置为：`$[0.01, 0.2]$` 或 `$[0.05, 0.3]$`
* `n_estimators`
    - 树模型基学习器的数量
* `max_depth`
    - 一棵树的最大深度。增加此值将使模型更复杂，并且更可能过度拟合
    - 默认 6
    - 范围：`$[0，\infty)$`
    - 典型值一般设置为：`$[3, 10]$`
* `min_child_weight`
    - 子节点中所有样本的权重和的最小值，如果新分裂的节点的样本权重和小于 `min_child_weight` 则停止分裂，
      可以用来减少过拟合，但是也不能太高，太高会导致欠拟合
    - 默认 1
    - 范围：`$[0，\infty)$`
    - 典型值一般设置为：1
* `gamma`(`min_split_loss`)
    - 树模型节点当分裂结果使得损失函数减少时，才会进行分裂。
      分裂节点时，损失函数减小值只有大于等于 gamma 时节点才分裂。
      gamma 值越大，算法越保守，越不容易过拟合，但性能就不一定能保证，需要平衡
    - 默认 0
    - 范围：`$[0，\infty)$`
    - 典型值一般设置为：`$[0.1, 0.2]$`
* `subsample`
    - 构建每棵树对样本的采样率
    - 默认 1
    - 范围：`$(0, 1]$`
    - 典型值一般设置为：`$[0.5, 0.9]$`
* `colsample_bytree`
    - 列采样率，也就是特征采样率
    - 默认 1
    - 范围：`$(0，1]$`
    - 典型值一般设置为：`$[0.5, 0.9]$`
* `scale_pos_weight`
    - 控制正负权重的平衡，在类别高度不平衡的情况下，将参数设置大于 0，可以加快收敛。
      Kaggle 竞赛一般设置 `sum(negativ instances) / sum(positive instances)`
* `lambda`(reg_lambda)：
    - 叶节点权重得分的 L2 正则参数。L2 正则化权重项，增加此值将使模型更加保守
    - 默认 1
* `alpha`(reg_alpha)：
    - 叶节点权重得分的l1正则参数。权重的 L1 正则化项，增加此值将使模型更加保守。适用于高维数据中, 能使得算法加速
    - 默认 0

一般不需要调参：

* `max_leaf_nodes`
    - 树中叶节点的最大数量
* `max_delta_step`：
    - 允许每个叶子输出的最大增量步长。如果将该值设置为 0，则表示没有约束。如果将其设置为正值，
      则可以帮助使更新步骤更加保守。通常不需要此参数，但是当类极度不平衡时，
      它可能有助于逻辑回归。将其设置为 `$[1, 10]$` 的值可能有助于控制更新
    - 默认 0
    - 范围：`$[0，\infty)$`
* `sampling_method`
    - 用于对训练样本进行采样的方法
    - 默认 `uniform`
    - `uniform`：每个训练实例的选择概率均等。通常将 subsample>=0.5 设置为良好的效果
    - `gradient_based`：每个训练实例的选择概率与规则化的梯度绝对值成正比，
      具体来说就是，subsample 可以设置为低至 0.1，而不会损失模型精度
* `tree_method`
    - XGBoost 中使用的树构建算法 
    - 默认 `auto`
    - auto：使用启发式选择最快的方法
    - exact：对于小型数据集，精确的贪婪算法，枚举所有拆分的候选点
    - approx：对于较大的数据集，使用分位数和梯度直方图的近似贪婪算法
    - hist：更快的直方图优化的近似贪婪算法(LightGBM 也是使用直方图算法)
    - gpu_hist：GPU hist 算法的实现
- `num_parallel_tree`
    - 每次迭代期间构造的并行树的数量。此选项用于支持增强型随机森林
    - 默认 1
- `monotone_constraints`
    - 可变单调性的约束，在某些情况下，如果有非常强烈的先验信念认为真实的关系具有一定的质量，
      则可以使用约束条件来提高模型的预测性能
    - 例如 `params_constrained['monotone_constraints'] = "(1,-1)"`，
      `(1,-1)` 表示对第一个预测变量施加增加的约束，对第二个预测变量施加减小的约束

## Linear Booster 参数

- `lambda`(reg_lambda)：
    - L2 正则化权重项。增加此值将使模型更加保守。归一化为训练示例数
    - 默认 0
- `alpha`(reg_alpha)：
    - 权重的 L1 正则化项。增加此值将使模型更加保守。归一化为训练示例数
    - 默认 0
- `updater`：默认 `shotgun`
    - `shotgun`：基于 shotgun 算法的平行坐标下降算法。使用 "hogwild" 并行性，因此每次运行都产生不确定的解决方案
    - `coord_descent`：普通坐标下降算法。同样是多线程的，但仍会产生确定性的解决方案
- `feature_selector`：特征选择和排序方法，默认 `cyclic`
    - `cyclic`：通过每次循环一个特征来实现的
    - `shuffle`：类似于 `cyclic`，但是在每次更新之前都有随机的特征变换
    - `random`：一个随机(有放回)特征选择器
    - `greedy`：选择梯度最大的特征(贪婪选择)
    - `thrifty`：近似贪婪特征选择(近似于 `greedy`)
- `top_k`：要选择的最重要特征数(在 `greedy` 和 `thrifty` 内)

## 学习任务参数

> 控制模型优化的表现

* `objective`：默认 `reg:squarederror`，表示最小平方误差
    - 回归
        - `reg:squarederror`：最小平方误差
        - `reg:squaredlogerror`：对数平方损失
        - `reg:logistic`：逻辑回归
        - `reg:pseudohubererror`：使用伪 Huber 损失进行回归，这是绝对损失的两倍可微选择
        - `reg:tweedie`：使用对数链接进行 Tweedie 回归
    - 二分类
        - `binary:logistic`：二元分类的逻辑回归，输出分类概率
        - `binary:logitraw`：用于二进制分类的逻辑回归，逻辑转换之前的输出得分
        - `binary:hinge`：二进制分类的铰链损失。这使预测为0或1，而不是产生概率。(SVM就是铰链损失函数)
    - 生存分析
        - `survival:cox`：针对正确的生存时间数据进行 Cox 回归(负值被视为正确的生存时间)
        - `survival:aft`：用于检查生存时间数据的加速故障时间模型
        - `aft_loss_distribution`：survival:aft 和 aft-nloglik 度量标准使用的概率密度函数
    - 多分类
        - `multi:softmax`：使用 softmax 目标函数进行多类分类，需要设置 `num_class`(类数)，返回分类标签
        - `multi:softprob`：与 softmax 相同，但输出向量，可以进一步重整为矩阵。
          结果包含属于每个类别的每个数据点的预测概率
    - 排序
        - `rank:pairwise`：使用 LambdaMART 进行成对排名，从而使成对损失最小化
        - `rank:ndcg`：使用 LambdaMART 进行列表式排名，使标准化折让累积收益(NDCG)最大化
        - `rank:map`：使用 LambdaMART 进行列表平均排名，使平均平均精度(MAP)最大化
        - `reg:gamma`：使用对数链接进行伽马回归。输出是伽马分布的平均值
    - `count:poisson`：计数数据的泊松回归，泊松分布的输出平均值
    - 自定义损失函数和评价指标
* `eval_metric`：验证数据的评估指标，将根据目标分配默认指标(回归均方根，分类误差，排名的平均平均精度)，
  用户可以添加多个评估指标
    - 回归
        - `rmse`，均方根误差
        - `rmsle`：均方根对数误差
        - `mae`：平均绝对误差
        - `mphe`：平均伪 Huber 错误
    - 二分类
        - `logloss`：负对数似然
        - `error`：二(进制)分类错误率
    - 多分类
        - `merror`：多类分类错误率
        - `mlogloss`：多类 `logloss`
    - `auc`：曲线下面积
    - `aucpr`：PR 曲线下的面积
    - `ndcg`：归一化累计折扣
    - `map`：平均精度
* `seed`：随机数种子，默认 0

## 参数调节

参数调优的一般策略:

1. 首先, 选择一个相对较大的 `eta`/`learning_rate`, 比如: 0.1(一般范围在: `$[0.05, 0.3]$`)
    - 根据这个选定的 `learning_rate`，对树的数量 `n_estimators` 进行 CV 调优，
      选择一个最优的 `n_estimators`
2. 依次调节树参数 
    - `max_depth`
        - `$[3, 10]$`
    - `min_child_weight`
        - 类别不平衡数据选择一个较小值
    - `gamma`
        - `$[0.1, 0.2]$`
    - `subsample`
        - `$[0.5, 0.9]$`
    - `colsample_bytree`
        - `$[0.5, 0.9]$`
    - `scale_pos_weight`
        - 类别不平衡数据选择一个较小值
3. 调节正则化参数
    - `lambda`: L2
    - `alpha`: L1
4. 减小 `learning_rate`, 增加决策树数量 `n_estimators`，并且优化其他参数
5. 调节对模型结果影响最大的参数
   - `max_depth`
   - `min_child_weight`

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

