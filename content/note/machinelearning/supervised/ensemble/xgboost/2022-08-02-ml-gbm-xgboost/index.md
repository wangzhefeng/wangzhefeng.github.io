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

- [XGBoost 简介](#xgboost-简介)
  - [XGBoost 特点](#xgboost-特点)
  - [XGBoost vs GBDT](#xgboost-vs-gbdt)
  - [XGBoost 优势](#xgboost-优势)
  - [XGBoost  优缺点](#xgboost--优缺点)
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
- [XGBoost 工程优化](#xgboost-工程优化)
  - [列块并行学习](#列块并行学习)
  - [缓存访问](#缓存访问)
  - [块式核外计算](#块式核外计算)
- [参考](#参考)
</p></details><p></p>

# XGBoost 简介

> XGBoost，eXtreme Gradient Boosting

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

## XGBoost vs GBDT

GBDT 基于负梯度（"残差"）优化偏差的训练方式容易使模型过拟合，
虽然 GBDT 在学习率、树数量、子采样比例和决策树结构参数上有做正则化去避免过拟合，
但是有没有其他更好的正则化手段呢？

另外，GBDT 的串行训练方式造成的计算开销大，能从其它地方优化，从而加快模型训练吗？
答案是有，2016 年陈天奇发表了 XGBoost，他在 GBDT 的基础上，优化了算法原理和工程实现，
但本质上它跟 GBDT 一样都是加性模型

![img](images/xgb_gbdt.png)

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

## XGBoost  优缺点

> XGBoost 的缺点也是 LightGBM 的出发点

优点：

* 预测精度比 GBDT 高，因为采用二阶泰勒展开
* 更有效避免过拟合，因为正则化手段更丰富了
    - 例如：在损失函数上加入正则化项（抑制叶子节点数量和叶子节点权重）、缩减、列采样等
* 针对缺失值和稀疏值有较好的处理手段，自动分配分裂方向
* 灵活性强，不仅支持 CART 树作为弱学习器，还支持其他线性分类器如逻辑回归
* 工程落地性好，支持特征粒度上的并行。因为使用了一系列工程优化方法，
  例如块存储，还有使用近似算法减少计算量

缺点：

* XGBoost 还是跟 GBDT 一样，处理不好类别型数据
    - 一旦类别型特征出现，XGBoost 把类别型特征当作数值型特征处理，用 Label Encoding 会增加模型学习难度，
      但陈天奇在项目问答模块回答道：“当类别数量小 (比如10-100) 时，可以考虑 One-hot Encoding。”，
      而如果类别数过大，个人觉得 One-hot 编码的风险会很大，因为带来了很大稀疏性，这会导致两种结果：
        - (1) 增加内存和时间开销
        - (2) 稀疏特征的分裂下的子树泛化性差，易拟合
* 存在内存开销，需要存储梯度统计值和索引指针
* 每轮迭代时, 都需要遍历整个训练数据集多次:
    - 如果把整个训练数据装进内存则会限制训练数据的大小
    - 如果不装进内存, 反复地读写训练数据又会消耗非常大的时间
* 预排序方法(pre-sorted)
    - 首先, 空间消耗大. 这样的算法需要保存数据的特征值, 
      还保存了特征排序的结果(例如排序后的索引, 为了后续快速地计算分割点), 
      这里需要消耗训练数据两倍的内存 
    - 其次时间上也有较大的开销, 在遍历每一个分割点的时候, 都需要进行分裂增益的计算, 消耗的代价大
* 对 cache 优化不友好:
    - 在预排序后, 特征对梯度的访问是一种随机访问, 并且不同的特征访问的顺序不一样, 
      无法对 cache 进行优化. 同时, 在每一层长树的时候, 需要随机访问一个行索引到叶子索引的数组, 
      并且不同特征访问的顺序也不一样, 也会造成较大的 cache miss

# XGBoost 模型理论

![img](images/flow.png)

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


# XGBoost 工程优化

前面花了大量篇幅讲解了作者在"原理优化"上的工作，我们现在来看下“工程优化”，
个人觉得这是论文的亮点之一，毕竟陈天奇的研究方向是大规模机器学习，
而且这也是 XGBoost 在工业界受欢迎的原因，就因为工程优化做的好。
作者在"工程优化"上主要有三点：列块并行学习、缓存访问和块式核外计算

## 列块并行学习

在上面近似算法中，我们知道要提前对特征进行值排序，才能进行扫描找最佳分裂点。
但排序是很耗时的，比如像Local局部分裂策略，我们如果每次分裂完又重新排序，这就耗时了。
所以为了减少时间开销，我们在 XGBoostx 训练前就先对特征进行值排序，
然后将排序好的特征值和对应样本的位置指针保存至块中 (Block)。
块中的数据是以稀疏矩阵 (即 Compressed Sparse Columns, CSC) 格式保存的，
使得后续扫描都能重复使用。由于一个块里放一列，所以这对并行化和列采样都有帮助

![img](images/list_block.png)

## 缓存访问

在我们扫描排序好的特征值时，我们需要基于对应位置指针找到一阶和二阶导数，
但这不是连续内存空间访问，所以可能会造成CPU缓存丢失，从而使分裂点寻找的过程变慢了。
对此，XGBoost 提出缓存预加载算法 (Cache-aware Prefetching Algorithm)，
为每个线程分配一个内部缓存区，将导数数据放在里面，然后以小批次的方式进行累加操作，
这样将不连续的读写尽可能变得连续。这种方法，对于贪心算法来说，提速2倍。
但对于近似算法，要注意平衡块大小，因为块太小会导致低效并行化，块太大会导致缓存丢失。
实验显示，每个块 `$2^{16}$` 个样本可以达到较好的平衡

## 块式核外计算

一般来说，我们是把数据主要加载至CPU内存中，把待加载数据放在硬盘中，等 CPU 运算完再读取硬盘数据，
可 CPU 和硬盘的 IO 不同，前者更快，后者更慢。为了在这块提速，
作者提出两个核外 (out-of-core) 计算技巧：

* 块压缩 (Block Compression)：采用列压缩，然后用一个独立线程对待加载入CPU的数据进行解压。这样减少了读取压力。
* 块分区 (Block Sharding)：将数据分区至多个硬盘中，另外给每个硬盘都分配个线程预加载，这增加了磁盘读取的吞吐

# 参考

* [Xgboost: A scalable tree boosting system]()
* [GitHub-Python-Package]()
* [GitHub-R-Package]()
* [GitHub-Microsoft]()
* [Doc](https://xgboost.readthedocs.io/en/latest/get_started.html)
* [Python 示例]()
* [《深入理解XGBoost》- Microstrong]()
* [陈天奇在项目问答模块](https://github.com/dmlc/xgboost/issues/21)
* [comment:re sklearn -- integer encoding vs 1-hot (py)](https://github.com/szilard/benchm-ml/issues/1)
* [务实基础-XGBoost](https://mp.weixin.qq.com/s?__biz=MzUyNzA1OTcxNg==&mid=2247485425&idx=1&sn=36b23b224ceb8e5595730042fc7797a8&chksm=fa04199acd73908c0498b323b7edb8b3ce52b17964edc2462cb842ff6fefe3fdc0b5cadd91ba&cur_album_id=1577157748566310916&scene=189#wechat_redirect)

