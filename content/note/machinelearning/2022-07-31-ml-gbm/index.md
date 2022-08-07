---
title: GBM
author: 王哲峰
date: '2022-07-31'
slug: ml-gbm
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

- [GBM模型原理](#gbm模型原理)
  - [函数估计问题](#函数估计问题)
  - [梯度提升模型(GBM)](#梯度提升模型gbm)
  - [梯度提升模型算法的应用](#梯度提升模型算法的应用)
- [GBM模型调参(Python)](#gbm模型调参python)
  - [参数类型](#参数类型)
  - [调参策略](#调参策略)
</p></details><p></p>


# GBM模型原理

随着 Breiman 对 AdaBoost 算法的突破性理解和解释的提出, Friedman,
Hastie 和 Tibshirani 将提升算法解释为在统计学框架下的拟合累加模型
(Additive Model) 的函数估计问题. 并且, Friedman
将提升算法扩展为一种利用类似最速下降法优化方法,
在具体的损失函数梯度方向上逐步拟合基本学习器的梯度提升器算法
(Gradient Boosting Machine), 也称为 梯度提升模型算法(Gradient
Boosting Modeling). 从而将提升算法扩展到许多应用上,
同时也产生了一系列具体的梯度提升算法, 例如:

- 对于回归问题, 利用损失函数为平方误差损失函数 `$L(y, f) = (y-f)^{2}$` 时产生的 `$L_{2}$` Boost 算法
- 对于分类问题, 应用对数似然损失函数 `$L(y, f) = log(1+e^{-yf})$` 得到了 LogitBoost 算法
- 选择指数损失函数 `$L(y, f)=exp(-yf)$`, 将会得到 AdaBoost 算法


## 函数估计问题

假设在一个函数估计问题中, 存在:

* 一个随机响应变量 `$y$`
* 一组随机解释变量 `$\mathbf{x}=\{x_{1}, \ldots, x_{d} \}$`, 其中 `$d$` 是解释变量的个数.
  
给定训练数据集  ``$\{(\mathbf{x}_{i}, y_{i}), i=1, 2, \ldots, N \}$``, 
为变量 `$(\mathbf{x}, y)$` 的观测值.
函数估计问题的目标就是利用解释变量 `$mathbf{x}$` 和响应变量 `$y$` 观测值的联合分布, 通过最小化一个特殊的损失函数 `$L(y, f(\mathbf{x}))$` 的期望值得到一个估计或近似函数 `$f(\mathbf{x})$`, 函数 `$f(\mathbf{x})$` 的作用就是将解释变量 `$mathbf{x}$` 映射到响应变量 `$y$`:

`$$f^{*}=\arg\underset{f}{\min}E_{y, \mathbf{x}}[L(y, f(\mathbf{x}))]$$`

其中:

- 损失函数 `$L(y, f(\mathbf{x}))$` 是为了评估响应变量与其函数估计值的接近程度.
- 实际应用中存在很多常用的损失函数:
   - 用来解决回归问题的平方误差损失函数 `$L(y, f)=(y-f)^{2}$` 和绝对误差损失函数 `$L(y, f)=|y-f|$`,
     其中 `$y \in \mathbf{R}$`;
   - 用来解决二分类问题的负二项对数似然损失 `$L(y, f)=log(1+e^{-2yf})$`, 其中 `$y \in \{-1, 1\}$`.

## 梯度提升模型(GBM)

梯度提升模型 (GBM) 是一种对"累加"("additive") 扩展模型的拟合方法, 在这里, 
"累加"(additive) 扩展模型是指由一簇"基本函数"(base function) 扩展成的函数空间中的函数组合. 
而这里的"基本函数"相当于在基分类器 `$G_{m}(\mathbf{x}) \in \{-1, 1\}$`. 
因此方程中关于响应变量 `$y$` 的估计函数 `$f(\mathbf{x})$` 可以表示为一种参数化的"累加"扩展形式:

`$$f(\mathbf{x}; \{\beta_{m},\gamma_{m}\}_{1}^{M}) = \sum_{m=1}^{M}\beta_{m}b(\mathbf{x};\gamma_{m}).$$`

其中:

- `$\{\beta_{m}, \gamma_{m}\}_{1}^{M}$` 是估计函数 `$f(\cdot)$` 的参数集合. 
  并且, 函数 `$b(\mathbf{x};\gamma_{m}), m=1, 2, \ldots, M$` 通常是关于解释变量 `$mathbf{x}$` 的简单学习器函数, 
  例如 `$b(\mathbf{x}; \gamma_{m})$` 可以是一个简单的回归树函数,
  其中参数 `$gamma_{m}$` 是回归树中的分裂变量及分裂位置值.

给定训练数据 `$\{(\mathbf{x}_{i}, y_{i}), i=1,2,\ldots, N\}$`,
将上面的累加模型代入函数估计问题中有

`$$\underset{\{\beta_{m}, \gamma_{m}\}^{M}_{m=1}}{\min}E_{y,\mathbf{x}}(L(y_{i}, \sum_{m=1}^{M}\beta_{m}b(\mathbf{x}_{i};\gamma_{m}))),$$`

即

`$$\underset{\{\beta_{m}, \gamma_{m}\}_{m=1}^{M}}{\min}\sum_{i=1}^{N} L(y_{i}, \sum_{m=1}^{M} \beta_{m} b(\mathbf{x}_{i}; \gamma_{m})),$$`

因此, 方程中的函数估计问题就变成了一个参数估计问题.

在梯度提升模型(GBM) 中,
对于上面的估计问题作者希望利用前向分步累加模型(Forward Stagewise
Additive Modeling) 算法进行求解, 前项分步累加模型算法如下

1. 初始化 `$f_{0}(\mathbf{x})=0$`.
2. 进行迭代, `$m=1, 2, \ldots, M$`, 计算

`$$(\beta_{m}, \gamma_{m})=\underset{\beta, \gamma}{\arg\min}\sum_{i=1}^{N}L(y_{i}, f_{m-1}(\mathbf{x}_{i})+\beta b(\mathbf{x}_{i}; \gamma)).$$`

3. 更新估计函数
  `$f_{m}(\mathbf{x})=f_{m-1}(\mathbf{x})+\beta_{m}b(\mathbf{x};\gamma_{m}).$`

在机器学习中, 上面的方程被称为提升(boosting), 函数 `$b(\mathbf{x};\gamma)$` 被称为弱分类器(weak learner)
或者基本学习器(base learner), 并且一般是一个分类树.

然而, 对于具体的损失函数 `$L(y, f(\mathbf{x}))$` 和基本学习器函数`$b(\mathbf{x}; \gamma)$`, 
前向分步累加模型很难得到最优解. 作者在这里采用了一种类似最速下降法来解决前向分步累加模型算法中的估计问题.
因为在前向分步累加模型的方程中, 如果给定估计函数 `$f_{m-1}(\mathbf{x})$`, 
则 `$beta_{m}b(\mathbf{x};\gamma_{m})$` 可以看成是最速下降算法中求解最优解 `$f_{M}^{*}(\mathbf{x})$`
的最优的贪婪迭代项. 因此, 应用最速下降法, 将估计函数 `$f(\mathbf{x})$` 的数值最解 `$f_{M}^{*} (\mathbf{x})$`
表示为下面的形式

`$$f_{M}^{*}(\mathbf{x})=\sum_{m=0}^{M}h_{m}(\mathbf{x}),$$`

其中:

- `$f_{0}(\mathbf{x})=h_{0}(\mathbf{x})$` 是一个初始化的猜测值,
   `$h_{m}(\mathbf{x}), m=1, 2, \ldots, M$` 是最速下降算法中定义的连续增量函数. 
   最速下降法定义上面的增量函数 `$h_{m}(\mathbf{x}), m=1, 2, \ldots, M$` 如下所示

`$$h_{m}(\mathbf{x})=-\eta _{m}g_{m}(\mathbf{x}),$$`

其中:

`$$g_m (\mathbf{x}) = \Bigg[\frac{\partial E_{y, \mathbf{x}}[L(y, f(\mathbf{x}))]}{\partial f(\mathbf{x})}\Bigg]_{f(\mathbf{x})=f_{m-1}(\mathbf{x})} \\ = E_{y, \mathbf{x}}\Bigg[\frac{\partial L(y, f(\mathbf{x}))}{\partial f(\mathbf{x})}\Bigg]_{f(\mathbf{x})=f_{m-1}(\mathbf{x})},$$`

其中:

* `$g_{m} (\mathbf{x} ) \in R^{N}$` 为损失函数 `$L(y, f(\mathbf{x}))$` 在 `$f(\mathbf{x})=f_{m-1}(\mathbf{x})$` 处的梯度向量. 并且

`$$f_{m-1}(\mathbf{x})=\sum^{m-1}_{i=0}f_{i}(\mathbf{x}),$$`

* 步长: `$eta_{m}, m=1,2,\ldots, M$` 可以通过线性搜索算法得到

`$$eta_{m}=\arg\underset{\eta}{\min}E_{y,\mathbf{x}}L(y_{i}, f_{m-1}(\mathbf{x})-\eta g_{m}(\mathbf{x})).$$`

上面的过程重复迭代, 直到满足算法设定的停止条件.
此时最速下降算法的函数更新形式为

`$$f_{m}(\mathbf{x}) = f_{m-1}(\mathbf{x})-\eta g_{m}(\mathbf{x}).$$`

可以看出, 最速下降法是一种十分贪婪的数值优化策略,
因为算法中负梯度 `$g_{m}$` 是函数空间 `$R^{N}$` 中,
损失函数 `$L(y, f)$` ~ `$f=f_{m-1}$` 处下降最快的局部方向.

如果在训练数据上最小化损失函数 `$L(y,f(\mathbf{x}))$` 是最终的目标, 那么最速下降法会是一种很好的解决方法. 
因为 `$g_{m}(\mathbf{x})$` 对于任何可导的损失函数 `$L(y, f(\mathbf{x}))$` 都是比较容易求得的.
然而, 最速下降法中计算得到的 `$g_{m}(\mathbf{x})$` 只是在训练数据上定义的,
而梯度提升算法的目的却是将最终的模型泛化到除了训练数据之外的未知数据上.
这样训练出的梯度提升算法才具有泛化能力.

因此, 梯度提升模型算法通过利用一个基本的学习器模型算法将负梯度向量进行拟合,
得到了负梯度向量值 `$-g_{m}(\mathbf{x}_{i}), i=1,2,\ldots, N$` 的近似估计向量, 
即和产生了一个在前项分步模型算法中的基本学习器.
然后将这个近似估计向量应用在最速下降算法中代替负梯度向量, 从而使得提升算法拥有泛化能力.
下面是利用平方误差损失拟合负梯度向量估计值的表达式

`$$a_{m}=\arg\underset{a}{\min}\sum_{i=1}^{N}[-g_{m}(\mathbf{x}_{i})-b(\mathbf{x}_{i}; a)]^{2},$$`

当然实际应用中也可以使用其他的一些基本学习器模型来进行拟合.
比较常用的有决策树模型拟合一棵树模型.
而基学习器的权重系数仍然使用最速下降算法中的线性搜索算法得到

`$$eta_{m}=\arg\underset{\eta}{\min}\sum_{i=1}^{N}L(y_{i}, f_{m-1}(\mathbf{x}_{i})+\eta b(\mathbf{x}_{i}; a_{m}))$$`

然后, 将估计近似函数进行更新

`$$f_{m}(\mathbf{x})=f_{m-1}(\mathbf{x})+\eta_{m}b(\mathbf{x}, a_{m})$$`

一般的梯度提升算法的伪代码如下:

1. 初始化 `$\hat{f}^{[0]}(\cdot)$` 为一个常数值. 通常的选择为:
    `$\hat{f}^{[0]}(\cdot) \equiv \arg \underset{c}{\min}\frac{1}{N}\sum_{i=1}^{n}L(y_{i}, c),$`
    或者为 `$\hat{f}^{[0]} \equiv 0$`, 令 `$m=0$`;
2. 增加 `$m=1$`. 计算负梯度 `$-\frac{\partial}{\partial f}$`
    并且计算负梯度在 `$\hat{f}^{[m-1]}(\mathbf{x}_{i})$` 处的值:
    `$U_{i}=-\frac{\partial}{\partial f}L(y_{i}, f)\Bigg|_{f=\hat{f}^{[m-1]}(\mathbf{x}_{i})}, i=1, \ldots, N;$`
3. 将负梯度向量~ `$U_{1}, \ldots, U_{N}$`
    通过一个基本的学习模型(例如: 回归) 拟合到预测变量的观测值向量
    `$mathbf{x}_{1}, \ldots, \mathbf{x}_{N}$`:
    `$(\mathbf{x}_{i}, U_{i})^{N}_{i=1} \rightarrow \hat{b}^{[m]}(\cdot);$`
4. 更新估计函数:
    `$\hat{f}^{[m]}(\cdot)=\hat{f}^{[m-1]}(\cdot)+\eta_{m} \cdot \hat{b}^{[m]}(\cdot),$`
    其中 `$0 \leqslant \eta_{m} \leqslant 1$` 是一个步长因子;
5. 重复第二步到第四步直到 `$m=M$`;
6. 输出训练得到的学习器 `$f^{[M]}(\cdot)$`



## 梯度提升模型算法的应用

在上一节, 我们已经给出了梯度提升模型算法的详细推导及其一般性算法伪代码. 可以看出, 在梯度提升算法中, 
算法在第2 步中对一个具体的损失函数 `$L(y, f(\mathbf{x}))$` 求负梯度向量 `$U_{i}, i=1,2,\ldots, N$`,
而在第 3 步则利用一个具体的基本学习器模型算法, 
对数据预测变量观测值和负梯度向量 `$U_{i}, i=1,2,\ldots, N$` 进行拟合产生负梯度向量 `$U_{i}, i=1,2,\ldots, N$` 的近似估计. 
因此, 在梯度提升模型算法中, 应用不同的损失函数 `$L(y,f(\mathbf{x}))$` 和不同的基本学习器模型算法可以得到不同的提升算法模型.

对于损失函数的选择, 只要损失函数 `$L(\cdot, \cdot)$` 满足对于它的第二个参数变量是光滑且凸的, 
就可以应用到梯度提升算法的第 2 步中. 这一小节, 具体讨论梯度提升算法的一些特殊应用算法.
包括应用平方误差损失函数的 `$L_{2}$` Boost 算法,
应用负二项对数似然损失的 LogitBoost 算法以及基于分位数回归模型的分位数提升分类(QBC) 算法.

下面是这些损失函数的形式

- 用在 `$L_{2}$` Boost 算法中的平方误差损失函数 `$L(y, f) = (y - f)^{2}/2,$`
- 用在 LogitBoost 算法中的对数似然损失函数 `$L(y, f) = log_{2}(1+exp(-2yf)),$`
- 用在分位数提升分类~(QBC) 算法中基于分位数回归模型产生的损失函数 `$L(y, f) = [y-(1-\tau)]K(f/h)$`
   - 其中: `$K(\cdot)$` 是一个标准正态分布的累积分布函数, h是一个给定的大于零的常数.



# GBM模型调参(Python)



## 参数类型

- 决策树参数
   - `min_samples_split`
      - 要分裂的树节点需要的最小样本数量, 若低于某个阈值, 则在此节点不分裂；
      - 用于控制过拟合, 过高会阻止模型学习, 并导致欠拟合；
      - 需要使用CV进行调参；
   - `min_samples_leaf`
      - 叶子节点中所需的最小样本数, 若低于某个阈值, 则此节点的父节点将不分裂, 此节点的父节点作为叶子结点；
      - 用于控制过拟合, 同`min_samples_split` ；
      - 一般选择一个较小的值用来解决不平衡类型样本问题；
   - `min_weight_fraction_leaf`
      - 类似于`min_sample_leaf` ；
      - 一般不进行设置, 上面的两个参数设置就可以了；
   - `max_depth`
      - 一棵树的最大深度；
      - 用于控制过拟合, 过大会导致模型比较复杂, 容易出现过拟合；
      - 需要使用CV进行调参；
   - `max_leaf_nodes`
      - 一棵树的最大叶子节点数量；
      - 一般不进行设置, 设置`max_depth` 就可以了；
   - `max_features`
      - 在树的某个节点进行分裂时的考虑的最大的特征个数, 一般进行随机选择, 较高的值越容易出现过拟合, 但也取决于具体的情况；
      - 一般取特征个数的平方根(跟随机森林的选择一样)；
- Boosting参数
   - `learning_rate`
      - 每棵树对
   - `n_estimators`
   - `subsample`
      - 构建每棵数时选择的样本数；
- 其他参数
   - `loss`
   - `init`
   - `random_state`
   - `verbose`
   - `warm_start`
   - `presort`


## 调参策略

- 一般参数调节策略: 
   - 选择一个相对来说较高的learning rate, 先选择默认值0.1(0.05-0.2)
   - 选择一个对于这个learning rate最优的树的数量(合适的数量为: 40-70)
      - 若选出的树的数量较小, 可以减小learning rate 重新跑GridSearchCV
      - 若选出的树的数量较大, 可以增加初始learning rate
         重新跑GridSearchCV
   - 调节基于树的参数
   - 降低learning rate, 增加学习期的个数得到更稳健的模型
- 对于learning rate的调节, 对其他树参数设置一些默认的值
   - min_samples_split = 500
      - 0.5-1% of total samples
      - 不平衡数据选择一个较小值
   - min_samples_leaf = 50
      - 凭感觉选择, 考虑不平衡数据, 选择一个较小值
   - max_depth = 8
      - 基于数据的行数和列数选择, 5-8
   - mat_features = 'sqrt'
   - subsample = 0.8
- 调节树参数
   - 调节`max_depth` , `min_samples_split`
   - 调节`min_samples_leaf`
   - 调节`max_features`

