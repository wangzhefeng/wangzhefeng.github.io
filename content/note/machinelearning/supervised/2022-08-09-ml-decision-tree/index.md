---
title: Decision Tree
author: wangzf
date: '2022-08-09'
slug: ml-decision-tree
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
img {
    pointer-events: none;
}
</style>

<details><summary>目录</summary><p>

- [决策树介绍](#决策树介绍)
  - [模型介绍](#模型介绍)
  - [模型构建](#模型构建)
  - [模型优缺点](#模型优缺点)
- [决策树学习基本算法](#决策树学习基本算法)
- [决策树节点分裂算法](#决策树节点分裂算法)
  - [信息熵](#信息熵)
  - [信息增益](#信息增益)
  - [信息增益率](#信息增益率)
  - [CART](#cart)
    - [模型介绍](#模型介绍-1)
    - [模型构建](#模型构建-1)
    - [节点分裂准则](#节点分裂准则)
- [决策树剪枝](#决策树剪枝)
  - [预剪枝](#预剪枝)
  - [后剪枝](#后剪枝)
  - [泛化性能评价](#泛化性能评价)
- [决策树实现](#决策树实现)
  - [分类](#分类)
  - [回归](#回归)
- [参考](#参考)
</p></details><p></p>

# 决策树介绍

## 模型介绍

决策树是一种典型的基本分类器，从本质上讲，决策树的分类思想是产生一系列规则，
然后通过这些规则进行数据分析的数据挖掘过程。该分类器的生成和决策过程分为三个部分：

1. 首先，通过对训练集进行递归分析, 生成一棵形状如倒立的树状结构
2. 然后，分析这棵树从根节点到叶节点的路径, 产生一系列规则
3. 最后，根据这些规则对新数据进行分类预测 

决策树为一个树状模型，树中包含三种节点：根节点、(中间)内部节点、叶节点：

* 树的每个节点表示样本对象的集合，根节点包含样本全集
* 从根节点出发，经过若干中间节点后，到达叶节点的路径表示某个规则，
  整个树表示由训练样本决定的规则集合
* 从每个节点出发的分叉路径代表某个可能的属性值，
  每个节点包含的样本集合根据属性测试的结果被划分到子节点中
* 每个叶节点对应从根节点到该叶节点所经历的路径所表示的对象的值。
  即叶节点对应于决策结果，其他每个节点则对应于一个属性测试(分割变量选择、分割节点选择)

## 模型构建

决策树模型构建的问题基本可以归结为两点: 

* 生成树(split)
    - 选择变量作为分割变量(分割节点)
        - 决策树的生成过程就是使用满足划分准则的特征不断的将数据集划分为纯度更高，不确定性更小的子集的过程。
          对于当前数据集的每一次的划分，都希望根据某特征划分之后的各个子集的纯度更高，不确定性更小
        - 特征选择准则: 
            - 使用某特征对数据集划分之后, 数据子集的纯度要比划分前的数据集 D 的纯度高(不确定性要比划分前数据集 D 的不确定性低)
            - 度量划分前后的纯度变化用子集的纯度之和与划分前的数据集 D 的纯度进行对比，
              划分后的纯度为各数据子集的纯度的加和(子集占比 `$\times$` 子集的经验熵)
    - 如何度量划分数据集前后的数据集的纯度（不确定性） 
        - 回归树
            - 树节点分裂准则
                - 预测误差，计算分裂前后集合的误差得分，取误差得分最小的进行分裂
                    - 均方误差
                    - 对数误差
            - 树叶节点预测值
                - 节点内样本目标变量均值
                - 通过最优化算法计算得到(XGBoost)
        - 分类树
            - 树节点分裂准则
                - 信息增益: ID3 模型
                - 信息增益率: C4.5 模型
                - Gini 指数: CART 模型
            - 树叶节点预测值
                - 通过投票法产生的类别值(少数服从多数)
* 树的剪枝(prune)
    - 预剪枝
    - 后剪枝
    - 正则化方法(XGBoost)

## 模型优缺点

决策树模型是一种简单易用的非参数分类器，算法的优缺点：

* 决策树模型的优点是 
    - 不需要对数据有任何的先验设计
    - 对噪声数据和缺失数据不敏感
    - 计算速度较快
    - 结果容易解释
    - 稳健性强
* 决策树模型的缺点是 
    - 1.精确度不高
        - 因为它是矩形的判别边界，使得精确度不高
        - 对回归问题不太适合。在处理回归问题时建议使用模型树(model tree)方法，即先将数据切分，再对各组数据进行线性回归
    - 2.单个决策树不太稳定. 数据微小的变化会造成模型结构变化 
    - 3.有变量选择偏向，即会选择那些有取值较多的变量
        - 一种改善的方法是使用条件推断树
        - 还可以采用集成学习方法
    - 4.分类规则复杂
        - 决策树算法在产生规则的时候采用了局部贪婪方法。
          每次只选取一个属性进行分析构造决策树，所以他产生的规则很复杂，
          针对此问题一般采用剪枝的方法实现 
    - 5.收敛到非全局的局部最优解(ID3) 
    - 6.容易过度拟合

# 决策树学习基本算法

* 输入：
    - 训练集：`$D=\{(x_{1}, y_{1}),(x_{2}, y_{2}),\ldots,(x_{n}, y_{n})\}$`
    - 变量集：`$A=\{a_{1}, a_{2},\ldots,a_{p}\}$`
* 过程：
    - 函数: `$TreeGenerate(D, A)$`
    1. 生成根节点 node
    2. **if** 【`$D$` 中样本全属于同一类别 `$C$`】**then**
        - 将根节点 node 标记为 `$C$` 类叶节点
        - **return**
    3. **end if**
    4. **if** 【`$A=\emptyset$` **or** `$D$` 中样本在 `$A$` 上取值相同】 **then**
        - 将根节点 node 标记为叶节点, 其类别标记为 `$D$` 中样本数最多的类
        - **return**
    5. **end if**
    6. **else**
        - 从 `$A$` 中选择最优划分变量 `$a_{*}$`
        - **for** 【`$a_{*}$` 的每一个值 `$a_{*}^{v}$`】
            **do**： 
            - 为 node 生成一个分支；令 `$D_{v}$` 表示 `$D$` 中在 `$a_{*}$` 上取值为 `$a_{*}^{v}$` 的样本子集
                - **if** 【`$D_{v}=\emptyset$`】 **then**
                    - 将根节点 node 标记为叶节点, 其类别标记为 `$D$` 中样本数最多的类
                - **else**
                    - 以 `$TreeGenerate(D, A \& \{a_{*}\})$` 为分支节点
                - **end if**
        - **end for**
* 输出：
    - 以 node 为根节点的一棵决策树

# 决策树节点分裂算法

## 信息熵

信息熵(information entropy)是度量样本集合(随机变量)纯度(不确定性)最常用的指标。
信息熵的值越小，则样本集合的纯度越高；信息熵的值越大，则样本集合的不确定性就越大

假设样本集合 `$D$` 中第 `$k$` 类样本所占的比例为 `$p_{k},k = 1, 2, \ldots, |\mathcal{Y}|$`，
其中 `$|\mathcal{Y}|$` 为样本集合 `$D$` 中的类别总数，则样本集合 `$D$` 的信息熵为: 

`$$Entropy(D) = -\sum^{|\mathcal{Y}|}_{k=1}p_{k}log_{2}p_{k}$$`

特别地，对于二分类的情况，`$D$` (根节点)中只有两个类别，
假设每个类别在样本集中的比例为：正例的比例 `$p_{+}$` , 
以及负例的比例 `$p_{-}$` , 则样本集合 `$D$` (根节点)的信息熵为: 

`$$Entropy(D) = -(p_{+}\log_2 p_{+} + p_{-}\log_{2} p_{-})$$`

## 信息增益

信息增益(information gain)主要作为 ID3 决策树[Quinlan,1986]的变量选择分裂准则，
这里对节点的划分不是二叉划分，是多叉划分。信息增益越大，则使用相应变量来进行划分所获得的"纯度提升”越大

假设离散变量 `$a$` 有 `$V$` 个不同取值 `$\{a^{1}, a^{2}, \ldots, a^{V}\}$`。
若使用变量 `$a$` 对样本集合 `$D$` 进行划分，则会产生 `$V$` 个分支节点。
其中第 `$v$` 个分支节点包含了 `$D$` 中所有在变量 `$a$` 上取值为 `$a^{v}$` 的样本，
记为 `$D^{v}$`。易知：

样本集合 `$D$` 的信息熵为：

`$$Entropy(D) = -\sum^{|\mathcal{Y}|}_{k=1}p_{k}log_{2}p_{k}$$`

分裂后的各个样本集合 `$D^{1},D^{2},\ldots D^{v}, \ldots, D^{V}$` 的总信息熵为：

`$$\sum^{V}_{v=1}Entropy(D^{v})$$`

再考虑到不同的分支节点所包含的样本数不同，所以给分支节点赋权重 `$\frac{|D^{v}|}{|D|}$`，
即样本数越多的分支节点影响越大，于是可计算出用变量 `$a$` 进行划分所获得的"信息增益”为: 

`$$Gain(D, a)=Entropy(D)-\sum^{V}_{v=1}\frac{|D^{v}|}{|D|}Entropy(D^{v})$$`

最终选择获得信息增益最大的变量为: 

`$$a_{*}=\underset{a\in A}{argmax} Gain(D, a)$$`

## 信息增益率

信息增益准则对可取值数目多的变量有所偏好，为减少这种偏好可能带来的不利影响，
C4.5 决策树算法[Quinlan,1993]不直接使用信息增益，
使用信息增益率(information gain ratio)来选择最优的划分变量

信息增益率准则对变量取值数目较少的属性有所偏好，因此 C4.5 并不是直接选择信息增益率最大的变量，
而是使用了一个启发式的：先从变量集合中找出信息增益率高于平均水平的变量，再从中选择信息增益率最高的

C4.5 仍然是多分叉树, 下面给出信息增益率的表达形式, 它是在信息增益的基础上进行改进得到的

`$$Gain\_ratio(D, a)=\frac{Gain(D, a)}{IV(a)}$$`

其中: 

* `$IV(a)$` 称为变量 `$a$` 的"固有值(intrinsic value)”, 
  变量 `$a$` 可能取值的数目越多(即 `$V$` 越大), 则 `$IV(a)$` 的值通常会越大: 

`$$IV(a)=-\sum^{V}_{v=1}\frac{|D^{v}|}{|D|}log_{2}\frac{|D^{v}|}{|D|}$$`

## CART

### 模型介绍

CART(Classification And Regression Tree，分类与回归树)是树模型算法的一种，
它先根据基尼指数从自变量集合中寻找最佳分割变量和最佳分割点，
将数据划分为**两组**。针对分组后的数据将上述步骤重复下去，直到满足某种停止条件。
这样反复分隔数据后使分组后的数据变得一致，纯度较高。同时可自动探测出复杂数据的潜在结构、
重要模式和关系, 探测出的知识又可用来构造精确和可靠地预测模型。
关于分类与回归树的深入理论，可以参考 Breiman 等人的著作 

CART 模型的构建可以看作是一个对变量进行递归分割选择的过程。
递归分割, 顾名思义也就是对变量进行逐层分隔，直到分割结果满足某种条件才停下来，
这里"分割的结果”可能是得到一些分类值(分类树)，也可能是一些描述统计量或预测值(回归树)。
建立的 CART 模型可分为分类树(classification tree)和回归树(regression tree)两种：

* 分类树用于因变量为分类数据的情况，树的末端为因变量的分类值
* 回归树则可以用于因变量为连续变量的情况，树的末端可以给出相应类别中的因变量描述或预测

### 模型构建

1. 对所有自变量和所有分割点进行评估，选择最优分割变量，生成一棵规模较大的 CART 树
     - 最佳的选择是使得分割后组内的数据纯度更高，即组内数据的目标变量变异更小。这种纯度可通过 Gini 指数或熵 Entropy 来度量
2. 对树进行剪枝(prune)
    - 要兼顾树的规模和误差的大小, 因此通常会使用 CP 参数(complexity parameter)来对树的复杂度进行控制，
       使预测误差和树的规模都尽可能小
    - 通常的做法是，先建立一个划分较细较为复杂的树模型，再根据交叉验证方法来估计不同剪枝条件下各模型的误差，选择误差最小的树模型
3. 输出最终结果，进行预测和解释

### 节点分裂准则

> Gini 指数，Gini Index

数据集的纯度可用基尼值(Gini value)来度量。基尼值越小，则数据集的纯度越高

CART[Breiamn et al.,1984]是二分叉树，其使用基尼指数(Gini index)来选择划分变量。
基尼指数越小，则使用相应变量来进行划分所获得的"纯度提升”越大

对于上面的样本集合 `$D$` 的纯度可用基尼值来度量：

`$$Gini(D)=\sum^{|\mathcal{Y}|}_{k=1}\sum_{k' \neq k}p_{k}p_{k'}=1-\sum^{|\mathcal{Y}|}_{k=1}p_{k}^{2}$$`

假设变量 `$a$` 有 `$V$` 个可能的取值 `${a^{1}, a^{2}, \ldots, a^{V}}$`。
若使用变量 `$a$` 对样本集合 `$D$` 进行划分

如果 `$a$` 为离散变量，对于变量 `$a$` 的每一个取值 `$a^{i}, (i=1, 2,\ldots,V)$`，会产生两个分支节点，

* 第一个分支节点(左分支)包含了 `$D$` 中所有在变量 `$a$` 上取值为 `$a^{i}$` 的样本，记为 `$D^{l}$`
* 第二个分支节点(右分支)包含了 `$D$` 中所有在变量 `$a$` 上取值不为 `$a^{i}$` 的样本，记为 `$D^{r}$`

如果 `$a$` 为连续变量，对于变量 `$a$` 的每一个取值 `$a^{i}, (i=1, 2,\ldots,V)$`，
（假设已经从小到大进行了排列: `$a^{1} \leqslant a^{2} \leqslant \ldots \leqslant a^{n}$`)
基于分割点 `$t$` 可将样本集合 `$D$` 划分为两个子集，即会产生两个分支节点：

* 第一个分支节点(左分支)包含了 `$D$` 中所有在变量 `$a$` 上取值为 `$a \leq t$` 的样本，记为 `$D^{l}$`
* 第二个分支节点(右分支)包含了 `$D$` 中所有在变量 `$a$` 上取值为 `$a > t$` 的样本，记为 `$D^{r}$`

显然对于相邻的两个取值 `$a^{i}$` 与 `$a^{i+1}$` 来说，
`$t$` 在区间 `$[a^{i},a^{i+1})$` 中取任意值所产生的划分结果都相同。
因此, 对连续变量，可考察包含 `$V-1$` 个元素的候选分割点集合：
     
`$$T_{a}=\bigg\{\frac{a^{i}+a^{i+1}}{2}|1 \leqslant i \leqslant V-1 \bigg\}$$`

即把区间 `$[a^{i},a^{i+1})$` 的中位点作为候选分割点，然后就可以像离散变量一样来考察这些分割点，选取最优的分割点进行样本集合的划分 

假设集合 `$D^{l}$` 的基尼值为 `$Gini(D^{l})$`，集合 `$D^{r}$` 的基尼值为 `$Gini(D^{r})$`。
再考虑到两个不同的分支节点所包含的样本数不同, 所以给左右两个分支节点分别赋权重 `$\frac{|D^{l}|}{|D|}$`、
`$\frac{|D^{r}|}{|D|}$`，即样本数越多的分支节点影响越大，于是可计算出用变量 `$a$` 进行划分的基尼指数为：

`$$Gini\_index(D,a)=\frac{|D^{l}|}{|D|}Gini(D^{l})+\frac{|D^{r}|}{|D|}Gini(D^{r})$$`

选择使得划分后基尼指最数小的变量作为最优划分变量：

`$$a_{*}=\underset{a \in A}{argmin}Gini\_index(D, a)$$`

# 决策树剪枝

剪枝(Purning)是决策树减轻"过拟合”的主要手段。在决策树学习中，为了尽可能地正确分类训练样本，
节点划分不断重复，有时会造成决策树分支过多，这时可能因为训练样本学得太好了，
以至于把训练样本自身的一些特点当做所有数据都具有的一般性质而导致过拟合。
因此可通过去掉一些分支来降低过拟合的风险 

决策树有两种常用的剪枝策略：

* 预剪枝(prepruning)
* 后剪枝(postpurning)

## 预剪枝

> prepruning

在决策树生成过程中，对每个节点在分割前先进行估计，若当前的分割不能带来决策树泛化性能的提升，
则停止分割并将当前节点标记为叶节点

在预剪枝过程中，决策树的很多分支都没有展开，这不仅降低了过拟合的风险，
还显著减少了决策树的训练时间和预测时间；但有些分支的当前分割虽不能提升泛化性能、甚至能导致泛化性能下降，
但在其基础上的后续分割却有可能导致泛化性能的显著提高，容易导致欠拟合

## 后剪枝

> postpurning

后剪枝的基本思想是：先从训练样本集生成一棵最大规模的完整的决策树，然后自底向上地对非叶结点进行考察，
将某非叶节点下的节点删除，若该剪枝能提升决策树的泛化性能，则将该子树替换为叶节点

后剪枝决策树比预剪枝决策树保留了更多的分支，后剪枝过程的欠拟合风险很小，
泛化性能往往优于预剪枝决策树

但后剪枝过程是在生成完全决策树之后进行的，并且要自底向上地对树中所有的非叶节点进行逐一考察，
因此训练时间开销比未剪枝决策树和预剪枝决策树都要大的多

## 泛化性能评价

评估方法：

* 留出法
* 交叉验证
* 自助法

评估性能度量

* 分类准确率(错误率)
* MSE

# 决策树实现

## 分类

示例 2：

```python
from sklearn import tree

# data
X = [[0, 0], 
    [1, 1]]
Y = [0, 1]

# train model
clf = tree.DecisionTreeClassifier()
clf = clf.fit(X, Y)

# predict
clf.predict([[2.0, 2.0]])

clf.preidct_proba([[2.0, 2.0]])
```

示例 2：

```python
from sklearn.datasets import load_iris
from sklearn import tree

# data
iris = load_iris()

# train model
clf = tree.DecisionTreeClassifier()
clf = clf.fit(iris.data, iris.target)

# export tree in Graphviz format
import graphviz
dot_data = tree.export_graphviz(clf, out_file = None)
graph = graphviz.Source(dot_data)
graph.render("iris")

dot_data = tree.export_graphviz(clf, 
                                out_file = None, 
                                feature_names = iris.feature_names, 
                                class_names = iris.target_names,
                                filled = True, 
                                rounded = True,
                                special_characters = True)
graph = graphviz.Source(dot_data)
graph
```

## 回归

```python
from sklearn import tree

# data
X = [[0, 0], [2, 2]]
y = [0.5, 0.25]

# model
clf = tree.DecisionTreeRegressor()
clf = clf.fit(X, y)

# model predict
clf.preidct()
```

# 参考

* [决策树--信息增益，信息增益比，Geni指数的理解](https://www.cnblogs.com/muzixi/p/6566803.html)
