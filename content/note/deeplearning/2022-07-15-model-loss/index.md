---
title: 损失函数
author: wangzf
date: '2022-07-15'
slug: model-loss
categories:
  - deeplearning
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

- [损失函数](#损失函数)
    - [回归](#回归)
        - [均方误差损失函数](#均方误差损失函数)
    - [二分类](#二分类)
        - [交叉熵损失函数](#交叉熵损失函数)
        - [二元交叉熵损失](#二元交叉熵损失)
    - [多分类](#多分类)
        - [多元交叉熵损失](#多元交叉熵损失)
        - [对数损失](#对数损失)
        - [负对数似然损失](#负对数似然损失)
        - [KL 散度](#kl-散度)
        - [余弦相似度](#余弦相似度)
        - [自适应对数 Softmax 损失](#自适应对数-softmax-损失)
- [损失函数的优化](#损失函数的优化)
    - [极大似然学习条件分布](#极大似然学习条件分布)
    - [梯度下降算法](#梯度下降算法)
    - [mini-batch 学习](#mini-batch-学习)
</p></details><p></p>

# 损失函数

为了使用基于梯度的学习方法，必须选择一个损失函数，并且必须选择如何表示模型的输出。

1. 在大多数情况下，参数模型定义了一个分布 `$p(y|x;\theta)$`，
   并且简单地使用极大似然原理，这意味着使用 **训练数据和模型预测间的交叉熵** 作为损失函数。
2. 有时使用一个更简单的方法，不是预测 `$y$` 的完整概率分布，
   而是仅仅预测 **在给定 `$x$` 的条件下，`$y$` 的某种统计量**。
3. 用于训练神经网络的完整的损失函数通常是在基本的机器学习损失函数的基础上结合一个正则项。

一般来说，监督学习的目标函数由损失函数和正则化项组成，

`$$Objective = Loss + Regularization$$`

## 回归

### 均方误差损失函数

> 均方误差损失函数，Mean Squared Error，MSE

若 `$p_{model}(y|x) = N(y;f(x;\theta)，I)$`：

`$$J(\theta)=\frac{1}{2}E_{x,y \sim \hat{p}_{data}} \parallel y-f(x;\theta)\parallel^{2} + const$$`
`$$E=\frac{1}{2} \sum_{k}(y_k-t_k)^2$$`

其中：

- `$y_k$`：神经网路的输出
- `$t_k$`：正确的目标变量
- `$k$`：数据的维数

## 二分类

### 交叉熵损失函数

> 交叉熵误差损失函数，Cross Entropy Error，CEE

`$$L = -\frac{1}{n}\sum_{i=0}^{n}\Big(y_i\log a_i + (1-y_i) \log (1-a_i)\Big)$$`
`$$E=-\sum_{k}t_k\log y_k$$`

其中：

- `$y_k$`：神经网路的输出
- `$t_k$`：正确的解标签，`$\{0，1\}$`
- `$k$`：数据的维数

### 二元交叉熵损失

> 二元交叉熵损失，Binary Cross Entropy Erro，BCE

`$$L(y,\hat{y}) = - \frac{1}{N}\sum_{i=0}^{N-1} (y_i log \hat{y_i} + (1-y_i) log(1-\hat{y_i}))$$`

推导过程如下：该公式由极大似然原理推导得来。

由于 `$\hat{y_i}$` 表示的是样本标签为 1 的概率，`$(1-\hat{y_i})$` 表示的是样本标签为 0 的概率，
那么训练集中的全部样本取得对应标签的概率即似然函数可以写成如下形式：

`$$L(y,\hat{y}) = \prod_{i=0}^{N-1} \hat{y_i}^{y_i} (1-\hat{y_i})^{(1-y_i)}$$`

其中：

* 当 `$y_i = 1$` 为时，连乘中的项为 `$\hat{y_i}$`，
* 当 `$y_i = 0$` 为时，连乘中的项为 `$(1-\hat{y_i})$`

转换成对数似然函数，得到 

`$$ln L(y,\hat{y}) = \sum_{i=0}^{N-1} y_i ln{\hat{y_i}} + (1-y_i)ln{(1-\hat{y_i})}$$`

对数似然函数求极大值，等价于对对数似然函数的负数求极小值，
考虑样本数量维度归一化，于是得到了二元交叉熵损失函数的形式。

## 多分类

### 多元交叉熵损失

> 多元交叉熵损失函数，Multiple Cross Entropy Error

多元交叉熵是二元交叉熵的自然拓展：

`$$L(y,\hat{y}) = - \frac{1}{N}\sum_{i=0}^{N-1} \sum_{k=0}^{K-1} I(y_i==k) log \hat{y}_{i,k}$$`

其中：

* `$I(x)$` 是指示函数，，`$x=True$` 时，`$I(x)=1$`，`$x=False$` 时，`$I(x)=0$`
* `$y_{i}$` 取 `$\{0,K-1\}$` 其中的一个类别编码序号
* `$\hat{y}_{i}$` 是一个长度为 `$K$` 的概率向量

多元交叉熵的类别数 `$K$` 取 2 时即可得到二元交叉熵对应的公式。

### 对数损失

> 对数损失函数，Log Loss

`$$L(y,\hat{y}) = - \frac{1}{N}\sum_{i=0}^{N-1}  log(\hat{y_{i}}[y_i])$$`

容易证明，对数损失函数与交叉熵函数完全等价，是交叉熵的另外一种视角： 
即每个样本对其标签对应类别的预测概率值求对数，求平均再取负数即可。

### 负对数似然损失

负对数似然损失，Negative Log Likelihood Loss, NLLLoss

`$$L(y,\hat{y}) = - \frac{1}{N}\sum_{i=0}^{N-1}y_{i}[y_{i}]$$`

注意：这里的 `$\hat{y}$` 实际上不是概率值，而是概率值取了对数

### KL 散度

KL 散度也叫做相对熵，可以衡量两个概率分布之间的差异。
KL 散度的计算公式是交叉熵减去信息熵，注意 KL 散度是不对称的，
即 `$KL(P,Q) \neq KL(Q,P)$`，所以不能叫做 KL 距离。

两个随机变量 `$P$` 和 `$Q$` 之间的 KL 散度定义如下：

`$$KL(P，Q) = \sum_{k=0}^{K-1}p_{k}ln\bigg(\frac{p_{k}}{q_{k}}\bigg) = \sum_{k=0}^{K-1}p_{k}(ln p_{k} - ln q_{k})$$`

对二分类情况：

`$$KL(Y,\hat{Y}) = - \frac{1}{N}\sum_{i=0}^{N-1} \Big(y_i log \hat{y_i} + (1-y_i) log(1-\hat{y_i})\Big) \\
                   + \frac{1}{N}\sum_{i=0}^{N-1} \Big(y_i log y_i + (1-y_i) log(1- y_i)\Big)$$`

在 `$y_{i}$` 取 0 或 1 的情况下，信息熵部分为 0，所以 KL 散度就等于交叉熵，但是在一些情况下，
例如使用标签平滑处理技术后，`$y_{i}$` 的取值不是 0 或 1，这时候，
KL 散度相当于在交叉熵的基础上减去了一个常数，KL 散度作为损失函数去优化模型的效果和交叉熵是完全一样的，
但是在这种情况下当模型完美拟合标签的情况下 KL 散度的最小值可取到 0，
而此时交叉熵能够取到的最小值是信息熵不为 0，所以这种情况下使用 KL 散度更符合我们对损失函数的一般认识。

### 余弦相似度

* TODO

### 自适应对数 Softmax 损失

* TODO

# 损失函数的优化

## 极大似然学习条件分布

大多数现代的神经网络使用极大似然来训练，这意味着损失函数就是**负的对数似然**，
它与训练数据和模型分布间的交叉熵等价，这个损失函数表示为：

`$$J(\theta)=E_{x,y \sim \hat{p}_{data}} \log p_{model}(y|x)$$`

损失函数的具体形式随着模型而改变，取决于 `$\log p_{model}$` 的形式。
使用极大似然来导出损失函数的方法的一个优势是，它减轻了为每个模型设计损失函数的负担。
明确一个模型 `$p(y|x)$` 则自动地确定了一个损失函数 `$\log p(y|x)$`。

贯穿神经网络设计的一个反复出现的主题是损失函数的梯度必须足够的大和具有足够的预测性，
来为学习算法提供一个好的指引。饱和(变得非常平)的的函数破坏了这一目标，
因为它们把梯度变得非常小，这在很多情况下都会发生，
因为用于产生隐藏单元或者输出单元的输出激活函数会饱和，
**负的对数似然**帮助在很多模型中避免这个问题。

## 梯度下降算法

对于交叉熵损失函数的优化，通常采用基于 **梯度下降(Gradient Descent)** 的算法对其进行优化迭代求解。

除了原始的梯度下降法之外，根据一次优化所需要的样本量的不同又可分为：

* **随机梯度下降(Stoictic Gradient Descent)**
* **小批量梯度下降(mini-batch Gradient Descent)**

之后又引入了 **带有历史梯度加权的带动量(momentum)的梯度下降法**、
**Rmsprop** 以及声名远扬的 **Adam 算法** 等等。

## mini-batch 学习

机器学习使用训练数据进行学习，严格来说就是针对训练数据计算损失函数的值，找出使该损失函数的值最小的参数。
因此，计算损失函数时必须将所有的训练数据作为对象。针对所有训练数据的损失函数如下：

* 均方误差：`$E = \frac{1}{2}\sum_n \sum_k (y_{nk} - t_{nk})^2$`，`$n$` 为训练数据的个数
* 交叉熵：`$E = -\frac{1}{n}\sum_n \sum_k t_{nk}\log y_{nk}$`，`$n$` 为训练数据的个数

通过对所有训练数据的损失函数除以 `$n$`，可以求得单个数据的“平均损失函数”，通过这样的平均化，
可以获得和训练数据的数量无关的统一指标。

在训练数据数量比较大时，如果以全部数据为对象求损失函数，计算过程需要花费较长的时间，
且对计算机空间的要求也会比较高。从全部数据中选出一部分，作为全部数据的“近似”，
对小部分数据进行学习，叫做 mini-batch 学习

<!-- ## mini-batch 交叉熵

```python
def cross_entropy_error(y，t):
   if y.ndim == 1:
      t = t.reshape(1，t.size)
      y = y.reshape(1，y.size)

   batch_size = y.shape[0]
   return -np.sum(t * np.log(y + 1e-7)) / batch_size
```

当监督数据是标签形式时：

```python
def cross_entropy_error(y，t):
   if y.ndim == 1:
      t = t.reshape(1，t.size)
      y = y.reshape(1，y.size)

   batch_size = y.shape[0]
   return -np.sum(np.log(y[np.arange(batch_size)，t] + 1e-7)) / batch_size
``` -->
