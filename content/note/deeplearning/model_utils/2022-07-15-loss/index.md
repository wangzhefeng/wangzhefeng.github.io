---
title: 损失函数
author: 王哲峰
date: '2022-07-15'
slug: loss
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
    - [均方误差损失函数(Mean Squared Error, MSE)](#均方误差损失函数mean-squared-error-mse)
  - [二分类](#二分类)
    - [交叉熵误差损失函数(Cross Entropy Error, CEE)](#交叉熵误差损失函数cross-entropy-error-cee)
    - [二元交叉熵损失(Binary Cross Entropy Erro, BCE)](#二元交叉熵损失binary-cross-entropy-erro-bce)
  - [多分类](#多分类)
    - [多元交叉熵](#多元交叉熵)
    - [对数损失函数](#对数损失函数)
    - [负对数似然损失](#负对数似然损失)
    - [KL 散度](#kl-散度)
    - [余弦相似度](#余弦相似度)
    - [自适应对数 Softmax 损失](#自适应对数-softmax-损失)
- [极大似然学习条件分布](#极大似然学习条件分布)
- [学习条件统计量](#学习条件统计量)
- [mini-batch 学习](#mini-batch-学习)
- [mini-batch 交叉熵](#mini-batch-交叉熵)
- [PyTorch 损失函数](#pytorch-损失函数)
  - [内置损失函数](#内置损失函数)
    - [常用内置损失函数 API](#常用内置损失函数-api)
    - [常用内置损失函数解释](#常用内置损失函数解释)
  - [创建自定义损失函数](#创建自定义损失函数)
    - [FocalLoss](#focalloss)
    - [SCELoss](#sceloss)
  - [自定义 L1 和 L2 正则化项](#自定义-l1-和-l2-正则化项)
    - [数据准备](#数据准备)
    - [定义模型](#定义模型)
    - [模型训练](#模型训练)
    - [结果可视化](#结果可视化)
    - [通过优化器实现 L2 正则化](#通过优化器实现-l2-正则化)
- [TensorFlow 损失函数](#tensorflow-损失函数)
  - [TensorFlow 内置损失函数](#tensorflow-内置损失函数)
    - [内置损失函数的两种形式](#内置损失函数的两种形式)
    - [回归损失](#回归损失)
    - [二分类损失](#二分类损失)
    - [多分类损失](#多分类损失)
    - [概率损失](#概率损失)
  - [创建自定义损失函数](#创建自定义损失函数-1)
    - [类形式的损失函数](#类形式的损失函数)
    - [函数形式的损失函数](#函数形式的损失函数)
  - [损失函数的使用—compile() \& fit()](#损失函数的使用compile--fit)
    - [通过实例化一个损失类创建损失函数](#通过实例化一个损失类创建损失函数)
    - [直接使用损失函数](#直接使用损失函数)
  - [损失函数的使用—单独使用](#损失函数的使用单独使用)
</p></details><p></p>

# 损失函数

为了使用基于梯度的学习方法, 必须选择一个损失函数, 并且必须选择如何表示模型的输出。
下面给出基于不同学习任务常用的损失函数:

| 任务 | 符号 | 名字  | 数学形式  |
|-----|------|------|----------|
| 分类 |      |      |          |
| 回归 |      |      |          |

神经网络损失函数-交叉熵损失函数(cross entropy):

1. 在大多数情况下, 参数模型定义了一个分布 `$p(y|x;\theta)$`, 
   并且简单地使用极大似然原理, 这意味着使用 **训练数据和模型预测间的交叉熵** 作为损失函数
2. 有时使用一个更简单的方法, 不是预测 `$y$` 的完整概率分布, 
   而是仅仅预测 **在给定 `$x$` 的条件下, `$y$` 的某种统计量**
3. 用于训练神经网络的完整的损失函数通常是在基本的机器学习损失函数的基础上结合一个正则项

## 回归

### 均方误差损失函数(Mean Squared Error, MSE)

若 `$p_{model}(y|x) = N(y;f(x;\theta), I)$`:

`$$J(\theta)=\frac{1}{2}E_{x,y \sim \hat{p}_{data}} \parallel y-f(x;\theta)\parallel^{2} + const$$`
`$$E=\frac{1}{2} \sum_{k}(y_k-t_k)^2$$`

其中:

- `$y_k$`: 神经网路的输出
- `$t_k$`: 正确的目标变量
- `$k$`: 数据的维数

```python
def mean_squared_error(y, t):
    error = 0.5 * np.sum((y - t) ** 2)
    return error
```

## 二分类

### 交叉熵误差损失函数(Cross Entropy Error, CEE)

`$$L = -\frac{1}{n}\sum_{i=0}^{n}(y_i\log a_i + (1-y_i \log (1-a_i)))$$`
`$$E=-\sum_{k}t_k\log y_k$$`

其中:

- `$y_k$`: 神经网路的输出
- `$t_k$`: 正确的解标签, `${0, 1}$`
- `$k$`: 数据的维数

```python
def cross_entropy_error(y, t):
    delta = 1e-7
    error = - np.sum(t * np.log(y + delta))

    return error
```

### 二元交叉熵损失(Binary Cross Entropy Erro, BCE)

`$$BinaryCrossEntropyLoss(Y,\hat{Y}) = - \frac{1}{N}\sum_{i=0}^{N-1} (y_i log \hat{y_i} + (1-y_i) log(1-\hat{y_i}))$$`

该公式由极大似然原理推导得来。由于 `$\hat{y_i}$` 表示的是样本标签为1的概率，
`$1-\hat{y_i}$` 表示的是样本标签为0的概率，
那么训练集中的全部样本取得对应标签的概率即似然函数可以写成如下形式

`$$L(Y,\hat{Y}) = \prod_{i=0}^{N-1} \hat{y_i}^{y_i} (1-\hat{y_i})^{(1-y_i)}$$`

注意当 `$y_i = 1$` 为时，连乘中的项为 `$\hat{y_i}$`，
当 `$y_i = 0$` 为时，连乘中的项为 `$(1-\hat{y_i})$`

转换成对数似然函数，得到 

`$$lnL(Y,\hat{Y}) = \sum_{i=0}^{N-1} y_i ln{\hat{y_i}} + (1-y_i)ln{(1-\hat{y_i})}$$`

对数似然函数求极大值，等价于对对数似然函数的负数求极小值，
考虑样本数量维度归一化，于是得到了二元交叉熵损失函数的形式




## 多分类

### 多元交叉熵

多元交叉熵是二元交叉熵的自然拓展

`$$CrossEntropyLoss(Y,\hat{Y}) = - \frac{1}{N}\sum_{i=0}^{N-1} \sum_{k=0}^{K-1} I(y_i==k) log \hat{y_{i,k}} \\
\text{where} I(x) \text{ is the Indicator function} \\
I(True)= 1 \text{ and } I(False) = 0$$`

其中:

* `$y_{i}$` 取 `$0 ~ K-1$` 其中的一个类别编码序号
* `$\hat{y}_{i}$` 是一个长度为 `$K$` 的概率向量

多元交叉熵的类别数 `$K$` 取 2 时即可得到二元交叉熵对应的公式


### 对数损失函数

`$$LogLoss(Y,\hat{Y}) = - \frac{1}{N}\sum_{i=0}^{N-1}  log(\hat{y_{i}}[y_i])$$`

容易证明，对数损失函数与交叉熵函数完全等价，是交叉熵的另外一种视角: 
即每个样本对其标签对应类别的预测概率值求对数，求平均再取负数即可

### 负对数似然损失

NLLLoss(Negative Log Likelihood Loss)，即负对数似然损失

数学表达式:

`$$NLLLoss(Y, \hat{Z}) = - \frac{1}{N}\sum_{i=0}^{N-1}z_{i}[y_{i}]$$`

注意的是这里的 `$\hat{Z}$` 实际上不是概率值，而是概率值取了对数

### KL 散度

KL 散度也叫做相对熵，可以衡量两个概率分布之间的差异。
KL 散度的计算公式是交叉熵减去信息熵，注意 KL 散度是不对称的，
即 `$KL(P, Q) \neq KL(Q, P)$`，所以不能够叫做 KL 距离

两个随机变量 `$P$` 和 `$Q$` 之间的 KL 散度定义如下:

`$$KL(P, Q) = \sum_{k=0}^{K-1}p_{k}ln\bigg(\frac{p_{k}}{q_{k}}\bigg) = \sum_{k=0}^{K-1}p_{k}(ln p_{k} - ln q_{k})$$`

对二分类情况:

`$$KL(Y,\hat{Y}) = - \frac{1}{N}\sum_{i=0}^{N-1} (y_i log \hat{y_i} + (1-y_i) log(1-\hat{y_i}))  + \frac{1}{N}\sum_{i=0}^{N-1} (y_i log y_i + (1-y_i) log(1- y_i))$$`

在 `$y_{i}$` 取 0 或 1 的情况下，信息熵部分为 0，所以 KL 散度就等于交叉熵，但是在一些情况下，
例如使用标签平滑处理技术后，`$y_{i}$` 的取值不是 0 或 1，
这时候，KL 散度相当于在交叉熵的基础上减去了一个常数，
KL 散度作为损失函数去优化模型的效果和交叉熵是完全一样的，
但是在这种情况下当模型完美拟合标签的情况下 KL 散度的最小值可取到 0，
而此时交叉熵能够取到的最小值是信息熵不为 0，
所以这种情况下使用 KL 散度更符合我们对 Loss 的一般认识

### 余弦相似度

### 自适应对数 Softmax 损失

# 极大似然学习条件分布

大多数现代的神经网络使用极大似然来训练, 这意味着损失函数就是负的对数似然, 
它与训练数据和模型分布间的交叉熵等价, 这个损失函数表示为:

`$$J(\theta)=E_{x,y \sim \hat{p}_{data}} \log p_{model}(y|x)$$`

损失函数的具体形式随着模型而改变, 取决于 `$\log p_{model}$` 的形式。

使用极大似然来导出损失函数的方法的一个优势是, 它减轻了为每个模型设计损失函数的负担。
明确一个模型 `$p(y|x)$` 则自动地确定了一个损失函数 `$\log p(y|x)$`

贯穿神经网络设计的一个反复出现的主题是损失函数的梯度必须足够的大和具有足够的预测性, 
来为学习算法提供一个好的指引。饱和(变得非常平)的的函数破坏了这一目标, 
因为它们把梯度变得非常小, 这在很多情况下都会发生, 
因为用于产生隐藏单元或者输出单元的输出激活函数会饱和。
负的对数似然帮助在很多模型中避免这个问题

对于交叉熵损失函数的优化, 通常采用基于 `梯度下降` 的算法框架对其进行优化迭代求解。
这其中除了原始的梯度下降法之外, 
根据一次优化所需要的样本量的不同又可分为 `随机梯度下降` 和 `小批量 (mini-batch) 梯度下降`。
之后又引入了 **带有历史梯度加权的带动量 (momentum) 的梯度下降法**、
**Rmsprop** 以及声名远扬的 **Adam 算法** 等等:

* 梯度下降(Gradient Descent)
* 随机梯度下降(Stoictic Gradient Descent)
* 小批量梯度下降(mini-batch Gradient Descent)
* 带有历史梯度加权的带动量的梯度下降法()
* Rmsprop
* Adam

# 学习条件统计量

# mini-batch 学习

* 机器学习使用训练数据进行学习, 严格来说就是针对训练数据计算损失函数的值, 找出使该损失函数的值最小的参数。
  因此, 计算损失函数时必须将所有的训练数据作为对象
* 针对所有训练数据的损失函数:
    - 均方误差: `$E = \frac{1}{2}\sum_n \sum_k (y_{nk} - t_{nk})^2$`，`$n$` 为训练数据的个数
    - 交叉熵: `$E = -\frac{1}{n}\sum_n \sum_k t_{nk}\log y_{nk}$`，`$n$` 为训练数据的个数
* 通过对所有训练数据的损失函数除以 `$n`, 可以求得单个数据的“平均损失函数”, 通过这样的平均化, 
  可以获得和训练数据的数量无关的统一指标
* 在训练数据数量比较大时, 如果以全部数据为对象求损失函数, 计算过程需要花费较长的时间, 
  且对计算机空间的要求也会比较高
* 从全部数据中选出一部分, 作为全部数据的“近似”, 对小部分数据进行学习, 叫做 mini-batch 学习

# mini-batch 交叉熵

```python
def cross_entropy_error(y, t):
   if y.ndim == 1:
      t = t.reshape(1, t.size)
      y = y.reshape(1, y.size)

   batch_size = y.shape[0]
   return -np.sum(t * np.log(y + 1e-7)) / batch_size
```

当监督数据是标签形式时:

```python
def cross_entropy_error(y, t):
   if y.ndim == 1:
      t = t.reshape(1, t.size)
      y = y.reshape(1, y.size)

   batch_size = y.shape[0]
   return -np.sum(np.log(y[np.arange(batch_size), t] + 1e-7)) / batch_size
```

# PyTorch 损失函数

一般来说，监督学习的目标函数由损失函数和正则化项组成，

`$$Objective = Loss + Regularization$$`

PyTorch 中的损失函数一般在模型训练的时候指定。PyTorch 中内置的损失函数的参数和 TensorFlow 不同:

* PyTorch：`y_pred` 在前，`y_true` 在后
* TensorFlow：`y_true` 在前，`y_pred` 在后

如果有需要，也可以自定义损失函数，自定义损失函数需要接收两个张量 `y_pred`、`y_true` 作为输入参数，
并输出一个标量作为损失函数

PyTorch 中的正则化项一般通过自定义的方式和损失函数一起添加作为目标函数。
如果仅仅使用 L2 正则化，也可以利用优化器的 `weight_decay` 参数来实现相同的效果

## 内置损失函数

PyTorch 内置损失函数一般有类的实现和函数的实现两种形式。
例如：`torch.nn.BCE` 和 `torch.nn.functional.binary_cross_entropy` 都是二元交叉熵损失函数，
但前者是类的实现形式，后者是函数的实现形式。
实际上，类的实现形式通常是调用函数的实现形式并用 `torch.nn.Module` 封装后得到的

一般常用的是类的实现形式，它们封装在 `torch.nn` 模块下，并且类名以 `Loss` 结尾

### 常用内置损失函数 API

回归:

* `torch.nn.MSELoss`
    - 均方误差损失，也叫做 L2 损失
* `torch.nn.L1Loss`
    - L1 损失，也叫做绝对误差损失
* `torch.nn.SmoothL1Loss`
    - 平滑 L1 损失，当输入在 `$[-1, 1]$` 之间时，平滑为 L2 损失，

二分类:

* `torch.nn.BCELoss`
    - 二元交叉熵损失，输入已经过 `torch.nn.Sigmoid` 激活，
      对于不平衡数据集可以用 `weights` 参数调整类别权重
* `torch.nn.BCEWithLogitsLoss`
    - 二元交叉熵损失，输入未经过 `torch.nn.Sigmoid` 激活

多分类:

* `torch.nn.CrossEntropyLoss`
    - 交叉熵损失函数
        - `y_true` 需要是一维的，是类别编码
        - `y_pred` 未经过 `torch.nn.Softmax` 激活
        - 对于不平衡数据集可以用 `weight` 参数调整类别权重
* `torch.nn.NLLLoss`
    - 负对数似然损失(The Negative Log Likelihood Loss)
    - 如果 `y_pred` 经过了 `torch.nn.LogSoftmax` 激活，
      这种方法和直接使用 `torch.nn.CrossEntropyLoss` 等价
* `torch.nn.KLDivLoss`
    - KL 散度，也叫相对熵，等于交叉熵减去信息熵
    - 要求输入经过 `torch.nn.LogSoftmax` 激活
    - 标签为概率值
* `torch.nn.CosineSimilarity`
    - 余弦相似度
* `torch.nn.AdaptiveLogSoftmaxWithLoss`
    - 一种非常适合多类别且类别分布很不均衡的损失函数，会自适应地将多个小类别合成一个 cluster

### 常用内置损失函数解释

1.二分类的交叉熵的计算公式是什么？为什么是这样一种形式

`$$BinaryCrossEntropyLoss(Y,\hat{Y}) = - \frac{1}{N}\sum_{i=0}^{N-1} (y_i log \hat{y_i} + (1-y_i) log(1-\hat{y_i}))$$`

该公式由极大似然原理推导得来。由于 `$\hat{y_i}$` 表示的是样本标签为 1 的概率，
`$1-\hat{y_i}$` 表示的是样本标签为 0 的概率，
那么训练集中的全部样本取得对应标签的概率即似然函数可以写成如下形式

`$$L(Y,\hat{Y}) = \prod_{i=0}^{N-1} \hat{y_i}^{y_i} (1-\hat{y_i})^{(1-y_i)}$$`

注意当 `$y_i = 1$` 为时，连乘中的项为 `$\hat{y_i}$`，
当 `$y_i = 0$` 为时，连乘中的项为 `$(1-\hat{y_i})$` 转换成对数似然函数，得到 

`$$lnL(Y,\hat{Y}) = \sum_{i=0}^{N-1} y_i ln{\hat{y_i}} + (1-y_i)ln{(1-\hat{y_i})}$$`

对数似然函数求极大值，等价于对对数似然函数的负数求极小值，
考虑样本数量维度归一化，于是得到了二元交叉熵损失函数的形式。

2.多元交叉熵的计算公式是什么？和二元交叉熵有什么联系?

`$$CrossEntropyLoss(Y,\hat{Y}) = - \frac{1}{N}\sum_{i=0}^{N-1} \sum_{k=0}^{K-1} I(y_i==k) log \hat{y_{i,k}} \\
\text{where} I(x) \text{ is the Indicator function} \\
I(True)= 1 \text{ and } I(False) = 0$$`

多元交叉熵是二元交叉熵的自然拓展，其中 `$y_i$` 取 `$0~K-1$` 其中的一个类别编码序号，
`$\hat{y_i}$` 是一个长度为 K 的概率向量。多元交叉熵的类别数 K 取 2 时即可得到二元交叉熵对应的公式

3.sklearn，catboost 等库中常常看到 logloss 对数损失函数，这个损失函数如何计算，和交叉熵有什么关系？

`$$LogLoss(Y,\hat{Y}) = - \frac{1}{N}\sum_{i=0}^{N-1}  log(\hat{y_{i}}[y_i])$$`

公式中的方括号和Python中的索引的用法一致，表示取 `$\hat{y_{i}}$` 的第 `$y_i$` 个元素

容易证明，对数损失函数与交叉熵函数完全等价，是交叉熵的另外一种视角: 
即每个样本对其标签对应类别的预测概率值求对数，求平均再取负数即可

4.PyTorch 中的 `nn.NLLLoss` 和 `nn.CrossEntropyLoss` 有什么区别和联系？

NLLoss 全称是 Negative Log Likelihood Loss，即负对数似然损失。其计算公式如下

`$$NLLoss(Y,\hat{Z}) = - \frac{1}{N}\sum_{i=0}^{N-1}  {z_{i}}[y_i]$$`

公式中的方括号和 Python 中的索引的用法一致，表示取 `$\hat{z_{i}}$` 的第 `$y_i$` 个元素

注意的是这里的 `$\hat{Z}$` 实际上不是概率值，而是概率值取了对数，
所以，和 LogLoss 一对比，很容易发现，`LogSoftmax` + `NLLLoss` 等价于 `Softmax` + `LogLoss`,
等价于 `Softmax` + `CrossEntropyLoss`。为了数值精度考虑，
PyTorch 中的 `nn.CrossEntropyLoss` 要求输入未经过 Softmax 激活，
所以有 `nn.LogSoftmax` + `nn.NLLLoss` 等价于 `nn.CrossEntropyLoss`

5.KL散度的计算公式是什么？有什么现实含义？和交叉熵有什么关系？

KL 散度也叫相对熵，可以衡量两个概率分布之间的差异

KL 散度的计算公式是交叉熵减去信息熵。注意 KL 散度是不对称的, 
即 `$KL(P,Q)\neq KL(Q,P)$`, 所以不能够叫做 KL 距离

两个随机变量 P 和 Q 之间的KL散度定义如下：

`$$KL(P,Q) = \sum_{k=0}^{K-1}p_k ln(\frac{p_k}{q_k}) = \sum_{k=0}^{K-1} p_k (ln{p_k} - ln{q_k})$$`

对二分类情况下，有：

`$$KL(Y,\hat{Y}) = - \frac{1}{N}\sum_{i=0}^{N-1} (y_i log \hat{y_i} + (1-y_i) log(1-\hat{y_i}))  + \frac{1}{N}\sum_{i=0}^{N-1} (y_i log y_i + (1-y_i) log(1- y_i))$$`

在 `$y_i$` 取 0 或 1 的情况下，信息熵部分为 0，所以 KL 散度就等于交叉熵，
但是在一些情况下，例如使用标签平滑处理技术后，`$y_i$` 的取值不是 0 或 1，
这时候，KL 散度相当于在交叉熵的基础上减去了一个常数，
KL 散度作为损失函数去优化模型的效果和交叉熵是完全一样的，
但是在这种情况下当模型完美拟合标签的情况下 KL 散度的最小值可取到 0，
而此时交叉熵能够取到的最小值是信息熵不为 0，
所以这种情况下使用KL散度更符合我们对 Loss 的一般认识

## 创建自定义损失函数

如果有需要，也可以自定义损失函数，自定义损失函数需要接收两个张量 `y_pred`、`y_true` 作为输入参数，
并输出一个标量作为损失函数

也可以对 `torch.nn.Module` 进行子类化，重写 `forward` 方法实现损失的计算逻辑，
从而得到损失函数的类的实现

### FocalLoss

Focal Loss 是一种对 `binary_crossentropy` 的改进损失函数的形式。
它在样本不均衡和存在较多易分类的样本时相比 `binary_crossentropy` 具有明显的优势

Focal Loss 的数学形式:

`$$\begin{split}
focal\_loss(y, p) =
\begin{cases}
-\alpha(1 - p)^{\gamma} \log(p) & \text{if } y = 1, \\\\
-(1 - \alpha)p^{\gamma} \log(1 - p) & \text{if } y = 0,
\end{cases}
\end{split}$$`

Focal Loss 只有两个可调参数，从而让模型更加聚焦在正样本和困难样本上，
这就是为什么这个损失函数叫做 Focal Loss:

* `alpha` 参数
    - 主要用于衰减负样本的权重
* `gamma` 参数
    - 主要用于衰减容易训练样本的权重

Focal Loss 介绍:

* https://zhuanlan.zhihu.com/p/80594704

Focal Loss 实现:

```python
import torch
from torch import nn


class FocalLoss(nn.Module):
    
    def __init__(self, gamma = 2.0, alpha = 0.75):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha
    
    def forward(self, y_pred, y_true):
        bce = torch.nn.BCELoss(reduction = "none")(y_pred, y_true)
        p_t = (y_true * y_pred) + ((1 - y_true) * (1 - y_pred))
        alpha_factor = y_true * self.alpha + (1 - y_true) * (1 - self.alpha)
        modulating_factor = torch.pow(1.0 - p_t, self.gamma)
        loss = torch.mean(alpha_factor * modulating_factor * bce)
        return loss

# 困难样本
y_pred_hard = torch.tensor([[0.5], [0.5]])
y_true_hard = torch.tensor([[1.0], [0.0]])
# 容易样本
y_pred_easy = torch.tensor([[0.9], [0.1]])
y_true_easy = torch.tensor([[1.0], [0.0]])

focal_loss = FocalLoss()
bce_loss = nn.BCELoss()

print("focal_loss(easy samples):", focal_loss(y_pred_easy, y_true_easy))
print("bce_loss(easy samples):", bce_loss(y_pred_easy, y_true_easy))

print("focal_loss(hard samples):", focal_loss(y_pred_hard, y_true_hard))
print("bce_loss(hard samples):", bce_loss(y_pred_hard, y_true_hard))
```

```
focal_loss(easy samples): tensor(0.0005)
bce_loss(easy samples): tensor(0.1054)
focal_loss(hard samples): tensor(0.0866)
bce_loss(hard samples): tensor(0.6931)
```

可见 focal_loss 让容易样本的权重衰减到原来的 0.0005/0.1054 = 0.00474。
而让困难样本的权重只衰减到原来的 0.0866/0.6931=0.12496。
因此相对而言，focal_loss 可以衰减容易样本的权重

### SCELoss

SCELoss(Symmetric Cross Entropy Loss) 也是一种对交叉熵损失的改进损失，
主要用在标签中存在明显噪声的场景

SCELoss 的数学形式:

`$$sce\_loss(y, p) = \alpha \text{ } ce\_loss(y, p) + \beta \text{ } rce\_loss(y, p)$$`

其中：

* `$ce\_loss(y, p) = -y log(p) - (1 - y) log(1 - p)$`
* `$rce\_loss(y, p) = -p log(y) - (1 - p) log(1 - y)$`
* `$rce\_loss(y, p) = ce\_loss(p, y)$`

SCELoss 介绍：

* https://zhuanlan.zhihu.com/p/420827592
* https://zhuanlan.zhihu.com/p/420913134

SCELoss 实现:

```python
import torch
from torch import nn
import torch.nn.functional as F

class SCELoss(nn.Module):

    def __init__(self, num_classes = 10, a = 1, b = 1):
        super(SCELoss, self).__init__()
        self.num_classes = num_classes
        self.a = a
        self.b = b
        self.cross_entropy = nn.CrossEntropyLoss()
    
    def forward(self, pred, labels):
        # CE 部分，正常的交叉熵损失
        ce = self.cross_entropy(pred, labels)
        # RCE
        pred = F.softmax(pred, dim = 1)
        pred = torch.clamp(pred, min = 1e-4, max = 1.0)
        label_one_hot = F.one_hot(labels, self.num_classes).float().to(pred.device)
        label_one_hot = torch.clamp(label_one_hot, min = 1e-4, max = 1.0)
        rce = (-1 * torch.sum(pred * torch.log(label_one_hot), dim = 1))
        # Loss
        loss = self.a * ce + self.b * rce.mean()
        return loss
```

## 自定义 L1 和 L2 正则化项

通常认为 L1 正则化可以产生稀疏权值矩阵，即产生一个稀疏模型，可以用于特征选择。
而 L2 正则化可以防止过拟合。一定程度上，L1 也可以防止过拟合

### 数据准备

```python
import numpy as np 
import pandas as pd 
from matplotlib import pyplot as plt
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import Dataset,DataLoader,TensorDataset
import torchkeras 
%matplotlib inline
%config InlineBackend.figure_format = 'svg'

#正负样本数量
n_positive,n_negative = 1000,6000

#生成正样本, 小圆环分布
r_p = 5.0 + torch.normal(0.0,1.0,size = [n_positive,1]) 
theta_p = 2*np.pi*torch.rand([n_positive,1])
Xp = torch.cat([r_p*torch.cos(theta_p),r_p*torch.sin(theta_p)],axis = 1)
Yp = torch.ones_like(r_p)

#生成负样本, 大圆环分布
r_n = 8.0 + torch.normal(0.0,1.0,size = [n_negative,1]) 
theta_n = 2*np.pi*torch.rand([n_negative,1])
Xn = torch.cat([r_n*torch.cos(theta_n),r_n*torch.sin(theta_n)],axis = 1)
Yn = torch.zeros_like(r_n)

#汇总样本
X = torch.cat([Xp,Xn],axis = 0)
Y = torch.cat([Yp,Yn],axis = 0)


#可视化
plt.figure(figsize = (6,6))
plt.scatter(Xp[:,0],Xp[:,1],c = "r")
plt.scatter(Xn[:,0],Xn[:,1],c = "g")
plt.legend(["positive","negative"]);


ds = TensorDataset(X,Y)

ds_train,ds_val = torch.utils.data.random_split(ds,[int(len(ds)*0.7),len(ds)-int(len(ds)*0.7)])
dl_train = DataLoader(ds_train,batch_size = 100,shuffle=True,num_workers=2)
dl_val = DataLoader(ds_val,batch_size = 100,num_workers=2)

features,labels = next(iter(dl_train))
```


### 定义模型

```python
class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(2,4)
        self.fc2 = nn.Linear(4,8) 
        self.fc3 = nn.Linear(8,1)
        
    def forward(self,x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        y = self.fc3(x)
        return y
        
net = Net() 

from torchkeras import summary

summary(net,features);
```

### 模型训练

```python
# L2正则化
def L2Loss(model,alpha):
    l2_loss = torch.tensor(0.0, requires_grad=True)
    for name, param in model.named_parameters():
        if 'bias' not in name: #一般不对偏置项使用正则
            l2_loss = l2_loss + (0.5 * alpha * torch.sum(torch.pow(param, 2)))
    return l2_loss

# L1正则化
def L1Loss(model,beta):
    l1_loss = torch.tensor(0.0, requires_grad=True)
    for name, param in model.named_parameters():
        if 'bias' not in name:
            l1_loss = l1_loss +  beta * torch.sum(torch.abs(param))
    return l1_loss




from torchkeras import KerasModel
from torchkeras.metrics import AUCROC 

net = Net()

# 将L2正则和L1正则添加到FocalLoss损失，一起作为目标函数
def focal_loss_with_regularization(y_pred,y_true):
    y_probs = torch.sigmoid(y_pred)
    focal = FocalLoss()(y_probs,y_true) 
    l2_loss = L2Loss(net,0.0001) #注意设置正则化项系数
    l1_loss = L1Loss(net,0.0001)
    total_loss = focal + l2_loss + l1_loss
    return total_loss


optimizer = torch.optim.Adam(net.parameters(),lr = 0.002)
model = KerasModel(net=net,
                   loss_fn = focal_loss_with_regularization ,
                   metrics_dict = {"auc":AUCROC()},
                   optimizer= optimizer )


dfhistory = model.fit(train_data=dl_train,
      val_data=dl_val,
      epochs=20,
      ckpt_path='checkpoint.pt',
      patience=3,
      monitor='val_auc',
      mode='max')
```

### 结果可视化

```python
# 结果可视化
fig, (ax1,ax2) = plt.subplots(nrows=1,ncols=2,figsize = (12,5))
ax1.scatter(Xp[:,0],Xp[:,1], c="r")
ax1.scatter(Xn[:,0],Xn[:,1],c = "g")
ax1.legend(["positive","negative"]);
ax1.set_title("y_true");

Xp_pred = X[torch.squeeze(torch.sigmoid(net.forward(X))>=0.5)]
Xn_pred = X[torch.squeeze(torch.sigmoid(net.forward(X))<0.5)]

ax2.scatter(Xp_pred[:,0],Xp_pred[:,1],c = "r")
ax2.scatter(Xn_pred[:,0],Xn_pred[:,1],c = "g")
ax2.legend(["positive","negative"]);
ax2.set_title("y_pred");
```
 
### 通过优化器实现 L2 正则化

如果仅仅需要使用 L2 正则化，那么也可以使用优化器的 `weight_decay` 参数来实现，
`weight_decay` 参数可以设置参数在训练过程中的衰减，这和 L2 正则化的作用效果等价

PyTorch 的优化器支持一种称之为 Per-parameter options 的操作，
就是对每一个参数进行特定的学习率，权重衰减率指定

```python
weight_params = [
    param for name, param in model.named_parameters() 
    if "bias" not in name
]
bias_params = [
    param for name, param in model.named_parameters() 
    if "bias" in name
]

optimizer = torch.optim.SGD([
    {
        "params": weight_params, 
        "weight_decay": 1e-5,
    },
    {
        "params": bias_params,
        "weight_decay": 0,
    }
], lr = 1e-2, momentum = 0.9)
```

# TensorFlow 损失函数

一般来说，监督学习的目标函数由损失函数和正则化项组成，

`$$Objective = Loss + Regularization$$`

对于 Keras 模型:

* 目标函数中的正则化项一般在各层中指定
    - 例如使用 `Dense` 的 `kernel_regularizer` 和 `bias_regularizer` 等参数指定权重使用 L1 或者 L2 正则化项，
      此外还可以用 `kernel_constraint` 和 `bias_constraint` 等参数约束权重的取值范围，这也是一种正则化手段
* 损失函数在模型编译时候指定
    - 对于回归模型，通常使用的损失函数是均方损失函数 `mean_squared_error`
    - 对于二分类模型，通常使用的是二元交叉熵损失函数 `binary_crossentropy`
    - 对于多分类模型
        - 如果 `label` 是 one-hot 编码的，则使用类别交叉熵损失函数 `categorical_crossentropy`。
        - 如果 `label` 是类别序号编码的，则需要使用稀疏类别交叉熵损失函数 `sparse_categorical_crossentropy`

如果有需要，也可以自定义损失函数，自定义损失函数需要接收两个张量 `y_true`, `y_pred` 作为输入参数，
并输出一个标量作为损失函数值

## TensorFlow 内置损失函数

### 内置损失函数的两种形式

内置的损失函数一般有类的实现和函数的实现两种形式，如

* 二分类损失函数
    - `BinaryCrossentropy` 和 `binary_crossentropy`
* 二分类、多分类
    - 类别交叉熵损失函数: `CategoricalCrossentropy` 和 `categorical_crossentropy`
    - 稀疏类别交叉熵损失函数: `SparseCategoricalCrossentropy` 和 `sparse_categorical_crossentropy`

* 语法

```python
tf.keras.losses.Class(
    from_loits = False, 
    label_smoothing = 0, 
    reduction = "auto", 
    name = ""
)
```

* 示例

```python
# data
y_ture = [[0., 1.], [0., 0.]]
y_pred = [[0.6, 0.4], [0.4, 0.6]]

# reduction="auto" or "sum_over_batch_size"
bce = tf.keras.losses.BinaryCrossentropy()
bce(y_true, y_pred).numpy()

# reduction=sample_weight
bce = tf.keras.losses.BinaryCrossentropy()
bce(y_true, y_pred, sample_weight = [1, 0]).numpy()

# reduction=sum
bce = tf.keras.losses.BinaryCrossentropy(
    reduction = tf.keras.losses.Reduction.SUM
)
bce(y_true, y_pred).numpy()

# reduction=none
bce = tf.keras.losses.BinaryCrossentropy(
    reduction = tf.keras.losses.Reduction.NONE
)
bce(y_true, y_pred).numpy()
```

### 回归损失

* `MeanSquaredError` 类、`mean_squared_error` 函数，MSE
    - 均方误差损失，mse
* `Huber` 类
    - Huber 损失
    - 介于 mse 和 mae 之间，对异常值比较鲁棒，相对 mse 有一定优势
* `MeanAbsoluteError` 类、`mean_absolute_error` 函数，MAE
    - 平均绝对值误差损失，mae
* `MeanAbsolutePercentageError` 类、`mean_absolute_percentage_error` 函数
    - 平均百分比误差损失，mape

### 二分类损失

* `BinaryCrossentropy` 类、`binary_crossentropy()` 函数
    - 二元交叉熵损失
* `Hinge` 类、`hinge` 函数
    - 合页损失
    - 最著名的应用是支持向量机的损失函数

### 多分类损失

* `CategoricalCrossentropy` 类、`categorical_crossentropy()` 函数
    - 类别交叉熵
    - 要求 label 为 one-hot 编码
* `SparseCategoricalCrossentropy` 类、`sparse_categorical_crossentropy()` 函数
    - 稀疏类别交叉熵
    - 多分类
    - 要求 label 为序号编码形式
* `CosineSimilarity` 类、`cosine_similarity` 函数
    - 余弦相似度

### 概率损失

* `KLDivergence` 类、`kl_divergence()` 函数，KLD
    - 相对熵损失，KL 散度
    - 常用于最大期望算法 EM 的损失函数，两个概率分布差异的一种信息度量

## 创建自定义损失函数

如果有需要，也可以自定义损失函数，自定义损失函数需要接收两个张量 `y_true`, `y_pred` 作为输入参数，
并输出一个标量作为损失函数值

### 类形式的损失函数

自定义损失函数需要继承 `tf.keras.losses.Loss` 类, 重写 `call` 方法即可, 
输入真实值 `y_true` 和模型预测值 `y_pred`, 输出模型预测值和真实值之间通
过自定义的损失函数计算出的损失值

Focal Loss 是一种对 `binary_crossentropy` 的改进损失函数形式。
它在样本不均衡和存在较多易分类的样本时相比 `binary_crossentropy` 具有明显的优势。
它有两个可调参数，alpha 参数和 gamma 参数。其中 alpha 参数主要用于衰减负样本的权重，
gamma 参数主要用于衰减容易训练样本的权重。从而让模型更加聚焦在正样本和困难样本上。
这就是为什么这个损失函数叫做 Focal Loss，其数学表达式如下：

`$$focal\_loss(y,p) = \begin{cases}
-\alpha  (1-p)^{\gamma}\log(p) &
\text{if y = 1}\\
-(1-\alpha) p^{\gamma}\log(1-p) &
\text{if y = 0}
\end{cases} $$`

```python  
import tensorflow as tf
from tensorflow.keras import losses

class FocalLoss(losses.Loss):

    def __init__(self, 
                 gamma = 2.0, 
                 alpha = 0.75, 
                 name = "focal_loss"):
        self.gamma = gamma
        self.alpha = alpha
    
    def call(self, y_true, y_pred):
        bce = tf.losses.binary_crossentropy(y_true, y_pred)
        p_t = (y_true * y_pred) + ((1 - y_true) * (1 - y_pred))
        alpha_factor = y_true * self.alpha + (1 - y_true) * (1 - self.alpha)
        modulating_factor = tf.pow(1.0 - p_t, self.gamma)
        loss = tf.reduce_sum(
            alpha_factor * modulating_factor * bce, 
            axis = -1
        )
        return loss
```

### 函数形式的损失函数

```python
import tensorflow as tf
from tensorflow.keras import losses

def focal_loss(gamma = 2.0, alpha = 0.75):
    def focal_loss_fixed(y_true, y_pred):
        bce = tf.losses.binary_crossentropy(y_true, y_pred)
        p_t = (y_true * y_pred) + ((1- y_true) * (1 - y_pred))
        alpha_factor = y_true * alpha + (1 - y_true) * (1 - alpha)
        modulating_factor = tf.pow(1.0 - p_t, gamma)
        loss = tf.reduce_sum(
            alpha_factor * modulating_factor * bce, 
            axis = -1
        )
        
    return focal_loss_fixed
```

## 损失函数的使用—compile() & fit()

### 通过实例化一个损失类创建损失函数

* 可以传递配置参数

```python
from tensorflow import keras
from tensorflow.keras import layers, losses

# 模型构建
model = keras.Sequential()
model.add(
    layers.Dense(
        64, 
        kernel_initializer = "uniform", 
        input_shape = (10,)
    )
)
model.add(layers.Activation("softmax"))

# 模型编译
model.compile(
    loss = losses.SparseCategoricalCrossentropy(from_logits = True), 
    optimizer = "adam", 
    metrics = ["acc"]
)
```

### 直接使用损失函数

```python
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.losses import sparse_categorical_crossentropy

# 模型构建
model = keras.Sequential()
model.add(
    layers.Dense(
        64, 
        kernel_initializer = "uniform", 
        input_shape = (10,)
    )
)
model.add(layers.Activation("softmax"))

# 模型编译
model.compile(
    loss = "sparse_categorical_crossentropy", 
    optimizer = "adam", 
    metrics = ["acc"]
)
```

## 损失函数的使用—单独使用

```python
tf.keras.losses.mean_squared_error(tf.ones((2, 2)), tf.zeros((2, 2)))
loss_fn = tf.keras.losses.MeanSquaredError(resuction = "sum_over_batch_size")
loss_fn(tf.ones((2, 2)), tf.zeros((2, 2)))

loss_fn = tf.keras.losses.MeanSquaredError(reduction = "sum")
loss_fn(tf.ones((2, 2)), tf.zeros((2, 2)))

loss_fn = tf.keras.losses.MeanSquaredError(reduction = "none")
loss_fn(tf.ones((2, 2)), tf.zeros((2, 2)))

loss_fn = tf.keras.losses.mean_squared_error
loss_fn(tf.ones((2, 2,)), tf.zeros((2, 2)))

loss_fn = tf.keras.losses.MeanSquaredError()
loss_fn(tf.ones((2, 2)), tf.zeros((2, 2)))
```
