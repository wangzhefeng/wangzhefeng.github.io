---
title: 损失函数
author: 王哲峰
date: '2022-07-15'
slug: dl-model-losses
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
</style>

<details><summary>目录</summary><p>

- [损失函数](#损失函数)
- [均方误差损失函数(Mean Squared Error, MSE)](#均方误差损失函数mean-squared-error-mse)
- [交叉熵误差损失函数(Cross Entropy Error, CEE)](#交叉熵误差损失函数cross-entropy-error-cee)
- [极大似然学习条件分布](#极大似然学习条件分布)
- [学习条件统计量](#学习条件统计量)
- [mini-batch 学习](#mini-batch-学习)
- [mini-batch 交叉熵](#mini-batch-交叉熵)
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

# 均方误差损失函数(Mean Squared Error, MSE)

若 `$p_{model}(y|x) = N(y;f(x;\theta), I)$`:

`$$J(\theta)=\frac{1}{2}E_{x,y \sim \hat{p}_{data}} ||y-f(x;\theta)||^{2} + const$$`
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

# 交叉熵误差损失函数(Cross Entropy Error, CEE)
    
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

- 梯度下降(Gradient Descent)
- 随机梯度下降(Stoictic Gradient Descent)
- 小批量梯度下降(mini-batch Gradient Descent)
- 带有历史梯度加权的带动量的梯度下降法()
- Rmsprop
- Adam

# 学习条件统计量

# mini-batch 学习

- 机器学习使用训练数据进行学习, 严格来说就是针对训练数据计算损失函数的值, 找出使该损失函数的值最小的参数。
  因此, 计算损失函数时必须将所有的训练数据作为对象
- 针对所有训练数据的损失函数:
    - 均方误差: `$E = \frac{1}{2}\sum_n \sum_k (y_{nk} - t_{nk})^2$`，`$n$` 为训练数据的个数
    - 交叉熵: `$E = -\frac{1}{n}\sum_n \sum_k t_{nk}\log y_{nk}$`，`$n$` 为训练数据的个数
- 通过对所有训练数据的损失函数除以 `$n`, 可以求得单个数据的“平均损失函数”, 通过这样的平均化, 
  可以获得和训练数据的数量无关的统一指标
- 在训练数据数量比较大时, 如果以全部数据为对象求损失函数, 计算过程需要花费较长的时间, 
  且对计算机空间的要求也会比较高
- 从全部数据中选出一部分, 作为全部数据的“近似”, 对小部分数据进行学习, 叫做 mini-batch 学习

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

