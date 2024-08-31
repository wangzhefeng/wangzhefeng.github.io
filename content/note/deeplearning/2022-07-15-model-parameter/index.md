---
title: 模型调参
author: 王哲峰
date: '2022-07-15'
slug: model-parameter
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

- [参数初始化](#参数初始化)
  - [Xavier初始值](#xavier初始值)
  - [He 初始值](#he-初始值)
  - [结论](#结论)
- [PyTorch 参数初始化](#pytorch-参数初始化)
  - [常用分布初始化](#常用分布初始化)
  - [常数初始值](#常数初始值)
  - [Glorot 初始化](#glorot-初始化)
  - [He 初始化](#he-初始化)
  - [狄利克雷初始化](#狄利克雷初始化)
  - [其他](#其他)
  - [稀疏初始化](#稀疏初始化)
- [TensorFlow 参数初始化](#tensorflow-参数初始化)
  - [Initializers 的使用方法](#initializers-的使用方法)
  - [常数初始化器](#常数初始化器)
  - [分布初始化器](#分布初始化器)
  - [矩阵初始化器](#矩阵初始化器)
  - [LeCun 分布初始化器](#lecun-分布初始化器)
  - [Glorot 分布初始化器](#glorot-分布初始化器)
  - [He 正态分布和均匀分布初始化器](#he-正态分布和均匀分布初始化器)
  - [自定义初始化器](#自定义初始化器)
- [超参数的调优](#超参数的调优)
- [参考](#参考)
</p></details><p></p>

# 参数初始化

在神经网络的学习中，权重 `$W$` 的的初始值特别重要。
设定什么样的权重初始值，经常关系到神经网络的学习能否成功。

不能将权重初始值全部设为 0，因为在误差反向传播中，所有权重都会进行相同的更新。
比如：在两层神经网络中，假设第 1 层和第 2 层的权重为 0，这样一来，正向传播时，
因为输入层的权重是 0，所有第 2 层的权重神经元全部会被传递相同的值。
第 2 层的神经元中全部输入相同的值，这意味着反向传播时第 2 层的权重全部都会进行相同的更新，
因此，权重被更新为相同的值，并拥有了对称的值，这使得神经网络拥有许多不同的权重的意义就丧失了。
为了防止“权重均一化”，必须随机生成初始值

梯度消失(gradient vanishing)：

* 当使用 Sigmoid 函数作为激活函数时，激活层的激活值呈偏向 0 和 1 的分布，随着输出不断靠近 0 或 1，
  它导数的值逐渐接近 0，因此，偏向 0 和 1 的数据分布会造成反向传播中梯度的值不断减小，
  最后消失。层次加深的深度学习中，梯度消失的问题更加严重。

表现力受限：

* 如果有多个神经元都输出几乎相同的值，那他们就没有存在的意义了。
  比如，如果 100 个神经元都输出几乎相同的值，那么也可以由1个神经元来表示基本相同的事情。
  因此，激活值在分布上有所偏向。

各层的激活函值的分布都要求有适当的广度：

* 因为通过在各层间传递多样性的数据，神经网络可以进行高效的学习，
  反之，如果传递的是有所偏向的数据，就会出现梯度消失或者“表现力受限”的问题，
  导致学习无法顺利进行。

## Xavier初始值

> Xavier 初始值

在 Xavier Glorot 等人的论文中，推荐了权重初始值，俗称 “Xavier初始值”。
在一般的深度学习框架中，Xavier 初始值已经被作为标准使用。

Xavier 的论文中，为了使各层的激活值呈现出具有相同广度的分布，推导了合适的权重尺度。结论就是：
如果前一层的节点数为 `$n$`，则初始值使用标准差为 `$\frac{1}{\sqrt{n}}$` 的分布。

## He 初始值

> He 初始值

Xavier 初始值是以**激活函数是线性函数**为前提推导出来的，
因为 Sigmoid 函数和 tanh 函数左右对称，
且中央附近可以视作线性函数，所以适合使用 Xavier 初始值。

当激活函数使用 ReLU 函数时，一般推荐 ReLU 专用的初始值，
Kaiming He 等人推荐了一种初始值，俗称 “He 初始值”。
结论就是：如果前一层的节点数为 `$n$`，则初始值使用标准差为 `$\sqrt{\frac{2}{n}}$` 的高斯分布。

## 结论

* 当激活函数使用 ReLU 时，权重初始值使用 **He 初始值**
* 当激活函数为 Sigmoid 或 tanh 等 S 型函数时，初始值使用 **Xavier 初始值**

# PyTorch 参数初始化

`torch.nn.init` 模块中的所有函数旨在用于初始化神经网络参数，
因此它们都在 `torch.no_grad()` 模式中运行，不会被 `autograd` 考虑在内。

* `torch.nn.init.calculate_gain(nonlinearity, param = None)`
    - 给定非线性函数的推荐增益值

## 常用分布初始化

* `torch.nn.init.uniform_(tensor, a = 0.0, b = 1.0)`
    - 从均匀分布 `$U(a, b)$` 中抽样填充输入张量的值
* `torch.nn.init.normal_(tensor, mean = 0.0, std = 1.0)`
    - 从正态分布 `$N(\mu, \sigma^{2})$` 中抽样填充输入张量的值

## 常数初始值

* `torch.nn.init.constant_(tensor, val)`
* `torch.nn.init.ones_(tensor)`
* `torch.nn.init.zeros_(tensor)`
* `torch.nn.init.eye_(tensor)`

## Glorot 初始化

* `torch.nn.init.xavier_uniform_(tensor, gain = 0.1)`
* `torch.nn.init.xavier_normal_(tensor, gain = 0.1)`

## He 初始化

* `torch.nn.init.kaiming_uniform_(
    tensor, 
    a = 0, 
    mode = "fan_in", 
    nonlinearity = "leaky_relu"
  )`
* `torch.nn.init.kaiming_normal_(
    tensor, 
    a = 0, 
    mode = "fan_in", 
    nonlinearity = "leaky_relu"
  )`

```python
import torch.nn as nn

w = torch.empty(3, 5)
nn.init.kaiming_uniform_(w)
```

## 狄利克雷初始化

Fills the {3, 4, 5}-dimensional input Tensor with the Dirac delta function. 
Preserves the identity of the inputs in Convolutional layers, 
where as many input channels are preserved as possible. 
In case of groups>1, each group of channels preserves identity

## 其他

* `torch.nn.init.dirac_(tensor, groups = 1)`
* `torch.nn.init.trunc_normal_(tensor, mean = 0.0, std = 1.0, a = -2.0, b = 2.0)`
    - Fills the input Tensor with values drawn from a truncated normal distribution
* `torch.nn.init.orthogonal_(tensor, gain = 1)`
    - Fills the input Tensor with a (semi) orthogonal matrix
    - The input tensor must have at least 2 dimensions, 
      and for tensors with more than 2 dimensions the trailing dimensions are flattened

## 稀疏初始化

* `torch.nn.init.sparse_(tensor, sparsity, std = 0.01)`
    - Fills the 2D input Tensor as a sparse matrix, 
      where the non-zero elements will be drawn from 
      the normal distribution `$N(0,0.01)$`

# TensorFlow 参数初始化

## Initializers 的使用方法

初始化定义了设置 Keras Layer 权重随机初始的方法

- `kernel_initializer` 参数
   - `"random_uniform"`
- `bias_initializer` 参数

## 常数初始化器

- `tf.keras.initializers.Initializer()`
    - 基类
- `tf.keras.initializers.Zeros()`
    - `0`
- `tf.keras.initializers.Ones()`
    - `1`
- `tf.keras.initializers.Constant()`
    - `tf.keras.initializers.Constant(value = 0)`
        - `0`
    - `tf.keras.initializers.Constant(value = 1)`
        - `1`

## 分布初始化器

`tf.keras.initializers.RandomNormal(mean = 0.0, stddev = 0.05, seed = None)`
   - 正态分布
`tf.keras.initializers.RandomUniform(minval = 0.05, maxval = 0.05, seed = None)`
   - 均匀分布
`tf.keras.initializers.TruncatedNormal(mean = 0.0, stddev = 0.05, seed = None)`
    - 截尾正态分布:生成的随机值与 `RandomNormal` 生成的类似, 
      但是在距离平均值两个标准差之外的随机值将被丢弃并重新生成。
      这是用来生成神经网络权重和滤波器的推荐初始化器

## 矩阵初始化器

`tf.keras.initializers.VarianveScaling(scale = 1.0, mode = "fan_in", distribution = "normal", seed = None)`
    - 根据权值的尺寸调整其规模
`tf.keras.initializers.Orthogonal(gain = 1.0, seed = None)`
    - [随机正交矩阵](http://arxiv.org/abs/1312.6120)
`tf.keras.initializers.Identity(gain = 1.0)`
    - 生成单位矩阵的初始化器。仅用于 2D 方阵

## LeCun 分布初始化器

`tf.keras.initializers.lecun_normal()`

* LeCun 正态分布初始化器
* 它从以 0 为中心, 标准差为 `$stddev = \sqrt{\frac{1}{fanin}}$` 的截断正态分布中抽取样本,  
  其中 `fanin` 是权值张量中的输入单位的数量

`tf.keras.initializers.lecun_uniform()`

- LeCun 均匀初始化器
- 它从 `$[-limit, limit]$` 中的均匀分布中抽取样本,  其中 limit 是 `$\sqrt{\frac{3}{fanin}}$`，
  其中 `fanin` 是权值张量中的输入单位的数量

## Glorot 分布初始化器

`tf.keras.initializers.glorot_normal()`

- Glorot 正态分布初始化器, 也称为 Xavier 正态分布初始化器
- 它从以 0 为中心, 标准差为 `$stddev = \sqrt{\frac{2}{fanin + fanout}}$` 的截断正态分布中抽取样本, 
  其中 `fanin` 是权值张量中的输入单位的数量, `fanout` 是权值张量中的输出单位的数量

`tf.keras.initializers.glorot_uniform()`

- Glorot 均匀分布初始化器, 也称为 Xavier 均匀分布初始化器
- 它从 `$[-limit, limit]$` 中的均匀分布中抽取样本,  其中 limit 是 `$\sqrt{\frac{6}{fanin + fanout}}$`， 
  `fanin` 是权值张量中的输入单位的数量, `fanout` 是权值张量中的输出单位的数量

## He 正态分布和均匀分布初始化器

`tf.keras.initializers.he_normal()`: He 正态分布初始化器

* 从以 0 为中心, 标准差为 `$stddev = \sqrt{\frac{2}{fanin}}$` 的截断正态分布中抽取样本, 
  其中 `fanin` 是权值张量中的输入单位的数量

`tf.keras.initializers.he_uniform()`: He 均匀分布方差缩放初始化器

* 它从 `$[-limit, limit]$` 中的均匀分布中抽取样本,  其中 `limit` 是 `$\sqrt{\frac{6}{fan\_in}}$`，
 其中 `fan_in` 是权值张量中的输入单位的数量

## 自定义初始化器


# 超参数的调优

神经网络中的超参数是指，各层的神经元数量、batch 大小、参数更新时的学习率、权值衰减参数(正则化参数)等

* 不能使用测试数据(test data)评估超参数的性能。调整超参数时，
  必须使用超参数专用的确认数据，用于调整超参数的数据一般称为验证数据(validation data)。
* 模型训练数据的使用:
    - 训练数据用于参数(权重和偏置)的学习
    - 验证数据用于超参数的性能评估
    - 测试数据确认泛化能力，要在最后使用(比较理想的是只用一次)

# 参考

* [torch.nn.init Doc](https://pytorch.org/docs/stable/nn.init.html#)
