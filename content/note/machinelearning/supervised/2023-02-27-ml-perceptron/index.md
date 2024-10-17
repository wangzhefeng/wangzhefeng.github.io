---
title: 感知机
author: wangzf
date: '2023-02-27'
slug: ml-perceptron
categories:
  - machinelearning
tags:
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
img {
    pointer-events: none;
}
</style>

<details><summary>目录</summary><p>

- [感知机模型介绍](#感知机模型介绍)
  - [感知机模型](#感知机模型)
  - [感知机训练](#感知机训练)
- [单层感知机的局限性](#单层感知机的局限性)
- [多层感知机可以实现异或门](#多层感知机可以实现异或门)
- [从感知机到神经网络](#从感知机到神经网络)
</p></details><p></p>

# 感知机模型介绍

## 感知机模型

**感知机, perceptron** 是由美国学者 Frank Rosenblatt 在 1957 年提出来的, 
感知机是神经网络(深度学习)的起源算法. 
因此, 学习感知机的构造也就是学习通向神经网络和深度学习的一种重要思想, 
感知机是神经网络的理论基础. 

感知机就是一个通过建立一个线性超平面, 对线性可分的数据集进行分类的线性模型. 
感知机接收多个输入信号, 输出一个信号.

假设 `$x_1, x_2, \cdots, x_p$` 是输入信号, `$\hat{y}$` 是输出信号,
`$w_1,w_2,\cdots, w_p$` 是权重, `$b$` 是偏置.输入信号被送往神经元时,
会被分别乘以固定的权重 `$(w_1x_1,w_2x_2,\cdots,w_px_p)$`.
神经元会计算传送过来的信号的总和, 只有当这个总和超过了某个界限值时, 才会输出 1.
这也被称为"神经元被激活”. 这里将这个界限值称为阈值, 用符号 `$\theta$` 表示.

感知机的多个输入信号都有各自固定的权重, 这些权重发挥着重要控制各个信号的重要性的作用.
也就是说,权重越大, 对应该权重的信号的重要性就越高:
 
`$$\hat{y}=\sigma\Big(\sum_{i=1}^{p} \omega_i x_i + b\Big)$$`

`$$\begin{cases}
y = 1, \text{if} (\sum_{i=1}^{p} \omega_{i} x_{i} + b) > \theta \\
y = 0, \text{if} (\sum_{i=1}^{p} \omega_{i} x_{i} + b) \leq \theta
\end{cases}$$`

感知机就是一个二分类线性分类器，其目的是从特征学习出一个分类模型 `$f(\cdot)$`：

`$$y=f(z), y \in \{0, 1\}$$`

感知机模型是将特征变量的线性组合作为自变量：

`$$z=\omega^{T}x + b$$`

由于自变量 `$x$` 取值的范围是 `$[-\infty, +\infty]$`，
因此需要使用阶跃函数(Step 函数)将线性模型 `$z=\omega^{T}x + b$` 映射到集合 `$\{0, 1\}$` 上

阶跃函数(step function) `$f(\cdot)$` 如下：

`$$f(z) = \begin{cases}
1 & & z \geq 0 \\
0 & & z < 0
\end{cases}$$`

感知机模型的目标就是从数据中学习得到 `$\omega, b$`，
使得正例 `$y=1$` 的特征 `$\omega^{T}x+b$` 远大于 `$0$`，
负例 `$y=0$` 的特征 `$\omega^{T}x + b$` 远小于 `$0$`


## 感知机训练

- 首先, 模型接受输入 `$x_{i}$` ,将输入 `$x_{i}$` 与权重(weights) `$\omega_i$` 
  和偏置(bias) `$b$` 进行加权求和 `$\sum_{i=1}^{p} \omega_i x_i + b$`, 并经过 
  `$\sigma(\cdot)$` 函数进行激活,将激活结果作为 `$\hat{y}$` 进行输出. 
  这便是感知机执行前向传播计算的基本过程;
- 其次, 当执行完前向传播计算得到输出 `$\hat{y}$` 之后, 模型需要根据输出 `$\hat{y}$` 和实际的样本标签(sample label)
  `$y$` 按照损失函数 `$L(y, \hat{y})$` 计算当前损失;
- 最后, 通过计算损失函数 `$L(y, \hat{y})$` 关于权重(weights) `$\omega_i$` 和偏置(bias) `$b$`
  的梯度, 根据梯度下降算法更新权重和偏置. 经过不断的迭代调整权重和偏置使得损失最小, 这便是完整的
  **单层感知机** 的训练过程.

感知机的学习就是寻找一个超平面能够将特征空间中的两个类别的数据分开，
即确定感知机模型参数 `$\omega, b$`, 所以需要定义学习的损失函数并将损失函数最小化

1. 定义学习损失函数

`$$L(\omega, b)=-\frac{1}{||\omega||}\sum_{x_i \in M}y_i(\omega^{T} x_i + b) \\
=-\sum_{x_i \in M}y_i(\omega^{T} x_i + b)$$`

其中: 

* 集合 `$M$` 是超平面 `$S$` 的误分类点集合

损失函数的意义是：误分类点到超平面的 `$S$` 的距离总和

2. 感知机学习算法

> 随机梯度下降算法(Stochastic gradient descent)

最优化问题: 

`$$\omega, b= argmin L(\omega, b)=-\sum_{x_i \in M}y_i(\omega^{T} x_i + b)$$`

算法: 

1. 选取初始值: `$\omega_0, b_0$`
2. 在训练数据中选取数据点 `$(x_i, y_i)$`
3. 如果 `$y_i(\omega\cdot x_i + b)<0$`
    - `$\omega \gets \omega + \eta y_i x_i$`
    - `$b \gets b + \eta y_i$`
4. 重新选取数据点，直到训练集中没有误分类点

# 单层感知机的局限性

- 单层感知机无法分离非线性空间
    - 单层感知机只能表示由一条直线(超平面)分割的空间
- 感知机无法实现异或门(XOR gate)
    - XOR 函数("异或"逻辑)是两个二进制值的运算, 当这些二进制值中恰好有一个为 1 时, 
      XOR 函数返回值为 1, 其余情况下为 0.

# 多层感知机可以实现异或门

- 感知机可以通过叠加层可以表示异或门, 实现非线性空间的分割:
    - **异或门** 可以通过组合 **与非门**、**或门**, 再将前两个逻辑门的组合和 **与门** 组合得到   
- 叠加了多层的感知机也称为 **多层感知机(MLP, Multi-Layered Perceptron)**

# 从感知机到神经网络

- **单层感知机** 包含两层神经元,即输入与输出神经元,可以非常容易的实现逻辑与、或和非等线性可分情形, 
  但终归而言,这样的一层感知机的学习能力是非常有限的, 对于像异或这样的非线性情形, 单层感知机就搞不定了.
  其学习过程会呈现一定程度的振荡,权值参数 `$\omega_i$` 难以稳定下来,最终不能求得合适的解;
- 对于 **非线性可分** 的情况, 在感知机的基础上一般有了两个解决方向:
    1. **支持向量机模型**
        - 支持向量机旨在通过 **核函数** 映射来处理非线性的情况;
    2. **神经网络模型**   
        - 神经网络模型也叫 **多层感知机**, 与单层的感知机在结构上的区别主要在于 MLP 多了若干 **隐藏层**, 
          这使得神经网络对非线性的情况拟合能力大大增强.

