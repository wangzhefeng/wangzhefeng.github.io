---
title: SVM
author: 王哲峰
date: '2022-09-13'
slug: ml-svm
categories:
  - machinelearning
tags:
  - machinelearning
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

- [什么是支持向量机？](#什么是支持向量机)
- [线性分类器](#线性分类器)
  - [数据集的线性可分性](#数据集的线性可分性)
  - [感知机](#感知机)
    - [感知机模型](#感知机模型)
    - [感知机模型学习](#感知机模型学习)
    - [Logistic 回归](#logistic-回归)
  - [支持向量机(形式)](#支持向量机形式)
- [函数间隔与几何间隔](#函数间隔与几何间隔)
  - [函数间隔](#函数间隔)
  - [几何间隔](#几何间隔)
- [参考](#参考)
</p></details><p></p>

- 适用于高维数据
- 不同的核函数

# 什么是支持向量机？

支持向量机是 90 年代中期发展起来的基于统计学习理论的一种 `有监督机器学习方法`, 
通过寻求 `结构化风险最小` 来提高学习器的泛化能力, 
实现 `经验风险` 和 `置信范围` 的最小化, 从而达到在统计样本量较少的情况下, 
也能获得良好的统计规律性. 

所谓支持向量机, 顾名思义, 分为两个部分了解: 

- 什么是支持向量: 支持或支撑平面上把两类类别划分开来的超平面的向量点；
- 机(machine, 机器): 一个算法；

# 线性分类器

## 数据集的线性可分性

给定一个数据集

`$$T = \{(x_1,y_1),(x_2, y_2),...,(x_n, y_n)\}$$`

其中: 

- `$x_i\in R^{n}$`
- `$y_i \in \{0, 1\}$`
- `$i=1,2, ..., n$`

如果存在某个超平面 `$S`:

`$$\omega^{T}x+b=0$$`

能够将数据集的正实例(`$y=1$`)和负实例(`$y=0$`)完全正确地分到超平面的两侧, 
即对所有的 `$y_i=1$` 的实例 `$i$`, 有 `$\omega^{T}x+b>0$`；
对所有的 `$y_i=0$` 的实例 `$i$`, 有 `$\omega^{T}x+b<0$`, 
则称数据集 `$T$` 为线性可分数据集, 否则称为线性不可分. 

一个二分类线性分类器就是要在 `$R^n$` 特征空间中找到一个超平面 `$S$`, 
其方程可以表示为: 

`$$\omega^{T}x+b = 0$$`

这个超平面将特征空间划分为两个部分, 位于两部分的点分别被分为两类. 

## 感知机

### 感知机模型

感知机就是一个二分类线性分类器, 
其目的是从特征学习出一个分类模型 `$f(\cdot)$`: `$y=f(z), y \in \{0, 1\}$`

感知机模型是将特征变量的线性组合作为自变量: 

`$$z=\omega^{T}x + b$$`

由于自变量 `$x$` 取值的范围是
`$[-\infty, +\infty]$`, 因此, 需要使用 `阶跃函数(Step函数)` 将自变量
`$z=\omega^{T}x + b$` 映射到范围 `$\{0, 1\}$` 上. 

这里 `$f(z)$` 是一个阶跃函數(step function): 

`$$f(z) = \begin{cases}
1 & & z \geq 0 \\
0 & & z < 0
\end{cases}$$`

> 感知机模型的目标就是从数据中学习得到 `$\omega, b$`, 
使得正例 `$y=1$` 的特征 `$\omega^{T}x+b$` 远大于 `$0$`, 
负例 `$y=0$` 的特征 `$\omega^{T}x + b$` 远小于 `$0$`. 

### 感知机模型学习

> 感知机的学习就是寻找一个超平面能够将特征空间中的两个类别的数据分开, 
  即确定感知机模型参数 `$\omega, b$`, 所以需要定义学习的损失函数并将损失函数最小化；

**1.定义学习损失函数:**

`$$L(\omega, b)=-\frac{1}{||\omega||}\sum_{x_i \in M}y_i(\omega^{T} x_i + b) \\
=-\sum_{x_i \in M}y_i(\omega^{T} x_i + b)$$`

其中: 

* 集合 `$M$` 是超平面 `$S$` 的误分类点集合

损失函数的意义是: 误分类点到超平面的$S$的距离总和；

**2.感知机学习算法:**

> 随机梯度下降算法(Stochastic gradient descent)

最优化问题: 

`$$\omega, b= argmin L(\omega, b)=-\sum_{x_i \in M}y_i(\omega^{T} x_i + b)$$`

算法: 

* 选取初始值: `$\omega_0, b_0$`
* 在训练数据中选取数据点 `$(x_i, y_i)$`
* 如果 `$y_i(\omega\cdot x_i + b)<0$`
   - `$\omega \gets \omega + \eta y_i x_i$`
   - `$b \gets b + \eta y_i$`
* 重新选取数据点, 直到训练集中没有误分类点；

### Logistic 回归

> Logistic Regression的目的是从特征学习出一个 0/1 分类模型 `$f(\cdot)$`: 
> `$$y = f(z), y \in \{0, 1\}$$`

Logistic Regression 模型是将特征变量的线性组合作为自变量: 

`$$z=\omega^{T}x + b$$`

由于自变量 `$x$` 取值的范围是 `$[-\infty, +\infty]$`, 因此, 
需要使用 Logistic 函数(Sigmoid 函数)将自变量 `$z=\omega^{T}x + b$` 映射到范围 `$[0, 1]$` 上, 
映射后的值被认为是 `$y=1$` 的概率. 假设: 

`$$h_{\omega,b}(x)=\sigma(\omega^{T}x + b)$$`

其中 `$\sigma(z)$` 是Sigmoid函数: 

`$$\sigma(z)=\frac{1}{1+e^{-z}}$$`

因此 Logistic Regression 模型的形式为: 

`$$\begin{cases}
P(y=1|x, \omega) = h *\{\omega, b\}(x) =\sigma(\omega^{T}x+b) \\ 
P(y=0|x, \omega) = 1 - h* \{\omega, b\}(x) =1-\sigma(\omega^{T}x+b) 
\end{cases}$$`

当要判别一个新来的数据点 `$x_{test}$` 属于哪个类别时, 
只需要求解 `$h_{\omega, b}(x_{test}) = \sigma(\omega^{T}x_{test} + b)$`: 

`$$ y_{test}=\left\{
\begin{array}{rcl}
1    &      & h_{\omega,b}(x_{test}) \geq 0.5 & \Leftrightarrow & \omega^{T}x_{test}+b \geq 0\\
0    &      & h_{\omega,b}(x_{test}) < 0.5 & \Leftrightarrow & \omega^{T}x_{test}+b < 0\\
\end{array} \right.$$`

Logistic Regression 的目标就是从数据中学习得到 `$\omega, b$`, 
使得正例 `$y=1$` 的特征 `$\omega^{T}x+b$` 远大于 `$0$`, 
负例 `$y=0$` 的特征 `$\omega^{T}x + b$` 远小于 `$0$`. 

## 支持向量机(形式)

在支持向量机中, 将符号进行变换: 

假设: 

`$$h_{\omega,b}(x)=\sigma(\omega^{T}x + b)$$`

这里, 只需考虑 `$\omega^{T}x + b$` 的正负问题, 
而不关心 `$\sigma(\cdot)$` 的形式. 
因此将 `$\sigma(\cdot)$` 进行简化, 将其简单地映射到 `$[-1, 1]$` 上: 

`$$\sigma(z)=\left\{
\begin{array}{rcl}
1     &      & z \geq 0\\
-1    &      & z < 0\\
\end{array} \right. $$`

# 函数间隔与几何间隔

> 函数间隔，Function Margin
> 几何间隔，Geometric Margin

- 一般而言, 一个数据点距离超平面( `$\omega^{T}x+b =0$`)的远近可以表示为分类预测的确信或准确程度. 
- 在超平面 `$\omega^{T}x+b$` 确定的情况下, 
  `$|\omega^{T}x+b|$` 能够相对表示点 `$x$` 距离超平面的远近；
  而 `$\omega^{T}x+b$` 的符号与类别标记 `$y$` 的符号是否一致表示分类是否正确, 
  可以用指标量 `$y\cdot (\omega^{T}x+b)$` 的正负性来判定或表示分类的正确性和确信度；

## 函数间隔

> Function Margin

函数间隔: 

`$$\hat{\gamma}=y\cdot (\omega^{T}x+b)=yf(x)$$`

超平面 `$\omega^{T}x+b$` 关于训练数据 `$T$` 的函数间隔为超平面 `$\omega^{T}x+b$` 
关于 `$T$` 中所有样本点 `$(x_i, y_i)$` 的函数间隔的最小值: 

`$$\hat{\gamma}=\min\hat{\gamma}_{i}, i = 1, 2,..., n$$`

- 上面定义的函数间隔虽然可以表示分类预测的正确性和确信度, 但在选择分类超平面时, 只有函数间隔是不够的；
- 如果成比例的改变 `$\omega$` 和 `$b$`, 
  比如将他们改变为 `$2\omega$` 和 `$2b$`, 
  虽然此时超平面没有改变, 但是函数间隔的值 `$yf(x)$` 却变成了原来的4倍. 
- 解决问题: 可以对法向量 `$\omega$` 加一些约束条件, 使其表面上看起来规范化；

## 几何间隔

> Geometric Margin



# 参考

