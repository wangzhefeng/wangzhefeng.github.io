---
title: 广义加性模型(GAM)
author: 王哲峰
date: '2022-09-23'
slug: model-gam
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
</style>

<details><summary>目录</summary><p>

- [广义加性模型简介](#广义加性模型简介)
- [线性回归模型](#线性回归模型)
- [广义加性模型](#广义加性模型)
  - [广义加性回归方程](#广义加性回归方程)
  - [平滑函数](#平滑函数)
    - [基函数](#基函数)
    - [多项式方程](#多项式方程)
    - [径向基函数](#径向基函数)
    - [样条函数](#样条函数)
      - [样条函数](#样条函数-1)
      - [样条回归](#样条回归)
  - [联系函数](#联系函数)
- [将一个线性回归方程建模为 GAM](#将一个线性回归方程建模为-gam)
- [GAM Python 库](#gam-python-库)
</p></details><p></p>

# 广义加性模型简介

广义加性模型(GAM)作为回归家族的一个扩展，是最强大的模型之一，可以为任何回归问题建模

线性模型简单、直观、便于理解，但是，在现实生活中，变量的作用通常不是线性的。
线性的假设很可能不能满足实际需求，甚至直接违背实际情况

1985 年 Stone 提出加性模型(additive models)，模型中每一个加性项使用单个光滑函数来估计，
在每一个加性项中可以解释因变量如何随自变量变化而变化，
解决了模型中自变量数量较多时，模型的方差会加大问题

1990 年，Hastie 和 Tibshirani 扩展了加性模型的应用范围，
提出了广义加性模型(Generalized Additive Models)

# 线性回归模型

`$$y = \omega_{1} x_{1} +  \omega_{2} x_{2} + \ldots +  \omega_{n} x_{n} + \sigma$$`

其中:

* `$x_{i}, i = 1, 2, \ldots, n$` 为自变量
* `$y$` 为因变量 
* `$\omega_{i}$` 是每个自变量函数 `$x_{i}$` 的权重，`$i = 1, 2, \ldots, n$`
* `$\sigma$` 为随机误差项

# 广义加性模型

## 广义加性回归方程

`$$g(y) = \omega_{1}F_{1}(x_{1}) + \omega_{2}F_{2}(x_{2}) + \ldots + \omega_{n}F_{n}(x_{n}) + \sigma$$`

其中:

* 自变量 `$x_{i}$` 被函数 `$F_{i}(x_{i})$` 嵌套，`$i = 1, 2, \ldots, n$`
* 因变量 `$y$` 不是 `$y$` 本身，而是一个函数 `$g(y)$`
* `$\omega_{i}$` 是每个自变量函数 `$F_{i}(x_{i})$` 的权重，`$i = 1, 2, \ldots, n$`
* `$\sigma$` 为随机误差项

## 平滑函数

广义加性模型中的 `$F_{i}, i = 1, 2, \ldots, n$` 是一组对每个自变量分别建模为目标变量的函数，
称为平滑函数(smoothing functions)，将所有这些函数加起来预测 `$g(y)$`

`$F_{i}$` 对于不同的自变量可以采用不同的表示，
对于一个自变量 `$x_{i}$`，`$F_{i}$` 可以表示为如下形式:

* 多项式方程
* 径向基函数(RBF)
* 回归样条函数(Regression Splines)[最常见]
* [Tensor](https://fda.readthedocs.io/en/latest/modules/autosummary/skfda.representation.basis.Tensor.html)

### 基函数

基函数(basic functions) 是一组可以用来表示复杂非线性函数的简单函数。

例如，有非线性函数:

`$$f(x) = 5 + 2 x^{2}$$`

为了表示这个复杂的非线性函数，可以使用基函数集: `$f_{1}(x) = 1, f_{2}(x) = x, f_{3}(x) = x^{2}$`。
因此非线性函数用基函数表示为:

`$$f(x) = 5f_{1}(x) + 0 f_{2}(x) + 2 f_{3}(x)$$`

基函数有很多种，最常见的可能就是径向基函数(RBF)，此外还有样条函数(Splines)、多项式方程等

### 多项式方程

### 径向基函数

### 样条函数

#### 样条函数

样条函数(Splines)是基函数的一种，它是由多项式分段定义的函数。
分段多项式基本上就是对变量的不同区间有不同表示的多项式

例如:

`$$g(x) = \left \{
\begin{array}{rcl}
mx + a,       &      & {x < 5} \\
mx + nx^{2},  &      & {5 < x < 10} \\
p x^{3},      &      & {x > 10} \\
\end{array} \right.$$`

根据 `$x$` 的不同区间改变多项式的表示，这样的多项式称为分段多项式。
根据样条的程度，可以有以下可能的基函数，来构造原始的复杂函数 `$g(x)$`:

* 0 阶 基函数：`$f_{1}(x) = 1$`
* 1 阶 基函数：`$f_{1}(x) = 1, f_{2}(x) = x$`
* 2 阶 基函数：`$f_{1}(x) = 1, f_{2}(x) = x, f_{3}(x) = x^{2}$`
* 3 阶 基函数：`$f_{1}(x) = 1, f_{2}(x) = x, f_{3}(x) = x^{2}, f_{4}(x) = x^{3}$`

#### 样条回归

样条回归是一组基础函数集的加权和，其中使用的基函数是样条函数

`$$F_{i}(x_{i}) = \sum_{j}\omega_{j}b_{j}(x_{i}), i = 1, 2, \ldots, n$$`

其中:

* `$F_{i}$` 是第 `$i$` 个自变量的平滑函数
* `$b_{j}$` 是样条回归的第 `$j$` 个基函数，因为样条回归由多个基函数组成

所以 GAM 方程是(如果只使用样条回归):

`$$g(y) = \sum_{k}\omega_{k}b_{k}(x_{1}) + \sum_{m}\omega_{m}b_{m}(x_{2}) + \ldots + \sum_{n}\omega_{n}b_{n}(x_{n}) + \sigma$$`

其中:

* `$k, m, n$` 是不同自变量的不同样条函数的阶

## 联系函数

如果自变量和目标变量之间的关系不是线性的，用于线性回归的线程方程就需要一些修改将目标映射到自变量，
这里的映射有可能是非线性关系，所以就需要将目标限制在某个特定范围内，也就是将 `$y$` 变为 `$g(y)$`

这里的 `$g(y)$` 称为联系函数(link function)，它的作用就是保持目标变量和自变量之间的线性关系。
正如模型的名字，“广义”这个词描述了 GAM 可以满足不同的回归场景，这些场景不需要遵循线性回归的基本假设，
所以这个 `$g(\cdot)$` 可以是任何函数

# 将一个线性回归方程建模为 GAM

GAM 方程:

`$$g(y) = \omega_{1}F_{1}(x_{1}) + \omega_{2}F_{2}(x_{2}) + \ldots + \omega_{n}F_{n}(x_{n}) + \sigma$$`

如果要将一个线性回归方程建模为 GAM，只需要:

1. 联系函数设置成恒等函数
2. `$F_{n}$` 设置成恒等函数

也就是说 

1. `$g(y) = x$`
2. `$F_{i}(x) = x, i = 1, 2, \ldots, n$`
  
所以有:

`$$y = \omega_{1}x_{1} + \omega_{2}x_{2} + \ldots + \omega_{n}x_{n} + \sigma$$`

# GAM Python 库

* [pygam](https://pygam.readthedocs.io/en/latest/notebooks/tour_of_pygam.html) 是 Python 中的 GAM 的实现

