---
title: 多目标规划
author: 王哲峰
date: '2023-08-20'
slug: multi-objective-optimization
categories:
  - optimizer algorithm
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

- [多目标优化简介](#多目标优化简介)
- [多目标优化的一般形式](#多目标优化的一般形式)
- [Pareto 最优解](#pareto-最优解)
    - [向量序](#向量序)
    - [非支配解](#非支配解)
    - [Pareto 最优解](#pareto-最优解-1)
    - [总结](#总结)
- [多目标优化求解方法](#多目标优化求解方法)
    - [评价函数法](#评价函数法)
    - [评价函数法](#评价函数法-1)
        - [理想点法](#理想点法)
        - [极大极小法](#极大极小法)
        - [线性加权法](#线性加权法)
    - [目标规划法](#目标规划法)
    - [分层序列法](#分层序列法)
    - [NSGA-II](#nsga-ii)
- [参考](#参考)
</p></details><p></p>

# 多目标优化简介

多目标优化(Multi-objective Optimization Problem, MOP)也叫多目标规划，
即同时优化多个目标的规划问题。

求解多目标规划的有两种方法：一种是<span style='border-bottom:1.5px dashed red;'>合成型</span>，
将多目标转化成单目标决策问题；另一种是<span style='border-bottom:1.5px dashed red;'>分层型</span>，
在保证第一目标的情况下，尽量优化第二、第三等目标。

因此，多目标规划一般有两种方法：一种是化多为少，即将多目标转化为比较容易求解的单目标规划方法；
另一种是分层序列法，即把目标按其重要性排序，每次都在前一个目标最优解集内求解下一个目标的最优解，
直到求出共同的最优解。

如何理解目标最优解集呢？在多目标规划中往往有多个最优解同时满足约束条件，
不同的解之间不能简单通过大小来比较，这点事同单目标规划的最大区别，
多个解组成的集合称为帕累托最优解集，组成的超平面称为帕累托前沿。

# 多目标优化的一般形式

多目标规划是由多个目标函数构成的，其数学模型一般描述如下：

`$$min f_{1}(x_{1}, x_{2}, \cdots, x_{n})$$`
`$$min f_{2}(x_{1}, x_{2}, \cdots, x_{n})$$`
`$$\cdots$$`
`$$min f_{m}(x_{1}, x_{2}, \cdots, x_{n})$$`

`$$s.t.\begin{cases}
g_{i}(x_{1}, x_{2}, \cdots, x_{n}) \leq 0, i = 1, 2, \cdots, p \\
h_{j}(x_{1}, x_{2}, \cdots, x_{n}) = 0, i = 1, 2,\cdots, q
\end{cases}$$`

该规划问题有 `$m$` 个目标函数，均为求最小值问题，最大化问题可以通过乘 `$-1$` 转化成最小化问题，
所以，写成向量的形式如下：

`$$\begin{align}
Z 
&= min F(x) \\
&= [f_{1}(\mathbf{X}),f_{2}(\mathbf{X}), \cdots, f_{m}(\mathbf{X})]^{T}
\end{align}$$`

`$$s.t.\begin{cases}
g_{i}(\mathbf{X}) \leq 0, i = 1, 2, \cdots, p \\
h_{j}(\mathbf{X}) = 0, i = 1, 2,\cdots, q
\end{cases}$$`

其中：

* `$g_{i}(\mathbf{X}) \leq 0$` 为等式约束
* `$h_{j}(\mathbf{X}) = 0$` 为等式约束

# Pareto 最优解

## 向量序

多目标规划问题中的目标函数是由多个目标函数组成的向量，在优化目标函数的过程中，
如果比较迭代求解过程中前后两个解（多个目标函数值组成的向量）的优劣呢？
由此引入<span style='border-bottom:1.5px dashed red;'>向量序</span>的概念。

对于向量 `$a = (a_{1}, a_{2}, \cdots, a_{n})$` 和向量 `$b = (b_{1}, b_{2}, \cdots, b_{n})$`：

1. 对于任意的 `$i$`，有 `$a_{i} = b_{i}$`，称向量 `$a$` 等于向量 `$b$`，记作 `$a=b$`；
2. 对于任意的 `$i$`，有 `$a_{i} \leq b_{i}$`，称向量 `$a$` 小于等于向量 `$b$`，记作 `$a \leq b$`；
3. 对于任意的 `$i$`，有 `$a_{i} < b_{i}$`，称向量 `$a$` 小于向量 `$b$`，记作 `$a < b$`。

## 非支配解

由于多目标优化是多个目标函数组成的向量，因此可以比较不同解之间的优劣关系。
假设决策变量 `$\mathbf{X}_{a}$` 和 `$\mathbf{X}_{b}$` 是两个可行解，
则 `$f_{k}(\mathbf{X}_{a})$` 和 `$f_{k}(\mathbf{X}_{b})$` 表示不同决策变量对应第 `$k$` 个目标函数值。

对于任意的 `$k$`，有 `$f_{k}(\mathbf{X}_{a}) \leq f_{k}(\mathbf{X}_{b})$`，且至少存在一个 `$k$`，
使得 `$f_{k}(\mathbf{X}_{a}) < f_{k}(\mathbf{X}_{b})$` 成立，
则称决策变量 `$\mathbf{X}_{a}$` 支配 `$\mathbf{X}_{b}$`。
如果存在一个变量 `$\mathbf{X}$`，不存在其他决策变量能够支配它，
那么就称该决策变量为<span style='border-bottom:1.5px dashed red;'>非支配解</span>。
从向量比较的角度来说，如果 `$\mathbf{X}_{a}$` 支配 `$\mathbf{X}_{b}$`，
则说明在该多目标规划中 `$\mathbf{X}_{a}$` 要优于 `$\mathbf{X}_{b}$`。

## Pareto 最优解

Pareto 是经济学中的一个概念，翻译过来叫作帕累托。
<span style='border-bottom:1.5px dashed red;'>Pareto 最优</span>在经济学中的意思是，
在一个经济系统中，不能再做任何改进，使得在不损害别人的效用情况下增加自己的效用。

在多目标规划中，Pareto 是这样的一个解，对其中一个目标的优化必然会导致其他目标变差，
即一个解可能在其中某个目标是最好的，但是在其他目标上可能是最差的，因此，不一定在所有目标上都是最优解。
在所有目标函数都是极小化的多目标规划问题中，对于任意的 `$k$`，
有 `$f_{k}(\mathbf{X}^{*}) \leq f_{k}(\mathbf{X})$`，
`$\mathbf{X}^{*}$` 支配其他解 `$\mathbf{X}$`，
称 `$\mathbf{X}^{*}$` 是多目标规划的一个<span style='border-bottom:1.5px dashed red;'>Pareto 最优解</span>，
又称<span style='border-bottom:1.5px dashed red;'>非劣最优解</span>。
所有的 Pareto 最优解组成<span style='border-bottom:1.5px dashed red;'>Pareto 最优集合</span>。
所有 Pareto 最优解组成的曲面称为<span style='border-bottom:1.5px dashed red;'>Pareto 前沿(Pareto Front)</span>，
见下图：

![img](images/pareto_front.png)

其中白色空心点表示 Pareto 最优解，它们互不相同，这一点与单目标规划是不相同的。
它不存在一个单独的最优解，而是一个最优解的集合，在这些解中没有一个绝对的解比另一个更好，
除非加入一些偏好信息。

## 总结

由于多目标规划的解是多个目标函数组成的向量，向量比较是通过向量的序来实现的。对于某个解 `$\mathbf{X}$`，
如果对每个子目标 `$f_{k}$` 都有 `$f_{k}(\mathbf{X}) \leq f_{k}(\mathbf{X}^{*})$`，
则 `$\mathbf{X}$` 是非劣解，也称 Pareto 解。对于两个非劣解 `$\mathbf{X}_{a}$` 和 `$\mathbf{X}_{b}$`，
如果对每个子目标 `$f_{k}$` 均有 `$f_{k}(\mathbf{X}_{a}) \leq f_{k}(\mathbf{X}_{b})$`，
则 `$\mathbf{X}_{a}$` 支配 `$\mathbf{X}_{b}$`。如果没有一个解 `$\mathbf{X}$` 能够支配 `$\mathbf{X}^{*}$`，
则 `$\mathbf{X}^{*}$` 是 Pareto 最优解。

# 多目标优化求解方法

为了求解多目标规划问题的非劣解，常常会将多目标规划问题转化为单目标规划问题去处理，实现这种转化的方法有：

* 评价函数法
* 目标规划法
* 分层序列法
* 智能优化算法，如：NSGA-II

## 评价函数法

评价函数法是一种常见的求解多目标规划的方法，
其基本原理就是用一个评价函数来集中反映各个目标的重要性等因素，
并最小化评价函数。常见的评价函数法有理想点法、极大极小法、线性加权法。

## 评价函数法

### 理想点法

### 极大极小法

### 线性加权法

## 目标规划法

目标规划法（功效系数法）是目前最流行的求解多目标规划的方法。目标规划的基本思想是，
给定若干个目标及实现这些目标的优先顺序，在资源有限的情况下，使总偏离目标的偏差值最小。

这里提到两个概念：

1. 优先顺序
    - 通过给目标赋予一个权重即可
2. 偏差值





## 分层序列法




## NSGA-II




# 参考

* 《Python 最优化算法实战》-- 苏振裕，北京大学出版社