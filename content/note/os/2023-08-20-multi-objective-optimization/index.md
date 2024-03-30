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
    - [Pareto](#pareto)
- [多目标优化求解方法](#多目标优化求解方法)
    - [评价函数法](#评价函数法)
        - [理想点法](#理想点法)
        - [极大极小法](#极大极小法)
        - [线性加权法](#线性加权法)
    - [目标规划法](#目标规划法)
    - [分层序列法](#分层序列法)
    - [NSGA-II](#nsga-ii)
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

由于多目标优化是多个目标函数组成的向量，因此可以比较不同解之间的优劣关系。
假设决策变量 `$\mathbf{X}_{a}$` 和 `$\mathbf{X}_{b}$` 是两个可行解，
则 `$f_{k}(\mathbf{X}_{a})$` 和 `$f_{k}(\mathbf{X}_{b})$` 表示不同决策变量对应第 `$k$` 个目标函数值。



## Pareto




# 多目标优化求解方法

## 评价函数法

### 理想点法

### 极大极小法


### 线性加权法

## 目标规划法

## 分层序列法

## NSGA-II



