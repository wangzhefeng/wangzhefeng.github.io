---
title: 多目标规划
subtitle: Multi-Objective Optimization
author: wangzf
date: '2024-08-29'
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
- [多目标优化一般形式](#多目标优化一般形式)
- [Pareto 最优解](#pareto-最优解)
    - [向量序](#向量序)
    - [非支配解](#非支配解)
    - [Pareto 最优解](#pareto-最优解-1)
    - [总结](#总结)
- [多目标优化求解方法](#多目标优化求解方法)
    - [评价函数法](#评价函数法)
        - [理想点法](#理想点法)
        - [极大极小法](#极大极小法)
        - [线性加权法](#线性加权法)
    - [目标规划法](#目标规划法)
        - [偏差变量](#偏差变量)
        - [优先等级和权重系数](#优先等级和权重系数)
        - [目标规划单纯形法](#目标规划单纯形法)
        - [目标规划单纯形法过程](#目标规划单纯形法过程)
    - [NSGA-II](#nsga-ii)
    - [总结](#总结-1)
- [多目标优化 Gurobi 实现](#多目标优化-gurobi-实现)
    - [Gurobi 多目标优化简介](#gurobi-多目标优化简介)
    - [Gurobi 多目标优化 API](#gurobi-多目标优化-api)
        - [合成型](#合成型)
        - [分层型](#分层型)
        - [混合型](#混合型)
    - [Gurobi 多目标优化示例](#gurobi-多目标优化示例)
        - [Gurobi 多目标优化示例 1](#gurobi-多目标优化示例-1)
        - [Gurobi 多目标优化示例 2](#gurobi-多目标优化示例-2)
- [参考](#参考)
</p></details><p></p>

# 多目标优化简介

**多目标优化(Multi-objective Optimization Problem, MOP)**也叫多**目标规划**，
即同时优化多个目标的规划问题。

求解多目标规划的有两种方法：一种是<span style='border-bottom:1.5px dashed red;'>合成型</span>，
将多目标转化成单目标决策问题；另一种是<span style='border-bottom:1.5px dashed red;'>分层型</span>，
在保证第一目标的情况下，尽量优化第二、第三等目标。
因此，多目标规划一般有两种方法：一种是化多为少，即将多目标转化为比较容易求解的单目标规划方法；
另一种是分层序列法，即把目标按其重要性排序，每次都在前一个目标最优解集内求解下一个目标的最优解，
直到求出共同的最优解。

如何理解**目标最优解集**呢？在多目标规划中往往有多个最优解同时满足约束条件，
不同的解之间不能简单通过大小来比较，这点是同单目标规划的最大区别，
多个解组成的集合称为**帕累托最优解集**，组成的超平面称为**帕累托前沿**。

# 多目标优化一般形式

多目标规划是由多个目标函数构成的，其数学模型一般描述如下：

`$$\text{min} \space f_{1}(x_{1}, x_{2}, \cdots, x_{n})$$`
`$$\text{min} \space f_{2}(x_{1}, x_{2}, \cdots, x_{n})$$`
`$$\cdots$$`
`$$\text{min} \space f_{m}(x_{1}, x_{2}, \cdots, x_{n})$$`

`$$\text{s.t.}\begin{cases}
g_{i}(x_{1}, x_{2}, \cdots, x_{n}) \leq 0, i = 1, 2, \cdots, p \\
h_{j}(x_{1}, x_{2}, \cdots, x_{n}) = 0, i = 1, 2,\cdots, q
\end{cases}$$`

该规划问题有 `$m$` 个目标函数，均为求最小值问题，最大化问题可以通过乘 `$-1$` 转化成最小化问题，
所以，写成向量的形式如下：

`$$\begin{align}
Z 
&= \text{min} \space F(\mathbf{X}) \\
&= \text{min} \space [f_{1}(\mathbf{X}),f_{2}(\mathbf{X}), \cdots, f_{m}(\mathbf{X})]^{T}
\end{align}$$`

`$$s.t.\begin{cases}
g_{i}(\mathbf{X}) \leq 0, i = 1, 2, \cdots, p \\
h_{j}(\mathbf{X}) = 0, i = 1, 2, \cdots, q
\end{cases}$$`

其中，`$g_{i}(\mathbf{X}) \leq 0$` 为不等式约束，`$h_{j}(\mathbf{X}) = 0$` 为等式约束。

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
<span style='border-bottom:1.5px dashed red;'>Pareto 最优</span>在经济学中的意思是：
**在一个经济系统中，不能再做任何改进，使得在不损害别人的效用情况下增加自己的效用。**

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

由于多目标规划的解是多个目标函数组成的向量，**向量比较是通过向量的序来实现的**。

对于某个解 `$\mathbf{X}$`，如果对每个子目标 `$f_{k}$` 都有 `$f_{k}(\mathbf{X}) \leq f_{k}(\mathbf{X}^{*})$`，
则 `$\mathbf{X}$` 是非劣解，也称 **Pareto 解**。
对于两个非劣解 `$\mathbf{X}_{a}$` 和 `$\mathbf{X}_{b}$`，
如果对每个子目标 `$f_{k}$` 均有 `$f_{k}(\mathbf{X}_{a}) \leq f_{k}(\mathbf{X}_{b})$`，
则 `$\mathbf{X}_{a}$` 支配 `$\mathbf{X}_{b}$`。

如果没有一个解 `$\mathbf{X}$` 能够支配 `$\mathbf{X}^{*}$`，则 `$\mathbf{X}^{*}$` 是 **Pareto 最优解**。

# 多目标优化求解方法

为了求解多目标规划问题的非劣解，常常会将多目标规划问题转化为单目标规划问题去处理，实现这种转化的方法有：

* 评价函数法
* 目标规划法
* 分层序列法
* 智能优化算法，如：NSGA-II

## 评价函数法

评价函数法是一种常见的求解多目标规划的方法，其基本原理就是用一个评价函数来集中反映各个目标的重要性等因素，
并最小化评价函数。常见的评价函数法有：理想点法、极大极小法、线性加权法。

### 理想点法

在决策优化过程中，每个目标函数都有一个**最优解**，假设为 `$f_{k}^{*}$`，
而在多目标求解过程的**实际解**是 `$f_{k}$`，那么**最优解和实际解之间存在差距可以用距离来表示**，
对于所有的目标，其评价函数为总的距离，表示如下：

`$$h(F(\mathbf{X})) = \text{min} \space Z = \sqrt{\sum_{k=1}^{m}\lambda_{k}[f_{k}^{*}(x)-f_{k}(x)]^{2}}$$`

式中，`$\lambda_{k}$` 为第 `$k$` 个目标函数的权重。上述表达式和最小二乘法的原理一样。

利用距离表达式也存在一些问题，如果两个目标函数之间的度量单位不同，则不能直接进行比较和加权，
如果目标函数 `$f_{1}$` 的量纲是 100，而目标函数 `$f_{2}$` 的量纲是 10000，
则 `$Z$` 的值基本上由 `$f_{2}$` 决定，对于 `$f_{1}$` 的影响可以忽略不计。

### 极大极小法

极大极小法考虑的是在最不利的情况下找出最好的结果，
其评价函数可以选择多个目标函数 `$f_{k}$` 中的最大值，即：

`$$h(F(\mathbf{X})) = \underset{l \leq k \leq m}{\text{max}}f_{k}(\mathbf{X})$$`

原问题可以归结为数值极小化问题，即：

`$$\text{min} \space h(F(\mathbf{X})) = \text{min}\Big(\underset{l\leq k \leq m}{\text{max}} f_{k}(\mathbf{X})\Big)$$`

为了求解上述问题，常常引入新的变量 `$v$`，构造一个与之等价的最优化问题，即：

`$$\text{min}\space v$$`
`$$\text{s.t.}\space f_{k}(\mathbf{X}) \leq v, k = 1,2,\cdots, m$$`

还可以根据各个目标的重要性赋予权重，用 `$w_{k}f_{k}(\mathbf{X})$` 代替 `$f_{k}(\mathbf{X})$`，
使各权重系数满足 `$\sum_{k=1}^{m}w_{k}=1$`。

极大极小法的思想类似整数规划中的分支定界法，
`$\underset{l\leq k \leq m}{\text{max}} f_{k}(\mathbf{X})$` 是整个问题的上界，
不断减小上界直到极值点。

### 线性加权法

线性加权法是一种简单的多目标优化方法，基本原理是对每个目标赋予权重，
然后将各个目标函数相加构成一个新的目标函数，并通过求解这个单个目标函数得到多目标的一个解，
即：

`$$h(F(\mathbf{X})) = \sum_{k=1}^{m}w_{k}f_{k}$$`
`$$\text{s.t.} \space \sum_{k=1}^{m}w_{k} = 1$$`

## 目标规划法

目标规划法（功效系数法）是目前最流行的求解多目标规划的方法。目标规划的基本思想是，
给定若干个目标及实现这些目标的优先顺序，在资源有限的情况下，使总偏离目标的偏差值最小。

这里提到两个概念：

1. 优先顺序
    - 通过给目标赋予一个权重即可
2. 偏差值

考虑如下单目标规划问题：

`$$\text{max}\space Z=8x_{1}+10x_{2}$$`
`$$\text{s.t.}\space\begin{cases}
2x_{1}+x_{2}\leq 11 \\
x_{1}+2x_{2} \leq 10 \\
x_{1}, x_{2} \geq 0
\end{cases}$$`

其中，`$x_{1}$` 和 `$x_{2}$` 分别是 A 产品、B 产品的生产数量，约束是指资源消耗的情况。

虽然利用单纯形法能很好地求解这个问题，但是在决策时还应考虑一系列因素，例如以下几种：

1. A 产品有销售下降趋势，就可以考虑 A 产品生产数量不要大于 B 产品；
2. 应该尽可能利用设备，但不希望有加班情况；
3. 应尽可能达到预期收益，如 56 元。

这里有多个目标，这些目标同时反映了不同类型的约束。有些约束不能违反，称为**硬约束**，
可以违反的约束称之为**软约束**，对于违反软约束的情况需要添加惩罚项，如允许加班，
但是加班的惩罚需要体现在成本里面。对于任何违反硬约束的解都是不可行解。

### 偏差变量

偏差变量表示未达到目标或超过目标的部分，通常用正偏差 `$d^{+}$` 表示目标值的部分，
用负偏差 `$d^{-}$` 表示未达到目标值的部分。注意：不可能出现正偏差的同时又出现负偏差，
也就是说，决策值不可能即超过目标值，又没有达到目标值，因此 `$d^{-} \times d^{+} = 0$`。

那该如何设置偏差变量呢？有以下三种基本形式，通过最小化问题进行了解：

1. 要求恰好达到目标值，即正负偏差都要尽可能小，这时 `$\text{min}\space Z = d^{-} + d^{+}$`；
2. 要求不超过目标值，即允许不达到目标值，即正偏差尽可能小，这时 `$\text{min}\space Z = d^{+}$`；
3. 要求超过目标值，即负偏差尽可能小，这时 `$\text{min}\space Z = d^{-}$`。

回到问题中，对于前面提到的一系列条件，考虑如下：

1. A 产品不大于 B 产品，即 `$\text{min}\space Z = d_{1}^{+}$`；
2. 充分利用设备，但不希望加班，即 `$\text{min}\space Z = d_{2}^{+} + d_{2}^{-}$`；
3. 最终产值不低于 56 元，即 `$\text{min}\space Z = d_{3}^{-}$`。

### 优先等级和权重系数

对于多目标规划而言，通常要在保证前一个目标值不会劣化的前提下优化下一个目标，
因此可以给每个目标赋予优先级 `$p_{i}$` 且 `$p_{i} \gg p_{1+i}$`，
以及给每个目标赋予权重系数 `$\omega_{i}$`。

加入权重系数的多目标优化可以表示为：

`$$\text{min}\space Z = \{p_{1}(\omega_{1}, d_{1}^{+}), p_{2}(\omega_{2}d_{2}^{+} + \omega_{2}d_{2}^{-}), p_{3}(\omega_{3}d_{3}^{-}) \}$$`
`$$\text{s.t.}\space\begin{cases}
2x_{1} + x_{2} + x_{3} = 11 \\
x_{1} - x_{2} + d_{1}^{-}  - d_{1}^{+} = 0 \\
x_{1} + 2x_{2} + d_{2}^{-} - d_{2}^{+} = 10 \\
8x_{1} + 10x_{2} + d_{3}^{-} - d_{3}^{+} = 56 \\
x_{1}, x_{2}, d_{i}^{-}, d_{i}^{+} \geq 0
\end{cases}$$`

此处 `$d_{i}^{-}-d_{i}^{+}$` 起到了松弛变量的作用。模型中的第一个约束 `$2_{1}+x_{2} \leq 11$` 是硬约束，
可以添加松弛变量 `$x_{3}$` 使之变成等式约束。

### 目标规划单纯形法

目标规划的数学模型结构和线性规划的数学模型结构没有本质区别，因此目标规划也可以用单纯形法求解。
由于目标规划中含有多个目标函数，因此单纯性表中的检验数慧有多行，检验数的行数由目标优先等级的个数决定，
在确认入基变量时，不但要根据本优先级的检验数，还要根据比它更高优先级的检验数来确定。

假设多目标函数都是求最小值问题，则要求所有检验数 `$R_{j} \geq 0$`。

目标规划单纯形法的计算步骤如下：

1. 建立初始单纯性表，在表中将检验数鞍优先因子个数分别列成 K 行，并设 `$k=1$`；
2. 检查第 `$k$` 行检验数是否存在负数，且对应前 `$k-1$` 行的系数为 0，若有则取其中最小者对应的变量为入基变量，然后转向第 3 步，否则转向第 5 步；
3. 按照最小比值法确定出基变量，当存在两个或两个以上的相同的最小比值时，选取较高优先级别的变量为出基变量；
4. 按单纯形法进行旋转运算，建立新的单纯形表，然后返回第 2 步；
5. 当 `$k=K$` 时，计算结束，表中的解为满意解。否则令 `$k=k+1$`，返回第 2 步。

### 目标规划单纯形法过程

下面设有如下多目标规划问题：

`$$\text{min}\space Z = \{p_{1}(d_{1}^{+}), p_{2}(d_{2}^{+} + d_{2}^{-}), p_{3}(d_{3}^{-}) \}$$`
`$$\text{s.t.}\space\begin{cases}
2x_{1} + x_{2} + x_{3} = 11 \\
x_{1} - x_{2} + d_{1}^{-}  - d_{1}^{+} = 0 \\
x_{1} + 2x_{2} + d_{2}^{-} - d_{2}^{+} = 10 \\
8x_{1} + 10x_{2} + d_{3}^{-} - d_{3}^{+} = 56 \\
x_{1}, x_{2}, d_{i}^{-}, d_{i}^{+} \geq 0
\end{cases}$$`

1. 建立初始单纯性表，如下：

| `$X_{B}$` | `$x_{1}$` | `$x_{2}$` | `$x_{3}$` | `$d_{1}^{-}$` | `$d_{1}^{+}$` | `$d_{2}^{-}$` | `$d_{2}^{+}$` | `$d_{3}^{-}$` | `$d_{3}^{+}$` | `$b$` | `$\theta$` |
|-----------|---------|-----------|-----------|---------------|---------------|---------------|---------------|---------------|---------------|-------|------------|
| 变量       | `$c_{j}$`    |    |     |   |   | 1  | 1  | 1 | 1  |    |    | |
|           | `$x_{3}$`     | 2  | 1   | 1 |   |    |    |    |   |    | 11 | |
|           | `$d_{1}^{-}$` | 1  | -1  |   | 1 | -1 |    |    |   |    | 0  | |
|           | `$d_{2}^{-}$` | 1  | 2   |   |   |    | 1  | -1 |   |    | 10 | |
|           | `$d_{3}^{-}$` | 8  | 10  |   |   |    |    |    | 1 | -1 | 56 | |
| `$R_{j}$` | `$p_{1}$`     |    |     |   |   | 1  |    |    |   |    |    | |
|           | `$p_{2}$`     | -1 | -2  |   |   |    |    | 2  |   |    |    | |
|           | `$p_{3}$`     | -8 | -10 |   |   |    |    |    |   |  1 |    | |

* TODO


## NSGA-II

多目标规划算法除了前面提到的目标规划法外，使用智能优化算法求解多目标规划问题也是常见的方法，
其中以 NSGA-II 应用较为广泛。

NSGA-II 是在 NSGA 的基础上提出的，而 **NSGA 是基于遗传算法的多目标优化算法**。
在遗传算法中，种群的交叉、变异、复制等操作在不同的领域问题有不同的形式，NSGA-II 相比于 NSGA，
其改进点在于采用了**快速非支配排序算法**，计算复杂度大大地降低，采用拥挤度比较算子，而不是 NSGA 中的共享半径，
使得搜索范围更广泛，种群个体能够扩展到整个 Pareto 域，在个体复制操作上采用精英策略，精英策略可以加速算法的执行速度，
而且在一定程度上也能确保已经找到的满意解不被丢失，提高算法的鲁棒性。

## 总结

大部分的优化问题都有两种方法：

1. 精确算法，如数学规划方法
2. 启发式算法，如遗传算法、粒子群算法

在实际建模过程中，如果数学规划方法不可行时，可以从启发式算法入手分析问题，解决问题。

# 多目标优化 Gurobi 实现

## Gurobi 多目标优化简介

在多目标优化中，可以直接把多个目标通过分配权重的方式组合成单目标优化问题，
但是如果多个目标函数之间的数量级差异很大，则应该使用分层优化的方法。

## Gurobi 多目标优化 API

在 Gurobi 中，可以通过 `Model.setObjectiveN` 函数来建立多目标优化模型，
多目标的 `setObjectiveN` 函数和单目标的 `setObjective` 函数用法基本一致，
不同的是多了目标优先级、目标劣化接受程度、多目标的权重等参数。

```python
setObjectiveN(expr, index, priority, weight, abstol, reltol, name)
```

参数说明如下：

1. `expr`: 目标函数表达式，如 `$x+2y+3z$`；
2. `index`: 目标函数对应的需要 `$(0, 1, 2, \cdots)$`，即第几个目标，注意目标函数序号从 `$0$` 开始；
3. `priority`: 优先级，为整数，值越大表示目标优先级越高；
4. `weight`: 权重（浮点数），在合成型多目标解法中使用该参数，表示不同目标之间的组合权重；
5. `abstol`: 允许的目标函数值最大的降低量 `abstol`（浮点数），即当前迭代的值相比最优值的可接受劣化程度；
6. `reltol`: `abstol` 的百分数表示，如 `reltol = 0.05` 表示可接受劣化程度是 5%； 
7. `name`: 目标函数名称；

需要注意的是，在 Gurobi 的多目标优化中，要求所有的目标函数都是线性的，并且目标函数的优化方向应一致，
即全部最大化或全部最小化，因此可以通过乘以 -1 实现不同的优化方向。

当前 Gurobi 支持 3 种多目标模式，分别是 Blend(合成型)、Hierarchical(分层型)、两者的混合型。

### 合成型

合成型(Blend)通过对多个目标赋予不同的权重实现将多目标转化成单目标函数，
权重扮演优先级的角色。例如，有如下两个优化目标：

`$$obj_{1}=x+2y, weight_{1} = 3$$`
`$$obj_{2}=x-3y, weight_{2} = 0.5$$`

经过合成后的单目标函数为：

`$$\begin{align}
obj
&=weight_{1} \times obj_{1} + weight_{2}\times obj_{2} \\
&=3\times (x+2y)-0.5\times(x-3y)\\
&=2.5x + 7.5
\end{align}$$`

Gurobi 使用方法如下：

```python
import gurobipy as grb

model = grb.Model()

x = model.addVars(name = "x")
y = model.addVars(name = "y")

# 添加第一个目标
model.setObjectiveN(x + 2 * y, index = 0, weight = 3, name = "obj1")
# 添加第二个目标
model.setObjectiveN(x - 3 * y, index = 1, weight = 0.5, name = "obj2")
```

```python
for i in range(model.NumObj):
    model.setParam(grb.GRB.Param.ObjNumber, i)
    print(f"第 {i} 个目标的优化值是 {model.objNVal}")
```

### 分层型

分层型(Hierarchical)有优先级，一般理解是在保证第一个目标值最优的情况下优化第二个目标，
或者在优化第二个目标时要保证第一个目标的最优值只能允许少量劣化。

例如，有如下两个优化目标：

`$$obj_{1}=x+2y, priority_{1}=2$$`
`$$obj_{2}=x-3y, priority_{2}=1$$`

此时 Gurobi 按照优先级大小进行优化（先优化 `$obj_{1}$`，再优化 `$obj_{2}$`）。
若没有设定 `abstol` 或 `reltol`，则在优化低优先级目标(`$obj_{2}$`)时，
不会改变高优先级的目标(`$obj_{1}$`)值。

```python
import gurobipy as grb

model = grb.Model()

x = model.addVars(name = "x")
y = model.addVars(name = "y")

# 添加第一个目标
model.setObjectiveN(x + 2 * y, index = 0, priority = 20, name = "obj1")
# 添加第二个目标
model.setObjectiveN(x - 3 * y, index = 1, priority = 1, name = "obj2")
```

```python
for i in range(model.NumObj):
    model.setParam(grb.GRB.Param.ObjNumber, i)
    print(f"第 {i} 个目标的优化值是 {model.objNVal}")
```

### 混合型

混合型的写法也很简单，将权重和优先级同时设定即可：

```python
import gurobipy as grb

model = grb.Model()

x = model.addVars(name = "x")
y = model.addVars(name = "y")

# 添加第一个目标
model.setObjectiveN(x + 2 * y, index = 0, weight = 3, priority = 20, name = "obj1")
# 添加第二个目标
model.setObjectiveN(x - 3 * y, index = 1, weight = 0.5, priority = 1, name = "obj2")
```

```python
for i in range(model.NumObj):
    model.setParam(grb.GRB.Param.ObjNumber, i)
    print(f"第 {i} 个目标的优化值是 {model.objNVal}")
```

## Gurobi 多目标优化示例

### Gurobi 多目标优化示例 1

假设工厂需要把 `$N$` 份工作分配给 `$N$` 个工人，每份工作只能由一个工人做，
且每个工人也只能做一份工作。假设工人 `$i$` 处理工作 `$j$` 需要的时间是 `$T_{ij}$`，
获得的利润是 `$C_{ij}$`，那么需要怎么安排才能使得总利润最大且总耗时最小呢？

这里有两个目标，最主要的目标是利润最大化，次要目标是耗时最小化。

为了编程方便，这里假设 `$N=10$`，`$T_{ij}$` 和 `$C_{ij}$` 通过随机数生成。

```python
import numpy as np
import gurobipy as grb

# 设置工人数和工作数量
N = 10
np.random.seed(1234)

# 用随机数初始化时间矩阵 T_{ij} 和成本矩阵 C_{ij}
Tij = {
    (i, j): np.random.randint(0, 100) 
    for i in rang(1, N + 1) 
    for j in rang(1, N + 1)
}
Cij = {
    (i, j): np.random.randint(0, 100) 
    for i in rang(1, N + 1) 
    for j in rang(1, N + 1)
}

# 定义模型
m = grb.Model("MultiObj")

# 添加变量
# x 是 0-1 变量，xij=1 表示第 i 个工人被分配到第 j 个工作中
x = m.addVars(Tij.keys(), vtype = grb.GRB.BINARY, name = "x")

# 添加约束
# 第 1 个约束表示一份工作只能分配给一个工人
m.addConstrs((x.sum("*", j) == 1 for j in range(1, N + 1)), name = "C1")
# 第 2 个约束表示一个工人制作一份工作
m.addConstrs((x.sum(i, "*") == 1 for i in range(1, N + 1)), name = "C2")
```

多目标方式 1: 合成型

```python
# 设置多重目标权重
m.setObjectiveN(x.prod(Tij), index = 0, weight = 0.1, name = "obj1")
m.setObjectiveN(-x.prod(Cij), index = 1, weight = 0.5, name = "obj2")

# 启动求解
m.optimize()

# 获取求解结果
for i in Tij.keys():
    if x[i].x > 0.9:
        print(f"工人 {i[0]} 分配工作 {i[1]}")

# 获取目标函数值
for i in rang(1, 3):
    m.setParam(grb.GRB.Param.ObjNumber, i)
    print(f"Obj{i} = {m.ObjNVal}")
```

```

```

多目标方式 2：分层型

```python
# 设置目标函数
m.setObjectiveN(
    x.prod(Tij), index = 0, priority = 1, 
    abstol = 0, reltol = 0, name = "obj1"
)
m.setObjectiveN(
    -x.prod(Cij), index = 1, priority = 2, 
    abstol = 100, reltol = 0, name = "obj2"
)

# 启动求解
m.optimize()

# 获取求解结果
for i in Tij.keys():
    if x[i].x > 0.9:
        print(f"工人 {i[0]} 分配工作 {i[1]}")

# 获取目标函数值
for i in rang(1, 3):
    m.setParam(grb.GRB.Param.ObjNumber, i)
    print(f"Obj{i} = {m.ObjNVal}")
```

```

```

多目标方式 3：混合型

```python
# 设置目标函数

# 启动求解
m.optimize()
```

### Gurobi 多目标优化示例 2

在 Gurobi 中，目标规划使用分层规划的方式实现。

问题：

`$$$$`
`$$$$`

```python
import gurobipy as grb

# 创建模型
m = grb.Model()

# 定义变量
d11 = m.addVars(lb = 0, vtype = grb.GRB.CONTINUOUS, name = "d11")
d12 = m.addVars(lb = 0, vtype = grb.GRB.CONTINUOUS, name = "d12")
d21 = m.addVars(lb = 0, vtype = grb.GRB.CONTINUOUS, name = "d21")
d22 = m.addVars(lb = 0, vtype = grb.GRB.CONTINUOUS, name = "d22")
d31 = m.addVars(lb = 0, vtype = grb.GRB.CONTINUOUS, name = "d31")
d32 = m.addVars(lb = 0, vtype = grb.GRB.CONTINUOUS, name = "d32")
x1 = m.addVars(lb = 0, vtype = grb.GRB.CONTINUOUS, name = "x1")
x2 = m.addVars(lb = 0, vtype = grb.GRB.CONTINUOUS, name = "x2")
x3 = m.addVars(lb = 0, vtype = grb.GRB.CONTINUOUS, name = "x3")

# 添加约束
m.addConstrs(2 * x1 + x2 + x3 == 11)
m.addConstrs(x1 - x2 + d11 - d12 == 0)
m.addConstrs(x1 + 2 * x2 + d21 - d22 == 10)
m.addConstrs(8 * x1 + 10 * x2 + d31 - d32 == 56)

# 添加目标
m.setObjectiveN(d12, index = 0, priority = 9, name = "obj1")
m.setObjectiveN(d21 + d22, index = 1, priority = 6, name = "obj2")
m.setObjectiveN(d31, index = 2, priority = 3, name = "obj3")

# 求解
m.optimize()

# 查看变量值
for v in m.getVars():
    print(f"{v.varName}={v.x}")

# 查看各个目标函数值
for i in range(3):
    m.setParam(grb.GRB.Param.ObjNumber, i)
    print(f"Obj{(i+1)} = {m.ObjNVal}")

# 查看最终的目标函数值
print(f"8 * x1 + 10 * x2 = {8 * 2 + 10 * 4}")
```

# 参考

* 《Python 最优化算法实战》-- 苏振裕，北京大学出版社
