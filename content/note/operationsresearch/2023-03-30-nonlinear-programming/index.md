---
title: 非线性规划
subtitle: NonLinear Programming
author: wangzf
date: '2023-03-30'
slug: nonlinear-programming
categories:
  - optimizer algorithm
tags:
  - algorithm
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

- [无约束非线性规划模型](#无约束非线性规划模型)
- [带约束的非线性规划模型](#带约束的非线性规划模型)
    - [选址-分配问题](#选址-分配问题)
        - [线性约束的非线性规划](#线性约束的非线性规划)
    - [汽油混合模型](#汽油混合模型)
    - [工程设计模型](#工程设计模型)
    - [特殊的非线性规划](#特殊的非线性规划)
        - [凸规划](#凸规划)
        - [可分离规划](#可分离规划)
        - [二次规划](#二次规划)
        - [正项几何规划](#正项几何规划)
</p></details><p></p>

线性规划模型使用连续决策变量、线性约束、线性目标函数。
而非线性规划(nonlinear programming, NLP)涵盖了所有其他单目标、
连续决策变量的规划模型。

非线性规划模型：

* 无约束非线性规划
    - 单目标
    - 连续决策变量
    - 非线性目标函数
* 带约束非线性规划
    - 单目标
    - 连续决策变量
    - 非线性约束或线性约束
    - 非线性目标函数或线性目标函数

# 无约束非线性规划模型

> 在无约束优化中，线性目标函数总是无界的（除了目标函数是常数的不重要情形），
> 但无约束的非线性规划可能存在有界的最优解。



# 带约束的非线性规划模型

## 选址-分配问题

选址问题以及将顾客分配到服务设施等问题，在非线性规划中是非常常见的。

下面将介绍一家名叫 Beer Belge 的比利时啤酒公司在经营中遇到的问题。
Beer Belge 希望重组现有的 17 个仓库以更加有效地服务遍布比利时全境地 24000 名顾客。
为了分析的方便，将所有顾客划分为 650 个区域。

下面是相关的下标定义：

`$$i \triangleq \text{仓库编号}(i=1,\cdots,17)$$`
`$$j \triangleq \text{顾客区域}(i=j,\cdots,650)$$`

公司的分析员可以根据地图和以往的经验来确定每个顾客区域的信息。

`$$h_{j} \triangleq \text{顾客区域}\space j \space\text{的中心的}\space x\space \text{坐标}$$`
`$$k_{j} \triangleq \text{顾客区域}\space j \space\text{的中心的}\space y\space \text{坐标}$$`
`$$d_{j} \triangleq \text{每年需要去区域} \space j\space \text{的交货次数}$$`

我们希望选择所有仓库的位置，然后将不同的顾客区域对应到 17 个仓库中，
目标是最小化总的旅行成本。

如所有的选址-分配(location-allocation)问题一样，Beer Belge 的决策变量体现在两个分离而相关的问题中。
首先，我们需要决定仓库放在什么位置；其次，我们需要将去区域 `$j$` 交付货物的次数分配给这 17 个仓库。
我们可以按照如下方式来定义决策变量：

`$$x_{i} \triangleq \text{仓库} \space i \space \text{位置的} \space x \space \text{坐标}$$`
`$$y_{i} \triangleq \text{仓库} \space i \space \text{位置的} \space y \space \text{坐标}$$`
`$$w_{i,j}\triangleq \text{仓库}\space i \space \text{负责配送区域} \space j \space \text{的交货次数}$$`

为了得到待优化的目标函数，假设从仓库 `$i$` 到区域 `$j$` 的往返成本与两个位置的直线（欧几里得）距离成正比。
所以，最小化总运输成本可以用下面的公式来表示：

`$$\text{min}\space \sum_{i}\sum_{j}(i \space \text{到} \space j \space \text{的次数})(i \space \text{到} \space j \space \text{的距离})$$`

利用先前对常数和决策变量定义好的符号，得到：

`$$\text{min}\space\sum_{i=1}^{17}\sum_{j=1}^{650}w_{i,j}\sqrt{(x_{i} - h_{j})^{2} + (y_{i}-k_{j})^{2}}$$`

考虑到调配方案的可行性，为了完成这个模型，我们必须要添加一些约束。最终的模型就变成了下面的样子：

`$$\text{min}\space\sum_{i=1}^{17}\sum_{j=1}^{650}w_{i,j}\sqrt{(x_{i} - h_{j})^{2} + (y_{i}-k_{j})^{2}}\qquad(\text{总配送距离})$$`

`$$\begin{align}
\text{s.t.}\space
& \sum_{i=1}^{17}w_{i,j}=d_{j}, j=1, \cdots, 650\qquad(\text{给区域}\space j\space \text{的配送次数}) \\ 
& w_{i,j} \geq 0, i = 1, \cdots, 17; j = 1,\cdots,650
\end{align}$$`

### 线性约束的非线性规划

> 许多可以被有效解决的大型非线性规划问题中所有或者几乎所有的约束都是线性的。

上述选址-分配模型说明了许多大规模的非线性规划包含的所有约束都是线性的。
而仅仅是因为目标函数不是线性的，所以才将这个问题称为非线性规划，也可简称为 NLP。

含有线性约束的非线性规划构成了一类重要的 NLP 问题，这主要因为线性规划是易于处理的，
而此类型的 NLP 仅仅将目标函数扩展为非线性。幸运的是，线性约束的 NLP 是非常常见的。

## 汽油混合模型

## 工程设计模型

> 对结构和过程的最优工程设计常常带来相对较少的变量和高度非线性的约束和目标函数。




## 特殊的非线性规划

### 凸规划

批量规划模型：批量规划模型的特殊之处在于它是一个凸规划(convex program)。

一个有如下形式的有约束线性规划：

`$$\text{max 或 min} f(\mathbf{x})$$`
`$$\text{s.t.}\space g_{i}(\mathbf{x})\begin{cases}
\geq b_{i} \\
\leq b_{i} , i=1,\cdots, m\\
= b_{i}
\end{cases}$$`

是一个凸规划，如果目标函数是最大化凹函数 `$f$` 或最小化凸函数 `$f$`，
每个满足 `$\geq$` 约束的 `$g_{i}$` 都是凹函数，每个满足 `$\leq$` 约束的 `$g_{i}$` 都是凸函数，
而每个满足 `$=$` 约束的 `$g_{i}$` 都是线性的。

凸规划的易处理性：




### 可分离规划

最优批量规划模型除了满足凸规划的条件之外，还有一种特殊的可分离性质。

如果函数 `$s(\mathbf{x})$` 可以被表示为下列部分之和：

`$$s(x_{i}, \cdots, x_{n}) \triangleq \sum_{j-1}^{n}s_{j}(x_{j})$$`

其中的每一个函数 `$s_{1}(x_{1}), \cdots, s_{n}(x_{n})$` 都是单变量函数，
则函数 `$s(\mathbf{x})$` 是可分离的。

也就是说一个函数是可分离的，当且仅当它可以写为一系列单变量函数的和。

当目标函数和所有的函数都是可分离的，便将这类 NLP 称为可分离规划。

一个有约束的非线性规划如果满足下面的函数形式：

`$$\text{max 或 min} f(\mathbf{x})$$`
`$$\text{s.t.}\space g_{i}(\mathbf{x})\begin{cases}
\geq b_{i} \\
\leq b_{i} , i=1,\cdots, m\\
= b_{i}
\end{cases}$$`

且 `$f$` 和所有的 `$g_{i}$` 都是可分离的，则该规划是一个可分离规划。

可分离规划的易处理性：

### 二次规划

二次型投资组合管理模型：

`$$\text{min}\space 66.51 x_{1}^{2} + 2(2.61)x_{1}x_{2}+2(2.18)x_{1}x_{3}+0.63x_{2}^{2}+2(0.48)x_{2}x_{3}+0.38 x_{3}^{2}$$`

`$$\begin{align}
\text{s.t.} \space
& x_{1}+x_{2}+x_{3} = 1 \\
& 13.22x_{1}+8.24x_{2}+9.03x_{3} \geq 11 \\
& x_{1}, x_{2}, x_{3} \geq 0
\end{align}$$`


二次规划定义：

一个有约束的非线性规划是二次规划，如果目标函数是二次型，也就是：

`$$f(\mathbf{x})\triangleq \sum_{j}c_{j}x_{j}+\sum_{i}\sum_{j}q_{i,j}x_{i}x_{j} = \mathbf{c}\cdot\mathbf{x}+\mathbf{xQx}$$`

而所有的约束都是线性的。


二次规划的易处理性：



### 正项几何规划



正项几何规划的已处理性：

