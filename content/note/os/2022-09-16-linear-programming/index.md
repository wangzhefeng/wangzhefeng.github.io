---
title: 线性规划
author: 王哲峰
date: '2022-09-16'
slug: linear-programming
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

- [线性规划的标准型](#线性规划的标准型)
- [线性规划的求解方式](#线性规划的求解方式)
  - [单纯形法](#单纯形法)
    - [单纯形法的数学规范型](#单纯形法的数学规范型)
    - [单纯形法过程](#单纯形法过程)
    - [单纯形法示例](#单纯形法示例)
  - [内点法](#内点法)
    - [内点法原理](#内点法原理)
    - [内点法过程](#内点法过程)
    - [内点法示例](#内点法示例)
  - [列生成法](#列生成法)
- [对偶问题](#对偶问题)
- [拉格朗日乘子法](#拉格朗日乘子法)
- [参考](#参考)
</p></details><p></p>

# 线性规划的标准型

对于线性规划，先来看一个简单的数学规划模型，即：

`$$max Z = 70x_{1} + 30x_{2}$$`

`$$s.t.\begin{cases}
3x_{1} + 9x_{2} \leq 540 \\
5x_{1} + 5x_{2} \leq 450 \\
9x_{1} + 3x_{2} \leq 720 \\
x_{1}, x_{2} \geq 0
\end{cases}$$`

显然这不是线性规划数学模型的标准形式，在线性规划求解方法中，模型的标准形式如下：

1. 目标函数求最大值；
2. 约束条件为等式约束；
3. 约束条件右边的常数项大于等于 0；
4. 所有变量大于或等于 0。

对于非标准形式的模型，约束方程可以通过引入 **松弛变量** 使不等式约束转化为等式约束：

* 如果目标函数是求最小值，则两边乘以 `$-1$` 将求 `$min$` 转成求 `$max$`；
* 如果遇到约束方程右边常数项为负数，则将约束方程乘以 `$-1$` 使常数项非负；
* 如果变量 `$x_{k}$` 没有约束，则既可以是正数也可以是负数，
  另 `$x_{k} = x_{k}^{'} - x_{k}^{''}$`，其中 `$x_{k}^{'}x_{k}^{''} \geq 0$`。

通过变换，上面模型的标准型如下：

`$$max Z = 70x_{1} + 30x_{2}$$`

`$$s.t.\begin{cases}
3x_{1} + 9x_{2} + x_{3} = 540 \\
5x_{1} + 5x_{2} + x_{4} = 450 \\
9x_{1} + 3x_{2} + x_{5} = 720 \\
x_{1}, x_{2}, x_{3}, x_{4}, x_{5} \geq 0
\end{cases}$$`

将线性规划模型准换成标准型后，就可以使用经典的线性规划方法求解了，包括单纯形法、内点法等。

# 线性规划的求解方式

## 单纯形法

单纯形法是求解线性规划的经典方法，与多元消去法求解多元一次方程的原理类似，在具体实现上，
<span style='border-bottom:1.5px dashed red;'>通过矩阵的变换对解空间进行搜索</span>。
由于线性规划模型的目标函数和约束方程都是凸函数，所以单纯形法能够以很高的效率求解线性规划问题。
单纯形法是最优算法的基础算法，也是后续其他整数规划等算法的基础。

### 单纯形法的数学规范型

> 单纯形法原理

由于线性规划模型中目标函数和约束方程都是凸函数，因此从凸优化的角度来说，
线性规划的最优解在可行域的顶点上，单纯形法的本质就是通过矩阵的线性变换来遍历这些顶点以计算最优解。

假设有一个规划问题，即：

`$$max \mathbf{Z} = \mathbf{CX}$$`
`$$\text{s.t.} \mathbf{AX} = \mathbf{b}$$`

将变量 `$X$` 拆解为基变量 `$X_{B}$` 和非基变量 `$X_{N}$` 两部分，
即 `$X=[X_{B}, X_{N}]$`，同理，将 `$C$` 拆解为 `$C=[C_{B}, C_{N}]$`，
将 `$A$` 拆解为 `$A=[A_{B}, A_{N}]$` 两部分，则规划问题可以表示为：

`$$max Z = [C_{B}, C_{N}]^{T} \cdot [X_{B}, X_{N}]$$`
`$$\text{s.t.} [A_{B}, A_{N}]^{T} \cdot [X_{B}, X_{N}] = b$$`

假设前 `$m$` 个变量 `$x_{1}, x_{2},\cdots,x_{m}$` 为基变量，
后面的 `$x_{m+1}, \cdots, x_{n}$` 为非基变量，则约束方程可以写为如下形式：

`$$\text{s.t.}\begin{cases}
x_{1} \quad \quad + a_{1,m+1}x_{m+1}+\cdots+a_{1,n}x_{n} = b_{1} \\
\quad x_{2} \quad + a_{2,m+1}x_{m+1}+\cdots+a_{2,n}x_{n} = b_{3} \\
\quad \cdots \quad \\
\quad \quad x_{m} + a_{m,m+1}x_{m+1}+\cdots+a_{m,n}x_{n} = b_{m}
\end{cases}$$`

则基变量的值为：

`$$x_{i}=b_{i}-(a_{i,m+1}x_{m+1} + \cdots + a_{i,n}x_{n})=b_{i}-\sum_{j=m+1}^{n}a_{i,j}x_{j}$$`

将 `$x_{i}$` 代入目标函数，并消去目标函数中的基变量 `$X_{B}$`，则：

`$$\begin{align}
max Z
&=\sum_{j=1}^{n}c_{j}x_{j} \\ 
&=\sum_{i=1}^{m}c_{i}x_{i}+\sum_{j=m+1}^{n}c_{j}x_{j} \\
&=\sum_{i=1}^{m}c_{i}[b_{i}-\sum_{j=m+1}^{n}a_{ij}x_{j}]+\sum_{j=m+1}^{n}c_{j}x_{j} \\
&=\sum_{i=1}^{m}c_{i}b_{i}-\sum_{i=1}^{m}c_{i}a_{ij}x_{j}+\sum_{j=m+1}^{n}c_{j}x_{j} \\
&=\sum_{i=1}^{m}c_{i}b_{i}+\sum_{j=m+1}^{n}[c_{j}-\sum_{i=1}^{m}c_{i}a_{ij}]x_{j} \\
&=Z_{0}+\sum_{j=m+1}^{n}[c_{j}-Z_{j}]x_{j} \\
&=Z_{0} + \sum_{j=m+1}^{n}R_{j}x_{j} \\
\end{align}$$`

式中，目标函数 `$Z_{0}$` 的值为：

`$$\begin{align}
Z_{0} 
&= \sum_{j=1}^{m}c_{j}b_{j} \\
&= (c_{1}, c_{2}, \cdots, c_{m})(b_{1}, b_{2}, \cdots, b_{m})^{T} \\
&= \mathbf{C}_{B}\mathbf{b}
\end{align}$$`

`$R_{j}$` 为非基变量检验数，即：

`$$\begin{align}R_{j}
&=c_{j}-\sum_{i=1}^{m}c_{i}a_{ij} \\
&=c_{j}-(c_{1},c_{2},\cdots,c_{m})(a_{1j},a_{2j},\cdots,a_{mj})^{T} \\
&= c_{j}-\mathbf{C}_{B}a_{j}, j=m+1,\cdots,n
\end{align}$$`

根据上面的公式推导，可以计算出目标函数的值、非基变量的检验数，
同时也说明了单纯形表变换的内在数学原理。

### 单纯形法过程




### 单纯形法示例


## 内点法

内点法也是求解线性规划的一个方法，相比单纯形法，
内点法在 <span style='border-bottom:1.5px dashed red;'>大规模线性优化、
二次优化、非线性规划</span> 方面都有比较好的表现。
内点法是 <span style='border-bottom:1.5px dashed red;'>多项式算法</span>，
随着问题规模的增大，计算的复杂度却不会急剧增大，
因此在大规模问题上比单纯形法有更广泛的应用。

### 内点法原理

内点法的求解思路同拉格朗日松弛法的思路类似，将约束问题转化为无约束问题，
通过无约束函数的梯度下降进行迭代直至得到有效解。

内点法就是在梯度下降的过程中，如果当前迭代点是在可行域外，则会给损失函数一个非常大的值，
这样就能约束在可行域内求解。但是，内点法不能处理等式约束，因为构造的内点惩罚函数是定义在可行域内的函数，
而等式约束优化问题不存在可行域空间。由此看来，内点法和单纯形法对优化问题的形式是不一样的。

### 内点法过程





### 内点法示例


## 列生成法

# 对偶问题

# 拉格朗日乘子法


# 参考

* [十分钟快速掌握单纯形法](https://mp.weixin.qq.com/s?__biz=MzU0NzgyMjgwNg==&mid=2247484683&idx=1&sn=32fbd323572549ebe1d7ceca7e5c79dd&chksm=fb49c8b2cc3e41a4005d70d926c48e4c538ebd573d5ffbdeeba6b10dadc4d03012cc311249c8&scene=21#wechat_redirect)
