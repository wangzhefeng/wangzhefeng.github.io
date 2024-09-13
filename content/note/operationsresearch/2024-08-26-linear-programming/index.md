---
title: 线性规划
subtitle: Linear Programming, LP
author: 王哲峰
date: '2024-08-26'
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
        - [单纯形法原理](#单纯形法原理)
        - [单纯形法过程](#单纯形法过程)
        - [单纯形法示例](#单纯形法示例)
    - [内点法](#内点法)
        - [内点法原理](#内点法原理)
        - [内点法过程](#内点法过程)
        - [内点法示例](#内点法示例)
    - [列生成法](#列生成法)
        - [列生成法原理](#列生成法原理)
        - [列生成法过程](#列生成法过程)
        - [列生成法示例](#列生成法示例)
- [对偶问题](#对偶问题)
    - [对偶问题的形式](#对偶问题的形式)
    - [对称形式对偶](#对称形式对偶)
    - [对偶单纯形](#对偶单纯形)
    - [对偶问题的应用](#对偶问题的应用)
- [拉格朗日乘子法](#拉格朗日乘子法)
    - [无约束优化](#无约束优化)
    - [等式约束优化](#等式约束优化)
    - [不等式约束优化](#不等式约束优化)
    - [拉格朗日对偶](#拉格朗日对偶)
- [线性规划求解示例](#线性规划求解示例)
    - [单纯形法代码](#单纯形法代码)
    - [Scipy](#scipy)
        - [示例 1](#示例-1)
        - [示例 2](#示例-2)
    - [Pyomo](#pyomo)
    - [docplex](#docplex)
- [参考](#参考)
</p></details><p></p>

# 线性规划的标准型

对于线性规划，先来看一个简单的数学规划模型，即：

`$$\text{max} \space Z = 70x_{1} + 30x_{2}$$`

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

`$$\text{max} \space Z = 70x_{1} + 30x_{2}$$`

`$$s.t.\begin{cases}
3x_{1} + 9x_{2} + x_{3} = 540 \\
5x_{1} + 5x_{2} + x_{4} = 450 \\
9x_{1} + 3x_{2} + x_{5} = 720 \\
x_{1}, x_{2}, x_{3}, x_{4}, x_{5} \geq 0
\end{cases}$$`

将线性规划模型准换成标准型后，就可以使用经典的线性规划方法求解了，包括单纯形法、内点法、列生成法等。

# 线性规划的求解方式

## 单纯形法

单纯形法是求解线性规划的经典方法，与多元消去法求解多元一次方程的原理类似，在具体实现上，
<span style='border-bottom:1.5px dashed red;'>通过矩阵的变换对解空间进行搜索</span>。

由于线性规划模型的目标函数和约束方程都是凸函数，所以单纯形法能够以很高的效率求解线性规划问题。
单纯形法是最优算法的基础算法，也是后续其他整数规划等算法的基础。

### 单纯形法原理

由于线性规划模型中目标函数和约束方程都是凸函数，因此从凸优化的角度来说，
**线性规划的最优解在可行域的顶点上**，
单纯形法的本质就是通过矩阵的线性变换来遍历这些顶点以计算最优解。

假设有一个规划问题，即：

`$$\text{max} \space \mathbf{Z} = \mathbf{CX}$$`
`$$\text{s.t.} \space \mathbf{AX} = \mathbf{b}$$`

将变量 `$X$` 拆解为基变量 `$X_{B}$` 和非基变量 `$X_{N}$` 两部分，
即 `$X=[X_{B}, X_{N}]$`，同理，将 `$C$` 拆解为 `$C=[C_{B}, C_{N}]$`，
将 `$A$` 拆解为 `$A=[A_{B}, A_{N}]$` 两部分，则规划问题可以表示为：

`$$\text{max} \space Z = [C_{B}, C_{N}]^{T} \cdot [X_{B}, X_{N}]$$`
`$$\text{s.t.}\space [A_{B}, A_{N}]^{T} \cdot [X_{B}, X_{N}] = b$$`

假设 `$\mathbf{C}=(c_{1},c_{2},\cdots,c_{m},c_{m+1},\cdots,c_{n})$`；
`$\mathbf{X}=(x_{1},x_{2},\cdots,x_{m},x_{m+1},\cdots,x_{n})$`，
前 `$m$` 个变量 `$x_{1}, x_{2},\cdots,x_{m}$` 为基变量，
后面的 `$x_{m+1}, \cdots, x_{n}$` 为非基变量。`$\mathbf{b}=(b_{1},b_{2},\cdots,b_{m})$`，
则约束方程可以写为如下形式：

`$$\text{s.t.}\begin{cases}
x_{1} \quad \quad + a_{1,m+1}x_{m+1}+\cdots+a_{1,n}x_{n} = b_{1} \\
\quad x_{2} \quad + a_{2,m+1}x_{m+1}+\cdots+a_{2,n}x_{n} = b_{2} \\
\quad \cdots \quad \quad \cdots \\
\quad \quad x_{m} + a_{m,m+1}x_{m+1}+\cdots+a_{m,n}x_{n} = b_{m}
\end{cases}$$`

则基变量的值为：

`$$x_{i}=b_{i}-(a_{i,m+1}x_{m+1} + \cdots + a_{i,n}x_{n})=b_{i}-\sum_{j=m+1}^{n}a_{i,j}x_{j}, \space i=1, \cdots, m$$`

将基变量 `$x_{i}$` 代入目标函数，并消去目标函数中的基变量 `$X_{B}$`，则：

`$$\begin{align}
\text{max} \space Z
&=\sum_{j=1}^{n}c_{j}x_{j} \\ 
&=\sum_{i=1}^{m}c_{i}x_{i}+\sum_{j=m+1}^{n}c_{j}x_{j} \\
&=\sum_{i=1}^{m}c_{i}[b_{i}-\sum_{j=m+1}^{n}a_{ij}x_{j}]+\sum_{j=m+1}^{n}c_{j}x_{j} \\
&=\sum_{i=1}^{m}c_{i}b_{i}-\sum_{i=1}^{m}\sum_{j=m+1}^{n}c_{i}a_{ij}x_{j}+\sum_{j=m+1}^{n}c_{j}x_{j} \\
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

式中，`$R_{j}$` 为非基变量检验数，即：

`$$\begin{align}R_{j}
&=c_{j}-\sum_{i=1}^{m}c_{i}a_{ij} \\
&=c_{j}-(c_{1},c_{2},\cdots,c_{m})(a_{1j},a_{2j},\cdots,a_{mj})^{T} \\
&= c_{j}-\mathbf{C}_{B}a_{j},\space j=m+1,\cdots,n
\end{align}$$`

根据上面的公式推导，可以计算出目标函数的值、非基变量的检验数，
同时也说明了单纯形表变换的内在数学原理。

`$$\text{max} \space Z = \mathbf{C}_{B}\mathbf{b} + \sum_{j=m+1}^{n}(c_{j}-\mathbf{C}_{B}a_{j})x_{j}$$`

### 单纯形法过程

> 单纯形法就是遍历可行域各顶点后判断最优解

要用单纯形法求解线性规划数学模型，还需要把模型转化成规范形，规范形的条件如下：

1. 数学模型已经是标准型。
2. 约束方程组系数矩阵中含有至少一个单位子矩阵，对应的变量称为 **基变量**，
   **基**的作用是得到 **初始基本可行解**，这个初始基本可行解通常是**原点**。
   在大部分的问题中，通常引入**松弛变量**得到单位子矩阵，即使约束条件是等式约束，
   也可以引入 `$x_{N} = 0$` 的松弛变量。

    上述示例的约束方程组系数矩阵是：

    `$$A = \begin{bmatrix}
    3 & 9 & 1 & 0 & 0 \\
    5 & 5 & 0 & 1 & 0 \\
    9 & 3 & 0 & 0 & 1
    \end{bmatrix} = (a_{1}, a_{2}, a_{3}, a_{4}, a_{5})$$`

    对应的单位子矩阵是：

    `$$B = \begin{bmatrix}
    1 & 0 & 0 \\
    0 & 0 & 0 \\
    0 & 0 & 1 
    \end{bmatrix}=(a_{3}, a_{4}, a_{5})$$`

> 如何理解线性规划中的**基变量**和**非基变量**呢？
> 
> 线性规划的最优解只能在顶点处取到，所以单纯形法的思想就是从一个顶点出发，
> 连续访问不同的顶点，在每一个顶点处检查是否有相邻的其他顶点能取到更优的目标函数值。
> 
> 线性规划里面的约束（等式或不等式）可以看作是超平面（Hyperplane）或半空间（Half space）。
> **可行域**可以看作是被这组约束，或者超平面和半空间定义（围起来）的区域。
> 那么某一个顶点其实就是某组超平面的交点，这一组超平面对应的约束就是在某一个顶点取到 `"="` 号的约束（也就是基）。
> 顶点对应的代数意义就是一组方程（取到等号的约束）的解。

3. 目标函数中不含基变量。在这里基变量 `$x_{3}$`、`$x_{4}$` 和 `$x_{5}$` 是在约束方程中引进的变量，
   所以目标函数中没有这些基变量。

---

单纯形法的计算过程可以表示成单纯形表：

| 说明        | `$x_{1}$`    | `$x_{2}$`    | `$x_{3}$`    | `$x_{4}$`       | `$x_{5}$`     | `$b$` | `$\theta$` |
|-------------|--------------|--------------|--------------|-----------------|--------------|-------|------------|
| 目标函数系数 | `$c_{1}=70$` | `$c_{2}=30$` |              |                 |               |       |            |
| 约束 1      | `$a_{11}=3$` | `$a_{21}=9$` | `$a_{31}=1$` |                 |              | `$b_{1}=540$` |    |
| 约束 2      | `$a_{12}=5$` | `$a_{22}=5$` |              | `$a_{42}=1$`    |               | `$b_{2}=450$` |    |
| 约束 3      | `$a_{13}=9$` | `$a_{23}=3$` |              |                 | `$a_{53}=1$`| `$b_{3}=720$` |    |

单纯形法的具体计算过程如下：

1. **确定初始基本可行解。** 利用规范型的数学模型，整理出目标函数和约束方程的形式。

    `$$\text{max} \space Z = 70x_{1} + 30x_{2}$$`
    `$$s.t.\begin{cases}
    3x_{1} + 9x_{2} + x_{3} = 540 \\
    5x_{1} + 5x_{2} + x_{4} = 450 \\
    9x_{1} + 3x_{2} + x_{5} = 720 \\
    x_{1}, x_{2}, x_{3}, x_{4}, x_{5} \geq 0
    \end{cases}$$`

   令非基变量 `$x_{j}=0,j=1,2$`，这样可以直接得到基变量的取值，即 `$x_{3}=540$`，
   `$x_{4}=450$`，`$x_{5}=720$`，将非基变量 `$x_{j} = 0,j=1,2$` 代入目标函数得到 `$Z = 0$`，
   初始基本可行解是：

   `$$\mathbf{X}=(x_{1},x_{2},x_{3},x_{4},x_{5})=(0,0,540,450,720)^{T}$$` 
   `$$Z=0$$`

   此时顶点位置是原点 `$O$`。

2. **判断当前点 `$\mathbf{X}$` 是否为最优解。** 对于最大化问题，
   目标函数中非基变量的系数 `$a_{i} \leq 0$` 时为最优解。
   而这里非基变量的系数 `$a_{1}=70>0$`，`$a_{2}=30>0$`，
   意味着只要在可行域内随着非基变量 `$x_{1}$` 和 `$x_{2}$` 的增大，
   目标函数就会继续增大，所以此时的解不是最优解。
   
   当然也可以利用梯度的知识来思考这个问题。对于最大化问题，
   只需要沿着梯度方向搜索即可找到最大值。线性规划是一个典型的凸优化问题，
   当梯度为零时，得到最大值。即当非基变量系数 `$a_{j} \leq 0$` 时，可得到问题最优解。

3. **基变量出基与非基变量入基。** 变量的入基和出基在几何图上表现为顶点的变化，
   如从 `$a$` 顶点变换到 `$b$` 顶点。
   
    - 选择使目标函数 `$Z$` 变化最快的非基变量入基，
      即**选择目标函数系数 `$a_{i}$` 最大且为正数的非基变量入基**，所以选择 `$x_{1}$` 入基(非基变量变为基变量)，
      此时仍然 `$x_{2}=0$`。从凸优化的角度来看，就是选择目标函数梯度最大的方向做下一步的计算。
    - 那该选择哪个基变量出基呢？可以利用计算约束方程中常数项（`$b_{j}$`）与 `$x_{1}$` 系数（`$a_{1j}$`）的比值 `$\theta$`，
      选择最小的 `$\theta$` 对应的约束方程的基变量出基，即 `$\theta=\frac{b_{j}}{a_{i}}$`，根据下面的单纯形表，因此选择 `$x_{5}$` 出基(基变量变为非基变量)。

    | 目标和基变量 | `$x_{1}$`    | `$x_{2}$`    | `$x_{3}$`    | `$x_{4}$`       | `$x_{5}$`     | `$b$` | `$\theta$` |
    |-------------|--------------|--------------|--------------|-----------------|--------------|-------|------------|
    | `$\mathbf{Z}$` | `$c_{1}=70$` | `$c_{2}=30$` |              |                 |               |               |    |
    | `$x_{3}$`      | `$a_{11}=3$` | `$a_{21}=9$` | `$a_{31}=1$` |                 |               | `$b_{1}=540$` | `$\theta_{1}=540/3=180$` |
    | `$x_{4}$`      | `$a_{12}=5$` | `$a_{22}=5$` |              | `$a_{42}=1$`    |               | `$b_{2}=450$` | `$\theta_{2}=450/5=90$` |
    | `$x_{5}$`     | `$a_{13}=9$` | `$a_{23}=3$` |              |                 | `$a_{53}=1$`| `$b_{3}=720$` | `$\theta_{3}=720/9=80$` |
 
    因为选择 `$x_{1}$` 入基，`$x_{5}$` 出基，因此约束方程的第三个式子变成：

    `$$\text{s.t.}_{3,\text{new}} \space x_{1} + \frac{1}{3}x_{2} + \frac{1}{9}x_{5} = 80 \Rightarrow x_{1} = 80-\frac{1}{3}x_{2}-\frac{1}{9}x_{5}$$`

    约束方程第一个式子不变，即：

    `$$\text{s.t.}_{1,\text{old}}\space 3x_{1}+9x_{2}+x_{3}=540$$`

    根据多元消去法，将 `$\text{s.t.}_{3,\text{new}}$` 中的 `$x_{1}$` 代入 `$\text{s.t.}_{1}$` 中：

    `$$\text{s.t.}_{1,\text{new}} \space 3\Big(80-\frac{1}{3}x_{2}-\frac{1}{9}x_{5}\Big)+9x_{2}+x_{3}=540 \Rightarrow 0x_{1} + 8x_{2}+x_{3}-\frac{1}{3}x_{5}=300$$`

    即：

    `$$\text{s.t.}_{1, \text{new}}=\text{s.t.}_{1, \text{old}} - 3 \times \text{s.t.}_{3,\text{new}}$$`

    同理，得到其他约束方程和目标函数的替代，即新的目标函数和新的约束表达式形式。

    下面重新计算新的单纯形表：

    | 目标和基变量 | `$x_{1}$`    | `$x_{2}$`    | `$x_{3}$`    | `$x_{4}$`       | `$x_{5}$`     | `$b$` | `$\theta$` |
    |-------------|--------------|--------------|--------------|-----------------|--------------|-------|------------|
    | `$\mathbf{Z}$`|  | `$c_{2}=\frac{20}{3}$` |              |                 | `$c_{5}=-\frac{70}{3}$` |  `$5600$`             |    |
    | `$x_{3}$`      |  | `$a_{21}=8$`           | `$a_{31}=1$` |                 | `$a_{51}=-\frac{1}{3}$` | `$b_{1}=300$` | `$\theta_{1}$` |
    | `$x_{4}$`      |  | `$a_{22}=\frac{10}{3}$` |              | `$a_{42}=1$`    | `$a_{52}=-\frac{5}{9}$` | `$b_{2}=50$` | `$\theta_{2}$` |
    | `$x_{5}$`     | `$a_{13}=1$` | `$a_{23}=\frac{1}{3}$` |              |                 | `$a_{53}=\frac{1}{9}$`| `$b_{3}=80$` | `$\theta_{3}$` |

    新的问题形式为：

    `$$\text{max} \space Z = \frac{20}{3}x_{2} - \frac{70}{3}x_{5}$$`
    `$$s.t.\begin{cases}
    -\frac{1}{3}x_{5} + 8x_{2} + x_{3} = 300 \\
    -\frac{5}{9}x_{5} + \frac{10}{3}x_{2} + x_{4} = 50 \\
    \frac{1}{9}x_{5} + \frac{1}{3}x_{2} + x_{1} = 80 \\
    x_{1}, x_{2}, x_{3}, x_{4}, x_{5} \geq 0
    \end{cases}$$`

4. **计算新的解 `$\mathbf{X}$`。** 令非基变量 `$x_{j}=0, j=2,5$`，
   求出基变量 `$x_{i}=b_{i},i=1,3,4$`，得到基变量的值：`$x_{1}=80, x_{3}=300, x_{4}=50$`，即：

    `$$\mathbf{X}=(x_{1}, x_{2}, x_{3},x_{4}, x_{5}) = (80,0,300,50,0)^{T}$$`
    `$$\mathbf{Z} =5600$$`

    变换后，`$x_{1}$` 的值从 0 变成 80，称为入基；`$x_{5}$` 的值从 720 变成 0，称为出基，此时对应顶点为 `$a$` 点。

5. **判断当前解 `$\mathbf{X}$` 是否最优。** 由于目标函数中 `$x_{2}$` 的系数仍然大于 0，因此当前位置还不是最优，
   因为在可行域内随着 `$x_{2}$` 增大目标函数还会增大。
6. **基变量出基与非基变量入基。** 在目标函数中，系数为正且最大的变量是 `$x_{2}$`，
   因此选择 `$x_{2}$` 入基，并计算 `$\theta$` 选择出基变量。

    | 目标和基变量 | `$x_{1}$`    | `$x_{2}$`    | `$x_{3}$`    | `$x_{4}$`       | `$x_{5}$`     | `$b$` | `$\theta$` |
    |-------------|--------------|--------------|--------------|-----------------|--------------|-------|------------|
    | `$\mathbf{Z}$`|  | `$c_{2}=\frac{20}{3}$` |              |                 | `$c_{5}=-\frac{70}{3}$` |  `$5600$`             |    |
    | `$x_{3}$`      |  | `$a_{21}=8$`           | `$a_{31}=1$` |                 | `$a_{51}=-\frac{1}{3}$` | `$b_{1}=300$` | `$\theta_{1}=37.5$` |
    | `$x_{4}$`      |  | `$a_{22}=\frac{10}{3}$` |              | `$a_{42}=1$`    | `$a_{52}=-\frac{5}{9}$` | `$b_{2}=50$` | `$\theta_{2}=15$` |
    | `$x_{1}$`     | `$a_{13}=1$` | `$a_{23}=\frac{1}{3}$` |              |                 | `$a_{53}=\frac{1}{9}$`| `$b_{3}=80$` | `$\theta_{3}=240$` |

    经过计算，选择 `$x_{4}$` 出基，重新计算单纯形表：

    | 目标和基变量 | `$x_{1}$`    | `$x_{2}$`    | `$x_{3}$`    | `$x_{4}$`       | `$x_{5}$`     | `$b$` | `$\theta$` |
    |-------------|--------------|--------------|--------------|-----------------|--------------|-------|------------|
    | `$\mathbf{Z}$`|  | `$c_{2}=-2$` |              |                 | `$c_{5}=-\frac{20}{3}$` |  `$5600$`             |    |
    | `$x_{3}$`    |  |  | `$a_{31}=1$` | `$a_{41}=\frac{12}{5}$` | `$a_{51}=-\frac{5}{3}$` | `$b_{1}=180$` | `$\theta_{1}$` |
    | `$x_{2}$`    |  | `$a_{22}=1$` |              | `$a_{42}=\frac{3}{10}$` | `$a_{52}=-\frac{1}{6}$` | `$b_{2}=15$` | `$\theta_{2}$` |
    | `$x_{1}$`    | `$a_{13}=1$` |  |              | `$a_{43}=\frac{1}{10}$` | `$a_{53}=\frac{1}{6}$`| `$b_{3}=75$` | `$\theta_{3}$` |

    新的问题形式为：

    `$$\text{max} \space Z = -2x_{2} - \frac{20}{3}x_{5}$$`
    `$$s.t.\begin{cases}
    \frac{12}{5}x_{4} - \frac{5}{3}x_{5} + x_{3} = 180 \\
    \frac{3}{10}x_{4} - \frac{1}{6}x_{5} + x_{2} = 15 \\
    \frac{1}{10}x_{4} + \frac{1}{6}x_{5} + x_{1} = 75 \\
    x_{1}, x_{2}, x_{3}, x_{4}, x_{5} \geq 0
    \end{cases}$$`

7. **确定新的解 `$\mathbf{X}$`。** 令非基变量 `$x_{j}=0, j=4,5$`，求出基变量 `$x_{i}=b_{i}, i=1,2,3$`，
   得到基变量的值 `$x_{1}=75, x_{2}=15, x_{3}=180$`，即：

    `$$\mathbf{X}=(x_{1}, x_{2}, x_{3},x_{4}, x_{5}) = (75,15,180,0,0)^{T}$$`
    `$$\mathbf{Z} =5700$$`

    变换后，`$x_{2}$` 的值从 0 变成 15，称为入基；`$x_{4}$` 的值从 50 变成 0，称为出基，此时对应顶点为 `$h$` 点。

8. **判断当前解 `$\mathbf{X}$` 是否最优。** 因为函数中所有变量的系数均小于 0，变量 `$x_{4}$` 和 `$x_{5}$` 变大只会使目标函数减小，
   所以当前解是最优解，即：

    `$$\mathbf{X}=(x_{1}, x_{2}, x_{3},x_{4}, x_{5}) = (75,15,180,0,0)^{T}$$`
    `$$\mathbf{Z} =5700$$`

    经过单纯形法的搜索，搜索路径是 `$O \rightarrow a \rightarrow$`，
    而不是 `$O \rightarrow b \rightarrow k \rightarrow h$`，最终得到最优解。

### 单纯形法示例

```python
# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd


# 定义线性规划求解函数
def lp_solver(matrix: pd.DataFrame):
    """
    输入线性规划的矩阵，根据单纯形法求解线性规划模型
    max cx
    s.t. ax <= b

    Args:
        matrix (pd.DataFrame): 
            b      x1    x2    x3   x4   x5
        obj 0.0    70.0  30.0  0.0  0.0  0.0
        x3  540.0  3.0   9.0   1.0  0.0  0.0
        x4  450.0  5.0   5.0   0.0  1.0  0.0
        x5  720.0  9.0   3.0   0.0  0.0  1.0

        - 第 1 行是目标函数的系数
        - 第 2~4 行是约束方程的系数
        - 第 1 列是约束方程的常数项
        - obj-b 交叉，即第 1 行第 1 列的元素是目标函数的负值
        - x3,x4,x5 既是松弛变量，也是初始可行解
    """
    # 检验数是否大于 0
    c = matrix.iloc[0, 1:]
    # TODO
    while c.max() > 0:
        # ------------------------------
        # 选择入基变量 
        # ------------------------------
        # 目标函数系数最大的变量入基
        c = matrix.iloc[0, 1:]
        print(c)
        in_x = c.idxmax()
        print(in_x)
        # in_x_v = c[in_x]  # 入基变量的系数
        # print(in_x_v)
        print("-" *40)
        # ------------------------------
        # 选择出基变量
        # ------------------------------
        # 选择正的最小比值对应的变量出基 min(b列/入基变量列)
        b = matrix.iloc[1:, 0]
        print(b)
        in_x_a = matrix.iloc[1:][in_x]  # 选择入基变量对应的列
        print(in_x_a)
        out_x = (b / in_x_a).idxmin()  # 得到出基变量
        print(out_x)
        # out_x_v = b[out_x]  # 出基变量的系数
        # print(out_x_v)
        print("-" * 40)
        # ------------------------------
        # 旋转操作
        # ------------------------------
        matrix.loc[out_x, :] = matrix.loc[out_x, :] / matrix.loc[out_x, in_x]
        # print(matrix)
        # print("-" * 40)
        for idx in matrix.index: 
            if idx != out_x:
                matrix.loc[idx, :] = matrix.loc[idx, :] - matrix.loc[out_x, :] * matrix.loc[idx, in_x]
        # print(matrix)
        # print("-" * 40)
        # 索引替换（入基与出基变量名称替换）
        matrix_index = matrix.index.tolist()
        i = matrix_index.index(out_x)
        print(matrix_index)
        print(in_x, out_x)
        print(i)
        matrix_index[i] = in_x
        print(matrix_index)
        print("-" * 40)
        matrix.index = matrix_index 
    # 打印结果
    print("最终的最优单纯形法是：")
    print(matrix)
    print(f"目标函数值是：{-matrix.iloc[0, 0]}")
    print("最优决策变量是：")
    x_count = (matrix.shape[1] - 1) - (matrix.shape[0] - 1)
    X = matrix.iloc[0, 1:].index.tolist()[:x_count]
    for xi in X:
        print(f"{xi} = {matrix.loc[xi, 'b']}")


# 测试代码 main 函数
def main():
    # 约束方程系数矩阵，包含常数项
    matrix = pd.DataFrame(
        np.array([
            [0.0, 70.0, 30.0, 0.0, 0.0, 0.0],
            [540.0, 3.0, 9.0, 1.0, 0.0, 0.0],
            [450.0, 5.0, 5.0, 0.0, 1.0, 0.0],
            [720.0, 9.0, 3.0, 0.0, 0.0, 1.0],
        ]),
        index = ["obj", "x3", "x4", "x5"],
        columns = ["b", "x1", "x2", "x3", "x4", "x5"]
    )
    print(matrix)
    print("-" * 40)
    # 调用前面定义的函数求解
    lp_solver(matrix)

if __name__ == "__main__":
    main()
```

```
最终的最优单纯形法是：
          b   x1   x2   x3   x4        x5
obj -5700.0  0.0  0.0  0.0 -2.0 -6.666667
x3    180.0  0.0  0.0  1.0 -2.4  1.000000
x2     15.0  0.0  1.0  0.0  0.3 -0.166667
x1     75.0  1.0  0.0  0.0 -0.1  0.166667
目标函数值是：5700.0
最优决策变量是：
x1 = 75.0
x2 = 15.0
```

## 内点法

内点法也是求解线性规划的一个方法，相比单纯形法，
内点法在 <span style='border-bottom:1.5px dashed red;'>大规模线性优化、
二次优化、非线性规划</span> 方面都有比较好的表现。
内点法是 <span style='border-bottom:1.5px dashed red;'>多项式算法</span>，
随着问题规模的增大，计算的复杂度却不会急剧增大，
因此在大规模问题上比单纯形法有更广泛的应用。

### 内点法原理

内点法的求解思路同 **拉格朗日松弛法** 的思路类似，**将约束问题转化为无约束问题，
通过无约束函数的梯度下降进行迭代直至得到有效解**。

内点法就是在梯度下降的过程中，如果当前迭代点是在可行域外，则会给损失函数一个非常大的值，
这样就能约束在可行域内求解。但是，**内点法不能处理等式约束**，
因为构造的内点惩罚函数是定义在可行域内的函数，而等式约束优化问题不存在可行域空间。
由此看来，内点法和单纯形法对优化问题的形式是不一样的。

下面介绍一下内点法是如何将**约束问题**转化为**无约束问题**的。

考虑一个最小化的线性规划问题：

`$$min Z = c^{T}x$$`
`$$s.t. Ax \leq b$$`

借鉴拉格朗日松弛法的思路，这个线性规划问题可以表示成如下函数：

`$$min f(x) = c^{T}x + \sum_{i=1}^{m}I(Ax - b)$$`

其中，`$m$` 是约束方程的个数，`$I$` 是指示函数，一般定义如下：

`$$I(u) = \begin{cases}
0, \text{if} \space u \leq 0 \\
\infty, \text{if} \space u > 0
\end{cases}$$`

通过指示函数可以将约束方程直接写到目标函数中，然后对目标函数求极小值。
但是这个指示函数 `$I(u)$` 是不可导的，需要用其他可导的函数近似替代，
常用的替代函数如下：

`$$I_{-}(u)=-\frac{1}{t}log(-u)$$`

当 `$u>0$` 时，`$I_{-}(u)=\infty$`，参数 `$t$` 决定 `$I_{-}(u)$` 对应 `$I(u)$` 的近似程度，
类似机器学习中损失函数的正则化参数的作用。因此，新的目标函数可以写成如下形式：

`$$\begin{align}
min f(x)
&=c^{T}x-\frac{1}{t}\sum_{i=1}^{m}log(-Ax+b)  \\ 
&=tc^{T}x - \sum_{i=1}^{m}log(-Ax+b)
\end{align}$$`

由于指示函数 `$I_{-}(u)$` 是凸函数，所以新的目标函数也是凸函数，
因此可以用凸优化中的方法求解该函数的极小值，例如梯度下降法、牛顿法、拟牛顿法、L-BFGS 等。
下面以经典的**牛顿法**讲解如何求函数最小化问题。

目标函数 `$f(x)$` 在 `$x_{0}$` 做二阶泰勒公式展开时得到：

`$$f(x) \approx f(x_{0}) + (x-x_{0})f'(x_{0}) + \frac{1}{2}(x-x_{0})^{2}f''(x_{0})$$`

上式成立的条件是 `$f(x)$` 近似等于 `$f(x_{0})$`，对上面等式两边同时对 `$(x-x_{0})$` 求导，
并令导数为 0，可以得到下面的方程：

`$$f'(x) = f'(x_{0})+(x-x_{0})f''(x_{0})$$`
`$$x = x_{0}-\frac{f'(x_{0})}{f''(x_{0})}$$`

这样就得到了下一点的位置，从 `$x_{0}$` 走到 `$x_{1}$`。
重复这个过程，直到到达导数为 0 的点，因此牛顿法的迭代公式是：

`$$x_{n+1}=x_{n}-\frac{f'(x_{n})}{f''(x_{n})}=x_{n}-H^{-1}\nabla f(x_{n})$$`

式中，`$H^{-1}$` 表示二阶导数矩阵（黑塞矩阵）的逆。因此，如果使用牛顿法求解目标函数最优值，
需要知道目标函数的一阶导数 `$f'(x_{n})$` 和二阶导数 `$f''(x_{n})$`。

### 内点法过程

再看前面的线性规划问题：

`$$\text{max} \space Z = 70x_{1} + 30x_{2}$$`
`$$s.t.\begin{cases}
3x_{1} + 9x_{2} \leq 540 \\
5x_{1} + 5x_{2} \leq 450 \\
9x_{1} + 3x_{2} \leq 720 \\
x_{1}, x_{2} \geq 0
\end{cases}$$`

问题转换为内点法需要的形式：

`$$\text{min} \space Z = -70x_{1} - 30x_{2}$$`
`$$s.t.\begin{cases}
3x_{1} + 9x_{2} - 540 \leq 0 \\
5x_{1} + 5x_{2} - 450 \leq 0 \\
9x_{1} + 3x_{2} - 720 \leq 0 \\
-x_{1} \leq 0 \\
-x_{2} \leq 0
\end{cases}$$`

问题转换为无约束优化问题形式：

`$$\begin{align}
\text{min} Z 
&= t(-70x_{1} - 30x_{2}) \\
&=\text{log}(-3x_{1} - 9x_{2} + 540) - \\
&=\text{log}(-5x_{1} - 5x_{2} + 450) - \\
&=\text{log}(-9x_{1} - 3x_{2} + 720) - \\
&=\text{log}(x_{1}) - \\
&=\text{log}(x_{2})
\end{align}$$`

在问题中，目标函数的一阶导数是：

`$$J=\begin{bmatrix}
\frac{\partial f}{\partial x_{1}} & \frac{\partial f}{\partial x_{2}}
\end{bmatrix}$$`

`$$\frac{\partial f}{\partial x_{1}} = -70t+\frac{3}{-3x_{1}-9x_{2}+540}+\frac{5}{-5x_{1}-5x_{2}+450}+\frac{9}{-9x_{1}-3x_{2}+720}+\frac{1}{-x_{1}}$$`
`$$\frac{\partial f}{\partial x_{2}} = -30t+\frac{9}{-3x_{1}-9x_{2}+540}+\frac{5}{-5x_{1}-5x_{2}+450}+\frac{3}{-9x_{1}-3x_{2}+720}+\frac{1}{-x_{2}}$$`

目标函数的二阶导数是：

`$$H=\begin{bmatrix}
\frac{\partial^{2} f}{\partial x_{1} \partial x_{1}} & \frac{\partial^{2} f}{\partial x_{1} \partial x_{2}} \\
\frac{\partial^{2} f}{\partial x_{2} \partial x_{1}} & \frac{\partial^{2} f}{\partial x_{2} \partial x_{2}}
\end{bmatrix}$$`

`$$\frac{\partial^{2} f}{\partial x_{1} \partial x_{1}} = \frac{9}{(3x_{1}+x_{2}-240)^{2}} + \frac{1}{(x_{1}+3x_{2}-180)^{2}} + \frac{1}{(x_{1}+x_{2}-90)^{2}} + \frac{1}{x_{1}^{2}}$$`
`$$\frac{\partial^{2} f}{\partial x_{1} \partial x_{2}} = \frac{3}{(3x_{1}+x_{2}-240)^{2}} + \frac{3}{(x_{1}+3x_{2}-180)^{2}} + \frac{1}{(x_{1}+x_{2}-90)^{2}}$$`
`$$\frac{\partial^{2} f}{\partial x_{2} \partial x_{1}} = \frac{3}{(3x_{1}+x_{2}-240)^{2}} + \frac{3}{(x_{1}+3x_{2}-180)^{2}} + \frac{1}{(x_{1}+x_{2}-90)^{2}}$$`
`$$\frac{\partial^{2} f}{\partial x_{2} \partial x_{2}} = \frac{1}{(3x_{1}+x_{2}-240)^{2}} + \frac{9}{(x_{1}+3x_{2}-180)^{2}} + \frac{1}{(x_{1}+x_{2}-90)^{2}} + \frac{1}{x_{2}^{2}}$$`

选择一个恰当的初始解 `$x_{0}$` 代入牛顿法迭代公式：

`$$x_{n+1}=x_{n}-H^{-1}\nabla f$$`

不断迭代直至得到最优解。

总结：内点法和单纯形法的结果相差很大，只是因为内点法的搜索路径是在可行域内部，
而不在可行域的边界上，这也是内点法的局限性。并且通过前面的求解过程发现，
内点法不仅局限在线性规划上，二次规划等也是可以求解的，因为其本质是利用函数梯度求最优值，
这同机器学习算法的思路是一致的，真正的难点在于如何保证新的目标函数是否存在一阶导数和二阶导数，
以及如何得到一阶导数和二阶导数的信息。此外，初始迭代点的选择也是很重要的，
在线性规划问题中能够保证最后得到的是最优解，而非线性规划问题中，函数是非凸的，
因此很难保证最后的解是全局最优解。

### 内点法示例

```python
import time
import numpy as np

def gradient(x1, x2, t):
    """
    计算目标函数在 x 处的一阶导数（雅可比矩阵 ）

    Args:
        x1 (_type_): _description_
        x2 (_type_): _description_
        t (_type_): _description_
    """
    j1 = -70*t + 3/(-3*x1 - 9*x2 + 540) \
               + 5/(-5*x1 - 5*x2 + 450) \
               + 9/(-9*x1 - 3*x2 + 720) \
               - 1/x1
    j2 = -30*t + 9/(-3*x1 - 9*x2 + 540) \
               + 5/(-5*x1 - 5*x2 + 450) \
               + 3/(-9*x1 - 3*x2 + 720) \
               - 1/x2

    return np.asmatrix([j1, j2]).T


def hessian(x1, x2):
    """
    计算目标函数在 x 处的二阶导数（黑塞矩阵）

    Args:
        x1 (_type_): _description_
        x2 (_type_): _description_
    """
    x1, x2 = float(x1), float(x2)
    h11 = 9/(3*x1 + x2 - 240)**2 + (x1 + 3*x2 - 180)**(-2) + (x1 + x2 - 90)**(-2) + x1**(-2)
    h12 = 3/(3*x1 + x2 - 240)**2 + 3/(x1 + 3*x2 - 180)**2 + (x1 + x2 - 90)**(-2)
    h21 = 3/(3*x1 + x2 - 240)**2 + 3/(x1 + 3*x2 - 180)**2 + (x1 + x2 - 90)**(-2)
    h22 = (3*x1 + x2 - 240)**(-2) + 9/(x1 + 3*x2 - 180)**2 + (x1 + x2 - 90)**(-2) + x2**(-2)
    
    return np.asmatrix([[h11, h12], [h21, h22]])


def invertible(H):
    """
    求黑塞矩阵的逆矩阵

    Args:
        H (_type_): _description_
    """
    H_inv = np.linalg.inv(H)

    return H_inv


def run():
    # 牛顿法的初始迭代值
    x = np.asmatrix(np.array([10, 10])).T
    # 指数函数中的 t
    t = 0.00001
    # 迭代停止的误差
    eps = 0.01
    # 记录迭代的次数
    iter_cnt = 0
    while iter_cnt < 20:
        iter_cnt += 1
        J = gradient(x[0, 0], x[1, 0], t)
        H = hessian(x[0, 0], x[1, 0])
        H_inv = invertible(H)
        # 牛顿法
        x_new = x - H_inv * J
        # 求二范数，判断迭代效果
        error = np.linalg.norm(x_new - x)
        print(f"迭代次数是：{iter_cnt}, x1={x_new[0, 0]:.2f}, x2={x_new[1, 0]:.2f}, 误差是: {error}")
        x = x_new
        if error < eps:
            break
        time.sleep(1)
    # 打印结果
    print(f"目标函数值是：{70*x[0, 0] + 30*x[1, 0]:.2f}")


# 测试代码 main 函数
def main():
    run()

if __name__ == "__main__":
    main()

```

```
迭代次数是：1, x1=15.91, x2=15.34, 误差是: 7.964345953591451
迭代次数是：2, x1=20.13, x2=18.19, 误差是: 5.09056325893654
迭代次数是：3, x1=21.00, x2=18.33, 误差是: 0.8795427059613212
迭代次数是：4, x1=21.02, x2=18.32, 误差是: 0.02676438103520062
迭代次数是：5, x1=21.02, x2=18.32, 误差是: 2.199053220249811e-05
目标函数值是：2021.17
```

## 列生成法

列生成法是一种用于求解**大规模线性优化问题**非常高效的算法。
本质上，**列生成算法就是单纯形法的一种形式**，它是用来求解线性规划问题的，
所不同的列生成法**改善了大规模优化问题中单纯形法基变换计算效率低的问题**，
列生成法在**整数规划**中已经得到了广泛应用。

### 列生成法原理

列生成法主要用于解决变量很多而约束相对较少的问题，特别是经常用于解决大规模整数规划问题。

单纯形法虽然能保证在数次迭代后找到最优解，但是其面对变量很多的线性规划问题就显得很弱了。
因为它需要在众多变量里进行基变换，所以这种遍历的计算量是很大的。

有人基于单纯形法提出了列生成法，
其思路是强制原问题(Master Problem)把一部分变量限定(Restrict)为基变量，
得到一个规模更小（即变量数比原问题少的）的限制性主问题(Restrict Master Problem)，
在限制主问题上用单纯形法求解最优解，但是此时求得的最优解只是限制性主问题的解，
并不是原问题的最优解，就需要通过一个子问题(Subproblem)去检查在那些未被考虑的变量中，
是否有使限制主问题的 ReducedCost 小于 0，如果有，就把这个变量的相关系数列加入到限制主问题的系数矩阵中。

列生成法的形象化表示过程如下，考虑线性规划问题，即：

`$$\text{min}c_{1}x_{1}+c_{2}x_{2}+\cdots+c_{n}x_{n}$$`
`$$\text{s.t.}\begin{cases}
a_{11}x_{1} + a_{12}x_{2} + \cdots + a_{1n}x_{n} = b_{1} \\
\cdots \\
a_{m1}x_{1} + a_{m2}x_{2} + \cdots + a_{mn}x_{n} = b_{m} \\
\end{cases}$$`

把前面 `$j$` 个变量强制设定为基变量，后面 `$n-j$` 个变量设定为非基变量，由于非基变量等于 0，
用矩阵表示的线性规划形式是：

`$$\text{min} c_{1}x_{1}+c_{2}x_{2}+\cdots+c_{j}x_{j}$$`
`$$\text{s.t.}\begin{bmatrix}
a_{11} & \cdots & a_{1j} \\
\cdots & \cdots & \cdots \\
a_{m1} & \cdots & a_{mj} \\
\end{bmatrix}x = b$$`

此时的问题为**限定主问题**，通过求解限定主问题先得到**对偶问题的最优解**，
然后用子问题检查是否存在新的变量使目标变量继续朝优化方向变化（检验数小于 0），
假设存在一个满足的变量 `$x_{j+1}$`，那么原问题则变成：

`$$\text{min} c_{1}x_{1}+c_{2}x_{2}+\cdots+c_{j}x_{j}+c_{j+1}x_{j+1}$$`
`$$\text{s.t.}\begin{bmatrix}
a_{11} & \cdots & a_{1j} & a_{1j+1} \\
\cdots & \cdots & \cdots & \cdots \\
a_{m1} & \cdots & a_{mj} & a_{mj+1} \\
\end{bmatrix}x = b$$`

此时系数矩阵多了一列，这就是列生成法名称的由来。此时求解**新限定主问题**得到**对偶问题的最优解**，
通过子问题得到新的变量，如此循环往复直到无法添加新的变量。

---

接下来讲解限定主问题和子问题的关系。

假设有如下线性规划问题：

`$$\text{min} c^{T}x$$`
`$$\text{s.t.}\begin{cases}
Ax = b \\
x \geq 0 
\end{cases}$$`

与单纯形的数学规范形的思路类似，令 `$x=[x_{B}, x_{N}]$`，其中 `$x_{B}$` 表示基变量，
`$x_{N}$` 表示非基变量。类似地，`$A = [A_{B}, A_{N}]$`，`$c^{T} = [c_{B}^{T}, c_{N}^{T}]$`，
约束方程变成：

`$$Ax = b$$`
`$$Bx_{B} + Nx_{N} = b$$`
`$$x_{B}=B^{-1}b-B^{-1}Nx_{N}$$`

因为非基变量 `$x_{N} = 0$`，所以 `$x_{B} = B^{-1}b$`。

对目标函数有如下推导关系：

`$$\begin{align}
\text{min} c^{T}x 
&\Leftrightarrow \text{min}c_{B}^{T}x_{B} + c_{N}^{T}x_{N} \\
&\Leftrightarrow \text{min}c_{B}^{T}(B^{-1}b - B^{-1}Nx_{N}) + c^{T}_{N}x_{N} \\
&\Leftrightarrow \text{min}(c_{N}^{T} - c_{B}^{T}B^{-1}N)x_{N} + c_{B}^{T}B^{-1}b
\end{align}$$`

其中：

* `$c_{B}^{T}B^{-1}b$` 是常数
* `$x_{N}$` 非负
* `$c_{N}^{T}-c_{B}^{T}B^{-1}N$` 称为 ReducedCost，
  ReducedCost 即非基变量的检验数，如果 `$\text{ReducedCost} < 0$`，
  则说明有非基变量 `$x_{N}$` 使得目标函数可以更优。



### 列生成法过程



### 列生成法示例

```python

```

# 对偶问题

线性规划的对偶问题可以将原问题和对偶问题看成是一个问题的两个视角，

如果在一定资源下如何安排生产才能使利润最大，
这个问题的另一个角度就是怎样购买这些生产资源使花钱最少。

从数学的角度来说，如果原问题不好求解，可以尝试从对偶问题的角度出发求解原问题，
如在求最小问题中，对偶问题就是寻找原问题目标函数的下界。

## 对偶问题的形式

## 对称形式对偶


## 对偶单纯形


## 对偶问题的应用




# 拉格朗日乘子法

## 无约束优化


## 等式约束优化


## 不等式约束优化


## 拉格朗日对偶

# 线性规划求解示例

## 单纯形法代码

```python
# -*- coding: utf-8 -*-

# ***************************************************
# * File        : simplex_method.py
# * Author      : Zhefeng Wang
# * Email       : wangzhefengr@163.com
# * Date        : 2024-08-26
# * Version     : 0.1.082616
# * Description : description
# * Link        : link
# * Requirement : 相关模块版本需求(例如: numpy >= 2.1.0)
# ***************************************************

# python libraries
import numpy as np
import pandas as pd

# global variable
LOGGING_LABEL = __file__.split('/')[-1][:-3]


# 定义线性规划求解函数
def lp_solver(matrix: pd.DataFrame):
    """
    输入线性规划的矩阵，根据单纯形法求解线性规划模型
    max cx
    s.t. ax <= b

    Args:
        matrix (pd.DataFrame): 
            b      x1    x2    x3   x4   x5
        obj 0.0    70.0  30.0  0.0  0.0  0.0
        x3  540.0  3.0   9.0   1.0  0.0  0.0
        x4  450.0  5.0   5.0   0.0  1.0  0.0
        x5  720.0  9.0   3.0   0.0  0.0  1.0

        - 第 1 行是目标函数的系数
        - 第 2~4 行是约束方程的系数
        - 第 1 列是约束方程的常数项
        - obj-b 交叉，即第 1 行第 1 列的元素是目标函数的负值
        - x3,x4,x5 既是松弛变量，也是初始可行解
    """
    # 检验数是否大于 0
    c = matrix.iloc[0, 1:]
    # TODO
    while c.max() > 0:
        # ------------------------------
        # 选择入基变量 
        # ------------------------------
        # 目标函数系数最大的变量入基
        c = matrix.iloc[0, 1:]
        print(c)
        in_x = c.idxmax()
        print(in_x)
        # in_x_v = c[in_x]  # 入基变量的系数
        # print(in_x_v)
        print("-" *40)
        # ------------------------------
        # 选择出基变量
        # ------------------------------
        # 选择正的最小比值对应的变量出基 min(b列/入基变量列)
        b = matrix.iloc[1:, 0]
        print(b)
        in_x_a = matrix.iloc[1:][in_x]  # 选择入基变量对应的列
        print(in_x_a)
        out_x = (b / in_x_a).idxmin()  # 得到出基变量
        print(out_x)
        # out_x_v = b[out_x]  # 出基变量的系数
        # print(out_x_v)
        print("-" * 40)
        # ------------------------------
        # 旋转操作
        # ------------------------------
        matrix.loc[out_x, :] = matrix.loc[out_x, :] / matrix.loc[out_x, in_x]
        # print(matrix)
        # print("-" * 40)
        for idx in matrix.index: 
            if idx != out_x:
                matrix.loc[idx, :] = matrix.loc[idx, :] - matrix.loc[out_x, :] * matrix.loc[idx, in_x]
        # print(matrix)
        # print("-" * 40)
        # 索引替换（入基与出基变量名称替换）
        matrix_index = matrix.index.tolist()
        i = matrix_index.index(out_x)
        print(matrix_index)
        print(in_x, out_x)
        print(i)
        matrix_index[i] = in_x
        print(matrix_index)
        print("-" * 40)
        matrix.index = matrix_index 
    # 打印结果
    print("最终的最优单纯形法是：")
    print(matrix)
    print(f"目标函数值是：{-matrix.iloc[0, 0]}")
    print("最优决策变量是：")
    x_count = (matrix.shape[1] - 1) - (matrix.shape[0] - 1)
    X = matrix.iloc[0, 1:].index.tolist()[:x_count]
    for xi in X:
        print(f"{xi} = {matrix.loc[xi, 'b']}")


# 测试代码 main 函数
def main():
    # 约束方程系数矩阵，包含常数项
    matrix = pd.DataFrame(
        np.array([
            [0.0, 70.0, 30.0, 0.0, 0.0, 0.0],
            [540.0, 3.0, 9.0, 1.0, 0.0, 0.0],
            [450.0, 5.0, 5.0, 0.0, 1.0, 0.0],
            [720.0, 9.0, 3.0, 0.0, 0.0, 1.0],
        ]),
        index = ["obj", "x3", "x4", "x5"],
        columns = ["b", "x1", "x2", "x3", "x4", "x5"]
    )
    print(matrix)
    print("-" * 40)
    # 调用前面定义的函数求解
    lp_solver(matrix)

if __name__ == "__main__":
    main()
```

```
最终的最优单纯形法是：
          b   x1   x2   x3   x4        x5
obj -5700.0  0.0  0.0  0.0 -2.0 -6.666667
x3    180.0  0.0  0.0  1.0 -2.4  1.000000
x2     15.0  0.0  1.0  0.0  0.3 -0.166667
x1     75.0  1.0  0.0  0.0 -0.1  0.166667
目标函数值是：5700.0
最优决策变量是：
x1 = 75.0
x2 = 15.0
```

## Scipy

### 示例 1

```python
# -*- coding: utf-8 -*-
import numpy as np
from scipy import optimize as op


'''
线性规划

问题：
    max z = 2x1 + 3x2 - 5x3
    s.t. x1 + x2 + x3 = 7
         2x1 - 5x2 + x3 >= 10
         x1 + 3x2 + x3 <= 12
         x1, x2, x3 >= 0

API：
    scipy.optimize.linprog(
        c, 
        A_ub = None, 
        b_ub = None, 
        A_eq = None, 
        b_eq = None, 
        bounds = None, 
        method = 'simplex', 
        callback = None, 
        options = None
    )

参数：
    * c 函数系数数组, 最大化参数为c, 最小化为-c, 函数默认计算最小化. 
    * A_ub 不等式未知量的系数, 默认转成 <= , 如果原式是 >= 系数乘负号. 
    * B_ub 对应A_ub不等式的右边结果
    * A_eq 等式的未知量的系数
    * B_eq 等式的右边结果
    * bounds 每个未知量的范围
'''


# 目标函数
c = np.array([2, 3, -5])

A_ub = np.array([[-2, 5, -1], [1, 3, 1]])
B_ub = np.array([-10, 12])
A_eq = np.array([[1, 1, 1]])
B_eq = np.array([7])
x1 = (0, 7)
x2 = (0, 7)
x3 = (0, 7)
res = op.linprog(-c, A_ub, B_ub, A_eq, B_eq, bounds = (x1, x2, x3))
print(res)


# 测试代码 main 函数
def main():
    pass

if __name__ == "__main__":
    main()
```

### 示例 2

```python
# -*- coding: utf-8 -*-
from scipy.optimize import minimize
import numpy as np


"""
目标函数:
    min[(2+x1)/(1+x2) -3 * x1 + 4 * x3]
约束条件:  
    x1, x2, x3 的范围都在 [0.1, 0.9] 范围内
"""


def fun(args):
    """
    待优化函数: [(2+x1)/(1+x2) -3 * x1 + 4 * x3]
    """
    a, b, c, d = args
    v = lambda x: (a + x[0]) / (b + x[1]) - c * x[0] + d * x[2]
    return v


def con(args):
    """
    约束条件: 
        x1 - x1_min >= 0
        x1_max - x1 >= 0
        x2 - x2_min >= 0
        x2_max - x2 >= 0
        x3 - x3_min >= 0
        x3_max - x3 >= 0
    """
    x1_min, x1_max, x2_min, x2_max, x3_min, x3_max = args
    cons = (
        {"type": "ineq", "fun": lambda x: x[0] - x1_min},
        {"type": "ineq", "fun": lambda x: -x[0] + x1_max},
        {"type": "ineq", "fun": lambda x: x[1] - x2_min},
        {"type": "ineq", "fun": lambda x: -x[1] + x2_max},
        {"type": "ineq", "fun": lambda x: x[2] - x3_min},
        {"type": "ineq", "fun": lambda x: -x[2] + x3_max}
    )
    return cons


def optimizer():
    """
    目标函数优化器
    """
    # 目标函数系数
    args_fun = (2, 1, 3, 4)
    # 约束条件参数范围
    args_con = (
        0.1, 0.9,
        0.1, 0.9,
        0.1, 0.9
    )
    # 构造约束条件
    cons = con(args_con)
    # 设置优化变量初始猜测值
    x0 = np.asarray((0.5, 0.5, 0.5))
    # 目标函数优化
    res = minimize(
        fun(args_fun), 
        x0, 
        method = "SLSQP", 
        constraints = cons
    )
    return res


# 测试代码 main 函数
def main():
    result = optimizer()
    print("优化得到的目标函数最小值: ", result.fun)
    print("优化状态: ", result.success)
    print("优化路径: ", result.x)

if __name__ == "__main__":
    main()
```

## Pyomo

```python
# -*- coding: utf-8 -*-

# ***************************************************
# * File        : lp_pyomo.py
# * Author      : Zhefeng Wang
# * Email       : wangzhefengr@163.com
# * Date        : 2023-04-12
# * Version     : 0.1.041223
# * Description : description
# * Link        : https://zhuanlan.zhihu.com/p/125179633
# * Requirement : 相关模块版本需求(例如: numpy >= 2.1.0)
# ***************************************************

# python libraries
from pyomo.environ import *


# ------------------------------
# Problem
# ------------------------------
# objective: profit = 40x + 30y
# constraint: x <= 40
#             x + y <= 80
#             2x + y <= 100
# ------------------------------


# model
model = ConcreteModel()

# 声明决策变量
model.x = Var(domain = NonNegativeReals)
model.y = Var(domain = NonNegativeReals)

# 声明目标函数
model.profit = Objective(expr = 40 * model.x + 30 * model.y, sense = maximize)

# 声明约束条件
model.demand = Constraint(expr = model.x <= 40)
model.laborA = Constraint(expr = model.x + model.y <= 80)
model.laborB = Constraint(expr = 2 * model.x + model.y <= 100)
model.pprint()

# 模型求解
SolverFactory("glpk", executable = "/usr/local/bin/glpsol").solve(model).write()

# 模型解
print(f"\nProfit = {model.profit()}")

print("\nDecision Variables:")
print(f"x = {model.x()}")
print(f"y = {model.y()}")

print("\nConstraints:")
print(f"Demand = {model.demand()}")
print(f"Labor A = {model.laborA()}")
print(f"Labor B = {model.laborB()}")




# 测试代码 main 函数
def main():
    pass

if __name__ == "__main__":
    main()
```

## docplex

```python
# -*- coding: utf-8 -*-

# ***************************************************
# * File        : lp1.py
# * Author      : Zhefeng Wang
# * Email       : wangzhefengr@163.com
# * Date        : 2023-04-12
# * Version     : 0.1.041216
# * Description : description
# * Link        : https://zhuanlan.zhihu.com/p/124422566
# * Requirement : 相关模块版本需求(例如: numpy >= 2.1.0)
# ***************************************************

# python libraries
import os
import json
import random

import pandas as pd
import docplex.mp.model as cpx

# global variable
LOGGING_LABEL = __file__.split('/')[-1][:-3]


# ------------------------------
# Problem:
# ------------------------------
# Objective: `$$min \sum_{i=1}^{n}\sum_{j=1}^{m} c_{ij}x_{ij}$$`
# Constraint: `$$\sum_{i=1}^{n}a_{ij}x_{ij} \leq b_{j}, \forall j$$`
#             `$$x_{ij} \geq l_{ij}, \forall i, j$$`
#             `$$x_{ij} \leq u_{ij}, \forall i,j$$`
# ------------------------------
# ------------------------------
# 决策变量：n * m = 10 * 5
# 输出参数：n, m, c, a, b, l, u
# ------------------------------
n = 10
m = 5
set_I = range(1, n + 1)
set_J = range(1, m + 1)
c = {(i, j): random.normalvariate(0, 1) for i in set_I for j in set_J}
a = {(i, j): random.normalvariate(0, 5) for i in set_I for j in set_J}
b = {j: random.randint(0, 30) for j in set_J}
l = {(i, j): random.randint(0, 10) for i in set_I for j in set_J}
u = {(i, j): random.randint(10, 20) for i in set_I for j in set_J}
print(c)
print("-" * 20)
print(a)
print("-" * 20)
print(json.dumps(b, indent = 4))
print("-" * 20)
print(l)
print("-" * 20)
print(u)

# ------------------------------
# 模型
# ------------------------------
opt_model = cpx.Model(name = "MIP Model")

# ------------------------------
# 决策变量
# ------------------------------
# 决策变量：continuous
x_vars = {
    (i, j): opt_model.continuous_var(
        lb = l[i, j], 
        ub = u[i, j], 
        name = f"x_{i}_{j}"
    )
    for i in set_I for j in set_J
}
print(f"\nVariables:\n {x_vars}")
# 决策变量：binary
# x_vars = {
#     (i, j): opt_model.binary_var(name = f"x_{i}_{j}")
#     for i in set_I for j in set_J
# }
# print(f"\nVariables: {x_vars}")
# 决策变量：integer
# x_vars = {
#     (i, j): opt_model.integer_var(lb = l[i, j], ub = u[i, j], name = f"x_{i}_{j}")
#     for i in set_I for j in set_J
# }
# print(f"\nVariables: {x_vars}")

# ------------------------------
# 约束条件
# ------------------------------
# 小于等于(<=)约束
constraints = {
    j: opt_model.add_constraint(
        ct = opt_model.sum(a[i, j] * x_vars[i, j] for i in set_I) <= b[j], 
        ctname = f"constraint_{j}",
    )
    for j in set_J
}
print(f"\n Constraints:\n {constraints}")
# 大于等于(>=)约束
# constraints = {
#     j: opt_model.add_constraint(
#         ct = opt_model.sum(a[i, j] * x_vars[i, j] for i in set_I) >= b[j],
#         ctname = f"constraint_{j}",
#     )
#     for j in set_J
# }
# print(f"\n Constraints: {constraints}")
# 等于(==)约束
# constraints = {
#     j: opt_model.add_constraint(
#         ct = opt_model.sum(a[i, j] * x_vars[i, j] for i in set_I) == b[j],
#         ctname = f"constraint_{j}",
#     )
#     for j in set_J
# }
# print(f"\n Constraints: {constraints}")

# ------------------------------
# 目标函数
# ------------------------------
# objective
objective = opt_model.sum(x_vars[i, j] * c[i, j] for i in set_I for j in set_J)

# maximization
# opt_model.maximize(objective)

# minimization
opt_model.minimize(objective)

# ------------------------------
# 模型求解
# ------------------------------
# local cplex
opt_model.solve()

# cloud cplex
# opt_model.solve(url = "your_cplex_cloud_url", key = "your_api_key")

# ------------------------------
# 模型求解结果
# ------------------------------
# 决策变量解解析
opt_df = pd.DataFrame.from_dict(x_vars, orient = "index", columns = ["variable_object"])
opt_df.index = pd.MultiIndex.from_tuples(opt_df.index, name = ["column_i", "column_j"])
opt_df.reset_index(inplace = True)
opt_df["solution_value"] = opt_df["variable_object"].apply(lambda item: item.solution_value)
opt_df.drop(columns = ["variable_object"], inplace = True)

# 结果保存
solution_path = "./models/optimization_solution.csv"
if not os.path.exists(solution_path):
    opt_df.to_csv(solution_path)


# 测试代码 main 函数
def main():
    pass

if __name__ == "__main__":
    main()
```


# 参考

* [十分钟快速掌握单纯形法](https://mp.weixin.qq.com/s?__biz=MzU0NzgyMjgwNg==&mid=2247484683&idx=1&sn=32fbd323572549ebe1d7ceca7e5c79dd&chksm=fb49c8b2cc3e41a4005d70d926c48e4c538ebd573d5ffbdeeba6b10dadc4d03012cc311249c8&scene=21#wechat_redirect)
