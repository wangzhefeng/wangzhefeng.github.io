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

`$$x_{i}=b_{i}-(a_{i,m+1}x_{m+1} + \cdots + a_{i,n}x_{n})=b_{i}-\sum_{j=m+1}^{n}a_{i,j}x_{j}$$`

将 `$x_{i}$` 代入目标函数，并消去目标函数中的基变量 `$X_{B}$`，则：

`$$\begin{align}
max Z
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

`$R_{j}$` 为非基变量检验数，即：

`$$\begin{align}R_{j}
&=c_{j}-\sum_{i=1}^{m}c_{i}a_{ij} \\
&=c_{j}-(c_{1},c_{2},\cdots,c_{m})(a_{1j},a_{2j},\cdots,a_{mj})^{T} \\
&= c_{j}-\mathbf{C}_{B}a_{j}, j=m+1,\cdots,n
\end{align}$$`

根据上面的公式推导，可以计算出目标函数的值、非基变量的检验数，
同时也说明了单纯形表变换的内在数学原理。

### 单纯形法过程

> 单纯形法就是遍历可行域各顶点后判断最优解。

要用单纯形法求解线性规划数学模型，还需要把模型转化成规范形，规范形的条件如下：

1. 数学模型已经是标准型。
2. 约束方程组系数矩阵中含有至少一个单位子矩阵，对应的变量称为 **基变量**，
   **基**的作用是得到 **初始基本可行解**，这个初始基本可行解通常是**原点**。
   在大部分的问题中，通常引入松弛变量得到单位子矩阵，即使约束条件是等式约束，
   也可以引入 `$x_{N} = 0$` 的松弛变量。

> 如何理解线性规划中的**基变量**和**非基变量**呢？
> 
> 线性规划的最优解只能在顶点处取到，所以单纯形法的思想就是从一个顶点出发，
> 连续访问不同的顶点，在每一个顶点处检查是否有相邻的其他顶点能取到更优的目标函数值。
> 线性规划里面的约束（等式或不等式）可以看作是超平面（Hyperplane）或半空间（Half space）。
> **可行域**可以看作是被这组约束，或者超平面和半空间定义（围起来）的区域。
> 那么某一个顶点其实就是某组超平面的交点，这一组超平面对应的约束就是在某一个顶点取到 `"="` 号的约束（也就是基）。
> 顶点对应的代数意义就是一组方程（取到等号的约束）的解。

3. 目标函数中不含基变量。在这里基变量 `$x_{3}$`、`$x_{4}$` 和 `$x_{5}$` 是在约束方程中引进的变量，
   所以目标函数中没有这些基变量。

单纯形法的具体计算过程如下：

1. 确定初始基本可行解。利用规范型的数学模型，整理出目标函数和约束方程的形式。

    `$$max Z = 70x_{1} + 30x_{2}$$`
    `$$s.t.\begin{cases}
    3x_{1} + 9x_{2} + x_{3} = 540 \\
    5x_{1} + 5x_{2} + x_{4} = 450 \\
    9x_{1} + 3x_{2} + x_{5} = 720 \\
    x_{1}, x_{2}, x_{3}, x_{4}, x_{5} \geq 0
    \end{cases}$$`

   令非基变量 `$x_{j}=0,j=1,2,3$`，这样可以直接得到基变量的取值，即 `$x_{3}=540$`，
   `$x_{4}=450$`，`$x_{5}=720$`，将非基变量 `$x_{j} = 0,j=1,2,3$` 代入目标函数得到 `$Z = 0$`，
   初始基本可行解是：

   `$$\mathbf{X}=(x_{1},x_{2},x_{3},x_{4},x_{5})=(0,0,540,450,720)^{T}$$` 
   `$$Z=0$$`

   此时顶点位置是原点 `$O$`。

2. 判断当前点 `$\mathbf{X}$` 是否为最优解。对于最大化问题，
   目标函数中非基变量的系数 `$a_{i} \leq 0$` 时为最优解。
   而这里非基变量的系数 `$a_{1}=70>0$`，`$a_{2}=30>0$`，
   意味着只要在可行域内随着非基变量 `$x_{1}$` 和 `$x_{2}$` 的增大，
   目标函数就会继续增大，所以此时的解不是最优解。
   
   当然也可以利用梯度的知识来思考这个问题。对于最大化问题，
   只需要沿着梯度方向搜索即可找到最大值。线性规划是一个典型的凸优化问题，
   当梯度为零时，得到最大值。即当非基变量系数 `$a_{j} \leq 0$` 时，可得到问题最优解。

3. 基变量出基与非基变量入基。变量的入基和出基在几何图上表现为顶点的变化，
   如从 `$a$` 顶点变换到 `$b$` 顶点。选择使目标函数 `$Z$` 变化最快的非基变量入基，
   即**选择目标函数系数 `$a_{i}$` 最大且为正数的非基变量入基**，所以选择 `$x_{1}$` 入基，
   此时仍然 `$x_{2}=0$`。从凸优化的角度来看，就是选择目标函数梯度最大的方向做下一步的计算。
   那该选择哪个基变量出基呢？可以利用计算约束方程中常数项（`$b_{j}$`）与 `$x_{1}$` 系数（`$a_{1j}$`）的比值 `$\theta$`，选择最小的 `$\theta$` 对应的约束方程的基变量出基，即 `$\theta=\frac{b}{a_{i}}$`。

4. 计算新的解 `$\mathbf{X}$`。
5. 判断当前解 `$\mathbf{X}$` 是否最优。
6. 基变量出基与非基变量入基。
7. 确定新的解 `$\mathbf{X}$`。
8. 判断当前解 `$\mathbf{X}$` 是否最优。

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

