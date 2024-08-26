---
title: KKT 条件
author: 王哲峰
date: '2022-10-20'
slug: kkt-condition
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

- [KKT 条件](#kkt-条件)
- [等式约束优化问题](#等式约束优化问题)
  - [等式约束优化问题](#等式约束优化问题-1)
  - [Lagrange 函数](#lagrange-函数)
  - [求解 Lagrange 函数的最优解](#求解-lagrange-函数的最优解)
- [不等式约束优化问题](#不等式约束优化问题)
  - [不等式约束优化问题的主要思想](#不等式约束优化问题的主要思想)
  - [一元函数的例子](#一元函数的例子)
    - [目标函数及约束条件](#目标函数及约束条件)
    - [引入松弛变量](#引入松弛变量)
    - [Lagrange 函数](#lagrange-函数-1)
    - [求解 Lagrange 函数的最优解](#求解-lagrange-函数的最优解-1)
  - [多元多次不等式约束问题示例](#多元多次不等式约束问题示例)
- [同时包含等式和不等式约束的一般优化问题](#同时包含等式和不等式约束的一般优化问题)
- [参考](#参考)
</p></details><p></p>

KKT(Karush-Kuhn-Tucker) 条件是在满足一些有规则的条件下，
一个非线性规划(Nonlinear Programming)问题能有最优化解法的一个必要条件。
这是一个使用广义拉格朗日函数的结果

KKT 条件将 Lagrange 乘数法(Lagrange Multipliers) 所处理涉及的约束优化问题推广至不等式。
在实际应用上，KKT 条件(方程组)一般不存在代数解，许多优化算法可供数值计算选用

# KKT 条件

对于具有等式和不等式约束的一般优化问题:

`$$min f(\textbf{x})$$`

`$$s.t. \begin{cases}
g_{j}(\textbf{x}) \leq 0, j = 1, 2, \ldots, m \\
h_{k}(\textbf{x}) = 0, k = 1, 2, \ldots, l
\end{cases}$$`

KKT 条件给出了判断 `$x^{*}$` 是否为最优解的必要条件，即:

`$$\begin{cases}
\frac{\partial f}{\partial x_{i}} + \sum_{j=1}^{m}\mu_{j}\frac{\partial g_{j}}{\partial x_{i}} + \sum_{k=1}^{l}\lambda_{k} \frac{\partial h_{k}}{\partial x_{i}} = 0, i = 1, 2, \ldots, n\\
h_{k}(\textbf{x}) = 0, k = 1, 2, \ldots, l  \\
\mu_{j}g_{j}(\textbf{x}), j = 1, 2, \ldots, m  \\
\mu_{j} \geq 0
\end{cases} $$`

# 等式约束优化问题

## 等式约束优化问题

`$$min f(x_{1}, x_{2}, \ldots, x_{n})$$`

`$$s.t. h_{k}(x_{1}, x_{2}, \ldots, x_{n}) = 0$$`

## Lagrange 函数

根据 Lagrange 乘数法，令

`$$L(\textbf{x}, \lambda) = f(\textbf{x}) + \sum_{k=1}^{l} \lambda_{k} h_{k}(\textbf{x})$$`

其中:

* 函数 `$L(\textbf{x}, y)$` 称为 Lagrange 函数
* 参数 `$\lambda$` 称为 Lagrange 乘子

## 求解 Lagrange 函数的最优解

对函数 `$L(\textbf{x}, y)$` 关于 `$\textbf{x}$` 和 `$\lambda$` 求偏导数：

`$$\begin{cases}
\frac{\partial L}{\partial x_{i}} = 0, i = 1, 2, \ldots, n \\
\frac{\partial L}{\partial \lambda_{k}} = 0, k = 1, 2, \ldots, l 
\end{cases} $$`

解方程组得到的解可能为极值点，具体是否为极值点需要根据问题本身的具体情况检验。
这个方程组称为 **等式约束的极值必要条件**

上式对 `$n$` 个 `$x_{i}$` 和 `$l$` 个 `$\lambda_{k}$` 分别求偏导，
回想在无约束优化问题 `$f(x_{1}, x_{2}, \ldots, x_{n})$` 中，根据极值的必要条件，
分别令 `$\frac{\partial f}{\partial x_{i}} = 0$`，求出可能的极值。
因此可以联想到：

> 等式约束下的 Lagrange 乘数法引入了 `$l$` 个 Lagrange 乘子，
> 可以把 `$\lambda_{k}$` 也看作优化变量(`$x_{i}$` 就叫做优化变量)。
> 相当于将优化变量个数增加到了 `$(n + l)$` 个，`$x_{i}$` 与 `$\lambda_{k}$` 一视同仁，
> 均为优化变量，均对它们求偏导

# 不等式约束优化问题

## 不等式约束优化问题的主要思想

不等式约束优化问题的主要思想是：

> 转化的思想 —— 将不等式约束条件变成等式约束条件。
> 具体做法是：引入松弛变量。松弛变量也是优化变量，也需要一视同仁求偏导

![img](images/neq.jpg)

## 一元函数的例子

### 目标函数及约束条件

`$$min f(x)$$`

`$$s.t. \begin{cases}
g_{1}(x) = a - x \leq 0 \\
g_{2}(x) = x - b \leq 0 \\
\end{cases}$$`

(优化问题中，我们必须求得一个确定的值，因此不妨令所有的不等式均取到等号，即 `$\leq$` 的情况)

### 引入松弛变量

对于约束 `$g_{1}(x)$` 和 `$g_{2}(x)$`，分别引入两个松弛变量 `$a_{1}^{2}$` 和 `$b_{1}^{2}$`，
得到 

`$$\begin{cases}
h_{1}(x, a_{1}) = g_{1}(x) + a_{1}^{2} = a - x + a_{1}^{2} = 0 \\
h_{2}(x, b_{1}) = g_{2}(x) + b_{1}^{2} = x - b + b_{1}^{2} = 0
\end{cases}$$`

注意，这里直接加上平方项 `$a_{1}^{2}$`、`$b_{1}^{2}$` 而非 `$a_{1}$`、`$b_{1}$`，
是因为 `$g_{1}(x)$`、`$g_{2}(x)$` 这两个不等式的左边必须加上一个正数才能使不等式变为等式。
若只加上 `$a_{1}$`、`$b_{1}$`，又会引入新的约束 `$a_{1} \geq 0$` 和 `$b_{1} \geq 0$`，
这不符合原先的意愿

### Lagrange 函数

由此，将不等式约束转化为了等式约束，并得到 Lagrange 函数：

`$$L(x, a_{1}, b_{1}, \mu_{1}, \mu_{2}) \\
= f(x) + \mu_{1} (g_{1}(x) + a_{1}^{2}) + \mu_{2} (g_{2}(x) + b_{1}^{2}) \\
= f(x) + \mu_{1} (a - x + a_{1}^{2}) + \mu_{2} (x - b + b_{1}^{2})$$`

### 求解 Lagrange 函数的最优解

按照等式约束优化问题(极值必要条件)对其求解，联立方程：

`$$\begin{cases}
\frac{\partial L}{\partial x} = \frac{\partial f}{\partial x} + \mu_{1} \frac{d g_{1}(x)}{d x} + \mu_{2} \frac{d g_{2}(x)}{d x} = \frac{\partial f}{\partial x} - \mu_{1} + \mu_{2} = 0 \\
\frac{\partial L}{\partial \mu_{1}} = g_{1}(x) + a_{1}^{2} = 0 \\
\frac{\partial L}{\partial \mu_{2}} = g_{2}(x) + b_{1}^{2} = 0 \\
\frac{\partial L}{\partial a_{1}} = 2 \mu_{1} a_{1} = 0 \\
\frac{\partial L}{\partial b_{1}} = 2 \mu_{2} b_{1} = 0 \\
\mu_{1} \geq 0, \mu_{2} \geq 0
\end{cases}$$`

解方程组：

* 对 `$\mu_{1} a_{1} = 0$` 有两种情况，综合两种情况得到 `$\mu_{1}g_{1}(x)=0$`，
  且在约束起作用时 `$\mu_{1} > 0$`，`$g_{1}(x)$`，约束不起作用时 `$\mu_{1}=0$`，`$g_{1}(x) \neq 0$`：
    - 情形 1：`$\mu_{1} = 0$`，`$a_{1} \neq 0$`。
      由于 `$\mu_{1} = 0$`，因此，`$g_{1}(x)$` 与其相乘为 0，可以理解为约束 `$g_{1}(x)$` 不起作用，
      且有：`$g_{1}(x) = a - x < 0$`
    - 情形 2：`$\mu_{1} \geq 0$`，`$a_{1} = 0$`。
      此时 `$g_{1}(x) = a - x = 0$` 且 `$\mu_{1} > 0$`，
      可以理解为：约束 `$g_{1}(x)$` 起作用，且有 `$g_{1}(x) = 0$`
* 同样，对 `$\mu_{2} b_{1} = 0$`，得到 `$\mu_{2}g_{2}(x)=0$`，
  且在约束起作用时 `$\mu_{2} > 0$`，`$g_{2}(x)$`，约束不起作用时 `$\mu_{2}=0$`，`$g_{2}(x) \neq 0$`

因此，方程组(极值必要条件)转换为：

`$$\begin{cases}
\frac{\partial f}{\partial x} + \mu_{1} \frac{d g_{1}(x)}{d x} + \mu_{2} \frac{d g_{2}(x)}{d x} = \frac{\partial f}{\partial x} - \mu_{1} + \mu_{2} = 0 \\
\mu_{1} g_{1}(x) = 0 \\
\mu_{2} g_{2}(x) = 0 \\
\mu_{1} \geq 0, \mu_{2} \geq 0
\end{cases}$$`

## 多元多次不等式约束问题示例

`$$min f(x)$$`

`$$s.t.
g_{j}(x) \leq 0, j = 1, 2, \ldots, m$$`

通过 Lagrange 乘数法求最优解有：

`$$\begin{cases}
\frac{\partial f(x^{*})}{\partial x_{i}} + \sum_{j = 1}^{m} \mu_{j} \frac{\partial g_{j}(x^{*})}{\partial x_{i}} = 0, i = 1, 2, \ldots, n\\
\mu_{j} g_{j}(x^{*}) = 0, j = 1, 2, \ldots, m \\
\mu_{j} \geq 0, j = 1, 2, \ldots, m
\end{cases}$$`

上式便称为不等式约束优化问题的 KKT 条件. `$\mu_{j}, j = 1, 2, \ldots, m$` 称为 KKT 乘子，
且约束起作用时，`$\mu_{j} \geq 0$`，`$g_{j}(x) = 0$`；约束不起作用时，`$\mu_{j} = 0$`，`$g_{j} < 0$`

# 同时包含等式和不等式约束的一般优化问题

`$$min f(\textbf{x})$$`

`$$s.t. \begin{cases}
g_{j}(\textbf{x}) \leq 0, j = 1, 2, \ldots, m \\
h_{k}(\textbf{x}) = 0, k = 1, 2, \ldots, l
\end{cases}$$`


KKT 条件（`$x^{*}$` 是最优解的必要条件) 为:

`$$\begin{cases}
\frac{\partial f}{\partial x_{i}} + \sum_{j=1}^{m}\mu_{j}\frac{\partial g_{j}}{\partial x_{i}} + \sum_{k=1}^{l}\lambda_{k} \frac{\partial h_{k}}{\partial x_{i}} = 0, i = 1, 2, \ldots, n\\
h_{k}(\textbf{x}) = 0, k = 1, 2, \ldots, l  \\
\mu_{j}g_{j}(\textbf{x}), j = 1, 2, \ldots, m  \\
\mu_{j} \geq 0
\end{cases} $$`

对于等式约束的 Lagrange 乘子，并没有非负的要求！以后求其极值点，不必再引入松弛变量，直接使用 KKT 条件判断

# 参考

* https://zhuanlan.zhihu.com/p/26514613
* https://zhuanlan.zhihu.com/p/38163970

