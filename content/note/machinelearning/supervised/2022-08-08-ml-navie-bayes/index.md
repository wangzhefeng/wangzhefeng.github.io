---
title: Navie Bayes
author: wangzf
date: '2022-08-08'
slug: ml-navie-bayes
categories:
  - machinelearning
tags:
  - machinelearning
  - model
---

<style>
h1 {
    background-color: #2B90B6;
    background-image: linear-gradient(45deg, #4EC5D4 10%, #146b8c 20%);
    background-size: 100%;
    -webkit-background-clip: text;
    -moz-background-clip: text;
    -webkit-text-fill-color: transparent;
    -moz-text-fill-color: transparent;
}
h2 {
    background-color: #2B90B6;
    background-image: linear-gradient(45deg, #4EC5D4 10%, #146b8c 20%);
    background-size: 100%;
    -webkit-background-clip: text;
    -moz-background-clip: text;
    -webkit-text-fill-color: transparent;
    -moz-text-fill-color: transparent;
}
h3 {
    background-color: #2B90B6;
    background-image: linear-gradient(45deg, #4EC5D4 10%, #146b8c 20%);
    background-size: 100%;
    -webkit-background-clip: text;
    -moz-background-clip: text;
    -webkit-text-fill-color: transparent;
    -moz-text-fill-color: transparent;
}
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

- [模型推导](#模型推导)
- [模型学习方法](#模型学习方法)
  - [极大似然估计](#极大似然估计)
  - [贝叶斯估计](#贝叶斯估计)
</p></details><p></p>


# 模型推导

给定数据集: `$\{(x_i, y_i)\}$`, 其中: `$i = 1, 2, \ldots, N$`, `$x_i \in R^p$`, `$y_i \in \{c_1, c_2, \ldots, c_K\}$`

假设

- 训练数据集 `$\{(x_i, y_i)\}, i = 1, 2, \ldots, N$` 由 `$P(x, y)$` 独立同分布产生
- `$P(x, y)$`:  是 `$x$` 和 `$y$` 的联合概率分布
- `$P(y = c_k), i = 1, 2, \ldots, K$`: 是目标变量 `$y$` 的先验分布
- `$P(x|y=c_k)$`: 是给定目标变量 `$y=c_k$` 下, 预测变量 `$x$` 条件分布

根据条件概率的条件独立性假设: 

`$$\begin{eqnarray}
   P(x|y=c_k) & & {} = P(x_{ij}|y_i=c_k) \nonumber \\
   		   & & {} = \prod_{j=1}^{p}P(x_{ij}|y_i=c_k) \nonumber
   \end{eqnarray}$$`

根据Bayesian定理, 求解给预测变量 `$x$` 下, 目标变量 `$y=c_k$` 的后验概率: 

`$$\begin{eqnarray}
   P(y=c_k|x) & & {} = \frac{P(x, y = c_k)}{P(x)} \nonumber \\
   		   & & {} = \frac{P(x|y=c_k)P(y=c_k)}{\sum_{k}P(x|y=c_k)P(Y=c_k)} \nonumber \\
   		   & & {} = \frac{P(y_i=c_k)\prod_{j}P(x_{ij}|y_i=c_k)}{\sum_k P(y_i=c_k)\prod_j P(x_{ij}|y_i=c_k)} \nonumber
   \end{eqnarray}$$`

朴素贝叶斯分类器: 

`$$\begin{eqnarray}
   y_i=f(x_i) & & {} = \arg\underset{c_k}{\max} P(y=c_k|x_i) \nonumber \\
   	       & & {} = \arg\underset{c_k}{\max} P(y_i=c_k)\prod_{j}P(x_{ij}|y_i=c_k) \nonumber
   \end{eqnarray}$$`

# 模型学习方法

## 极大似然估计

**朴素贝叶斯分类器:**

`$$\begin{eqnarray}
   y_i=f(x_i) & & {} = \arg\underset{c_k}{\max} P(y_i=c_k|x_i) \nonumber \\
   	       & & {} = \arg\underset{c_k}{\max} P(y_i=c_k)\prod_{j}P(x_{ij}|y_i=c_k) \nonumber
   \end{eqnarray}$$`

**估计 `$P(y_i=c_k)$`:**

`$$P(y_i=c_k)=\frac{\sum_{i=1}^{N}I(y_i=c_k)}{N}$$`

**估计 `$P(x_{ij}|y_i=c_k)$`:**

假设第`$j$`个特征`$x_{ij}$`的取值集合为 `$\{a_{j1}, a_{j2}, \ldots, a_{jS_j}\}$`

`$$P(x_{ij} = a_{jl}|y_i=c_k)=\frac{\sum_{i=1}^{N}I(x_{ij} = a_{jl},y_i=c_k)}{\sum_{i=1}^{N}I(y_i=c_k)}, j= 1, 2, \ldots, p; l = 1, 2, \ldots, S_j$$`

**算法:**

给定训练数据集 `$T = \{(x_i, y_i), i= 1, 2, \ldots, N\}$`

其中: 

- `$x_i = (x_{i1}, x_{i2}, \ldots, x_{ip})$`
- `$x_{ij}$`是第 `$i$` 个样本的第 `$j$` 个 特征
- `$x_{ij}\in \{a_{j1}, a_{j2}, \ldots, a_{jS_j}\}$`, `$a_{jl}, l=1, 2, \ldots, S_j$` 是第 `$j$` 个特征可能取的第 `$l$` 个值
- `$y\in \{c_1, c_2, \ldots, c_k\}$`

**Note:**

1. 计算先验概率及条件概率

`$$P(y_i=c_k)=\frac{\sum_{i=1}^{N}I(y_i=c_k)}{N}, i = 1, 2, \ldots, N, k = 1, 2, \ldots, K$$`
`$$P(x_{ij} = a_{jl}|y_i=c_k)=\frac{\sum_{i=1}^{N}I(x_{ij} = a_{jl},y_i=c_k)}{\sum_{i=1}^{N}I(y_i=c_k)}, j= 1, 2, \ldots, p; l = 1, 2, \ldots, S_j$$`

2. 对于给定的样本 `$(x_{i1}, x_{i2}, \ldots, x_{ip})$`, 计算 

`$$P(y_i=c_k)\prod_{j=1}^{p}P(x_{ij}=a_{jl}|y_i=c_k), k = 1, 2, \ldots, K$$`

3. 确定样本 `$x_i$` 的类 

`$$y_i = \arg\underset{c_k}{\max}P(y_i=c_k)\prod_{j=1}^{p}P(x_{ij}=a_{jl}|y_i=c_k)$$`



## 贝叶斯估计

极大似然估计可能会出现所要估计得概率值为0的情况, 这时会影响到后验概率的计算结果, 使分类产生偏差

估计先验概率: 

`$$P_\lambda(y_i=c_k)=\frac{\sum_{i=1}^{N}I(y_i=c_k)+\lambda}{N+K\lambda}, i = 1, 2, \ldots, N, k = 1, 2, \ldots, K$$`

估计条件概率: 

`$$P_{\lambda}(x_{ij} = a_{jl}|y_i=c_k)=\frac{\sum_{i=1}^{N}I(x_{ij} = a_{jl},y_i=c_k)+\lambda}{\sum_{i=1}^{N}I(y_i=c_k)+S_j\lambda}, j= 1, 2, \ldots, p; l = 1, 2, \ldots, S_j;\lambda\geq 0$$`

- 当 `$\lambda = 0$` 时, 极大似然估计
- 当 `$\lambda = 1$` 时, 拉普拉斯平滑(Laplace smoothing), 常取 `$\lambda = 1$`
