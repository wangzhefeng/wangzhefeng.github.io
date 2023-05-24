---
title: 特征构建
author: 王哲峰
date: '2022-09-13'
slug: feature-engine-build
categories:
  - feature engine
tags:
  - machinelearning
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

- [交互特征](#交互特征)
- [特征组合](#特征组合)
- [多项式特征](#多项式特征)
- [自动特征工程](#自动特征工程)
  - [featuretools 库](#featuretools-库)
    - [安装和加载](#安装和加载)
    - [简单使用](#简单使用)
- [参考](#参考)
</p></details><p></p>

# 交互特征

两个特征的乘积可以组成一个简单的交互特征，这样可以捕获特征之间的交互作用。
交互特征的构造非常简单，但是使用起来代价很高

```python
df = pd.DataFrame(
    {},
    columns = ["x1", "x2", "x3", "x4", "x5"]
)
df["x1_x2"] = df["x1"] * df["x2"]
```

# 特征组合

为了提高复杂关系的拟合能力，在特征工程中经常把一阶离散特征两两组合，构成高阶组合特征。
并不是所有的特征组合都有意义，可以使用基于决策树的特征组合方法寻找组合特征，
决策树中每一条从根节点到叶节点的路径都可以看成是一种特征组合的方式


# 多项式特征


- 多项式特征可以生成特征的更高阶特征和交互特征

`$$(X_{1}, X_{2}) \rightarrow (1, X_1, X_2, X_{1}^{2}, X_{2}^{2}, X_{1} X_{2})$$`

```python
import numpy as np
from sklearn.preprocessing import PolynomialFeatures

# data
X = np.arange(6).reshape(3, 2)

# 多项式特征构造
poly = PolynomialFeatures(degree = 2, interaction_only = True)
new_X = poly.fit_transform(X)
new_X
```

# 自动特征工程

## featuretools 库

### 安装和加载

安装：

```bash
$ pip install featuretools
```

加载：

```python
import featuretools as ft
```

### 简单使用

```python
data = ft.demo.load_mock_customer()
customer_df = df["customers"]
sessions_df = data["sessions"]
transactions_df = data["transactions"]

entities = {
    "customers": (customers_df, "customer_id"),
    "sessions": (sessions_df, "session_id", "session_start")
}
```

# 参考
