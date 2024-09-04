---
title: 粒子群算法
subtitle: Particle Swarm Optimization
author: 王哲峰
date: '2024-09-04'
slug: particle-swarm-optimization
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

- [粒子群算法原理](#粒子群算法原理)
- [粒子群算法求解无约束优化问题](#粒子群算法求解无约束优化问题)
- [粒子群算法求解约束优化问题](#粒子群算法求解约束优化问题)
- [粒子群算法求解旅行商问题](#粒子群算法求解旅行商问题)
</p></details><p></p>

> Particle Swarm Optimization, PSO

# 粒子群算法原理


# 粒子群算法求解无约束优化问题

用粒子群算法求解 Rastrigin 函数的极小值，Rastrigin 是一个典型的非线性多峰函数，
在搜索区域内存在许多极大值和极小值，导致寻找全局最小值比较困难，常用来测试寻优算法的性能。

Rastrigin 函数的表达式如下：

`$$Z = 2a + x^{2} - a cos 2 \pi x + y^{2} - a cos 2 \pi y$$`

这是一个典型非凸优化问题，通过 Python 绘制函数图形如下：

```python
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D

# 生成 X 和 Y 的数据
X = np.arange(-5, 5, 0.1)
Y = np.arange(-5, 5, 0.1)
X, Y = np.meshgrid(X, Y)

```

![img](images/)

# 粒子群算法求解约束优化问题

# 粒子群算法求解旅行商问题



