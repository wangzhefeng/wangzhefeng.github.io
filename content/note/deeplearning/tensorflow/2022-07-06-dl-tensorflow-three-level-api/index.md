---
title: TensorFlow 三阶 API
author: 王哲峰
date: '2022-09-20'
slug: dl-tensorflow-three-level-api
categories:
  - tensorflow
tags:
  - tool
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
</style>

<details><summary>目录</summary><p>

- [低阶 API](#低阶-api)
- [中阶 API](#中阶-api)
- [高阶 API](#高阶-api)
- [线性回归模型](#线性回归模型)
  - [载入 Python 依赖](#载入-python-依赖)
  - [数据准备](#数据准备)
  - [模型构建](#模型构建)
  - [模型训练](#模型训练)
  - [模型结果可视化](#模型结果可视化)
</p></details><p></p>

# 低阶 API

低阶 API 主要包括:

* 张量操作
* 计算图
* 自动微分

# 中阶 API

TensorFlow 中阶 API 主要包括:

* 各种模型层
* 损失函数
* 优化器
* 数据管道
* 特征列


# 高阶 API

TensorFlow 的高阶 API 主要为 `tf.keras.models` 提供的模型的类接口

TensorFlow 高阶 API 主要包括:

* 使用 `Sequential` 按层顺序构建模型
* 使用函数式 API 构建任意结构模型
* 继承 `Model` 基类构建自定义模型

# 线性回归模型

## 载入 Python 依赖

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import tensorflow as tf
from tensorflow.keras import models, layers, losses, metrics, optimizers
```

## 数据准备

* 生成数据

```python
# 样本数量
num_samples = 400

# 生成测试用数据集
X = tf.random.uniform([n, 2], minval = -10, maxval = 10)
w0 = tf.constant([[2.0], [-3.0]])
b0 = tf.constant([[3.0]])
Y = X@w0 + b0 + tf.random.normal([n, 1], mean = 0.0, stddev = 2.0)
```

* 数据可视化

```python
%matplotlib inline
%config InlineBackend.figure_format = "svg"

plt.figure(figsize = (12, 5))

ax1 = plt.subplot(121)
ax1.scatter(X[:, 0], Y[:, 0], c = "b")
plt.xlabel("x1")
plt.ylabel("y", rotation = 0)

ax2 = plt.subplot(122)
ax2.scatter(X[:, 1], Y[:, 0], c = "g")
plt.xlabel("x2")
plt.ylabel("y", rotation = 0)

plt.show()
```

## 模型构建

```python
tf.keras.backend.clear_session()

model = models.Sequential()
model.add(layers.Dense(1, input_shape = (2,)))

model.summary()
```

## 模型训练

```python
model.compile(
    optimizer = "adam",
    loss = "mse",
    metrics = ["mae"],
)
model.fit(X, Y, batch_size = 10, epochs = 200)

tf.print(f"w = {model.layers[0].kernel}")
tf.print(f"b = {model.layers[0].bias}")
```

## 模型结果可视化

```python
%matplotlib inline
%config InlineBackend.figure_format = "svg"

w, b = model.variables

plt.figure(figsize = (12, 5))

ax1 = plt.subplot(121)
ax1.scatter(X[:, 0], Y[:, 0], c = "b", label = "samples")
ax1.plot(X[:, 0], w[0] * X[:, 0] + b[0], "-r", linewidth = 5.0, label = "model")
ax1.legend()
plt.xlabel("x1")
plt.ylabel("y", rotation = 0)

ax2 = plt.subplot(122)
ax2.scatter(X[:, 1], Y[:, 0], c = "g", label = "samples")
ax2.plot(X[:, 1], w[1] * X[:, 1] + b[0], "-r", linewidth = 5.0, label = "model")
ax2.legend()
plt.xlabel("x2")
plt.ylabel("y", rotation = 0)

plt.show()
```
