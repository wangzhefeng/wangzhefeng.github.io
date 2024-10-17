---
title: TensorFlow 自动微分机制
author: wangzf
date: '2022-07-03'
slug: dl-tensorflow-autograd
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
img {
    pointer-events: none;
}
</style>

<details><summary>目录</summary><p>

- [利用梯度磁带求导数](#利用梯度磁带求导数)
  - [变量张量求导](#变量张量求导)
  - [常量张量求导](#常量张量求导)
  - [求二阶导数](#求二阶导数)
  - [在 AutoGraph 中使用梯度磁带求导](#在-autograph-中使用梯度磁带求导)
- [利用梯度磁带和优化器求最小值](#利用梯度磁带和优化器求最小值)
  - [使用 optimizer.apply\_gradients](#使用-optimizerapply_gradients)
  - [使用 optimizer.minimize](#使用-optimizerminimize)
  - [在 AutoGraph 中完成最小值求解](#在-autograph-中完成最小值求解)
</p></details><p></p>

神经网络通常依赖反向传播求梯度来更新网络参数，求梯度过程通常是一件非常复杂而容易出错的事情，
而深度学习框架可以帮助自动地完成求梯度运算

TensorFlow 一般使用梯度磁带 `tf.GradientTape` 来记录正向运算过程，然后反播磁带自动得到梯度值，
这种利用 `tf.GradientTape` 求微分的方法叫做 TensorFlow 的自动微分机制

# 利用梯度磁带求导数

求以下函数的导数:

`$$f(x) = a \times x^{2} + b \times x + c$$`

## 变量张量求导

```python
import tensorflow as tf
import numpy as np

x = tf.Variable(0.0, name = "x", dtype = tf.float32)
a = tf.constant(1.0)
b = tf.constant(-2.0)
c = tf.constant(1.0)

with tf.GradientTape() as tape:
    y = a * tf.pow(x, 2) + b * x + c

dy_dx = tape.gradient(y, x)
print(dy_dx)
```

```
tf.Tensor(-2.0, shape=(), dtype=float32)
```

## 常量张量求导

* 对常量张量也可以求导，需要增加 `watch`

```python
import tensorflow as tf
import numpy as np

x = tf.Variable(0.0, name = "x", dtype = tf.float32)
a = tf.constant(1.0)
b = tf.constant(-2.0)
c = tf.constant(1.0)

with tf.GradientTape() as tape:
    tape.watch([a, b, c])
    y = a * tf.pow(x, 2) + b * x + c

dy_dx, dy_da, dy_db, dy_dc = tape.gradient(y, [x, a, b, c])
print(dy_da)
print(dy_db)
print(dy_dc)
```

```
tf.Tensor(0.0, shape=(), dtype=float32)

tf.Tensor(1.0, shape=(), dtype=float32)
```

## 求二阶导数

```python
with tf.GradientTape() as tape2:
    with tf.GradientTape() as tape1:
        y = a * tf.pow(x, 2) + b * x + c
    dy_dx = tape1.gradient(y, x)
dy2_dx2 = tape2.gradient(dy_dx, x)

print(dy2_dx2)
```

```
tf.Tensor(2.0, shape=(), dtype=float32)
```

## 在 AutoGraph 中使用梯度磁带求导

```python
@tf.function
def f(x):
    # 自变量转换成 tf.float32
    x = tf.cast(x, tf.float32)
    a = tf.constant(1.0)
    b = tf.constant(-2.0)
    c = tf.constant(1.0)
    
    with tf.GradientTape() as tape:
        tap.watch(x)
        y = a * tf.pow(x, 2) + b * x + c
    dy_dx = tape.gradient(y, x)

    return ((dy_dx, y))

tf.print(f(tf.constant(0.0)))
tf.print(f(tf.constant(1.0)))
```

# 利用梯度磁带和优化器求最小值

求以下函的最小值:

`$$f(x) = a \times x^{2} + b \times x + c$$`

## 使用 optimizer.apply_gradients

```python
x = tf.Variable(0.0, name = "x", dtype = tf.int32)
a = tf.constant(1.0)
b = tf.constant(-2.0)
c = tf.constant(1.0)

optimizer = tf.keras.optimizers.SGD(learning_rate = 0.01)

for _ in range(1000):
    with tf.GradientTape() as  tape:
        y = a * tf.pow(x, 2) + b * x + c
    dy_dx = tape.gradient(y, x)
    optimizer.apply_gradients(grad_and_vars = [(dy_dx, x)])

tf.print(f"y = {y}; x = {x}")
```

## 使用 optimizer.minimize

`optimizer.minimize` 相当于先用 `tape` 求 `gradient`，再 `apply_gradients`

```python
x = tf.Variable(0.0, name = "x", dtype = tf.int32)

def f():
    a = tf.constant(1.0)
    b = tf.constant(-2.0)
    c = tf.constant(1.0)
    y = a * tf.pow(x, 2) + b * x + c
    return (y)

optimizer = tf.keras.optimizers.SGD(learning_rate = 0.01)

for _ in range(1000):
    optimizer.minimize(f, [x])

tf.print(f"y = {f()}; x = {x}")
```

## 在 AutoGraph 中完成最小值求解

* 使用 `optimizer.apply_gradients`

```python
x = tf.Variable(0.0, name = "x", dtype = tf.float32)
optimizer = tf.keras.optimizers.SGD(learning_rate = 0.01)

@tf.function
def minimizef():
    a = tf.constant(1.0)
    b = tf.constant(-2.0)
    c = tf.constant(1.0)
    for _ in tf.range(1000):
        with tf.GradientTape() as tape:
            y = a * tf.pow(x, 2) + b * x + c
        dy_dx = tape.gradient(y, x)
        optimizer.apply_gradients(grads_and_vars = [(dy_dx, x)])
    y = a * tf.pow(x, 2) + b * x + c
    return y

tf.print(minimizef())
tf.print(x)
```

* 使用 `optimizer.minimize`

```python
x = tf.Variable(0.0, name = "x", dtype = tf.float32)
optimizer = tf.keras.optimizers.SGD(learning_rate = 0.01)

@tf.function
def f():
    a = tf.constant(1.0)
    b = tf.constant(-2.0)
    c = tf.constant(1.0)
    y = a * tf.pow(x, 2) + b * x + c
    return (y)

@tf.function
def train(epoch):
    for _ in tf.range(epoch):
        optimizer.minimize(f, [x])
    return (f())

tf.print(train(1000))
tf.print(x)
```

