---
title: TensorFlow 张量、自动微分机制和计算图
author: 王哲峰
date: '2022-09-10'
slug: dl-tensorflow-tensor-autograd-graph
categories:
  - deeplearning
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

- [张量](#张量)
  - [张量数据结构](#张量数据结构)
    - [常量张量](#常量张量)
      - [数据类型](#数据类型)
      - [数据维度](#数据维度)
      - [数据转换](#数据转换)
    - [变量张量](#变量张量)
  - [张量数据操作](#张量数据操作)
    - [创建张量](#创建张量)
      - [张量创建](#张量创建)
      - [均匀分布](#均匀分布)
      - [正态分布](#正态分布)
      - [特殊矩阵](#特殊矩阵)
    - [索引切片](#索引切片)
    - [维度变换](#维度变换)
    - [合并分割](#合并分割)
      - [合并](#合并)
      - [分割](#分割)
  - [张量数学运算](#张量数学运算)
    - [标量运算](#标量运算)
    - [向量运算](#向量运算)
    - [矩阵运算](#矩阵运算)
    - [广播机制](#广播机制)
- [自动微分机制](#自动微分机制)
  - [利用梯度磁带求导数](#利用梯度磁带求导数)
  - [利用梯度磁带和优化器求最小值](#利用梯度磁带和优化器求最小值)
- [计算图](#计算图)
  - [静态计算图](#静态计算图)
  - [动态计算图](#动态计算图)
  - [AutoGraph](#autograph)
    - [AutoGraph 使用规范](#autograph-使用规范)
    - [AutoGraph 机制原理](#autograph-机制原理)
    - [AutoGraph 和 tf.Module](#autograph-和-tfmodule)
</p></details><p></p>

张量和计算图是 TensorFlow 的核心概念

# 张量

TensorFlow 的基本数据结构是张量。张量即多维数组，TensorFlow 的张量和 Numpy 中的 array 很类似。
从行为特性来开，有两种类型的张量:
    
* 常量 Constant
    - 常量的值在计算图中不可以被重新赋值
* 变量 Variable
    - 变量可以在计算图中用 `assign` 等算子重新赋值

## 张量数据结构

### 常量张量

#### 数据类型

张量的数据类型和 `numpy.array` 数据类型基本一一对应

```python
import numpy as np
import tensorflow as tf

i = tf.constant(1)  # tf.int32 类型常量
l = tf.constant(1, dtype = tf.int64)  # tf.int64 类型常量
f = tf.constant(1.23)  # tf.float32 类型常量
d = tf.constant(3.14, dtype = tf.double)  # tf.double 类型常量
s = tf.constant("hello world")  # tf.string 类型常量
b = tf.constant(True)  # tf.bool 类型常量

print(tf.int64 == np.int64)  # True
print(tf.double == np.double)  # True
print(tf.string == np.unicode)  # False
print(tf.bool == np.bool)  # True
```

#### 数据维度

不同类型的数据可以用不同维度(rank)的张量来表示:

* 标量为 0 维张量
* 向量为 1 维张量
* 矩阵为 2 维张量
* 彩色图像有 RGB 三个通道，可以表示为 3 维张量
* 视频还有时间维，可以表示为 3 维张量

可以简单地总结为: 有几层总括号，就是多少维的张量

```python

```

#### 数据转换

* 可以用 `tf.cast` 改变张量的数据类型

```python
h = tf.constant([123, 456], dtype = tf.int32)
f = tf.cast(h, tf.float32)
print(h.dtype)  # <dtype: 'int32'>
print(f.dtype)  # <dtype: 'float32'>
```

* 可以用 `.numpy()` 的方法将 TensorFlow 中的张量转换为 Numpy 中的张量
* 可以用 `shape` 方法查看张量尺寸

```python
y = tf.constant([[1.0, 2.0], [3.0, 4.0]])
print(y.numpy())
print(y.shape)
```

```
[[1. 2.]
 [3. 4.]]

(2, 2)
```

* 可以用 `.numpy().decode()` 对张量进行字符编码

```python
u = tf.constant(u"你好，世界")
print(u.numpy())
print(u.numpy().decode("utf-8"))
```

```
b'\xe4\xbd\xa0\xe5\xa5\xbd \xe4\xb8\x96\xe7\x95\x8c'
你好 世界
```

### 变量张量

模型中需要别训练的参数一般被设置成变量

* 常量值不可以改变，常量的重新赋值相当于创造新的内存空间

```python
c = tf.constant([1.0, 2.0])
print(c)
print(id(c))
```

```
tf.Tensor([1. 2.], shape=(2,), dtype=float32)
5276289568
```

```python
c = c + tf.constant([1.0, 1.0])
print(c)
print(id(c))
```

```
tf.Tensor([2. 3.], shape=(2,), dtype=float32)
5276290240
```

* 变量的值可以改变，可以通过 `assign`、`assign_add` 等方法给变量重新赋值

```python
v = tf.Variable([1.0, 2.0], name = "v")
print(v)
print(id(v))
```

```
<tf.Variable 'v:0' shape=(2,) dtype=float32, numpy=array([1., 2.], dtype=float32)>
5276259888
```

```python
v.assign_add([1.0, 1.0])
print(v)
print(id(v))
```

```
<tf.Variable 'v:0' shape=(2,) dtype=float32, numpy=array([2., 3.], dtype=float32)>
5276259888
```

## 张量数据操作

### 创建张量

张量创建的许多方法和 Numpy 中创建 array 的方法很像

#### 张量创建

* tf.constant
* tf.range
* tf.linspace
* tf.zeros
* tf.zeros_like
* tf.ones
* tf.ones_like
* tf.fill

```python
import tensorflow as tf
import numpy as np

a = tf.constant([1, 2, 3], dtype = tf.float32)
tf.print(a)

b = tf.range(1, 10, delta = 2)
tf.print(b)

c = tf.linspace(0.0, 2 * 3.14, 100)
tf.print(c)

d = tf.zeros([3, 3])
tf.print(d)

a = tf.ones([3, 3])
b = tf.zeros_like(a, dtype = tf.float32)
tf.print(a)
tf.print(b)

b = tf.fill([3, 2], 5)
tf.print(b)
```

#### 均匀分布

* tf.random.set_seed
* tf.random.uniform

```python
tf.random.set_seed(1.0)
a = tf.random.uniform([5], minval = 0, maxval = 10)
tf.print(a)
```

#### 正态分布

```python
tf.random.set_seed(1.0)
b = tf.random.normal([3, 3], mean = 0.0, stddev = 1.0)
tf.print(b)
```

* 正态分布，剔除 2 倍方差意外数据重新生成

```python
tf.random.set_seed(1.0)
c = tf.random.truncated_normal(
    (5, 5), 
    mean = 0.0, 
    stddev = 1.0, 
    dtype = tf.float32
)
tf.print(c)
```

#### 特殊矩阵

* 单位矩阵

```python
I = tf.eye(3, 3)
tf.print(I)
```

* 对角矩阵

```python
t = tf.linalg.diag([1, 2, 3])
tf.print(t)
```

### 索引切片

张量的索引切片方式和 Numpy 几乎是一样的，切片时支持省略参数和省略号

* 对于 `tf.Variable`，可以通过索引和切片对部分元素进行修改
* 对于提取张量的连续子区域，可使用 `tf.slice`
* 对于不规则的切片提取，可以使用 `tf.gather`、`tf.gather_nd`、`tf.boolean_mask`
* `tf.boolean_mask` 功能最为强大，它可以实现 `tf.gather`、`tf.gather_nd` 的功能，
  并且 `tf.boolean_mask` 还可以实现布尔索引
* 如果要通过修改张量的某些元素得到新的张量，可以使用 `tf.where`、`tf.scatter_nd`

```python
import tensorflow as tf

tf.random.set_seed(3)
t = tf.random.uniform([5, 5], minval = 0, maxval = 10, dtype = tf.int32)
a = tf.random.uniform([3, 3, 3], minval = 0, maxval = 10, dtype = tf.int32)
tf.print(t)
tf.print(a)

# 第 0 行
tf.print(t[0])

# 倒数第 1 行
tf.print(t[-1])

# 第 1 行 第 3 列
tf.print(t[1, 3])
tf.print(t[1][3])


# 第 1 行至第 3 行
tf.print(t[1:4, :])
tf.print(tf.slice(t, [1, 0], [3, 5]))  # tf.slice(input, begin_vector, size_vector)

# 第 1 行至最后一行，第 0 列到最后一列每隔两列取一列
tf.print(t[1:4, :4:2])

# 对变量来说，还可以使用索引和切片修改部分元素
x = tf.Variable([[1, 2], [3, 4]], dtype = tf.float32)
x[1, :].assign(tf.constant([0.0, 0.0]))
tf.print(x)

# 省略号可以表示多个冒号
tf.print(a[..., 1])
```

### 维度变换

维度变换相关的函数有:

* tf.reshape
    - 改变张量形状
* tf.squeeze
    - 减少维度
* tf.expand_dim
    - 增加维度
* tf.transpose
    - 交换维度








### 合并分割

* 可以用 `tf.concat` 和 `tf.stack` 方法对多个张量进行合并
* 可以用 `tf.split` 方法把一个张量分割成多个张量

#### 合并

`tf.concat` 和 `tf.stack` 有略微的区别，`tf.concat` 是连接，不会增加维度，
而 `tf.stack` 是堆叠，会增加维度

```python
a = tf.constant(
    [[1.0, 2.0], 
     [3.0, 4.0]]
)
tf.print(a.shape)  # shape=(2, 2)

b = tf.constant(
    [[5.0, 6.0], 
     [7.0, 8.0]]
)
tf.print(b.shape)  # shape=(2, 2)

c = tf.constant(
    [[9.0, 10.0], 
     [11.0, 12.0]]
)
tf.print(c.shape)  # shape=(2, 2)
```

* tf.concat

```python
tf.concat([a, b, c], axis = 0)
tf.concat([a, b, c], aixs = 1)
```

```
<tf.Tensor: shape=(6, 2), dtype=float32, numpy=
array([[ 1.,  2.],
       [ 3.,  4.],
       [ 5.,  6.],
       [ 7.,  8.],
       [ 9., 10.],
       [11., 12.]], dtype=float32)>

<tf.Tensor: shape=(2, 6), dtype=float32, numpy=
array([[ 1.,  2.,  5.,  6.,  9., 10.],
       [ 3.,  4.,  7.,  8., 11., 12.]], dtype=float32)>
```

* tf.stack

```python
tf.stack([a, b, c])
tf.stack([a, b, c], axis = 1)
```

```
<tf.Tensor: shape=(3, 2, 2), dtype=float32, numpy=
array([[[ 1.,  2.],
        [ 3.,  4.]],

       [[ 5.,  6.],
        [ 7.,  8.]],

       [[ 9., 10.],
        [11., 12.]]], dtype=float32)>

<tf.Tensor: shape=(2, 3, 2), dtype=float32, numpy=
array([[[ 1.,  2.],
        [ 5.,  6.],
        [ 9., 10.]],

       [[ 3.,  4.],
        [ 7.,  8.],
        [11., 12.]]], dtype=float32)>
```

#### 分割

`tf.split` 是 `tf.concat` 的逆运算，
可以指定分割份数平均分割，也可以通过指定每份的记录数量进行分割

* `tf.split(value, num_or_size_splits, axis)`

```python
a = tf.constant(
    [[1.0, 2.0], 
     [3.0, 4.0]]
)
tf.print(a.shape)  # shape=(2, 2)

b = tf.constant(
    [[5.0, 6.0], 
     [7.0, 8.0]]
)
tf.print(b.shape)  # shape=(2, 2)

c = tf.constant(
    [[9.0, 10.0], 
     [11.0, 12.0]]
)
tf.print(c.shape)  # shape=(2, 2)

d = tf.concat([a, b, c], axis = 0)
tf.print(d)
```

```
<tf.Tensor: shape=(6, 2), dtype=float32, numpy=
array([[ 1.,  2.],
       [ 3.,  4.],
       [ 5.,  6.],
       [ 7.,  8.],
       [ 9., 10.],
       [11., 12.]], dtype=float32)>
```

```python
# 指定分割份数，平均分割
tf.split(d, 3, axis = 0)
```

```
[<tf.Tensor: shape=(2, 2), dtype=float32, numpy=
 array([[1., 2.],
        [3., 4.]], dtype=float32)>,
 <tf.Tensor: shape=(2, 2), dtype=float32, numpy=
 array([[5., 6.],
        [7., 8.]], dtype=float32)>,
 <tf.Tensor: shape=(2, 2), dtype=float32, numpy=
 array([[ 9., 10.],
        [11., 12.]], dtype=float32)>]
```

```python
# 指定每份的记录数量
tf.split(d, [2, 2, 2], axis = 0)
```

```
[<tf.Tensor: shape=(2, 2), dtype=float32, numpy=
 array([[1., 2.],
        [3., 4.]], dtype=float32)>,
 <tf.Tensor: shape=(2, 2), dtype=float32, numpy=
 array([[5., 6.],
        [7., 8.]], dtype=float32)>,
 <tf.Tensor: shape=(2, 2), dtype=float32, numpy=
 array([[ 9., 10.],
        [11., 12.]], dtype=float32)>]
```

## 张量数学运算

张量的数学运算符可以分为标量运算符、向量运算符、矩阵运算符

### 标量运算

标量运算符的特点是对张量实施逐元素运算，
有些标量运算符对常用的数学运算符进行了重载，并且支持类似 numpy 的广播特性

加、减、乘、除、乘方、三角函数、指数、对数等常见函数，
逻辑比较运算符等都是标量运算符，许多标量运算符都在 `tf.math` 模块下

```python
import tensorflow as tf
import numpy as np

a = tf.constant([[1.0, 2], [-3, 4.0]])
b = tf.constant([[5.0, 6], [7.0, 8.0]])

# 运算符重载
tf.print(a + b)
tf.print(a - b)
tf.print(a * b)
tf.print(a / b)
tf.print(a ** 2)
tf.print(a ** (0.5))
tf.print(a % 3)
tf.print(a // 3)
tf.print(a >= 2)
tf.print((a >= 2) & (a <= 3))
tf.print((a >= 2) | (a <= 3))
tf.print(a == 5)  # tf.equal(a, 5)
tf.print(tf.sqrt(a))
```


```python
import tensorflow as tf
import numpy as np

a = tf.constant([2.6, -2.7])
b = tf.constant([5.0, 6.0])
c = tf.constant([6.0, 7.0])

tf.print(tf.add_n([a, b, c]))
tf.print(tf.maximum(a, b))
tf.print(tf.minimum(a, b))

tf.print(tf.math.round(a))  # 保留整数部分，四舍五入
tf.print(tf.math.floor(a))  # 保留整数部分，向下归整
tf.print(tf.math.ceil(a))  # 保留整数部分，向上归整
```

* 幅值裁剪

```python
x = tf.constant([0.9, -0.8, 100.0, -20.0, 0.7])

y = tf.clip_by_value(x, clip_value_min = -1, clip_value_max = 1)
z = tf.clip_by_norm(x, clip_norm = 3)
tf.print(y)
tf.print(z)
```


### 向量运算

向量运算符只在一个特定轴上运算，将一个向量映射到一个标量或者另一个向量。
许多向量运算符都以 `reduce` 开头

* 向量 reduce

```python
a = tf.range(1, 10)

tf.print(tf.reduce_sum(a))
tf.print(tf.reduce_mean(a))
tf.print(tf.reduce_max(a))
tf.print(tf.reduce_min(a))
tf.print(tf.reduce_prod(a))
```

* 张量指定维度进行 reduce

```python
a = tf.range(1, 10)
b = tf.reshape(a, (3, 3))

tf.print(tf.reduce_sum(b, axis = 1, keepdims = True))
tf.print(tf.reduce_sum(b, axis = 1, keepdims = True))
```

* bool 类型的 reduce

```python
p = tf.constant([True, False, False])
q = tf.constant([False, False, True])
tf.print(tf.reduce_all(p))
tf.print(tf.reduce_any(q))
```




### 矩阵运算


### 广播机制


# 自动微分机制

神经网络通常依赖反向传播求梯度来更新网络参数，求梯度过程通常是一件非常复杂而容易出错的事情，
而深度学习框架可以帮助自动地完成求梯度运算

TensorFlow 一般使用地图磁带 `tf.GradientTape` 来记录正向运算过程，然后反播磁带自动得到梯度值，
这种利用 `tf.GradientTape` 求微分的方法叫做 TensorFlow 的自动微分机制

## 利用梯度磁带求导数

```python
```


## 利用梯度磁带和优化器求最小值

```python

```






# 计算图

计算图由节点(nodes)和线(edges)组成:

* 节点表示操作符 Operation，或者称之为算子
* 线表示计算间的依赖

实现表示有数据的传递依赖，传递的数据即张量，虚线通常可以表示控制依赖，即执行先后顺序

TensorFlow 有三种计算图的构建方式

* 静态计算图
* 动态计算图
* AutoGraph

## 静态计算图

TensorFlow 1.0 采用的是静态计算图，需要先使用 TensorFlow 的各种算子创建计算图，
然后再开启一个会话 Session，显式执行计算图

在 TensorFlow 1.0 中，使用静态计算图分两步:

1. 定义计算图
2. 在会话中执行计算图



## 动态计算图

TensorFlow 2.0 采用的是动态计算图，即每使用一个算子后，
该算子会被动态加入到隐含的默认计算图中立即执行得到结果，
而无需开启 Session

* 使用动态计算图(Eager Excution)的好处是方便调试程序
    - 动态计算图会让 TensorFlow 代码的表现和 Python 原生代码的表现一样，
      写起来就像 Numpy 一样，各种日志打印，控制流全部都是可以使用的
* 使用动态图的缺点是运行效率相对会低一点
    - 因为使用动态图会有许多词 Python 进程和 TensorFlow 的 C++ 进程之间的通信
    - 而静态计算图构建完成之后几乎全部在 TensorFlow 内核上使用 C++ 代码执行，效率更高。
      此外，静态图会对计算步骤进行一定的优化，去除和结果无关的计算步骤 

## AutoGraph

如果需要在 TensorFlow 中使用静态图，
可以使用 `@tf.function` 装饰器将普通 Python 函数转换成对应的 TensorFlow 计算图构建代码。
运行该函数就相当于在 TensorFlow 1.0 中用 Session 执行代码。使用 `@tf.function` 构建静态图的方式叫做 AutoGraph


### AutoGraph 使用规范


### AutoGraph 机制原理


### AutoGraph 和 tf.Module


