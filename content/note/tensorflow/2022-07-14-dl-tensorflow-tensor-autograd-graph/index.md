---
title: TensorFlow 张量、自动微分机制和计算图
author: 王哲峰
date: '2022-07-14'
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
      - [常量值不可改变](#常量值不可改变)
      - [变量值可以改变](#变量值可以改变)
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
      - [tf.broadcast_to 以显式的方式按照广播机制扩展张量的维度](#tfbroadcast_to-以显式的方式按照广播机制扩展张量的维度)
      - [计算广播后计算结果的形状，静态形状，TensorShape 类型参数](#计算广播后计算结果的形状静态形状tensorshape-类型参数)
      - [计算广播后计算结果的形状，动态形状，Tensor 类型参数](#计算广播后计算结果的形状动态形状tensor-类型参数)
      - [广播效果](#广播效果)
- [自动微分机制](#自动微分机制)
  - [利用梯度磁带求导数](#利用梯度磁带求导数)
    - [变量张量求导](#变量张量求导)
    - [常量张量求导](#常量张量求导)
    - [求二阶导数](#求二阶导数)
    - [在 AutoGraph 中使用梯度磁带求导](#在-autograph-中使用梯度磁带求导)
  - [利用梯度磁带和优化器求最小值](#利用梯度磁带和优化器求最小值)
- [计算图](#计算图)
  - [静态计算图](#静态计算图)
    - [TensorFlow 1.0 静态计算图](#tensorflow-10-静态计算图)
    - [TensorFlow 2.0 计算图](#tensorflow-20-计算图)
  - [动态计算图](#动态计算图)
    - [普通动态计算图](#普通动态计算图)
    - [动态计算图封装](#动态计算图封装)
  - [AutoGraph](#autograph)
    - [AutoGraph 示例](#autograph-示例)
    - [AutoGraph 使用规范](#autograph-使用规范)
      - [使用 TensorFlow 函数](#使用-tensorflow-函数)
      - [避免装饰的函数内部定义 tf.Variable](#避免装饰的函数内部定义-tfvariable)
      - [装饰的函数内部不可修改外部 Python 数据结构](#装饰的函数内部不可修改外部-python-数据结构)
    - [AutoGraph 机制原理](#autograph-机制原理)
      - [创建装饰函数原理](#创建装饰函数原理)
      - [调用装饰函数原理](#调用装饰函数原理)
      - [再次调用装饰函数原理](#再次调用装饰函数原理)
      - [再次调用装饰函数原理](#再次调用装饰函数原理-1)
    - [AutoGraph 和 tf.Module](#autograph-和-tfmodule)
      - [AutoGraph 和 tf.Module 概述](#autograph-和-tfmodule-概述)
      - [应用 tf.Module 封装 AutoGraph](#应用-tfmodule-封装-autograph)
      - [tf.Module 和 tf.keras.Model, tf.keras.layers.Layer](#tfmodule-和-tfkerasmodel-tfkeraslayerslayer)
</p></details><p></p>

> 张量和计算图是 TensorFlow 的核心概念

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
print(tf.string == np.unicode)  # False, tf.string 类型和 np.unicode 类型不等价
print(tf.bool == np.bool)  # True
```

#### 数据维度

不同类型的数据可以用不同维度(rank)的张量来表示:

* 标量为 0 维张量
* 向量为 1 维张量
* 矩阵为 2 维张量
* 彩色图像有 RGB 三个通道，可以表示为 3 维张量
* 视频还有时间维，可以表示为 4 维张量

可以简单地总结为: 有几层中括号，就是多少维的张量

* 标量，0 维张量

```python
scalar = tf.constant(True)
print(tf.rank(scalar))

# tf.rank 的作用和 numpy 的 ndim 方法相同
print(scalar.numpy().ndim)
```

```
tf.Tensor(0, shape=(), dtype=int32)
0
```

* 向量，1 维张量

```python
vector = tf.constant([1.0, 2.0, 3.0, 4.0])
print(tf.rank(vector))

# tf.rank 的作用和 numpy 的 ndim 方法相同
print(np.ndim(vector.numpy()))
```

```
tf.Tensor(1, shape=(), dtype=int32)
1
```

* 矩阵，2 维张量

```python
matrix = tf.constant(
    [[1.0, 2.0],
     [3.0, 4.0]]
)
print(tf.rank(matrix))

# tf.rank 的作用和 numpy 的 ndim 方法相同
print(np.ndim(matrix).numpy())
```

```
2
2
```

* 3 维张量

```python
tensor3 = tf.constant(
    [[[1.0, 2.0],
      [3.0, 4.0]],
     [[5.0, 6.0],
      [7.0, 8.0]]]
)
print(tensor3)
print(tf.rank(tensor3))
```

```
tf.Tensor(
[[[1. 2.]
  [3. 4.]]

 [[5. 6.]
  [7. 8.]]], shape=(2, 2, 2), dtype=float32)

tf.Tensor(3, shape=(), dtype=int32)
```

* 4 维张量

```python
tensor4 = tf.constant(
    [[[[1.0, 1.0], 
       [2.0, 2.0]], 
      [[3.0, 3.0], 
       [4.0, 4.0]]],
     [[[5.0, 5.0], 
       [6.0, 6.0]], 
      [[7.0, 7.0], 
       [8.0, 8.0]]]],
)
print(tensor4)
print(tf.rank(tensor4))
```

```
tf.Tensor(
[[[[1. 1.]
   [2. 2.]]

  [[3. 3.]
   [4. 4.]]]


 [[[5. 5.]
   [6. 6.]]

  [[7. 7.]
   [8. 8.]]]], shape=(2, 2, 2, 2), dtype=float32)
tf.Tensor(4, shape=(), dtype=int32)
```

#### 数据转换

* 可以用 `tf.cast` 改变张量的数据类型

```python
h = tf.constant([123, 456], dtype = tf.int32)
f = tf.cast(h, tf.float32)
print(h.dtype)  # <dtype: 'int32'>
print(f.dtype)  # <dtype: 'float32'>
```

* 可以用 `.numpy()` 的方法将 TensorFlow 中的张量转换为 Numpy 中的张量，
  使用 `shape` 方法查看张量尺寸

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

模型中需要被训练的参数一般被设置成变量

#### 常量值不可改变

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

#### 变量值可以改变

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
    - `tf.boolean_mask` 功能最为强大，它可以实现 `tf.gather`、`tf.gather_nd` 的功能，
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

```python
# 考虑班级成绩册的例子，有4个班级，每个班级10个学生，每个学生7门科目成绩。
# 可以用一个4×10×7的张量来表示
scores = tf.random.uniform((4, 10, 7), minval = 0, maxval = 100 , dtype = tf.int32)
tf.print(scores)

# #抽取每个班级第0个学生，第5个学生，第9个学生的全部成绩
p = tf.gather(scores, [0, 5, 9], axis = 1)
tf.print(p)
# or
p = tf.boolean_mask(
    scores, 
    [True, False, False, False, False, 
     True, False, False, False, True], 
axis = 1
)
tf.print(p)

# 抽取每个班级第0个学生，第5个学生，第9个学生的第1门课程，第3门课程，第6门课程成绩
q = tf.gather(tf.gather(scores, [0, 5, 9], axis = 1), [1, 3, 6], axis = 2)
tf.print(q)

# 抽取第0个班级第0个学生，第2个班级的第4个学生，第3个班级的第6个学生的全部成绩
# indices的长度为采样样本的个数，每个元素为采样位置的坐标
s = tf.gather_nd(scores, indices = [(0, 0), (2, 4), (3, 6)])
s

#抽取第0个班级第0个学生，第2个班级的第4个学生，第3个班级的第6个学生的全部成绩
s = tf.boolean_mask(scores,
    [[True,False,False,False,False,False,False,False,False,False],
     [False,False,False,False,False,False,False,False,False,False],
     [False,False,False,False,True,False,False,False,False,False],
     [False,False,False,False,False,False,True,False,False,False]])
tf.print(s)


# 利用tf.boolean_mask可以实现布尔索引
# 找到矩阵中小于0的元素
c = tf.constant([[-1,1,-1],[2,2,-2],[3,-3,3]],dtype=tf.float32)
tf.print(c,"\n")

tf.print(tf.boolean_mask(c,c<0),"\n") 
tf.print(c[c<0]) #布尔索引，为boolean_mask的语法糖形式
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

`tf.reshape` 可以改变张量的形状，但是其本质上不会改变张量元素的存储顺序，
所以，该操作实际上非常迅速，并且是可逆的

```python
a = tf.random.uniform(
    shape = [1, 3, 3, 2],
    minval = 0,
    maxval = 255,
    dtype = tf.int32,
)
tf.print(a.shape)
tf.print(a)
```

```python
# 改成 (3, 6) 形状
b = tf.reshape(a, [3, 6])
tf.print(b.shape)
tf.print(b)
```

```python
# 改成 [1, 3, 3, 2] 形状的张量
c = tf.reshape(b, [1, 3, 3, 2])
tf.print(c)
```

如果张量在某个维度上只有一个元素，利用 `tf.squeeze` 可以消除这个维度，
和 `tf.reshape` 相似，它本质上不会改变张量元素的存储顺序。
张量的各个元素在内存中是线性存储的，其一般规律是，同一层级中的相邻元素的物理地址也相邻

```python
s = tf.squeeze(a)
tf.print(s.shape)
tf.print(s)
```

在第 0 维插入长度为 1 的一个维度

```python
d = tf.expand_dims(s, axis = 0)
d
```


`tf.transpose` 可以交换张量的维度，与 `tf.reshape` 不同，它会改变张量元素的存储顺序。
`tf.transpose` 常用于图片存储格式的变换上

```python
# Batch, Height, Width, Channel
a = tf.random.uniform(shape = [100, 600, 600, 4], minval = 0, maxval = 255, dtype = tf.int32)
tf.print(a.shape)
```

```python
# Channel, Height, Width, Batch
s = tf.transpose(a, perm = [3, 1, 2, 0])
tf.print(s.shape)
```




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
有些标量运算符对常用的数学运算符进行了重载，并且支持类似 numpy 的广播特性，
许多标量运算符都在 `tf.math` 模块下

* 加、减、乘、除、乘方、三角函数、指数、对数等常见函数，
  逻辑比较运算符等都是标量运算符

```python
import numpy as np
import tensorflow as tf

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

* TODO

```python
import numpy as np
import tensorflow as tf

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
import numpy as np
import tensorflow as tf

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

* `tf.foldr` 实现 `tf.reduce_sum`

```python
s = tf.foldr(lambda a, b: a + b, tf.range(10))
tf.print(s)
```

* cum 扫描累计

```python
a = tf.range(1, 10)

tf.print(tf.math.cumsum(a))
tf.print(tf.math.cumprod(a))
```

* arg 最大、最小值索引

```python
a = tf.range(1, 10)
tf.print(tf.argmax(a))
tf.print(tf.argmin(a))
```

* `tf.math.top_k` 可以用于对张量排序

```python
a = tf.constant([1, 3, 7, 5, 4, 8])
values, indices = tf.math.top_k(a, 3, sorted = True)
tf.print(values)
tf.print(indices)
```

* 使用 `tf.math.top_k` 可以在 TensorFlow 中实现 KNN 算法

```python
# TODO
```

### 矩阵运算

矩阵运算包括:

* 矩阵乘法
* 矩阵转置
* 矩阵求逆
* 矩阵求迹
* 矩阵范数
* 矩阵行列式
* 矩阵求特征值
* 矩阵分解

除了一些常用的运算外，大部分和矩阵有关的运算都在 `tf.linalg` 子包中

* 矩阵乘法

```python
a = tf.constant([[1, 2], [3, 4]])
b = tf.constant([[2, 0], [0, 2]])
a@b
# 等价于
tf.matmul(a, b)
```

* 矩阵转置

```python
a = tf.constant([[1, 2], [3, 4]])
tf.transpose(a)
```

* 矩阵求逆

必须为 `tf.float32` 或 `tf.double`

```python
a = tf.constant([[1.0, 2], [3, 4]], dtype = tf.float32)
tf.linalg.inv(a)
```

* 矩阵求迹(trace)

```python
a = tf.constant([[1.0, 2], [3, 4]], dtype = tf.float32)
tf.linalg.trace(a)
```

* 矩阵范数

```python
a = tf.constant([[1.0, 2], [3, 4]])
tf.linalg.norm(a)
```

* 矩阵行列式

```python
a = tf.constant([[1,0, 2], [3, 4]])
tf.linalg.det(a)
```

* 矩阵求特征值

```python
a = tf.constant([[1.0, 2], [-5, 4]])
tf.linalg.eigvals(a)
```

* 矩阵分解

```python
# 矩阵 QR 分解，讲一个方阵分解为一个正交矩阵 Q 和上三角矩阵
# QR 分解实际上是对矩阵 A 实施 Schmidt 正交化得到 Q
a = tf.constant([[1.0, 2.0], [3.0, 4.0]], dtype = tf.float32)
q, r = tf.linalg.qr(a)
tf.print(q)
tf.print(r)
tf.print(q@r)


# 矩阵 SVD 分解
# SVD 分解可以将任意一个矩阵分解为一个正交矩阵 U，一个对角矩阵 S 和一个正交矩阵 V.t() 的乘积
# SVD 常用于矩阵压缩和降维
a = tf.constant([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]], dtype = tf.float32)
s, u, v = tf.linalg.svd(a)
tf.print(u, "\n")
tf.print(s, "\n")
tf.print(v, "\n")
tf.print(u@tf.linalg.diag(s)@tf.transpose(v))
```

### 广播机制

TensorFlow 的广播规则和 Numpy 是一样的:

1. 如果张量的维度不同，将维度较小的张量进行扩展，直到两个张量的维度都一样
2. 如果两个张量在某个维度上的长度是相同的，或者其中一个张量在该维度上的长度为 1，
   那么我们就说这两个张量在该维度上是相容的
4. 如果两个张量在所有维度上都是相容的，它们就能使用广播
5. 广播之后，每个维度的长度将取两个张量在该维度长度的较大值
6. 在任何一个维度上，如果一个张量的长度为 1，另一个张量长度大于 1，那么在该维度上，
   就好像是对第一个张量进行了复制

#### tf.broadcast_to 以显式的方式按照广播机制扩展张量的维度

```python
a = tf.constant([1, 2, 3])
b = tf.constant(
    [[0, 0, 0],
     [1, 1, 1],
     [2, 2, 2]]
)
b + a
# 等价于
b + tf.broadcast_to(a, b.shape)
```

#### 计算广播后计算结果的形状，静态形状，TensorShape 类型参数

```python
tf.broadcast_static_shape(a.shape, b.shape)
```

#### 计算广播后计算结果的形状，动态形状，Tensor 类型参数

```python
c = tf.constant([1, 2, 3])
d = tf.constant([[1], [2], [3]])
tf.broadcast_dynamic_shape(tf.shape(c), tf.shape(d)) 
```

#### 广播效果

```python
c + d
# 等价于
tf.broadcast_to(c, [3, 3]) + tf.broadcast_to(d, [3, 3])
```

# 自动微分机制

神经网络通常依赖反向传播求梯度来更新网络参数，求梯度过程通常是一件非常复杂而容易出错的事情，
而深度学习框架可以帮助自动地完成求梯度运算

TensorFlow 一般使用地图磁带 `tf.GradientTape` 来记录正向运算过程，然后反播磁带自动得到梯度值，
这种利用 `tf.GradientTape` 求微分的方法叫做 TensorFlow 的自动微分机制

## 利用梯度磁带求导数

* 求 `$f(x) = a \times x^{2} + b \times x + c$` 的导数

```python
import tensorflow as tf
import numpy as np

x = tf.Variable(0.0, name = "x", dtype = tf.float32)
a = tf.constant(1.0)
b = tf.constant(-2.0)
c = tf.constant(1.0)
```

### 变量张量求导

```python
with tf.GradientTape() as tape:
    y = a * tf.pow(x, 2) + b * x + c

dy_dx = tape.gradient(y, x)
print(dy_dx)
```

```
tf.Tensor(-2.0, shape=(), dtype=float32)
```

### 常量张量求导

对常量张量也可以求导，需要增加 watch

```python
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

### 求二阶导数

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

### 在 AutoGraph 中使用梯度磁带求导

```python
@tf.function
def f(x):
    a = tf.constant(1.0)
    b = tf.constant(-2.0)
    c = tf.constant(1.0)

    # 自变量转换成 tf.float32
    x = tf.cast(x, tf.float32)
    with tf.GradientTape() as tape:
        tap.watch(x)
        y = a * tf.pow(x, 2) + b * x + c
    dy_dx = tape.gradient(y, x)

    return ((dy_dx, y))

tf.print(f(tf.constant(0.0)))
tf.print(f(tf.constant(1.0)))
```

## 利用梯度磁带和优化器求最小值

```python

```











# 计算图

计算图由节点(nodes)和线(edges)组成:

* 节点表示操作符(Operation)，或者称之为算子
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

### TensorFlow 1.0 静态计算图

```python
import tensorflow as tf

# 定义计算图
graph = tf.Graph()
with graph.as_default():
    # placeholder 为占位符，执行会话的时候指定填充对象
    x = tf.placeholder(name = "x", shape = [], dtype = tf.string)
    y = tf.placeholder(name = "y", shape = [], dtype = tf.string)
    z = tf.string_join([x, y], name = "join", separator = " ")

# 执行计算图
with tf.Session(graph = graph) as sess:
    print(sess.run(
        fetches = z, 
        feed_dict = {x: "hello", y: "world"}
    ))
```

### TensorFlow 2.0 计算图

TensorFlow 2.0 为了确保对老版本 TensorFlow 项目的兼容性，
在 `tf.compat.v1` 子模块中保留了对 TensorFlow 1.0 静态计算图构建风格的支持。
已经不推荐使用了

```python
import tensorflow as tf

# 定义计算图
graph = tf.compat.v1.Graph()
with graph.as_default():
    # placeholder 为占位符，执行会话的时候指定填充对象
    x = tf.compat.v1.placeholder(
        name = "x", 
        shape = [], 
        dtype = tf.string
    )
    y = tf.compat.v1.placeholder(
        name = "y", 
        shape = [], 
        dtype = tf.string
    )
    z = tf.strings.join([x, y], name = "join", separator = " ")

# 执行计算图
with tf.compat.v1.Session(graph = graph) as sess:
    # fetches 的结果非常像一个函数的返回值
    # feed_dict 中的占位符相当于函数的参数序列
    result = sess.run(
        fetches = z,
        feed_dict = {
            x: "hello",
            y: "world",
        }
    )
    print(result)
```

## 动态计算图

TensorFlow 2.0 采用的是动态计算图，即每使用一个算子后，
该算子会被动态加入到隐含的默认计算图中立即执行得到结果，
而无需开启 Session

在 TensorFlow 2.0 中，使用的是动态计算图和 AutoGraph。
动态计算图已经不区分计算图的定义和执行了，
而是定义后立即执行，因此称之为 Eager Excution，立即执行

* 使用动态计算图(Eager Excution)的好处是方便调试程序
    - 动态计算图会让 TensorFlow 代码的表现和 Python 原生代码的表现一样，
      写起来就像 Numpy 一样，各种日志打印，控制流全部都是可以使用的
* 使用动态图的缺点是运行效率相对会低一点
    - 因为使用动态图会有许多次 Python 进程和 TensorFlow 的 C++ 进程之间的通信
    - 而静态计算图构建完成之后几乎全部在 TensorFlow 内核上使用 C++ 代码执行，效率更高。
      此外，静态图会对计算步骤进行一定的优化，去除和结果无关的计算步骤 

### 普通动态计算图

* 动态计算图在每个算子处都进行构建，构建后立即执行

```python
x = tf.constant("hello")
y = tf.constant("world")
z = tf.strings.join([x, y], separator = " ")
tf.print(z)
```

```
hello world
```

### 动态计算图封装

* 可以将动态计算图代码的输入和输出关系封装成函数

```python
def strjoin(x, y):
    z = tf.strings.join([x, y], separator = " ")
    tf.print(z)
    return z

result = strjoin(
    x = tf.constant("hello"),
    y = tf.constant("world"),
)
pritn(result)
```

```
hello world
tf.Tensor(b'hello world', shape=(), dtype=string)
```

## AutoGraph

动态计算图运行效率相对较低，如果需要在 TensorFlow 中使用静态图，
可以使用 `@tf.function` 装饰器将普通 Python 函数转换成对应的 TensorFlow 计算图构建代码。
运行该函数就相当于在 TensorFlow 1.0 中用 Session 执行代码。
使用 `@tf.function` 构建静态图的方式叫做 AutoGraph

在 TensorFlow 2.0 中，使用 AutoGraph 的方式使用计算图分两步:

1. 定义计算图变成了定义函数
2. 执行计算图变成了调用函数

在 AutoGraph 中不需要使用会话，一切都像原始的 Python 语法一样自然。
实践中，一般会先用动态计算图调试，
然后在需要提高性能的地方利用 `@tf.function` 切换成 AutoGraph 获得更高的效率。
当然，`@tf.function` 的使用需要遵循一定的规范

### AutoGraph 示例

* 使用 AutoGraph 构建静态图

```python
import tensorflow as tf

@tf.function
def strjoin(x, y):
    z = tf.strings.join([x, y], separator = " ")
    tf.print(z)
    return z

result = strjoin(
    x = tf.constant("hello"),
    y = tf.constant("world"),
)
print(result)
```

```
hello world
tf.Tensor(b'hello world', shape=(), dtype=string)
```

* 创建日志

```python
import os
import datetime
from pathlib import Path

stamp = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:S')
logdir = os.path.join("data", "autograph", stamp)
# or
stamp = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:S')
logdir = str(Path("./data/autograph/" + stamp))
```

* 查看计算图

```python
import tensorflow as tf

# 日志写入器
writer = tf.summary.create_file_writer(logdir)

# 开启 AutoGraph 跟踪
tf.summary.trace_on(graph = True, profiler = True)

# 执行 AutoGraph
result = strjoin(
    x = "hello",
    y = "world",
)

# 将计算图信息写入日志
with writer.as_default():
    tf.summary.trace_export(
        name = "autograph",
        step = 0,
        profile_outdir = logdir,
    )
```

* 启动 tensorboard 在 jupyter 中的魔法命令

```python
%load_ext tensorboard
%tensorboard --logdir ./data/autograph/
```

### AutoGraph 使用规范

* 静态计算图执行效率很高，但较难调试
* 动态计算图易于调试，编码效率低，执行效率偏低
* AutoGraph 机制可以将动态图转换成静态计算图，兼收执行效率和编码效率之利

AutoGraph 机制能够转换的代码并不是没有任何约束的，
有一些编码规范需要遵循，否则有可能会转换失败或者不符合预期。
所以这里总结了 AutoGraph 编码规范

#### 使用 TensorFlow 函数

被 `@tf.function` 修饰的函数应尽可能使用 TensorFlow 中的函数而不是 Python 中的其他函数。例如:

- 使用 `tf.print()` 而不是 `print()`
- 使用 `tf.range()` 而不是 `range()`
- 使用 `tf.constant(True)` 而不是 `True`

解释：Python 中的函数仅仅会在跟踪执行函数以创建静态图的阶段使用，
普通 Python 函数是无法嵌入到静态计算图中的，所以在计算图构建好之后再次调用的时候，
这些 Python 函数并没有被计算，而 TensorFlow 中的函数则可以嵌入到计算图中。
使用普通的 Python 函数会导致 被 `@tf.function` 修饰前【eager 执行】和
被 `@tf.function` 修饰后【静态图执行】的输出不一致

```python
import numpy as np
import tensorflow as tf

@tf.function
def np_random():
    a = np.random.randn(3, 3)
    tf.print(a)

# np_random() 每次执行都是一样的结果
np_random()
np_random()
```

```python
import tensorflow as tf

@tf.function
def tf_random():
    a = tf.random.normal((3, 3)
    tf.print(a)

# tf_random() 每次执行都会重新生成随机数
tf_random()
tf_random()
```

#### 避免装饰的函数内部定义 tf.Variable

避免在 `@tf.function` 修饰的函数内部定义 `tf.Variable`

解释：如果函数内部定义了 `tf.Variable`, 那么在【eager执行】时，
这种创建 `tf.Variable` 的行为在每次函数调用时候都会发生。但是在【静态图执行】时，
这种创建 `tf.Variable` 的行为只会发生在第一步跟踪 Python 代码逻辑创建计算图时，
这会导致被 `@tf.function` 修饰前【eager执行】和被 `@tf.function` 修饰后【静态图执行】的输出不一致。
实际上，TensorFlow 在这种情况下一般会报错

```python
import tensorflow as tf

x = tf.Variable(1.0, dtype = tf.float32)

@tf.function
def outer_var():
    x.assign_add(1.0)
    tf.print(x)
    return (x)

outer_var()
```

```python
import tensorflow as tf

@tf.function
def inner_var():
    x = tf.Variable(1.0, dtype = tf.float32)
    x.assign_add(1.0)
    tf.print(x)
    return (x)

# 执行将报错
inner_var()
```

#### 装饰的函数内部不可修改外部 Python 数据结构

被 `@tf.function` 修饰的函数不可修改该函数外部的 Python 列表或字典等数据结构变量

解释：静态计算图是被编译成 C++ 代码在 TensorFlow 内核中执行的。
Python 中的列表和字典等数据结构变量是无法嵌入到计算图中，
它们仅仅能够在创建计算图时被读取，
在执行计算图时是无法修改 Python 中的列表或字典这样的数据结构变量的

```python
tensor_list = []

def append_tensor(x):
    tensor_list.append(x)
    return tensor_list

append_tensor(tf.constant(5.0))
append_tensor(tf.constant(6.0))
print(tensor_list)
```

```
[<tf.Tensor: shape=(), dtype=float32, numpy=5.0>, 
 <tf.Tensor: shape=(), dtype=float32, numpy=6.0>]
```

```python
tensor_list = []

@tf.function  # 加上这一行切换成 AutoGraph 结果将不符合预期
def append_tensor(x):
    tensor_list.append(x)
    return tensor_list

append_tensor(tf.constant(5.0))
append_tensor(tf.constant(6.0))
print(tensor_list)
```

```
[<tf.Tensor 'x:0' shape=() dtype=float32>]
```

### AutoGraph 机制原理

#### 创建装饰函数原理

当使用 `@tf.function` 装饰一个函数的时候，背后发生了什么？

- 背后什么都没发生，仅仅是在 Python 堆栈中记录了这样一个函数的签名

```python
import numpy as np
import tensorflow as tf

@tf.function(autograph = True)
def my_add(a, b):
    for i in tf.range(3):
        tf.print(i)
    c = a + b
    print("tracing")
    
    return c
```

#### 调用装饰函数原理

当第一次调用被 `@tf.function` 装饰的函数时，背后发生了什么？

```python
my_add(
    a = tf.constant("hello"),
    b = tf.constant("world"),
)
```

```
traceing
0
1
2
```

1. 创建计算图

即创建一个静态计算图，跟踪执行一遍函数体中的 Python 代码，
确定各个变量的 Tensor 类型，并根据执行顺序将算子添加到计算图中

在这个过程中，如果开启了 `autograph=True` (默认开启)，
会将 Python 控制流转换成 TensorFlow 图内控制流。 
主要是将 `if` 语句转换成 `tf.cond` 算子表达，
将 `while` 和 `for` 循环语句转换成 `tf.while_loop` 算子表达，
并在必要的时候添加 `tf.control_dependencies` 指定执行顺序依赖关系

相当于在 TensorFlow 1.0 执行了类似下面的语句:

```python
graph = tf.Graph()
with graph.as_default():
    a = tf.placeholder(shape = [], dtype = tf.string)
    b = tf.placeholder(shape = [], dtype = tf.string)
    cond = lambda i: i < tf.constant(3)
    def body(i):
        tf.print(i)
        return (i + 1)
    loop = tf.while_loop(cond, body, loop_vars = [0])
    loop
    with tf.control_dependencies(loop):
        c = tf.strings.join([a, b])
    print("tracing")
```

2. 执行计算图

相当于在 TensorFlow 1.0 中执行了下面的语句

```python
with tf.Session(graph = graph) as sess:
    sess.run(c, feed_dict = {
        a: tf.constant("hello"),
        b: tf.constant("world"),
    })
```

#### 再次调用装饰函数原理

当再次用相同的输入参数类型调用被 `@tf.function` 装饰的函数时，背后发生了什么？

```python
my_add(
    a = tf.constant("good"),
    b = tf.constant("morning"),
)
```

```
0
1
2
```

#### 再次调用装饰函数原理

当再次用不同的的输入参数类型调用被 `@tf.function` 装饰的函数时，背后到底发生了什么？

* 由于输入参数的类型已经发生变化，已经创建的计算图不能够再次使用。
  需要重新做2件事情：创建新的计算图、执行计算图

```python
my_add(
    a = tf.constant(1),
    b = tf.constant(2),
)
```

```
tracing
0
1
2
```

* 需要注意的是，如果调用被 `@tf.function` 装饰的函数时输入的参数不是 `Tensor` 类型，
  则每次都会重新创建计算图

```python
my_add("hello", "world")
my_add("good", "moning")
```

```
tracing
0
1
2
tracing
0
1
2
```

### AutoGraph 和 tf.Module

#### AutoGraph 和 tf.Module 概述

AutoGraph 的编码规范中提到，在构建 AutoGraph 时应该避免在 `@tf.function` 修饰的函数内部定义 `tf.Variable`。
但是如果在被修饰的函数外部定义 `tf.Variable`，又会显得这个函数由外部变量依赖，封装不够完美。
一种简单的思路是定义一个类，并将相关的 `tf.Variable` 创建放在类的初始化方法中。而将函数的逻辑放在其他方法中。

TensorFlow 提供了一个基类 `tf.Module`，通过继承它构建子类，不仅可以解决上述问题，而且可以非常方便地管理变量，
还可以非常方便地管理它引用的其他 Module，最重要的是，能够利用 `tf.saved_model` 保存模型并实现跨平台部署使用

实际上，`tf.keras.models.Model`、`tf.keras.layers.Layer` 都是继承自 `tf.Module` 的，
提供了方便的变量管理和所引用的子模块管理的功能

#### 应用 tf.Module 封装 AutoGraph

* 定义一个简单的 function

```python
import tensorflow as tf

x = tf.Variable(1.0, dtype = tf.float32)

@tf.function(
    input_signature = [tf.TensorSpec(shape = [], dtype = tf.float32)]
)
def add_print(a):
    x.assign_add(a)
    tf.print(x)
    return (x)

add_print(tf.constant(3.0))
add_print(tf.constant(3))  # 输入不符合张量签名的参数将报错
```

```
4
```

* 利用 `tf.Module` 封装函数

```python
class DemoModule(tf.Module):
    def __init__(self, init_value = tf.constant(0.0), name = None):
        super(DemoModule, self).__init__(name = name)
        with self.name_scope:
            self.x = tf.Variable(
                init_value, 
                dtype = tf.float32, 
                trainable = True
            )
    
    @tf.function(input_signature = [tf.TensorSpec(shape = [], dtype = tf.float32)])
    def add_print(self, a):
        with self.name_scope:
            self.x.assign_add(a)
            tf.print(self.x)
            return (self.x)
```

* 调用类

```python
demo = DemoModule(init_value = tf.constant(1.0))
result = demo.add_print(tf.constant(5.0))
```

```
6
```

* 查看模块中的全部变量和全部可训练变量

```python
print(demo.variables)
print(demo.trainable_variables)
```

```
(<tf.Variable 'demo_module/Variable:0' shape=() dtype=float32, numpy=6.0>,)
(<tf.Variable 'demo_module/Variable:0' shape=() dtype=float32, numpy=6.0>,)
```

* 查看模块中的全部子模块

```python
demo.submodules
```

* 使用 `tf.saved_model` 保存模型，并指定需要跨平台部署的方法

```python
tf.saved_model.save(
    demo, 
    "./data/demo/1", 
    signatures = {
        "serving_default": demo.add_print
    }
)
```

* 加载模型

```python
demo2 = tf.saved_model.load("./data/demo/1")
demo2.add_print(tf.constant(5.0))
```

```
11
```

* 查看模型文件相关信息

```bash
$ !sabed_model_cli show --dir ./data/demo/1 --all
```

![img](images/)

* 在 TensorBoard 中查看计算图，模块会被添加模块名 `demo_module`，方便层次化呈现计算图结构

```python
import datetime

# 创建日志
stamp = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:S')
logdir = f"./data/demomodule/{stamp}"
writer = tf.summary.create_file_writer(logdir)

# 开启 autograph 跟踪
tf.summary.trace_on(graph = True, profiler = True)

# 执行 autograph
demo = DemoModule(init_value = tf.constant(0.0))
result = demo.add_print(tf.constant(5.0))

# 将计算图信息写入日志
with writer.as_default():
    tf.summary.trace_export(
        name = "demomodule",
        step = 0,
        profiler_outdir = logdir,
    )

# 启动 tensorboard 在 jupyter 中的魔法命令
%reload_ext tensorboard
from tensorboard import notebook
notebook.start("--logdir ./data/demomodule/")
```

* 通过给 `tf.Module` 添加属性的方法进行封装

```python
my_module = tf.Module()
my_module.x = tf.Variable(0.0)

@tf.function(input_signature = [tf.TensorSpec(shape = [], dtype = tf.float32)])
def add_print(a):
    my_module.x.assign_add(a)
    tf.print(my_module.x)
    return (my_module.x)

my_module.add_print = add_print
my_module.add_print(tf.constant(1.0)).numpy()
print(my_module.variables)

# 使用 tf.saved_model 保存模型
tf.saved_model(
    my_module,
    "./data/my_module",
    signatures = {
        "serving_default": my_module.add_print
    }
)

# 加载模型
my_module2 = tf.saved_model.load("./data/my_module")
my_module2.add_print(tf.constant(5.0))
```

```
1.0
(<tf.Variable 'Variable:0' shape=() dtype=float32, numpy=0.0>,)

INFO:tensorflow:Assets written to: ./data/mymodule/assets
5
```

#### tf.Module 和 tf.keras.Model, tf.keras.layers.Layer

`tf.keras` 中的模型和层都是继承 `tf.Module` 实现的，也具有变量管理和子模块管理功能

```python
import tensorflow as tf
from tensorflow.keras import models, layers, losses, metrics

print(issubclass(tf.keras.Model, tf.Module))
print(issubclass(tf.keras.layers.Layer, tf.Module))
print(issubclass(tf.keras.Model, tf.keras.layers.Layer))
```

```
True
True
True
```

```python
tf.keras.backend.clear_session()

model = models.Sequential()
model.add(layers.Dense(4, input_shape = (10,)))
model.add(layers.Dense(2))
model.add(layers.Dense(1))

model.summary()
model.variables
model.layers[0].trainable = False  # 冻结第 0 层的边变量，使其不可训练
model.trainable_variable
model.submodules
model.layers
model.name
model.name_scope()
```

