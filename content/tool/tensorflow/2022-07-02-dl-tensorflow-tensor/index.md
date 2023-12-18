---
title: TensorFlow 张量
author: 王哲峰
date: '2022-07-02'
slug: dl-tensorflow-tensor
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
    - [tf.broadcast\_to 以显式的方式按照广播机制扩展张量的维度](#tfbroadcast_to-以显式的方式按照广播机制扩展张量的维度)
    - [计算广播后计算结果的形状，静态形状，TensorShape 类型参数](#计算广播后计算结果的形状静态形状tensorshape-类型参数)
    - [计算广播后计算结果的形状，动态形状，Tensor 类型参数](#计算广播后计算结果的形状动态形状tensor-类型参数)
    - [广播效果](#广播效果)
</p></details><p></p>

> 张量和计算图是 TensorFlow 的核心概念

TensorFlow 的基本数据结构是张量。张量即多维数组，TensorFlow 的张量和 Numpy 中的 array 很类似。
从行为特性来开，有两种类型的张量:
    
* 常量 Constant
    - 常量的值在计算图中不可以被重新赋值
* 变量 Variable
    - 变量可以在计算图中用 `assign` 等算子重新赋值

# 张量数据结构

## 常量张量

### 数据类型

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

### 数据维度

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

### 数据转换

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

## 变量张量

模型中需要被训练的参数一般被设置成变量

### 常量值不可改变

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

### 变量值可以改变

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

# 张量数据操作

## 创建张量

张量创建的许多方法和 Numpy 中创建 array 的方法很像

### 张量创建

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

### 均匀分布

* tf.random.set_seed
* tf.random.uniform

```python
tf.random.set_seed(1.0)
a = tf.random.uniform([5], minval = 0, maxval = 10)
tf.print(a)
```

### 正态分布

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

### 特殊矩阵

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

## 索引切片

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

## 维度变换

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




## 合并分割

* 可以用 `tf.concat` 和 `tf.stack` 方法对多个张量进行合并
* 可以用 `tf.split` 方法把一个张量分割成多个张量

### 合并

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

### 分割

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

# 张量数学运算

张量的数学运算符可以分为标量运算符、向量运算符、矩阵运算符

## 标量运算

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

## 向量运算

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

## 矩阵运算

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

## 广播机制

TensorFlow 的广播规则和 Numpy 是一样的:

1. 如果张量的维度不同，将维度较小的张量进行扩展，直到两个张量的维度都一样
2. 如果两个张量在某个维度上的长度是相同的，或者其中一个张量在该维度上的长度为 1，
   那么我们就说这两个张量在该维度上是相容的
4. 如果两个张量在所有维度上都是相容的，它们就能使用广播
5. 广播之后，每个维度的长度将取两个张量在该维度长度的较大值
6. 在任何一个维度上，如果一个张量的长度为 1，另一个张量长度大于 1，那么在该维度上，
   就好像是对第一个张量进行了复制

### tf.broadcast_to 以显式的方式按照广播机制扩展张量的维度

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

### 计算广播后计算结果的形状，静态形状，TensorShape 类型参数

```python
tf.broadcast_static_shape(a.shape, b.shape)
```

### 计算广播后计算结果的形状，动态形状，Tensor 类型参数

```python
c = tf.constant([1, 2, 3])
d = tf.constant([[1], [2], [3]])
tf.broadcast_dynamic_shape(tf.shape(c), tf.shape(d)) 
```

### 广播效果

```python
c + d
# 等价于
tf.broadcast_to(c, [3, 3]) + tf.broadcast_to(d, [3, 3])
```

