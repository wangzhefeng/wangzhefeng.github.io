---
title: PyTorch 张量
author: 王哲峰
date: '2022-07-16'
slug: dl-pytorch-api-tensor
categories:
  - pytorch
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

- [tensor 介绍](#tensor-介绍)
  - [tensor](#tensor)
  - [tensor 与 variable](#tensor-与-variable)
- [tensor 数据类型](#tensor-数据类型)
  - [基本数据类型](#基本数据类型)
  - [自动推断数据类型](#自动推断数据类型)
  - [指定数据类型](#指定数据类型)
  - [使用特定类型构造函数](#使用特定类型构造函数)
  - [不同类型转换](#不同类型转换)
- [tensor 维度](#tensor-维度)
  - [标量](#标量)
  - [向量](#向量)
  - [矩阵](#矩阵)
  - [三维张量](#三维张量)
  - [四维张量](#四维张量)
- [tensor 尺寸](#tensor-尺寸)
  - [标量](#标量-1)
  - [向量](#向量-1)
  - [矩阵](#矩阵-1)
  - [三维张量](#三维张量-1)
  - [四维张量](#四维张量-1)
- [tensor 设备](#tensor-设备)
- [tensor 创建](#tensor-创建)
  - [直接创建](#直接创建)
    - [tensor](#tensor-1)
    - [tensor 和 array](#tensor-和-array)
  - [依数值创建](#依数值创建)
  - [依概率分布创建](#依概率分布创建)
    - [设置随机数种子](#设置随机数种子)
    - [生成随机数 tensor](#生成随机数-tensor)
- [tensor 结构操作](#tensor-结构操作)
  - [拼接和分割](#拼接和分割)
    - [cat](#cat)
    - [stack](#stack)
    - [split](#split)
  - [索引和切片](#索引和切片)
    - [选取行、列](#选取行列)
    - [不规则切片](#不规则切片)
  - [维度变换](#维度变换)
    - [view 和 reshape](#view-和-reshape)
    - [squeeze 和 unsqueeze](#squeeze-和-unsqueeze)
    - [transpose 和 permute](#transpose-和-permute)
    - [t](#t)
- [tensor 数学运算](#tensor-数学运算)
  - [In-place 操作](#in-place-操作)
  - [标量运算](#标量运算)
  - [向量运算](#向量运算)
    - [统计值](#统计值)
    - [CUM 扫描](#cum-扫描)
    - [排序](#排序)
  - [矩阵运算](#矩阵运算)
    - [矩阵加法](#矩阵加法)
    - [矩阵乘法](#矩阵乘法)
    - [矩阵转置](#矩阵转置)
    - [矩阵逆](#矩阵逆)
    - [矩阵求迹](#矩阵求迹)
    - [矩阵范数](#矩阵范数)
    - [矩阵行列式](#矩阵行列式)
    - [矩阵特征值和特征向量](#矩阵特征值和特征向量)
    - [矩阵分解](#矩阵分解)
  - [爱因斯坦求和函数](#爱因斯坦求和函数)
    - [规则](#规则)
    - [einsum 基础范例](#einsum-基础范例)
    - [einsum 高级范例](#einsum-高级范例)
- [torch 广播机制](#torch-广播机制)
- [参考](#参考)
</p></details><p></p>

# tensor 介绍

tensor 是 PyTorch 中最基本的概念，其参与了整个运算过程，
这里主要介绍 tensor 的概念和属性，如 data，variable，device 等，
并且介绍 tensor 的基本创建方法，如直接创建、依数值创建、依概率分布创建等

## tensor

tensor 其实是多维数组，它是标量、向量、矩阵的高维拓展

## tensor 与 variable

variable(已经没用了) 是 `torch.autograd` 中的数据类型，
在 PyTorch 0.4.0 版本之后 variable 已经并入 tensor，但是 variable 这个数据类型对于理解 tensor 来说很有帮助

variable(`torch.autograd.variable`) 有 5 个属性，这些属性都是为了 tensor 的自动求导而设置的：

* `data`
* `grad`
* `grad_fn`
* `requires_grad`
* `is_leaf`

tensor(`torch.tensor`) 有 8 个属性:

* 与数据本身相关
    - `data`
        - 被包装的 tensor
    - `dtype`
        - tensor 的数据类型
    - `dim()`
        - tensofr 的维度
    - `shape`、`size()`
        - tensor 的形状
    - `device`
        - tensor 所在的设备，gpu(`cuda`)/cpu(`cpu`)，tensor 放在 gpu 上才能使用加速
* 与梯度求导相关
    - `grad`
        - `data` 的梯度
    - `grad_fn`
        - `fn` 表示 function 的意思，记录创建 tensor 时用到的方法
    - `requires_grad`
        - 是否需要梯度 
    - `is_leaf`
        - 是否是叶子节点(tensor)

# tensor 数据类型

## 基本数据类型

tensor 的数据类型和 numpy `array` 基本一一对应，但是不支持 `str` 类型，包括：

* `torch.uint8`
* `torch.int8`
* `torch.int16`
* `torch.int32`(`torch.int`)
* `torch.int64`(`torch.long`)
* `torch.float16`
* `torch.float32`(`torch.float`)
* `torch.float64`(`torch.double`)
* `torch.bool`

一般神经网络建模使用的都是 `torch.float32`

## 自动推断数据类型

```python
i = torch.tensor(1)
print(i, i.dtype)  # tensor(1) torch.int64

x = torch.tensor(2.0)
print(x, x.dtype)  # tensor(2.) torch.float32

b = torch.tensor(True)
print(b, b.dtype)  # tensor(True) torch.bool
```

## 指定数据类型

* `dtype` 参数

```python
i = torch.tensor(1, dtype = torch.int32)
print(i, i.dtype)  # tensor(1, dtype=torch.int32) torch.int32

x = torch.tensor(2.0, dtype = torch.double)
print(x, x.dtype)  # tensor(2., dtype=torch.float64) torch.float64
```

## 使用特定类型构造函数

* `torch.IntTensor()`
* `torch.Tensor(sequence)`
* `torch.BoolTensor(sequence)`

```python
i = torch.IntTensor(1)
print(i, i.dtype)  # tensor([0], dtype=torch.int32) torch.int32

x = torch.Tensor(np.array(2.0))
print(x, x.dtype)  # tensor(2.) torch.float32

b = torch.BoolTensor(np.array([1, 0, 2, 0]))
print(b, b.dtype)  # tensor([ True, False,  True, False]) torch.bool
```

## 不同类型转换

* `.float()`
* `.type(type)`
* `.type_as(tensor)`

```python
i = torch.tensor(1)
print(i, i.dtype)  # tensor(1) torch.int64

x = i.float()
print(x, x.dtype)  # tensor(1.) torch.float32

y = i.type(torch.float)
print(y, y.dtype)  # tensor(1.) torch.float32

z = i.type_as(x)
print(z, z.dtype)  # tensor(1.) torch.float32
```

# tensor 维度

不同类型的数据可以用不同维度(dimension)的张量来表示，有几层中括号，就是多少维的张量

* 标量为 0 维张量
* 向量为 1 维张量
* 矩阵为 2 维张量
* 彩色图像有 RGB 3 个通道，可以表示为 3 维张量
* 视频还有时间维，可以表示 4 维张量

## 标量

```python
scalar = torch.tensor(True)
print(scalar.dim())  # 0
```

## 向量

```python
vector = torch.tensor([1.0, 2.0, 3.0, 4.0])
print(vector.dim())  # 1
```

## 矩阵

```python
matrix = torch.tensor(
    [[1.0, 2.0],
     [3.0, 4.0]]
)
print(matrix.dim())  # 2
```

## 三维张量

```python
tensor3 = torch.tensor(
    [[[1.0, 2.0],
      [3.0, 4.0]],
     [[5.0, 6.0],
      [7.0, 8.0]]]
)
print(tensor3.dim())  # 3
```

## 四维张量

```python
tensor4 = torch.tensor(
    [[[[1.0,1.0],
       [2.0,2.0]],
      [[3.0,3.0],
       [4.0,4.0]]],
     [[[5.0,5.0],
       [6.0,6.0]],
      [[7.0,7.0],
       [8.0,8.0]]]]
)
print(tensor4.dim())  # 4
```

# tensor 尺寸

* 可以使用 `shape` 属性或者 `size()` 方法查看张量在每一维的长度
* 可以使用 `view()` 方法改变张量的尺寸，如果 `view()` 方法改变尺寸失败，可以使用 `reshape()` 方法

## 标量

```python
scalar = torch.tensor(True)
print(scalar.size())  # torch.Size([])
print(scalar.shape)  # torch.Size([])
```

## 向量

```python
vector = torch.tensor([1.0, 2.0, 3.0, 4.0])
print(vector.size())  # torch.Size([4])
print(vector.shape)  # torch.Size([4])
```

## 矩阵

```python
matrix = torch.tensor(
    [[1.0, 2.0],
     [3.0, 4.0]]
)
print(matrix.size())  # torch.Size([2, 2])
print(matrix.shape)  # torch.Size([2, 2])
```

## 三维张量

```python
tensor3 = torch.tensor(
    [[[1.0, 2.0],
      [3.0, 4.0]],
     [[5.0, 6.0],
      [7.0, 8.0]]]
)
print(tensor3.size())  # torch.Size([2, 2, 2])
print(tensor3.shape)  # torch.Size([2, 2, 2])
```

## 四维张量

```python
tensor4 = torch.tensor(
    [[[[1.0,1.0],
       [2.0,2.0]],
      [[3.0,3.0],
       [4.0,4.0]]],
     [[[5.0,5.0],
       [6.0,6.0]],
      [[7.0,7.0],
       [8.0,8.0]]]]
)
print(tensor4.size())  # torch.Size([2, 2, 2, 2])
print(tensor4.shape)  # torch.Size([2, 2, 2, 2])
```

# tensor 设备

如果 CUDA 可用，可以使用 `torch.device` 对象将 tensors 移出或放入 GPU

```python
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# tensor
x = torch.tensor([1])
# directly create a tensor on gpu
y = torch.ones_like(x, device = device)
# or just use strings `.to("cuda")`
x = x.to(device)
# `.to` can also change dtype together
z = x + y
z.to("cpu", torch.double)
```

# tensor 创建

* 直接创建
    - `torch.tensor()`
    - `torch.from_numpy()`
* 依数值创建
    - `torch.empty()`
    - `torch.ones()`
    - `torch.zeros()`
    - `torch.eye()`：单位矩阵
    - `torch.diag()`：对角矩阵
    - `torch.fill_()`
    - `torch.arange()`
    - `torch.linspace()`
* 依概率分布创建
    - `torch.manual_seed()`
    - `torch.normal()`：标准正态
    - `torch.randn()`：正态分布
    - `torch.rand()`：均匀分布
    - `torch.randint()`
    - `torch.randperm()`：整数随机排列

## 直接创建

### tensor

* API

```python
torch.tensor(
    data,  # scale, list, numpy
    dtype = none,
    device = none,
    requires_grad = false,
    pin_memory = false,  # 是否存于锁页内存
)
```

* 示例

```python
arr = np.ones((3, 3))

t = torch.tensor(arr, device = "cuda")
print(t)
```

### tensor 和 array

可以用 `tensor.numpy()` 方法从 tensor 得到 numpy array，也可以用 `torch.from_numpy(array)` 从 numpy array 得到 tensor。
这两种方法关联的 tensor 与 numpy array 是共享内存的，当修改其中一个数据的时候，另一个的值也会被改动

1. numpy array To torch tensor

```python
import numpy as np
import torch

# numpy array
a = np.ones(5)
# torch tensor
b = torch.from_numpy(a)
print(a)  # array([1., 1., 1., 1., 1.])
print(a)  # tensor([1., 1., 1., 1., 1.], dtype=torch.float64)

np.add(a, 1, out = a)  # array([2., 2., 2., 2., 2.])
print(a)  # array([2., 2., 2., 2., 2.])
print(b)  # array([2., 2., 2., 2., 2.])
```

2. torch tensor To numpy array

```python
import torch

# torch tensor
a = torch.ones(5)
# numpy array
b = a.numpy()
print(a)  # tensor([1., 1., 1., 1., 1.])
print(b)  # array([1., 1., 1., 1., 1.], dtype=float32)

a.add_(1)  # tensor([2., 2., 2., 2., 2.])
print(a)  # tensor([2., 2., 2., 2., 2.])
print(b)  # array([2., 2., 2., 2., 2.], dtype=float32)
```

3. 如果有需要，可以用张量的 `.clone()` 方法拷贝 tensor，中断这种关联

```python
import torch

# torch tensor
tensor = torch.zeros(3)
# numpy array
# 使用 clone 方法拷贝张量, 拷贝后的张量和原始张量内存独立
arr = tensor.clone().numpy()  
# 也可以使用 tensor.data.numpy()

print("before add 1:")
print(tensor)  # tensor([0., 0., 0.])
print(arr)  # array([0., 0., 0.], dtype=float32)

print("\nafter add 1:") # 使用带下划线的方法表示计算结果会返回给调用张量
tensor.add_(1)  # tensor([1., 1., 1.])
print(tensor)  # tensor([1., 1., 1.])
print(arr)  # array([0., 0., 0.], dtype=float32)
```

4. 可以使用 `.item()` 方法从标量张量得到对应的 Python 数值

```python
import torch

# tensor
scalar = torch.tensor(1.0)
# python scalar
s = scalar.item()

print(s)  # 1.0
print(type(s))  # <class 'float'>
```

5. 使用 `tolist()` 方法从 tensor 得到对应的 Python 数值列表

```python
import torch

# tensor
tensor = torch.rand(2, 2)
# python list
t = tensor.tolist()

print(t)  # [[0.3403051495552063, 0.6483253240585327], [0.243993878364563, 0.7659801244735718]]
print(type(t))  # <class 'list'>
```

## 依数值创建

* `torch.empty()`：空矩阵
* `torch.ones()`
* `torch.zeros()`
* `torch.eye()`：单位矩阵
* `torch.diag()`：对角矩阵
* `torch.fill_()`
* `torch.arange()`
* `torch.linspace()`

## 依概率分布创建

### 设置随机数种子

* `torch.seed()`
    - 设置生成随机数的种子为非确定性随机数
* `torch.manual_seed()`
    - 设置生成随机数的种子
* `torch.initial_seed()`
    - 返回用于生成随机数的初始种子

### 生成随机数 tensor

* 从伯努利分布中提取二进制随机数(0 或 1)
    - `torch.bernoulli()`
* 正态分布
    - `torch.normal()`：正态分布
    - `torch.randn()`：标准正态分布
    - `torch.randn_like()`：标准正态分布
    - `torch.multinomial()`：多元正态分布
* 均匀分布
    - `torch.rand()`：`$[0, 1)$` 区间的均匀分布
    - `torch.rand_like()`：`$[0, 1)$` 区间的均匀分布
    - `torch.randint()`：从给定区间的均匀分布中取整数
    - `torch.randint_like()`：从给定区间的均匀分布中取整数
* 泊松分布
    - `torch.pisson()`
* 整数随机排列
    - `torch.randperm()`：

# tensor 结构操作

* tensor 的拼接
    - `torch.cat()`
    - `torch.stack()`
    - `torch.hstack()`
    - `torch.vstack()`
* tensor 的分割
    - `torch.split()` 
    - `torch.chunk()`
* tensor 的索引和切片
    - `[]`、`[:]`、`[...]`...
    - 不规则切片
        - `torch.index_select()`：不规则切片
        - `torch.masked_select()`：不规则切片
        - `torch.take()`：不规则切片
    - `torch.gather()`
    - 如果要通过修改张量的部分元素值得到新的张量
        - `torch.where()`：可以理解为 if 的张量版本
        - `torch.index_fill()`：选取元素逻辑和 `torch.index_select` 相同 
        - `torch.masked_fill()`：选取元素逻辑和 `torch.masked_select` 相同
* tensor 的变换
    - `torch.view()`：改变张量的形状
    - `torch.reshape()`：改变张量的形状
    - `torch.squeeze()`：减少维度
    - `torch.unsqueeze()`：增加维度
    - `torch.transpose()`：交换维度
    - `torch.permute()`：交换维度
    - `torch.t`

## 拼接和分割

可以用 `torch.cat` 方法和 `torch.stack` 方法将多个张量合并，
可以用 `torch.split` 方法把一个张量分割成多个张量

* `torch.cat` 和 `torch.stack` 有略微的区别，`torch.cat` 是连接，
  不会增加维度，而 `torch.stack` 是堆叠，会增加维度
* `torch.split` 是 `torch.cat` 的逆运算，可以指定分割份数平均分割，
  也可以通过指定每份的记录数量进行分割

### cat

```python
a = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
b = torch.tensor([[5.0, 6.0], [7.0, 8.0]])
c = torch.tensor([[9.0, 10.0], [11.0, 12.0]])

abc_cat = torch.cat([a, b, c], dim = 0)
print(abc_cat.shape)
print(abc_cat)
```

```
torch.Size([6, 2])
tensor([[ 1.,  2.],
        [ 3.,  4.],
        [ 5.,  6.],
        [ 7.,  8.],
        [ 9., 10.],
        [11., 12.]])
```


```python
abc_cat = torch.cat([a, b, c], dim = 1)
print(abc_cat.shape)
print(abc_cat)
```

```
torch.Size([2, 6])
tensor([[ 1.,  2.,  5.,  6.,  9., 10.],
        [ 3.,  4.,  7.,  8., 11., 12.]])
```

### stack

```python
abc_stack = torch.stack([a, b, c], axis = 0)  # torch 中的 dim 和 axis 参数名可以混用
print(abc_stack.shape)
print(abc_stack)
```

```
torch.Size([3, 2, 2])
tensor([[[ 1.,  2.],
         [ 3.,  4.]],

        [[ 5.,  6.],
         [ 7.,  8.]],

        [[ 9., 10.],
         [11., 12.]]])
```

```python
abc_stack = torch.stack([a, b, c], axis = 1)  # torch 中的 dim 和 axis 参数名可以混用
print(abc_stack.shape)
print(abc_stack)
```

```
torch.Size([2, 3, 2])
tensor([[[ 1.,  2.],
         [ 5.,  6.],
         [ 9., 10.]],

        [[ 3.,  4.],
         [ 7.,  8.],
         [11., 12.]]])
```

### split

```python
print(abc_cat)
a, b, c = torch.split(
    abc_cat, 
    split_size_or_sections = 2, 
    dim = 0
)  # 每份 2 个进行分割
print(a)
print(b)
print(c)
```

```
tensor([[ 1.,  2.],
        [ 3.,  4.],
        [ 5.,  6.],
        [ 7.,  8.],
        [ 9., 10.],
        [11., 12.]])

tensor([[1., 2.],
        [3., 4.]])

tensor([[5., 6.],
        [7., 8.]])

tensor([[ 9., 10.],
        [11., 12.]])
```

```python
print(abc_cat)
p, q, r = torch.split(
    abc_cat, 
    split_size_or_sections = [4, 1, 1], 
    dim = 0
)  # 每份分别为 [4, 1, 1]
print(p)
print(q)
print(r)
```

```
tensor([[ 1.,  2.],
        [ 3.,  4.],
        [ 5.,  6.],
        [ 7.,  8.],
        [ 9., 10.],
        [11., 12.]])

tensor([[1., 2.],
        [3., 4.],
        [5., 6.],
        [7., 8.]])

tensor([[ 9., 10.]])

tensor([[11., 12.]])
```

## 索引和切片

### 选取行、列

```python
torch.manual_seed(0)
minval, maxval = 0, 10
t = torch.floor(minval + (maxval - minval) * torch.rand([5, 5])).int()
print(t)

a = torch.arange(27).view(3, 3, 3)
print(a)
```

```
tensor([[4, 7, 0, 1, 3],
        [6, 4, 8, 4, 6],
        [3, 4, 0, 1, 2],
        [5, 6, 8, 1, 2],
        [6, 9, 3, 8, 4]], dtype=torch.int32)

tensor([[[ 0,  1,  2],
         [ 3,  4,  5],
         [ 6,  7,  8]],

        [[ 9, 10, 11],
         [12, 13, 14],
         [15, 16, 17]],

        [[18, 19, 20],
         [21, 22, 23],
         [24, 25, 26]]])
```

```python
t[0]  # 第 0 行
t[-1]  # 倒数第一行
t[1, 3]  # 第 1 行第 3 列
t[1][3]  # 第 1 行第 3 列
t[1:4, :]  # 第 1 行至第 3 行
t[1:4, 4:2]  # 第 1 行至最后一行，第 0 列到最后一列每隔两列取一列
a[..., 1]  # 省略号可以表示多个冒号
```

### 不规则切片

```python
# 考虑班级成绩册的例子，有4个班级，每个班级5个学生，
# 每个学生7门科目成绩。可以用一个4×5×7的张量来表示
minval = 0
maxval = 100
scores = torch.floor(minval + (maxval - minval) * torch.rand([4, 5, 7])).int()
print(scores)
```

```
tensor([[[55, 95,  3, 18, 37, 30, 93],
         [17, 26, 15,  3, 20, 92, 72],
         [74, 52, 24, 58,  3, 13, 24],
         [81, 79, 27, 48, 81, 99, 69],
         [56, 83, 20, 59, 11, 15, 24]],

        [[72, 70, 20, 65, 77, 43, 51],
         [61, 81, 98, 11, 31, 69, 91],
         [93, 94, 59,  6, 54, 18,  3],
         [94, 88,  0, 59, 41, 41, 27],
         [69, 20, 68, 75, 85, 68,  0]],

        [[17, 74, 60, 10, 21, 97, 83],
         [28, 37,  2, 49, 12, 11, 47],
         [57, 29, 79, 19, 95, 84,  7],
         [37, 52, 57, 61, 69, 52, 25],
         [73,  2, 20, 37, 25, 32,  9]],

        [[39, 60, 17, 47, 85, 44, 51],
         [45, 60, 81, 97, 81, 97, 46],
         [ 5, 26, 84, 49, 25, 11,  3],
         [ 7, 39, 77, 77,  1, 81, 10],
         [39, 29, 40, 40,  5,  6, 42]]], dtype=torch.int32)
```

```python
# 抽取每个班级第0个学生，第2个学生，第4个学生的全部成绩
torch.index_select(socres, dim = 1, index = torch.tensor([0, 2, 4]))

# 抽取每个班级第0个学生，第2个学生，第4个学生的第1门课程，第3门课程，第6门课程成绩
q = torch.index_select(
    torch.index_select(scores, dim = 1, index = torch.tensor([0, 2, 4])), 
    dim = 2, 
    index = torch.tensor([1, 3, 6])
)

# 抽取第0个班级第0个学生的第0门课程，第2个班级的第3个学生的第1门课程，第3个班级的第4个学生第6门课程成绩
# take将输入看成一维数组，输出和index同形状
s = torch.take(
    socres, 
    torch.tensor([
        0 * 5 * 7 + 0, 
        2 * 5 * 7 + 3 * 7 + 1,
        3 * 5 * 7 + 4 * 7 + 6
    ])
)

# 抽取分数大于等于 80 分的分数(布尔索引)，结果是 1 维张量
g = torch.masked_select(scores, scores >= 80)
print(g)
```

* 如果要通过修改张量的部分元素值得到新的张量

```python
# 如果分数大于 60 分，赋值成 1，否则赋值成 0
ifpass = torch.where(scores > 60, torch.tensor(1), torch.tensor(0))

# 将每个班级第0个学生，第2个学生，第4个学生的全部成绩赋值成满分
torch.index_fill(scores, dim = 1, index = torch.tensor(0, 2, 4), value = 100)
scores.index_fill(dim = 1, index = torch.tensor([0, 2, 4]), value = 100)

# 将分数小于60分的分数赋值成60分
b = torch.masked_fill(scores, scores < 60, 60)
b = scores.masked_fill(scores < 60, 60)
```

## 维度变换

### view 和 reshape

使用 `view` 可以改变张量尺寸:

```python
vector = torch.arange(0,12)
print(vector)
print(vector.shape)

matrix34 = vector.view(3, 4)
print(matrix34)
print(matrix34.shape)

# -1 表示该位置长度由程序自动推断
matrix43 = vector.view(4, -1)
print(matrix43)
print(matrix43.shape)
```

```
tensor([ 0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11])
torch.Size([12])

tensor([[ 0,  1,  2,  3],
        [ 4,  5,  6,  7],
        [ 8,  9, 10, 11]])
torch.Size([3, 4])

tensor([[ 0,  1,  2],
        [ 3,  4,  5],
        [ 6,  7,  8],
        [ 9, 10, 11]])
torch.Size([4, 3])
```

有些操作会让张量存储结构扭曲，直接使用 `view` 会失败，可以用 `reshape` 方法:

```python
matrix26 = torch.arange(0, 12).view(2, 6)
print(matrix26)
print(matrix26.shape)
```

```
tensor([[ 0,  1,  2,  3,  4,  5],
        [ 6,  7,  8,  9, 10, 11]])
torch.Size([2, 6])
```

* 转置操作让张量存储结构扭曲

```python
matrix62 = matrix26.t()
print(matrix62.is_contiguous())
```

```
False
```

* 直接使用 `view` 方法会失败，可以使用 `reshape` 方法

```python
matrix34 = matrix62.view(3, 4)  # error!
matrix34 = matrix62.reshape(3, 4)
print(matrix34)

# 等价于
matrix34 = matrix62.contiguous().view(3, 4)
print(matrix34)
```

```
tensor([[ 0,  6,  1,  7],
        [ 2,  8,  3,  9],
        [ 4, 10,  5, 11]])

tensor([[ 0,  6,  1,  7],
        [ 2,  8,  3,  9],
        [ 4, 10,  5, 11]])
```

### squeeze 和 unsqueeze

如果张量在某个维度上只有一个元素，利用 `torch.squeeze` 可以消除这个维度

```python
a = torch.tensor([[1.0, 2.0]])
s = torch.squeeze(a)
print(a)
print(s)
print(a.shape)
print(s.shape)
```

```
tensor([[1., 2.]])
tensor([1., 2.])
torch.Size([1, 2])
torch.Size([2])
```

`torch.unsqueeze` 的作用和 `torch.squeeze` 的作用相反，
`torch.unsqueeze` 在指定维插入长度为 1 的维度

```python
d = torch.unsqueeze(s, axis = 0)
print(s)
print(d)
print(s.shape)
print(d.shape)
```

```
tensor([1., 2.])
tensor([[1., 2.]])
torch.Size([2])
torch.Size([1, 2])
```

### transpose 和 permute

`torch.transpose` 可以交换张量的维度，常用于图片存储格式的变换上。
如果是二维的矩阵，通常会调用矩阵的转置方法 `matrix.t()`，
等价于 `torch.transpose(matrix, 0, 1)`

```python
minval = 0
maxval = 255
# (batch, height, width, channel)
data = torch.floor(minval + (maxval - minval) * torch.rand([100, 256, 256, 4])).int()
print(data.shape)

# 转换成 PyTorch 默认的图片格式 (batch, channel, height, width)
data_t = torch.transpose(torch.transpose(data, 1, 2), 1, 3)
print(data.shape)

# 转换成 PyTorch 默认的图片格式 (batch, channel, height, width)
data_p = torch.permute(data, [0, 3, 1, 2])
print(data_p.shape)
```

### t

```python
matrix = torch.tensor([[1, 2, 3], [4, 5, 6]])
print(matrix)
print(matrix.t())
# 等价于
print(torch.transpose(matrix, 0, 1))
```

# tensor 数学运算

tensor 数学运算主要有:

* 标量运算
* 向量运算
* 矩阵运算
* 任意阶张量运算
    - 爱因斯坦求和函数 `torch.einsum()` 
* 广播机制

## In-place 操作

带有 `_` 后缀的操作叫做 in-place 操作，也就是就地计算的，
比如下面的操作会直接改变 `$x$`：

* `x.add_(y)`
* `x.copy_(y)`
* `x.t_()`
* ...

## 标量运算

标量运算的特点是对张量实施逐元素运算，操作的张量至少是 0 维，
有些标量运算符对常用的数学运算符进行了重载。
并且支持类似 numpy 的广播特性。以下是常见的标量运算符：

* 加
* 减
* 乘
* 除
* 求模
* 指数
* 对数
* 三角函数
* 逻辑比较运算符

```python
import numpy as np
import torch

a = torch.tensor([1.0, 8.0])
b = torch.tensor([5.0, 6.0])
c = torch.tensor([6.0, 7.0])
x = torch.tensor([2.6, -2.7])
y = torch.tensor([0.9, -0.8, 100.0, -20.0, 0.7])

a + b
a - b
a * b
a / b
torch.div(a, b, rounding_mode = "floor")  # 地板除法
a % 3  # 求模
a ** 2  # 乘方
torch.sqrt(a)
a >= 2  # torch.ge(a, 2)
(a >= 2) & (a <= 3)
(a >= 2) | (a <= 3)
a == 5  # torch.eq(a, 5)
torch.max(a, b)
torch.min(a, b)
torch.round(x)  # 保留整数部分，四舍五入
torch.floor(x)  # 保留整数部分，向下归整
torch.ceil(x)  # 保留整数部分，向上归整
torch.trunc(x)  # 保留整数部分，向 0 归整
torch.fmod(x, 2)  # 作除法取余数
torch.remainder(x, 2)  # 作除法取剩余的部分，结果恒正
torch.clamp(y, min = -1, max = 1)  # 幅值裁剪
torch.clamp(y, max = 1)  # 幅值裁剪
relu = lambda x: x.clamp(min = 0.0)  # 幅值裁剪
relu(torch.tensor(5.0))
```

## 向量运算

向量运算符只在一个特定轴上运算，将一个向量映射到一个标量或者另外一个向量

### 统计值

```python
a = torch.arange(1, 10).float().view(3, 3)

torch.sum(a)
torch.mean(a)
torch.max(a)
torch.min(a)
torch.prod(a)  # 累计乘积
torch.std(a)
torch.var(a)
torch.median(a)
torch.max(a, dim = 0)
torch.max(a, dim = 1)
```

### CUM 扫描

```python
a = torch.arange(1, 10)

torch.cumsum(a, 0)
torch.cumprod(a, 0)
torch.cummax(a, 0).values
torch.cummax(a, 0).indices
torch.cummin(a, 0).values
torch.cummin(a, 0).indices
```

### 排序

```python
a = torch.tensor([[9, 7, 8], [1, 3, 2], [5, 6, 4]]).float()

torch.topk(a, 2, dim = 0)
torch.topk(a, 2, dim = 1)
torch.sort(a, dim = 1)
```

## 矩阵运算

矩阵运算包括：

* 矩阵加法
* 矩阵乘法
* 矩阵逆
* 矩阵求迹
* 矩阵范数
* 矩阵行列式
* 矩阵求特征值
* 矩阵分解

### 矩阵加法

```python
import torch

x = torch.zeros(5, 3, dtype = torch.long)
y = torch.rand(5, 3)

# method 1
print(x + y)

# method 2
print(torch.add(x, y))

# method 3
result = torch.empty(5, 3)
torch.add(x, y, out = result)
print(result)

# method 4
y.add_(x)
print(y)
```

### 矩阵乘法

* 逐元素相乘

```python
a = torch.tensor([[1, 2], [3, 4]])
b = torch.tensor([[2, 0], [0, 2]])

a.mul(b)
a * b
```

* 点积

```python
a = torch.tensor([[1, 2], [3, 4]])
b = torch.tensor([[2, 0], [0, 2]])

a @ b
torch.matmul(a, b)
torch.mm(a, b)
```

* 高维张量的矩阵乘法在后面的维度上进行

```python
a = torch.randn(5, 5, 6)
b = torch.randn(5, 6, 4)
(a @ b).shape
```

### 矩阵转置

```python
a = torch.tensor([[1.0, 2], [3, 4]])
a.t()
```

### 矩阵逆

```python
a = torch.tensor([[1.0, 2], [3, 4]])
torch.inverse(a)
```

### 矩阵求迹

```python
a = torch.tensor([[1.0, 2], [3, 4]])
torch.trace(a)
```

### 矩阵范数

```python
a = torch.tensor([[1.0, 2], [3, 4]])
torch.norm(a)
```

### 矩阵行列式

```python
a = torch.tensor([[1.0, 2], [3, 4]])
torch.det(a)
```

### 矩阵特征值和特征向量

```python
a = torch.tensor([[1.0, 2], [-5, 4]], dtype = torch.float)
torch.linalg.eig(a)
```

### 矩阵分解

* QR 分解
    - 矩阵 QR 分解，是将一个方阵分解为一个正交矩阵 `$q$` 和上三角矩阵 `$r$`，
      QR 分解实际上是对矩阵 `$a$` 实施 Schmidt 正交化得到 `$q$`

```python
a = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
q, r = torch.linalg.qr(a)
print(q)
print(r)
print(q @ r)
```

* SVD 分解
    - SVD 分解可以将任意一个矩阵分解为一个正交矩阵 `$u$`、一个对角阵 `$s$` 和一个正交矩阵 `$v.t()$` 的乘积
    - SVD 常用于矩阵压缩和降维

```python
import torch
import torch.nn.functional as F

a = torch.tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
u, s, v = torch.linalg.svd(a)
print(u)
print(s)
print(v)

print(u @ F.pad(torch.diag(s), (0, 0, 0, 1)) @ v.t())
```

## 爱因斯坦求和函数

如果问 PyTorch 中最强大的一个数学函数是什么？我会说是 `torch.einsum`：爱因斯坦求和函数。
它几乎是一个"万能函数"：能实现超过一万种功能。不仅如此，和其它 PyTorch 中的函数一样，
`torch.einsum` 是支持求导和反向传播的，并且计算效率非常高

`torch.einsum` 提供了一套既简洁又优雅的规则，可实现包括但不限于：内积，外积，矩阵乘法，
转置和张量收缩(tensor contraction)等张量操作，熟练掌握 `torch.einsum` 可以很方便的实现复杂的张量操作，
而且不容易出错。尤其是在一些包括 batch 维度的高阶张量的相关计算中，
若使用普通的矩阵乘法、求和、转置等算子来实现很容易出现维度匹配等问题，若换成 `torch.einsum` 则会特别简单。
套用一句深度学习 paper 标题当中非常时髦的话术，einsum is all you needed！

### 规则

矩阵乘法：

`$$C_{ij} = \underset{k}{\sum}A_{ik}B_{kj}$$`

爱因斯坦求和约定：只出现在公式一边的指标叫做哑指标，针对哑指标求和符号可以省略

`$$C_{ij} = A_{ik}B_{kj}$$`

einsum 函数：

`C = torch.einsum("ik,kj->ij", A, B)`

einsum 函数的规则原理：

1. 用元素计算公式来表达张量运算
2. 只出现在元素计算公式箭头左边的指标叫做哑指标
3. 省略元素计算公式中对哑指标的求和符号

```python
import torch

A = torch.tensor([[1, 2], [3, 4.0]])
B = torch.tensor([[5, 6], [7, 8.0]])

C1 = A @ B
C2 = torch.einsum("ik,kj->ij", [A, B])
```

### einsum 基础范例

* 张量转置

```python
A = torch.randn(3, 4, 5)
# B = torch.permute(A, [0, 2, 1])
B = torch.einsum("ijk->ikj", A)
print(f"before:{A.shape}")
print(f"after:{B.shape}")
```

* 取对角元

```python
A = torch.randn(5, 5)
# B = torch.diagonal(A)
B = torch.einsum("ii->i", A)
print(f"before:{A.shape}")
print(f"after:{B.shape}")
```

* 求和降维

```python
A = torch.randn(4, 5)
# B = torch.sum(A, 1)
B = torch.einsum("ij->i", A)
print(f"before:{A.shape}")
print(f"after:{B.shape}")
```

* 哈达玛积

```python
A = torch.randn(5, 5)
B = torch.randn(5, 5)
# C = A * B
C = torch.einsum("ij,ij->ij", A, B)
print(f"before:{A.shape, B.shape}")
print(f"after:{C.shape}")
```

* 向量内积

```python
A = torch.randn(10)
B = torch.randn(10)
# C = torch.dot(A, B)
C = torch.einsum("i,i->", A, B)
print(f"before:{A.shape, B.shape}")
print(f"after:{C.shape}")
```

* 向量外积(类似笛卡尔积)

```python
A = torch.randn(10)
B = torch.randn(5)
# C = torch.outer(A, B)
C = torch.einsum("i,j->ij", A, B)
print(f"before:{A.shape, B.shape}")
print(f"after:{C.shape}")
```

* 矩阵乘法

```python
A = torch.randn(5, 4)
B = torch.randn(4, 6)
# C = torch.matmul(A, B)
C = torch.einsum("ik,kj->ij", A, B)
print(f"before:{A.shape, B.shape}")
print(f"after:{C.shape}")
```

* 张量缩并

```python
A = torch.randn(3, 4, 5)
B = torch.randn(4, 3, 6)
# C = torch.tensordot(A, B, dims = [(0, 1), (1, 0)])
C = torch.einsum("ijk,jih->kh", A, B)
print(f"before:{A.shape, B.shape}")
print(f"after:{C.shape}")
```

### einsum 高级范例

einsum 可用于超过两个张量的计算。例如：双线性变换，
这是向量内积的一种扩展，一种常用的注意力机制实现方式

不考虑 batch 维度时，双线性变换的公式如下：

`$$A=qWk^{T}$$`

考虑 batch 维度时，无法用矩阵乘法表示，可以用元素计算公式表达如下：

`$$A_{ij}=\underset{k}{\sum}\underset{l}{\sum}Q_{ik}W_{jkl}K_{il}=Q_{ik}W_{jkl}K_{il}$$`

```python
# bilinear 注意力机制

# ==== 不考虑 batch 维度 ====
q = torch.randn(10)  # query_features
k = torch.randn(10)  # key_features
W = torch.randn(5, 10, 10)  # out_features, query_features, key_features
b = torch.randn(5)  # out_features
# a = q @ W @ k.t() + b
a = torch.bilinear(q, k, W, b)
print(f"a.shape:{a.shape}")

# ==== 考虑 batch 维度 ====
Q = torch.randn(8, 10)  # batch_size, query_features
K = torch.randn(8, 10)  # batch_size, key_features
W = torch.randn(5, 10, 10)  # out_features, query_features, key_features
b = torch.randn(5)  # out_features
A = torch.bilinear(Q, K, W, b)
A = torch.einsum("bq,oqk,bk->bo", Q, W, K) + b
print(f"A.shape:{A.shape}")
```

也可以用 einsum 来实现更常见的 scaled-dot-product 形式的 Attention

不考虑 batch 维度时，scaled-dot-product 形式的 Attention 用矩阵乘法公式表示如下：

`$$a = softmax(\frac{qk^{T}}{d_{k}})$$`

考虑 batch 维度时，无法用矩阵乘法表示，可以用元素计算公式表达如下：

`$$A_{ij}=softmax(\frac{Q_{in}K_{ijn}}{d_{k}})$$`

```python
# scaled-dot-product 注意力机制

# ==== 不考虑 batch 维度 ====
q = torch.randn(10)  # query_features
k = torch.randn(6, 10)  # key_size, key_features

d_k = k.shape[-1]
a = torch.softmax(q @ k.t() / d_k, -1)
print(f"a.shape:{a.shape}")

# ==== 不考虑 batch 维度 ====
Q = torch.randn(8, 10)
K = torch.randn(8, 6, 10)

d_k = K.shape[-1]
A = torch.softmax(torch.einsum("in,ijn->ij", Q, K) / d_k, -1)
print(f"A.shape:{A.shape}")
```

# torch 广播机制

PyTorch 的广播规则和 Numpy 是一样的:

* 如果张量的维度不同，将维度较小的张量进行扩展，直到两个张量的维度都一样
* 如果两个张量在某个维度上的长度是相同的，或者其中一个张量在该维度上的长度为 1，那么就说这两个张量在该维度上是相容的
* 如果两个张量在所有维度上都是相容的，它们就能使用广播
* 广播之后，每个维度的长度将取两个张量在该维度长度的较大值
* 在任何一个维度上，如果一个张量的长度为 1，另一个张量的长度大于 1，那么在该维度上，就好像是对第一个张量进行了复制

`torch.broadcast_tensors` 可以将多个张量根据广播规则转换成相同的维度。
维度扩展允许的操作有两种：

1. 增加一个维度
2. 对长度为 1 的维度进行复制扩展

```python
a = torch.tensor([1, 2, 3])
b = torch.tensor([[0, 0, 0], [1, 1, 1], [2, 2, 2]])

b + a
torch.cat([a[None, :]] * 3, dim = 0) + b
a_broad, b_broad = torch.broadcast_tensors(a, b)
print(a_broad)
print(b_broad)
a_broad + b_broad
```

# 参考

* [爱因斯坦求和约定](https://www.zhihu.com/question/439496333)
* [PyTorch Doc](https://pytorch.org/docs/stable/tensors.html#)
