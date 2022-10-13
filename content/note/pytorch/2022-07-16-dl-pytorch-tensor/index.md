---
title: PyTorch 张量
author: 王哲峰
date: '2022-07-16'
slug: dl-pytorch-tensor
categories:
  - deeplearning
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

- [PyTorch tensor](#pytorch-tensor)
  - [tensor 介绍](#tensor-介绍)
    - [tensor](#tensor)
    - [tensor 与 variable](#tensor-与-variable)
  - [tensor 数据类型](#tensor-数据类型)
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
    - [size 和 shape](#size-和-shape)
    - [view 和 reshape](#view-和-reshape)
  - [tensor 创建](#tensor-创建)
    - [直接创建](#直接创建)
    - [tensor 和 numpy array](#tensor-和-numpy-array)
  - [tensor 结构操作](#tensor-结构操作)
    - [索引和切片](#索引和切片)
  - [tensor 数学运算](#tensor-数学运算)
    - [相加](#相加)
    - [index](#index)
    - [resize](#resize)
    - [object trans](#object-trans)
    - [torch tensor To numpy array](#torch-tensor-to-numpy-array)
    - [numpy array To torch tensor](#numpy-array-to-torch-tensor)
    - [标量运算](#标量运算)
    - [向量运算](#向量运算)
    - [矩阵运算](#矩阵运算)
    - [爱因斯坦求和函数](#爱因斯坦求和函数)
    - [广播机制](#广播机制)
  - [cuda tensor](#cuda-tensor)
</p></details><p></p>

# PyTorch tensor

## tensor 介绍

tensor 是 PyTorch 中最基本的概念, 其参与了整个运算过程, 
这里主要介绍 tensor 的概念和属性, 如 data, variable, device 等,
并且介绍 tensor 的基本创建方法, 如直接创建、依数值创建、依概率分布创建等

### tensor

tensor 其实是多维数组,它是标量、向量、矩阵的高维拓展

### tensor 与 variable

在 PyTorch 0.4.0 版本之后 variable 已经并入 tensor, 
但是 variable 这个数据类型对于理解 tensor 来说很有帮助,
variable 是 `torch.autograd` 中的数据类型

variable(`torch.autograd.variable`) 有 5 个属性, 
这些属性都是为了 tensor 的自动求导而设置的:

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
    - `shape`
        - tensor 的形状
    - `device`
        - tensor 所在的设备, gpu/cup, tensor 放在 gpu 上才能使用加速
* 与梯度求导相关
    - `requires_grad`
        - 是否需要梯度
    - `grad`
        - `data` 的梯度
    - `grad_fn`
        - fn 表示 function 的意思，记录创建 tensor 时用到的方法
    - `is_leaf`
        - 是否是叶子节点(tensor)

## tensor 数据类型

tensor 的数据类型和 numpy array 基本一一对应，但是不支持 str 类型，包括:

* `torch.float64`(`torch.double`)
* `torch.float32`(`torch.float`)
* `torch.float16`
* `torch.int64`(`torch.long`)
* `torch.int32`(`torch.int`)
* `torch.int16`
* `torch.int8`
* `torch.uint8`
* `torch.bool`

一般神经网络建模使用的都是 `torch.float32`

### 自动推断数据类型

```python
import numpy as np
import torch

i = torch.tensor(1)
print(i, i.dtype)  # tensor(1) torch.int64

x = torch.tensor(2.0)
print(x, x.dtype)  # tensor(2.) torch.float32

b = torch.tensor(True)
print(b, b.dtype)  # tensor(True) torch.bool
```

### 指定数据类型

```python
i = torch.tensor(1, dtype = torch.int32)
print(i, i.dtype)  # tensor(1, dtype=torch.int32) torch.int32

x = torch.tensor(2.0, dtype = torch.double)
print(x, x.dtype)  # tensor(2., dtype=torch.float64) torch.float64
```

### 使用特定类型构造函数

```python
i = torch.IntTensor(1)
print(i, i.dtype)

x = torch.Tensor(np.array(2.0))
print(x, x.dtype)  # torch.FloatTensor

b = torch.BoolTensor(np.array([1, 0, 2, 0]))
print(b, b.dtype)
```

### 不同类型转换

```python
i = torch.tensor(1)
print(i, i.dtype)

x = i.float()
print(x, x.dtype)

y = i.type(torch.float)
print(y, y.dtype)

z = i.type_as(x)
print(z, z.dtype)
```

## tensor 维度

不同类型的数据可以用不同维度(dimension)的张量来表示

* 标量为 0 维张量
* 向量为 1 维张量
* 矩阵为 2 维张量
* 彩色图像有 RGB 3 个通道，可以表示为 3 维张量
* 视频还有时间维，可以表示 4 维张量

有几层中括号，就是多少维的张量

### 标量

```python
scalar = torch.tensor(True)
print(scalar)  # tensor(True)
print(scalar.dim())  # 0
```

### 向量

```python
vector = torch.tensor([1.0, 2.0, 3.0, 4.0])
print(vector)  # tensor([1., 2., 3., 4.])
print(vector.dim())  # 1
```

### 矩阵

```python
matrix = torch.tensor(
    [[1.0, 2.0],
     [3.0, 4.0]]
)
print(matrix)  # tensor([[1.0, 2.0],[3.0, 4.0]])
print(matrix.dim())  # 2
```

### 三维张量

```python
tensor3 = torch.tensor(
    [[[1.0, 2.0],
      [3.0, 4.0]],
     [[5.0, 6.0],
      [7.0, 8.0]]]
)
print(tensor3)
print(tensor3.dim())  # 3
```

### 四维张量

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
print(tensor4)
print(tensor4.dim())  # 4
```

## tensor 尺寸

* 可以使用 `shape` 属性或者 `size()` 方法查看张量在每一维的长度
* 可以使用 `view()` 方法改变张量的尺寸，如果 `view()` 方法改变尺寸失败，
  可以使用 `reshape()` 方法

### size 和 shape

```python
scalar = torch.tensor(True)
print(scalar.size())
print(scalar.shape)
```

```python
vector = torch.tensor([1.0, 2.0, 3.0, 4.0])
print(vector.size())
print(vector.shape)
```

```python
matrix = torch.tensor(
    [[1.0, 2.0],
     [3.0, 4.0]]
)
print(matrix.size())
print(matrix.shape)
```

### view 和 reshape

使用 `view` 可以改变张量尺寸:

```python
vector = torch.arange(0,12)
print(vector)
print(vector.shape)

matrix34 = vector.view(3,4)
print(matrix34)
print(matrix34.shape)

# -1表示该位置长度由程序自动推断
matrix43 = vector.view(4,-1)
print(matrix43)
print(matrix43.shape)
```

有些操作会让张量存储结构扭曲，直接使用 `view` 会失败，可以用 `reshape` 方法:

```python
matrix26 = torch.arange(0, 12).view(2, 6)
print(matrix26)
print(matrix26.shape)

# 转置操作让张量存储结构扭曲
matrix62 = matrix26.t()
print(matrix62.is_contiguous())

# 直接使用 view 方法会失败，可以使用 reshape方法
matrix34 = matrix62.view(3, 4)  # error!
matrix34 = matrix62.reshape(3, 4)
print(matrix34)
# 等价于
matrix34 = matrix62.contiguous().view(3, 4)
print(matrix34)
```

## tensor 创建

* 直接创建
    - `torch.tensor()`
    - `torch.from_numpy()`
* 依数值创建
    - `torch.empty()`
    - `torch.ones()`
    - `torch.zeros()`
    - `torch.eye()`: 单位矩阵
    - `torch.diag()`: 对角矩阵
    - `torch.fill_()`
    - `torch.arange()`
    - `torch.linspace()`
* 依概率分布创建
    - `torch.manual_seed()`
    - `torch.normal()`
    - `torch.randn()`: 正态分布
    - `torch.rand()`: 均匀分布
    - `torch.randint()`
    - `torch.randperm()`: 整数随机排列

### 直接创建

* API

```python
torch.tensor(
    data,  # list, numpy
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

### tensor 和 numpy array

* 可以用 `numpy()` 方法从 tensor 得到 numpy array，
  也可以用 `torch.from_numpy()` 从 numpy array 得到 tensor。
  这两种方法关联的 tensor 与原 numpy array 是共享内存的，
  当修改其中一个数据的时候，另一个的值也会被改动

```python
import numpy as np
import torch

arr = np.zeros(3)
tensor = torch.from_numpy(arr)
print("before add 1:")
print(arr)
print(tensor)

print("\nafter add 1:")
np.add(arr, 1, out = arr)
print(arr)
print(tensor)
```

* 如果有需要，可以用张量 `clone()` 方法拷贝 tensor，中断这种关联

```python
tensor = torch.zeros(3)

# 使用 clone 方法拷贝张量, 拷贝后的张量和原始张量内存独立
arr = tensor.clone().numpy()  # 也可以使用 tensor.data.numpy()
print("before add 1:")
print(tensor)
print(arr)

print("\nafter add 1:")
# 使用带下划线的方法表示计算结果会返回给调用张量
tensor.add_(1)
print(tensor)
print(arr)
```

* 可以使用 `item()` 方法从标量张量得到对应的 Python 数值

```python
# item方法和tolist方法可以将张量转换成Python数值和数值列表
scalar = torch.tensor(1.0)
s = scalar.item()
print(s)
print(type(s))
```

* 使用 `tolist()` 方法从 tensor 得到对应的 Python 数值列表

```python
tensor = torch.rand(2,2)
t = tensor.tolist()
print(t)
print(type(t))
```

## tensor 结构操作

* tensor 的拼接
    - `torch.cat()`
    - `torch.stack()`
* tensor 的切分
    - `torch.chunk()`
    - `torch.split()`
* tensor 的索引和切片
    - `[]`、`[:]`、`[...]`...
    - 不规则切片
        - `torch.index_select()`: 不规则切片
        - `torch.masked_select()`: 不规则切片
        - `torch.take()`: 不规则切片
    - `torch.gather()`
    - 如果要通过修改张量的部分元素值得到新的张量
        - `torch.where()`: 可以理解为 if 的张量版本
        - `torch.masked_fill()`: 选取元素逻辑和 `torch.masked_select` 相同
        - `torch.index_fill()`: 选取元素逻辑和 `torch.index_select` 相同
* tensor 的变换
    - `torch.view()` 
    - `torch.reshape()`: 改变张量的形状
    - `torch.squeeze()`: 减少维度
    - `torch.unsqueeze()`: 增加维度
    - `torch.transpose()`: 交换维度
    - `torch.permute()`: 交换维度
    - `torch.t`

### 索引和切片

* 选取行、列

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

* 不规则切片

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

## tensor 数学运算

tensor 数学运算主要有:

* 标量运算
* 向量运算
* 矩阵运算
* 爱因斯坦求和函数 `torch.einsum()` 进行任意阶张量运算
* 广播机制

tensor 数学运算 API:

* 相加
    - `+`
    - `torch.add(, out)`
    - `.add_()`
* index
    - `[:, :]`
* resize
    - `.view()`
    - `.size()`
* object trans
    - `.items()`
* numpy.array to torch.tensor
    - `torch.from_numpy()`
* torch.tensor to numpy.array
    - `.numpy()`

### 相加

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

### index

```python
import torch

x = torch.zeros(5, 3, dtype = torch.long)
print(x[:, 1])
```

### resize

```python
import torch

x = torch.randn(4, 4)
y = x.view(16)
z = x.view(-1, 8)
print(x.size(), y.size(), z.size())
```

### object trans

```python
import torch

x = torch.randn(1)
print(x)
print(x.item()) # python number
```

### torch tensor To numpy array

```python
import torch

a = torch.ones(5)
b = a.numpy()
print(a)
print(b)

a.add_(1)
print(a)
print(b)
```

### numpy array To torch tensor

```python
import numpy as np

a = np.ones(5)
b = torch.from_numpy(a)
np.add(a, 1, out = a)

print(a)
print(b)
```

### 标量运算


### 向量运算


### 矩阵运算


### 爱因斯坦求和函数


### 广播机制

PyTorch 的广播规则和 Numpy 是一样的:

* 如果张量的维度不同，将维度较小的张量进行扩展，直到两个张量的维度都一样
* 如果两个张量在某个维度上的长度是相同的，或者其中一个张量在该维度上的长度为 1，那么就说这两个张量在该维度上是相容的
* 如果两个张量在所有维度上都是相容的，它们就能使用广播
* 广播之后，每个维度的长度将取两个张量在该维度长度的较大值
* 在任何一个维度上，如果一个张量的长度为 1，另一个张量的长度大于 1，那么在该维度上，就好像是对第一个张量进行了复制



## cuda tensor

* CUDA 可用
* 使用 `torch.device` 对象将 tensors 移出或放入 GPU

```python
x = torch.tensor([1])
if torch.cuda.is_available():
    device = torch.device("cuda")  # a cuda device object
    y = torch.ones_like(x, device = device)  # directly create a tensor on gpu
    x = x.to(device)  # or just use strings `.to("cuda")`
    z = x + y
    z.to("cpu", torch.double)  # `.to` can also change dtype together!
```

