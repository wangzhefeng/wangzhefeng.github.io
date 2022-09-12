---
title: PyTorch 张量、自动微分机制与动态计算图机制
author: 王哲峰
date: '2022-07-16'
slug: dl-pytorch-tensor-autograd-graph
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

- [内容](#内容)
- [PyTorch tensor](#pytorch-tensor)
  - [tensor 介绍](#tensor-介绍)
  - [tensor API](#tensor-api)
    - [tensor 创建 API](#tensor-创建-api)
    - [tensor 操作 API](#tensor-操作-api)
  - [tensor 创建](#tensor-创建)
    - [直接创建](#直接创建)
    - [依数值创建](#依数值创建)
    - [依概率分布创建](#依概率分布创建)
  - [tensor 操作](#tensor-操作)
  - [cuda tensor](#cuda-tensor)
  - [tensor 数据结构](#tensor-数据结构)
  - [tensor 结构操作](#tensor-结构操作)
  - [tensor 数学运算](#tensor-数学运算)
    - [标量运算](#标量运算)
    - [广播机制](#广播机制)
- [PyTorch 数据并行](#pytorch-数据并行)
  - [让模型跑在 GPU 上](#让模型跑在-gpu-上)
  - [让模型跑在多个 GPU 上](#让模型跑在多个-gpu-上)
- [PyTorch 自动微分机制](#pytorch-自动微分机制)
  - [使用 backward() 方法求导数](#使用-backward-方法求导数)
    - [标量的反向传播](#标量的反向传播)
    - [非标量的反向传播](#非标量的反向传播)
    - [非标量的反向传播可以用标量的反向传播实现](#非标量的反向传播可以用标量的反向传播实现)
  - [使用 grad() 方法求导数](#使用-grad-方法求导数)
    - [标量的反向传播](#标量的反向传播-1)
    - [多个自变量求导](#多个自变量求导)
  - [使用自动微分和优化器求最小值](#使用自动微分和优化器求最小值)
  - [autograd 包功能](#autograd-包功能)
- [PyTorch 动态计算图机制](#pytorch-动态计算图机制)
  - [计算图](#计算图)
  - [PyTorch 动态计算图机制](#pytorch-动态计算图机制-1)
  - [PyTorch 计算图中的张量节点](#pytorch-计算图中的张量节点)
    - [计算图的正向传播时立即执行的](#计算图的正向传播时立即执行的)
    - [计算图在反向传播后立即销毁](#计算图在反向传播后立即销毁)
  - [PyTorch 计算图中的 Function 节点](#pytorch-计算图中的-function-节点)
  - [PyTorch 计算图与反向传播](#pytorch-计算图与反向传播)
  - [PyTorch 计算图叶节点和非叶节点](#pytorch-计算图叶节点和非叶节点)
  - [PyTorch 计算图在 TensorBoard 中的可视化](#pytorch-计算图在-tensorboard-中的可视化)
</p></details><p></p>


# 内容

- 计算图
   - 描述运算的有向无环图
      - Tensor 的 `is_leaf` 属性
      - Tensor 的 `grad_fn` 属性
- PyTorch 动态图机制
   - 动态图与静态图
- PyTorch 自动微分机制
   - `torch.autograd.backward()` 方法自动求取梯度
   - `torch.autograd.grad(  )` 方法可以高阶求导

**Note**

- 梯度不自动清零
- 依赖叶节点的节点, `requires_grad` 默认为 True
- 叶节点不能执行原位操作

# PyTorch tensor

## tensor 介绍

tensor 是 PyTorch 中最基本的概念, 其参与了整个运算过程, 
这里主要介绍 tensor 的概念和属性, 如 data, variable, device 等,
并且介绍 tensor 的基本创建方法, 如直接创建、依数值创建、依概率分布创建等

- tensor
    - tensor 其实是多维数组,它是标量、向量、矩阵的高维拓展
- tensor 与 variable
    - 在 PyTorch 0.4.0 版本之后 variable 已经并入 tensor, 
      但是 variable 这个数据类型对于理解 tensor 来说很有帮助,
      variable 是 `torch.autograd` 中的数据类型. 
    - variable(`torch.autograd.variable`) 有 5 个属性, 
      这些属性都是为了 tensor 的自动求导而设置的:
        - `data`
        - `grad`
        - `grad_fn`
        - `requires_grad`
        - `is_leaf`
    - tensor(`torch.tensor`) 有 8 个属性:
        - 与数据本身相关
            - `data`: 被包装的 tensor
            - `dtype`: tensor 的数据类型, 如 
                - `torch.floattensor`
                - `torch.cuda.floattensor`
                - `float32`
                - `int64(torch.long)`
            - `shape`: tensor 的形状
            - `device`: tensor 所在的设备, gpu/cup, tensor 放在 gpu 上才能使用加速
        - 与梯度求导相关
            - `requires_grad`: 是否需要梯度
            - `grad`: `data` 的梯度
            - `grad_fn`: fn 表示 function 的意思，记录创建 tensor 时用到的方法
            - `is_leaf`: 是否是叶子节点(tensor)

## tensor API

### tensor 创建 API

- 直接创建
    - `torch.tensor()`
    - `torch.from_numpy()`
- 依数值创建
    - `torch.empty()`
    - `torch.ones()`
    - `torch.zeros()`
    - `torch.eye()`
    - `torch.full()`
    - `torch.arange()`
    - `torch.linspace()`
- 依概率分布创建
    - `torch.normal()`
    - `torch.randn()`
    - `torch.rand()`
    - `torch.randint()`
    - `torch.randperm()`

### tensor 操作 API

- tensor 的基本操作
    - tensor 的拼接
        - `torch.cat()`
        - `torch.stack()`
    - tensor 的切分
        - `torch.chunk()`
        - `torch.split()`
    - tensor 的索引
        - `index_select()`
        - `masked_select()`
    - tensor 的变换
        - `torch.reshape()`
        - `torch.transpose()`
        - `torch.t`
- tensor 的数学运算
    - `add(input, aplha, other)`

## tensor 创建

```python
from __future__ import print_function
import numpy as np
import torch
```

### 直接创建

1. 从 data 创建 tensor api

- API

    ```python
    torch.tensor(
        data,                   # list, numpy
        dtype = none,
        device = none,
        requires_grad = false,
        pin_memory = false,      # 是否存于锁页内存
    )
    ```

- 示例

    ```python
    arr = np.ones((3, 3))

    t = torch.tensor(arr, device = "cuda")
    print(t)
    ```

2. 通过 numpy array 创建 tensor api

> 创建的 tensor 与原 ndarray 共享内存，当修改其中一个数据的时候，另一个也会被改动

- API

    ```python
    torch.from_numpy(ndarray)
    ```

- 示例

    ```python
    # np.ndarray
    arr = np.array(
        [
            [1, 2, 3], 
            [4, 5, 6]
        ]
    )
    print(arr)
    
    # torch.tensor
    t = torch.from_numpy(arr)
    print(t)

    # 修改 arr    
    arr[0, 0] = 0
    print(arr, t)

    # 修改 torch.tensor
    t[1, 1] = 100
    print(arr, t)
    ```

### 依数值创建

- API

    ```python
    torch.zeros(
        *size,
        out = none,  # 输出张量，就是把这个张量赋值给另一个张量，但这两个张量一样，指的是同一个内存地址
        dtype = none,
        layout = torch.strided,  # 内存的布局形式
        device = none,
        requires_grad = false,
    )
    ```

- 示例

    ```python
    out_t = torch.tensor([1])
    t = torch.zeros((3, 3), out = out_t)
    print(out_t, t)
    print(id(out_t), id(t), id(t) == id(out_t))
    ```

### 依概率分布创建

* TODO

## tensor 操作

> * 相加
>   - `+`
>   - `torch.add(, out)`
>   - `.add_()`
> * index
>   - `[:, :]`
> * resize
>   - `.view()`
>   - `.size()`
> * object trans
>   - `.items()`
> * numpy.array to torch.tensor
>   - `torch.from_numpy()`
> * torch.tensor to numpy.array
>   - `.numpy()`

- add

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

- index

    ```python
    import torch

    x = torch.zeros(5, 3, dtype = torch.long)
    print(x[:, 1])
    ```

- resize

    ```python
    import torch

    x = torch.randn(4, 4)
    y = x.view(16)
    z = x.view(-1, 8)
    print(x.size(), y.size(), z.size())
    ```

- object trans

    ```python
    import torch

    x = torch.randn(1)
    print(x)
    print(x.item()) # python number
    ```

- torch tensor To numpy array

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

- numpy array To torch tensor

    ```python
    import numpy as np
    
    a = np.ones(5)
    b = torch.from_numpy(a)
    np.add(a, 1, out = a)
    
    print(a)
    print(b)
    ```

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

## tensor 数据结构



## tensor 结构操作


## tensor 数学运算

tensor 数学运算主要有:

* 标量运算
* 向量运算
* 矩阵运算
* 爱因斯坦求和函数 `torch.einsum()` 进行任意阶张量运算
* 广播机制

### 标量运算




### 广播机制

PyTorch 的广播规则和 Numpy 是一样的:

* 如果张量的维度不同，将维度较小的张量进行扩展，直到两个张量的维度都一样
* 如果两个张量在某个维度上的长度是相同的，或者其中一个张量在该维度上的长度为 1，那么就说这两个张量在该维度上是相容的
* 如果两个张量在所有维度上都是相容的，它们就能使用广播
* 广播之后，每个维度的长度将取两个张量在该维度长度的较大值
* 在任何一个维度上，如果一个张量的长度为 1，另一个张量的长度大于 1，那么在该维度上，就好像是对第一个张量进行了复制

# PyTorch 数据并行

## 让模型跑在 GPU 上

```python
import torch

# 让模型在 GPU 上运行
device = torch.device("cuda:0")
model.to(device)

# 将 tensor 复制到 GPU 上
my_tensor = torch.ones(2, 2, dtype = torch.double)
mytensor = my_tensor.to(device)
```

## 让模型跑在多个 GPU 上

> * PyTorch 默认使用单个 GPU 执行运算

```python
model = nn.DataParallel(model)
```

```python
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

# Parameters
input_size = 5
output_size = 2

batch_size = 30
data_size = 100

# Device
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


# data 
class RandomDataset(Dataset):

    def __init__(self, size, length):
        self.len = length
        self.data = torch.randn(length, size)

    def __len__(self):
        return self.len
    
    def __getitem__(self, index):
        return self.data[index]

rand_loader = DataLoader(
    dataset = RandomDataset(input_size, data_size), 
    batch_size = batch_size, 
    shuffle = True
)

# model
class Model(nn.Module):

    def __init__(self, input_size, output_size):
        super(Model, self)__init__()
        self.fc = nn.Linear(input_size, output_size)

    def forward(self, input):
        output = self.fc(input)
        print("\tIn Model: input size", input.size(),
                "output size", output.size())

        return output

model = Model(input_size, output_size)
if torch.cuda.device_count() > 1:
    print("Let's use", torch.cuda.device_count(), "GPUs!")
    model = nn.DataParallel(model)

model.to(device)


for data in rand_loader:
    input = data.to(device)
    output = model(input)
    print("Outside: input size", input.size(),
        "output_size", output.size())
```

# PyTorch 自动微分机制

神经网络通常依赖反向传播算法求梯度来更新网络参数，
而求梯度过程通常是一件非常复杂繁琐而且容易出错的事情。
深度学习框架可以帮助自动地完成这种求梯度的运算

PyTorch 一般通过反向传播 `torch.autograd.backward()` 方法实现求梯度计算，
该方法求得的梯度将保存在对应自变量张量的 `grad` 属性下。
除此之外，也能够调用 `torch.autograd.grad()` 函数实现求梯度计算。
这就是 PyTorch 的自动微分机制

## 使用 backward() 方法求导数

PyTorch 自动微分机制使用的是 `torch.autograd.backward()` 方法，
功能就是自动求取梯度

* `torch.autograd.backward()` 方法通常在一个标量张量上调用，
  该方法求得的梯度将保存在对应自变量张量的 `grad` 属性下
* 如果调用的张量非标量，则要传入一个和它形状相同的 `gradient` 参数张量，
  相当于用该 `gradient` 参数张量与调用张量作向量点乘，得到的标量结果再反向传播

下面是 `torch.autograd.backward()` API:

```python
torch.autograd.backward(
   tensors, 
   gard_tensors = None, 
   retain_graph = None, 
   create_graph = False
)
```

### 标量的反向传播

```python
import numpy as np
import torch

"""
# 目标:
# 求 f(x) = a*x**2 + b*x + c 关于 x 的在 a=1.0,b=-2.0,c=1.0 处的导数
"""

# 自变量(标量)
x = torch.tensor(0.0, requires_grad = True)  # x 需要被求导
# 参数
a = torch.tensor(1.0)
b = torch.tensor(-2.0)
c = torch.tensor(1.0)
# 因变量
y = a * torch.pow(x, 2) + b * x + c

# 因变量 y 反向传播计算对 自变量 x 求导
y.backward()
dy_dx = x.grad
print(dy_dx)
```

```
tensor(-2.)
```

### 非标量的反向传播

```python
import numpy as np
import torch

"""
# 目标:
# 求 f(x) = a*x**2 + b*x + c 关于 x 的在 a=1.0,b=-2.0,c=1.0 处的导数
"""

# 自变量(张量)
x = torch.tensor([
    [0.0, 0.0],
    [1.0, 2.0],
], requires_grad = True)  # x 需要被求导
# 参数
a = torch.tensor(1.0)
b = torch.tensor(-2.0)
c = torch.tensor(1.0)
# 因变量
y = a * torch.pow(x, 2) + b * x + c

# gradient = torch.tensor([[1.0, 1.0], [1.0, 1.0]])
gradient = torch.ones_like(x)

print(f"x:\n{x}")
print(f"y:\n{y}")
y.backward(gradient = gradient)
x_grad = x.grad
print(f"x_grad:\n{x_grad}")
```

```
x:
 tensor([[0., 0.],
        [1., 2.]], requires_grad=True)
y:
 tensor([[1., 1.],
        [0., 1.]], grad_fn=<AddBackward0>)
x_grad:
 tensor([[-2., -2.],
        [ 0.,  2.]])
```

### 非标量的反向传播可以用标量的反向传播实现

```python
import numpy as np
import torch

"""
# 目标:
# 求 f(x) = a*x**2 + b*x + c 关于 x 的在 a=1.0,b=-2.0,c=1.0 处的导数
"""

# 自变量(张量)
x = torch.tensor([
    [0.0, 0.0],
    [1.0, 2.0],
], requires_grad = True)  # x 需要被求导
# 参数
a = torch.tensor(1.0)
b = torch.tensor(-2.0)
c = torch.tensor(1.0)
# 因变量
y = a * torch.pow(x, 2) + b * x + c

gradient = torch.tensor([[1.0, 1.0], [1.0, 1.0]])
z = torch.sum(y * gradient)

print(f"x:\n{x}")
print(f"y:\n{y}")
z.backward(gradient = gradient)
x_grad = x.grad
print(f"x_grad:\n{x_grad}")
```

```
x: 
tensor([[0., 0.],
        [1., 2.]], requires_grad=True)
y: 
tensor([[1., 1.],
        [0., 1.]], grad_fn=<AddBackward0>)
x_grad:
tensor([[-2., -2.],
        [ 0.,  2.]])
```

## 使用 grad() 方法求导数

### 标量的反向传播

```python
import numpy as np
import torch

"""
# 目标:
# 求 f(x) = a*x**2 + b*x + c 关于 x 的在 a=1.0,b=-2.0,c=1.0 处的导数
"""

# 自变量(标量)
x = torch.tensor(0.0, requires_grad = True)  # x 需要被求导
# 参数
a = torch.tensor(1.0)
b = torch.tensor(-2.0)
c = torch.tensor(1.0)
# 因变量
y = a * torch.pow(x, 2) + b * x + c

# create_graph 设置为 True 将允许创建更高阶的导数
dy_dx = torch.autograd.grad(y, x, create_graph = True)[0]
print(dy_dx.data)

# 求二阶导数
dy2_dx2 = torch.autograd.grad(dy_dx, x)[0]
print(dy2_dx2.data)
```

```
tensor(-2.)
tensor(2.)
```

### 多个自变量求导

```python
import numpy as np
import torch

x1 = torch.tensor(1.0, requires_grad = True)  # x1 需要被求导
x2 = torch.tensor(2.0, requires_grad = True)  # x2 需要被求导
y1 = x1 * x2
y2 = x1 + x2

# 允许同时对多个自变量求导数
(dy1_dx1, dy1_dx2) = torch.autograd.grad(
    outputs = y1,
    inputs = [x1, x2], 
    retain_graph = True
)
print(dy1_dx1, dy1_dx2)

# 如果有多个因变量，相当于把多个因变量的梯度结果求和
(dy12_dx1, dy12_dx2) = torch.autograd.grad(
    outputs = [y1, y2], 
    inputs = [x1, x2],
)
print(dy12_dx1, dy12_dx2)
```

```
tensor(2.) tensor(1.)
tensor(3.) tensor(2.)
```

## 使用自动微分和优化器求最小值

```python
import numpy as np
import torch

"""
# 目标:
# 求 f(x) = a*x**2 + b*x + c 的最小值
"""

# 自变量(标量)
x = torch.tensor(0.0, requires_grad = True)  # x 需要被求导
# 参数
a = torch.tensor(1.0)
b = torch.tensor(-2.0)
c = torch.tensor(1.0)
# 因变量函数
def f(t):
    result = a * torch.pow(t, 2) + b * x + c
    return result

# 优化器
optimizer = torch.optim.SGD(params = [x], lr = 0.01)

for i in range(500):
    optimizer.zero_grad()
    y = f(x)
    y.backward()
    optimizer.step()

print(f"y = {f(x).data}; x = {x.data}")
```

```
y= tensor(0.) ; x= tensor(1.0000)
```

## autograd 包功能

`torch.autograd` 包提供了对所有 Tensor 的自动微分操作:

- `torch.Tensor(requires_grad = True)`
    - 跟踪 `torch.Tensor` 上所有的操作
- `requires_grad` 属性
    - `torch.Tensor` 是否被跟踪
- `torch.autograd.backward()` 方法
    - 自动计算所有的梯度
- `grad` 属性
    - `torch.Tensor` 上的梯度
- `torch.autograd.grad()` 方法
    - 自动计算所有的梯度
- `.detach()`
    - 停止跟踪 `torch.Tensor` 上的跟踪历史、未来的跟踪
- `with torch.no_grad(): pass`
- `.grad_fn`
- `.zero_grad()`

```python
import torch

# --------------------
# 创建 Tensor 时设置 requires_grad 跟踪前向计算
# --------------------
x = torch.ones(2, 2, requires_grad = True)
print("x:", x)

y = x + 2
print("y:", y)
print("y.grad_fn:", y.grad_fn)

z = y * y * 3
print("z:", z)
print("z.grad_fn", z.grad_fn)

out = z.mean()
print("out:", out)
print("out.grad_fn:", out.grad_fn)
out.backward()
print("x.grad:", x.grad)

# --------------------
# .requires_grad_() 能够改变一个已经存在的 Tensor 的 `requires_grad`
# --------------------
a = torch.randn(2, 2)
a = ((a * 3) / (a - 1))
print("a.requires_grad", a.requires_grad)
a.requires_grad_(True)
print("a.requires_grad:", a.requires_grad)
b = (a * a).sum()
print("b.grad_fn", b.grad_fn)



# 梯度
x = torch.randn(3, requires_grad = True)
y = x * 2
while y.data.norm() < 1000:
      y = y * 2
v = torch.tensor([0.1, 1.0, 0.0001], dtype = torch.float)
y.backward(v)
print(x.grad)

# .requires_grad
print(x.requires_grad)
print((x ** 2).requires_grad)

with torch.no_grad():
      print((x ** 2).requires_grad)

# .detach()
print(x.requires_grad)
y = x.detach()
print(y.requires_grad)
print(x.eq(y).all())
```

# PyTorch 动态计算图机制

## 计算图

计算图是用来描述运算的有向无环图。主要有两个因素:节点、边。
其中节点表示数据，如向量、矩阵、张量；而边表示运算，如加减乘除、卷积等。

使用计算图的好处不仅是让计算看起来更加简洁，
还有个更大的优势是让梯度求导也变得更加方便

示例:

```python
x = torch.tensor([2.], requires_grad = True)
w = torch.tensor([1.], requires_grad = True)

a = torch.add(w, x)
b = torch.add(w, 1)

y = torch.mul(a, b)

y.backward()
print(w.grad)
```

## PyTorch 动态计算图机制

![img](./images/torch动态图.gif)

PyTorch 的计算图由节点和边组成，节点表示张量或者 Function，边表示张量和 Function 之间的依赖关系。
PyTorch 中的计算图是动态度，这里的动态主要有两重含义:

* 计算图的正向传播时立即执行的
    - 无需等待完成的计算图创建完毕，每条语句都会在计算图中动态添加节点和边，并立即执行正向传播得到计算结果
* 计算图在反向传播后立即销毁，下次调用需要重新构建计算图
    - 如果在程序中使用了 `backward` 方法执行了反向传播，或者利用了 `torch.autograd.grad()` 方法计算了梯度，
      那么创建的计算图会被立即销毁，释放了存储空间，下次调用需要重新创建

## PyTorch 计算图中的张量节点

### 计算图的正向传播时立即执行的

```python
import torch

w = torch.tensor([[3.0, 1.0]], requires_grad = True)
b = torch.tensor([[3.0]], requires_grad = True)
X = torch.randn(10, 2)
Y = torch.randn(10, 1)
# Y_hat 定义后其正向传播被立即执行，与其后面的 loss 创建语句无关
Y_hat = X @ w.t() + b
loss = torch.mean(torch.pow(Y_hat - Y, 2))

print(loss.data)
print(Y_hat.data)
```

```
tensor(17.8969)
tensor([[3.2613],
        [4.7322],
        [4.5037],
        [7.5899],
        [7.0973],
        [1.3287],
        [6.1473],
        [1.3492],
        [1.3911],
        [1.2150]])
```

### 计算图在反向传播后立即销毁

```python
import torch

w = torch.tensor([[3.0, 1.0]], requires_grad = True)
b = torch.tensor([[3.0]], requires_grad = True)
X = torch.randn(10, 2)
Y = torch.randn(10, 1)
# Y_hat 定义后其正向传播被立即执行，与其后面的 loss 创建语句无关
Y_hat = X @ w.t() + b
loss = torch.mean(torch.pow(Y_hat - Y, 2))

# 计算图在反向传播后立即销毁
loss.backward()  # 如果再次执行反向传播(loss.backward())将报错

# 如果需要保留计算图，需要设置 retain_graph = True
# loss.backward(retain_graph = True)
```

## PyTorch 计算图中的 Function 节点

计算图中的 Function 节点就是 PyTorch 中各种对张量操作的函数。
这些 Function 与 Python 中的函数有一个较大的区别是它同时包括正向计算逻辑和反向传播逻辑。
可以通过继承 `torch.autograd.Function` 来创建这种支持反向传播的 Function

```python
class MyReLU(torch.autograd.Function):

    # 正向传播逻辑，可以用 ctx 存储一些值，供反向传播使用
    @staticmethod
    def forward(ctx, input):
        ctx.save_for_backward(input)
        return input.clamp(min = 0)
    
    # 反向传播逻辑
    @staticmethod
    def backward(ctx, grad_output):
        input, = ctx.saved_tensors
        grad_input = grad_output.clone()
        grad_input[input < 0] = 0
        return grad_input
```

```python
import torch

w = torch.tensor([[3.0, 1.0]], requires_grad = True)
b = torch.tensor([[3.0]], requires_grad = True)
X = torch.tensor([[-1.0, -1.0], [1.0, 1.0]])
Y = torch.tensor([[2.0, 3.0]])

relu = MyReLU.apply
Y_hat = relu(X @ w.t() + b)
loss = torch.mean(torch.pow(Y_hat - Y, 2))

print(w.grad)
print(b.grad)
# Y_hat 的梯度函数即是定义的 MyReLU.backward
print(Y_hat.grad_fn)
```

## PyTorch 计算图与反向传播

从计算图理解反向传播的原理和过程

```python
import torch

x = torch.tensor(3.0, requires_grad = True)
y1 = x + 1
y2 = 2 * x
loss = (y1 - y2) ** 2

loss.backward()
```

`loss.backward()` 语句调用后，依次发生以下计算过程:

1. `loss` 自己的 `grad` 梯度赋值为 1，即对自身的梯度 为 1
    - `loss.grad` = dloss_dloss(x = 3) = 1
2. `loss` 根据其自身梯度以及关联的 `backward()` 方法，
   计算出其对应的自变量即 `y1` 和 `y2` 的梯度，
   将该值赋值到 `y1.grad` 和 `y2.grad`
    - `y1.grad` = dloss_dy1(x = 3) = 2 * (y1 - y2) * 1 = 2 * (4 - 6) * 1 = -4
    - `y2.grad` = dloss_dy2(x = 3) = 2 * (y1 - y2) * (-1)= 2 * (4 - 6) * (-1) = 4
3. `y1` 和 `y2` 根据其自身梯度以及关联的 `backward()` 方法，
   分别计算出其对应的自变量 `x` 的梯度，`x.grad` 将其收到多个梯度值累加
    - `x.grad` = dloss_dx = dloss_loss * (dloss_dy1 * dy1_dx + dloss_dy2 * dy2_dx) = 1 * (-4 * 1 + 4 * 2) = 4

因为求导链式法则衍生的梯度累加规则，张量的grad梯度不会自动清零，在需要的时候需要手动置零


## PyTorch 计算图叶节点和非叶节点

```python
import torch

x = torch.tensor(3.0, requires_grad = True)
y1 = x + 1
y2 = 2 * x
loss = (y1 - y2) ** 2

loss.backward()

print(f"loss.grad: {loss.grad}")
print(f"y1.grad: {y1.grad}")
print(f"y2.grad: {y2.grad}")
print(f"x.grad: {x.grad}")

print(f"loss.is_leaf: {loss.is_leaf}")
print(f"y1.is_leaf: {y1.is_leaf}")
print(f"y2.is_leaf: {y2.is_leaf}")
print(f"x.is_leaf: {x.is_leaf}")
```

```
loss.grad: None
y1.grad: None
y2.grad: None
tensor(4.)

loss.is_leaf: False
y1.is_leaf: False
y2.is_leaf: False
x.is_leaf: True
```

在反向传播的过程中，只有 `is_leaf = True` 的叶节点需要求导的张量的导数结果才会被保留下来。
叶节点张量需要满足两个条件:

1. 叶节点张量是由用户直接创建的张量，而非由某个 Function 通过计算得到的张量
2. 叶节点张量的 `requires_grad` 属性必须为 `True`

PyTorch 这样的设计规则主要是为了节约内存或者显存空间，因为几乎所有时候，用户只会关心他自己直接创建的张量的梯度。
所有依赖于叶节点张量的张量，其 `requires_grad = True`，但其梯度值只在计算过程中被用到，不会最终存储到 `grad` 中。
如果需要保留中间计算结果的梯度到 `grad` 属性中，可以使用 `retain_grad` 方法。如果仅仅是为了调试代码产看梯度值，
可以利用 `register_hook` 打印日志，可以查看非叶节点的梯度值

```python
import torch 

#正向传播
x = torch.tensor(3.0, requires_grad = True)
y1 = x + 1
y2 = 2 * x
loss = (y1 - y2) ** 2

#非叶子节点梯度显示控制
y1.register_hook(lambda grad: print('y1 grad: ', grad))
y2.register_hook(lambda grad: print('y2 grad: ', grad))
loss.retain_grad()

#反向传播
loss.backward()

print(f"loss.grad: {loss.grad}")
print(f"y1.grad: {y1.grad}")
print(f"y2.grad: {y2.grad}")
print(f"x.grad: {x.grad}")
```

```
y2 grad: tensor(4.)
y1 grad: tensor(-4.)
loss.grad: tensor(1.)
x.grad: tensor(4.)
```

## PyTorch 计算图在 TensorBoard 中的可视化

可以使用 `torch.utils.tensorboard` 将计算图导出到 TensorBoard 进行可视化

```python
import torch
from torch import nn
from torch.utils.tensorboard import SummaryWriter
from tensorboard import notebook
# %load_ext tensorboard
# %tensorboard --logdir ./data/tensorboard

class Net(nn.Module):
    
    def __init__(self):
        super(Net, self).__init__()
        self.w = nn.Parameter(torch.randn(2, 1))
        self.b = nn.Parameter(torch.zeros(1, 1))
    
    def forward(self, x):
        y = x@self.w + self.b
        return y

net = Net()

# TODO
writer = SummaryWriter("./tensorboard")
writer.add_graph(net, input_to_model = torch.rand(10, 2))
writer.close()

notebook.list()
# 在 tensorboard 中查看模型
notebook.start("--logdir ./data/tensorboard")
notebook.list()
```