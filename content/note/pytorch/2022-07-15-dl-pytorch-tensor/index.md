---
title: PyTorch tensor
author: 王哲峰
date: '2022-07-15'
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
  - [tensor API](#tensor-api)
    - [tensor 创建 API](#tensor-创建-api)
    - [tensor 操作 API](#tensor-操作-api)
  - [tensor 创建](#tensor-创建)
    - [直接创建](#直接创建)
    - [依数值创建](#依数值创建)
    - [依概率分布创建](#依概率分布创建)
  - [tensor 操作](#tensor-操作)
  - [cuda tensor](#cuda-tensor)
- [PyTorch 数据并行](#pytorch-数据并行)
  - [让模型跑在 GPU 上](#让模型跑在-gpu-上)
  - [让模型跑在多个 GPU 上](#让模型跑在多个-gpu-上)
</p></details><p></p>

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

