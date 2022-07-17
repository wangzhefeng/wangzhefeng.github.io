---
title: PyTorch 基础
author: 王哲峰
date: '2022-07-15'
slug: dl-pytorch
categories:
  - deeplearning
tags:
  - tool
---

<style>
h1 {
  background-color: #2B90B6;
  background-image: linear-gradient(45deg, #4EC5D4 10%, #146b8c 20%);
  background-size: 100%;
  -webkit-background-clip: text;
  -moz-background-clip: text;
  -webkit-text-fill-color: transparent;
  -moz-text-fill-color: transparent;
}
h2 {
  background-color: #2B90B6;
  background-image: linear-gradient(45deg, #4EC5D4 10%, #146b8c 20%);
  background-size: 100%;
  -webkit-background-clip: text;
  -moz-background-clip: text;
  -webkit-text-fill-color: transparent;
  -moz-text-fill-color: transparent;
}

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

- [需要掌握的内容](#需要掌握的内容)
- [PyTorch tensor 的创建](#pytorch-tensor-的创建)
  - [tensor 的介绍](#tensor-的介绍)
  - [tensor 的创建](#tensor-的创建)
    - [直接创建](#直接创建)
    - [依数值创建](#依数值创建)
  - [tensor 的操作](#tensor-的操作)
- [PyTorch 数据读取与预处理](#pytorch-数据读取与预处理)
  - [Dataset 和 DataLoader](#dataset-和-dataloader)
    - [使用 TorchVision, TorchText, TorchAudio dataset](#使用-torchvision-torchtext-torchaudio-dataset)
    - [使用文件创建自定义的 Dataset](#使用文件创建自定义的-dataset)
    - [2.1.3 torch.utils.data API](#213-torchutilsdata-api)
- [PyTorch 数据预处理](#pytorch-数据预处理)
  - [Transforms 示例](#transforms-示例)
  - [torchvision transforms](#torchvision-transforms)
  - [torchtext transforms](#torchtext-transforms)
  - [torchaudio transforms](#torchaudio-transforms)
- [PyTorch 数据并行](#pytorch-数据并行)
  - [让模型跑在 GPU 上](#让模型跑在-gpu-上)
  - [让模型跑在多个 GPU 上](#让模型跑在多个-gpu-上)
- [PyTorch 动态图、自动求导](#pytorch-动态图自动求导)
  - [计算图](#计算图)
  - [Pytorch 动态图机制](#pytorch-动态图机制)
  - [PyTorch 自动求导机制](#pytorch-自动求导机制)
- [PyTorch Cheat Sheet](#pytorch-cheat-sheet)
  - [Imports](#imports)
  - [Tensors](#tensors)
  - [Deep Learning](#deep-learning)
  - [Data Utilities](#data-utilities)
</p></details><p></p>

# 需要掌握的内容

* [ ] Get Stated
    - [ ] Start Locally
    - [ ] Start via Partners
* [ ] Tutorials
    - Learn the Basics
        - [ ] Introduction to PyTorch
            - [x] Learn the Basics
            - [ ] Quickstart
            - [ ] Tensors
            - [ ] Datasets & DataLoaders
            - [ ] Transform
            - [x] Save and Load the Model
        - [ ] Learning PyTorch
        - [ ] Image and Video
        - [ ] Text
        - [ ] Deploying PyTorch Model in Production
        - [ ] Model Optimization
        - [ ] Parallel and Distributed Training
    - PyTorch Recipes
        - [ ] 快速入门
        - [ ] 自定义层、训练循环
        - [ ] 分布式训练
    - Additional Resources
        - [ ] Examples of PyTorch
        - [ ] PyTorch Cheat Sheet
        - [ ] Tutorials on GitHub
        - [ ] Run Tutorials on Google Colab
* [ ] Docs
    - [ ] PyTorch
    - [ ] torchaudio
    - [ ] torchtext
    - [ ] torchvision
    - [ ] TorchElastic
    - [ ] TorchServe
    - [ ] PyTorch on XLA Devices
* [ ] GitHub
    - [ ] https://github.com/pytorch/pytorch
    - [ ] https://github.com/onnx/tutorials

# PyTorch tensor 的创建

* 1.tensor 的简介及创建
   - tensor 是多维数组
   - tensor 的创建
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
* 2.tensor 的操作
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

## tensor 的介绍

tensor 是 PyTorch 中最基本的概念,其参与了整个运算过程, 
这里主要介绍 tensor 的概念和属性, 如 data, variable, device 等,
并且介绍 tensor 的基本创建方法, 如直接创建、依属主创建、依概率分布创建等

- tensor
    - tensor 其实是多维数组,它是标量、向量、矩阵的高维拓展
- tensor 与 variable 
    - 在 PyTorch 0.4.0 版本之后 variable 已经并入 tensor, 
      但是 variable 这个数据类型对于理解 tensor 来说很有帮助,
      variable 是 `torch.autograd` 中的数据类型. 
      variable(`torch.autograd.variable`) 有 5 个属性, 
      这些属性都是为了 tensor 的自动求导而设置的:
        - `data`
        - `grad`
        - `grad_fn`
        - `requires_grad`
        - `is_leaf`
    - tensor(`torch.tensor`) 有 8 个属性:
        - 与数据本身相关
            - `data`: 被包装的 tensor
            - `dtype`: tensor 的数据类型, 如 `torch.floattensor`, `torch.cuda.floattensor`, `float32`, `int64(torch.long)`
            - `shape`: tensor 的形状
            - `device`: tensor 所在的设备, gpu/cup, tensor 放在 gpu 上才能使用加速
        - 与梯度求导相关
            - `requires_grad`: 是否需要梯度
            - `grad`: `data` 的梯度
            - `grad_fn`: fn 表示 function 的意思，记录创建 tensor 时用到的方法
            - `is_leaf`: 是否是叶子节点(tensor)

## tensor 的创建

```python
from __future__ import print_function
import numpy as np
import torch
```

### 直接创建

1. torch.tensor(): 从 data 创建 tensor api

- API

    ```python
    torch.tensor(
        data,                   # list, numpy
        dtype = none,
        device = none,
        requires_grad = false,
        pin_memory = false      # 是否存于锁页内存
    )
    ```

- 示例

    ```python
    arr = np.ones((3, 3))

    t = torch.tensor(arr, device = "cuda")
    print(t)
    ```

2. 通过 numpy array 来创建 tensor api

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
        out = none,             # 输出张量，就是把这个张量赋值给另一个张量，但这两个张量一样，指的是同一个内存地址
        dtype = none,
        layout = torch.strided, # 内存的布局形式
        device = none,
        requires_grad = false
    )
    ```

- 示例

    ```python
    out_t = torch.tensor([1])
    t = torch.zeros((3, 3), out = out_t)
    print(out_t, t)
    print(id(out_t), id(t), id(t) == id(out_t))
    ```

## tensor 的操作

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

- torch tensor 2 numpy array

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

- numpy array 2 torch tensor

    ```python
    import numpy as np
    
    a = np.ones(5)
    b = torch.from_numpy(a)
    np.add(a, 1, out = a)
    
    print(a)
    print(b)
    ```

##cuda tensor

```python
# let us run this cell only if cuda is available
# we will use `torch.device` objects to move tensors in and out of gpu
x = torch.tensor([1])
if torch.cuda.is_available():
   device = torch.device("cuda")            # a cuda device object

   y = torch.ones_like(x, device = device)  # directly create a tensor on gpu
   
   x = x.to(device)                         # or just use strings `.to("cuda")`
   
   z = x + y
   z.to("cpu", torch.double)                # `.to` can also change dtype together!
```




# PyTorch 数据读取与预处理

## Dataset 和 DataLoader

> - Dataset 保存了数据的样本和标签
> - DataLoader 将 Dataset 封装为可迭代对象, 便于访问样本

### 使用 TorchVision, TorchText, TorchAudio dataset

* 加载相关库

```python
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor
```

* 下载、读取数据

```python
training_data = datasets.FashionMNIST(
    root = "data",
    trian = True,
    download = True,
    transform = ToTensor(),
)

test_data = datasets.FashionMNIST(
    root = "data",
    train = False,
    download = True,
    transform = ToTensor(),
)
```

* 生成数据加载器

```python
batch_size = 64

# create data loaders
train_dataloader = DataLoader(trianing_data, batch_size = batch_size)
test_dataloader = DataLoader(test_data, batch_size = batch_size)

for X, y in test_dataloader:
    print(f"Shape of X [N, C, H, W]: {X.shape}")
    print(f"Shape of y: {y.shape} {y.dtype}")
    break

# Display image and label.
train_features, train_labels = next(iter(train_dataloader))
print(f"Feature batch shape: {train_features.size()}")
print(f"Labels batch shape: {train_labels.size()}")
img = train_features[0].squeeze()
label = train_labels[0]
plt.imshow(img, cmap="gray")
plt.show()
print(f"Label: {label}")
```

### 使用文件创建自定义的 Dataset

> * `Dataset` 类需要实现以下方法
>     - `__init__`
>     - `__len__`
>     - `__getitem__`

* 记载相关库

```python
import os
import pandas as pd
import torchvision.io import read_image
# TODO import torchtext.iop import *
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
```

* 创建 Dataset 类

    - 文件所在目录结构如下 TODO

    ```
    - img_dir
        - 
        - labels.csv
            tshirt1.jpg, 0
            tshirt2.jpg, 0
            ...
            ankleboot999.jpg, 9
    ```

```python
class CustomImage(Dataset):

    def __init__(self, annotations_file, img_dir, transform = None, target_transform = None):
        self.img_labels = pd.read_csv(annotations_file)
        self.img_dir = img_dir
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.img_labels)
    
    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.img_labels.iloc[idx, 0])
        image = read_image(img_path)
        label = self.img_labels.iloc[idx, 1]
        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)
        return image, label
```

* 使用 DataLoader 准备训练数据集

```python
training_data = CustomImage(
    annotations_file = "",
    img_dir = "",
    transform = ToTensor(),
    target_transform = ToTensor(),
)
test_data = CustomImage(
    annotations_file = "",
    img_dir = "",
    transform = ToTensor(),
    target_transform = ToTensor(),
)

train_dataloader = DataLoader(training_data, batch_size = 64, shuffle = True)
test_dataloader = DataLoader(test_data, batch_size = 64, shuffle = True)

# Display image and label.
train_features, train_labels = next(iter(train_dataloader))
print(f"Feature batch shape: {train_features.size()}")
print(f"Labels batch shape: {train_labels.size()}")
img = train_features[0].squeeze()
label = train_labels[0]
plt.imshow(img, cmap="gray")
plt.show()
print(f"Label: {label}")
```

### 2.1.3 torch.utils.data API

* torch.utils
    - torch.utils.benchmark
    - torch.utils.bottleneck
    - torch.utils.checkpoint
    - torch.utils.cpp_extension
    - **torch.utils.data**
    - torch.utils.dlpack
    - torch.utils.mobile_optimizer
    - torch.utils.model_zoo
    - torch.utils.tensorboard





# PyTorch 数据预处理

* torchvision.transforms
* torchtext.transforms
* torchaudio.transforms

## Transforms 示例

```python
import torch
from torchvision import datasets
from torchvision import ToTensor, Lambda

training_data = datasets.FashionMNIST(
    root = "data",
    train = True,
    download = True,
    transform = ToTensor(),
    target_transform = Lambda(
        lambda y: torch.zeros(10, dtype = torch.float)
                       .scatter_(dim = 0, torch.tensor(y), value = 1)
    )
)
test_data = datasets.FashionMNIST(
    root = "data",
    train = False,
    download = True,
    transform = ToTensor(),
    target_transform = Lambda(
        lambda y: torch.zeros(10, dtype = torch.float)
                       .scatter_(dim = 0, torch.tensor(y), value = 1)
    )
)
```

## torchvision transforms

* Scriptable transforms
* Compositions of transforms
* Transforms on PIL Image and torch.*Tensor
* Transforms on PIL Image only
* Transforms on torch.*Tensor only
* Conversion transforms
* Generic transforms
* Automatic Augmentation transforms
* Functional transforms

## torchtext transforms



## torchaudio transforms







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

# Parameters and DataLoaders
input_size = 5
output_size = 2

batch_size = 30
data_size = 100

# Device
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

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


















# PyTorch 动态图、自动求导

- 4.1 计算图
   - 描述运算的有向无环图
      - Tensor 的 is_leaf 属性
      - Tensor 的 grad_fn 属性
- 4.2 PyTorch 动态图机制
   - 动态图与静态图
- 4.3 PyTorch 自动求导机制
   - torch.autograd.backward() 方法自动求取梯度
   - torch.autograd.grad() 方法可以高阶求导
   - note
      - 梯度不自动清零
      - 依赖叶节点的节点, requires_grad 默认为 True
      - 叶节点不能执行原位操作

## 计算图

计算图是用来描述运算的有向无环图。主要有两个因素:节点、边。
其中节点表示数据，如向量、矩阵、张量；而边表示运算，如加减乘除、卷积等。

使用计算图的好处不仅是让计算看起来更加简洁，还有个更大的优势是让梯度求导也变得更加方便。

- 示例:

```python
x = torch.tensor([2.], requires_grad = True)
w = torch.tensor([1.], requires_grad = True)

a = torch.add(w, x)
b = torch.add(w, 1)

y = torch.mul(a, b)

y.backward()
print(w.grad)
```

## Pytorch 动态图机制


## PyTorch 自动求导机制

- package `autograd`
- torch.Tensor
- .requires_grad = True
- .backward()
- .grad
- .detach()
- with torch.no_grad(): pass
- .grad_fn

PyTorch 自动求导机制使用的是 ``torch.autograd.backward`` 方法，功能就是自动求取梯度。

- API:

```python
torch.autograd.backward(
   tensors, 
   gard_tensors = None, 
   retain_graph = None, 
   create_graph = False
)
```

`autograd` 包提供了对所有 Tensor 的自动微分操作

```python
#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
`autograd` package:
-------------------
跟踪 torch.Tensor 上所有的操作:torch.Tensor(requires_grad = True)
自动计算所有的梯度:.backward()
torch.Tensor 上的梯度:.grad
torch.Tensor 是否被跟踪:.requires_grad
停止跟踪 torch.Tensor 上的跟踪历史、未来的跟踪:.detach()

with torch.no_grad():
      pass

Function
.grad_fn
"""

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

# PyTorch Cheat Sheet

## Imports

```python
# General
import torch # root package
from torch.utils.data import Dataset, Dataloader # dataset representation and loading

# Neural Network API
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F
import torch.autograd as autograd
import torch.optim as optim
from torch.jit import script, trace

# Torchscript and JIT
torch.jit.trace()
@script

# ONNX
torch.onnx.export(model, dummy data, xxxx.proto)
model = onnx.load("alexnet.proto")
onnx.checker.check_model(model)
onnx.helper.printable_graph(model.graph)

# Vision
from torchvision import datasets, models, transforms
import torchvision.transforms as transforms

# Distributed Training
import torch.distributed as dist
from torch.multiprocessing import Process
```

## Tensors

```python   
import torch

# Creation
x = torch.randn(*size)
x = torch.[ones|zeros](*size)
x = torch.tensor(L)
y = x.clone()
with torch.no_grad():
requires_grad = True

# Dimensionality
x.size()
x = torch.cat(tensor_seq, dim = 0)
y = x.view(a, b, ...)
y - x.view(-1,a)
y = x.transpose(a, b)
y = x.permute(*dims)
y = x.unsqueeze(dim)
y = x.unsqueeze(dim = 2)
y = x.squeeze()
y = x.squeeze(dim = 1)

# Algebra
ret = A.mm(B)
ret = A.mv(x)
x = x.t()

# GUP Usage
torch.cuda.is_available
x = x.cuda()
x = x.cpu()
if not args.disable_cuda and torch.cuda.is_available():
   args.device = torch.device("cuda")
else:
   args.device = torch.device("cpu")
net.to(device)
x = x.to(device)
```

## Deep Learning

```python
import torch.nn as nn

nn.Linear(m, n)
nn.ConvXd(m, n, s)
nn.MaxPoolXd(s)
nn.BatchNormXd
nn.RNN
nn.LSTM
nn.GRU
nn.Dropout(p = 0.5, inplace = False)
nn.Dropout2d(p = 0.5, inplace = False)
nn.Embedding(num_embeddings, embedding_dim)

# Loss Function
nn.X

# Activation Function
nn.X

# Optimizers
opt = optim.x(model.parameters(), ...)
opt.step()
optim.X

# Learning rate scheduling
scheduler = optim.X(optimizer, ...)
scheduler.step()
optim.lr_scheduler.X
```

## Data Utilities

```python
# Dataset
Dataset
TensorDataset
Concat Dataset

# Dataloaders and DataSamplers
DataLoader(dataset, batch_size = 1, ...)
sampler.Sampler(dataset, ...)
sampler.XSampler where ...
```

