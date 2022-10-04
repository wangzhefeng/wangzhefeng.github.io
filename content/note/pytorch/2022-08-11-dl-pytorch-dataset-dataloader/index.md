---
title: PyTorch 数据管道
author: 王哲峰
date: '2022-08-11'
slug: dl-pytorch-dataset-dataloader
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

- [PyTorch 数据读取构建](#pytorch-数据读取构建)
  - [Dataset 和 DataLoader 原理](#dataset-和-dataloader-原理)
    - [获取一个 batch 数据的步骤](#获取一个-batch-数据的步骤)
    - [Dataset 和 DataLoader 的功能分工](#dataset-和-dataloader-的功能分工)
  - [使用 Dataset 创建数据集](#使用-dataset-创建数据集)
  - [使用 DataLoader 加载数据集](#使用-dataloader-加载数据集)
    - [API](#api)
    - [数据加载顺序和 Sampler](#数据加载顺序和-sampler)
    - [自动内存锁定](#自动内存锁定)
- [PyTorch 内置数据集](#pytorch-内置数据集)
  - [内置数据集](#内置数据集)
  - [使用 TorchVision datasets](#使用-torchvision-datasets)
  - [使用 TorchText datasets](#使用-torchtext-datasets)
  - [使用 TorchAudio datasets](#使用-torchaudio-datasets)
  - [自定义数据集的基本类](#自定义数据集的基本类)
- [使用文件创建自定义 Dataset](#使用文件创建自定义-dataset)
  - [加载相关库](#加载相关库)
  - [文件所在目录结构如下](#文件所在目录结构如下)
  - [创建 Dataset 类](#创建-dataset-类)
  - [构建 DataLoader 训练数据集](#构建-dataloader-训练数据集)
- [PyTorch 数据预处理](#pytorch-数据预处理)
  - [Transforms 示例](#transforms-示例)
  - [torchvision transforms](#torchvision-transforms)
    - [常用转换](#常用转换)
  - [torchtext transforms](#torchtext-transforms)
  - [torchaudio transforms](#torchaudio-transforms)
- [PyTorch 数据管道构建](#pytorch-数据管道构建)
  - [结构化数据管道](#结构化数据管道)
  - [图像数据管道](#图像数据管道)
  - [文本数据管道](#文本数据管道)
- [参考](#参考)
</p></details><p></p>

# PyTorch 数据读取构建

## Dataset 和 DataLoader 原理

### 获取一个 batch 数据的步骤

从一个数据集中获取一个 batch 的数据需要以下步骤，假定数据集的特征和标签分别为张量 `$X$` 和 `$Y$`，
数据集可以表示为 `$(X, Y)$`，假定 batch 大小为 `$m$`:

1. 首先，要确定数据集的长度 `$n$`
    - 结果类似 `$n = 1000$`
2. 然后，从 `$0$` 到 `$n - 1$` 的范围内抽取 `$m$` 个数(batch 的大小)
    - 假定 `m = 4`，拿到的结果是一个列表，类似 `indices = [1, 4, 8, 9]`
3. 从数据集中去取这 `m` 个数对应下标的元素
    - 拿到的结果是一个元组列表，类似 `samples = [(X[1], Y[1]), (X[4], Y[4]), (X[8], Y[8]), (X[9], Y[9])]`
4. 最后，将结果整理成两个张量作为输出
    - 拿到的结果是两个张量，类似 `batch = (features, labels)`，其中 `features = torch.stack([X[1], x[4], X[8], X[9]])`，
      `labels = troch.stack([Y[1], Y[4], Y[8], Y[9]])`

### Dataset 和 DataLoader 的功能分工

上述第一个步骤确定数据集的长度是由 `Dataset` 的 `__len__` 方法实现的。
第二个步骤从 `$0$` 到 `$n-1$` 的范围中抽取出 `$m$` 个数的方法
是由 `DataLoader` 的 `sampler` 和 `batch_sampler` 参数指定的



## 使用 Dataset 创建数据集

`Dataset` 定义了数据集的内容，保存了数据的样本和标签，
它相当于一个类似列表的数据结构，具有确定的长度，能够用索引获取数据集中的元素。
`Dataset` 类还是：

* `torchvision.dataset` 的类
* `torchtext.datasets` 的类
* `torchaudio.datasets` 的类
* 自定义 `Dataset` 对象的父类

## 使用 DataLoader 加载数据集

PyTorch 数据加载工具里面最核心的类是: `torch.utils.data.DataLoader`，
`DataLoader` 将 `Dataset` 封装为可迭代对象, 便于访问样本。
`DataLoader` 定义了按 batch 加载数据集的方法，它是一个实现了 `__iter__` 方法的可迭代对象，
每次迭代输出一个 batch 的数据。

`DataLoader` 对象功能如下：

* map-style、可迭代(iterable-style)数据集
* 自定义数据加载顺序，控制 batch 中元素的采样方法
* 自动控制 batch 的大小
* 将 batch 结果整理成模型所需输入形式的方法
* 单进程、多进程数据加载
* 自动内存锁定(pinning)

### API

```python
DataLoader(
    dataset,  # Dataset 对象, tensorvision,torchtext,torchaudio,自定义 Dataset,IterableDataset
    batch_size = 1,  # 批次大小
    shuffle = False,  # 是否乱序
    sampler = None,  # 样本采样函数，一般无需设置
    batch_sampler = None,  # 批次采样函数，一般无需设置
    num_workers = 0,  # 使用多进程读取数据，设置的进程数
    collate_fn = None,  # 整理一个批次数据的函数
    pin_memory = False,  # 是否设置为锁业内存。默认为 False，锁业内存不会使用虚拟内存(硬盘)，从锁业内存拷贝到 GPU 上的速度会更快
    drop_last = False,  # 是否丢弃最后一个样本数量不足 batch_size 批次数据
    timeout = 0,  # 加载一个数据批次的最长等待时间，一般无需设置
    worker_init_fn = None,  # 每个 worker 中的 dataset 的初始化函数，常用于 IterableDateset，一般不使用
    multiprocessing_context = None,  # 多进程数据加载
)
``` 

一般情况下，会配置 `dataset`、`batch_size`、`shuffle`、`num_workers`、`pin_memory`、`drop_last` 这六个参数。
有时候，对于一些复杂的数据集，还需要自定义 `collate_fn` 函数，其他参数一般使用默认值即可

DataLoader 除了可以加载 `torch.utils.data.Dataset` 外，还能够加载另一种数据集 `torch.utils.data.IterableDataset`。
和 `Dataset` 数据集相当于一种列表结构不同，`IterableDataset` 相当于一种迭代器结构，它更加复杂，一般较少使用

### 数据加载顺序和 Sampler

* 对于可迭代(iterable-style)数据集，数据加载顺序完全由用户定义的方式迭代控制。
  这允许更容易实现块读取和动态批量大小
* 对于 map-style 的情况，`torch.utils.data.Sampler` 类用于指定数据加载中使用 key/value 的顺序。

### 自动内存锁定

对于数据加载，给 `DataLoader` 传递 `pin_memory = True` 将自动将获取的 tensor 放入固定内存中，
从而更快地将数据传输到支持 CUDA 的 GPU

```python
class SimpleCustomBatch:
    
    def __init__(self, data):
        transposed_data = list(zip(*data))
        self.inp = torch.stack(transposed_data[0], 0)
        self.tgt = torch.stack(transposed_data[1], 0)
    
    def pin_memory(self):
        self.inp = self.inp.pin_memory()
        self.tgt = self.tgt.pin_memory()
        return self
    
def collate_wrapper(batch):
    return SimpleCustomBatch(batch)

# data
inps = torch.arange(10 * 5, dtype = torch.float32).view(10, 5)
tgts = torch.arange(10 * 5, dtype = torch.float32).view(10, 5)
dataset = TensorDataset(inps, tgts)

# data loader
loader = DataLoader(
    dataset, 
    batch_size = 2,
    collate_fn = collate_wrapper,
    pin_memory = True,
)

for batch_ndx, sample in enumerate(loader):
    print(sample.inp.is_pinned())
    print(sample.tgt.is_pinned())
```

# PyTorch 内置数据集

Torchvision 在模块 `torchvision.datasets` 中提供了很多内置数据集，
以及一些用于构建自定义数据集的使用功能类

## 内置数据集

内置的数据集都是 `torch.utils.data.Dataset` 的子类，
所以它们都具有 `__getitem__` 和 `__len__` 实现方法，
因此，它们都可以被传递给可以使用 `torch.multiprocessing` 
多进程的 `torch.utils.data.DataLoader` 并行加载多个样本的

这些数据集都有类似的 API，它们都有两个共同的转换参数:

* `transform()`
* `target_transform()`

## 使用 TorchVision datasets

* 加载相关库

```python
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor
```

* 下载、读取数据

```python
# Dataset 实例对象
training_data = datasets.FashionMNIST(
    root = "data",
    trian = True,
    download = True,
    transform = ToTensor(),
    target_transform = ToTensor(),
)

# Dataset 实例对象
test_data = datasets.FashionMNIST(
    root = "data",
    train = False,
    download = True,
    transform = ToTensor(),
    target_transform = ToTensor(),
)
```

* 生成数据加载器

```python
batch_size = 64

# create data loaders
train_dataloader = DataLoader(trianing_data, batch_size = batch_size, shuffle = True)
test_dataloader = DataLoader(test_data, batch_size = batch_size, shuffle = True)

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

```python
import torch
from torchvision import datasets

imagenet_data = datasets.ImageNet("path/to/imagenet_root/") 
data_loader = torch.utils.data.DataLoader(
    imagenet_data,
    batch_size = 4,
    shuffle = True,
    num_workers = args.nThreads,
)
```

## 使用 TorchText datasets

```python
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torchtext import datasets
from torchtext.transforms import *
```

```python
from torchtext.datasets import IMDB

train_iter = IMDB(split = "train")

def tokenize(label, line):
    return line.split()

tokens = []
for label, line in train_iter:
    tokens += tokenize(label, line)
```


## 使用 TorchAudio datasets

```python
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torchaudio import datasets
from torchaudio.transforms import *
```

```python
import torch
from torchaudio import datasets

yesno_data = datasets.YESNO(".", download = True)
data_loader = torch.utils.data.DataLoader(
    yesno_data,
    batch_size = 1,
    shuffle = True,
    num_workers = args.nThreads,
)
```

## 自定义数据集的基本类

* `DatasetFolder(root, loader, Any],...)`
* `ImageFolder(root, transform,...)`
* `VisionDataset(root, transforms, transform,...)`







# 使用文件创建自定义 Dataset

在绝大部分情况下，用户只需实现 `Dataset` 类的以下方法，即可轻松构建自己的数据集，
并用默认数据管道进行加载:

* `__init__`
* `__len__`
* `__getitem__`

## 加载相关库

```python
import os
import pandas as pd

import torchvision.io import read_image
# TODO import torchtext.iop import *
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
```

## 文件所在目录结构如下

```
- img_dir
    - 
    - labels.csv
        tshirt1.jpg, 0
        tshirt2.jpg, 0
        ...
        ankleboot999.jpg, 9
```

## 创建 Dataset 类

```python
class CustomImageDataset(Dataset):

    def __init__(self, 
                 annotations_file, 
                 img_dir, 
                 transform = None, 
                 target_transform = None):
        self.img_labels = pd.read_csv(annotations_file)
        self.img_dir = img_dir
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        """
        return the number of samples in dataset
        """
        return len(self.img_labels)
    
    def __getitem__(self, idx):
        """
        loads and returns a sample from 
        the dataset at the given index `idx`
        """
        # image path
        img_path = os.path.join(
            self.img_dir, 
            self.img_labels.iloc[idx, 0]
        )
        # image tensor
        image = read_image(img_path)
        # image label
        label = self.img_labels.iloc[idx, 1]
        # image transform
        if self.transform:
            image = self.transform(image)
        # image label transform
        if self.target_transform:
            label = self.target_transform(label)

        return image, label
```

## 构建 DataLoader 训练数据集

```python
training_data = CustomImageDataset(
    annotations_file = "",
    img_dir = "",
    transform = ToTensor(),
    target_transform = ToTensor(),
)
test_data = CustomImageDataset(
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

# PyTorch 数据预处理

可以使用 `transforms` 对数据集进行转换操作，使得数据集可以作为机器学习算法可以使用的形式

* `torchvision.transforms`
* `torchtext.transforms`
* `torchaudio.transforms`

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
    target_transform = Lambda(lambda y: 
        torch.zeros(10, dtype = torch.float)
             .scatter_(dim = 0, torch.tensor(y), value = 1)
    ),
)
test_data = datasets.FashionMNIST(
    root = "data",
    train = False,
    download = True,
    transform = ToTensor(),
    target_transform = Lambda(lambda y: 
        torch.zeros(10, dtype = torch.float)
             .scatter_(dim = 0, torch.tensor(y), value = 1)
    ),
)
```

## torchvision transforms

所有的 TorchVision datasets 都有两个接受包含转换逻辑的可调用对象的参数:

* transform
    - 修改特征
* target_transform
    - 修改标签

`torchvision.transform` 模块提供了多个常用转换

* ToTensor()
    - 将 PIL 格式图像或 Numpy `ndarra` 转换为 `FloatTensor`
    - 将图像的像素强度值(pixel intensity values)缩放在 `[0, 1]` 范围内
* Lambda 换换
    - 可以应用任何用户自定义的 lambda 函数

### 常用转换

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








# PyTorch 数据管道构建

## 结构化数据管道

## 图像数据管道

PyTorch 中构建图像数据管道通常有两种方法:

* 使用 `torchvision` 中的 `datasets.ImageFolder` 来读取图像，然后用  `DataLoader` 进行加载
* 通过继承 `torch.utils.data.Dataset` 实现用户自定义读取逻辑，然后用 `DataLoader` 来并行加载
    - 这种方法是读取用户自定义数据集的通用方法，即可以读取图像数据集，也可以读取文本数据集

## 文本数据管道

# 参考

- [DataLoader & Dataset](https://pytorch.org/docs/stable/data.html#)

