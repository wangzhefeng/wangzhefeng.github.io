---
title: PyTorch 数据管道
author: 王哲峰
date: '2022-08-11'
slug: dl-pytorch-data-pipeline
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
img {
    pointer-events: none;
}
</style>

<details><summary>目录</summary><p>

- [PyTorch 数据管道](#pytorch-数据管道)
  - [Dataset 和 DataLoader 原理](#dataset-和-dataloader-原理)
    - [获取 batch 数据](#获取-batch-数据)
    - [Dataset 和 DataLoader 功能分工](#dataset-和-dataloader-功能分工)
    - [DataLoader 内部调用方式](#dataloader-内部调用方式)
    - [Dataset 和 DataLoader 核心源码](#dataset-和-dataloader-核心源码)
  - [使用 Dataset 创建数据集](#使用-dataset-创建数据集)
    - [根据 Tensor 创建 Dataset](#根据-tensor-创建-dataset)
    - [使用图片目录创建 Dataset](#使用图片目录创建-dataset)
    - [使用文件创建自定义 Dataset](#使用文件创建自定义-dataset)
    - [创建自定义 Dataset](#创建自定义-dataset)
  - [使用 DataLoader 加载数据集](#使用-dataloader-加载数据集)
    - [数据加载顺序和 Sampler](#数据加载顺序和-sampler)
    - [自动内存锁定](#自动内存锁定)
- [PyTorch 内置数据集](#pytorch-内置数据集)
  - [内置数据集](#内置数据集)
  - [TorchVision datasets](#torchvision-datasets)
  - [TorchText datasets](#torchtext-datasets)
  - [TorchAudio datasets](#torchaudio-datasets)
  - [自定义数据集的基本类](#自定义数据集的基本类)
- [PyTorch 数据预处理](#pytorch-数据预处理)
  - [torchvision transforms](#torchvision-transforms)
    - [常用转换](#常用转换)
  - [torchtext transforms](#torchtext-transforms)
  - [torchaudio transforms](#torchaudio-transforms)
- [PyTorch 数据管道构建](#pytorch-数据管道构建)
  - [结构化数据管道](#结构化数据管道)
  - [图像数据管道](#图像数据管道)
  - [语音数据管道](#语音数据管道)
  - [视屏数据管道](#视屏数据管道)
  - [文本数据管道](#文本数据管道)
- [参考](#参考)
</p></details><p></p>

# PyTorch 数据管道

PyTorch 通常使用 `torch.utils.data.Dataset` 和 `torch.utils.data.DataLoader` 这两个工具类来构建数据通道

* `Dataset` 定义了数据集的内容，保存了数据的样本和标签，它相当于一个类似列表的数据结构
    - 具有确定的长度(`__len__`)
    - 能够用索引获取数据集中的元素(`__getitem__`)
* `DataLoader` 定义了按 batch 加载数据集的方法，它是一个实现了 `__iter__` 方法的可迭代对象，
  每次迭代输出一个 batch 的数据。`DataLoader` 能够控制 batch 的大小，batch 中元素的采样方法，
  以及将 batch 结果整理成模型所需输入形式的方法，并且能够使用多进程读取数据

另外，在绝大部分情况下，用户只需实现 `Dataset` 的 `__len__` 方法和 `__getitem__` 方法，
就可以轻松构建自己的数据集，并用默认数据管道进行加载

## Dataset 和 DataLoader 原理

* `Dataset`
    - 方法
        - `__len__`
        - `__getitem__`
* `DataLoader`
    - 方法
        - `__next__`
        - `__iter__`
    - 参数
        - `dataset`
        - `batch_size`
        - `collate_fn`
        - `shuffle`：`sampler`、`RandomSampler`、`SequentialSampler`
        - `drop_last`：`batch_sampler`、`BatchSampler`

### 获取 batch 数据

假定数据集的特征和标签分别为张量 `$X$` 和 `$y$`，数据集可以表示为 `$(X, y)$`，假定 batch 大小为 `$m$`。
从一个数据集中获取一个 batch 的数据需要以下步骤:

1. 首先，要确定数据集的长度 `$n$`。假设 `$n=1000$`
2. 然后，从 `$[0, n-1]$` 的范围内抽取 `$m$` 个数(batch 的大小)。假定 `m =4`，拿到的结果是一个列表

    ```
    indices = [1, 4, 8, 9]
    ```

3. 从数据集 `$(X, y)$` 中去取这 `m` 个数对应下标的元素。拿到的结果是一个元组列表

    ```
    samples = [
        (X[1], y[1]), 
        (X[4], y[4]), 
        (X[8], y[8]), 
        (X[9], y[9])
    ]
    ```

4. 最后，将结果整理成两个张量作为输出。拿到的结果是两个张量，类似 `batch = (features, labels)`，其中：

    ```
    features = torch.stack([X[1], x[4], X[8], X[9]])
    labels = troch.stack([Y[1], Y[4], Y[8], Y[9]])
    ```

### Dataset 和 DataLoader 功能分工

从上述获取一个 batch 数据的步骤分析 `Dataset` 和 `DataLoader` 的功能分工：

1. 第一个步骤确定数据集的长度是由 `Dataset` 的 `__len__` 方法实现的
2. 第二个步骤从 `$[0, n-1]$` 的范围中抽取出 `$m$` 个数的方法是由 `DataLoader` 的 `sampler` 和 `batch_sampler` 参数指定的
    - `sampler` 参数制定单个元素的抽样方法，一般无需用户设置，
      程序默认在 `DataLoader` 的参数 `shuffle = True` 时采用随机抽样，
      `shuffle = False` 时采用顺序抽样
    - `batch_sampler` 参数将多个抽样的元素整理成一个列表，一般无需用户设置，
      默认方法在 `DataLoader` 的参数 `drop_last = True` 时丢弃数据集最后一个长度不能被 batch 大小整除的批次，
      在 `drop_last = False` 时保留最后一个批次
3. 第三个步骤的核心逻辑根据下标取数据集中的元素，是由 `Dataset` 的 `__getitem__` 方法实现的
4. 第四个步骤的逻辑由 `DataLoader` 的参数 `collate_fn` 指定，一般情况下也无需用户设置

### DataLoader 内部调用方式

Dataset 和 DataLoader 使用方式：

```python
import torch
from torch.utils.data import TensorDataset, DataLoader
from torch.utils.data import Sampler, RandomSampler, BatchSampler


dataset = TensorDataset(
    torch.randn(1000, 3),
    torch.randint(low = 0, high = 2, size = (1000,)).float(),
)
dataloader = DataLoader(
    dataset,
    batch_size = 4,
    drop_last = False,
)

features, labels = next(iter(dataloader))
print(f"features = {features}")
print(f"labels = {labels}")
```

```
features =  tensor([[-0.3979,  0.4728, -0.9796],
                    [-1.0995,  0.7045,  0.7593],
                    [-0.9703, -0.6259, -0.2886],
                    [-1.1529, -0.7042, -0.8151]])
labels =  tensor([1., 0., 0., 0.])
```

* step1：确定数据集长度(`Dataset` 的 `__len__` 方法实现)

```python
dataset = TensorDataset(
    torch.randn(1000, 3),
    torch.randint(low = 0, high = 2, size = (1000,)).float(),
)
print("n = ", len(dataset))  # len(ds) 等价于 dataset.__len__()
```

```
n = 1000
```

* step2：确定抽样索引(`DataLoader` 中的 `Sampler` 和 `BatchSampler` 实现)

```python
sampler = RandomSampler(data_source = dataset)
batch_sampler = BatchSampler(
    sampler = sampler, 
    batch_size = 4, 
    drop_last = False
)
for idxs in batch_sampler:
    indices = idxs
    break 
print("indices = ", indices)
```

```
indices =  [776, 144, 127, 140]
```

* step3：取出一批样本 batch (`Dataset` 的 `__getitem__` 方法实现)

```python
batch = [dataset[i] for i in  indices]  #  dataset[i] 等价于 dataset.__getitem__(i)
print("batch = ", batch)
```

```
batch =  [(tensor([-0.1744, -1.1102,  0.3292]), tensor(0.)), 
          (tensor([1.0112, 1.3905, 1.7684]), tensor(0.)), 
          (tensor([ 0.6682,  0.6509, -1.1334]), tensor(0.)), 
          (tensor([-0.2228,  0.7622,  0.0318]), tensor(1.))]
``` 

* step4: 整理成 features 和 labels (`DataLoader` 的 `collate_fn` 方法实现)

```python
def collate_fn(batch):
    features = torch.stack([sample[0] for sample in batch])
    labels = torch.stack([sample[1] for sample in batch])
    return features, labels 

features,labels = collate_fn(batch)
print("features = ", features)
print("labels = ", labels)
```

```
features =  tensor([[-0.1744, -1.1102,  0.3292],
        [ 1.0112,  1.3905,  1.7684],
        [ 0.6682,  0.6509, -1.1334],
        [-0.2228,  0.7622,  0.0318]])
labels =  tensor([0., 0., 0., 1.])
```

### Dataset 和 DataLoader 核心源码

以下是 `Dataset` 和 `DataLoader` 的核心源码，省略了为了提升性能而引入的诸如多进程读取数据相关的代码

```python
import torch 

class Dataset(object):
    def __init__(self):
        pass
    
    def __len__(self):
        raise NotImplementedError
        
    def __getitem__(self, index):
        raise NotImplementedError


class DataLoader(object):
    def __init__(self, 
                 dataset, 
                 batch_size, 
                 collate_fn = None, 
                 shuffle = True, 
                 drop_last = False):
        self.dataset = dataset
        self.collate_fn = collate_fn
        self.sampler = torch.utils.data.RandomSampler if shuffle \ 
            else torch.utils.data.SequentialSampler
        self.batch_sampler = torch.utils.data.BatchSampler
        self.sample_iter = self.batch_sampler(
            self.sampler(self.dataset),
            batch_size = batch_size,
            drop_last = drop_last
        )
        self.collate_fn = collate_fn if collate_fn is not None \
            else torch.utils.data._utils.collate.default_collate
     
    def __next__(self):
        indices = next(iter(self.sample_iter))
        batch = self.collate_fn([self.dataset[i] for i in indices])
        return batch
    
    def __iter__(self):
        return self
```

测试：

```python
class ToyDataset(Dataset):

    def __init__(self, X, Y):
        self.X = X
        self.Y = Y 

    def __len__(self):
        return len(self.X)

    def __getitem__(self, index):
        return self.X[index], self.Y[index]

# data
X = torch.randn(1000, 3)
Y = torch.randint(low = 0, high = 2, size = (1000,)).float()
# dataset
dataset = ToyDataset(X, Y)
# dataloader
dataloader = DataLoader(dataset, batch_size = 4, drop_last = False)

# test
features, labels = next(iter(dl))
print("features = ", features )
print("labels = ", labels )  
```

```
features =  tensor([[ 0.6718, -0.5819,  0.9362],
                    [-0.4208, -0.1517,  0.3838],
                    [ 2.1848, -1.2617,  0.7580],
                    [ 0.1418, -1.6424,  0.3673]])
labels =  tensor([0., 1., 1., 0.])
```

## 使用 Dataset 创建数据集

`Dataset` 定义了数据集的内容，保存了数据的样本和标签，它相当于一个类似列表的数据结构，
具有确定的长度(`__len__`)，能够用索引获取数据集中的元素(`__getitem__`)。
`Dataset` 类还是：

* `torchvision.dataset` 的父类
* `torchtext.datasets` 的父类
* `torchaudio.datasets` 的父类
* 自定义 `Dataset` 对象的父类

`Dataset` 创建数据集常用的方法有：

1. 使用 `torch.utils.data.TensorDataset` 根据 Tensor 创建数据集。
   Numpy 的 `array`、Pandas 的 `DataFrame` 需要先转换成 `torch.Tensor`
2. 使用 `torchvision.datasets.ImageFolder` 根据图片目录创建图片数据集
3. 集成 `torch.utils.data.Dataset` 创建自定义数据集，通过实现以下方法
    - `__init__`
    - `__len__`
    - `__getitem__`

此外，还可以通过：

* `torch.utils.data.random_split` 将一个数据集分割成多份，常用于分割数据集，验证集和测试集
* 调用 `Dataset` 的加法运算符 `+` 将多个数据集合并成一个数据集

### 根据 Tensor 创建 Dataset

使用 `torch.utils.data.TensorDataset` 根据 Tensor 创建数据集。
Numpy 的 `array`、Pandas 的 `DataFrame` 需要先转换成 `torch.Tensor`

```python
import numpy as np
from sklearn import datasets
import torch
from torch.utils.data import (
    TensorDataset, 
    Dataset, 
    DataLoader, 
    random_split
)


# data
iris = datasets.load_iris()

# dataset
dataset_iris = TensorDataset(
    torch.tensor(iris.data),
    torch.tensor(iris.target),
)

# train and test dataset split
num_train = int(len(dataset_iris) * 0.8)
num_valid = len(dataset_iris) - num_train
train_dataset, valid_dataset = random_split(dataset_iris, [num_train, num_valid])

# dataloader
train_dataloader = DataLoader(
    train_dataset, 
    batch_size = 8,
    shuffle = True,
)
valid_dataloader = DataLoader(
    valid_dataset,
    batch_size = 8,
    shuffle = False,
)

# test
for features, labels in train_dataloader:
    print(features, labels)
    break

# 演示加法运算符 `+` 的合并作用
dataset_iris = train_dataset + valid_dataset
print(f"len(train_dataset) = {len(train_dataset)}")
print(f"len(valid_dataset) = {len(valid_dataset)}")
print(f"len(train_dataset+valid_dataset) = {len(dataset_iris)}")
print((type(dataset_iris)))
```

### 使用图片目录创建 Dataset

使用 `torchvision.datasets.ImageFolder` 根据图片目录创建图片数据集。
图片目录形式如下：

```
- ./cifar2
    - train
        - img1.png
        - img2.png
    - test
        - img1.png
        - img2.png
```

```python
import os
import sys

import numpy as np
from PIL import Image
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision import datasets


"""
# 图片
img = Image.open("./data/cat.jpeg")
# 随机数值翻转
transforms.RandomVerticalFlip()(img)
# 随机旋转
transforms.RandomRotation(45)(img)
"""


# transforms
transform_train = transforms.Compose([
    transforms.RandomHorizontalFlip(),  # 随机水平翻转
    transforms.RandomVerticalFlip(),  # 随机垂直翻转
    transforms.RandomRotation(45),  # 随机在 45 角度内旋转
    transforms.ToTensor(),  # 转换成张量
])
transform_valid = transforms.Compose([
    transforms.ToTensor()
])
def transform_label(x):
    return torch.tensor([x]).float()

# dataset
train_dataset = datasets.ImageFolder(
    root = "./cifar2/train/",
    train = True,
    transform = transform_train,
    target_transform = transform_label,
    download = False,
)
valid_dataset = datasets.ImageFolder(
    root = "./cifar2/test/",
    train = False,
    transform = transform_valid,
    target_transform = transform_label,
    download = False,
)
print(train_dataset.class_to_idx)

# dataloader
train_dataloader = DataLoader(
    train_dataset, 
    batch_size = 50, 
    shuffle = True
)
valid_dataloader = DataLoader(
    valid_dataset, 
    batch_size = 50, 
    shuffle = False
)

# test
for features, labels in train_dataloader:
    print(features.shape)
    print(labels.shape)
    break
```

### 使用文件创建自定义 Dataset

在绝大部分情况下，用户只需实现 `Dataset` 类的以下方法，即可轻松构建自己的数据集，
并用默认数据管道进行加载:

* `__init__`
* `__len__`
* `__getitem__`

文件所在目录结构如下

```
- img_dir
    - tshirt1.jpg
    - tshirt2.jpg
    ...
    - ankleboot999.jpg
    - labels.csv
        tshirt1.jpg, 0
        tshirt2.jpg, 0
        ...
        ankleboot999.jpg, 9
```


```python
import os
import sys
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.io import read_image


# global variable
LOGGING_LABEL = __file__.split('/')[-1][:-3]


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
        # # image tensor
        img_path = os.path.join(self.img_dir, self.img_labels.iloc[idx, 0])  # image path
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


# dataset
train_dataset = CustomImageDataset(
    annotations_file = "",
    img_dir = "",
    transform = transforms.ToTensor(),
    target_transform = transforms.ToTensor(),
)
test_dataset = CustomImageDataset(
    annotations_file = "",
    img_dir = "",
    transform = transforms.ToTensor(),
    target_transform = transforms.ToTensor(),   
)

# dataloader
train_dataloader = DataLoader(
    train_dataset, 
    batch_size = 64, 
    shuffle = True,
)
test_dataloader = DataLoader(
    test_dataset, 
    batch_size = 64, 
    shuffle = False,
)

# ------------------------------
# test
# ------------------------------
# test data
train_features, train_labels = next(iter(train_dataloader))
print(f"Feature batch shape: {train_features.size()}")
print(f"Labels batch shape: {train_labels.size()}")
# test plot
img = train_features[0].squeeze()
label = train_labels[0]
plt.imshow(img, cmap="gray")
plt.show()
print(f"Label: {label}")
```

### 创建自定义 Dataset

```python
from pathlib import Path
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms


train_dir = "./cifar2/train/"
test_dir = "./cifar2/test/"


class Cifar2Dataset(Dataset):

    def __init__(self, imgs_dir, img_transform):
        self.files = list(Path(imgs_dir).rglob("*.jpg"))
        self.transform = img_transform

    def __len__(self):
        return len(self.files)

    def __getitem__(self, i):
        # features
        file_i = str(self.files[i])
        img = Image.open(file_i)
        tensor = self.transform(img)
        # labels
        label = torch.tensor([1.0]) if "1_automobile" in file_i else torch.tensor([0.0])

        return tensor, label


# transforms
transform_train = transforms.Compose([
    transforms.RandomHorizontalFlip(),  # 随机水平翻转
    transforms.RandomVerticalFlip(),  # 随机垂直翻转
    transforms.RandomRotation(45),  # 随机在 45 角度内旋转
    transforms.ToTensor(),  # 转换成张量
])
transform_valid = transforms.Compose([
    transforms.ToTensor()
])

# dataset
train_dataset = Cifar2Dataset(
    train_dir, 
    transform_train
)
valid_dataset = Cifar2Dataset(
    test_dir, 
    transform_valid
)

# dataloader
train_dataloader = DataLoader(
    train_dataset, 
    batch_size = 50, 
    shuffle = True,
)
valid_dataloader = DataLoader(
    valid_dataset, 
    batch_size = 50, 
    shuffle = True,
)

# test
for features, labels in train_dataloader:
    print(features.shape)
    print(labels.shape)
    break
```

## 使用 DataLoader 加载数据集

PyTorch 数据加载工具里面最核心的类是：`torch.utils.data.DataLoader`，`DataLoader` 将 `Dataset` 封装为可迭代对象，便于访问样本

`DataLoader` 定义了按 batch 加载数据集的方法，它是一个实现了 `__iter__` 方法的可迭代对象，每次迭代输出一个 batch 的数据

`DataLoader` 能够控制 batch 的大小，batch 中元素的采样方法，
以及将 batch 结果整理成模型所需输入形式的方法，并且能够使用多进程读取数据

`DataLoader` 对象功能如下：

* map-style、可迭代(iterable-style)数据集
* 自定义数据加载顺序，控制 batch 中元素的采样方法
* 自动控制 batch 的大小
* 将 batch 结果整理成模型所需输入形式的方法
* 单进程、多进程数据加载
* 自动内存锁定(pinning)

`DataLoader` API：

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

## 内置数据集

PyTorch 内置的数据集都是 `torch.utils.data.Dataset` 的子类，
所以它们都具有 `__getitem__` 和 `__len__` 实现方法，
因此，它们都可以被传递给可以使用 `torch.multiprocessing` 
多进程的 `torch.utils.data.DataLoader` 并行加载多个样本

这些数据集都有类似的 API，它们都有两个共同的转换参数:

* `transform`
* `target_transform`

## TorchVision datasets

```python
from torchvision import datasets
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

# dataset
imagenet_data = datasets.ImageNet(
    root = "path/to/imagenet_root/",
    trian = True,
    download = True,
    transform = transfroms.ToTensor(),
    target_transform = transforms.ToTensor(),
) 
data_loader = DataLoader(
    imagenet_data,
    batch_size = 4,
    shuffle = True,
    num_workers = args.nThreads,
)
```

## TorchText datasets

```python
from torchtext import datasets
from torch.utils.data import Dataset, DataLoader
from torchtext.transforms import *

# dataset
train_iter = datasets.IMDB(split = "train")

def tokenize(label, line):
    return line.split()

tokens = []
for label, line in train_iter:
    tokens += tokenize(label, line)
```

## TorchAudio datasets

```python
from torchaudio import datasets
from torch.utils.data import Dataset, DataLoader
from torchaudio.transforms import *

yesno_data = datasets.YESNO(".", download = True)
data_loader = DataLoader(
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

# PyTorch 数据预处理

可以使用 `transforms` 对数据集进行转换操作，使得数据集可以作为机器学习算法可以使用的形式

* `torchvision.transforms`
* `torchtext.transforms`
* `torchaudio.transforms`

## torchvision transforms

> torchvision.transforms

所有的 torchvision datasets 都有两个接受包含转换逻辑的可调用对象的参数:

* transform
    - 修改特征
* target_transform
    - 修改标签

大部分 transform 同时接受 PIL 图像和 tensor 图像，但是也有一些 tansform 只接受 PIL 图像，或只接受 tensor 图像

* PIL
* tensor

transform 接受 tensor 图像或批量 tensor 图像

* tensor 图像的 shape 格式是 `(C, H, W)`
* 批量 tensor 图像的 shape 格式是 `(B, C, H, W)`

`torchvision.transform` 模块提供了多个常用转换

* ToTensor()
    - 将 PIL 格式图像或 Numpy `ndarra` 转换为 `FloatTensor`
    - 将图像的像素强度值(pixel intensity values)缩放在 `[0, 1]` 范围内
* Lambda 换换
    - 可以应用任何用户自定义的 lambda 函数
    - `scatter_`: 在标签给定的索引上设置 `value`

### 常用转换

* Scriptable transforms
    - `torch.nn.Sequential`
    - `torch.jit.script`
* Compositions of transforms
    - `Compose`: 将多个 transform 串联起来
* Transforms on PIL Image and `torch.*Tensor`
* Transforms on PIL Image only
    - `RandomChoice`
    - `RandomOrder`
* Transforms on `torch.*Tensor` only
    - `LinearTransformation`
    - `Normalize`
    - `RandomErasing`
    - `ConvertImageDtype`
* Conversion transforms
    - `ToPILImage`: tensor/ndarray -> PIL Image
    - `ToTensor`: PIL Image/numpy.ndarray -> tensor
    - `PILToTensor`: PIL Image -> tensor
* Generic transforms
    - `Lambda`
* Automatic Augmentation transforms
    - `AutoAugmentPolicy`
    - `AutoAgument`
    - `RandAugment`
    - `TrivialAugmentWide`
    - `AugMix`
* Functional transforms
    - 函数式转换提供了对转换管道的细粒度控制。与上述转换相反，
      函数式转换不包含用于其参数的随机数生成器。
      这意味着必须指定/生成所有参数，但函数转换将提供跨调用的可重现结果
    - `torchvision.transform.functional`

## torchtext transforms

## torchaudio transforms


# PyTorch 数据管道构建

## 结构化数据管道

## 图像数据管道

PyTorch 中构建图像数据管道通常有两种方法:

* 使用 `torchvision` 中的 `datasets.ImageFolder` 来读取图像，然后用  `DataLoader` 进行加载
* 通过继承 `torch.utils.data.Dataset` 实现用户自定义读取逻辑，然后用 `DataLoader` 来并行加载
    - 这种方法是读取用户自定义数据集的通用方法，即可以读取图像数据集，也可以读取文本数据集

## 语音数据管道

## 视屏数据管道

## 文本数据管道

# 参考

* [DataLoader & Dataset API](https://pytorch.org/docs/stable/data.html#)
