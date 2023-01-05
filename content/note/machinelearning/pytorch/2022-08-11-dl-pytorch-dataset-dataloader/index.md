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
    - [Dataset 和 DataLoader 的一般使用方式](#dataset-和-dataloader-的一般使用方式)
    - [DataLoader 内部调用方式步骤拆解](#dataloader-内部调用方式步骤拆解)
    - [Dataset 和 DataLoader 的核心源码](#dataset-和-dataloader-的核心源码)
  - [使用 Dataset 创建数据集](#使用-dataset-创建数据集)
    - [根据 Tensor 创建数据集](#根据-tensor-创建数据集)
    - [根据图片目录创建图片数据集](#根据图片目录创建图片数据集)
    - [使用文件创建自定义 Dataset](#使用文件创建自定义-dataset)
    - [创建自定义数据集](#创建自定义数据集)
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
- [PyTorch 数据预处理](#pytorch-数据预处理)
  - [Transforms 示例](#transforms-示例)
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


PyTorch 通常使用 `Dataset` 和 `DataLoader` 这两个工具类来构建数据通道

* `Dataset` 定义了数据集的内容，保存了数据的样本和标签，
  它相当于一个类似列表的数据结构，具有确定的长度，能够用索引获取数据集中的元素
* `DataLoader` 定义了按 batch 加载数据集的方法，它是一个实现了 `__iter__` 方法的可迭代对象，
  每次迭代输出一个 batch 的数据。`DataLoader` 能够控制 batch 的大小，batch 中元素的采样方法，
  以及将 batch 结果整理成模型所需输入形式的方法，并且能够使用多进程读取数据

另外，在绝大部分情况下，用户只需实现 `Dataset` 的 `__len__` 方法和 `__getitem__` 方法，
就可以轻松构建自己的数据集，并用默认数据管道进行加载

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
    - 拿到的结果是两个张量，类似 `batch = (features, labels)`，其中：
        - `features = torch.stack([X[1], x[4], X[8], X[9]])`
        - `labels = troch.stack([Y[1], Y[4], Y[8], Y[9]])`

### Dataset 和 DataLoader 的功能分工

从上述获取一个 batch 数据的步骤分析 `Dataset` 和 `DataLoader` 的功能分工：

* 第一个步骤确定数据集的长度是由 `Dataset` 的 `__len__` 方法实现的。
* 第二个步骤从 `$0$` 到 `$n-1$` 的范围中抽取出 `$m$` 个数的方法
  是由 `DataLoader` 的 `sampler` 和 `batch_sampler` 参数指定的
    - `sampler` 参数制定单个元素的抽样方法，一般无需用户设置，
      程序默认在 `DataLoader` 的参数 `shuffle = True` 时采用随机抽样，`shuffle = False` 时采用顺序抽样
    - `batch_sampler` 参数将多个抽样的元素整理成一个列表，一般无需用户设置，
      默认方法在 `DataLoader` 的参数 `drop_last = True` 时丢弃数据集最后一个长度不能被 batch 大小整除的批次，
      在 `drop_last = False` 时保留最后一个批次
* 第三个步骤的核心逻辑根据下标取数据集中的元素，是由 `Dataset` 的 `__getitem__` 方法实现的
* 第四个步骤的逻辑由 `DataLoader` 的参数 `collate_fn` 指定，一般情况下也无需用户设置

### Dataset 和 DataLoader 的一般使用方式

```python
import torch
from torch.utils.data import TensorDataset, Dataset, DataLoader
from torch.utils.data import RandomSampler, BatchSampler

ds = TensorDataset(
    torch.randn(1000, 3),
    torch.randint(low = 0, high = 2, size = (1000,)).float(),
)
dl = DataLoader(
    ds,
    batch_size = 4,
    drop_last = False,
)
features, labels = next(iter(dl))
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

### DataLoader 内部调用方式步骤拆解

* step1: 确定数据集长度 (Dataset 的 `__len__` 方法实现)

```python
ds = TensorDataset(
    torch.randn(1000,3),
    torch.randint(low = 0, high = 2, size = (1000,)).float()
)
print("n = ", len(ds))  # len(ds)等价于 ds.__len__()
```

```
n =  1000
```

* step2: 确定抽样 indices (DataLoader 中的 `Sampler` 和 `BatchSampler` 实现)

```python
sampler = RandomSampler(data_source = ds)
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

* step3: 取出一批样本 batch (Dataset 的 `__getitem__` 方法实现)

```python
batch = [ds[i] for i in  indices]  #  ds[i] 等价于 ds.__getitem__(i)
print("batch = ", batch)
```

```
batch =  [(tensor([-0.1744, -1.1102,  0.3292]), tensor(0.)), 
          (tensor([1.0112, 1.3905, 1.7684]), tensor(0.)), 
          (tensor([ 0.6682,  0.6509, -1.1334]), tensor(0.)), 
          (tensor([-0.2228,  0.7622,  0.0318]), tensor(1.))]
``` 

* step4: 整理成 features 和 labels (DataLoader 的 `collate_fn` 方法实现)

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

### Dataset 和 DataLoader 的核心源码

以下是 Dataset 和 DataLoader 的核心源码，省略了为了提升性能而引入的诸如多进程读取数据相关的代码

```python
import torch 
class Dataset(object):
    def __init__(self):
        pass
    
    def __len__(self):
        raise NotImplementedError
        
    def __getitem__(self,index):
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
        self.collate_fn = collate_fn if collate_fn is not None 
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
    
X = torch.randn(1000, 3)
Y = torch.randint(low = 0, high = 2, size = (1000,)).float()
ds = ToyDataset(X, Y)
dl = DataLoader(ds, batch_size = 4, drop_last = False)
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

`Dataset` 定义了数据集的内容，保存了数据的样本和标签，
它相当于一个类似列表的数据结构，具有确定的长度，能够用索引获取数据集中的元素。
`Dataset` 类还是：

* `torchvision.dataset` 的父类
* `torchtext.datasets` 的父类
* `torchaudio.datasets` 的父类
* 自定义 `Dataset` 对象的父类

`Dataset` 创建数据集常用的方法有：

* 使用 `torch.utils.data.TensorDataset` 根据 Tensor 创建数据集。
  numpy 的 array，Pandas 的 DataFrame 需要先转换成 Tensor
* 使用 `torchvision.datasets.ImageFolder` 根据图片目录创建图片数据集
* 集成 `torch.utils.data.Dataset` 创建自定义数据集

此外，还可以通过：

* `torch.utils.data.random_split` 将一个数据集分割成多份，常用于分割数据集，验证集和测试集
* 调用 `Dataset` 的加法运算符 `+` 将多个数据集合并成一个数据集

### 根据 Tensor 创建数据集

```python
import numpy as np
from sklearn import datasets
import torch
from torch.utils.data import TensorDataset, Dataset, DataLoader, random_split

# 根据 Tensor 创建数据集
iris = datasets.load_iris()
ds_iris = TensorDataset(
    torch.tensor(iris.data),
    torch.tensor(iris.target),
)

# 分割成训练集和预测集
n_train = int(len(ds_iris) * 0.8)
n_val = len(ds_train) - n_train
ds_train, ds_val = random_split(ds_iris, [n_train, n_val])

# 使用 DataLoader 加载数据集
dl_train = DataLoader(
    ds_train, 
    batch_size = 8,
)
dl_val = DataLoader(
    ds_val,
    batch_size = 8,
)
for features, labels in dl_train:
    print(features, labels)
    break

# 演示加法运算符 `+` 的合并作用
ds_data = ds_train + ds_val
print(f"len(ds_train) = {len(ds_train)}")
print(f"len(ds_val) = {len(ds_val)}")
print(f"len(ds_train+ds_val) = {len(ds_data)}")
print((type(ds_data)))
```
### 根据图片目录创建图片数据集

```python
import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision import transforms, datasets
from PIL import Image

# 图片
img = Image.open("./data/cat.jpeg")
# 随机数值翻转
transforms.RandomVerticalFlip()(img)
# 随机旋转
transforms.RandomRotation(45)(img)
```

```python
# 定义图片增强操作
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

# 根据图片目录创建数据集
ds_train = datasets.ImageFolder(
    "./cifar2/train/",
    transform = transform_train,
    target_transform = transform_label,
)
ds_val = datasets.ImageFolder(
    "./cifar2/test/",
    transform = transform_valid,
    target_transform = transform_label,
)
print(ds_train.class_to_idx)
```

```python
# 使用 DataLoader 加载数据集
dl_train = DataLoader(ds_train, batch_size = 50, shuffle = True)
dl_val = DataLoader(ds_val, batch_size = 50, shuffle = False)
for features, labels in dl_train:
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

加载相关库

```python
import os
from pathlib import Path

import pandas as pd
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.io import read_image
import torchtext.iop import *
```

文件所在目录结构如下

```
- img_dir
    - 
    - labels.csv
        tshirt1.jpg, 0
        tshirt2.jpg, 0
        ...
        ankleboot999.jpg, 9
```

创建 Dataset 类

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

构建 DataLoader 训练数据集

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

### 创建自定义数据集

```python
from pathlib import Path
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader


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


# 定义图片数据增强
transform_train = transforms.Compose([
    transforms.RandomHorizontalFlip(),  # 随机水平翻转
    transforms.RandomVerticalFlip(),  # 随机垂直翻转
    transforms.RandomRotation(45),  # 随机在 45 角度内旋转
    transforms.ToTensor(),  # 转换成张量
])
transform_valid = transforms.Compose([
    transforms.ToTensor()
])

ds_train = Cifar2Dataset(train_dir, transform_train)
ds_val = Cifar2Dataset(test_dir, transform_val)

dl_train = DataLoader(ds_train, batch_size = 50, shuffle = True)
dl_val = DataLoader(ds_val, batch_size = 50, shuffle = True)

for features, labels in dl_train:
    print(features.shape)
    print(labels.shape)
    break


```

## 使用 DataLoader 加载数据集

PyTorch 数据加载工具里面最核心的类是: `torch.utils.data.DataLoader`，
`DataLoader` 将 `Dataset` 封装为可迭代对象, 便于访问样本

`DataLoader` 定义了按 batch 加载数据集的方法，它是一个实现了 `__iter__` 方法的可迭代对象，
每次迭代输出一个 batch 的数据。

`DataLoader` 能够控制 batch 的大小，batch 中元素的采样方法，
以及将 batch 结果整理成模型所需输入形式的方法，并且能够使用多进程读取数据

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

## 内置数据集

内置的数据集都是 `torch.utils.data.Dataset` 的子类，
所以它们都具有 `__getitem__` 和 `__len__` 实现方法，
因此，它们都可以被传递给可以使用 `torch.multiprocessing` 
多进程的 `torch.utils.data.DataLoader` 并行加载多个样本

这些数据集都有类似的 API，它们都有两个共同的转换参数:

* `transform`
* `target_transform`

## 使用 TorchVision datasets

* 加载相关库

```python
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor

imagenet_data = datasets.ImageNet(
    root = "path/to/imagenet_root/",
    trian = True,
    download = True,
    transform = ToTensor(),
    target_transform = ToTensor(),
) 
data_loader = DataLoader(
    imagenet_data,
    batch_size = 4,
    shuffle = True,
    num_workers = args.nThreads,
)
```

## 使用 TorchText datasets

```python
from torch.utils.data import Dataset, DataLoader
from torchtext import datasets
from torchtext.transforms import *

train_iter = datasets.IMDB(split = "train")

def tokenize(label, line):
    return line.split()

tokens = []
for label, line in train_iter:
    tokens += tokenize(label, line)
```


## 使用 TorchAudio datasets

```python
from torch.utils.data import Dataset, DataLoader
from torchaudio import datasets
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
* Transforms on PIL Image and `torch.*Tensor`
* Transforms on PIL Image only
* Transforms on `torch.*Tensor` only
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

## 语音数据管道

## 视屏数据管道

## 文本数据管道




# 参考

* [DataLoader & Dataset](https://pytorch.org/docs/stable/data.html#)

