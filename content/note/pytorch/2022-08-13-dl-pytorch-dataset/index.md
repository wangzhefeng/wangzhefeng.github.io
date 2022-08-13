---
title: PyTorch 数据集
author: 王哲峰
date: '2022-08-13'
slug: dl-pytorch-dataset
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
  - [Dataset 和 DataLoader](#dataset-和-dataloader)
    - [使用 TorchVision datasets](#使用-torchvision-datasets)
    - [使用 TorchText datasets](#使用-torchtext-datasets)
    - [使用 TorchAudio datasets](#使用-torchaudio-datasets)
  - [使用文件创建自定义 Dataset](#使用文件创建自定义-dataset)
    - [加载相关库](#加载相关库)
    - [文件所在目录结构如下](#文件所在目录结构如下)
    - [创建 Dataset 类](#创建-dataset-类)
    - [构建 DataLoader 训练数据集](#构建-dataloader-训练数据集)
  - [torch.utils.data API](#torchutilsdata-api)
    - [API](#api)
  - [参考](#参考)
- [PyTorch 数据预处理](#pytorch-数据预处理)
  - [Transforms 示例](#transforms-示例)
  - [torchvision transforms](#torchvision-transforms)
  - [torchtext transforms](#torchtext-transforms)
  - [torchaudio transforms](#torchaudio-transforms)
</p></details><p></p>

# PyTorch 数据读取构建

## Dataset 和 DataLoader

- Dataset 保存了数据的样本和标签
    - torchvision.dataset 的类
    - torchtext.datasets 的类
    - torchaudio.datasets 的类
    - 自定义 Dataset 对象的父类
- DataLoader 将 Dataset 封装为可迭代对象, 便于访问样本。DataLoader 对象功能：
    - automatic batching
    - sampling
    - shuffling
    - multiprocess data loading

### 使用 TorchVision datasets

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

### 使用 TorchText datasets

```python
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torchtext import datasets
from torchtext.transforms import *
```

### 使用 TorchAudio datasets

```python
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torchaudio import datasets
from torchaudio.transforms import *
```

## 使用文件创建自定义 Dataset

* `Dataset` 类需要实现以下方法
    - `__init__`
    - `__len__`
    - `__getitem__`

### 加载相关库

```python
import os
import pandas as pd
import torchvision.io import read_image
# TODO import torchtext.iop import *
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
```

### 文件所在目录结构如下

```
- img_dir
    - 
    - labels.csv
        tshirt1.jpg, 0
        tshirt2.jpg, 0
        ...
        ankleboot999.jpg, 9
```

### 创建 Dataset 类

```python
class CustomImage(Dataset):

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
        return len(self.img_labels)
    
    def __getitem__(self, idx):
        img_path = os.path.join(
            self.img_dir, 
            self.img_labels.iloc[idx, 0]
        )
        image = read_image(img_path)
        label = self.img_labels.iloc[idx, 1]
        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)

        return image, label
```

### 构建 DataLoader 训练数据集

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

## torch.utils.data API

PyTorch 数据加载工具里面最核心的类是: `torch.utils.data.DataLoader`，
该类生成可迭代的数据集:

* map-style、可迭代(iterable-style)数据集
* 自定义数据记载顺序
* 自动 batching
* 单进程、多进程数据加载
* 自动内存锁定(pinning)

### API

```python
DataLoader(
    dataset,  # Dataset 对象, tensorvision,torchtext,torchaudio,自定义 Dataset
    batch_size=1,  # 自动 batching
    shuffle=False,  # shuffling
    sampler=None,
    batch_sampler=None, 
    num_workers=0,  # 设置进程数
    collate_fn=None,
    pin_memory=False,  # 内存锁定 
    drop_last=False, 
    timeout=0,
    worker_init_fn=None, 
    *, 
    prefetch_factor=2,
    persistent_workers=False
)
```


## 参考

- [DataLoader & Dataset](https://pytorch.org/docs/stable/data.html#)


# PyTorch 数据预处理

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
    target_transform = Lambda(lambda y: torch.zeros(10, dtype = torch.float).scatter_(dim = 0, torch.tensor(y), value = 1))
)
test_data = datasets.FashionMNIST(
    root = "data",
    train = False,
    download = True,
    transform = ToTensor(),
    target_transform = Lambda(lambda y: torch.zeros(10, dtype = torch.float).scatter_(dim = 0, torch.tensor(y), value = 1))
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

