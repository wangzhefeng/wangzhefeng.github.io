---
title: PyTorch 模型训练
author: 王哲峰
date: '2022-08-13'
slug: dl-pytorch-model-compile
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

- [权重初始化](#权重初始化)
- [损失函数](#损失函数)
- [优化器](#优化器)
- [评价指标](#评价指标)
- [模型训练](#模型训练)
  - [脚本风格](#脚本风格)
  - [函数风格](#函数风格)
  - [类风格 torchkeras.KerasModel](#类风格-torchkeraskerasmodel)
  - [类风格 torchkeras.LightModel](#类风格-torchkeraslightmodel)
</p></details><p></p>

# 权重初始化

# 损失函数


# 优化器


# 评价指标


# 模型训练

PyTorch 通常需要用户编写自定义训练循环，训练循环的代码风格因人而异。
有三类典型的训练循环代码风格:

* 脚本形式训练循环
* 函数形式训练循环
* 类形式训练循环
* 通用的 Keras 风格的脚本形式训练循环

```python
import torch
from torch import nn
import torchvision
from torchvision import transforms

transform = transforms.Compose([transforms.ToTensor()])

ds_train = torchvision.datasets.MNIST(
    root = "./data/minist/", 
    train = True,
    download = True,
    transform = transform
)
ds_val = torchvision.datasets.MNIST(
    root = "./data/minist/",
    train = False,
    download = True,
    transform = transform
)

dl_train =  torch.utils.data.DataLoader(
    ds_train, 
    batch_size = 128, 
    shuffle = True, 
    num_workers = 4
)
dl_val =  torch.utils.data.DataLoader(
    ds_val, 
    batch_size = 128, 
    shuffle = False, 
    num_workers = 4
)

print(len(ds_train))
print(len(ds_val))
```

## 脚本风格


## 函数风格

## 类风格 torchkeras.KerasModel

## 类风格 torchkeras.LightModel