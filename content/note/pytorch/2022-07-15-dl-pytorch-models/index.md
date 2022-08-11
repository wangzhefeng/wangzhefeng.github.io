---
title: PyTorch Model
author: 王哲峰
date: '2022-07-15'
slug: dl-pytorch-models
categories:
  - deeplearning
  - pytorch
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

- [PyTorch torch.nn](#pytorch-torchnn)
  - [torch.nn 内容](#torchnn-内容)
  - [nn.Module 内容](#nnmodule-内容)
- [PyTorch 模型构建](#pytorch-模型构建)
  - [PyTorch 模型的创建](#pytorch-模型的创建)
  - [PyTorch 模型容器](#pytorch-模型容器)
  - [PyTorch 权重初始化](#pytorch-权重初始化)
  - [PyTorch 损失函数](#pytorch-损失函数)
  - [PyTorhc 优化器](#pytorhc-优化器)
  - [PyTorch 评价指标](#pytorch-评价指标)
- [PyTorch 模型保存和加载](#pytorch-模型保存和加载)
  - [保存和加载模型权重参数](#保存和加载模型权重参数)
  - [保存和加载整个模型](#保存和加载整个模型)
  - [导出模型为 ONNX](#导出模型为-onnx)
  - [保存和加载 checkpoint](#保存和加载-checkpoint)
- [PyTorch 模型部署](#pytorch-模型部署)
</p></details><p></p>

# PyTorch torch.nn

## torch.nn 内容

torch.nn 包含了构建神经网络计算图的基本模块:

* Containers
* Convolution Laysers
* Pooling Layers
* Padding Layers
* Non-linear Activations
* Normalization Layers
* Recurrent Layers
* Transformer Layers
* Linear Layers
* Dropout Layers
* Sparse Layers
* Distance Functions
* Loss Functions
* Vision Layers
* Shuffle Layers
* DataParallel Layers
* Utilities
* Quantized Functions
* Lazy Modules Initialization

## nn.Module 内容



# PyTorch 模型构建

## PyTorch 模型的创建

## PyTorch 模型容器

## PyTorch 权重初始化

## PyTorch 损失函数

## PyTorhc 优化器

## PyTorch 评价指标

# PyTorch 模型保存和加载

## 保存和加载模型权重参数

PyTorch 将模型训练的学习到的权重参数保存在一个状态字典中, `state_dict`

- 模型保存

```python
import torch
import torchvision.models as models

# 保存模型
model = models.vgg16(pretrained = True)
MODEL_PATH = "models/model_weights.pth"
torch.save(
    model.state_dict(), 
    MODEL_PATH
)
```

- 模型加载

```python
import torch
import torchvision.models as models

# do not specify pretrained=True, i.e. do not load default weights
model = models.vgg16()
MODEL_PATH = "models/model_weights.pth"
model.load_state_dict(
    torch.load(MODEL_PATH)
)
model.eval()
```

## 保存和加载整个模型

- 模型保存

```python
import torch
import torchvision.models as models

model = models.vgg16(pretrained = True)
MODEL_PATH = "models/model.pth"
torch.save(the_model, MODEL_PATH)
```

- 模型加载

```python
import torch

MODEL_PATH = "models/model.pth"
model = torch.load(MODEL_PATH)  # require pickle module
```

## 导出模型为 ONNX

* PyTorch 还具有原生 ONNX 导出支持。然而，鉴于 PyTorch 执行图的动态特性，
  导出过程必须遍历执行图以生成持久的 ONNX 模型。出于这个原因，
  应该将适当大小的测试变量传递给导出例程
* [ONNX 教程](https://github.com/onnx/tutorials)

```python
import torch
import torch.onnx as onnx

input_image = torch.zeros((1, 3, 224, 224))
```

## 保存和加载 checkpoint

- 加载相关库

```python
import torch
import torch.nn as nn
import torch.optim as optim
```

- 定义和初始化模型

```python
class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

net = Net()
print(net)
```

- 初始化优化器

```python
optimizer = optim.SGD(net.parameters(), lr = 0.001, momentum = 0.9)
```

- 保存 checkpoint

```python
EPOCH = 5
PATH = "model.pt"
LOSS = 0.4

torch.save({
    "epoch": EPOCH,
    "model_state_dict": net.state_dict(),
    "optimizer_state_dict": optimizer.state_dict(),
    "loss": LOSS,
}, PATH)
```

- 加载 checkpoint

```python
PATH = "model.pt"

# 初始化模型
model = Net()
# 初始化优化器
optimizer = optim.SGD(net.parameters(), lr = 0.001, momentum = 0.9)

# 加载 checkpoint
checkpoint = torch.load(PATH)

model.load_state_dict(checkpoint["model_state_dict"])
optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

epoch = checkpoint["epoch"]
loss = checkpoint["loss"]

model.eval()
# - or - 
model.train()
```

# PyTorch 模型部署

