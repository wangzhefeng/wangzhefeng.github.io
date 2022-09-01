---
title: PyTorch 模型构建
author: 王哲峰
date: '2022-08-13'
slug: dl-pytorch-model-building
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

- [PyTorch 模型的创建](#pytorch-模型的创建)
- [使用 nn.Sequential 按层顺序构建模型](#使用-nnsequential-按层顺序构建模型)
  - [用 add_module() 方法](#用-add_module-方法)
  - [使用变长参数](#使用变长参数)
  - [使用 OrderedDict](#使用-ordereddict)
- [继承 nn.Module 基类构建自定义模型](#继承-nnmodule-基类构建自定义模型)
  - [模型类](#模型类)
  - [模型层](#模型层)
  - [模型参数](#模型参数)
- [继承 nn.Module 基类构建面模型并辅助应用模型容器进行封装](#继承-nnmodule-基类构建面模型并辅助应用模型容器进行封装)
  - [Sequential](#sequential)
  - [ModuleList](#modulelist)
  - [ModuleDict](#moduledict)
</p></details><p></p>

# PyTorch 模型的创建

使用 PyTorch 通常有三种方式构建模型:

* 使用 nn.Sequential 按层顺序构建模型
* 继承 nn.Module 基类构建自定义模型
* 继承 nn.Module 基类构建面模型并辅助应用模型容器进行封装
    - `nn.Sequential`
    - `nn.ModuleList`
    - `nn.ModuleDict`

# 使用 nn.Sequential 按层顺序构建模型

使用 `nn.Sequential` 按层顺序构建模型无需定义 `forward` 方法，仅仅适用于简单的模型

## 用 add_module() 方法

```python
net = nn.Sequential()
net.add_module("conv1",nn.Conv2d(in_channels=3,out_channels=32,kernel_size = 3))
net.add_module("pool1",nn.MaxPool2d(kernel_size = 2,stride = 2))
net.add_module("conv2",nn.Conv2d(in_channels=32,out_channels=64,kernel_size = 5))
net.add_module("pool2",nn.MaxPool2d(kernel_size = 2,stride = 2))
net.add_module("dropout",nn.Dropout2d(p = 0.1))
net.add_module("adaptive_pool",nn.AdaptiveMaxPool2d((1,1)))
net.add_module("flatten",nn.Flatten())
net.add_module("linear1",nn.Linear(64,32))
net.add_module("relu",nn.ReLU())
net.add_module("linear2",nn.Linear(32,1))
print(net)
```

## 使用变长参数

* 不能给每个层指定名称

```python
net = nn.Sequential(
    nn.Conv2d(in_channels=3,out_channels=32,kernel_size = 3),
    nn.MaxPool2d(kernel_size = 2,stride = 2),
    nn.Conv2d(in_channels=32,out_channels=64,kernel_size = 5),
    nn.MaxPool2d(kernel_size = 2,stride = 2),
    nn.Dropout2d(p = 0.1),
    nn.AdaptiveMaxPool2d((1,1)),
    nn.Flatten(),
    nn.Linear(64,32),
    nn.ReLU(),
    nn.Linear(32,1)
)

print(net)
```

## 使用 OrderedDict

```python
from collections import OrderedDict

net = nn.Sequential(OrderedDict(
          [("conv1",nn.Conv2d(in_channels=3,out_channels=32,kernel_size = 3)),
            ("pool1",nn.MaxPool2d(kernel_size = 2,stride = 2)),
            ("conv2",nn.Conv2d(in_channels=32,out_channels=64,kernel_size = 5)),
            ("pool2",nn.MaxPool2d(kernel_size = 2,stride = 2)),
            ("dropout",nn.Dropout2d(p = 0.1)),
            ("adaptive_pool",nn.AdaptiveMaxPool2d((1,1))),
            ("flatten",nn.Flatten()),
            ("linear1",nn.Linear(64,32)),
            ("relu",nn.ReLU()),
            ("linear2",nn.Linear(32,1))
          ])
        )
print(net)
```

# 继承 nn.Module 基类构建自定义模型

## 模型类

```python
from torch import nn

class NeuralNetwork(nn.Module):
    def __init__(self):
        super(NeuralNetwork, self).__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(28*28, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 10),
        )

    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits
```

```python
model = NeuralNetwork().to(device)
print(model)
```

```python
import torch

X = torch.rand(1, 28, 28, device = device)
logits = model(X)
pred_probab = nn.Softmax(dim = 1)(logits)
y_pred = pred_probab.argmax(1)
print(f"Predicted class: {y_pred}")
```

## 模型层

* 三个图像

```python
import torch

input_image = torch.rand(3, 28, 28)
print(input_image.size())
```

* nn.Flatten

```python
from torch import nn

flatten = nn.Flatten()
flat_image = flatten(input_image)
print(flat_image.size())
```

* nn.Linear

```python
from torch import nn

layer1 = nn.Linear(in_features = 28 * 28, out_features = 20)
hidden1 = layer1(flat_image)
print(hidden1.size())
```

* nn.ReLU

```python
from torch import nn

print(f"Before ReLU: {hidden1}\n\n")
hidden1 = nn.ReLU()(hidden1)
print(f"After ReLU: {hidden1}")
```

* nn.Sequential

```python
import torch
from torch import nn

seq_modules = nn.Sequential(
    flatten,
    layer1,
    nn.ReLU(),
    nn.Linear(20, 10)
)
input_image = torch.rand(3, 28, 28)
logits = seq_modules(input_image)
```

* nn.Softmax

```python
from torch import nn

softmax = nn.Softmax(dim = 1)
pred_probab = softmax(logits)
```

## 模型参数

```python
print(f"Model structure: {model}\n\n")

for name, param in model.named_parameters():
    print(f"Layer: {name} | Size: {param.size()} | Values: {param[:2]}\n")
```

# 继承 nn.Module 基类构建面模型并辅助应用模型容器进行封装

当模型的结构比较复杂时，可以应用模型容器 `nn.Sequential`、`nn.ModuleList`、
`nn.ModuleDict` 对模型的部分结构进行封装。
这样做会让模型整体更加有层次感，有时候也能减少代码量。
模型容器的使用是非常灵活的，可以在一个模型中任意组合任意嵌套使用。

## Sequential


## ModuleList


## ModuleDict









