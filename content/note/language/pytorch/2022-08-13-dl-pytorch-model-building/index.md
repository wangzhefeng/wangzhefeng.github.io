---
title: PyTorch 模型构建
author: 王哲峰
date: '2022-08-13'
slug: dl-pytorch-model-building
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
</style>

<details><summary>目录</summary><p>

- [模型创建简介](#模型创建简介)
- [使用 nn.Sequential 按层顺序构建模型](#使用-nnsequential-按层顺序构建模型)
  - [用 add\_module 方法](#用-add_module-方法)
  - [使用变长参数](#使用变长参数)
  - [使用 OrderedDict](#使用-ordereddict)
- [继承 nn.Module 基类构建自定义模型](#继承-nnmodule-基类构建自定义模型)
  - [模型类示例 1](#模型类示例-1)
  - [模型类示例 2](#模型类示例-2)
  - [模型层](#模型层)
  - [模型参数](#模型参数)
- [继承 nn.Module 基类构建模型并辅助应用模型容器进行封装](#继承-nnmodule-基类构建模型并辅助应用模型容器进行封装)
  - [nn.Sequential](#nnsequential)
  - [nn.ModuleList](#nnmodulelist)
  - [nn.ModuleDict](#nnmoduledict)
</p></details><p></p>

# 模型创建简介

使用 PyTorch 通常有三种方式构建模型:

* 使用 `torch.nn.Sequential` 按层顺序构建模型
* 继承 `torch.nn.Module` 基类构建自定义模型
* 继承 `torch.nn.Module` 基类构建模型并辅助应用模型容器进行封装
    - `torch.nn.Sequential`
    - `torch.nn.ModuleList`
    - `torch.nn.ModuleDict`

# 使用 nn.Sequential 按层顺序构建模型

使用 `nn.Sequential` 按层顺序构建模型无需定义 `forward` 方法，仅仅适用于简单的模型

## 用 add_module 方法

```python
import torch
from torch import nn
from torchkeras import summary

net = nn.Sequential()
net.add_module("conv1", nn.Conv2d(
    in_channels = 3, 
    out_channels = 32, 
    kernel_size = 3)
)
net.add_module("pool1", nn.MaxPool2d(kernel_size = 2,stride = 2))
net.add_module("conv2", nn.Conv2d(
    in_channels = 32, 
    out_channels = 64, 
    kernel_size = 5)
)
net.add_module("pool2", nn.MaxPool2d(kernel_size = 2, stride = 2))
net.add_module("dropout", nn.Dropout2d(p = 0.1))
net.add_module("adaptive_pool", nn.AdaptiveMaxPool2d((1, 1)))
net.add_module("flatten", nn.Flatten())
net.add_module("linear1", nn.Linear(64, 32))
net.add_module("relu", nn.ReLU())
net.add_module("linear2", nn.Linear(32, 1))

print(net)
summary(net, input_shape = (3, 32, 32))
```

## 使用变长参数

* 不能给每个层指定名称

```python
import torch
from torch import nn
from torchkeras import summary


net = nn.Sequential(
    nn.Conv2d(in_channels = 3, out_channels = 32, kernel_size = 3),
    nn.MaxPool2d(kernel_size = 2, stride = 2),
    nn.Conv2d(in_channels = 32, out_channels = 64, kernel_size = 5),
    nn.MaxPool2d(kernel_size = 2, stride = 2),
    nn.Dropout2d(p = 0.1),
    nn.AdaptiveMaxPool2d((1, 1)),
    nn.Flatten(),
    nn.Linear(64, 32),
    nn.ReLU(),
    nn.Linear(32, 1),
)

print(net)
summary(net, input_shape = (3, 32, 32))
```

## 使用 OrderedDict

```python
import torch
from torch import nn
from torchkeras import summary
from collections import OrderedDict

net = nn.Sequential(
    OrderedDict([
        ("conv1", nn.Conv2d(in_channels = 3, out_channels = 32, kernel_size = 3)),
        ("pool1", nn.MaxPool2d(kernel_size = 2, stride = 2)),
        ("conv2", nn.Conv2d(in_channels = 32, out_channels = 64, kernel_size = 5)),
        ("pool2", nn.MaxPool2d(kernel_size = 2, stride = 2)),
        ("dropout", nn.Dropout2d(p = 0.1)),
        ("adaptive_pool", nn.AdaptiveMaxPool2d((1, 1))),
        ("flatten", nn.Flatten()),
        ("linear1", nn.Linear(64, 32)),
        ("relu", nn.ReLU()),
        ("linear2", nn.Linear(32, 1)),
    ])
)
print(net)
summary(net, input_shape = (3, 32, 32))
```

# 继承 nn.Module 基类构建自定义模型

## 模型类示例 1

```python
from torch import nn
from torchkeras import summary

class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()
         self.conv1 = nn.Conv2d(in_channels = 3, out_channels = 32, kernel_size = 3)
         self.pool1 = nn.MaxPool2d(kernel_size = 2, stride = 2)
         self.conv2 = nn.Conv2d(in_channels = 32, out_channels = 64, kernel_size = 5)
         self.pool2 = nn.MaxPool2d(kernel_size = 2, stride = 2)
         self.dropout = nn.Dropout2d(p = 0.1)
         self.adaptive_pool = nn.AdaptiveMaxPool2d((1, 1))
         self.flatten = nn.Flatten()
         self.linear1 = nn.Linear(64, 32)
         self.relu = nn.ReLU()
         self.linear2 = nn.Linear(32, 1)
    
    def forward(self, x):
        x = self.conv1(x)
        x = self.pool1(x)
        x = self.conv2(x)
        x = self.pool2(x)
        x = self.dropout(x)
        x = self.adaptive_pool(x)
        x = self.flatten(x)
        x = self.linear1(x)
        x = self.relu(x)
        y = self.linear2(x)
        return y
    
net = Net()
print(net)
summary(net, input_shape = (3, 32, 32))
```

## 模型类示例 2

```python
import torch
from torch import nn
print(f"torch.__version__ = {torch.__version__}")


device = "cuda:0" if torch.cuda.is_available() else "cpu"
print(f"Using {device} device.")


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

model = NeuralNetwork().to(device)
print(model)
```

```python
# 模型输入
X = torch.rand(1, 28, 28, device = device)

# 模型训练
logits = model(X)

# 模型预测
pred_probab = nn.Softmax(dim = 1)(logits)
y_pred = pred_probab.argmax(1)
print(f"Predicted class: {y_pred}")
```

## 模型层

* 三通道图像

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

# 继承 nn.Module 基类构建模型并辅助应用模型容器进行封装

当模型的结构比较复杂时，可以应用模型容器 `nn.Sequential`、`nn.ModuleList`、
`nn.ModuleDict` 对模型的部分结构进行封装。
这样做会让模型整体更加有层次感，有时候也能减少代码量。
模型容器的使用是非常灵活的，可以在一个模型中任意组合任意嵌套使用

## nn.Sequential

```python
import torch
from torch import nn


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels = 3, out_channels = 32, kernel_size = 3),
            nn.MaxPool2d(kernel_size = 2, stride = 2),
            nn.Conv2d(in_channels = 32, out_channels = 64, kernel_size = 5),
            nn.MaxPool2d(kernel_size = 2, stride = 2),
            nn.Dropout2d(p = 0.1),
            nn.AdaptiveMaxPool2d((1, 1)),
        )
        self.dense = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
        )
    
    def forward(self, x):
        x = self.conv(x)
        y = self.dense(x)
        return y

net = Net()
print(net)
```

## nn.ModuleList

`nn.ModuelList` 不能用 Python 中的 List 代替

```python
import torch
from torch import nn


class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        self.layer_list = nn.ModuleList([
            nn.Conv2d(in_channels = 3, out_channels = 32, kernel_size = 3),
            nn.MaxPool2d(kernel_size = 2, stride = 2),
            nn.Conv2d(in_channels = 32, out_channels = 64, kernel_size = 5)
            nn.MaxPool2d(kernel_size = 2, stride = 2),
            nn.Dropout2d(p = 0.1),
            nn.AdaptiveMaxPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
        ])
    
    def forward(self, x):
        for layer in self.layer_list:
            x = layer(x)
        return x

net = Net()
print(net)
```

## nn.ModuleDict

`nn.ModuleDict` 不能用 Python 中的 Dict 代替

```python
import torch
from torch import nn


class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        self.layers_dict = nn.ModuleDict({
            "conv1": nn.Conv2d(in_channels = 3, out_channels = 32, kernel_size = 3),
            "pool": nn.MaxPool2d(kernel_size = 2, stride = 2),
            "conv2": nn.Conv2d(in_channels = 32, out_channels = 64, kernel_size = 5),
            "dropout": nn.Dropout2d(p = 0.1),
            "adaptive": nn.AdaptiveMaxPool2d((1, 1)),
            "flatten": nn.Flatten(),
            "linear1": nn.Linear(64, 32),
            "relu": nn.ReLU(),
            "linear2": nn.Linear(32, 1),
        })

    def forward(self, x):
        layers = [
            "conv1",
            "pool",
            "conv2",
            "pool",
            "dropout",
            "adaptive",
            "flatten",
            "linear",
            "relu",
            "linear2",
        ]
        for layer in layers:
            x = self.layers_dict[layer](x)
        return x

net = Net()
print(net)
```

