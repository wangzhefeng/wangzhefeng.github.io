---
title: PyTorch 模型保存和加载
author: 王哲峰
date: '2022-08-16'
slug: dl-pytorch-model-save-load
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

- [保存和加载模型权重参数](#保存和加载模型权重参数)
  - [模型保存](#模型保存)
  - [模型加载](#模型加载)
- [保存和加载整个模型](#保存和加载整个模型)
  - [模型保存](#模型保存-1)
  - [模型加载](#模型加载-1)
- [导出模型为 ONNX](#导出模型为-onnx)
- [保存和加载 checkpoint](#保存和加载-checkpoint)
  - [定义和初始化模型](#定义和初始化模型)
  - [初始化优化器](#初始化优化器)
  - [保存 checkpoint](#保存-checkpoint)
  - [加载 checkpoint](#加载-checkpoint)
</p></details><p></p>

# 保存和加载模型权重参数

PyTorch 将模型训练学习到的权重参数保存在一个状态字典 `state_dict` 中，

## 模型保存

* `torch.save(model.state_dict(), model_path)`

```python
import torch
import torchvision.models as models

# 模型保存路径
MODEL_PATH = "models/model_weights.pth"

# 模型
model = models.vgg16(pretrained = True)

# 模型保存
torch.save(
    model.state_dict(), 
    MODEL_PATH,
)
```

## 模型加载

* `model.load_state_dict(torch.load(model_path))`
* `model.eval()`

```python
import torch
import torchvision.models as models

# 模型保存路径
MODEL_PATH = "models/model_weights.pth"

# 模型加载
model = models.vgg16()  # do not specify pretrained=True, i.e. do not load default weights
model.load_state_dict(
    torch.load(MODEL_PATH)
)
model.eval()
```

# 保存和加载整个模型

## 模型保存

* `torch.save(model, model_path)`

```python
import torch
import torchvision.models as models

# 模型保存路径
MODEL_PATH = "models/model.pth"

# 模型
model = models.vgg16(pretrained = True)

# 模型保存
torch.save(
    model, 
    MODEL_PATH
)
```

## 模型加载

* `torch.load(model_path)`

```python
import torch

# 模型保存路径
MODEL_PATH = "models/model.pth"

# 模型加载
model = torch.load(MODEL_PATH)  # require pickle module
```

# 导出模型为 ONNX

* PyTorch 还具有原生 ONNX 导出支持。然而，鉴于 PyTorch 执行图的动态特性，
  导出过程必须遍历执行图以生成持久的 ONNX 模型。出于这个原因，
  应该将适当大小的测试变量传递给导出例程
* [ONNX 教程](https://github.com/onnx/tutorials)

```python
import torch
import torch.onnx as onnx

input_image = torch.zeros((1, 3, 224, 224))
```

# 保存和加载 checkpoint

```python
import torch
import torch.nn as nn
import torch.optim as optim
```

## 定义和初始化模型

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

## 初始化优化器

```python
optimizer = optim.SGD(
    net.parameters(), 
    lr = 0.001, 
    momentum = 0.9
)
```

## 保存 checkpoint

* `torch.save({})`

```python
EPOCH = 5
MODEL_PATH = "model.pt"
LOSS = 0.4

torch.save({
    "epoch": EPOCH,
    "model_state_dict": net.state_dict(),
    "optimizer_state_dict": optimizer.state_dict(),
    "loss": LOSS,
}, MODEL_PATH)
```

## 加载 checkpoint

* `torch.load(model_path)`
* `.load_state_dict()`

```python
MODEL_PATH = "model.pt"

# 初始化模型
model = Net()

# 初始化优化器
optimizer = optim.SGD(
    net.parameters(), 
    lr = 0.001, 
    momentum = 0.9
)

# 加载 checkpoint
checkpoint = torch.load(MODEL_PATH)

model.load_state_dict(checkpoint["model_state_dict"])
optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
epoch = checkpoint["epoch"]
loss = checkpoint["loss"]

model.eval()
# - or - 
model.train()
```

