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
  - [模型训练设备](#模型训练设备)
  - [模型类](#模型类)
  - [模型层](#模型层)
  - [模型参数](#模型参数)
- [PyTorch 模型容器](#pytorch-模型容器)
- [PyTorch 权重初始化](#pytorch-权重初始化)
- [PyTorch 损失函数](#pytorch-损失函数)
- [PyTorhc 优化器](#pytorhc-优化器)
- [PyTorch 评价指标](#pytorch-评价指标)
</p></details><p></p>


* [torch.nn](https://pytorch.org/docs/stable/nn.html)
* [torch.nn.Module](https://pytorch.org/docs/stable/generated/torch.nn.Module.html)

```python
import torch
from torch import nn
```

# PyTorch 模型的创建

## 模型训练设备

```python
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using {device} device")
```

## 模型类

```python
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
X = torch.rand(1, 28, 28, device = device)
logits = model(X)
pred_probab = nn.Softmax(dim = 1)(logits)
y_pred = pred_probab.argmax(1)
print(f"Predicted class: {y_pred}")
```

## 模型层

* 三个图像

```python
input_image = torch.rand(3, 28, 28)
print(input_image.size())
```

* nn.Flatten

```python
flatten = nn.Flatten()
flat_image = flatten(input_image)
print(flat_image.size())
```

* nn.Linear

```python
layer1 = nn.Linear(in_features = 28 * 28, out_features = 20)
hidden1 = layer1(flat_image)
print(hidden1.size())
```

* nn.ReLU

```python
print(f"Before ReLU: {hidden1}\n\n")
hidden1 = nn.ReLU()(hidden1)
print(f"After ReLU: {hidden1}")
```

* nn.Sequential

```python
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
softmax = nn.Softmax(dim = 1)
pred_probab = softmax(logits)
```

## 模型参数

```python
print(f"Model structure: {model}\n\n")

for name, param in model.named_parameters():
    print(f"Layer: {name} | Size: {param.size()} | Values: {param[:2]}\n")
```





# PyTorch 模型容器

# PyTorch 权重初始化

# PyTorch 损失函数

# PyTorhc 优化器

# PyTorch 评价指标
