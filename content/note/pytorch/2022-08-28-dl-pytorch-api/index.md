---
title: PyTorch API
author: 王哲峰
date: '2022-08-28'
slug: dl-pytorch-api
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

- [低阶 API](#低阶-api)
- [中阶 API](#中阶-api)
- [高阶 API](#高阶-api)
- [torch.nn.functional 和 torch.nn.Module](#torchnnfunctional-和-torchnnmodule)
  - [torch.nn.functional](#torchnnfunctional)
  - [torch.nn.Module](#torchnnmodule)
    - [使用 nn.Module 管理参数](#使用-nnmodule-管理参数)
    - [使用 nn.Module 管理子模块](#使用-nnmodule-管理子模块)
</p></details><p></p>

# 低阶 API

PyTorch 低阶 API 主要包括:

* 张量操作
* 计算图
* 自动微分


# 中阶 API

PyTorch 中阶 API 主要包括

* 模型层
* 损失函数
* 优化器
* 数据管道


# 高阶 API

PyTorch 没有官方的高阶 API，一般需要用户自己实现训练循环、验证循环、预测循环


# torch.nn.functional 和 torch.nn.Module

PyTorch 和神经网络相关的功能组件大多都封装在 `torch.nn` 模块下，
这些功能组件的绝大部分既有函数形式实现，也有类形式实现

## torch.nn.functional

`torch.nn.functional` 有各种功能组件的函数实现:

```python
import torch.nn.functional as F
```

* 激活函数
    - `F.relu`
    - `F.sigmoid`
    - `F.tanh`
    - `F.softmax`
* 模型层
    - `F.linear`
    - `F.conv2d`
    - `F.max_pool2d`
    - `F.dropout2d`
    - `F.embedding`
* 损失函数
    - `F.binary_cross_entropy`
    - `F.mse_loss`
    - `F.cross_entropy`

```python
import torch
import torch.nn.functional as F

torch.relu(torch.tensor(-1.0))
F.relu(torch.tensor(-1.0))
```

```
tensor(0.)
tensor(0.)
```

## torch.nn.Module

为了便于对参数进行管理，一般通过继承 `torch.nn.Module` 转换称为类的实现形式，
并直接封装在 `torch.nn` 模块下

```python
from torch import nn
```

* 激活函数
    - `nn.ReLU`
    - `nn.Sigmoid`
    - `nn.Tanh`
    - `nn.Softmax`
* 模型层
    - `nn.Linear`
    - `nn.Conv2d`
    - `nn.MaxPool2d`
    - `nn.Dropout2d`
    - `nn.Embedding`
* 损失函数
    - `nn.BCELoss`
    - `nn.MSELoss`
    - `nn.CrossEntropyLoss`

实际上，`torch.nn.Module` 除了可以管理其引用的各种参数，
还可以管理其引用的子模块，功能十分强大

### 使用 nn.Module 管理参数

在 PyTorch 中，模型的参数是需要被优化器训练的，
因此，通常要设置参数为 `requires_grad = True` 的张量。
同时，在一个模型中，往往有许多的参数，要手动管理这些参数并不是一件容易的事情

PyTorch 一般将参数用 `nn.Parameter` 来表示，
并且用 `nn.Module` 来管理其结构下的所有参数

* 载入 Python 依赖

```python
import torch
from torch import nn
import torch.nn.functional as F
```

* 设置参数为 `requires_grad = True` 的张量

```python
torch.randn(2, 2, requires_grad = True)
```

* `nn.Parameter()` 具有 `require_grad = True` 属性

```python
w = nn.Parameter(torch.randn(2, 2))
print(w)
print(w.requires_grad)
```

* `nn.ParameterList()` 可以将多个 `nn.Parameter()` 组成一个列表

```python
params_list = nn.ParameterList([
    nn.Parameter(torch.rand(8, i))
    for i in range(1, 3)
])
print(params_list)
print(params_list[0].requires_grad)
```

* `nn.ParameterDict()` 可以将多个 `nn.Parameter()` 组成一个字典

```python
params_dict = nn.ParameterDict({
    "a": nn.Parameter(torch.rand(2, 2)),
    "b": nn.Parameter(torch.zeros(2)),
})
print(params_dict)
print(params_dict["a"].requires_grad)
```

* 用 `nn.Module()` 将 `nn.Parameter`、`nn.ParameterList()`、`nn.ParameterDict()` 管理起来

```python
module = nn.Module()
module.w = nn.Parameter(
    torch.randn(2, 2)
)
module.params_list = nn.ParameterList([
    nn.Parameter(torch.rand(8, i))
    for i in range(1, 3)
])
module.param_dict = nn.ParameterDict({
    "a": nn.Parameter(torch.rand(2, 2)),
    "b": nn.Parameter(torch.zeros(2)),
})

num_param = 0
for param in module.named_parameters():
    print(param, "\n")
    num_param = num_param + 1
print(f"Number of Parameters = {num_param}")
```

* 实践当中，一般通过继承 `nn.Module` 来构建模块类，并将所有含有需要学习的部分放在构造函数中

```python
class Linear(nn.Module):
    __constants__ = ["in_features", "out_features"]

    def __init__(self, in_features, out_features, bias = True):
        super(Linear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(
            torch.Tensor(out_features, in_features)
        )
        if bias:
            self.bias = nn.Parameter(torch.Tensor(out_features))
        else:
            self.register_parameter("bias", None)
        
    def forward(self, input):
        return F.linear(input, self.weight, self.bias)
```


### 使用 nn.Module 管理子模块

一般情况下，很少直接使用 `nn.Parameter` 来定义参数构建模型，而是通过一些拼装一些常用的模型层来构造模型。
这些模型层也是继承自 `nn.Module` 的对象，本身也包括参数，属于要定义的模块的子模块

`nn.Module` 提供了一些方法可以管理这些子模块:

* `children()`: 返回生成器，包括模块下的所有子模块
* `named_children()`: 返回一个生成器，包括模块下的所有子模块，以及它们的名字
* `modules()`: 返回一个生成器，包括模块下的所有各个层级的模块，包括模块本身
* `named_modules()`: 返回一个生成器，包括模块下的所有各个层级的模块以及它们的名字，包括模块本身

其中:

* `children()` 和 `named_children()` 方法较多使用
* `modules()` 和 `named_modules()` 方法较少使用，其功能可以通过多个 `named_children()` 的嵌套使用实现

```python
class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()

        self.embedding = nn.Embedding(
            num_embedding = 10000, 
            embedding_dim = 3, 
            padding_idx = 1
        )

        self.conv = nn.Sequential()
        self.conv.add_module(
            "conv_1", 
            nn.Conv1d(in_channels = 3, out_channels = 16, kernel_size = 5),
        )
        self.conv.add_module(
            "pool_1",
            nn.MaxPool1d(kernel_size = 2),
        )
        self.conv.add_module(
            "relu",
            nn.ReLU(),
        )
        self.conv.add_module(
            "conv_2",
            nn.Conv1d(in_channels = 16, out_channels = 128, kernel_size = 2),
        )
        self.conv.add_module(
            "pool_2",
            nn.MaxPool1d(kernel_size = 2),
        )
        self.conv.add_module(
            "relu_2",
            nn.ReLU(),
        )

        self.dense = nn.Sequential()
        self.dense.add_module("flatten", nn.Flatten())
        self.dense.add_module("linear", nn.Linear(6144, 1))
    
    def forward(self, x):
        x = self.embedding(x).transpose(1, 2)
        x = self.conv(x)
        y = self.dense(x)
        return y

net = Net()
```

```python
i = 0
for child in net.children():
    i += 1
    print(child, "\n")
print("child number", i)
```

```
Embedding(10000, 3, padding_idx=1) 

Sequential(
  (conv_1): Conv1d(3, 16, kernel_size=(5,), stride=(1,))
  (pool_1): MaxPool1d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
  (relu_1): ReLU()
  (conv_2): Conv1d(16, 128, kernel_size=(2,), stride=(1,))
  (pool_2): MaxPool1d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
  (relu_2): ReLU()
) 

Sequential(
  (flatten): Flatten(start_dim=1, end_dim=-1)
  (linear): Linear(in_features=6144, out_features=1, bias=True)
) 

child number 3
```

```python
i = 0
for name, child in net.named_children():
    i += 1
    print(name, ":", child, "\n")
print("child number", i)
```

```
embedding : Embedding(10000, 3, padding_idx=1) 

conv : Sequential(
  (conv_1): Conv1d(3, 16, kernel_size=(5,), stride=(1,))
  (pool_1): MaxPool1d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
  (relu_1): ReLU()
  (conv_2): Conv1d(16, 128, kernel_size=(2,), stride=(1,))
  (pool_2): MaxPool1d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
  (relu_2): ReLU()
) 

dense : Sequential(
  (flatten): Flatten(start_dim=1, end_dim=-1)
  (linear): Linear(in_features=6144, out_features=1, bias=True)
) 

child number 3
```

```python
i = 0
for module in net.modules():
    i += 1
    print(module)
print("module number:", i)
```

```
Net(
  (embedding): Embedding(10000, 3, padding_idx=1)
  (conv): Sequential(
    (conv_1): Conv1d(3, 16, kernel_size=(5,), stride=(1,))
    (pool_1): MaxPool1d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
    (relu_1): ReLU()
    (conv_2): Conv1d(16, 128, kernel_size=(2,), stride=(1,))
    (pool_2): MaxPool1d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
    (relu_2): ReLU()
  )
  (dense): Sequential(
    (flatten): Flatten(start_dim=1, end_dim=-1)
    (linear): Linear(in_features=6144, out_features=1, bias=True)
  )
)
Embedding(10000, 3, padding_idx=1)
Sequential(
  (conv_1): Conv1d(3, 16, kernel_size=(5,), stride=(1,))
  (pool_1): MaxPool1d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
  (relu_1): ReLU()
  (conv_2): Conv1d(16, 128, kernel_size=(2,), stride=(1,))
  (pool_2): MaxPool1d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
  (relu_2): ReLU()
)
Conv1d(3, 16, kernel_size=(5,), stride=(1,))
MaxPool1d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
ReLU()
Conv1d(16, 128, kernel_size=(2,), stride=(1,))
MaxPool1d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
ReLU()
Sequential(
  (flatten): Flatten(start_dim=1, end_dim=-1)
  (linear): Linear(in_features=6144, out_features=1, bias=True)
)
Flatten(start_dim=1, end_dim=-1)
Linear(in_features=6144, out_features=1, bias=True)
module number: 12
```

* 通过 `named_children` 方法找到 `embedding` 层，并将其参数设置为不可训练，相当于冻结 embedding 层

```python
children_dict = {
    name: module for name, module in net.named_children()
}
print(children_dict)

embedding = children_dict["embedding"]
embedding.requires_grad_(False)
```

```
{'embedding': Embedding(10000, 3, padding_idx=1), 'conv': Sequential(
  (conv_1): Conv1d(3, 16, kernel_size=(5,), stride=(1,))
  (pool_1): MaxPool1d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
  (relu_1): ReLU()
  (conv_2): Conv1d(16, 128, kernel_size=(2,), stride=(1,))
  (pool_2): MaxPool1d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
  (relu_2): ReLU()
), 'dense': Sequential(
  (flatten): Flatten(start_dim=1, end_dim=-1)
  (linear): Linear(in_features=6144, out_features=1, bias=True)
)}
Embedding(10000, 3, padding_idx=1)
```

```python
# 第一层的参数已经不可以被训练
for param in embedding.parameters():
    print(param.requires_grad)
    print(param.numel())
```

```
False
30000
```

