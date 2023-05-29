---
title: PyTorch 模型层
author: 王哲峰
date: '2022-08-12'
slug: dl-pytorch-model-layer
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

- [模型层简介](#模型层简介)
- [基础层](#基础层)
  - [全连接层](#全连接层)
  - [Embedding 层](#embedding-层)
  - [Normalization 层](#normalization-层)
    - [BatchNormalization 层](#batchnormalization-层)
    - [其他 Normalization 层](#其他-normalization-层)
  - [Dropout 层](#dropout-层)
  - [Padding 层](#padding-层)
  - [限幅层](#限幅层)
- [卷积网络相关层](#卷积网络相关层)
  - [卷积层](#卷积层)
    - [一维卷积](#一维卷积)
    - [二维卷积](#二维卷积)
    - [三维卷积](#三维卷积)
  - [池化层](#池化层)
    - [最大池化层](#最大池化层)
    - [平均池化层](#平均池化层)
  - [其他](#其他)
- [循环网络相关层](#循环网络相关层)
  - [RNN 层](#rnn-层)
  - [LSTM 层](#lstm-层)
  - [GRU 层](#gru-层)
- [Transformer 相关层](#transformer-相关层)
- [自定义模型层](#自定义模型层)
- [functional 和 Module](#functional-和-module)
  - [functional](#functional)
  - [Module](#module)
    - [使用 Module 管理参数](#使用-module-管理参数)
    - [使用 Module 管理子模块](#使用-module-管理子模块)
- [参考](#参考)
</p></details><p></p>

# 模型层简介

深度学习模型一般由各种模型层组合而称。`torch.nn` 中内置了非常丰富的各种模型层，
它们都属于 `torch.nn.Module` 的子类，具备参数管理功能

如果这些内置的模型层不能满足需求，也可以通过继承 `torch.nn.Module` 基类构建自定义的模型层。
实际上 PyTorch 不区分模型和模型层，都是通过继承 `torch.nn.Module` 进行构建，
因此，只要继承 `torch.nn.Module` 基类并实现 `forward` 方法即可自定义模型层

# 基础层

## 全连接层

`nn.Linear`：全连接层

- 参数个数 = 输入层特征数 × 输出层特征数(weight) ＋ 输出层特征数(bias)

`nn.Flatten`：压平层

- 用于将多维张量样本压成一维张量样本

## Embedding 层

`nn.Embedding`：嵌入层

- 一种比 One-Hot 更加有效的对离散特征进行编码的方法
- 一般用于将输入中的单词映射为稠密向量，嵌入层的参数需要学习

## Normalization 层

> 归一化层，Normalization

### BatchNormalization 层

* `nn.BatchNorm1d`：一维批标准化层
    - 通过线性变换将输入批次缩放平移到稳定的均值和标准差。可以增强模型对输入不同分布的适应性，加快模型训练速度，有轻微正则化效果
    - 一般在激活函数之前使用。可以用 `affine` 参数设置该层是否含有可以训练的参数
* `nn.BatchNorm2d`：二维批标准化层
    - 常用于 CV 领域
* `nn.BatchNorm3d`：三维批标准化层

### 其他 Normalization 层

* `nn.GroupNorm`：组归一化
    - 一种替代批归一化的方法，将通道分成若干组进行归一，不受 batch 大小限制
* `nn.LayerNorm`：层归一化
    - 常用于 NLP 领域，不受序列长度不一致影响
* `nn.InstanceNorm2d`：样本归一化
    - 一般在图像风格迁移任务中效果较好
* `SwitchableNorm`：可自适应归一化
    - 将 BatchNorm、LayerNorm、InstanceNorm 结合，赋予权重，让网络自己去学习归一化层应该使用什么方法
 
## Dropout 层

Dropout 层是一种正则化手段

* `nn.Dropout`：一维随机丢弃层
* `nn.Dropout2d`：二维随机丢弃层
* `nn.Dropout3d`：三维随机丢弃层

## Padding 层

* `nn.ReplicationPad1d`：一维复制填充层
    - 对一维张量样本通过复制边缘值填充扩展长度
* `nn.ZeroPad2d`：二维零值填充层
    - 对二维张量样本在边缘填充 0 值
* `nn.ConstantPad2d`：二维常数填充层
    - 对二维张量样本在边缘填充常数扩展长度

## 限幅层

* `nn.Threshold`：限幅层
    - 当输入大于或小于阈值范围时，截断之

# 卷积网络相关层

## 卷积层

### 一维卷积

`nn.Conv1d`：普通一维卷积，常用于文本

* 参数个数 = 输入通道数 × 卷积核尺寸(如3) × 卷积核个数 + 卷积核尺寸(如3)

### 二维卷积

`nn.Conv2d`：普通二维卷积，常用于图像

* 参数个数 = 输入通道数 × 卷积核尺寸(如 3x3) × 卷积核个数 + 卷积核尺寸(如 3x3) 
* 空洞卷积：
    - 通过调整 `dilation` 参数大于 1，可以变成空洞卷积，增加感受野
* 分组卷积(二维深度卷积)
    - 通过调整 `groups` 参数不为 1，可以变成分组卷积。分组卷积中每个卷积核仅对其对应的一个分组进行操作。
      当 `groups` 参数数量等于输入通道数时
    - 相当于 TensorFlow 中的二维深度卷积层 `tf.keras.layers.DepthwiseConv2D`
* 二维深度可分离卷积
    - 利用分组卷积和 `$1 \times 1$` 卷积的组合操作，可以构造相当于 TensorFlow 中的二维深度可分离卷积层 `tf.keras.layers.SeparableConv2D`

### 三维卷积

`nn.Conv3d`：普通三维卷积，常用于视频

- 参数个数 = 输入通道数 × 卷积核尺寸(如 3x3x3) × 卷积核个数 + 卷积核尺寸(如 3x3x3)

## 池化层


### 最大池化层

* `nn.MaxPool1d`：一维最大池化
* `nn.MaxPool2d`：二维最大池化
    - 一种下采样方式，没有需要训练的参数
* `nn.MaxPool3d`：三维最大池化
* `nn.AdaptiveMaxPool2d`：二维自适应最大池化
    - 无论输入图像的尺寸如何变化，输出的图像尺寸是固定的
    - 该函数的实现原理，大概是通过输入图像的尺寸和要得到的输出图像的尺寸来反向推算池化算子的 `padding`、`stride` 等参数
* `nn.FractionalMaxPool2d`：二维分数最大池化
    - 普通最大池化通常输入尺寸是输出的整数倍，而分数最大池化则可以不必是整数
    - 分数最大池化使用了一些随机采样策略，有一定的正则效果，可以用它来代替普通最大池化和 Dropout 层

### 平均池化层

* `nn.AvgPool2d`：二维平均池化
* `nn.AdaptiveAvgPool2d`：二维自适应平均池化
    - 无论输入的维度如何变化，输出的维度是固定的

## 其他

- `nn.ConvTranspose2d`：二维卷积转置层，俗称反卷积层
    - 并非卷积的逆操作，但在卷积核相同的情况下，当其输入尺寸是卷积操作输出尺寸的情况下，
      卷积转置的输出尺寸恰好是卷积操作的输入尺寸。在语义分割中可用于上采样
- `nn.Upsample`：上采样层，操作效果和池化相反。可以通过 `mode` 参数控制上采样策略为：
    - `"nearest"`：最邻近策略
    - `"linear"`：线性插值策略
* `nn.Fold`：逆滑动窗口提取层
* `nn.Unfold`：滑动窗口提取层
    - 其参数和卷积操作 `nn.Conv2d` 相同
    - 实际上，卷积操作可以等价于 `nn.Unfold` 和 `nn.Linear` 以及 `nn.Fold` 的一个组合。
      其中 `nn.Unfold` 操作可以从输入中提取各个滑动窗口的数值矩阵，并将其压平成一维。
      利用 `nn.Linear` 将 `nn.Unfold` 的输出和卷积核做乘法后，
      再使用 `nn.Fold` 操作将结果转换成输出图片形状

# 循环网络相关层

## RNN 层

* `nn.RNN`：简单循环网络层(支持多层)
    - 容易存在梯度消失，不能够适用长期依赖问题。一般较少使用
* `nn.RNNCell`：简单循环网络单元
    - 和 `nn.RNN` 在整个序列上迭代相比，它仅在序列上迭代一步。一般较少使用

## LSTM 层

* `nn.LSTM`：长短记忆循环网络层(支持多层)
    - 最普遍使用的循环网络层。具有携带轨道、遗忘门、更新门、输出门，可以较为有效地缓解梯度消失问题，从而能够适用长期依赖问题
    - 设置 `bidirectional = True` 时可以得到双向 LSTM
    - 需要注意的时，默认的输入和输出形状是 `(seq, batch, feature)`，如果需要将 `batch` 维度放在第 `0` 维，则要设置 `batch_first = True`
* `nn.LSTMCell`：长短记忆循环网络单元
    - 和 `nn.LSTM` 在整个序列上迭代相比，它仅在序列上迭代一步。一般较少使用

## GRU 层

* `nn.GRU`：门控循环网络层(支持多层)
    - LSTM 的低配版，不具有携带轨道，参数数量少于 LSTM，训练速度更快
* `nn.GRUCell`：门控循环网络单元
    - 和 `nn.GRU` 在整个序列上迭代相比，它仅在序列上迭代一步。一般较少使用

# Transformer 相关层

Transformer 网络结构是替代循环网络的一种结构，解决了循环网络难以并行，难以捕捉长期依赖的缺陷。
它是目前 NLP 任务的主流模型的主要构成部分

* `nn.Transformer`：Transformer 网络结构
* `nn.TransformerEncoder`：Transformer 编码器结构，由多个 `nn.TransformerEncoderLayer` 编码器层组成
* `nn.TransformerDecoder`：Transformer 解码器结构，由多个 `nn.TransformerDecoderLayer` 解码器层组成
* `nn.TransformerEncoderLayer`：Transformer 的编码器层。主要由以下结构组成：
    - Multi-Head self-Attention
    - Feed-Forward 前馈网络
    - LayerNorm 归一化层
    - 残差连接层
* `nn.TransformerDecoderLayer`：Transformer 的解码器层。主要由以下结构组成：
    - Masked Multi-Head self-Attention
    - Multi-Head cross-Attention
    - Feed-Forward 前馈网络
    - LayerNorm 归一化层
    - 残差连接层
* `nn.MultiheadAttention`：多头注意力层(Multi-Head self-Attention)
    - 用于在序列方向上融合特征
    - 使用的是 Scaled Dot Production Attention，并引入了多个注意力头

# 自定义模型层

如果这些内置的模型层不能满足需求，也可以构建自定义的模型层。实际上 PyTorch 不区分模型和模型层，
因此，只要继承 `torch.nn.Module` 基类并实现 `forward` 方法即可自定义模型层

可以仿照下面的 `torch.nn.Linear` 层源码自定义模型层：

```python
import math

import torch
from torch import nn
import torch.nn.functional as F


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
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.weight, a = math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = nn.init.calculate_fan_in_and_fan_out(
                self.weight
            )
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.bias, -bound, bound)
    
    def forward(self, input):
        return F.linear(input, self.weight, self.bias)

    def extra_repr(self):
        return f"in_features={self.in_features}, \
                 out_features={self.out_features}, \
                 bias={self.bias is not None}"
```

# functional 和 Module

PyTorch 和神经网络相关的功能组件大多都封装在 `torch.nn` 模块下，
这些功能组件的绝大部分既有函数形式实现，也有类形式实现

* 函数形式：`torch.nn.functional`
* 类形式：`torch.nn.Module`

## functional

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

示例：

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

## Module

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

### 使用 Module 管理参数

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


### 使用 Module 管理子模块

一般情况下，很少直接使用 `nn.Parameter` 来定义参数构建模型，而是通过一些拼装一些常用的模型层来构造模型。
这些模型层也是继承自 `nn.Module` 的对象，本身也包括参数，属于要定义的模块的子模块

`nn.Module` 提供了一些方法可以管理这些子模块:

* `children()`：返回生成器，包括模块下的所有子模块
* `named_children()`：返回一个生成器，包括模块下的所有子模块，以及它们的名字
* `modules()`：返回一个生成器，包括模块下的所有各个层级的模块，包括模块本身
* `named_modules()`：返回一个生成器，包括模块下的所有各个层级的模块以及它们的名字，包括模块本身

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

# 参考

