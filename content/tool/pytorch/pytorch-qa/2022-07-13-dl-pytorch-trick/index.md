---
title: PyTorch 技巧
author: 王哲峰
date: '2022-07-13'
slug: dl-pytorch-trick
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

- [基本配置](#基本配置)
  - [导入包和版本查询](#导入包和版本查询)
  - [可复现性](#可复现性)
    - [随机种子](#随机种子)
    - [Benchmark](#benchmark)
- [GPU 设置](#gpu-设置)
  - [GPU 参数查看](#gpu-参数查看)
  - [获取 GPU ID 信息](#获取-gpu-id-信息)
  - [GPU 启动模式设置](#gpu-启动模式设置)
  - [节点分配](#节点分配)
  - [指定 GPU 编号](#指定-gpu-编号)
  - [清除显存](#清除显存)
- [Tensor 数据类型](#tensor-数据类型)
  - [tensor 数据类型概述](#tensor-数据类型概述)
  - [tensor 类型](#tensor-类型)
  - [tensor 命名](#tensor-命名)
  - [tensor 类型转换](#tensor-类型转换)
    - [设置默认类型](#设置默认类型)
    - [torch tensor 与 numpy ndarray 转换](#torch-tensor-与-numpy-ndarray-转换)
- [Tensor 操作](#tensor-操作)
  - [](#)
- [模型参数](#模型参数)
  - [计算模型参数量](#计算模型参数量)
  - [查看模型参数](#查看模型参数)
  - [模型信息打印](#模型信息打印)
  - [模型可视化](#模型可视化)
- [梯度](#梯度)
  - [梯度裁剪](#梯度裁剪)
- [学习率](#学习率)
  - [学习率衰减](#学习率衰减)
- [参考](#参考)
</p></details><p></p>

> 所有基于 GPU 的操作都是在 Colab 上进行

# 基本配置

## 导入包和版本查询

```python
import torch
import torch.nn as nn

print(torch.__version__)
print(torch.version.cuda)
print(torch.backends.cudnn.version())
print(torch.cuda.get_device_name(0))
```

```
# colab GPU/T4
2.0.1+cu118
11.8
8700
Tesla T4
```

## 可复现性

### 随机种子

在硬件设备(CPU、GPU)不同时，完全的可复现性无法保证，即使随机种子相同。但是，在同一个设备上，应该保证可复现性。
具体做法是，在程序开始的时候固定 `torch` 的随机种子，同时也把 `numpy` 的随机种子固定

```python
import random
import numpy as np
import torch

fix_seed = 2021
random.seed(fix_seed)
np.random.seed(fix_seed)
torch.manual_seed(fix_seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(0)
``` 

### Benchmark

```python
if torch.cuda.is_available():
    torch.backends.cudnn.deterministic = True
    # cuDNN 使用非确定性算法，可以使用 torch.backends.cudnn.enabled = False 来进行禁用
    torch.backends.cudnn.benchmark = False
```

设置 cuDNN 使用非确定性算法，设置这个 flag 可以让内置的 cuDNN 的 auto-tuner 自动寻找最适合当前配置的高效算法，
来达到优化运行效率的问题。将会让程序在开始时花费一点额外时间，为整个网络的每个卷积层搜索最适合它的卷积实现算法，进而实现网络的加速：

```python
torch.backends.cudnn.enable = True
```

一般来讲，应该遵循以下准则：

1. 如果网络的输入数据维度或类型上变化不大，输入形状包括 batch size，图片大小，输入的通道，上述设置会增加运行效率
2. 如果卷积层的设置一直变化，网络的输入数据在每次 iteration 都变化的话，会导致 cnDNN 每次都会去寻找一遍最优配置，
   这样反而会降低运行效率

设置 cuDNN 不使用非确定算法：

```python
torch.backends.cudnn.benchmark = False
```

benchmark 模式会提升计算速度，但是由于计算中有随机性，每次网络前馈结果略有差异。
如果想要避免这种结果波动，设置：

```python
torch.backends.cudnn.deterministic = True
```

# GPU 设置

## GPU 参数查看

NVIDIA系统管理界面(`nvidia-smi`)是基于 NVIDIA Management Library(NVML) 的命令行实用程序，
旨在帮助管理和监视 NVIDIA GPU 设备

```bash
$ nvidia-smi
```

```
Sun Mar 28 02:40:38 2021
+-----------------------------------------------------------------------------+
| NVIDIA-SMI 418.56       Driver Version: 418.56       CUDA Version: 10.1     |
|-------------------------------+----------------------+----------------------+
| GPU  Name        Persistence-M| Bus-Id        Disp.A | Volatile Uncorr. ECC |
| Fan  Temp  Perf  Pwr:Usage/Cap|         Memory-Usage | GPU-Util  Compute M. |
|===============================+======================+======================|
|   0  GeForce GTX 108...  On   | 00000000:02:00.0 Off |                  N/A |
| 23%   29C    P8     9W / 250W |    611MiB / 11178MiB |      0%      Default |
+-------------------------------+----------------------+----------------------+
|   1  GeForce GTX 108...  On   | 00000000:03:00.0 Off |                  N/A |
| 23%   30C    P8     9W / 250W |      0MiB / 11178MiB |      0%      Default |
+-------------------------------+----------------------+----------------------+
|   2  GeForce GTX 108...  On   | 00000000:82:00.0 Off |                  N/A |
| 23%   30C    P8     9W / 250W |      0MiB / 11178MiB |      0%      Default |
+-------------------------------+----------------------+----------------------+
|   3  GeForce GTX 108...  On   | 00000000:83:00.0 Off |                  N/A |
| 23%   30C    P8     9W / 250W |      0MiB / 11178MiB |      0%      Default |
+-------------------------------+----------------------+----------------------+

+-----------------------------------------------------------------------------+
| Processes:                                                       GPU Memory |
|  GPU       PID   Type   Process name                             Usage      |
|=============================================================================|
|    0     33777      C   /usr/bin/python                              601MiB |
+-----------------------------------------------------------------------------+
```

上面是 `GEFORCE GTX 1080 Ti` GPU 服务器的运行信息：

* 第一行分别为：命令行工具版本、GPU 驱动版本、CUDA 版本
* 第一栏分别为：`GPU`(GPU 卡号，`0`～`4`)、`Fan`(风扇转速，`0`～`100%`)
* 第二栏分别为：`Name`(显卡名字)、`Temp`(温度，摄氏度)
* 第三栏分别为：`Perf`(性能状态，`P0`~`P12`，最高性能为 `P0`，最低性能为 `P12`)
* 第四栏分别为：`Persistence-M`(持续模式，默认为关闭，比较节能，如果设置成 `ON`，耗能比较大，但新的 GPU 应用启动时，花费的时间会更短)、`Pwr:Usage/Cap`(能耗)
* 第五栏分别为：`Bus-Id`(GPU 总线，`domain:bus:device.function`)
* 第六栏分别为：`Disp.A`(GPU 的显示是否初始化)、`Memory-Usage`(显存利用率)
* 第七栏分别为：`Volatile GPU-Util`(GPU 浮动利用率)
* 第八栏分别为：`Uncorr. ECC`(Error Correcting Code 错误检查和纠正码)、`Compute M.`(计算模式)

下面一张表为：每个 GPU Processes 的资源占用情况

> 注：显存占用和 GPU 占用是不一样的，显卡是由 GPU 和显存等组成的，显存和 GPU 的关系可简单理解为内存和 CPU 的关系

## 获取 GPU ID 信息

从左到右依次为：GPU 卡号、GPU 型号、GPU 物理 UUIID 号

```bash
$ nvidia-smi -L
```

```
GPU 0: GeForce GTX 1080 Ti (UUID: GPU-5da6e67e-fd5a-88fb-7a0e-109c3284f7bf)
GPU 1: GeForce GTX 1080 Ti (UUID: GPU-ce9189e4-2e58-3a19-4332-cb5c7fac1aa6)
GPU 2: GeForce GTX 1080 Ti (UUID: GPU-242b3020-8e5c-813a-42d9-475766d52f9d)
GPU 3: GeForce GTX 1080 Ti (UUID: GPU-8f3d825f-7246-3daf-eaa1-37845b03aa03)
```

单独过滤出 GPU 卡号信息：

```bash
$ nvidia-smi -L | cut -d ' ' -f 2 | cut -c 1
```

```
0
1
2
3
```

## GPU 启动模式设置

```bash
# 设置 GPU 持续模式：Persistence-M
$ sudo nvidia-smi -pm 1
```

## 节点分配

解决卡性能不均匀问题，如果是四卡机器，只使用两个节点优先选择 `0` 和 `3`，边界卡槽有利于散热

## 指定 GPU 编号

如果只需要一张显卡，设置当前使用的 GPU 设备仅为 0 号设备，设备名称为 `/gpu:0`

```python
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)
```

```
device(type="cuda")
```

或：

```python
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
print(os.environ["CUDA_VISIBLE_DEVICES"])
```

```
0
```

如果需要指定多张显卡，比如 0、1 号 显卡，设置当前使用的 GPU 设备为 0、1 号两个设备，
名称依次为 `/gpu:0`、`/gpu:1`。根据顺序表示优先使用 0 号设备，然后使用 1 号设备

```python
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"
```

```
0,1
```

在命令行运行时设置 GPU：

```bash
$ CUDA_VISIBLE_DEVICES=0,1 python train.py
```

## 清除显存

```python
torch.cuda.empty_cache()
```

或者，可以使用命令行重置 GPU 指令：

```bash
$ nvidia-smi --gpu-reset -i [gpu_id]
```

```
GPU 00000000:00:04.0 is currently in use by another process.

1 device is currently being used by one or more other processes (e.g., Fabric Manager, CUDA application, graphics application such as an X server, or a monitoring application such as another instance of nvidia-smi). Please first kill all processes using this device and all compute applications running in the system.
```

# Tensor 数据类型

## tensor 数据类型概述

PyTorch 有 9 种 CPU Tensor 类型和 9 种 GPU Tensor 类型：

| 数据类型 | dtype | CPU tensor dtype | GPU tensor dtype |
|---------|----|----|----|
| 16-bit floating | `torch.float16` | `torch.HalfTensor` | `torch.cuda.HalfTensor` |
| 32-bit floating | `torch.float32` | `torch.FloatTensor` | `torch.cuda.FloatTensor` |
| 64-bit floating | `torch.float64` | `torch.DoubleTensor` | `torch.cuda.DoubleTensor` |
| 8-bit integer(unsigned) | `torch.uint8` | `torch.ByteTensor` | `torch.cuda.ByteTensor` |
| 8-bit integer | `torch.int8` | `torch.CharTensor` | `torch.cuda.CharTensor` |
| 16-bit integer | `torch.int16` | `torch.ShortTensor` | `torch.cuda.ShortTensor` |
| 32-bit integer | `torch.int32` | `torch.IntTensor` | `torch.cuda.IntTensor` |
| 64-bit integer | `torch.int64` | `torch.LongTensor` | `torch.cuda.LongTensor` |
| Boolean | `torch.bool` | `torch.BoolTensor` | `torch.cuda.BoolTensor` |

## tensor 类型

```python
tensor = torch.randn(3, 4, 5)
print(tensor.type())
```

## tensor 命名

> PyTorch 1.3 之后版本

```python
NCHW = ["N", "C", "H", "W"]

images = torch.randn(32, 3, 56, 56, names = NCHW)
images.sum("C")
images.select("C", index = 0)

# 排序
tensor = tensor.align_to("N", "C", "H", "W")
```

## tensor 类型转换

### 设置默认类型

PyTorch 中的 `FloatTensor` 远远快于 `DoubleTensor`

```python
torch.set_default_tensor_type(torch.FloatTensor)

# 类型转换
tensor = tensor.cuda()
tensor = tensor.cpu()
tensor = tensor.float()
tensor = tensor.long()
```

### torch tensor 与 numpy ndarray 转换

除了 `CharTensor`，其他所有 CPU 上的张量都支持转换为 numpy 格式然后再转换回来

# Tensor 操作

## 






# 模型参数

## 计算模型参数量

```python
num_params = sum(torch.numel(param) for param in model.parameters())
```

## 查看模型参数

可以通过 `model.state_dict()` 或者 `model.named_parameters()` 函数查看现在的全部可训练参数(包括通过继承得到的父类中的参数)

```python
params = list(model.named_parameters())
(name, param) = param[28]
print(name)
print(param)
print(param.grad)
```

## 模型信息打印

> torchinfo

```python
from torchinfo import summary

model_stats = summary(your_model, (1, 3, 28, 28), verbose=0)
summary_str = str(model_stats)
```

* [torchinfo](https://github.com/TylerYep/torchinfo)

## 模型可视化

> torchviz

```python
import torch.nn as nn
from torchviz import make_dot

model = nn.Sequential()
model.add_module('W0', nn.Linear(8, 16))
model.add_module('tanh', nn.Tanh())
model.add_module('W1', nn.Linear(16, 1))

x = torch.randn(1, 8)
y = model(x)

make_dot(y.mean(), params=dict(model.named_parameters()), show_attrs=True, show_saved=True)
```

* [torchviz](https://github.com/szagoruyko/pytorchviz)

# 梯度

## 梯度裁剪

> Gradient Clipping

梯度裁剪在某些任务上会额外消耗大量的计算时间

```python
import torch.nn as nn

outputs = model(data)

# forward
loss = loss_fn(outputs, target)
optimizer.zero_grad()

# bakcward
loss.backward()
nn.utils.clip_grad_norm_(
    model.parameters(), 
    max_norm = 20, 
    norm_type = 2
)
optimizer.step()
```

`nn.utils.clip_grad_norm_` 的参数：

* `parameters`：一个基于变量的迭代器，会进行梯度归一化
* `max_norm`：梯度的最大范数
* `norm_type`：规定范数的类型，默认为 L2

# 学习率

## 学习率衰减

```python
import torch.optim as optim
from torch.optim import lr_scheduler

# 训练前的初始化
optimizer = optim.Adam(net.parameters(), lr = 1e-3)
scheduler = lr_scheduler.StepLR(optimizer, 10, 0.1)

# 训练过程中
for n in n_epoch:
    scheduler.step()
```

# 参考

* [PyTorch Tricks](https://github.com/demuxin/pytorch_tricks)
* [PyTorch 高频代码段集锦](https://mp.weixin.qq.com/s/VcczUaRSbr3zGzKiuxT9JQ)
* [竞赛中神经网络训练技巧汇总](https://mp.weixin.qq.com/s/o0snk_dsLXJGRvtdMrq_uw)
