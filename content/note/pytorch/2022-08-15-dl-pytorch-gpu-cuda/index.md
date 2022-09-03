---
title: PyTorch GPU训练 和 CUDA
author: 王哲峰
date: '2022-08-15'
slug: dl-pytorch-gup-cuda
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

- [PyTorch 使用 GPU 训练模型](#pytorch-使用-gpu-训练模型)
  - [GPU 训练模型示例](#gpu-训练模型示例)
  - [多 GPU 训练模型示例](#多-gpu-训练模型示例)
  - [矩阵乘法示例](#矩阵乘法示例)
  - [线性回归示例](#线性回归示例)
  - [图片分类示例](#图片分类示例)
  - [训练循环中使用 GPU](#训练循环中使用-gpu)
- [PyTorch CUDA](#pytorch-cuda)
  - [CUDA 语义](#cuda-语义)
  - [torch.cuda](#torchcuda)
    - [随机数生成](#随机数生成)
    - [Communication collectives](#communication-collectives)
    - [Streams 和 events](#streams-和-events)
    - [Graphs](#graphs)
    - [内存管理](#内存管理)
    - [NVIDIA 工具扩展(NVTX)](#nvidia-工具扩展nvtx)
    - [Jiterator](#jiterator)
</p></details><p></p>

# PyTorch 使用 GPU 训练模型

## GPU 训练模型示例

PyTorch 中使用 GPU 加速模型非常简单，只要将模型和数据移动到 GPU 上，核心代码只有几行

```python
import torch
print(f"torch.__version__ = {torch.__version__}")


# ------------------------------
# 设备
# ------------------------------
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"Using {device} device")

# ------------------------------
# 定义模型
# ------------------------------
class Net(torch.nn.Module):

    def __init__(self):
        super(Net, self).__init__()
    
    def forward(self, x):
        output = x
        return output

model = Net()

# ------------------------------
# 移动模型到 CUDA
# ------------------------------
model.to(device)

# ------------------------------
# 训练模型
# ------------------------------
model.fit()

# ------------------------------
# 移动数据到 CUDA
# ------------------------------
features = features.to(device)
labels = labels.to(deivce)
# labels = labels.cuda() if torch.cuda.is_available() else labels
```

## 多 GPU 训练模型示例 

如果要使用多个 GPU 训练模型，只需要将模型设置为数据并行风格模型。
模型移动到 GPU 上之后，会在每一个 GPU 上拷贝一个副本，并把数据平分到各个 GPU 上进行训练

```python
import torch
print(f"torch.__version__ = {torch.__version__}")

# ------------------------------
# 设备
# ------------------------------
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"Using {device} device")


# ------------------------------
# 定义模型
# ------------------------------
class Net(torch.nn.Module):

    def __init__(self):
        super(Net, self).__init__()
    
    def forward(self, x):
        output = x
        return output

model = Net()


# ------------------------------
# 包装为并行风格模型
# ------------------------------
if torch.cuda.device_count() > 1:
    model = nn.DataParallel(model)

# ------------------------------
# 训练模型
# ------------------------------
model.fit()

# ------------------------------
# 移动数据到 CUDA
# ------------------------------
features = features.to(device)
labels = labels.to(deivce)
# labels = labels.cuda() if torch.cuda.is_available() else labels
```

## 矩阵乘法示例





## 线性回归示例



## 图片分类示例


## 训练循环中使用 GPU









# PyTorch CUDA

## CUDA 语义

`torch.cuda` 用于设置和运行 CUDA 操作，它跟踪当前选择的 GPU，
默认情况下，分配的所有 CUDA 张量都将在该设备上创建。
可以使用 `torch.cuda.device` 上下文管理器更改所选设备


## torch.cuda

* torch.cuda 支持 CUDA tensor 类型
* 延迟初始化的，因此可以随时导入 torch.cuda
* 可以使用 is_available() 查看系统是否支持 CUDA

### 随机数生成


### Communication collectives


### Streams 和 events


### Graphs


### 内存管理


### NVIDIA 工具扩展(NVTX)


### Jiterator

