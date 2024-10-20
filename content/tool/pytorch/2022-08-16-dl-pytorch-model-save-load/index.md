---
title: PyTorch 模型保存和加载
author: wangzf
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
img {
    pointer-events: none;
}
</style>

<details><summary>目录</summary><p>

- [模型保存与加载简介](#模型保存与加载简介)
    - [序列化与反序列化](#序列化与反序列化)
    - [模型保存和加载的两种方式](#模型保存和加载的两种方式)
    - [checkpoint resume](#checkpoint-resume)
- [保存和加载整个模型](#保存和加载整个模型)
    - [模型保存](#模型保存)
    - [模型加载](#模型加载)
- [保存和加载模型权重参数](#保存和加载模型权重参数)
    - [模型保存](#模型保存-1)
    - [模型加载](#模型加载-1)
- [保存和加载模型 checkpoint](#保存和加载模型-checkpoint)
    - [定义和初始化模型](#定义和初始化模型)
    - [初始化优化器](#初始化优化器)
    - [保存 checkpoint](#保存-checkpoint)
    - [加载 checkpoint](#加载-checkpoint)
- [参考](#参考)
</p></details><p></p>

# 模型保存与加载简介

## 序列化与反序列化

因为在内存中的数据运行结束会进行释放，所以需要将数据保存到硬盘中，
以二进制序列的形式进行长久存储，便于日后使用。

在 PyTorch 中，对象就是模型，所以序列化和反序列化，
就是将训练好的模型从内存中保存到硬盘里，当要使用的时候，再从硬盘中加载。

PyTorch 提供的序列化和反序列化的 API 如下：

* **序列化** 即把对象转换为字节序列的过程
    - 功能：保存对象到硬盘中
    - 主要参数：
        - `obj` – saved object
        - `f` – a file-like object (has to implement write and flush) or 
          a string or os.PathLike object containing a file name
        - `pickle_module` – module used for pickling metadata and objects
        - `pickle_protocol` – can be specified to override the default protocol

```python
torch.save(
    obj, 
    f, 
    pickle_module = pickle, 
    pickle_protocol = DEFAULT_PROTOCOL, 
    _use_new_zipfile_serialization = True,
)
```

* **反序列化** 则把字节序列恢复为对象
    - 功能：加载硬盘中对象
    - 主要参数： 
        - `f`：文件路径
        - `map_location`：指定存储位置
            - 如 `map_location='cpu'`、`map_location={'cuda:1':'cuda:0'}`
            - `map_location` 经常需要手动设置，否者会报错。具体可参考以下形式：
                - GPU->CPU：`torch.load(model_path, map_location = 'cpu')`
                - CPU->GPU：`torch.load(model_path, map_location = lambda storage, loc: storage)`
        - `pickle_module`
        - `weights_only`
        - `mmap`
        - `pickle_load_args`

```python
torch.load(
    f, 
    map_location = None, 
    pickle_module = pickle, 
    *, 
    weights_only = False, 
    mmap = None, 
    **pickle_load_args
)
```

## 模型保存和加载的两种方式

一个 Module 当中包含了很多信息，不仅仅是模型的参数 `parameters`，
还包含了 `buffers`、`hooks` 和 `modules` 等一系列信息。
对于模型应用，最重要的是模型的 `parameters`，
其余的信息是可以通过 `model` 类再去构建的，所以模型保存就有两种方式：

* 整个模型
* 模型权重参数

通常，只需要保存模型的参数，在使用的时候再通过 `torch.load_state_dict()` 方法加载参数。
由于第一种方法不常用，并且在加载过程中还需要指定的类方法，而第二种方法代码非常简单，比较常用。

## checkpoint resume

在模型开发过程中，往往不是一次就能训练好模型，经常需要反复训练，
因此需要保存训练的 “状态信息”，以便于基于某个状态继续训练，这就是常说的 resume，
可以理解为 **断点续训练**。

在整个训练阶段，除了模型参数需要保存，还有优化器的参数、
学习率调整器的参数和迭代次数等信息也需要保存，因此推荐在训练时，
采用以下代码段进行模型保存。

```python
checkpoint = {
    "model": model_without_ddp.state_dict(),
    "optimizer": optimizer.state_dict(),
    "lr_scheduler": lr_scheduler.state_dict(),
    "epoch": epoch,
}

# save
path_save = "model_{}.pth".format(epoch)
torch.save(checkpoint, path_save)

# resume
checkpoint = torch.load(path_save, map_location = "cpu")
model.load_state_dict(checkpoint["model"])
optimizer.load_state_dict(checkpoint["optimizer"])
lr_scheduler.load_state_dict(checkpoint["lr_scheduler"])
start_epoch = checkpoint["epoch"] + 1
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

# 保存和加载模型权重参数

PyTorch 将模型训练学习到的权重参数保存在一个状态字典 `state_dict` 中。

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
# do not specify pretrained=True, i.e. do not load default weights
model = models.vgg16()  
model.load_state_dict(
    torch.load(MODEL_PATH)
)
model.eval()
```

# 保存和加载模型 checkpoint

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

* `torch.save({}, model_path)`

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



# 参考

* [模型保存与加载](https://tingsongyu.github.io/PyTorch-Tutorial-2nd/chapter-7/7.1-serialization.html)
