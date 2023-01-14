---
title: PyTorch 总结
author: 王哲峰
date: '2023-01-03'
slug: dl-pytorch-base
categories:
  - pytorch
tags:
  - tool
---

# PyTorch 建模流程

使用 PyTorch 实现神经网络模型的一般流程包括：

1. 准备数据
2. 定义模型
3. 训练模型
4. 评估模型
5. 使用模型
6. 保存模型

# PyTorch 核心概念

PyTorch 是一个基于 Python 的机器学习库。它主要提供了两种核心功能：

1. 支持 GPU 加速的张量计算
2. 方便优化模型的自动微分机制

PyTorch 的主要优点：

* 简洁易懂：Pytorch 的 API 设计的相当简洁一致。基本上就是 `tensor`, `autograd`, `nn`三级封装
* 便于调试：PyTorch 采用动态图，可以像普通 Python 代码一样进行调试
* 强大高效：PyTorch 提供了非常丰富的模型组件，可以快速实现想法。
  并且运行速度很快。目前大部分深度学习相关的 Paper 都是用 Pytorch 实现的

俗话说，万丈高楼平地起，PyTorch 这座大厦也有它的地基。
PyTorch 底层最核心的概念是张量，动态计算图以及自动微分

# PyTorch 层次结构

![](https://tva1.sinaimg.cn/large/e6c9d24egy1h5hw3zgu7ij212w0lwjt3.jpg)

PyTorch 的层次结构从低到高可以分成如下五层：

* 最底层为硬件层，PyTorch 支持 CPU、GPU 加入计算资源池
* 第二层为 C++ 实现的内核
* 第三层为 Python 实现的操作符，提供了封装 C++ 内核的低级 API 指令，
  主要包括各种张量操作算子、自动微分、变量管理. 如 `torch.tensor`, 
  `torch.cat`, `torch.autograd.grad`, `torch.nn.Module`. 
  如果把模型比作一个房子，那么第三层 API 就是模型之砖
* 第四层为 Python 实现的模型组件，对低级 API 进行了函数封装，
  主要包括各种模型层，损失函数，优化器，数据管道等等。
  如 `torch.nn.Linear`, `torch.nn.BCE`, `torch.optim.Adam`, 
  `torch.utils.data.DataLoader`. 如果把模型比作一个房子，
  那么第四层 API 就是模型之墙
* 第五层为 Python 实现的模型接口。Pytorch 没有官方的高阶 API。
  为了便于训练模型，仿照 keras 中的模型接口，使用了不到 300 行代码，
  封装了 PyTorch 的高阶模型接口 `torchkeras.KerasModel`。
  此外，有一个非常流行的非官方 `PyTorch` 的高阶API库，
  叫做 `pytorch_lightning`, 作者通过引用和借鉴它的一些能力，
  设计了一个和 `torchkeras.KerasModel` 功能类似的高阶模型接口 `torchkeras.LightModel`，
  功能更加强大。如果把模型比作一个房子，那么第五层API就是模型本身，即模型之屋

# PyTorch 低阶 API

PyTorch 低阶 API 主要包括:

* 张量操作
* 动态计算图
* 自动微分

# PyTorch 中阶 API

PyTorch 中阶 API 主要包括

* 数据管道
* 模型层
* 损失函数
* 优化器
* TensorBoard 可视化

# PyTorch 高阶 API

PyTorch 没有官方的高阶 API，一般通过 `nn.Module` 来构建模型并编写自定义训练循环实现训练循环、验证循环、预测循环。
为了更加方便地训练模型，可以使用仿 Keras 的 PyTorch 模型接口 `torchkeras.KerasModel/LightModel`，
作为 Pytorch 的高阶 API

* 构建模型的三种方法
    - 使用 `torch.nn.Sequential` 
    - 继承 `torch.nn.Module` 基类
    - 继承 `torch.nn.Module`，辅助应用模型容器
* 训练模型的三种方法
    - 脚本风格
    - 函数风格
    - `torchkeras` 类风格
* 使用 GPU 训练模型


# PyTorch 与广告推荐

根据业务和技术方向的差异性，目前工业界的算法工程师主要可以分成：

* 广告算法工程师
* 推荐算法工程师
* 风控算法工程师
* CV 算法工程师
* NLP 算法工程师
* ...

除了一些多模态数据外(文本、图像、社交网络等)，广告算法工程师、推荐算法工程师、
风控算法工程师处理的数据类型主要还是是结构化数据。
风控领域目前工业界用到的模型还是树模型为主，广告和推荐领域目前工业界是深度模型为主。
由于数据结构的相似性，广告的 CTR 预估模型和推荐系统的精排模型，基本是通用的。
广告、推荐领域常用的神经网络模型为：

* FM
* DeepFM
* FiBiNET
* DeepCross
* DIN
* DIEN

# PyTorch 周边工具

* Kaggle GPU 使用
* Streamlit 构建机器学习应用
* Optuna 调参工具


## Kaggle GPU


## Streamlit



## Optuna



