---
title: PyTorch 概述
author: 王哲峰
date: '2023-01-03'
slug: dl-pytorch-summary
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

- [PyTorch 建模流程](#pytorch-建模流程)
- [PyTorch 核心概念](#pytorch-核心概念)
- [PyTorch 层次结构](#pytorch-层次结构)
- [PyTorch 低阶 API](#pytorch-低阶-api)
- [PyTorch 中阶 API](#pytorch-中阶-api)
- [PyTorch 高阶 API](#pytorch-高阶-api)
- [PyTorch 周边工具](#pytorch-周边工具)
</p></details><p></p>

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

* 简洁易懂：PyTorch 的 API 设计的相当简洁一致。
  基本上就是 `tensor`, `autograd`, `nn` 三级封装
* 便于调试：PyTorch 采用动态图，可以像普通 Python 代码一样进行调试
* 强大高效：PyTorch 提供了非常丰富的模型组件，可以快速实现想法。
  并且运行速度很快。目前大部分深度学习相关的 Paper 都是用 PyTorch 实现的

俗话说，万丈高楼平地起，PyTorch 这座大厦也有它的地基。
PyTorch 底层最核心的概念是张量、动态计算图以及自动微分

# PyTorch 层次结构

![](https://tva1.sinaimg.cn/large/e6c9d24egy1h5hw3zgu7ij212w0lwjt3.jpg)

PyTorch 的层次结构从低到高可以分成如下五层：

* 最底层为硬件层，PyTorch 支持 CPU、GPU 加入计算资源池
* 第二层为 C++ 实现的内核
* 第三层为 Python 实现的操作符，提供了封装 C++ 内核的低级 API 指令，
  主要包括各种张量操作算子、自动微分、变量管理
    - 如 `torch.tensor`, `torch.cat`, `torch.autograd.grad`, `torch.nn.Module`
    - 如果把模型比作一个房子，那么第三层 API 就是模型之砖
* 第四层为 Python 实现的模型组件，对低级 API 进行了函数封装，
  主要包括各种模型层、损失函数、优化器、数据管道等等
    - 如 `torch.nn.Linear`, `torch.nn.BCE`, `torch.optim.Adam`, 
      `torch.utils.data.DataLoader`
    - 如果把模型比作一个房子，那么第四层 API 就是模型之墙
* 第五层为 Python 实现的模型接口，PyTorch 没有官方的高阶 API，但有一些第三方的高阶 API 库
    - [pytorch_lightning](https://www.pytorchlightning.ai/)
    - [torchkeras](https://github.com/lyhue1991/torchkeras)
        - 为了便于训练模型，这个仓库仿照 keras 中的模型接口，
          封装了 PyTorch 的高阶模型接口 `torchkeras.KerasModel`
        - 此外，该仓库同样通过引用和借鉴 `pytorch_lightning` 的一些能力，
          设计了一个和 `torchkeras.KerasModel` 功能类似的高阶模型接口 `torchkeras.LightModel`，功能更加强大
    - 如果把模型比作一个房子，那么第五层 API 就是模型本身，即模型之屋

# PyTorch 低阶 API

* 张量操作
* 动态计算图
* 自动微分

# PyTorch 中阶 API

* 数据管道
* 模型层
* 损失函数
* 优化器
* 评价指标
* TensorBoard 可视化

# PyTorch 高阶 API

PyTorch 没有官方的高阶 API，
一般通过 `torch.nn.Module` 来构建模型并编写自定义训练循环实现训练循环、验证循环、预测循环。
为了更加方便地训练模型，可以使用仿 Keras 的 PyTorch 模型接口 `torchkeras.KerasModel/LightModel`，其仓库为 [torchkeras](https://github.com/lyhue1991/torchkeras)，作为 PyTorch 的高阶 API

* 构建模型的三种方法
    - 使用 `torch.nn.Sequential` 
    - 继承 `torch.nn.Module` 基类
    - 继承 `torch.nn.Module`，辅助应用模型容器
        - `nn.Sequential`
        - `nn.ModuelList`
        - `nn.ModuleDict`
* 训练模型的三种方法
    - 脚本风格
    - 函数风格
    - `torchkeras` 类风格
* 使用 GPU 训练模型

# PyTorch 周边工具

* Kaggle GPU 使用
* Streamlit 构建机器学习应用
* Optuna 调参工具

