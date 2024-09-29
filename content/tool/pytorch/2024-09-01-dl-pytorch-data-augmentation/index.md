---
title: PyTorch 数据增强
author: 王哲峰
date: '2024-09-01'
slug: dl-pytorch-data-augmentation
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

- [数据增强](#数据增强)
- [PyTorch transforms](#pytorch-transforms)
    - [torchvision transforms](#torchvision-transforms)
        - [transforms 简介](#transforms-简介)
        - [transforms 机制](#transforms-机制)
        - [常用转换](#常用转换)
    - [torchtext transforms](#torchtext-transforms)
        - [transforms 简介](#transforms-简介-1)
        - [常用转换](#常用转换-1)
    - [torchaudio transforms](#torchaudio-transforms)
        - [transforms 简介](#transforms-简介-2)
        - [常用转换](#常用转换-2)
- [AIbumentations](#aibumentations)
- [参考](#参考)
</p></details><p></p>

# 数据增强

数据增强(Data Augmentation)已经成为深度学习时代的常规做法，
数据增强目的是为了增加训练数据的丰富度，
让模型接触多样性的数据以增加模型的泛化能力。

通常，数据增强可分为在线(online)与离线(offline)两种方式：

* 离线方式指的是在训练开始之前将数据进行变换，变换后的图片保存到硬盘当中；
* 在线方式则是在训练过程中，每一次加载训练数据时对数据进行变换，
  以实现让模型看到的图片都是增强之后的。
  
实际上，这两种方法理论上是等价的，一般的框架都采用在线方式的数据增强，
PyTorch 的 `transforms` 就是在线方式。

# PyTorch transforms

可以使用 `transforms` 对数据集进行转换操作，使得数据集可以作为机器学习算法可以使用的形式：

* `torchvision.transforms`
* `torchtext.transforms`
* `torchaudio.transforms`

## torchvision transforms

> `torchvision.transforms`

### transforms 简介

`torchvision.transforms` 是广泛使用的图像变换库，包含二十多种基础方法以及多种组合功能，
通常可以用 `torchvision.transforms.Compose([])` 把各方法串联在一起使用。
大多数的 `transforms` 类都有对应的 functional transforms，可供用户自定义调整。

在 `torchvision.transforms` 库中包含二十多种变换方法，那么多的方法里应该如何挑选，
以及如何设置参数呢？**数据增强的方向一定是测试数据集中可能存在的情况。**
举个例子，做人脸检测可以用水平翻转（如前置相机的镜像就是水平翻转），
但不宜采用垂直翻转（这里指一般业务场景，特殊业务场景有垂直翻转的人脸就另说）。
因为真实应用场景不存在倒转（垂直翻转）的人脸，因此在训练过程选择数据增强时就不应包含垂直翻转。

所有的 `torchvision` datasets 都有两个接受包含转换逻辑的可调用对象的参数:

* `transform`
    - 修改特征
* `target_transform`
    - 修改标签

大部分 transform 同时接受 PIL 图像和 tensor 图像，
但是也有一些 tansform 只接受 PIL 图像，或只接受 tensor 图像。
对于 tensor 图像，`transform` 接受 tensor 图像或批量 tensor 图像：

* tensor 图像的 shape 格式是 `(C, H, W)`
* 批量 tensor 图像的 shape 格式是 `(B, C, H, W)`

### transforms 机制

开始采用 `torchvision.transforms.Compose` 把变换的方法包装起来，放到 `Dataset` 中；
在 `DataLoader` 依次读数据时，调用 `Dataset` 的 `__getitem__`，
每个 `sample` 读取时，会根据 `compose` 里的方法依次地对数据进行变换，
以此完成在线数据增强。而具体的 `transforms` 方法通常包装成一个 `Module` 类，
具体实现会在各 `functional` 中。

### 常用转换

`torchvision.transform` 模块提供了多个常用转换

* Scriptable transforms
    - `torch.nn.Sequential`
    - `torch.jit.script`
* Compositions of transforms
    - `Compose`: 将多个 transform 串联起来
* Transforms on PIL Image and `torch.*Tensor`
* Transforms on PIL Image only
    - `RandomChoice`
    - `RandomOrder`
* Transforms on `torch.*Tensor` only
    - `LinearTransformation`
    - `Normalize`
    - `RandomErasing`
    - `ConvertImageDtype`
* Conversion transforms
    - `ToPILImage`: tensor/ndarray -> PIL Image
    - `ToTensor`: PIL Image/numpy.ndarray -> tensor
        - 将 PIL 格式图像或 Numpy `ndarra` 转换为 `FloatTensor`
        - 将图像的像素强度值(pixel intensity values)缩放在 `[0, 1]` 范围内
    - `PILToTensor`: PIL Image -> tensor
* Generic transforms
    * `Lambda` 变换
        - 可以应用任何用户自定义的 lambda 函数
        - `scatter_`: 在标签给定的索引上设置 `value`
* Automatic Augmentation transforms
    - `AutoAugmentPolicy`
    - `AutoAgument`
    - `RandAugment`
    - `TrivialAugmentWide`
    - `AugMix`
* Functional transforms
    - 函数式转换提供了对转换管道的细粒度控制。与上述转换相反，
      函数式转换不包含用于其参数的随机数生成器。
      这意味着必须指定/生成所有参数，但函数转换将提供跨调用的可重现结果
    - `torchvision.transform.functional`

## torchtext transforms

### transforms 简介

### 常用转换

## torchaudio transforms

### transforms 简介

### 常用转换

# AIbumentations




# 参考

* [transforms 的二十二个方法](https://zhuanlan.zhihu.com/p/53367135)
* [AIbumentations](https://albumentations.ai/)
