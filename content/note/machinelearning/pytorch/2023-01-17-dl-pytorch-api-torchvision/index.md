---
title: torchvision
author: 王哲峰
date: '2023-01-17'
slug: dl-pytorch-torchvision
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

- [数据集](#数据集)
  - [内置数据集](#内置数据集)
  - [自定义图像数据集的基本类](#自定义图像数据集的基本类)
- [数据集读写](#数据集读写)
  - [视屏](#视屏)
  - [图像](#图像)
- [数据转换与增强](#数据转换与增强)
  - [数据转换与增强简介](#数据转换与增强简介)
  - [常用转换](#常用转换)
- [特征提取](#特征提取)
- [模型和预训练权重](#模型和预训练权重)
  - [模型类型](#模型类型)
- [工具](#工具)
- [操作](#操作)
- [任务](#任务)
  - [classification](#classification)
  - [detection](#detection)
  - [segmentation](#segmentation)
  - [similarity learning](#similarity-learning)
  - [video classification](#video-classification)
</p></details><p></p>

# 数据集

> * torchvision.datasets
> * torch.utils.data.DataLoader

## 内置数据集

示例：

```python
from torch.utils.data import DataLoader
from torchvision import datasets

imagenet_data = datasets.ImageNet("path/to/imagenet_root/")
data_loader = DataLoader(
    imagenet_data,
    batch_size = 4,
    shuffle = True,
    num_workers = args.nThreads
)
```

类型：

* Image classification
* Image detection 或 segmentation
* Optical Flow
* Stereo Matching
* Image pairs
* Image captioning
* Video classification

## 自定义图像数据集的基本类

* `DatasetFolder(root, loader[, extensions, ...])`
    - 通用数据加载器
* `ImageFolder(root, transform, ...)`
    - 图像以默认情况方式排列时的通用数据加载器
* `VisionDataset(root, [transorms, transform,...])`
    - 用于制作与 torchvision 兼容的数据集

# 数据集读写

> torchvision.io

## 视屏

* read_video
* read_video_timestamps
* write_video

```python
import torchvision
video_path = "path to a test video"
# Constructor allocates memory and a threaded decoder
# instance per video. At the moment it takes two arguments:
# path to the video file, and a wanted stream.
reader = torchvision.io.VideoReader(video_path, "video")

# The information about the video can be retrieved using the
# `get_metadata()` method. It returns a dictionary for every stream, with
# duration and other relevant metadata (often frame rate)
reader_md = reader.get_metadata()

# metadata is structured as a dict of dicts with following structure
# {"stream_type": {"attribute": [attribute per stream]}}
#
# following would print out the list of frame rates for every present video stream
print(reader_md["video"]["fps"])

# we explicitly select the stream we would like to operate on. In
# the constructor we select a default video stream, but
# in practice, we can set whichever stream we would like
video.set_current_stream("video:0")
```

## 图像

* ImageReadMode
* JPEG 和 PNG
    - decode_image
    - encode_image
    - read_image
* JPEG
    - encode_jpeg
    - decode_jpeg
    - write_jpeg
* PNG
    - encode_png
    - decode_png
    - write_png
* unit8 Tensor
    - read_file
    - write_file

# 数据转换与增强

> torchvision.transforms

## 数据转换与增强简介

所有的 torchvision datasets 都有两个接受包含转换逻辑的可调用对象的参数:

* transform
    - 修改特征
* target_transform
    - 修改标签

大部分 transform 同时接受 PIL 图像和 tensor 图像，但是也有一些 tansform 只接受 PIL 图像，或只接受 tensor 图像

* PIL
* tensor

transform 接受 tensor 图像或批量 tensor 图像

* tensor 图像的 shape 格式是 `(C, H, W)`
* 批量 tensor 图像的 shape 格式是 `(B, C, H, W)`

一个 tensor 图像像素值的范围由 tensor dtype 严格控制

* float: `$[0, 1)$`
* integer: `[0, MAX_DTYPE]`

转换的形式：

* Module transforms
* functional transforms

## 常用转换

* Scriptable transforms
    - `torch.nn.Sequential`
    - `torch.jit.script`
* Compositions of transforms
    - `Compose`: 将多个 transform 串联起来
* Transforms on PIL Image and `torch.*Tensor`
    - ToTensor()
        - 将 PIL 格式图像或 Numpy `ndarra` 转换为 `FloatTensor`
        - 将图像的像素强度值(pixel intensity values)缩放在 `[0, 1]` 范围内
    - Lambda 换换
        - 可以应用任何用户自定义的 lambda 函数
        - `scatter_`: 在标签给定的索引上设置 `value`
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
    - `PILToTensor`: PIL Image -> tensor
* Generic transforms
    - `Lambda`
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


# 特征提取




# 模型和预训练权重

> torchvision.models

## 模型类型

* image classification
* pixelwise semantic setmentation
* object detection
* instance segmentation
* person keypoint detection
* video classification
* optical flow


# 工具


# 操作

# 任务

## classification


## detection


## segmentation




## similarity learning

## video classification

