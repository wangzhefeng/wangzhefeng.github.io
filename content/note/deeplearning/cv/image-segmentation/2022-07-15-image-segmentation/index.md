---
title: 图像分割
author: 王哲峰
date: '2022-07-15'
slug: image-segmentation
categories:
  - deeplearning
tags:
  - model
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

- [图像分割简介](#图像分割简介)
  - [图像分割应用](#图像分割应用)
  - [图像分割类型](#图像分割类型)
  - [图像分割方法](#图像分割方法)
- [参考](#参考)
</p></details><p></p>

# 图像分割简介

图像分割是将图像分割成片段（也称为对象）的过程。检测图像中存在的对象并为它们着色以将它们彼此分开。
它主要专注于检测物体的边界，因此可以很容易地将它们分开。很多时候，甚至标记检测到的每个片段/对象

## 图像分割应用

* 人脸检测
* 视频监控
* 自动驾驶汽车用它来检测物体（路标、其他汽车、行人等）
* 检测卫星图像中的物体（道路、农作物、建筑物等）
* 用于检测肿瘤的医学成像
* 基于内容的图像检索，它搜索图像的内容而不是元数据、名称等来检索数据

## 图像分割类型

![img](images/seg.png)

* 实例分割
    - 所有相同类型的对象都用不同的颜色/标签标记。每个对象都有自己的颜色/标签
    - 例如，图像中的个人会有不同的颜色/标签
* 语义分割
    - 所有相同类型的对象都用一种颜色/标签标记
    - 例如，图像中的所有人都将具有相同的颜色/标签

## 图像分割方法

多年来，已经开发了许多方法来解决图像分割任务。其中一些使用机器学习（深度学习），而另一些则使用非 ML 解决方案。
python 库 scikit-image 具有大多数非 ML 方法的实现。我们在下面列出了一些使用非 ML 方法解决图像分割任务的著名方法

* 基于阈值的方法
* 基于聚类的方法
* 基于直方图的方法
* 区域增长方法
* 边缘检测
* 流域改造
* 基于图形的方法

大多数 ML 方法涉及使用由卷积、密集等层组成的深度神经网络。下面，我们列出了一些解决图像分割任务的著名神经网络

* U-Net
* Fast-FCN(Fully Convolutional Network)
* Mask R-CNN
* DeepLab
* LRASPP
* Gates-SCNN




# 参考

* [Image Segmentation using Pre-Trained Models (torchvision)](https://coderzcolumn.com/tutorials/artificial-intelligence/pytorch-image-segmentation-using-pre-trained-models)
