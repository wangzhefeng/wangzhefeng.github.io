---
title: VGG-Net
subtitle: VGG-16、VGG-19
author: 王哲峰
date: '2023-03-23'
slug: cnn-vgg16
categories:
  - deeplearning
tags:
  - model
---

![img](images/vgg16.png)

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

- [VGG-Net 简介](#vgg-net-简介)
- [VGG 16 网络结构](#vgg-16-网络结构)
- [参考](#参考)
</p></details><p></p>

# VGG-Net 简介

> VGG16，VGG19

由于不断的积累实践和日益强大的计算能力，使得研究人员敢于将神经网络的结构推向更深层。
在 2014 年提出的 VGG-Net 中，首次将卷积网络结构拓展至 16 和 19 层，也就是著名的 VGG16 和 VGG19。

相较于此前的 LeNet-5 和 AlexNet 的 `$5 \times 5$` 卷积和 `$11 \times 11$` 卷积，VGGNet
结构中大量使用 `$3 \times 3$` 的卷积核和 `$2 \times 2$` 的池化核

VGGNet 的网络虽然开始加深但其结构并不复杂，但作者的实践却证明了卷积网络深度的重要性。
深度卷积网络能够提取图像低层次、中层次和高层次的特征，因而网络结构需要的一定的深度来提取图像不同层次的特征

在论文中，作者使用了 A-E 五个不同深度水平的卷积网络进行试验，从 A 到 E 网络深度不断加深，网络的具体信息如下:

![img](images/VGG-Net2.png)

# VGG 16 网络结构

VGG 的网络结构非常规整，2-2-3-3-3 的卷积结构也非常利于编程实现。卷积层的滤波器数量的变化也存在明显的规律，
由 64 到 128 再到 256 和 512，每一次卷积都是像素成规律的减少和通道数成规律的增加。
VGG-16 在当年的 ILSVRC 以 32% 的 top5 错误率取得了当年大赛的第二名。这么厉害的网络为什么是第二名？
因为当年有比 VGG 更厉害的网络，也就是致敬 LeNet-5 的 GoogLeNet

![img](images/vgg16.webp)

![img](images/vgg162.png)

# 参考

* [Very Deep Convolutional Networks for Large-Scale Image Recognition](https://arxiv.org/pdf/1409.1556.pdf?ref=blog.paperspace.com)
