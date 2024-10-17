---
title: LeNet-5
author: wangzf
date: '2023-01-27'
slug: cnn-lenet-5
categories:
  - deeplearning
tags:
  - paper
---

![img](images/lenet-5-slide.png)

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

- [LeNet-5 简介](#lenet-5-简介)
- [LeNet-5 网络结构](#lenet-5-网络结构)
- [参考](#参考)
</p></details><p></p>

# LeNet-5 简介

在神经网络的和深度学习领域，Yann LeCun 在 1998 年在 IEEE 上发表了 (Gradient-based learning applied to document recognition，1998)，
文中首次提出了 **卷积-池化-全连接** 的神经网络结构，
由 LeCun 提出的七层网络命名为 LeNet-5，因而也为他赢得了 CNN 之父的美誉。

# LeNet-5 网络结构

作为标准的卷积网络结构，LeNet-5 对后世的影响深远，以至于在 16 年后，谷歌提出 Inception 网络时也将其命名为 GoogLeNet，
以致敬 Yann LeCun 对卷积神经网络发展的贡献。然而 LeNet-5 提出后的十几年里，由于神经网络的可解释性问题和计算资源的限制，
神经网络的发展一直处于低谷。

LeNet-5 共有 5 层(输入输出层不计入层数，池化层与卷积层算 1 层). 
每层都有一定的训练参数，其中三个卷积层的训练参数较多，每层都有多个滤波器(特征图)，
每个滤波器都对上一层的输出提取不同的像素特征：

> * Layer input：输入
> * Layer 1：卷积-池化
> * Layer 2：卷积-池化
> * Layer 3：卷积(全连接)
> * Layer 4：全连接
> * Layer 5：全连接
> * Layer output：输出

原始论文中的网络结构：

![img](images/lenet-5.png)

下面是对网络结构的详解：

> As the name indicates，LeNet5 has 5 layers with two convolutional and three fully connected layers. 
> Let's start with the input. LeNet5 accepts as input a greyscale image of 32x32，
> indicating that the architecture is not suitable for RGB images (multiple channels). 
> So the input image should contain just one channel. After this，we start with our convolutional layers.
> 
> The first convolutional layer has a filter size of  5x5 with 6 such filters. 
> This will reduce the width and height of the image while increasing the depth (number of channels). 
> The output would be 28x28x6. After this，pooling is applied to decrease the feature map by half，
> i.e，14x14x6. Same filter size (5x5) with 16 filters is now applied to the output followed by a pooling layer. 
> This reduces the output feature map to 5x5x16.
> 
> After this，a convolutional layer of size 5x5 with 120 filters is applied to flatten the feature map to 120 values. 
> Then comes the first fully connected layer，with 84 neurons. 
> Finally，we have the output layer which has 10 output neurons，
> since the MNIST data have 10 classes for each of the represented 10 numerical digits.

# 参考

* [Gradient-Based Learning Applied to Document Recognition](http://yann.lecun.com/exdb/publis/pdf/lecun-01a.pdf)
* [卷积神经网络（CNN）基础及经典模型介绍](https://zhuanlan.zhihu.com/p/344562609)
