---
title: CNN-目标检测
author: 王哲峰
date: '2022-07-15'
slug: dl-cnn-target-detection
categories:
  - deeplearning
tags:
  - model
---

## 从 R-CNN 到 yoloi

- 目标检测:让计算机不仅能够识别出输入图像中的目标物体, 还要能够给出目标物体所在图像中的位置; 
- 在深度学习正式成为计算机视觉领域的主题之前, 传统的手工特征图像算法一直是目标检测的主要方法。
  在早期计算资源不发达的背景下, 研究人员的图像特征表达方法有限, 
  只能尽可能的设计更加多元化的检测算法来进行弥补, 
  包括早期的SIFT 检测算法、HOG 检测算法和后来著名的 DPM 模型等; 
- 深度学习之前的早期目标检测算法的发展历程如上图左边浅蓝色部分所示:
- 2013 年之后, 神经网络和深度学习逐渐取代了传统的图像检测算法而成为目标检测的主流方法。
  纵观这几年的深度学习目标检测发展历程, 基于深度学习算法的一系列目标检测算法大致可以分为两大流派:
- 两步走(two-stage)算法:先产生候选区域然后再进行CNN分类(RCNN系列); 
- 一步走(one-stage)算法:直接对输入图像应用算法并输出类别和相应的定为(yolo系列); 

## 两部走(two-stage)算法系列

### R-CNN

R-CNN 作为将深度学习引入目标检测算法的开山之作, 在目标检测算法发展历史上具有重大意义。R-CNN 算法是两步走方法的代表, 
即先生成候选区域 (region proposal) , 然后再利用 CNN 进行识别分类。由于候选框对于算法的成败起着关键作用, 
所以该方法就以 Region 开头首字母 R 加 CNN 进行命名。

相较于传统的滑动卷积窗口来判断目标的可能区域, R-CNN 采用 selective search 的方法来预先提取一些较可能是目标物体的候选区域, 
速度大大提升, 计算成本也显著缩小。总体而言, R-CNN 方法分为四个步骤:

- 生成候选区域
- 对候选区域使用CNN进行特征提取
- 将提取的特征送入SVM分类器
- 最后使用回归器对目标位置进行修正

虽然 R-CNN 在 2013年的当时可谓横空出世, 但也存在许多缺陷:selective search 方法生成训练网络的正负样本候选区域在速度上非常慢, 
影响了算法的整体速度; CNN 需要分别对每一个生成的候选区域进行一次特征提取, 存在着大量的重复运算, 制约了算法性能


- 论文:Rich feature hierarchies for accurate object detection and semantic segmentation
- R-CNN TensorFlow实现参考: https://github.com/yangxue0827/RCNN

### SPP-Net

针对 R-CNN 的问题, 提出 ResNet 的何恺明大佬提出了
SPP-Net。该算法通过在网络的卷积层和全连接层之间加入空间进字体池化层 (Spatial
Pyramid Pooling) 来对利用 CNN
进行卷积特征提取之前的候选区域进行裁剪和缩放使 CNN 的输入图像尺寸一致。

空间金字塔池化解决了输入候选区域尺寸不一致的问题, 但更重要的意义在于减少了
R-CNN 中的重复计算, 大大提高的算法的速度和性能。

SPP-Net 的缺点在于经过空间金字塔层的处理后, 虽然 CNN
的输入尺寸一致了, 但候选框的感受野因而也变得很大, 使得卷积神经网络在训练时无法有效更新模型权重。

- 论文:Spatial Pyramid Pooling in Deep Convolutional Networks for Visual Recognition
- SPP-Net TensorFlow 实现参考: https://github.com/chengjunwen/spp_net

### Fast-CNN

针对 SPP-Net 的问题, 2015年微软研究院在借鉴了 SPP-Net
的空间金字塔层的基础之上, 对 R-CNN 算法进行了有效的改进。

Fast R-CNN 的结构如上图所示。Fast R-CNN 的改进之处在于设计了一种 ROI
Pooling 的池化层结构, 有效解决了 R-CNN
算法必须将图像区域剪裁、缩放到相同尺寸大小的操作。提出了多任务损失函数, 每一个
ROI 都有两个输出向量:softmax 概率输出向量和每一类的边界框回归位置向量。

Fast R-CNN 虽然借鉴了 SPP-Net 的思想, 但对于 R-CNN 的 selective search
的候选框生成方法依然没做改进, 这使得 Fast R-CNN 依然有较大的提升空间。

- 论文: Fast R-CNN
- Fast R-CNN caffe 源码参考: https://github.com/rbgirshick/fast-rcnn


### Faster-CNN

### Mask R-CNN

### SPP-Net



## 一步走(one-stage)算法系列

纵然两步走的目标检测算法在不断进化, 检测准确率也越来越高, 但两步走始终存在的速度的瓶颈。在一些实时的目标检测需求的场景中, R-CNN
系列算法终归是有所欠缺。因而一步走 (one-stage) 算法便应运而生了, 其中以 yolo 算法系列为代表, 演绎了一种端到端的深度学习系统的实时目标检测效果。
yolo 算法系列的主要思想就是直接从输入图像得到目标物体的类别和具体位置, 不再像 R-CNN 系列那样产生候选区域。这样做的直接效果便是快。

### yolo v1

- 论文:You Only Look Once: Unified, Real-Time Object Detection
- yolo v1 pytorch参考:https://github.com/xiongzihua/pytorch-YOLO-v1

### SSD

- 论文:SSD: Single Shot MultiBox Detector
- SSD TensorFlow 源码参考:https://github.com/balancap/SSD-TensorFlow

### yolo v2/yolo9000

- 论文:YOLO9000: Better, Faster, Stronger
- yolo 9000 源码参考:https://github.com/philipperemy/yolo-9000

### yolo v3

- 论文:YOLOv3: An Incremental Improvement
- yolo v3 源码参考:https://github.com/ayooshkathuria/pytorch-yolo-v3
