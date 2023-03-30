---
title: 图像目标检测
author: 王哲峰
date: '2022-07-15'
slug: image-detection
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

- [目标检测](#目标检测)
  - [目标检测概述](#目标检测概述)
  - [目标检测模型](#目标检测模型)
- [目标检测数据集](#目标检测数据集)
- [目标检测性能指标](#目标检测性能指标)
  - [IoU](#iou)
  - [AP 和 mAP](#ap-和-map)
  - [FPS](#fps)
- [两部走(two-stage)算法系列](#两部走two-stage算法系列)
  - [R-CNN](#r-cnn)
  - [SPP-Net](#spp-net)
  - [Fast R-CNN](#fast-r-cnn)
  - [Faster R-CNN](#faster-r-cnn)
  - [Mask R-CNN](#mask-r-cnn)
  - [SPP-Net](#spp-net-1)
- [一步走(one-stage)算法系列](#一步走one-stage算法系列)
  - [yolo v1](#yolo-v1)
  - [SSD](#ssd)
  - [yolo v2/yolo9000](#yolo-v2yolo9000)
  - [yolo v3](#yolo-v3)
- [参考](#参考)
</p></details><p></p>

# 目标检测

> 目标检测: 让计算机不仅能够识别出输入图像中的目标物体，还要能够给出目标物体所在图像中的位置

## 目标检测概述

图像分类，检测及分割是计算机视觉领域的三大任务。图像分类模型是将图像划分为单个类别，
通常对应于图像中最突出的物体。但是现实世界的很多图片通常包含不只一个物体，
此时如果使用图像分类模型为图像分配一个单一标签其实是非常粗糙的，并不准确。
对于这样的情况，就需要目标检测模型，目标检测模型可以识别一张图片的多个物体，
并可以定位出不同物体(给出边界框)。目标检测在很多场景有用，如无人驾驶和安防系统

在深度学习正式成为计算机视觉领域的主题之前，
传统的手工特征图像算法一直是目标检测的主要方法。
在早期计算资源不发达的背景下，研究人员的图像特征表达方法有限，
只能尽可能的设计更加多元化的检测算法来进行弥补，
包括早期的 SIFT 检测算法、HOG 检测算法和后来著名的 DPM 模型等; 
深度学习之前的早期目标检测算法的发展历程如上图左边浅蓝色部分所示:

![img](images/todo.png)

## 目标检测模型

2013 年之后，神经网络和深度学习逐渐取代了传统的图像检测算法而成为目标检测的主流方法。
纵观这几年的深度学习目标检测发展历程，基于深度学习算法的一系列目标检测算法大致可以分为两大流派:

* (1) two-stage 检测算法，其将检测问题划分为两个阶段，
  首先产生候选区域(region proposals)，然后对候选区域分类(一般还需要对位置精修)，
  这类算法的典型代表是基于 region proposal 的 R-CNN 系算法，
  如 R-CNN、Fast R-CNN、Faster R-CNN 等
* (2) one-stage 检测算法，其不需要 region proposal 阶段，
  直接产生物体的类别概率和位置坐标值，比较典型的算法如 YOLO 和 SSD

简单来说就是：

* 两步走(two-stage)算法: 先产生候选区域然后再进行 CNN 分类(R-CNN 系列)
* 一步走(one-stage)算法: 直接对输入图像应用算法并输出类别和相应的定为(yolo 系列)

目标检测模型的主要性能指标是检测准确度和速度，
对于准确度，目标检测要考虑物体的定位准确性，而不单单是分类准确度。
一般情况下，two-stage 算法在准确度上有优势，而 one-stage 算法在速度上有优势。
不过，随着研究的发展，两类算法都在两个方面做改进

Google 在 2017 年开源了 [TensorFlow Object Detection API](https://link.zhihu.com/?target=https%3A//github.com/tensorflow/models/tree/master/research/object_detection)，
并对主流的 Faster R-CNN、R-FCN 及 SSD 三个算法在 MS COCO 数据集上的性能做了细致对比(见 Huang et al. 2017)，如下图所示

![img](images/gpu_time.jpeg)
<center>Faster R-CNN，R-FCN及SSD算法在MS COCO数据集上的性能对比</center>

近期，Facebook 的 FAIR 也开源了基于 Caffe2 的目标检测平台 [Detectron](https://link.zhihu.com/?target=https%3A//github.com/facebookresearch/Detectron)，
其实现了最新的 Mask R-CNN，RetinaNet 等检测算法，
并且给出了这些算法的 [Baseline Results](https://link.zhihu.com/?target=https%3A//github.com/facebookresearch/Detectron/blob/master/MODEL_ZOO.md) 。
不得不说，准确度(accuracy)和速度(speed)是一对矛盾体，
如何更好地平衡它们一直是目标检测算法研究的一个重要方向

# 目标检测数据集

目标检测常用的数据集包括 PASCAL VOC、ImageNet、MS COCO 等数据集，
这些数据集用于研究者测试算法性能或者用于竞赛

# 目标检测性能指标

目标检测的性能指标要考虑检测物体的位置以及预测类别的准确性

## IoU

目标检测问题同时是一个回归和分类问题。首先，为了评估定位精度，
需要计算 IoU(Intersection over Union，介于 0 到 1 之间)，
其表示预测框与真实框(ground-truth box)之间的重叠程度。
IoU 越高，预测框的位置越准确。因而，在评估预测框时，
通常会设置一个 IoU 阈值(如 0.5)，只有当预测框与真实框的 IoU 值大于这个阈值时，
该预测框才被认定为真阳性(True Positive，TP)，反之就是假阳性(False Positive，FP)

## AP 和 mAP

对于二分类，AP(Average Precision)是一个重要的指标，这是信息检索中的一个概念，
基于 precision-recall 曲线计算出来，详情见[这里](https://link.zhihu.com/?target=https%3A//en.wikipedia.org/w/index.php%3Ftitle%3DInformation_retrieval%26oldid%3D793358396%23Average_precision)。
对于目标检测，首先要单独计算各个类别的 AP 值，这是评估检测效果的重要指标。
取各个类别的 AP 的平均值，就得到一个综合指标 mAP(Mean Average Precision)，
mAP 指标可以避免某些类别比较极端化而弱化其它类别的性能这个问题

对于目标检测，mAP 一般在某个固定的 IoU 上计算，但是不同的 IoU 值会改变 TP 和 FP 的比例，
从而造成 mAP 的差异。COCO 数据集提供了[官方的评估指标](https://link.zhihu.com/?target=https%3A//github.com/cocodataset/cocoapi)，
它的 AP 是计算一系列 IoU 下(0.5:0.05:0.9，见[说明](https://link.zhihu.com/?target=http%3A//cocodataset.org/%23detection-eval))AP 的平均值，
这样可以消除 IoU 导致的 AP 波动。其实对于 PASCAL VOC 数据集也是这样，
Facebook 的 Detectron 上的有比较清晰的[实现](https://link.zhihu.com/?target=https%3A//github.com/facebookresearch/Detectron/blob/05d04d3a024f0991339de45872d02f2f50669b3d/lib/datasets/voc_eval.py%23L54)

## FPS

除了检测准确度，目标检测算法的另外一个重要性能指标是速度，只有速度快，才能实现实时检测，这对一些应用场景极其重要。
评估速度的常用指标是每秒帧率(Frame Per Second，FPS)，即每秒内可以处理的图片数量。
当然要对比 FPS，你需要在同一硬件上进行。另外也可以使用处理一张图片所需时间来评估检测速度，时间越短，速度越快

# 两部走(two-stage)算法系列

## R-CNN

R-CNN 作为将深度学习引入目标检测算法的开山之作，在目标检测算法发展历史上具有重大意义。
R-CNN 算法是两步走方法的代表，即先生成候选区域(region proposal)，
然后再利用 CNN 进行识别分类。由于候选框对于算法的成败起着关键作用，
所以该方法就以 Region 开头首字母 R 加 CNN 进行命名

相较于传统的滑动卷积窗口来判断目标的可能区域，
R-CNN 采用 selective search 的方法来预先提取一些较可能是目标物体的候选区域，
速度大大提升，计算成本也显著缩小。总体而言，R-CNN 方法分为四个步骤:

* 生成候选区域
* 对候选区域使用CNN进行特征提取
* 将提取的特征送入SVM分类器
* 最后使用回归器对目标位置进行修正

虽然 R-CNN 在 2013年的当时可谓横空出世，但也存在许多缺陷: 
selective search 方法生成训练网络的正负样本候选区域在速度上非常慢，
影响了算法的整体速度; CNN 需要分别对每一个生成的候选区域进行一次特征提取，
存在着大量的重复运算，制约了算法性能

* 论文: Rich feature hierarchies for accurate object detection and semantic segmentation
* [R-CNN TensorFlow 实现参考](https://github.com/yangxue0827/RCNN)

## SPP-Net

针对 R-CNN 的问题，提出 ResNet 的何恺明大佬提出了
SPP-Net。该算法通过在网络的卷积层和全连接层之间加入空间进字体池化层 (Spatial
Pyramid Pooling) 来对利用 CNN
进行卷积特征提取之前的候选区域进行裁剪和缩放使 CNN 的输入图像尺寸一致

空间金字塔池化解决了输入候选区域尺寸不一致的问题，但更重要的意义在于减少了
R-CNN 中的重复计算，大大提高的算法的速度和性能

SPP-Net 的缺点在于经过空间金字塔层的处理后，虽然 CNN
的输入尺寸一致了，但候选框的感受野因而也变得很大，
使得卷积神经网络在训练时无法有效更新模型权重

* 论文: Spatial Pyramid Pooling in Deep Convolutional Networks for Visual Recognition
* [SPP-Net TensorFlow 实现参考](https://github.com/chengjunwen/spp_net)

## Fast R-CNN

针对 SPP-Net 的问题，2015年微软研究院在借鉴了 SPP-Net
的空间金字塔层的基础之上，对 R-CNN 算法进行了有效的改进

Fast R-CNN 的结构如上图所示。Fast R-CNN 的改进之处在于设计了一种 ROI
Pooling 的池化层结构，有效解决了 R-CNN
算法必须将图像区域剪裁、缩放到相同尺寸大小的操作。提出了多任务损失函数，每一个
ROI 都有两个输出向量:softmax 概率输出向量和每一类的边界框回归位置向量

Fast R-CNN 虽然借鉴了 SPP-Net 的思想，但对于 R-CNN 的 selective search
的候选框生成方法依然没做改进，这使得 Fast R-CNN 依然有较大的提升空间

* 论文: Fast R-CNN
* [Fast R-CNN caffe 源码参考](https://github.com/rbgirshick/fast-rcnn)

## Faster R-CNN

## Mask R-CNN

## SPP-Net

# 一步走(one-stage)算法系列

纵然两步走的目标检测算法在不断进化，检测准确率也越来越高，但两步走始终存在的速度的瓶颈。
在一些实时的目标检测需求的场景中，R-CNN 系列算法终归是有所欠缺。
因而一步走 (one-stage) 算法便应运而生了，其中以 yolo 算法系列为代表，
演绎了一种端到端的深度学习系统的实时目标检测效果。
yolo 算法系列的主要思想就是直接从输入图像得到目标物体的类别和具体位置，
不再像 R-CNN 系列那样产生候选区域。这样做的直接效果便是快

## yolo v1

* 论文: You Only Look Once: Unified, Real-Time Object Detection
* [yolo v1 pytorch 参考](https://github.com/xiongzihua/pytorch-YOLO-v1)

## SSD

* 论文: SSD: Single Shot MultiBox Detector
* [SSD TensorFlow 源码参考](https://github.com/balancap/SSD-TensorFlow)

## yolo v2/yolo9000

* 论文: YOLO9000: Better, Faster, Stronger
* [yolo 9000 源码参考](https://github.com/philipperemy/yolo-9000)

## yolo v3

* 论文: YOLOv3: An Incremental Improvement
* [yolo v3 源码参考](https://github.com/ayooshkathuria/pytorch-yolo-v3)

# 参考

* []()