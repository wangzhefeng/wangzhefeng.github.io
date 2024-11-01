---
title: YOLOv1
# subtitle: 
author: wangzf
date: '2022-07-13'
slug: yolov1
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
img {
    pointer-events: none;
}
</style>

<details><summary>目录</summary><p>

- [YOLOv1 网络结构](#yolov1-网络结构)
- [YOLOv1 检测原理](#yolov1-检测原理)
- [参考](#参考)
</p></details><p></p>

# YOLOv1 网络结构

YOLOv1 的思想十分简洁：仅使用一个卷积神经网络来端到端地检测目标。
这一思想是对应当时主流的以 R-CNN 系列为代表的 two-stage 流派。
从网络结构上来看，YOLOv1 仿照 GoogLeNet 网络来设计主干网络，
但没有采用 GoogLeNet 的 Inception 模块，
而是使用串联的 `$1\times 1$` 卷积和 `$3\times 3$` 卷积所组成的模块，
所以它的主干网络的结构非常简单。

![img](images/)


# YOLOv1 检测原理




# 参考

* https://pjreddie.com/darknet/yolov2/
