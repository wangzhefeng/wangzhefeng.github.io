---
title: 图像处理概览
author: 王哲峰
date: '2023-04-29'
slug: cv-processing
categories:
  - computer vision
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

- [图像处理概述](#图像处理概述)
- [图像处理基本操作](#图像处理基本操作)
</p></details><p></p>

# 图像处理概述

推荐使用 OpenCV 而不是 Pillow 和 Matplotlib，因为 OpenCV 具有更多的功能。
但是，只在显示图像或进行批量图像处理时使用 Pillow。
当在 Jupyter Notebook 中显示图像时，建议使用 Matplotlib

# 图像处理基本操作

* 图像读取
* 图像显示
* 图像保存
* Pad 边缘填充
* Resize
* Crop 裁剪
    - center crop
    - five crop(top_left, top_right, bottom_left, bottom_right, center)
    - random crop
    - random resized crop
* gray scale
* color jitter 改变图片亮度、明暗、色调
    - brightness
    - saturation
    - ...
* Blur 模糊化
    - Gaussian
* Perspective 透视 
* Rotation 旋转
* Affine 仿射
* Invert 反转
* Posterize 调色（改变每个通道的颜色值）
* Solarize 曝光
* Adjust Sharpness 调节锐度
* Adjust Contrast 对比度
* Equalize 均衡
* Augment 增强
* Trivial Augment Wide
* Flip 翻转
    - Horizontal Flip
    - Vertical Flip

