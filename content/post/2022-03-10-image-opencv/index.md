---
title: OpenCV 图像处理
author: 王哲峰
date: '2022-03-10'
slug: image-opencv
categories: [图像处理]
tags:
  - tool
---

<style>
h1 {
  background-color: #2B90B6;
  background-image: linear-gradient(45deg, #4EC5D4 10%, #146b8c 20%);
  background-size: 100%;
  -webkit-background-clip: text;
  -moz-background-clip: text;
  -webkit-text-fill-color: transparent;
  -moz-text-fill-color: transparent;
}
h2 {
  background-color: #2B90B6;
  background-image: linear-gradient(45deg, #4EC5D4 10%, #146b8c 20%);
  background-size: 100%;
  -webkit-background-clip: text;
  -moz-background-clip: text;
  -webkit-text-fill-color: transparent;
  -moz-text-fill-color: transparent;
}
</style>

# OpenCV 简介

> OpenCV (Open Source Computer Vision Library)，一个开源的计算机视觉库，官方网站为 [http://opencv.org](http://opencv.org/)。它提供了很多函数，
> 这些函数非常高效地实现了计算机视觉算法，从最基本的滤波到高级的物体检测皆有涵盖。
>
> 1999 年，Gary Bradski（加里·布拉德斯基）当时在英特尔任职，怀着通过为计算机视觉和人工智能的从业者提供稳定的基础架构并以此来推动产业发展的美好愿景，他启动了 OpenCV 项目

- OpenCV 的一个目标是提供易于使用的计算机视觉接口，从而帮助人们快速建立精巧的视觉应用
- OpenCV 设计用于进行高效的计算，十分强调实时应用的开发。它由 C++ 语言编写并进行了深度优化，从而可以享受多线程处理的优势。同时也提供了 Python、Java、MATLAB 等语言的接口
- OpenCV 是跨平台的，可以在 Windows、Linux、macOS、Android、iOS 等系统上运行
- OpenCV 库包含从计算机视觉各个领域衍生出来的 500 多个函数，包括工业产品质量检验、医学图像处理、安保领域、交互操作、相机校正、双目视觉以及机器人学。OpenCV 的应用非常广泛，包括图像拼接、图像降噪、产品质检、人机交互、人脸识别、动作识别、动作跟踪、无人驾驶等
- 因为计算机视觉和机器学习经常在一起使用，所以 OpenCV 也包含一个完备的、具有通用性的机器学习库（ML模块）。这个子库聚焦于统计模式识别以及聚类。ML 模块对 OpenCV 的核心任务（计算机视觉）相当有用，但是这个库也足够通用，可以用于任意机器学习问题

- IPPICV 加速

	- 如果希望得到更多在英特尔架构上的自动优化，可以购买英特尔的集成性能基元（IPP）库，该库包含了许多算法领域的底层优化程序。在库安装完毕的情况下 OpenCV 在运行的时候会自动调用合适的 IPP 库。

	- 从 OpenCV 3.0 开始，英特尔许可 OpenCV 研发团队和 OpenCV 社区拥有一个免费的 IPP 库的子库（称 IPPICV），该子库默认集成在 OpenCV 中并在运算时发挥效用。如果你使用的是英特尔的处理器，那么 OpenCV 会自动调用 IPPICV。

	- IPPICV 可以在编译阶段链接到 OpenCV，这样一来，会替代相应的低级优化的C语言代码（在 cmake 中设置`WITH_IPP=ON/OFF`来开启或者关闭这一功能，默认情况为开启）。使用 IPP 获得的速度提升非常可观。

# OpenCV-Python

> OpenCV-Python 依赖于 Numpy 库，所有的 OpenCV 数组结构都能够与 Numpy 数组相互转换，这也使得使用 OpenCV 与 Numpy 的其他库的集成变得更加容易

## 依赖

- OpenCV 3.x
- Python 3
- Numpy

## OpenCV-Python 安装

- macOS
- Windows
- Linux

## OpenCV-Python 使用

```python
import cv2 as cv
print(cv.__version__)
```

## 图像读取、图像显示、图像保存

### API

- cv.imread(img_path, img_mode)
	- cv.IMREAD_COLOR：加载彩色图像。任何图像的透明度都会被忽视，默认。参数编码：1
	- cv.IMREAD_GRAYSCALE：以灰度模式加载图像。参数编码：0
	- cv.IMREAD_UNCHANGED：加载图像，包括 alpha 通道。编码：-1
- cv.imshow('window_name', img_obj)
- cv.imwrite

### 图像读取

使用 `cv.imread`读取图像时，图像应该在工作目录或者给出图像的完整文件路径

```python
import numpy as np
import cv2 as cv

img = cv.imread("img.jpg", 0)
```



### 图像显示

使用 `cv.imshow` 显示图像时，显示图像的窗口自动适合图像尺寸

```python
cv.imshow("image", img)
cv.waitKey(0)
cv.destoryAllWindows()
```







### 图像保存



