---
title: OpenCV BGR 像素强度图
author: 王哲峰
date: '2022-08-31'
slug: cv-opencv-bgr-pixel
categories:
  - computer vision
tags:
  - article
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

- [彩色图像、颜色通道](#彩色图像颜色通道)
- [图像属性、像素属性](#图像属性像素属性)
- [BGR 像素强度线(计数图)](#bgr-像素强度线计数图)
</p></details><p></p>

# 彩色图像、颜色通道

颜色通道:

* B: 蓝色
* G: 绿色
* R: 红色

# 图像属性、像素属性

```python
import numpy as np
import matplotlib.pyplot as plt

import cv2

img = cv2.imread("./data/img.jpg")
print(f"Image shape: {img.shape}")
print(f"Pixels num: {img.size}")
print(f"First pixel: {img[0][0]}")
```

# BGR 像素强度线(计数图)


```python
colors = ("blue", "green", "red")
label = ("Blue", "Green", "Red")
for count, color in enumerate(colors):
    histogram = cv2.calcHist(
        images = [img], 
        channels = [count], 
        mask = None, 
        histSize = [256], 
        ranges = [0, 256]
    )
    plt.plot(
        histogram, 
        color = color, 
        label = label[count] + str(" Pixels")
    )
    plt.title("Histogram Showing Number Of Pixels Belonging To Respective Pixel Intensity", color = "crimson")
    plt.ylabel("Number Of Pixels", color="crimson")
    plt.xlabel("Pixel Intensity", color="crimson")
    plt.legend(numpoints=1, loc="best")
    plt.show()
```

