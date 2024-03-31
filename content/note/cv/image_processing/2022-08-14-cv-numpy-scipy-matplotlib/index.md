---
title: Scipy Matplotlib Skimage 图像处理
author: 王哲峰
date: '2022-08-14'
slug: cv-numpy-scipy-matplotlib
categories:
  - deeplearning
  - computer vision
tags:
  - tool
---

* numpy
* scipy.ndimage
    - [Scipy.ndimage](https://docs.scipy.org/doc/scipy/tutorial/ndimage.html)
    - [Scipy-Numpy](http://scipy-lectures.org/advanced/image_processing/index.html)
* skimage
    - [skimage-doc](https://scikit-image.org/docs/stable/api/skimage.html#module-skimage)
    - [skimage-tutorial](http://scipy-lectures.org/packages/scikit-image/index.html#scikit-image) 
* matplotlib

# 任务

* Input/Output, displaying images
* Basic manipulations: cropping, flipping, rotating, …
* Image filtering: denoising, sharpening
* Image segmentation: labeling pixels corresponding to different objects
* Classification
* Feature extraction
* Registration
* ...

# 图像读写

```python
import matplotlib.pyplot as plt
from scipy import misc
import imageio  # Image module(PIL)

f = misc.face()
imageio.imsave("face.png", f)
plt.imshow()
plt.show()
```


# 图像去噪




# 特征提取


