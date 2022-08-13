---
title: Pillow 图像处理
author: 王哲峰
date: '2022-03-10'
slug: cv-pillow
categories:
  - deeplearning
  - computer vision
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

- [图像处理基本操作](#图像处理基本操作)
- [Pillow 概览](#pillow-概览)
- [Pillow 安装](#pillow-安装)
- [Pillow 核心](#pillow-核心)
  - [`Image` class](#image-class)
  - [图像读、写、转换](#图像读写转换)
  - [图像剪切、粘贴、拼接](#图像剪切粘贴拼接)
  - [图像几何变换](#图像几何变换)
  - [图像颜色转换](#图像颜色转换)
  - [图像增强](#图像增强)
  - [图像序列](#图像序列)
  - [PostScript 打印](#postscript-打印)
  - [图像读取](#图像读取)
  - [控制图像编码](#控制图像编码)
- [Pillow API](#pillow-api)
</p></details><p></p>

# 图像处理基本操作

- 图像读取
- 图像显示
- 图像保存
- pad 边缘填充
- resize 
- Crop 裁剪
    - center crop
    - five crop(top_left, top_right, bottom_left, bottom_right, center)
    - random crop
    - random resized crop
- gray scale
- color jitter 改变图片亮度、明暗、色调
    - brightness
    - saturation
    - ...
- Blur 模糊化
    - Gaussian
- Perspective 透视 
- Rotation 旋转
- Affine 仿射
- Invert 反转
- Posterize 调色（改变每个通道的颜色值）
- Solarize 曝光
- Adjust Sharpness 调节锐度
- Adjust Contrast 对比度
- Equalize 均衡
- Augment 增强
- Trivial Augment Wide
- Flip 翻转
    - Horizontal Flip
    - Vertical Flip

# Pillow 概览

PIL: Python Imaging Library, Python 图像处理库，提供了如下的功能:
    
- Image Archives, 图像存档和批处理
    - 图像读取(read image)
    - 图像打印(print image)
    - 图像缩略图(create thumbnails)
    - 图像格式变换(convert file format)
- Image Display, 图像显示
    - TK PhotoImage
    - TK BitmapImage
    - Windows DIB interface[PythonWin, Windows-based tookits]
    - show[debugging]
- Image Processing, 图像处理
    - 图像点操作(point operations)
    - 图像卷积核滤波(filtering with a set of build in convolution kernel)
    - 图像颜色空间转换(image colour space conversions)
    - 图像缩放(image resizing)
    - 图像旋转(image rotation)
    - 图像仿射变换(image arbitray affine transforms)
    - 图像统计信息分析直方图(histogram method)

# Pillow 安装

- macOS 安装:

```bash
$ python3 -m pip install --upgrade pip
$ python3 -m pip install --upgrade Pillow
```

- Windows 安装:

```bash
$ python3 -m pip install --upgrade pip
$ python3 -m pip install --upgrade Pillow
```

- Linux 安装:

```bash
$ python3 -m pip install --upgrade pip
$ python3 -m pip install --upgrade Pillow
```

# Pillow 核心

## `Image` class

API:

- `from PIL import Image`
- `Image.open`
- `Image.format`
- `Image.size`
- `Image.mode`
- `Image.show`

示例:

```python
import os
from PIL import Image

image_path = "/Users/zfwang/project/machinelearning/deeplearning/deeplearning/src/pillow_src/images"
image_name = "lena"
try:
    with Image.open(os.path.join(image_path, image_name + ".png")) as im:
        # 图像格式
        print(im.format)

        # 图像尺寸
        print(im.size)

        # 图像模式
        print(im.mode)

        # 图像打印
        im.show()
except IOError as e:
    print(f"Can't open {image_name}")
```

![images](images/lena.png)

.. note:: 

- `.format`: 图像格式
- `.size`: 图像尺寸(width_pixels, height_pixels)
- `.mode`: 图像中条带的数量和名称、像素类型、像素深度

## 图像读、写、转换

API:

- `Image.open`
- `Image.save`
- `Image.thumbnail(size = (width, height))`

示例:

- 将图像转换为 JPEG 格式:

```python
import os, sys
from PIL import Image

for infile in sys.argv[1:]:
    f, e = os.path.splitext(inflie)
    outfile = f + ".jpg"
    if infile != outfile:
        try:
            with Image.open(infile) as im:
                im.save(outfile)
        except OSError:
            print("cannot convert", infile)
```

```python
import os
from PIL import Image

image_path = "/Users/zfwang/project/machinelearning/deeplearning/deeplearning/src/pillow_src/images"
image_name = "lena"
try:
    with Image.open(os.path.join(image_path, image_name + ".png")) as im:
        im.save(os.path.join(image_path, image_name + ".jpg"))
except IOError as e:
    print(f"Can't open {image_name}")
```

![images](images/lena.jpg)
    
- 创建 JPEG 缩略图(thumbnails)

```python
import os, sys
from PIL import Image

size = (128, 128)

for inflie in sys.argv[1:]:
    outfile = os.path.splitext(infile)[0] + ".thumbnail"
    if infile != outfile:
        try:
            with Image.open(infile) as im:
                im.thumbnail(size)
                im.save(outfile, "JPEG")
        except OSError:
            print(f"Can't create thumbnail for{infile}")
```

```python
import os
from PIL import Image

image_path = "/Users/zfwang/project/machinelearning/deeplearning/deeplearning/src/pillow_src/images"
image_name = "lena"
size = (128, 128)

try:
    with Image.open(os.path.join(image_path, image_name + ".png")) as im:
        im.thumbnail(size)
        im.save(os.path.join(image_path, image_name + ".JPEG"))
except IOError as e:
    print(f"Can't open {image_name}")
```

![images](images/lena.JPEG)

- 识别图像文件

```python

import sys
from PIL import Image

for infile in sys.argv[1:]:
    try:
        with Image.open(infile) as im:
            print(infile, im.format, f"{im.size}x{im.mode}")
    except OSError:
        pass
```

```python
import os
from PIL import Image

image_path = "/Users/zfwang/project/machinelearning/deeplearning/deeplearning/src/pillow_src/images"
image_name = "lena"

try:
    with Image.open(os.path.join(image_path, image_name + ".png")) as im:
        print(image_name, im.format, f"{im.size}x{im.mode}")
except IOError as e:
    print(f"Can't open {image_name}")
```

## 图像剪切、粘贴、拼接

API:

- `Image.crop`: 从图像中复制子矩形
- `Image.past`
- `Image.merge`
- `Image.split`
- `Image.transpose(Image.ROTATE_180)`

示例:

    - 从图像中复制子矩形

```python

    import os
    from PIL import Image

    image_path = "/Users/zfwang/project/machinelearning/deeplearning/deeplearning/src/pillow_src/images"
    image_name = "lena"

    try:
        box = (100, 100, 200, 200)
        with Image.open(os.path.join(image_path, image_name + ".png")) as im:
            region = im.crop(box)
            region.show()
            region.save(os.path.join(image_path, image_name + "_region" + ".png"))
    except IOError as e:
        print(f"Can't open {image_name}")
```

![images](images/lena_region.png)
        
.. note:: 

  - PIL 中图像左上角坐标为 `(0, 0)`
  - `box(left, upper, right, lowe)`


- 从图像中复制子矩形、将子矩形粘贴回去

```python

import os
from PIL import Image

image_path = "/Users/zfwang/project/machinelearning/deeplearning/deeplearning/src/pillow_src/images"
image_name = "lena"

try:
    box = (100, 100, 200, 200)
    with Image.open(os.path.join(image_path, image_name + ".png")) as im:
        region = im.crop(box)
        region = region.transpose(Image.ROTATE_180)
        im.paste(region, box)
        im.save(os.path.join(image_path, image_name + "_region_paste" + ".png"))
except IOError as e:
    print(f"Can't open {image_name}")
```

![images](images/lena_region_paste.png)


- 图像滚动(image roll)

```python
import os
from PIL import Image

image_path = "/Users/zfwang/project/machinelearning/deeplearning/deeplearning/src/pillow_src/images"
image_name = "lena"

def roll(image, delta):
    """Roll an image sideways"""
    xsize, ysize = image.size
    delta = delta % xsize
    if delta == 0:
        return image
    part1 = image.crop((0, 0, delta, ysize))
    part2 = image.crop((delta, 0, xsize, ysize))
    image.paste(part1, (xsize - delta, 0, xsize, ysize))
    image.paste(part2, (0, 0, xsize - delta, ysize))

    return image

try:
    with Image.open(os.path.join(image_path, image_name + ".png")) as im:
        im = roll(im, 10)
        im.save(os.path.join(image_path, image_name + "_roll" + ".png"))
except OSError:
    print(f"cannot open {image_name}")
```

![images](images/lena_roll.png)

- RGB波段拆分、合并

```python
import os
from PIL import Image

image_path = "/Users/zfwang/project/machinelearning/deeplearning/deeplearning/src/pillow_src/images"
image_name = "lena"

try:
    with Image.open(os.path.join(image_path, image_name + ".png")) as im:
        r, g, b = im.split()
        im = Image.merge("RGB", (b, g, r))
        im.save(os.path.join(image_path, image_name + "_merge_gbr" + ".png"))
except OSError:
    print(f"cannot open {image_name}")
```

![images](images/lena.png)
![images](images/lena_merge_rbg.png)
![images](images/lena_merge_brg.png)
![images](images/lena_merge_bgr.png)
![images](images/lena_merge_grb.png)
![images](images/lena_merge_gbr.png)

.. note:: 

    - 对于单波段图像(single-band)，`Image.split` 返回图像本身
    - 为了对单个颜色波段进行处理，需要首先将图像转换为 RGB

## 图像几何变换

API:

- `Image.Image.resize`
- `Image.Image.rotate`
- `Image.transpose`
- `Image.transform`

示例:

- 简单的几何变换-改变图像像素大小

```python

import os
from PIL import Image

image_path = "/Users/zfwang/project/machinelearning/deeplearning/deeplearning/src/pillow_src/images"
image_name = "lena"

try:
    with Image.open(os.path.join(image_path, image_name + ".png")) as im:
        out = im.resize((1000, 1000))
        out.save(os.path.join(image_path, image_name + "_resize" + ".png"))
except OSError:
    print(f"cannot open {image_name}")
```

![images](images/lena_resize.png)

- 简单的几何变换-图像逆时针旋转一定的角度

```python
import os
from PIL import Image

image_path = "/Users/zfwang/project/machinelearning/deeplearning/deeplearning/src/pillow_src/images"
image_name = "lena"

try:
    with Image.open(os.path.join(image_path, image_name + ".png")) as im:
        out = im.rotate(45)
        out.save(os.path.join(image_path, image_name + "_rotate" + ".png"))
except OSError:
    print(f"cannot open {image_name}")
```

![images](images/lena_rotate.png)

- 图像转置

```python
import os
from PIL import Image

image_path = "/Users/zfwang/project/machinelearning/deeplearning/deeplearning/src/pillow_src/images"
image_name = "lena"

try:
    with Image.open(os.path.join(image_path, image_name + ".png")) as im:
        out1 = im.transpose(Image.FLIP_LEFT_RIGHT)
        out2 = im.transpose(Image.FLIP_TOP_BOTTOM)
        out3 = im.transpose(Image.ROTATE_90)
        out4 = im.transpose(Image.ROTATE_180)
        out5 = im.transpose(Image.ROTATE_270)
        out1.save(os.path.join(image_path, image_name + "_rotate_1" + ".png"))
        out2.save(os.path.join(image_path, image_name + "_rotate_2" + ".png"))
        out3.save(os.path.join(image_path, image_name + "_rotate_3" + ".png"))
        out4.save(os.path.join(image_path, image_name + "_rotate_4" + ".png"))
        out5.save(os.path.join(image_path, image_name + "_rotate_5" + ".png"))
except OSError:
    print(f"cannot open {image_name}")
```

![images](images/lena.png)
![images](images/lena_rotate_1.png)
![images](images/lena_rotate_2.png)
![images](images/lena_rotate_3.png)
![images](images/lena_rotate_4.png)
![images](images/lena_rotate_5.png)

.. note:: 

    - `trasnpose(ROTATE)` 与 `Image.Image.rotate` 效果相同
    - `transform()` 能进行更多形式的图像转换

## 图像颜色转换

## 图像增强


## 图像序列

## PostScript 打印

## 图像读取


## 控制图像编码



# Pillow API

- Module
    - Image
    - ImageChops
    - ImageCms
    - ImageColor
    - ImageDraw
    - ImageEnhance
    - ImageFile
    - ImageFilter
    - ImageFont
    - ImageGrab
    - ImageMath
    - ImageMorph
    - ImageOps
    - ImagePaletee
    - ImagePath
    - ImageQt
    - ImageSequence
    - ImageShow
    - ImageStat
    - ImageTK
    - ImageWin
    - ExifTags
    - TiffTags
    - JpegPressets
    - PSDraw
    - PixelAccess
    - PyAccess
    - features
- PIL Package
    - PIL
- Plugin reference
- Internal Reference Docs

