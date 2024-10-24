---
title: CNN 网络概览
author: wangzf
date: '2022-07-15'
slug: cnn
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

- [CNN 发展历史](#cnn-发展历史)
- [CNN 整体结构](#cnn-整体结构)
- [CNN 卷积层](#cnn-卷积层)
  - [卷积的含义](#卷积的含义)
    - [卷积的数学解释](#卷积的数学解释)
    - [卷积的图形解释](#卷积的图形解释)
  - [卷积类型](#卷积类型)
    - [单通道卷积](#单通道卷积)
    - [多通道卷积](#多通道卷积)
    - [3D 卷积](#3d-卷积)
    - [转置卷积](#转置卷积)
    - [1x1 卷积](#1x1-卷积)
    - [深度可分离卷积](#深度可分离卷积)
    - [空洞卷积](#空洞卷积)
  - [卷积步幅](#卷积步幅)
  - [卷积填充](#卷积填充)
  - [卷积输出维度](#卷积输出维度)
  - [卷积滤波器初始化和学习](#卷积滤波器初始化和学习)
- [CNN 池化层](#cnn-池化层)
  - [池化层介绍](#池化层介绍)
  - [池化层的作用](#池化层的作用)
  - [池化层操作](#池化层操作)
- [CNN 全连接层 Full Connected layer](#cnn-全连接层-full-connected-layer)
- [参考](#参考)
</p></details><p></p>

# CNN 发展历史

CNN 在计算机视觉的三大领域: **图像识别** 、**目标检测**、**语义分割(图像分割)** 有着广泛的应用

1985年，Rumelhart 和 Hinton 等人提出了 [反向传播算法，Back-Propagaion](http://www.cs.utoronto.ca/~hinton/absps/naturebp.pdf)，
即著名的反向传播算法训练神经网络模型，奠定了神经网络的理论基础

深度学习三巨头(Yann LeCun, Geoffrey Hinton, Yoshua Bengio)之一 Yann LeCun 在 BP 算法提出之后三年，
发现可以用 BP 算法训练一种构造出来的多层卷积网络结构，并用其训练出来的卷积网络识别手写数字。
LeCun 正式提出了[卷积神经网络(Convolutional Neural Network, CNN)](http://yann.lecun.com/exdb/publis/pdf/lecun-89e.pdf)的概念

LeCun 正式提出 CNN 概念之后，在 1998年 提出了 CNN 的开山之作：[LeNet-5 网络](http://yann.lecun.com/exdb/publis/pdf/lecun-01a.pdf)

进入 21 世纪后，由于计算能力和可解释性等多方面的原因，神经网络的发展经历了短暂的低谷，
直到 2012 年 ILSVRC 大赛上 AlexNet 一举夺魁，此后大数据兴起，
以 CNN 为代表的深度学习方法逐渐成为计算机视觉、语音识别和自然语言处理等领域的主流方法，
CNN 才开始正式发展起来

# CNN 整体结构

全连接的神经网络或 DNN 中，Affine 层(仿射层)后面跟着激活函数 ReLU 层。
最后是 Affine 层加 Softmax 层输出最终结果(概率)，但是全连接层存在一些问题：

* 因为图像是 3 维的，这个形状应该含有重要的空间信息。比如，空间上相邻的像素为相似的值、
  RGB 的各个通道之间分别有密切的关联性、相距较远的像素之间没有什么关联等
* 全连接层会忽略数据的形状，它会将输入数据作为相同的神经元(同一维度的神经元)处理，
  所以无法利用与形状相关的信息

一个典型的 CNN 通常包括下面这三层：

* **卷积层(Convolutional layer)**
    - CNN 相较于 DNN，其主要区别就在于卷积层，卷积层的存在使得神经网络具备更强的学习和特征提取能力
    - 卷积层可以保持数据形状不变，当输入数据是图像时，卷积层会以 3 维数据的形式接收输入数据，
      并同样以 3 维数据的形式输出至下一层。因此，在 CNN 中，可以正确理解图像等具有形状的数据
- **池化层(Pooling layer)**
    - CNN 中在卷积层后面会跟着一个池化层，池化层的存在使得 CNN 的稳健性更好
- **全连接层(Full Connected layer)**
    - CNN 的最后是 DNN 中常见的全连接层

# CNN 卷积层 

> Convolutional layer

## 卷积的含义

从数学的角度解释, 卷积可以理解为一种类似于 **加权运算** 一样的操作。
在图像处理中, 针对图像的像素矩阵, 
**卷积操作就是用一个卷积核来逐行逐列的扫描像素矩阵, 并与像素矩阵做元素相乘, 以此得到新的像素矩阵**, 这个过程称为卷积(Convolutional). 其中：

* 卷积核也叫作 `过滤器` 或者 `滤波器(filter)`
* 滤波器在输入的像素矩阵上扫过的面积称为 `感受野(receptive field)`
* 卷积层的输入输出数据称为 `特征图(feature map)`
    - 卷积层的输入数据称为 `输入特征图`
    - 卷积层的输出数据称为 `输出特征图`

### 卷积的数学解释

在泛函分析中, 卷积也叫做 **旋积** 或者 **褶积**, 
是一种通过两个函数 `$x(t)$` 和 `$h(t)$` 生成的数学算子, 
其计算公式如下:

* 连续形式

`$$x(t)h(t)(\tau)=\int_{- \infty}^{+ \infty} x(\tau)h(\tau - t)dt$$`

* 离散形式

`$$x(x)h(x)(\tau)=\sum_{\tau=-\infty}^{+ \infty} x(\tau)h(\tau - t)$$`

两个函数的卷积就是:

* 卷
    - 将一个函数 `$h(t)$` 进行翻转(Reverse), 然后再做一个平移(Shift), 得到 `$h(\tau - t)$` 
* 积
    - 将平移后的函数 `$h(\tau - t)$` 与另外一个函数 `$x(\tau)$`  对应元素相乘求和/积分

所以卷积本质上就是一个 Reverse-Shift-Weighted Summation 的操作

### 卷积的图形解释

函数 `$x(t)$` 和 `$h(t)$` 图像如下:

![img](images/conv_fun.png)

* 卷
    - 对函数 `$h(t)$` 进行翻转(Reverse), 得到 `$h(-\tau)$`:
  
    ![img](images/h_reverse.png)
  
    - 对函数 `$h(-\tau)$` 进行平移(Shift), 得到 `$h(t-\tau)$` :
  
    ![img](images/h_shift.png)
* 积

![img](images/conv_sum.png)

## 卷积类型

因为图像有单通道图像(灰度图)和多通道图(RGB 图)，所以对应常规卷积方式可以分为单通道卷积和多通道卷积。
二者本质上并无太大差异，无非对每个通道都要进行卷积而已。
针对图像的像素矩阵，卷积操作就是用一个卷积核来逐行逐列的扫描像素矩阵，
并与像素矩阵做元素相乘，以此得到新的像素矩阵

### 单通道卷积

一个标准的单通道卷积如下图所示：

![img](images/conv.gif)

对于单通道卷积，假设输入图像(输入特征图)维度为 `$n \times n \times c = 5 \times 5 \times 1$`，
滤波器维度为 `$f \times f \times c = 3 \times 3 \times 1$`，步长(stide)为 `$s=1$`，填充(padding) 大小为 `$p=0$`，
那么输出图像(输出特征图)维度可以计算为：

`$$\Bigg(\frac{n+2p-f}{s} + 1\Bigg) \times \Bigg(\frac{n+2p-f}{s} + 1\Bigg) \times c =3 \times 3 \times 1$$`

### 多通道卷积

![img](images/3Dconv1.png)
![img](images/3Dconv2.png)
![img](images/3Dconv3.png)

3 维卷积运算的输入图像数据为 3 通道(channel)的 RGB 数据

- 2 维图像数据的卷积运算都是以高、长方向的 2 维形状为对象的, 通道数为 1
    - 2 维图像数据形状为 `$(height, width)$`
- 3 维图像数据除了高、长方向外还需要处理 **通道(channel)** 方向, 通道数为 3
    - 3 维图像数据形状为 `$(channel, height, width)$`

3 维图像数据卷积运算

- 3维数据在通道方向上特征图增加了, 通道方向上有多个特征图时, 
    会按照通道进行输入图像数据和滤波器的卷积运算, 并将结果相加, 从而得到输出特征图
- 输入图像数据和滤波器的通道数要设为相同的值, 并且每个通道上的滤波器形状应相同
    - 输入图像形状: `$(channel, input\_height, input\_width)$`
    - 滤波器: `$(channel, filter\_height, filter\_width)$`
- 3 维图像数据卷积运算输出数据形状
    - 假设原始输入像素矩阵中: `$shape = (3, n, n)$` , 滤波器: `$shape = (3, f, f)$` , 步幅: `$s$`, 
        使用 same padding: 填充 `$p$` 个像素
        - 输出像素矩阵: `$shape = \Big(1, \frac{(n + 2p - f)}{s + 1}, \frac{(n + 2p - f)}{s + 1}\Big)$`
- 3 维卷积运算对于比较复杂的图像数据进行特征提取时, 可以使用多个滤波器(filter)
- 3 维卷积层的参数数量
    - 一个滤波器的参数数量为 `$f * f * nc$` , 其中 `$nc$` 是通道数量, 
    `$k$` 个滤波器的参数数量为 `$f * f * nc * k$`

多通道图卷积与单通道卷机相似，比如现在有一个输入图像维度为 `$5 \times 5 \times 3$` 的 RGB 3 通道图像，可以将其看成是 3 张 `$5 \times 5$` 图像的堆叠，
这时候把原先的单通道滤波器乘 3，用 3 个滤波器分别对着 3 张图像进行卷积，
将卷积得到 3 个特征图加总起来便是最后结果。这里强调一点：
滤波器的通道数一定要跟输入图像的通道数一致，不然会漏下某些通道得不到卷积

现在用 `$f \times f \times c = 3 \times 3 \times 3$` 的滤波器对 `$n \times n \times c = 5 \times 5 \times 3$` 的输入进行卷积，
得到的输出维度为 `$3 \times 3$`。这里少了通道数，所以一般会用多个 3 通道滤波器来进行卷积，
假设这里用了 10 个 `$3 \times 3 \times 3$` 的滤波器，那最后的输出便为 `$3 \times 3 \times 10$`，
滤波器的个数变成了输出特征图的通道数

![img](images/n_conv.gif)
![img](images/n_conv2.gif)

### 3D 卷积

将 2D 卷积增加一个深度维便可扩展为 3D 卷积。输入图像是 3 维的，滤波器也是 3 维的，对应的卷积输出同样是 3 维的

也可以从 3D 的角度来理解多通道卷积：可以将 `$3 \times 3 \times 3$` 的滤波器想象为一个三维的立方体，
为了计算立方体滤波器在输入图像上的卷积操作，首先将这个三维的滤波器放到左上角，
让三维滤波器的 27 个数依次乘以红绿蓝三个通道中的像素数据，即滤波器的前 9 个数乘以红色通道中的数据，
中间 9 个数乘以绿色通道中的数据，最后 9 个数乘以蓝色通道中的数据。
将这些数据加总起来，就得到输出像素的第一个元素值。示意图如下所示

![img](images/conv2d.png)

可以把 2D 卷积的计算输出公式进行扩展，可以得到 3D 卷积的输出维度计算公式。
假设输入图像大小为 `$a_{1} \times a_{2} \times a_{3} \times c$`，通道数为 `$c$`，
滤波器大小为 `$f \times f \times f \times c$`，
滤波器数量为 `$n$`，则输出维度可以表示为

`$$(a_{1} - f) \times (a_{2} - f) \times (a_{3} - f) \times n$$`

3D 卷积在医学影像数据、视频分类等领域都有着较为广泛的应用。
相较于 2D 卷积，3D 卷积的一个特点就是卷积计算量巨大，对计算资源要求相对较高

### 转置卷积

转置卷积(Transposed Convolution)也叫解卷积(Deconvolution)、反卷积。
在常规卷积时, 每次得到的卷积特征图尺寸是越来越小的, 但在图像分割等领域, 
需要逐步恢复输入时的尺寸, 如果把常规卷积时的特征图不断变小叫做 **下采样**, 
那么通过转置卷积来恢复分辨率的操作可以称作 **上采样**

![img](images/normal_conv.gif)
![img](images/transposed_conv.gif)

### 1x1 卷积

* TODO

### 深度可分离卷积

* TODO

### 空洞卷积

空洞卷积也叫扩张卷积、膨胀卷积, 
简单来说就是在卷积核元素之间加入一些空格(零)来扩大卷积核的的过程, 
可以用一个扩张率 `$a$`  来表示卷积核扩张的程度

* `$a=1, 2, 4$` 的时候卷积核的感受野

![img](images/dilated_conv.png)

* 扩展率 `$a=2$` 时的卷积过程

![img](images/dilated_conv.gif)

加入空洞之后的实际卷积核尺寸与原始卷积尺寸之间的关系如下:

`$$K = k+(k-1)(a-1)$$`

其中:

* `$K$` :加入空洞之后的卷积核尺寸
* `$k$` :原始卷积核尺寸
* `$a$`: :卷积扩张率

空洞卷积优点:

* 一个直接的作用就是可以扩大卷积感受野, 
 空洞卷积几乎可以在零成本的情况下就可以获得更大的感受野来扩充更多信息, 
 这有助于检测和分割任务中提高准确率
* 另一个优点就是可以捕捉多尺度的上下文信息, 
 当使用不同的扩展率来进行卷积核叠加时, 
 获取的感受野就丰富多样

## 卷积步幅

应用滤波器的位置的间隔称为 **步幅(Stirde)**，滤波器移动的步幅为 1 时, 
即滤波器在像素矩阵上一格一格平移, 滤波器也可以以 2 个或更多的单位平移

## 卷积填充

* **问题**:
    - 在进行卷积运算时, 原始输入像素矩阵的边缘和角落位置的像素点只能被滤波器扫描到一次, 
      而靠近像素中心点位置的像素点则会被多次扫描到进行卷积, 这使得边缘和角落里的像素特征提取不足; 
    - 在 CNN 中进行卷积运算, 输出数据的大小会越来越小, 反复进行多次卷积运算后, 
      某个时刻的输出大小就有可能变为 `$1 \times 1$`, 导致无法再应用卷积运算, 
      为了避免这样的情况, 就需要使用卷积填充(Padding), 
      可以使得卷积运算在保持空间大小不变的情况下降数据传给下一层
* **Padding 定义**:
    - 在进行卷积层处理之前, 需要向输入数据的周围填充固定的数据(比如 0), 
      这称为填充(Padding)
* **Padding 作用**:
    - 使用 Padding 的主要目的是为了调整输入数据的大小，
      使得输入像素矩阵中边缘和角落的像素特征可以被充分提取
    - 使得卷积运算后的输出数据大小保持合适
* **Padding 方法**:
    - valid Padding
        - 不填充
    - same Padding
        - 填充后输入和输出大小是一致的;
        - 对于 `$n \times n$` 的输入像素矩阵, 
          如果对输入像素矩阵每个边缘填充 `$p$` 个像素, 
          `$n$` 就变成了 `$n + 2p$`, 
          最后的输出像素矩阵的形状就变成了 `$shape = ((n+2p -f)/s+ 1, (n+2p -f)/s+ 1)$`
        - 如果想让输入数据的大小与输出数据的大小相等, 即 `$(n+2p-f)/s + 1 = n$`, 
          则对输入像素矩阵每个边缘 Padding 的像素个数为 `$p = (f-1)/2$`
        - 综上, 一般而言, 滤波器的大小 `$f$` 都会选择为奇数个; 
* **Padding 实现**:

```python
import numpy as np

def zero_pad(X, pad):
    X_pad = np.pad(
        X, 
        ((0, 0), (pad, pad), (pad, pad), (0, 0)), 
        "constant"
    )
    return X_pad

np.random.seed(1)
x = np.random.randn(4, 3, 3, 2)
x_pad = zero_pad(x, 2)

fig, ax = plt.subplots(1, 2)
ax[0].set_title("x")
ax[0].imshow(x[0, :, :, 0])
ax[1].set_title("x_pad")
ax[1].imshow(x_pad[0, :, :, 0])
```

![img](images/padding.png)

## 卷积输出维度

> 如何确定经过卷积后的输出矩阵的维度？

对于单通道卷积和多通道卷积，假设输入图像维度为 `$n \times n \times c$`，
滤波器维度为 `$f\times f \times c$`，卷积步长(stide)为 `$s$`，填充(padding) 大小为 `$p$`，
那么输出维度可以计算为：

`$$\Bigg(\frac{n+2p-f}{s} + 1\Bigg) \times \Bigg(\frac{n+2p-f}{s} + 1\Bigg)$$`

## 卷积滤波器初始化和学习

* 在 DNN 中, 参数有权重和偏置, 在 CNN 中, 滤波器的参数就对应 DNN 中的权重, 并且, CNN 中也存在偏置, 通常是一个标量数字; 
* 在训练 CNN 时, 需要初始化滤波器中的卷积参数, 在训练中不断迭代得到最好的滤波器参数; 
* 卷积层的参数通常在于滤波器, 可以计算一个滤波器的参数数量为 `$f \times f \times nc$` , 其中 `$nc$` 是通道数量

# CNN 池化层

> Pooling layer

## 池化层介绍

* 通常在设计 CNN 时, 卷积层后会跟着一个池化层 
* 池化层的操作类似于卷积, 
  只是将滤波器与感受野之间的元素相乘改成了利用树池对感受野直接进行采样(最大/平均) 
* 池化层的参数:
    - 滤波器的大小 `$f$`
    - 步幅 `$s$`
* 池化层只是计算神经网路某一层的静态属性, 中间没有学习过程 

![img](images/pooling.png)

## 池化层的作用

* 缩减模型大小, 对输入矩阵的高度和宽度进行缩小
* 提高模型计算速度
* 提高所提取特征的稳健性 

## 池化层操作

* 最大池化(max pooling)
    - 设置一个树池:
        - `$f \times f$` 的滤波器
        - 步幅: `$s$`
    - 将输入矩阵拆分为不同的区域
    - 输出输入矩阵不同区域的最大元素值
* 平均池化(average pooling)
    - 设置一个树池:
        - `$f \times f$` 的滤波器
        - 步幅: `$s$`
    - 将输入矩阵拆分为不同的区域
    - 输出输入矩阵不同区域的元素值的平均值

# CNN 全连接层 Full Connected layer

池化完成之后就是标准 DNN 中的全连接层了。相邻层的所有神经元之间都有连接, 
这称为全连接层(Full Connected layer), 可以使用 Affine 层实现全连接层

# 参考

* [CNN卷积方法一览](https://mp.weixin.qq.com/s/9RvkFMOxmUTdZGDqjZfELw)
* [Kernel](https://en.wikipedia.org/wiki/Kernel_(image_processing)?ref=blog.paperspace.com)
