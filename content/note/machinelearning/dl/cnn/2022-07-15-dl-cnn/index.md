---
title: CNN
author: 王哲峰
date: '2022-07-15'
slug: dl-cnn
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

- [CNN 发展历史](#cnn-发展历史)
- [CNN 整体结构](#cnn-整体结构)
- [卷积层 Convolutional layer](#卷积层-convolutional-layer)
  - [卷积的含义](#卷积的含义)
    - [卷积的数学解释](#卷积的数学解释)
    - [卷积的图形解释](#卷积的图形解释)
  - [卷积的类型](#卷积的类型)
    - [常规卷积](#常规卷积)
    - [3D 卷积](#3d-卷积)
    - [转置卷积](#转置卷积)
    - [1x1 卷积](#1x1-卷积)
    - [深度可分离卷积](#深度可分离卷积)
    - [空洞卷积](#空洞卷积)
  - [卷积计算](#卷积计算)
  - [卷积步幅](#卷积步幅)
  - [卷积填充](#卷积填充)
  - [卷积输出维度](#卷积输出维度)
  - [卷积滤波器初始化和学习](#卷积滤波器初始化和学习)
  - [三维卷积运算](#三维卷积运算)
- [池化层 Pooling layer](#池化层-pooling-layer)
  - [池化层介绍](#池化层介绍)
  - [池化层的作用](#池化层的作用)
  - [池化层操作](#池化层操作)
- [全连接层 Full Connected layer](#全连接层-full-connected-layer)
- [CNN 图像学习过程](#cnn-图像学习过程)
  - [CNN 的直观理解](#cnn-的直观理解)
  - [卷积](#卷积)
  - [CNN 结构](#cnn-结构)
    - [全连接层](#全连接层)
    - [池化层](#池化层)
    - [卷积层](#卷积层)
    - [卷积层与池化层级联](#卷积层与池化层级联)
    - [CNN 完整结构](#cnn-完整结构)
  - [卷积层](#卷积层-1)
  - [池化层](#池化层-1)
  - [激活函数](#激活函数)
  - [全连接层](#全连接层-1)
    - [原理](#原理)
    - [示例](#示例)
    - [对模型的影响](#对模型的影响)
  - [图像在 CNN 网络中的变化](#图像在-cnn-网络中的变化)
    - [数据](#数据)
    - [模型](#模型)
    - [一个卷积层](#一个卷积层)
    - [一个池化层](#一个池化层)
    - [激活函数](#激活函数-1)
    - [新卷积层](#新卷积层)
    - [新采样层](#新采样层)
    - [新激活函数](#新激活函数)
    - [将激活函数 relu 修改为 sigmoid](#将激活函数-relu-修改为-sigmoid)
    - [将激活函数 relu 修改为 tanh](#将激活函数-relu-修改为-tanh)
    - [新增两个卷积层和激活函数](#新增两个卷积层和激活函数)
    - [全连接层](#全连接层-2)
    - [不同 kernel size 对比](#不同-kernel-size-对比)
- [参考](#参考)
</p></details><p></p>

# CNN 发展历史

CNN 在计算机视觉的三大领域: **图像识别** 、**目标检测**、**语义分割(图像分割)** 有着广泛的应用

* 1985年, Rumelhart 和 Hinton 等人提出了 
  **BP神经网络** [Learning internal representations by error propagation, 1985](https://web.stanford.edu/class/psych209a/ReadingsByDate/02_06/PDPVolIChapter8.pdf)), 
  即著名的反向传播算法训练神经网络模型, 奠定了神经网络的理论基础.
* 深度学习三巨头(Yann LeCun, Geoffrey Hinton, Yoshua Bengio)之一 Yann LeCun 在 BP 神经网络提出之后三年, 
  发现可以用 BP 算法训练一种构造出来的多层卷积网络结构, 并用其训练出来的卷积网络识别手写数字。
  在 [Backpropagation applied to handwritten zip code recognition, 1989](http://yann.lecun.com/exdb/publis/pdf/lecun-89e.pdf)) 一文中, 
  LeCun 正式提出了 **卷积神经网络(Convolutional Neural Network,CNN)** 的概念.
* LeCun 正式提出 CNN 概念之后, 
  在 1998年 提出了 CNN 的开山之作 —— **LeNet-5 网络** [Gradient-based learning applied to document recognition, 1998](http://yann.lecun.com/exdb/publis/pdf/lecun-01a.pdf)).
* 进入21世纪后, 由于计算能力和可解释性等多方面的原因, 神经网络的发展经历了短暂的低谷, 直到 2012 年 ILSVRC 大赛上 **AlexNet** 一举夺魁, 
  此后大数据兴起, 以 CNN 为代表的深度学习方法逐渐成为计算机视觉、语音识别和自然语言处理等领域的主流方法, CNN 才开始正式发展起来。

# CNN 整体结构

* 全连接的神经网络或 DNN 中, Affine 层后面跟着激活函数 ReLU 层。最后是 Affine 层加 Softmax 层输出最终结果(概率)
    - 全连接层存在的问题:
        - 因为图像是3维的, 这个形状应该含有重要的空间信息。比如, 
          空间上相邻的像素为相似的值、RGB 的各个通道之间分别有密切的关联性、
          相距较远的像素之间没有什么关联等
        - 全连接层会忽略数据的形状, 它会将输入数据作为相同的神经元(同一维度的神经元)处理, 
          所以无法利用与形状相关的信息
* 一个典型的 CNN 通常包括下面这三层:
    - **卷积层(Convolutional layer)**
        - CNN 相较于 DNN, 其主要区别就在于 **卷积层(Convolutional layer)**, 
          卷积层的存在使得神经网络具备更强的学习和特征提取能力
        - 卷积层可以保持数据形状不变, 当输入数据是图像时, 卷积层会以 3 维数据的形式接收输入数据, 
          并同样以 3 维数据的形式输出至下一层。因此, 在 CNN 中, 可以正确理解图像等具有形状的数据
    - **池化层(Pooling layer)**
        - CNN 中在卷积层后面会跟着一个 **池化层(Pooling layer)**, 池化层的存在使得 CNN 的稳健性更好
    - **全连接层(Full Connected layer)**
        - CNN 的最后是 DNN 中常见的 **全连接层(Full Connected layer)**

# 卷积层 Convolutional layer

## 卷积的含义

从数学的角度解释, 卷积可以理解为一种类似于 **加权运算** 一样的操作。在图像处理中, 针对图像的像素矩阵, 
**卷积操作就是用一个卷积核来逐行逐列的扫描像素矩阵, 并与像素矩阵做元素相乘, 以此得到新的像素矩阵**, 
这个过程称为卷积(Convolutional). 其中：

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

## 卷积的类型

### 常规卷积

因为图像有单通道图像(灰度图)和多通道图(RGB图)，所以对应常规卷积方式可以分为单通道卷积和多通道卷积。
二者本质上并无太大差异，无非对每个通道都要进行卷积而已。
针对图像的像素矩阵，卷积操作就是用一个卷积核来逐行逐列的扫描像素矩阵，
并与像素矩阵做元素相乘，以此得到新的像素矩阵。其中卷积核也叫过滤器或者滤波器，
滤波器在输入像素矩阵上扫过的面积称之为感受野。一个标准的单通道卷积如下图所示：

对于单通道卷积，假设输入图像维度为 `$n \times n \times c = 5 \times 5 \times 1$`，
滤波器维度为 `$f \times f \times c = 3 \times 3 \times 1$`，卷积步长(stide)为 `$s=1$`，填充(padding) 大小为 `$p=0$`，
那么输出维度可以计算为：

`$$\Bigg(\frac{n+2p-f}{s} + 1\Bigg) \times \Bigg(\frac{n+2p-f}{s} + 1\Bigg)=3 \times 3$$`

![img](images/conv.gif)

多通道图卷积与单通道卷机相似，比如现在有一个输入图像维度为 `$5 \times 5 \times 3$` 的 RGB 3 通道图像，
可以将其看成是 3 张 `$5 \times 5$` 图像的堆叠，这时候把原先的单通道滤波器 `$\times 3$`，
用 3 个滤波器分别对着 3 张图像进行卷积，将卷积得到 3 个特征图加总起来便是最后结果。
这里强调一点：滤波器的通道数一定要跟输入图像的通道数一致，不然会漏下某些通道得不到卷积

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
那么通过装置卷积来恢复分辨率的操作可以称作 **上采样**

![img](images/normal_conv.gif)
![img](images/transposed_conv.gif)

### 1x1 卷积

* TODO

### 深度可分离卷积

* TODO

### 空洞卷积

空洞卷积也叫扩张卷积、膨胀卷积, 简单来说就是在卷积核元素之间加入一些空格(零)来扩大卷积核的的过程, 
可以用一个扩张率 `$a$`  来表示卷积核扩张的程度。

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

## 卷积计算

![img](images/conv.gif)

* 在上面的图中, 用一个 `$3 \times 3$` 的滤波器扫描一个 `$5 \times 5$`
  的输入像素矩阵。用滤波器中每一个元素与像素矩阵中感受野内的元素进行乘积运算, 可以得到一个
  `$3 \times 3$` 的输出像素矩阵。这个输出的 `$3 \times 3$`
  的输出像素矩阵能够较大程度地提取原始像素矩阵的图像特征, 这也是卷积神经网络之所以有效的原因
* 将各个位置上滤波器的元素和输入像素矩阵的对应元素相乘, 然后再求和。最后, 将这个结果保存到输出的对应位置; 
  将这个过程在输入像素矩阵的所有位置都进行一遍, 就可以得到卷积运算的输出

## 卷积步幅

* 应用滤波器的位置的间隔称为 **步幅(Stirde)**
* 滤波器移动的步幅为 1 时, 即滤波器在像素矩阵上一格一格平移, 滤波器也可以以 2 个或更多的单位平移

## 卷积填充

* **问题**:
    - 在进行卷积运算时, 原始输入像素矩阵的边缘和角落位置的像素点只能被滤波器扫描到一次, 
      而靠近像素中心点位置的像素点则会被多次扫描到进行卷积, 这使得边缘和角落里的像素特征提取不足; 
    - 在 CNN 中进行卷积运算, 输出数据的大小会越来越小, 反复进行多次卷积运算后, 某个时刻的输出大小就有可能变为
      `$1 \times 1$`, 导致无法再应用卷积运算, 为了避免这样的情况, 就需要使用卷积填充(Padding), 
      可以使得卷积运算在保持空间大小不变的情况下降数据传给下一层
* **Padding 定义**:
    - 在进行卷积层处理之前, 需要向输入数据的周围填充固定的数据(比如0), 这称为 **填充(Padding)**
* **Padding 作用**:
    - 使用 Padding 的主要目的是为了调整输入数据的大小
        - 使得输入像素矩阵中边缘和角落的像素特征可以被充分提取
        - 使得卷积运算后的输出数据大小保持合适
* **Padding 方法**:
    - valid Padding
        - 不填充
    - same Padding
        - 填充后输入和输出大小是一致的;
        - 对于 `$n \times n$` 的输入像素矩阵, 如果对输入像素矩阵每个边缘填充 `$p$` 个像素, 
          `$n$` 就变成了 `$n + 2p$`, 最后的输出像素矩阵的形状就变成了 `$shape = ((n+2p -f)/s+ 1, (n+2p -f)/s+ 1)$`。
        - 如果想让输入数据的大小与输出数据的大小相等, 即 `$n+2p-f + 1 = n$`, 
          则对输入像素矩阵每个边缘 Padding 的像素个数为 `$p = (f-1)/2$`
        - 综上, 一般而言, 滤波器的大小 `$f$` 都会选择为奇数个; 
* **Padding 实现**:

```python
import numpy as np

def zero_pad(X, pad):
    X_pad = np.pad(X, ((0, 0), (pad, pad), (pad, pad), (0, 0)), "constant")
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

假设原始输入像素矩阵中: `$shape = (n, n)$`, 滤波器: `$shape = (f, f)$`, 步幅: `$stride = s$`

- 如果滤波器的步幅 `$s=1$`, 那么输出像素矩阵:
   - `$shape = (n-f+1, n-f+1)$`
- 如果滤波器的步幅 `$s \geq 1$`, 那么输出像素矩阵:
   - `$shape = \Big(\frac{(n-f)}{s+1}, \frac{(n-f)}{s+1}\Big)$`

## 卷积滤波器初始化和学习

* 在 DNN 中, 参数有权重和偏置, 在 CNN 中, 滤波器的参数就对应 DNN 中的权重, 并且, CNN 中也存在偏置, 通常是一个标量数字; 
* 在训练 CNN 时, 需要初始化滤波器中的卷积参数, 在训练中不断迭代得到最好的滤波器参数; 
* 卷积层的参数通常在于滤波器, 根据滤波器的带下, 可以计算一个滤波器的参数数量为 `$f * f * nc$` , 其中 `$nc$` 是通道数量

## 三维卷积运算

![img](images/3Dconv1.png)
![img](images/3Dconv2.png)
![img](images/3Dconv3.png)

* 3 维卷积运算的输入图像数据为 3 通道(channel)的 RGB 数据
    - 2 维图像数据的卷积运算都是以高、长方向的 2 维形状为对象的, 通道数为 1
        - 2 维图像数据形状为 `$(height, width)$`
    - 3 维图像数据除了高、长方向外还需要处理 **通道(channel)** 方向, 通道数为 3
        - 3 维图像数据形状为 `$(channel, height, width)$`
* 3 维图像数据卷积运算
    - 3维数据在通道方向上特征图增加了, 通道方向上有多个特征图时, 
      会按照通道进行输入图像数据和滤波器的卷积运算, 并将结果相加, 从而得到输出特征图
    - 输入图像数据和滤波器的通道数要设为相同的值, 并且每个通道上的滤波器形状应相同
        - 输入图像形状: `$(channel, input_height, input_width)$`
        - 滤波器: `$(channel, filter_height, filter_width)$`
    - 3 维图像数据卷积运算输出数据形状
        - 假设原始输入像素矩阵中: `$shape = (3, n, n)$` , 滤波器: `$shape = (3, f, f)$` , 步幅: `$s$`, 
          使用 same padding: 填充 `$p$` 个像素
            - 输出像素矩阵: `$shape = \Big(1, \frac{(n + 2p - f)}{s + 1}, \frac{(n + 2p - f)}{s + 1}\Big)$`
    - 3 维卷积运算对于比较复杂的图像数据进行特征提取时, 可以使用多个滤波器(filter)
    - 3 维卷积层的参数数量
      - 一个滤波器的参数数量为 `$f * f * nc$` , 其中 `$nc$` 是通道数量, 
        `$k$` 个滤波器的参数数量为 `$f * f * nc * k$`

# 池化层 Pooling layer

## 池化层介绍

* 通常在设计 CNN 时, 卷积层后会跟着一个池化层; 
* 池化层的操作类似于卷积, 只是将 `滤波器与感受野之间的元素相乘` 改成了 `利用树池对感受野直接进行采样(最大/平均)`; 
* 池化层的参数:
    - 滤波器的大小 `$f$`
    - 步幅 `$s$`
* 池化层只是计算神经网路某一层的静态属性, 中间没有学习过程; 

![img](images/pooling.png)

## 池化层的作用

* 缩减模型大小, 对输入矩阵的高度和宽度进行缩小; 
* 提高模型计算速度; 
* 提高所提取特征的稳健性; 

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

# 全连接层 Full Connected layer

池化完成之后就是标准 DNN 中的全连接层了。相邻层的所有神经元之间都有连接, 
这称为全连接层(Full Connected layer), 可以使用 Affine 层实现全连接层

# CNN 图像学习过程

## CNN 的直观理解

从可视化的角度观察 CNN 每一层在图像识别过程中到底都学到了什么. 
2014 年 Zeiler 等人在 ECCV 上发表了一篇基于可视化角度理解 CNN 的经典论文, 
可谓是卷积神经网络可视化的开山之作(Visualizing and Understanding Convolutional Networks, 2014)

CNN在学习过程中是 **逐层对图像特征进行识别和检验** 的, CNN 的不同层负责检测输入图像的不同层级的图像特征。在 CNN 中

* 前几层网络用于检测 **图像的边缘特征**, 包括图像的基本轮廓
    - 边缘检测的目的就是检测出图像中亮度变化和特征较为明显的点和线
* 中间网络层用于检测 **图像中物体的部分区域**
* 后几层网络用于检测 **图像中完整的物体**

## 卷积

卷积(Convolution)公式：

`$$\int_{-\infty}^{\infty}f(\tau)g(x- \tau)d \tau$$`

卷积的物理意义大概可以理解为：

* 系统某一时刻的输出是由多个输入共同作用(叠加)的结果

卷积放在图像分析里：

* `$f(x)$` 可以理解为原始像素点(source pixel)，
  所有的原始像素点叠加起来，就是原始图像了
* `$g(x)$` 可以称为作用点，所有作用点合起来我们称为卷积核(convolution kernel)

卷积核上所有作用点依次作用于原始像素点后(乘起来)，线性叠加的输出结果，
即是最终卷积的输出，也是我们想要的结果，称为特征像素(destination pixel)

![img](images/convolution.jpg)

图像的锐化和边缘检测很像，先检测边缘，然后把边缘叠加到原来的边缘上，
原本图像边缘的值如同被加强了一般，亮度没有变化，但是更加锐利，
仔细想想卷积的物理意义，是不是仿佛有点卷积的感觉

> 一般通过什么操作检测边缘？

对于一维函数 `$f(x)$`，其一、二阶微分定义如下：

一阶微分的基本定义是差值：

`$$\frac{\partial f}{\partial x} = f(x + 1) - f(x)$$`

二阶微分为如下差分：

`$$\frac{\partial ^{2} f}{\partial x^{2}} = f(x + 1) + f(x - 1) - 2f(x)$$`

假设 `$f(x)$` 为图像边缘像素值(`$x$`)的灰度函数，
图像边缘的灰度分布图以及将 `$f(x)$` 的一、二阶微分作用于图像边缘上灰度的变化：

![img](images/gray_dist.jpg)

可以看到，在边缘(也就是台阶处)，二阶微分值非常大，其他地方值比较小或者接近 0。 
那就会得到一个结论，微分算子的响应程度与图像在用算子操作的这一点的突变程度成正比，
这样，图像微分增强边缘和其他突变(如噪声)，而削弱灰度变化缓慢的区域。
也就是说，微分算子(尤其是二阶微分)，对边缘图像非常敏感

很多时候，我们最关注的是一种各向同性的滤波器，这种滤波器的响应与滤波器作用的图像的突变方向无关。也就是说，各向同性滤波器是旋转不变的，即将原图像旋转之后进行滤波处理，与先对图像滤波再旋转的结果应该是相同的。可以证明，最简单的各向同性微分算子是拉普拉斯算子

一个二维图像函数 `$f(x, y)$` 的拉普拉斯算子定义为：

`$$\nabla^{2} f(x, y) = \frac{\partial^{2} f}{\partial x^{2}} + \frac{\partial^{2} f}{\partial y^{2}}$$`

那么对于一个二维图像 `$f(x, y)$`，可以用如下方法找到这个拉普拉斯算子：

`$$\frac{\partial^{2} f}{\partial x^{2}} = f(x + 1, y) + f(x - 1, y) - 2f(x, y)$$`

`$$\frac{\partial^{2} f}{\partial y^{2}} = f(x, y + 1) + f(x, y - 1) - 2f(x, y)$$`

因此：

`$$\begin{align}\nabla^{2} f(x, y) 
&= \frac{\partial^{2} f}{\partial x^{2}} + \frac{\partial^{2} f}{\partial y^{2}} \\
&= f(x + 1, y) + f(x - 1, y) + f(x, y + 1) + f(x, y - 1)- 4f(x, y)
\end{align}$$`

这个结果看起来太复杂，用别的方式重新表达一下，如果以 `$x, y$` 为坐标轴中心点，
算子为：

![img](images/img1.png)

这个有点和上面提到的卷积核(convolutional kernel)有点像了

由于拉普拉斯是一种微分算子，因此其应用强调的是图像中的灰度突变。
将原图像和拉普拉斯图像算子叠加在一起，从而得到锐化后的结果，于是模版就变为：

![img](images/img2.png)

上面这个，就是一个锐化卷积核模板了。原始边缘与它卷积，
得到的就是强化了的边缘(destination pixel)，图像变得更加锐利

> 如果所使用的模板定义有负的中心系数，那么必须将原图像减去经拉普拉斯变换后的图像，而不是加上他

另外，同样提取某个特征，经过不同卷积核卷积后效果也不一样，因为 CNN 里面卷积核的大小是有讲究的

![img](images/img3.png)

可以发现同样是锐化，下图 5x5 的卷积核要比上图 3x3 的卷积核效果细腻不少

综上，可以说明：

1. 原始图像通过与卷积核的数学运算，可以提取出图像的某些指定特征(features)
2. 不同卷积核，提取的特征也是不一样的
3. 提取的特征一样，不同的卷积核，效果也不一样

CNN 实际上也就是一个不断提取特征，进行特征选择，然后进行分类的过程，
卷积在 CNN 里，就是充当前排步兵，首先对原始图像进行特征提取。
所以我们首先要弄懂卷积在干什么，才能弄懂 CNN

## CNN 结构

### 全连接层

正常情况下，人眼看某张灰色图像，立马就能识别出图像中的对象，机器看某张图像，
实际上就是把所有像素(pixels)全放进一个神经网络，让神经网络去处理这些像素，
这个神经网络先叫全连接层：

![img](images/cnn_1.png)

抛开准确率不说，大概会得到 `$3 \times 32 \times 32 \times 1024 + 1024 \times 512 + 512 \times 10 + $` `$1024(bias) + 512(bias) = 3676672$` 个参数，
这些参数量是非常大的了，而这仅仅是一个 `$32 \times 32$` 的图像，
运算量非常大了。同时，由于参数过多，极其容易产生过拟合

### 池化层

但我们仔细观察图像，我们肉眼其实也是有选择性的，我们并不太关心上图中灰色的区域，
以及图像中对象的的黑色区域，我们更关心图像中对象与灰色区域的相交的边缘，
因为这才是我们判断一个对象是什么的主要依据。那我们会不会也可以在计算机里也这么做，
主要提取边缘特征，对于灰色和黑色这种冗余或者不重要的的区域特征，
我们尽量丢弃或者少保留，那么这样可能会减少参数或者减少提参数的过程。
既然这样，那我们干脆在全连接层前面，对输入图像进行一个预处理吧，
把那些没用的统统扔掉，于是我们加了个采集模块。
我们只保留那些我们想要的，或者相对比较重要的像素，叫它池化层或采样层。
所以，我们得到下面这个网络

![img](images/cnn_2.png)

### 卷积层

如果把图像分成块，一块一块看，仍然能识别图像中的对象，
也就是说人的大脑会把这些分散的图片组合起来进行识别，如果把分割的块进行位置调换，
就基本识别不出来了，也就是说，我们发现了两个现象：

* 如果我们只知道局部的图片，以及局部的相对位置，
  只要我们能将它正确组合起来，我们也可以对物体进行识别
* 局部与局部之间关联性不大，也就是局部的变化，很少影响到另外一个局部

我们还要解决两个问题：

1. 输入的只是原始图片，还没有提取图片的特征
2. 目前要处理的参数仍然非常多，需要对原始输入进行降维或者减少参数

卷积的作用就是提取图像中的特征，卷积再加上一些人眼识别图像的性质，
那么就会发现，卷积加上就可以得到如下神经网络：

![img](images/cnn_3.png)

### 卷积层与池化层级联

实际上，我们还会遇到两个问题：

* 一张图片特征这么多，一个卷积层提取的特征数量有限的，提取不过来啊！
* 我怎么知道最后采样层选出来的特征是不是重要的呢？

> 级联分类器(cascade of classifiers)大概意思就是从一堆弱分类器里面，
> 挑出一个最符合要求的弱分类器，用着这个弱分类器把不想要的数据剔除，
> 保留想要的数据。然后再从剩下的弱分类器里，再挑出一个最符合要求的弱分类器，
> 对上一级保留的数据，把不想要的数据剔除，保留想要的数据。
> 最后，通过不断串联几个弱分类器，进过数据层层筛选，最后得到我们想要的数据

那么，针对刚才的问题，也可以级联一个卷积层和采样层，
最简单的一个卷积神经网络，就诞生了：

![img](images/cnn_4.png)

### CNN 完整结构

CNN 主要由 3 钟模块构成：

* 卷积层
* 采样层
* 全连接层

大致上可以理解为：

1. 通过第一个卷积层提取最初特征，输出特征图(feature map)
2. 通过第一个采样层对最初的特征图(feature map)进行特征选择，去除多余特征,重构新的特征图
3. 第二个卷积层是对上一层的采样层的输出特征图(feature map)进行二次特征提取
4. 第二个采样层也对上层输出进行二次特征选择
5. 全连接层就是根据得到的特征进行分类

## 卷积层

> 卷积层如何提取特征？

下面是一个最简单的 CNN，起到一个分类器的作用

![img](images/cnn_4.png)

其中：

* 卷积层负责提取特征
* 采样层负责特征选择
* 全连接层负责分类

正常情况下，输入图像是 RGB 格式的，分别对应红(R)、绿(G)、蓝(B)三个颜色。
RGB 格式就是通过对红(R)、绿(G)、蓝(B)三个颜色通道的变化，
以及它们相互之间的叠加来得到各式各样的颜色, 这三个颜色通道叠加之后，
就是看到的 RGB 图片了

![img](images/rgb_1.png)

假设 RGB 三个分量的像素分别如下表示：

![img](images/rgb_2.png)

假设已经有合适的滤波器(卷积核, convolution kernel)了，接下来就是利用卷积核提取特征。
图像和卷积核卷积运算就可以得到特征值，就是 destination value。
卷积核放在神经网络里，就代表对应的权重(weight)，卷积核和图像进行点乘(dot product), 
就代表卷积核里的权重单独对相应位置的 pixel 进行作用。这里强调一下点乘，虽说称为卷积，
实际上是位置一一对应的点乘，不是真正意义的卷积。至于为什么要把点乘完所有结果加起来，
实际上就是把所有作用效果叠加起来，就好比前面提到的 RGB 图片，
红绿蓝分量叠加起来产生了一张真正意义的图像

下面是 RGB 三个分量和对应的三个卷积核，里面的数字相当于权重：

![img](images/rgb_3.png)

现在知道输入，知道神经元的权重了，根据神经网络公式进行卷积计算：

`$$y = \underset{i}{\sum}\omega_{i}x_{i} + b$$`

![img](images/rgb_4.png)

卷积可以提取特征，但是也不能随机找图像的 pixels 进行卷积。
卷积输出的特征图(feature map)，除了特征值本身外，还包含相对位置信息，
即提取出的相应的特征值也是按照顺序排列的。所以，卷积的方式也希望按照正确的顺序，
因此，实现卷积运算最后的方式就是从左到右，每隔 `$x$` 列 pixel，
向右移动一次卷积核进行卷积(`$x$` 可以自己定义)

下图中 黄->蓝->紫，就是卷积核移动的顺序，这里 `$x = 1$`：

![img](images/rgb_5.png)

当已经到最右，再从上到下，每隔 `$x$` 行 pixel，向下移动一次卷积核，移动完成，
再继续如上所述，从左到右进行

![img](images/rgb_6.png)

就这样，先从左到右，再从上到下，直到所有 pixels 都被卷积核过了一遍，
完成输入图片的第一层卷积层的特征提取

这里的 `$x$` 叫作 stride，就是步长的意思，如果 `$x = 2$`, 
就是相当每隔两行或者两列进行卷积。另外，分量的 pixel 外面还围了一圈 0，
称为补 0(zero padding)。因为添了一圈 0，实际上什么信息也没有添，但是
同样是 stride `$x=1$` 的情况下，补 0 比原来没有添 0 的情况下进行卷积，
从左到右，从上到下都多赚了 2 次卷积，这样第一层卷积层输出的特征图(feature map)仍然为 5x5，
和输入图片的大小一致。这样有什么好处呢

1. 获得的更多更细致的特征信息，上面那个例子我们就可以获得更多的图像边缘信息
2. 可以控制卷积层输出的特征图的尺寸，从而可以达到控制网络结构的作用，还是以上面的例子，
   如果没有做 zero-padding 以及第二层卷积层的卷积核仍然是 3x3, 
   那么第二层卷积层输出的特征图就是 1x1，CNN 的特征提取就这么结束了。
   同样的情况下加了 zero-padding 的第二层卷积层输出特征图仍然为 5x5，
   这样我们可以再增加一层卷积层提取更深层次的特征

## 池化层

> 池化层 pooling，也叫采样层 subsample

卷积层输出的特征图(feature map)到了池化层(pooling，也叫采样层 subsample)，
池化层实际上就是一个特征选择的过程

* max pooling
    - 通过 pooling 滤波器选取特征图(特征值矩阵)中的最大值，
      这个最大值可以理解为能代表这个特征的程度，
      比如上一层卷积层的滤波器是边缘滤波器，那这个最大值就代表在这个区域，
      这一块部位最符合边缘特征，max pooling 就是在这个区域内选出最能代表边缘的值，
      然后丢掉那些没有多大用的信息
    - 如果不进行 pooling，会过拟合，并且参数过多导致运算量大，
      可能还会难以平衡上一次卷积层或下一层卷积层的关系，无法满足模型结构需求
* average pooling
    - 通过 pooling 滤波器选取特征图(特征值矩阵)中的平均值

池化层还有一些性质，比如它可以一定程度提高空间不变性，
比如说平移不变性、尺度不变性、形变不变性。为什么会有空间不变性呢？
因为上一层卷积本身就是对图像一个区域一个区域去卷积，因此对于 CNN 来说，
重要是单独区域的特征，以及特征之间的相对位置(而不是绝对位置)图像细微的变换。
经过卷积，maxpooling 之后，输出结果和原来差别可能不算大，或者没有差别

Pooling 层说到底还是一个特征选择，信息过滤的过程，也就是说损失了一部分信息，
这是一个和计算性能的一个妥协，随着运算速度的不断提高，这个妥协会越来越小

特征提取的误差主要来自两个方面：

1. 邻域大小受限
2. 卷积层权值参数误差

目前主流观点认为对于 average pooling 和 max pooling 的主要区别在于：

* average pooling 能减小第一种误差，更多的保留图像的背景信息
* max pooling 能减小第二种误差，更多的保留纹理信息

## 激活函数

各种激活函数层出不穷，各有优点，比如下面这些：

![img](images/active_func.jpg)

每个激活函数都要考虑输入、输出以及数据变化，所以谨慎选择：

* Sigmoid 只会输出正数，以及靠近 0 的输出变化率最大
* tanh 和 sigmoid 不同的是，tanh 输出可以是负数
* ReLU 是输入只能大于 0，如果输入含有负数，ReLU 就不适合；
  如果输入是图片格式，ReLU 就挺常用的

激活函数对于提高模型鲁棒性、增加非线性表达能力、缓解梯度消失问题，
将特征图映射到新的特征空间从而更有利于训练、加速模型收敛等问题都有很好的帮助

## 全连接层

> 全连接层(Fully Connected Layer)

### 原理


全连接层之前的作用是提取特征，全连接层的作用是分类。
全连接层中的每一层是由许多神经元组成的的平铺结构

![img](images/cnn.jpg)

![img](images/cnn_to_full.png)

很简单,可以理解为在中间做了一个卷积

![img](images/cnn2full.jpg)

从上图我们可以看出，用一个 3x3x5 的 filter 去卷积激活函数的输出，
得到的结果就是一个 fully connected layer 的一个神经元的输出，
这个输出就是一个值。因为有 4096 个神经元，
实际就是用一个 3x3x5x4096 的卷积层去卷积激活函数的输出

这一步卷积一个非常重要的作用，就是把分布式特征 representation 映射到样本标记空间。
即它把特征 representation 整合到一起，输出为一个值，
这样做的一个好处就是大大减少特征位置对分类带来的影响

![img](images/final.jpg)

还有，发现有些全连接层有两层 1x4096 fully connected layer 平铺结构(有些网络结构有一层的，
或者二层以上的)，但是大部分是两层以上，根据泰勒公式，就是用多项式函数去拟合光滑函数。
这里的全连接层中一层的一个神经元就可以看成一个多项式，
用许多神经元去拟合数据分布，但是只用一层 fully connected layer 有时候没法解决非线性问题，
而如果有两层或以上 fully connected layer 就可以很好地解决非线性问题了

全连接层参数特多(可占整个网络参数 80% 左右)，
近期一些性能优异的网络模型如 ResNet 和 GoogLeNet 等均用全局平均池化(global average pooling，GAP)取代全连接层来融合学到的深度特征。
需要指出的是，用 GAP 替代全连接层的网络通常有较好的预测性能

### 示例

![img](images/cat.jpg)

从上图可以看出，猫在不同的位置，输出的特征值相同，但是位置不同。
对于电脑来说，特征值相同，但是特征值位置不同，那分类结果也可能不一样。
而这时全连接层 filter 的作用就相当于

猫在哪我不管，我只要猫，于是我让 filter 去把这个猫找到。
实际就是把 feature map 整合成一个值，这个值大，有猫；
这个值小，那就可能没猫，和这个猫在哪关系不大了有没有，
鲁棒性有大大增强了有没有。因为空间结构特性被忽略了，
所以全连接层不适合用于在方位上找 Pattern 的任务，比如 segmentation

### 对模型的影响

全连接层对模型影响的参数有三个：

1. 全连接层的总层数(长度)
2. 单个全连接层的神经元数(宽度)
3. 激活函数

如果全连接层宽度不变，增加长度：

* 优点：神经元个数增加，模型复杂度提升；
  全连接层数加深，模型非线性表达能力提高。
  理论上都可以提高模型的学习能力

如果全连接层长度不变，增加宽度：

* 优点：神经元个数增加，模型复杂度提升。理论上可以提高模型的学习能力

难度长度和宽度都是越多越好？肯定不是

* 缺点：
    - 学习能力太好容易造成过拟合
    - 运算时间增加，效率变低

那么怎么判断模型学习能力如何？

看 Training Curve 以及 Validation Curve，在其他条件理想的情况下，
如果 Training Accuracy 高，Validation Accuracy 低，也就是过拟合了，
可以尝试去减少层数或者参数。如果 Training Accuracy 低，说明模型学的不好，
可以尝试增加参数或者层数。至于是增加长度和宽度，这个又要根据实际情况来考虑了。
很多时候设计一个网络模型，不光考虑准确率，
也常常得在 Accuracy/Efficiency 里寻找一个好的平衡点

## 图像在 CNN 网络中的变化

> CNN 前向传播 forward propagation

图像数据经过卷积层发生了什么变化，经过采样层发生了什么变化，
经过激活层发生了什么变化，相当于实现了前向传播

```python
import numpy as np
import cv2

import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Activation, Flatten
from tensorflow.keras.layers import BatchNormalization, Dropout, UpSampling2D
from tensorflow.keras.layers import Input, add
from tensorflow.keras.models import Model, Sequential, load_model

print(f"tensorflow version: {tf.__version__}")
```

### 数据

数据读取及查看：

```python
girl = cv2.imread("girl.jpg")
print(f"girl shape: {girl.shape}")

while True:
    cv2.imshow("image", girl)
    if cv2.waitKey(1) & 0xFF == 27:
        break
cv2.destoryAllWindows()
```

![img](images/girl.jpg)

数据增维：

```python
# 由于 keras 只能按批处理数据，因此需要把单个数据提高一个维度
# (575, 607, 3) -> (1, 575, 607, 3)
girl_batch = np.expand_dims(girl, axis = 0)
print(f"girl_batch shape: {girl_batch.shape}")
```

图像数据可视化函数：

```python
def visualize(img, filter_, kernel_width, kernel_height, name):
    # 数据降维
    img = np.squeeze(img, axis = 0)
    # import pdb
    # pdb.set_trace()
    max_img = np.max(img)
    min_img = np.min(img)
    img = img - min_img
    img = img / (max_img - min_img)
    img = img * 255
    # img = img.reshape(img.shape[:2])
    cv2.imwrite(
        f"{name}_filter{str(filter_)}_\
        {str(kernel_width)}x{kernel_height}.jpg", 
        img
    )
```

### 模型

```python
kernel_width = 3
kernel_height = 3
filter_ = 3

model = Sequential()
model.add(Conv2D(filter_, kernel_width, kernel_height, input_shape = girl.shape, name = "conv_1"))  # 卷积层，filter 数量为 3，卷积核 size 为 (3,3)
model.add(MaxPooling2D(pool_size = (3, 3)))  # pooling 层，size 为 (3, 3)
model.add(Activation("relu"))  # 激活函数, 只保留大于 0 的值

model.add(Conv2D(filter_, kernel_width, kernel_height, input_shape = girl.shape, name = "conv_2"))
model.add(MaxPooling2D(pool_size = (2, 2)))
model.add(Activation("relu"))

model.add(Conv2D(filter_, kernel_width, kernel_height, input_shape = girl.shape, name = "conv_3"))
model.add(Activation("relu"))

model.add(Conv2D(filter_, kernel_width, kernel_height, input_shape = girl.shape, name = "conv_4"))
model.add(Activation("relu"))

model.add(Flatten())  # 把上层输出平铺
model.add(Dense(8, activation = "relu", name = "dens_1"))  # 加入全连接层，分为 8 类

# 因为 `Conv2d` 这个函数对于权值是随机初始化的，
# 每运行一次程序权值就变了，权值变了就没有比较意义了，
# 而我们不用 pretrained model，所以我们要保存第一次初始化的权值
model.save_weights("girl.h5")
```

### 一个卷积层

```python
model = Sequential()
model.add(Conv2D(3, 3, 3, input_shape = girl.shape, name = "conv_1"))

model.load_weights("girl.h5", by_name = True)
```

`by_name` 表示只给和保存模型相同卷积层名字的卷积层导入权值，
这里就是把上一步 `conv_1` 的权值导入这一步 `conv_1`，当然结构得相同

查看卷积层的输出-特征图：

```python
# 前向传播
conv_girl = model.predict(girl_batch)
girl_img = np.squeeze(conv_girl, axis = 0)
visualize(
    girl_img, 
    filter_ = 3, 
    kernel_width = 3, 
    kernel_height = 3, 
    name = "conv_girl"
)
```

![img](images/conv.png)

可以看到图像的一些纹理，边缘，或者颜色信息被一定程度上提取出来了，shape 也发生了变化

### 一个池化层

```python
model = Sequential()
model.add(Conv2D(3, 3, 3, input_shape = girl.shape, name = "conv_1"))
model.add(MaxPooling2D(pool_size = (2, 2)))

model.load_weights("girl.h5", by_name = True)
```

查看卷积层的输出-特征图：

```python
# 前向传播
conv_pooling_girl = model.predict(girl_batch)
girl_img = np.squeeze(conv_pooling_girl, axis = 0)
visualize(
    girl_img, 
    filter_ = 3, 
    kernel_width = 3, 
    kernel_height = 3, 
    name = "conv_pooling_girl"
)
```

![img](images/conv_pooling.png)

从上图可以明显的看到特征更加明显，并且 shape 减为三分之一了

### 激活函数

```python
model = Sequential()
model.add(Conv2D(3, 3, 3, input_shape = girl.shape, name = 'conv_1'))
model.add(MaxPooling2D(pool_size = (3, 3)))
model.add(Activation('relu'))  # 只保留大于 0 的值

model.load_weights("girl.h5", by_name = True)
```

查看结果：

```python
# 前向传播
conv_pooling_girl = model.predict(girl_batch)
girl_img = np.squeeze(conv_pooling_girl, axis = 0)
visualize(
    girl_img, 
    filter_ = 3, 
    kernel_width = 3, 
    kernel_height = 3, 
    name = "conv_pooling_relu_girl"
)
```

可以看到只有一些边缘的特征被保留下来了

### 新卷积层

```python
model = Sequential()
model.add(Conv2D(3, 3, 3, input_shape = girl.shape, name = "conv_1"))  # 卷积层，filter 数量为 3，卷积核 size 为 (3,3)
model.add(MaxPooling2D(pool_size = (3, 3)))  # pooling 层，size 为 (3, 3)
model.add(Activation("relu"))  # 激活函数, 只保留大于 0 的值

model.add(Conv2D(3, 3, 3, input_shape = girl.shape, name = "conv_2"))
model.load_weights("girl.h5", by_name = True)
```

查看结果：

```python
# 前向传播
conv_pooling_girl = model.predict(girl_batch)
girl_img = np.squeeze(conv_pooling_girl, axis = 0)
visualize(
    girl_img, 
    filter_ = 3, 
    kernel_width = 3, 
    kernel_height = 3, 
    name = "conv_pooling_relu_conv_girl"
)
```

纹理的信息更明显了

### 新采样层

```python
model = Sequential()
model.add(Conv2D(3, 3, 3, input_shape = girl.shape, name = "conv_1"))  # 卷积层，filter 数量为 3，卷积核 size 为 (3,3)
model.add(MaxPooling2D(pool_size = (3, 3)))  # pooling 层，size 为 (3, 3)
model.add(Activation("relu"))  # 激活函数, 只保留大于 0 的值

model.add(Conv2D(3, 3, 3, input_shape = girl.shape, name = "conv_2"))
model.add(MaxPooling2D(pool_size = (2, 2)))

model.load_weights("girl.h5", by_name = True)
```

查看结果：

```python
# 前向传播
conv_pooling_girl = model.predict(girl_batch)
girl_img = np.squeeze(conv_pooling_girl, axis = 0)
visualize(
    girl_img, 
    filter_ = 3, 
    kernel_width = 3, 
    kernel_height = 3, 
    name = "conv_pooling_relu_conv_pooling_girl"
)
```

### 新激活函数

```python
model = Sequential()
model.add(Conv2D(3, 3, 3, input_shape = girl.shape, name = "conv_1"))  # 卷积层，filter 数量为 3，卷积核 size 为 (3,3)
model.add(MaxPooling2D(pool_size = (3, 3)))  # pooling 层，size 为 (3, 3)
model.add(Activation("relu"))  # 激活函数, 只保留大于 0 的值

model.add(Conv2D(3, 3, 3, input_shape = girl.shape, name = "conv_2"))
model.add(MaxPooling2D(pool_size = (2, 2)))
model.add(Activation("relu"))

model.load_weights("girl.h5", by_name = True)
```

查看结果：

```python
# 前向传播
conv_pooling_girl = model.predict(girl_batch)
girl_img = np.squeeze(conv_pooling_girl, axis = 0)
visualize(
    girl_img, 
    filter_ = 3, 
    kernel_width = 3, 
    kernel_height = 3, 
    name = "conv_pooling_relu_conv_pooling_relu_girl"
)
```

### 将激活函数 relu 修改为 sigmoid

```python
model = Sequential()
model.add(Conv2D(3, 3, 3, input_shape = girl.shape, name = 'conv_1'))
model.add(MaxPooling2D(pool_size = (3, 3)))
model.add(Activation('sigmoid'))

model.add(Conv2D(3, 3, 3, input_shape = girl.shape, name = 'conv_2'))
model.add(MaxPooling2D(pool_size = (2, 2)))
model.add(Activation('sigmoid'))

model.load_weights("girl.h5", by_name = True)
```

查看结果：

```python
# 前向传播
conv_pooling_girl = model.predict(girl_batch)
girl_img = np.squeeze(conv_pooling_girl, axis = 0)
visualize(
    girl_img, 
    filter_ = 3, 
    kernel_width = 3, 
    kernel_height = 3, 
    name = "conv_pooling_sigmoid_conv_pooling_sigmoid_girl"
)
```

### 将激活函数 relu 修改为 tanh

```python
model = Sequential()
model.add(Conv2D(3, 3, 3, input_shape = girl.shape, name = 'conv_1'))
model.add(MaxPooling2D(pool_size = (3, 3)))
model.add(Activation('tanh'))

model.add(Conv2D(3, 3, 3, input_shape = girl.shape, name = 'conv_2'))
model.add(MaxPooling2D(pool_size = (2, 2)))
model.add(Activation('tanh'))

model.load_weights("girl.h5", by_name = True)
```

查看结果：

```python
# 前向传播
conv_pooling_girl = model.predict(girl_batch)
girl_img = np.squeeze(conv_pooling_girl, axis = 0)
visualize(
    girl_img, 
    filter_ = 3, 
    kernel_width = 3, 
    kernel_height = 3, 
    name = "conv_pooling_tanh_conv_pooling_tanh_girl"
)
```

### 新增两个卷积层和激活函数

```python
model = Sequential()
model.add(Conv2D(3, 3, 3, input_shape = girl.shape, name = "conv_1"))  # 卷积层，filter 数量为 3，卷积核 size 为 (3,3)
model.add(MaxPooling2D(pool_size = (3, 3)))  # pooling 层，size 为 (3, 3)
model.add(Activation("relu"))  # 激活函数, 只保留大于 0 的值

model.add(Conv2D(3, 3, 3, input_shape = girl.shape, name = "conv_2"))
model.add(MaxPooling2D(pool_size = (2, 2)))
model.add(Activation("relu"))

model.add(Conv2D(3, 3, 3, input_shape = girl.shape, name = "conv_3"))
model.add(Activation("relu"))

model.add(Conv2D(3, 3, 3, input_shape = girl.shape, name = "conv_4"))
model.add(Activation("relu"))

model.load_weights("girl.h5", by_name = True)
```

查看结果：

```python
# 前向传播
conv_pooling_girl = model.predict(girl_batch)
girl_img = np.squeeze(conv_pooling_girl, axis = 0)
visualize(
    girl_img, 
    filter_ = 3, 
    kernel_width = 3, 
    kernel_height = 3, 
    name = "conv_pooling_tanh_conv_pooling_tanh_conv2_conv2_girl"
)
```

### 全连接层

```python
model = Sequential()
model.add(Conv2D(3, 3, 3, input_shape = girl.shape, name = "conv_1"))  # 卷积层，filter 数量为 3，卷积核 size 为 (3,3)
model.add(MaxPooling2D(pool_size = (3, 3)))  # pooling 层，size 为 (3, 3)
model.add(Activation("relu"))  # 激活函数, 只保留大于 0 的值

model.add(Conv2D(3, 3, 3, input_shape = girl.shape, name = "conv_2"))
model.add(MaxPooling2D(pool_size = (2, 2)))
model.add(Activation("relu"))

model.add(Conv2D(3, 3, 3, input_shape = girl.shape, name = "conv_3"))
model.add(Activation("relu"))

model.add(Conv2D(3, 3, 3, input_shape = girl.shape, name = "conv_4"))
model.add(Activation("relu"))

model.add(Flatten())  # 把上层输出平铺
model.add(Dense(8, activation = "relu", name = "dens_1"))  # 加入全连接层，分为 8 类

model.load_weights("girl.h5", by_name = True)
```

查看结果：

```python
# 前向传播
conv_pooling_girl = model.predict(girl_batch)
girl_img = np.squeeze(conv_pooling_girl, axis = 0)
visualize(
    girl_img, 
    filter_ = 3, 
    kernel_width = 3, 
    kernel_height = 3, 
    name = "conv_pooling_tanh_conv_pooling_tanh_conv2_conv2_girl"
)
```

### 不同 kernel size 对比

```python
model = Sequential()
model.add(Conv2D(
    3, 3, 3, 
    kernel_initializer = keras.initializers.Constant(value = 0.12), input_shpae = girl.shape, 
    name = "conv_333"
))
```

查看卷积层的输出-特征图：

```python
# 前向传播
conv_girl = model.predict(girl_batch)
girl_img = np.squeeze(conv_girl, axis = 0)
visualize(
    girl_img, 
    filter_ = 3, 
    kernel_width = 3, 
    kernel_height = 3, 
    name = "conv_333"
)
```

```python
model = Sequential()
model.add(Conv2D(
    3, 2, 2, 
    kernel_initializer = keras.initializers.Constant(value = 0.12), input_shpae = girl.shape, 
    name = "conv_322"
))
```

查看卷积层的输出-特征图：

```python
# 前向传播
conv_girl = model.predict(girl_batch)
girl_img = np.squeeze(conv_girl, axis = 0)
visualize(
    girl_img, 
    filter_ = 3, 
    kernel_width = 2, 
    kernel_height = 2, 
    name = "conv_322"
)
```

```python
model = Sequential()
model.add(Conv2D(
    3, 24, 24, 
    kernel_initializer = keras.initializers.Constant(value = 0.12), input_shpae = girl.shape, 
    name = "conv_32424"
))
```

查看卷积层的输出-特征图：

```python
# 前向传播
conv_girl = model.predict(girl_batch)
girl_img = np.squeeze(conv_girl, axis = 0)
visualize(
    girl_img, 
    filter_ = 3, 
    kernel_width = 24, 
    kernel_height = 24, 
    name = "conv_32424"
)
```

# 参考

* [CNN卷积方法一览](https://mp.weixin.qq.com/s/9RvkFMOxmUTdZGDqjZfELw)
* [如何理解卷积神经网络的结构](https://zhuanlan.zhihu.com/p/31249821)
* [什么是卷积](https://zhuanlan.zhihu.com/p/30994790)
* [图像卷积与滤波的一些知识点](https://blog.csdn.net/zouxy09/article/details/49080029)
* [卷积层是如何提取特征的](https://zhuanlan.zhihu.com/p/31657315)
* [什么是采样层](https://zhuanlan.zhihu.com/p/32299939)
* [什么是激活函数](https://zhuanlan.zhihu.com/p/32824193)
* [什么是全连接层](https://zhuanlan.zhihu.com/p/33841176)
* [图片在卷积神经网络中是怎么变化的](https://zhuanlan.zhihu.com/p/34222451)

