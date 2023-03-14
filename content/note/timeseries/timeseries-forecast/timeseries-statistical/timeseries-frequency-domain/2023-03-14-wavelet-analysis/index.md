---
title: 小波分析
author: 王哲峰
date: '2023-03-14'
slug: wavelet-analysis
categories:
  - timeseries
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

- [小波变换的维基百科介绍](#小波变换的维基百科介绍)
  - [小波分析介绍](#小波分析介绍)
  - [小波的定义](#小波的定义)
  - [小波转换](#小波转换)
    - [小波转换的优点](#小波转换的优点)
    - [小波转换的缺点](#小波转换的缺点)
    - [小波转换和波的比较](#小波转换和波的比较)
    - [小波转换，傅立叶转换，加伯转换的比较](#小波转换傅立叶转换加伯转换的比较)
  - [小波转换的分类](#小波转换的分类)
- [小波变换教程](#小波变换教程)
  - [变换什么](#变换什么)
- [PyWavelets 库](#pywavelets-库)
  - [安装和使用](#安装和使用)
  - [主要功能](#主要功能)
- [参考](#参考)
</p></details><p></p>

# 小波变换的维基百科介绍

## 小波分析介绍

小波分析（wavelet analysis）或小波变换（英语：wavelet transform）是指用有限长或快速衰减的“母小波”（mother wavelet）的振荡波形来表示信号。该波形被缩放和平移以匹配输入的信号。

“小波”（英语：wavelet）一词由吉恩·莫莱特和阿列克斯·格罗斯曼在 1980 年代早期提出。他们用的是法语词 ondelette，意思就是“小波”。后来在英语里，“onde” 被改为 “wave” 而成了 wavelet。

小波变化的发展，承袭 Gabor transform 的局部化思想，并且克服了傅里叶和 Gabor transform 的部分缺陷，小波变换提供了一个可以调变的时频窗口，窗口的宽度 (width) 随着频率变化，频率增高时，时间窗口的宽度就会变窄，以提高分辨率．小波在整个时间范围内的振幅平均值为0，具有有限的持续时间和突变的频率与震幅，可以是不规则，或不对称的信号。

小波变换分成两个大类：离散小波变换 (DWT)  和连续小波变换 (CWT)。两者的主要区别在于，连续变换在所有可能的缩放和平移上操作，而离散变换采用所有缩放和平移值的特定子集。

## 小波的定义

wavelet 是指小型波(在傅立叶分析里的弦波是大型波)，简单说来，小波(wavelet)是一个衰减迅速的振荡。

有几种定义小波（或者小波族）的方法:

* 缩放滤波器
    - 小波完全通过缩放滤波器 `$g$`——一个低通有限脉冲响应（FIR）长度为 `$2N$` 和为 `$1$` 的滤波器——来定义。
      在双正交小波的情况，分解和重建的滤波器分别定义。
    - 高通滤波器的分析作为低通的 QMF 来计算，而重建滤波器为分解的时间反转。例如 Daubechies 和 Symlet 小波。
* 缩放函数
    - 小波由时域中的小波函数 `$\psi(t)$`(即母小波)和缩放函数 `$\phi(t)$`(也称为父小波)来定义。
    - 小波函数实际上是带通滤波器，每一级缩放将带宽减半。这产生了一个问题，如果要覆盖整个谱需要无穷多的级。
      缩放函数滤掉变换的最低级并保证整个谱被覆盖到。
    - 对于有紧支撑的小波，`$\phi(t)$` 可以视为有限长，并等价于缩放滤波器 `$g$`。例如 Meyer 小波。
* 小波函数
    - 小波只有时域表示，作为小波函数 `$\psi(t)$`。例如墨西哥帽小波

## 小波转换

如果函数 `$f \in L_{2}(R)$`，那么级数 

`$$\sum_{j \in Z}\sum_{k\in Z}\langle f, \psi_{j,k}\rangle \psi_{j,k}(t)$$` 

称作 `$f$` 的小波级数，且

`$$\langle f, \psi_{j,k}\rangle = \int_{-\infty}^{\infty}f(t)\psi_{j,k}(t)dt$$`

为 `$f$` 的小波系数

存在着大量的小波变换，每个适合不同的应用。完整的列表参看小波相关的变换列表，常见的如下：

* 连续小波变换（CWT）
* 离散小波变换（DWT）
* 快速小波转换（FWT）
* 小波包分解（Wavelet packet decomposition）（WPD）

### 小波转换的优点

* 可以同时观察频率和时间轴，在频率高时有较好的时间解析度，在频率低时有较好的频率解析度。
* 有快速小波转换可以加速运算。
* 可以分离出信号的精细或粗糙成分。
* 在小波理论中，可以用较少的小波系数去逼近一个函数。
* 对讯号去噪或压缩讯号时，不会对讯号造成明显的破坏。
* 适用于分析突变讯号，以及奇异讯号
* 可以分析讯号不同 scale 大小样貌

### 小波转换的缺点

* 运算量大，比较难做到即时处理
* 母小波挑选的限制

### 小波转换和波的比较

* 小波的大小对比波的频率
* 小波的 duration( window size) 对比波的 infinite duration
* 小波的 temporal localization 对比波的 no temporal localization

### 小波转换，傅立叶转换，加伯转换的比较

傅立叶转换具有局部性，加伯转换没有具有局部性

小波转换具有局部性，并且可以改变参数来调整频谱的窗口和结构形状，进而做到"变焦"的作用.

因此小波分析可以达到多解析度的效果

## 小波转换的分类

小波转换以输入、输出的连续或离散性质来区分，有三种:

|     | 输入 | 输出	| 小波转换名称 |
|-----|-----|------|------------|
| 第一种 | 连续函数	| 连续函数	| 连续小波变换（Continuous Wavelet Transform）
| 第二种 | 连续函数	| 离散函数	| 离散系数连续小波变换（Continuous Wavelet Transform with Discrete Coefficients)，有时候被称为discrete wavelet transform |
| 第三种 | 离散函数 | 离散函数 | 离散小波变换（Discrete Wavelet Transform）|

傅立叶转换(Fourier Transform)与小波转换比较共有四种类型：

|       | 输入    | 输出     | 傅立叶转换名称 |
|-------|--------|----------|-------------|
| 第一种 | 连续函数 | 连续函数 | 傅立叶转换(Fourier Transform) |
| 第二种 | 连续函数 | 离散函数 | 傅立叶级数(Fourier Series) |
| 第三种 | 离散函数 | 离散函数 | 离散傅立叶转换(Discrete Fourier Transform) |
| 第四种 | 离散函数 | 连续函数 | 离散时间傅立叶转换(Discrete-time Fourier Transform) |

两相比较我们可以看出，小波转换并没有输入为离散函数、输出为连续函数的类型（傅立叶转换表格的第四种），
原因在于该种类型并不实用

# 小波变换教程

## 变换什么

变换是将数学变换应用于信号，以从该信号中获取原始信号中不易获得的更多信息

假设一个时域信号为原始信号，一个已通过任何可用数学变换“变换”的信号为处理后的信号。
可以应用的变换有很多，其中傅立叶变换可能是迄今为止最受欢迎的

实践中的大多数信号的原始格式都是时域的。也就是说，无论该信号正在测量什么，都是时间的函数。
换句话说，当我们绘制信号时，其中一个轴是时间（自变量），另一个轴（因变量）通常是振幅。
当我们绘制时域信号时，我们获得了信号的时间幅度表示。对于大多数与信号处理相关的应用，
此表示并不总是信号的最佳表示。在许多情况下，最明显的信息隐藏在信号的频率内容中。
信号的频谱基本上是该信号的频率分量（频谱分量）。信号的频谱显示了信号中存在的频率

凭直觉，我们都知道频率与某种事物的速率变化有关。如果某物（一个数学或物理变量，在技术上是正确的术语）变化很快，
我们说它是高频的，而如果这个变量没有快速变化，即它变化平稳，我们说它是低频。如果这个变量根本没有变化，
那么我们说它的频率为零，或者没有频率。例如，日报的出版频率高于月刊（出版频率更高）









# PyWavelets 库

> Wavelet Transform in Python

## 安装和使用

```bash
$ pip install PyWavelets
```

使用简单：

```python
import pywt

cA, cD = pywt.dwt([1, 2, 3, 4], "db1")
```

## 主要功能

* 1D, 2D and nD Forward and Inverse Discrete Wavelet Transform (DWT and IDWT)
* 1D, 2D and nD Multilevel DWT and IDWT
* 1D, 2D and nD Stationary Wavelet Transform (Undecimated Wavelet Transform)
* 1D and 2D Wavelet Packet decomposition and reconstruction
* 1D Continuous Wavelet Transform
* Computing Approximations of wavelet and scaling functions
* Over 100 built-in wavelet filters and support for custom wavelets
* Single and double precision calculations
* Real and complex calculations
* Results compatible with Matlab Wavelet Toolbox (TM)


# 参考

* [维基百科](https://zh.wikipedia.org/wiki/%E5%B0%8F%E6%B3%A2%E5%88%86%E6%9E%90)
* [The Wavelet Tutorial](https://users.rowan.edu/~polikar/WTtutorial.html)
* [PyWavelets](https://pywavelets.readthedocs.io/en/latest/index.html)

