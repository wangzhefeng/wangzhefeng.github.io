---
title: 小波分析与傅里叶变换
author: 王哲峰
date: '2022-03-11'
slug: wavelet-fourier-transform
categories:
  - 数学、统计学
tags:
  - note
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

# 傅里叶变换

傅里叶变换的维基百科介绍：

> **傅里叶变换**（法语：Transformation de Fourier、英语：Fourier transform）是一种线性积分变换，用于信号在[时域](https://zh.wikipedia.org/wiki/時域)（或空域）和[频域](https://zh.wikipedia.org/wiki/频域)之间的变换，在物理学和工程学中有许多应用。因其基本思想首先由法国学者约瑟夫·傅里叶系统地提出，所以以其名字来命名以示纪念。实际上傅里叶变换就像化学分析，确定物质的基本成分；信号来自自然界，也可对其进行分析，确定其基本成分。
>
> - 时域 (time domain) 是描述数学函数或物理信号对时间的关系。例如一个信号的时域波形可以表达信号随着时间的变化
> - 频域 (frequency domain) 在电子学、控制系统、统计学中是指在对函数或信号进行分析时，分析其和频率有关的部分，而不是和时间有关的部分


## 傅里叶变换的应用

- TODO



# 小波分析

小波分析的维基百科介绍：

> **小波分析**（英语：wavelet analysis）或**小波变换**（英语：wavelet transform）是指用有限长或快速衰减的“母小波”（mother wavelet）的振荡波形来表示信号。该波形被缩放和平移以匹配输入的信号。
>
> “小波”（英语：wavelet）一词由吉恩·莫莱特和阿列克斯·格罗斯曼在 1980 年代早期提出。他们用的是法语词 ondelette，意思就是“小波”。后来在英语里，“onde” 被改为 “wave” 而成了 wavelet。
>
> 小波变化的发展，承袭 Gabor transform 的局部化思想，并且克服了傅里叶和 Gabor transform 的部分缺陷，小波变换提供了一个可以调变的时频窗口，窗口的宽度 (width) 随着频率变化，频率增高时，时间窗口的宽度就会变窄，以提高分辨率．小波在整个时间范围内的振幅平均值为0，具有有限的持续时间和突变的频率与震幅，可以是不规则，或不对称的信号。
>
> 小波变换分成两个大类：离散小波变换 (DWT)  和连续小波变换 (CWT)。两者的主要区别在于，连续变换在所有可能的缩放和平移上操作，而离散变换采用所有缩放和平移值的特定子集。


## 小波分析的应用

- TODO




# 参考资料

- https://zh.wikipedia.org/wiki/%E5%82%85%E9%87%8C%E5%8F%B6%E5%8F%98%E6%8D%A2
- https://www.leiphone.com/category/yanxishe/HJWOsm2lCtCIpVVK.html
- https://www.jezzamon.com/fourier/zh-cn.html
- https://github.com/Jezzamonn/fourier
- https://betterexplained.com/articles/an-interactive-guide-to-the-fourier-transform/
- https://zh.wikipedia.org/wiki/%E5%B0%8F%E6%B3%A2%E5%88%86%E6%9E%90
- http://users.rowan.edu/~polikar/WTpart1.html
- http://users.rowan.edu/~polikar/WTpart2.html
- http://users.rowan.edu/~polikar/WTpart3.html
