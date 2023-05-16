---
title: 时间序列预测-深度学习
author: 王哲峰
date: '2022-04-25'
slug: timeseries-forecast-dl
categories:
  - timeseries
tags:
  - machinelearning
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

- [循环神经网络 RNN](#循环神经网络-rnn)
  - [LSTM](#lstm)
  - [GRU](#gru)
  - [DeepAR](#deepar)
  - [LSTNet](#lstnet)
    - [网络结构](#网络结构)
    - [网络参数](#网络参数)
  - [Seq2Seq](#seq2seq)
  - [WaveNet](#wavenet)
  - [TFT](#tft)
- [卷积神经网络 CNN](#卷积神经网络-cnn)
- [格拉姆角场 GAF](#格拉姆角场-gaf)
- [短时傅里叶变换 STFT](#短时傅里叶变换-stft)
- [时间卷积网络 TCN](#时间卷积网络-tcn)
- [基于注意力机制的模型 Attention](#基于注意力机制的模型-attention)
- [CNN 和 RNN 和 Attention](#cnn-和-rnn-和-attention)
- [Transformer](#transformer)
  - [N-Beats](#n-beats)
- [参考](#参考)
</p></details><p></p>

深度学习方法近年来逐渐替代机器学习方法，成为人工智能与数据分析的主流，
对于时间序列的分析，有许多方法可以进行处理。
常见的利用神经网络技术来做时间序列预测的方法有 CNN、RNN、LSTM、GRU、Seq2Seq 等

相对于传统的树模型需要人工构建相关模型特征, 神经网络模型通常需要喂入大量的数据来进行训练, 
因此如果同类时序的数据量够多(有够多彼此间相关性较强的时序), 
那么训练一个通用型的端对端的神经网络预测有时也有不错的效果, 比如使用 LSTM 一方面可以较容易地整合外部变量, 
另一方面 LSTM 有能较好地自动提取时序特征的能力

深度学习的思路是尽量只使用原始的序列和其它相关输入信息，基本不做特征工程，
希望通过各类模型结构自动学习到时序的隐含表达，进而做端到端的预测输出

# 循环神经网络 RNN

循环神经网络(RNN)框架及其变种(LSTM/GRU/...)是为处理序列型而生的模型，
天生的循环自回归的结构是对时间序列的很好的表示。所采用的方式也是监督学习，
不过不需要人为的构建时序特征，可以通过深度学习网络拟合时序曲线，
捕捉时间先后顺序关系，长期依赖，进行特征学习与预测

## LSTM


## GRU

## DeepAR

亚马逊提出的一个网络架构，也是基于 Seq2Seq，不过 DeepAR 的输出是一个概率分布，
这也是它与传统 RNN 系列最大的一个区别

![img](images/deepar.webp)

## LSTNet

Long-and Short-term Time-series Network(LSTNet) 专门设计用于时间序列预测的深度学习网络，
该网络特点有如下5点：

* CNN：一维普通卷积(没有使用因果扩张卷积)，用于捕捉短期局部信息
* RNN：使用 LSTM 或 GRU，捕捉长期宏观信息
* Skip-RNN：对输入数据维度整理，使用 LSTM 或 GRU，
  捕捉更长期的信息并充分利用序列的周期特性，但此时的周期 P 是引入的另一个超参数
* Attention：论文中有提到使用 Attention 机制，灵活调整周期 P，但在代码中没见到
* HighWay：使用 Dense 模拟 AR 自回归过程，为预测添加线性成份，同时使输出可以响应输入的尺度变化

### 网络结构

![img](images/lstnet.png)

预测包含两部分，最终结果是非线性预测和线性预测的结合：

* 非线性层：CNN 层、RNN 层、Skip-RNN 层
* 线性层：AR 层

### 网络参数

## Seq2Seq

![img](images/seq2seq.jpeg)

基本是借鉴了 NLP 里的经典架构，使用 RNN/GRU/LSTM 作为基本单元，
encoder 中做训练窗口中的信息提取，然后在 decoder 中做预测 horizon 的多步输出

## WaveNet

NN 系列模型不好并行，所以突然发现谷歌提出的这个空洞因果卷积感觉很高级，
性能理论上也比 RNN 之类的好很多，它的结构大致长这样

![img](images/wavenet.jpeg)

## TFT

可以与树模型匹敌的深度学习模型了！这就是 Google AI 提出的 Temporal Fusion Transformers。
也是本文提到的第一个带 transformer 结构的模型

# 卷积神经网络 CNN

传统的卷积神经网络(CNN)一般认为不太适合时序问题的建模，这主要由于其卷积核大小的限制，
不能很好的抓取长时的依赖信息。但是最近也有很多的工作显示，
特定的卷积神经网络结构也可以达到很好的效果，通常将时间序列转化为图像，再应用基于卷积神经网络的模型做分析

# 格拉姆角场 GAF

格拉姆角场，Gramian Angular Field(GAF) 的核心思想是：将笛卡尔坐标系下的一维时间序列，
转化为极坐标系表示，再使用三角函数生成 GAF 矩阵。计算过程如下：

* 数值缩放：将笛卡尔坐标系下的时间序列缩放到 `$[0,1]$` 或 `$[-1,1]$` 区间
* 极坐标转换：使用坐标变换公式，将笛卡尔坐标系序列转化为极坐标系时间序列
* 角度和/差的三角函数变换：若使用两角和的 cos 函数则得到 GASF，若使用两角差的 cos 函数则得到 GADF

# 短时傅里叶变换 STFT

短时傅里叶变换，Short Time Fourier Transform(STFT) 在语音信号处理场景使用很广泛，
其目标主要将时间序列转为时频图像，进而采用卷积网络进行特征分析

# 时间卷积网络 TCN

时间卷积网络，Time Convolution Network(TCN) 是一种特殊的卷积神经网络，针对一维空间做卷积，迭代多层捕捉长期关系。
具体的，对于上一层 `$t$` 时刻的值，只依赖于下一层 `$t$` 时刻及其之前的值。
和传统的卷积神经网络的不同之处在于，TCN 不能看到未来的数据，它是单向的结构，不是双向的。
也就是说只有有了前面的因才有后面的果，是一种严格的时间约束模型，因此又被称为因果卷积

# 基于注意力机制的模型 Attention

在 RNN 中分析时间序列需要一步步按顺序处理从 `$t-n$` 到 `$t$` 的所有信息，
而当它们相距较远(`$n$` 非常大)时 RNN 的效果常常较差，且由于其顺序性处理效率也较低。
基于注意力机制(Attention)的模型，采用跳步的方式计算每个数值之间的两两关联，
然后组合这些关联分数得到一个加权的表示。该表示通过前馈神经网络的学习，可以更好的考虑到时序的上下文的信息

# CNN 和 RNN 和 Attention

结合 CNN + RNN + Attention，作用各不相同互相配合，主要设计思想：

* CNN 捕捉短期局部依赖关系
* RNN 捕捉长期宏观依赖关系
* Attention 为重要时间段或变量加权


# Transformer


## N-Beats

出自 Element AI，Bengio 是其中的 Co-Founder。第一次见到它是来自 M5 比赛亚军的分享，
不过他也只是在 top-level 的预测中使用了一下 N-Beats 模型

![img](images/n-beats.webp)

N-Beats 专注于做单变量的时序预测，且可以具有一定的 seasonality，trend 的可解释性，
跟 Prophet 很相似。从论文的实验来看，作者使用了非常重的 ensemble，
每个序列搞了 180 个模型的 bagging。这感觉有点过于“杀鸡用牛刀”了……我们实测下来也没有取得很好的效果，
而且看起来还不好加额外的特征变量，使用场景很受限

# 参考

* LSTNet
    - [blog](https://zhuanlan.zhihu.com/p/61795416)
    - [paper](https://arxiv.org/pdf/1703.07015.pdf)
    - [code](https://github.com/Lorne0/LSTNet_keras)
* TPA-LSTM
    - [blog](https://zhuanlan.zhihu.com/p/63134630)
    - [paper](https://arxiv.org/pdf/1809.04206v2.pdf)
    - [code](https://github.com/gantheory/TPA-LSTM)
* LSTM
   - [blog](https://cloud.tencent.com/developer/article/1041442)
* [时间序列预测](https://mp.weixin.qq.com/s?__biz=Mzg3NDUwNTM3MA==&mid=2247484974&idx=1&sn=d841c644fd9289ad5ec8c52a443463a5&chksm=cecef3dbf9b97acd8a9ededc069851afc00db422cb9be4d155cb2c2a9614b2ee2050dc7ab4d7&scene=21#wechat_redirect)
* [深度学习时间序列预测经典方案汇总](https://mp.weixin.qq.com/s?__biz=MzIyOTUyMDIwNg==&mid=2247484061&idx=1&sn=8c78cc2ebfcd8112580eb4f5863e2f94&chksm=e84024bcdf37adaa996ac40be1b3f594e6c653532738e7a062804830bb94bbb6e688f2ad1f6f&scene=21#wechat_redirect)
