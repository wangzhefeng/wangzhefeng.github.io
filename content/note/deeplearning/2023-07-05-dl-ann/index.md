---
title: 深度学习概述
# subtitle: 
author: wangzf
date: '2022-07-12'
slug: dl-ann
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

- [深度学习](#深度学习)
- [神经网络](#神经网络)
  - [人工神经元](#人工神经元)
  - [神经网络结构](#神经网络结构)
    - [前馈网络](#前馈网络)
    - [记忆网络](#记忆网络)
    - [图网络](#图网络)
- [神经网络](#神经网络-1)
  - [单层感知机](#单层感知机)
  - [多层感知机](#多层感知机)
  - [DNN](#dnn)
  - [FCFFNN](#fcffnn)
  - [CNN](#cnn)
  - [RNN](#rnn)
  - [深度生成模型](#深度生成模型)
    - [玻尔兹曼机](#玻尔兹曼机)
    - [深度信念网络](#深度信念网络)
    - [GAN](#gan)
  - [AutoEncoder](#autoencoder)
</p></details><p></p>

# 深度学习

**机器学习** 就是从历史数据中探索和训练出数据的普遍规律，将其归纳为相应的数学模型，
并对未知的数据进行预测的过程。在这个过程中会碰到各种各样的问题，
比如下面一系列关乎机器学习模型生死的问题：数据质量、模型评价标准、训练优化方法、过拟合。

在机器学习中，有很多已经相当成熟的模型，在这些机器学习模型中，
**人工神经网络** 就是一种比较厉害的模型；人工神经网络从早期的感知机发展而来，
对任何函数都有较好的拟合性。但自上个世纪 90 年代一直到 2012 年深度学习集中爆发前夕，
神经网络受制于计算资源的限制和较差的可解释性，一直处于发展的低谷阶段。之后大数据兴起，计算资源也迅速跟上，
加之 2012 年 ImageNet 竞赛冠军采用的 AlexNet 卷积神经网络一举将图片预测的 top5 错误率降至 16.4%，
震惊了当时的学界和业界。从此之后，原本处于研究边缘状态的神经网络又迅速热了起来，深度学习也逐渐占据了计算机视觉的主导地位。

**以神经网络为核心的深度学习理论**是机器学习的一个领域分支，
所以深度学习其本质上也必须遵循一些机器学习的基本要义和法则。
从机器学习的角度来开，**神经网络一般可以看作一个非线性模型，
其基本组成单元为具有非线性激活函数的神经元，通过大量神经元之间的连接，
使得神经网络成为一种高度非线性的模型。** 神经元之间的连接权重就是需要学习的参数，
可以在机器学习的框架下通过梯度下降方法来进行学习。

# 神经网络

## 人工神经元

人工神经元(Artificial Neuron)，简称神经元(Neuron)，是构成神经网络(Neural Network, NN)的基本单元，
接收一组输入信号并产生输出。

现代神经网络中的神经元和 MLP 神经元的结构并无太多变化。不同的是，
MLP 神经元中的激活函数 `$f$` 为 0 或 1 的阶跃函数，
而现代神经元中的激活函数通常要求是连续可导的函数。

假设一个神经元接收 `$D$` 个输入 `$x_{1}, x_{2}, \ldots, x_{D}$`，
令向量 `$\boldsymbol{x}=[x_{1}; x_{2}; \ldots; x_{D}]$` 来表示这组输入，
并用净输入(Net Input) `$\textbf{z} \in \mathbb{R}$` 表示一个神经元所获得的输入信号 `$x$` 的加权和

`$$\begin{aligned}
\textbf{z} &= \sum_{d=1}^{D}\omega_{d}x_{d} + b \\
           &=\boldsymbol{\omega}^{T}\boldsymbol{x}+b
\end{aligned}$$`

其中：

* `$\boldsymbol{\omega} = [\omega_{1}; \omega_{2}; \ldots; \omega_{D}] \in \mathbb{R}^{D}$` 是 `$D$` 维的权重向量
* `$b \in \mathbb{R}$` 是偏置

净输入 `$\textbf{z}$` 在经过一个非线性函数 `$f(\cdot)$` 后，得到神经元的活性值(Activation) 

`$$a = f(\textbf{z})$$`

其中非线性函数 `$f(\cdot)$` 称为激活函数(Activation Function)

![img](images/neuron.png)

## 神经网络结构

神经网络的结构指的是“神经元”之间如何连接，它可以是任意深度

![img](images/network.png)

### 前馈网络

> forward feedward Neural Network, FFNN

前馈网络中各个神经元按接收信息的先后分为不同的组。每一组可以看作一个神经层。
每一层中的神经元接收前一层神经元的输出，并输出到下一层神经元。
整个网络中的信息是朝一个方向传播，没有反向的信息传播，可以用一个有向无环路图表示。
前馈网络包括全连接前馈网络(FCFNN)和卷积神经网络(CNN)等。

前馈网络可以看作一个函数，通过简单非线性函数的多次复合，实现输入空间到输出空间的复杂映射。
这种网络结构简单，易于实现。

### 记忆网络

记忆网络，也称为反馈网络，网络中的神经元不但可以接收其他神经元的信息，也可以接收自己的历史信息。
和前馈网络相比，记忆网络中的神经元具有记忆功能，在不同的时刻具有不同的状态。
记忆神经网络中的信息传播可以是单向或双向传递，因此可用一个有向循环图或无向图来表示。
记忆网络包括循环神经网络(RNN)、Hopfield 网络、玻尔兹曼机、受限玻尔兹曼机等.

记忆网络可以看作一个程序，具有更强的计算和记忆能力。

为了增强记忆网络的记忆容量，可以引入外部记忆单元和读写机制，用来保存一些网络的中间状态，
称为记忆增强神经网络(Memory Augmented Neural Network，MANN)，比如神经图灵机和记忆网络。

### 图网络

前馈网络和记忆网络的输入都可以表示为向量或向量序列。但实际应用中很多数据是图结构的的数据，
比如知识图谱、社交网络、分子(Molecular)网络等。前馈网络和记忆网络很难处理图结构的数据。

图网络是定义在图结构数据上的神经网络。图中每个节点都由一个或一组神经元构成。节点之间的连接可以是有向的，
也可以是无向的。每个节点可以收到来自相邻节点或自身的信息。

图网络是前馈网络和记忆网络的泛化，包含很多不同的实现方式，
比如图卷积网络(Graph Convolutional Netword，GCN)、
图注意力网络(Graph Attention Network，GAT)、
消息传递神经网络(Message Passing Neural Network，MPNN)等。

# 神经网络

## 单层感知机

> * 感知机，Perceptron，
> * 单层感知机，Singel-layer Perceptron SLP

## 多层感知机

> * 人工神经网络，Artificial Neural Networks，ANN
> * 多层感知机，Multi-layer Perceptron，MLP

多层感知机(MLP) `$\approx$` 人工神经网络(ANN)。追根溯源的话，
(人工)神经网络(ANN)的基础模型是感知机(Perceptron)，
因此(人工)神经网络也可以叫做多层感知机，感知机叫做单层感知机(SLP)。

## DNN

> 深度神经网络，Deep Neural Networks，DNN

那么多层到底是几层？一般来说有 1-2 个隐藏层的神经网络就可以叫做多层，
准确的说是(浅层)神经网络(Shallow Neural Networks, SNN)。

随着隐藏层的增多，更深的神经网络(一般来说超过 5 层)就都叫做深度神经网络(DNN)、
深度学习(Deep Learning)。然而，“深度”只是一个商业概念，
很多时候工业界把 3 层隐藏层也叫做 “深度学习”，
所以不要在层数上太较真。在机器学习领域的约定俗成是，
名字中有深度(Deep)的网络仅代表其有超过 5-7 层的隐藏层。

需要特别指出的是，卷积网络(CNN)和循环网络(RNN)一般不加 Deep 在名字中的原因是：
它们的结构一般都较深，因此不需要特别指明深度。
想对比的，自编码器(Auto Encoder)可以是很浅的网络，也可以很深。
所以你会看到人们用 Deep Auto Encoder 来特别指明其深度

## FCFFNN

> 全连接的前馈深度神经网络，Fully Connected Feed Forward Neural Networks，FCFFNN

FCFFNN 也就是 DNN，适用于大部分分类(Classification)任务，比如数字识别等。
但一般的现实场景中我们很少有那么大的数据量来支持 DNN，所以纯粹的全连接网络应用性并不是很强。

## CNN

> 卷积神经网络，Convolutional Neural Network，CNN

卷积网络早已大名鼎鼎，从某种意义上也是为深度学习打下良好口碑的功臣。
不仅如此，卷积网络也是一个很好的计算机科学借鉴神经科学的例子。
卷积网络的精髓其实就是在多个空间位置上共享参数，据说我们的视觉系统也有相类似的模式。

首先简单说什么是卷积。卷积运算是一种数学计算，和矩阵相乘不同，卷积运算可以实现稀疏相乘和参数共享，
可以压缩输入端的维度。和普通 DNN 不同，CNN 并不需要为每一个神经元所对应的每一个输入数据提供单独的权重。
与池化(pooling)相结合，CNN 可以被理解为一种公共特征的提取过程，
不仅是 CNN 大部分神经网络都可以近似的认为大部分神经元都被用于特征提取。

应用场景：虽然我们一般都把 CNN 和图片联系在一起，
但事实上 CNN 可以处理大部分格状结构化数据(Grid-like Data)。
举个例子，图片的像素是二维的格状数据，时间序列在等时间上抽取相当于一维的的格状数据，
而视频数据可以理解为对应视频帧宽度、高度、时间的三维数据。

## RNN

> * 循环神经网络，Recurrent Neural Networks, RNN
> * 递归神经网络，Recursive Neural Networks，RNN

虽然很多时候我们把这两种网络都叫做 RNN，但事实上这两种网路的结构事实上是不同的。
而我们常常把两个网络放在一起的原因是：它们都可以处理有序列的问题，比如时间序列等。

举个最简单的例子，我们预测股票走势用 RNN 就比普通的 DNN 效果要好，原因是股票走势和时间相关，
今天的价格和昨天、上周、上个月都有关系。而 RNN 有“记忆”能力，可以“模拟”数据间的依赖关系(Dependency)。
为了加强这种“记忆能力”，人们开发各种各样的变形体，如非常著名的 Long Short-term Memory(LSTM)，
用于解决“长期及远距离的依赖关系”。
同理，另一个循环网络的变种：**双向循环网络(Bi-directional RNN)** 也是现阶段自然语言处理和语音分析中的重要模型。
开发双向循环网络的原因是语言/语音的构成取决于上下文，即“现在”依托于“过去”和“未来”。
单向的循环网络仅着重于从“过去”推出“现在”，而无法对“未来”的依赖性有效的建模。

递归神经网络和循环神经网络不同，它的计算图结构是树状结构而不是网状结构。
递归循环网络的目标和循环网络相似，也是希望解决数据之间的长期依赖问题。
而且其比较好的特点是用树状可以降低序列的长度，从 `$O(n)$` 降低到 `$O(log(n))$`，
熟悉数据结构的朋友都不陌生。但和其他树状数据结构一样，
如何构造最佳的树状结构如平衡树/平衡二叉树并不容易。

## 深度生成模型

> 深度生成模型，Deep Generative Models，DGM

说到生成模型，大家一般想到的无监督学习中的很多建模方法，比如拟合一个高斯混合模型或者使用贝叶斯模型。
深度学习中的生成模型主要还是集中于想使用无监督学习来帮助监督学习，毕竟监督学习所需的标签代价往往很高。
所以请大家不要较真我把这些方法放在了无监督学习中

### 玻尔兹曼机

> * 玻尔兹曼机(Boltzmann Machines,, BM)
> * 受限玻尔兹曼机(Restricted Boltzmann Machines, RBM)

每次一提到玻尔兹曼机和受限玻尔兹曼机我其实都很头疼。简单的说，玻尔兹曼机是一个很漂亮的基于能量的模型，
一般用最大似然法进行学习，而且还符合 Hebb's Rule 这个生物规律。但更多的是适合理论推演，有相当多的实际操作难度。
而受限玻尔兹曼机更加实际，它限定了其结构必须是二分图(Biparitite Graph)且隐藏层和可观测层之间不可以相连接。
此处提及RBM的原因是因为它是深度信念网络的构成要素之一

应用场景：实际工作中一般不推荐单独使用RBM

### 深度信念网络

> 深度信念网络，Deep Belief Neural Networks，DBNN

DBN 是祖师爷 Hinton 在 06 年提出的，主要有两个部分: 

1. 堆叠的受限玻尔兹曼机(Stacked RBM) 
2. 一层普通的前馈网络
   
DBN 最主要的特色可以理解为两阶段学习：

* 阶段 1 用堆叠的 RBM 通过无监督学习进行预训练(Pre-train)，
* 阶段 2 用普通的前馈网络进行微调

神经网络的精髓就是进行特征提取。和后文将提到的自动编码器相似，
我们期待堆叠的 RBF 有数据重建能力，及输入一些数据经过 RBF 我们还可以重建这些数据，
这代表我们学到了这些数据的重要特征。将 RBF 堆叠的原因就是将底层 RBF 学到的特征逐渐传递的上层的 RBF 上，
逐渐抽取复杂的特征。比如下图从左到右就可以是低层 RBF 学到的特征到高层 RBF 学到的复杂特征。
在得到这些良好的特征后就可以用第二部分的传统神经网络进行学习

多说一句，特征抽取并重建的过程不仅可以用堆叠的 RBM，也可以用后文介绍的自编码器

应用场景：现在来说 DBN 更多是了解深度学习“哲学”和“思维模式”的一个手段，
在实际应用中还是推荐 CNN/RNN 等，类似的深度玻尔兹曼机也有类似的特性但工业界使用较少

### GAN

> 生成式对抗网络，Generative Adversarial Networks，GAN

生成式对抗网络用无监督学习同时训练两个模型，内核哲学取自于博弈论。简单的说，GAN 训练两个网络：

1. 生成网络用于生成图片使其与训练数据相似
2. 判别式网络用于判断生成网络中得到的图片是否是真的是训练数据还是伪装的数据

生成网络一般有逆卷积层(deconvolutional layer)，而判别网络一般就是上文介绍的 CNN

![img](images/gan.png)

熟悉博弈论的朋友都知道零和游戏(zero-sum game)会很难得到优化方程，或很难优化，GAN 也不可避免这个问题。
但有趣的是，GAN 的实际表现比我们预期的要好，而且所需的参数也远远按照正常方法训练神经网络，
可以更加有效率的学到数据的分布。另一个常常被放在 GAN 一起讨论的模型叫做变分自编码器(Variational Auto-encoder, VAE)

应用场景：现阶段的 GAN 还主要是在图像领域比较流行，但很多人都认为它有很大的潜力大规模推广到声音、视频领域

## AutoEncoder

> 自编码器，Auto Encoder

自编码器是一种从名字上完全看不出和神经网络有什么关系的无监督神经网络，
而且从名字上看也很难猜测其作用。让我们看一幅图了解它的工作原理：

![img](images/autoencoder.png)

如上图所示，AutoEncoder 主要有 2 个部分：

1. 编码器(Encoder)
2. 解码器(Decoder)

将输入从左端输入后，经过了编码器和解码器，得到了输出一个 `2`。
但事实上真正学习到是中间的用红色标注的部分，即数在低维度的压缩表示。

评估自编码器的方法是**重建误差**，即输出的那个数字 `2` 和原始输入的数字 `2` 之间的差别，当然越小越好。
和主成分分析(PCA)类似，自编码器也可以用来进行数据压缩(Data Compression)，从原始数据中提取最重要的特征。
输入的那个数字 `2` 和输出的数字 `2` 略有不同，这是因为数据压缩中的损失，非常正常。

应用场景：主要用于降维(Dimension Reduction)，这点和 PCA 比较类似。
同时也有专门用于去除噪音还原原始数据的去噪编码器(Denoising Auto-encoder)。
