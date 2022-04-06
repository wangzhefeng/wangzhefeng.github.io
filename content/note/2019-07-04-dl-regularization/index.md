---
title: DL Regularization
author: 王哲峰
date: '2019-07-04'
slug: dl-regularization
categories:
  - deeplearning
tags:
  - note
---

# Dropout



# Batch Normalization

## Batch Normalization 是什么？

在神经网络模型训练过程中，当我们更新之前的权重(weight)时，
每个中间激活层的输出分布会在每次迭代时发生变化，这种现象
称为内部协变量位移(ICS)。所以很自然的一件事就是，如果想
防止这种情况发生，就得需要修正所有的分布。简单地说，如果分布变动了，
会限制这个分布，不让它移动，以帮助梯度优化和防止梯度消失，
这样能帮助神经网络更快。因此减少这种内部协变量位移是推动 Batch Normalization 
发展的关键原则。

## Batch Normalization 原理

Batch Normalization 通过在 Batch 上减去经验平均值，
除以经验标准差来对前一个输出层的输出进行归一化。
这将是数据看起来想高斯分布。

$$\bar{x_{i}} = \frac{x_{i} - \mu_{B}}{\sqrt{\sigma^{2}_{B} + \epsilon}}$$

其中：

- $\mu_{B}$ 为 batch 均值
- $\sigma^{2}_{B}$ 为 batch 方差

$$y_{i} \leftarrow \gamma \hat{x}_{i} + \beta$$

并且，学习了新的平均值 $\gamma$ 和协方差 $\beta$，即，
可以认为 Batch Normalization 能够帮助控制 batch 分布的一阶和二阶动量。

## Batch Normalization 的优点

- 更快地收敛
- 降低初始权重的重要性
- 鲁棒的超参数
- 需要较少的数据进行泛化

![bn](images/bn.png)

## Batch Normalization 的缺点

### Batch Normalization 在使用 batch size 的时候不稳定

Batch Normalization 在训练时的时候必须计算平均值和方差，以便在 batch 中对之前的输出进行归一化。
如果 batch 比较大的话，这种统计估计是比较准确的，而随着 batch 减小，估计的准确性持续减小。

![bn](images/bn1.png)

以上是 ResNet-50 的验证错误图。可以推断，如果 batch 大小保持 为32，
它的最终验证误差在 23 左右，并且随着 batch 大小的减小，
误差会继续减小(batch 大小不能为 1，因为它本身就是平均值)。
损失有很大的不同(大约 10%)。

如果 batch 大小是一个问题，为什么我们不使用更大的 batch？
我们不能在每种情况下都使用更大的 batch。在 finetune 的时候，
我们不能使用大的 batch，以免过高的梯度对模型造成伤害。
在分布式训练的时候，大的 batch 最终将作为一组小 batch 分布在各个实例中。


### Batch Normalization 导致训练时间增加

NVIDIA 和卡耐基梅隆大学进行的实验结果表明，“尽管 Batch Normalization 不是计算密集型，
而且收敛所需的总迭代次数也减少了。” 但是每个迭代的时间显著增加了，
而且还随着batch大小的增加而进一步增加。

![bn](images/bn2.png)

<center>ResNet-50 在ImageNet上使用 Titan X Pascal</center>

你可以看到，batch normalization 消耗了总训练时间的 1/4。
原因是 batch normalization 需要通过输入数据进行两次迭代，
一次用于计算 batch 统计信息，另一次用于归一化输出。

### Batch Normalization 训练和推理时结果不一样

例如，在真实世界中做“物体检测”。在训练一个物体检测器时，
我们通常使用大 batch(YOLOv4 和 Faster-RCNN 都是在默认 batch=64 的情况下训练的)。
但在投入生产后，这些模型的工作并不像训练时那么好。
这是因为它们接受的是大 batch 的训练，而在实时情况下，它们的 batch 大小等于 1，
因为它必须一帧帧处理。考虑到这个限制，
一些实现倾向于基于训练集上使用预先计算的平均值和方差。
另一种可能是基于你的测试集分布计算平均值和方差值。

### Batch Normalization 对于在线学习不友好

![online-learning](images/online-learning.png)

<center>典型的在线学习 Pipeline</center>

与batch学习相比，在线学习是一种学习技术，在这种技术中，
系统通过依次向其提供数据实例来逐步接受训练，可以是单独的，
也可以是通过称为mini-batch的小组进行。每个学习步骤都是快速和便宜的，
所以系统可以在新的数据到达时实时学习。由于它依赖于外部数据源，
数据可能单独或批量到达。由于每次迭代中batch大小的变化，
对输入数据的尺度和偏移的泛化能力不好，最终影响了性能。

### Batch Normalization 对循环神经网络不友好

虽然 Batch Normalization可以显著提高卷积神经网络的训练和泛化速度，
但它们很难应用于递归结构。Batch Normalization 可以应用于RNN堆栈之间，
其中归一化是“垂直”应用的，即每个RNN的输出。但是它不能“水平地”应用，
例如在时间步之间，因为它会因为重复的重新缩放而产生爆炸性的梯度而伤害到训练。

## Batch Normalization 的可替代方法

在 Batch Normalization 无法很好工作的情况下，有几种可替代方法可用：

- Layer Normalization
- Instance Normalization
- Group Normalization (+ weight standardization)
- Synchronous Batch Normalization

