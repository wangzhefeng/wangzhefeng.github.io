---
title: 模型层
author: 王哲峰
date: '2023-03-26'
slug: layer
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

- [归一化层 Normalization](#归一化层-normalization)
  - [结构化数据的 BatchNorm1d](#结构化数据的-batchnorm1d)
  - [图片数据的 BatchNorm2d](#图片数据的-batchnorm2d)
  - [文本数据的 LayerNorm](#文本数据的-layernorm)
  - [自适应学习 SwitchableNorm](#自适应学习-switchablenorm)
  - [对 BatchNorm 需要注意的几点](#对-batchnorm-需要注意的几点)
  - [归一化层 PyTorch 示例](#归一化层-pytorch-示例)
- [卷积层](#卷积层)
  - [普通卷积](#普通卷积)
  - [空洞卷积](#空洞卷积)
  - [分组卷积](#分组卷积)
  - [深度可分离卷积](#深度可分离卷积)
  - [转置卷积](#转置卷积)
  - [卷积层 PyTorch 示例](#卷积层-pytorch-示例)
- [上采样层](#上采样层)
  - [上采样层 PyTorch 示例](#上采样层-pytorch-示例)
</p></details><p></p>


# 归一化层 Normalization

重点说说各种归一化层:

`$$y = \frac{x - \mathrm{E}[x]}{ \sqrt{\mathrm{Var}[x] + \epsilon}} \cdot \gamma + \beta$$`

## 结构化数据的 BatchNorm1d

结构化数据的主要区分度来自每个样本特征在全体样本中的排序，将全部样本的某个特征都进行相同的放大缩小平移操作，
样本间的区分度基本保持不变，所以结构化数据可以做 BatchNorm，但 LayerNorm 会打乱全体样本根据某个特征的排序关系，引起区分度下降

![](https://tva1.sinaimg.cn/large/e6c9d24egy1h5mbd2ill5j20a808z0ta.jpg)

## 图片数据的 BatchNorm2d

图片数据的主要区分度来自图片中的纹理结构，所以图片数据的归一化一定要在图片的宽高方向上操作以保持纹理结构，
此外在 Batch 维度上操作还能够引入少许的正则化，对提升精度有进一步的帮助

![](https://tva1.sinaimg.cn/large/e6c9d24egy1h5m92dtnd0j20tn07ztab.jpg)

## 文本数据的 LayerNorm

文本数据的主要区分度来自于词向量(Embedding 向量)的方向，所以文本数据的归一化一定要在特征(通道)维度上操作以保持词向量方向不变。
此外文本数据还有一个重要的特点是不同样本的序列长度往往不一样，所以不可以在 Sequence 和 Batch 维度上做归一化，
否则将不可避免地让 padding 位置对应的向量变成非零向量

<img src="https://tva1.sinaimg.cn/large/e6c9d24egy1h5m903lv0nj20jc0iawfx.jpg" width=50% height="50%" />

## 自适应学习 SwitchableNorm

有论文提出了一种可自适应学习的归一化：`SwitchableNorm`，可应用于各种场景且有一定的效果提升

`SwitchableNorm` 是将 BatchNorm、LayerNorm、InstanceNorm 结合，赋予权重，让网络自己去学习归一化层应该使用什么方法

## 对 BatchNorm 需要注意的几点

1. BatchNorm 放在激活函数前还是激活函数后？
    - 原始论文认为将 BatchNorm 放在激活函数前效果较好，
      后面的研究一般认为将 BatchNorm 放在激活函数之后更好
2. BatchNorm 在训练过程和推理过程的逻辑是否一样？
    - 不一样！训练过程 BatchNorm 的均值和方差和根据 mini-batch 中的数据估计的，
     而推理过程中 BatchNorm 的均值和方差是用的训练过程中的全体样本估计的。
     因此预测过程是稳定的，相同的样本不会因为所在批次的差异得到不同的结果，
     但训练过程中则会受到批次中其他样本的影响，所以有正则化效果
3. BatchNorm 的精度效果与 batch_size 大小有何关系? 
    - 如果受到 GPU 内存限制，不得不使用很小的 batch_size，
      训练阶段时使用的 mini-batch 上的均值和方差的估计和预测阶段时使用的全体样本上的均值和方差的估计差异可能会较大，
      效果会变差。这时候，可以尝试 `LayerNorm` 或者 `GroupNorm` 等归一化方法

## 归一化层 PyTorch 示例

BatchNorm2d：

```python
import torch
from torch import nn

batch_size, channel, height, width = 32, 16, 128, 128
tensor = torch.arange(0, 32 * 16 * 128 * 128).view(32, 16, 128, 128).float()
bn = nn.BatchNorm2d(num_features = channel, affine = False)
bn_out = bn(tensor)

channel_mean = torch.mean(bn_out[:, 0, :, :])
channel_std = torch.std(bn_out[:, 0, :, :])
print(f"channel mean: {channel_mean.item()}")
print(f"channel std: {channel_std.item()}")
```

LayerNorm：

```python
import torch
from torch import nn

batch_size, sequence, features = 32, 100, 2048
tensor = torch.arange(0, 32 * 100 * 2048).view(32, 100, 2048).float()
ln = nn.LayerNorm(
    normalized_shape = [features], 
    elementwise_affine = False
)
ln_out = ln(tensor)

token_mean = torch.mean(ln_out[0, 0, :])
token_std = torch.std(ln_out[0, 0, :])
print(f"token mean: {token_mean.item()}")
print(f"token std: {token_std.item()}")
```

# 卷积层

卷积层的输出尺寸计算：

* 卷积输出尺寸计算

`$$o = (i + 2p -k) / s + 1$$`

* 空洞卷积

`$$k=d(k-1)+1$$`

其中：

* `$o$` 是输出尺寸
* `$i$` 是输入尺寸
* `$p$` 是 padding 大小
* `$k$` 是卷积核尺寸
* `$s$` 是 stride 步长
* `$d$` 是空洞卷积 dilation 膨胀系数

## 普通卷积

普通卷积的操作分为 3 个维度:

* 在空间维度(Height 和 Width 维度)，是共享卷积核权重，滑窗相乘求和
    - 融合空间信息
* 在输入通道维度，是每一个通道使用不同的卷积核参数，并对输入通道维度求和
    - 融合通道信息
* 在输出通道维度，操作方式是并行堆叠(多种)，有多少个卷积核就有多少个输出通道

> 普通卷积层的参数数量 = 输入通道数 × 卷积核尺寸(如 3x3) × 输出通道数(即卷积核个数) + 输出通道数(考虑偏置时）

![](https://tva1.sinaimg.cn/large/e6c9d24egy1h5nhe0lsutg20az0aln03.gif)

## 空洞卷积

和普通卷积相比，空洞卷积可以在保持较小参数规模的条件下增大感受野，
常用于图像分割领域

其缺点是可能产生网格效应，即有些像素被空洞漏过无法利用到，
可以通过使用不同膨胀因子的空洞卷积的组合来克服该问题

![](https://tva1.sinaimg.cn/large/e6c9d24egy1h5nhe0y7x1g20az0al0vu.gif)

* [Dilated Convolution(空洞卷积、膨胀卷积)详解](https://developer.orbbec.com.cn/v/blog_detail/892)

## 分组卷积

和普通卷积相比，分组卷积将输入通道分成 g 组，卷积核也分成对应的 g 组，
每个卷积核只在其对应的那组输入通道上做卷积，最后将 g 组结果堆叠拼接

由于每个卷积核只需要在全部输入通道的 1/g 个通道上做卷积，参数量降低为普通卷积的 1/g。
分组卷积要求输入通道和输出通道数都是 g 的整数倍

![](https://tva1.sinaimg.cn/large/e6c9d24egy1h5npy1zyalj20ie0erwf8.jpg)

* [理解分组卷积和深度可分离卷积如何降低参数量](https://zhuanlan.zhihu.com/p/65377955)

## 深度可分离卷积

深度可分离卷积的思想是将融合空间信息和融合通道信息的操作在卷积中分成独立的两步完成。
深度可分离卷积的思想是先用 `$g=m$`(输入通道数)的分组卷积逐通道作用融合空间信息，
再用 `$n$`(输出通道数)个 `$1 \times 1$`卷积融合通道信息。
其参数量为 `$(m \times k \times k)+ n \times m$`, 
相比普通卷积的参数量 `$m \times n \times k \times k$` 显著减小。
同时，由于深度可分离卷积融合空间信息与融合通道信息相互分离，往往还能比普通卷积取得更好的效果

![](https://tva1.sinaimg.cn/large/e6c9d24egy1h5npbiuvzvj20uq0e7dge.jpg)

## 转置卷积

一般的卷积操作后会让特征图尺寸变小，但转置卷积(也被称为反卷积)可以实现相反的效果，即放大特征图尺寸

两种方式理解转置卷积:

* 第一种方式是转置卷积是一种特殊的卷积，通过设置合适的 padding 的大小来恢复特征图尺寸
* 第二种理解基于卷积运算的矩阵乘法表示方法，转置卷积相当于将卷积核对应的表示矩阵做转置，
  然后乘上输出特征图压平的一维向量，即可恢复原始输入特征图的大小

![](https://tva1.sinaimg.cn/large/e6c9d24egy1h5ns98iiamj20v70u075e.jpg)

* [转置卷积(Transpose Convolution)](https://zhuanlan.zhihu.com/p/115070523)

## 卷积层 PyTorch 示例

卷积输出尺寸关系演示：

```python
import torch 
from torch import nn 
import torch.nn.functional as F 

# 卷积输出尺寸计算公式 o = (i + 2*p -k')//s  + 1 
# 对空洞卷积 k' = d(k-1) + 1
# o是输出尺寸，i 是输入尺寸，p是 padding大小， k 是卷积核尺寸， s是stride步长, d是dilation空洞参数

inputs = torch.arange(0,25).view(1, 1, 5, 5).float()  # i= 5
filters = torch.tensor([[[[1.0, 1], [1, 1]]]])  # k = 2

outputs = F.conv2d(inputs, filters) # o = (5+2*0-2)//1+1 = 4
outputs_s2 = F.conv2d(inputs, filters, stride = 2)  #o = (5+2*0-2)//2+1 = 2
outputs_p1 = F.conv2d(inputs, filters, padding = 1) #o = (5+2*1-2)//1+1 = 6
outputs_d2 = F.conv2d(inputs, filters, dilation = 2) #o = (5+2*0-(2(2-1)+1))//1+1 = 3

print("--inputs--")
print(inputs)
print("--filters--")
print(filters)

print("--outputs--")
print(outputs, "\n")

print("--outputs(stride=2)--")
print(outputs_s2, "\n")

print("--outputs(padding=1)--")
print(outputs_p1, "\n")

print("--outputs(dilation=2)--")
print(outputs_d2, "\n")
```

```
--inputs--
tensor([[[[ 0.,  1.,  2.,  3.,  4.],
          [ 5.,  6.,  7.,  8.,  9.],
          [10., 11., 12., 13., 14.],
          [15., 16., 17., 18., 19.],
          [20., 21., 22., 23., 24.]]]])
--filters--
tensor([[[[1., 1.],
          [1., 1.]]]])
--outputs--
tensor([[[[12., 16., 20., 24.],
          [32., 36., 40., 44.],
          [52., 56., 60., 64.],
          [72., 76., 80., 84.]]]]) 

--outputs(stride=2)--
tensor([[[[12., 20.],
          [52., 60.]]]]) 

--outputs(padding=1)--
tensor([[[[ 0.,  1.,  3.,  5.,  7.,  4.],
          [ 5., 12., 16., 20., 24., 13.],
          [15., 32., 36., 40., 44., 23.],
          [25., 52., 56., 60., 64., 33.],
          [35., 72., 76., 80., 84., 43.],
          [20., 41., 43., 45., 47., 24.]]]]) 

--outputs(dilation=2)--
tensor([[[[24., 28., 32.],
          [44., 48., 52.],
          [64., 68., 72.]]]]) 
```

卷积层参数数量演示：

```python
import torch
from torch import nn

features = torch.randn(8, 64, 128, 128)
print("features.shape:", features.shape)
print("\n")

# 普通卷积
print("--conv--")
conv = nn.Conv2d(in_channels = 64, out_channels = 32, kernel_size = 3)
conv_out = conv(features)
print("conv_out.shape:", conv_out.shape) 
print("conv.weight.shape:", conv.weight.shape)
print("\n")

#分组卷积
print("--group conv--")
conv_group = nn.Conv2d(in_channels = 64, out_channels = 32, kernel_size = 3, groups = 8)
group_out = conv_group(features)
print("group_out.shape:", group_out.shape) 
print("conv_group.weight.shape:", conv_group.weight.shape)
print("\n")

#深度可分离卷积
print("--separable conv--")
depth_conv = nn.Conv2d(in_channels = 64, out_channels = 64, kernel_size = 3, groups = 64)
oneone_conv = nn.Conv2d(in_channels = 64, out_channels = 32, kernel_size = 1)
separable_conv = nn.Sequential(depth_conv,oneone_conv)
separable_out = separable_conv(features)
print("separable_out.shape:", separable_out.shape) 
print("depth_conv.weight.shape:", depth_conv.weight.shape)
print("oneone_conv.weight.shape:", oneone_conv.weight.shape)
print("\n")

#转置卷积
print("--conv transpose--")
conv_t = nn.ConvTranspose2d(in_channels = 32, out_channels = 64, kernel_size = 3)
features_like = conv_t(conv_out)
print("features_like.shape:", features_like.shape)
print("conv_t.weight.shape:", conv_t.weight.shape)
```

```
features.shape: torch.Size([8, 64, 128, 128])


--conv--
conv_out.shape: torch.Size([8, 32, 126, 126])
conv.weight.shape: torch.Size([32, 64, 3, 3])


--group conv--
group_out.shape: torch.Size([8, 32, 126, 126])
conv_group.weight.shape: torch.Size([32, 8, 3, 3])


--separable conv--
separable_out.shape: torch.Size([8, 32, 126, 126])
depth_conv.weight.shape: torch.Size([64, 1, 3, 3])
oneone_conv.weight.shape: torch.Size([32, 64, 1, 1])


--conv transpose--
features_like.shape: torch.Size([8, 64, 128, 128])
conv_t.weight.shape: torch.Size([32, 64, 3, 3])
```

# 上采样层

除了使用转置卷积进行上采样外，在图像分割领域更多的时候一般是使用双线性插值的方式进行上采样，
该方法没有需要学习的参数，通常效果也更好，除了双线性插值之外，
还可以使用最邻近插值的方式进行上采样，但使用较少。此外，还有一种上采样方法是反池化。使用也不多

![](https://tva1.sinaimg.cn/large/e6c9d24egy1h5nsi5pt4eg20na0co74k.gif)

## 上采样层 PyTorch 示例

```python
import torch 
from torch import nn 

inputs = torch.arange(1, 5, dtype = torch.float32).view(1, 1, 2, 2)
print("inputs:")
print(inputs)
print("\n")

nearest = nn.Upsample(scale_factor = 2, mode = 'nearest')
bilinear = nn.Upsample(scale_factor = 2, mode = "bilinear", align_corners = True)

print("nearest(inputs): ")
print(nearest(inputs))
print("\n")
print("bilinear(inputs): ")
print(bilinear(inputs)) 
```

```
inputs:
tensor([[[[1., 2.],
          [3., 4.]]]])


nearest(inputs)：
tensor([[[[1., 1., 2., 2.],
          [1., 1., 2., 2.],
          [3., 3., 4., 4.],
          [3., 3., 4., 4.]]]])


bilinear(inputs)：
tensor([[[[1.0000, 1.3333, 1.6667, 2.0000],
          [1.6667, 2.0000, 2.3333, 2.6667],
          [2.3333, 2.6667, 3.0000, 3.3333],
          [3.0000, 3.3333, 3.6667, 4.0000]]]])
```

* [Differentiable Learning-To-Normalize Via Switchable Normalization](https://arxiv.org/pdf/1806.10779.pdf)
