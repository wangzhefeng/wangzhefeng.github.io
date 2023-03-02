---
title: PyTorch 模型层
author: 王哲峰
date: '2022-08-12'
slug: dl-pytorch-model-layer
categories:
  - pytorch
tags:
  - tool
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

- [模型层简介](#模型层简介)
- [基础层](#基础层)
  - [模型层 API](#模型层-api)
  - [归一化层](#归一化层)
    - [结构化数据的 BatchNorm1d](#结构化数据的-batchnorm1d)
    - [图片数据的 BatchNorm2d](#图片数据的-batchnorm2d)
    - [文本数据的 LayerNorm](#文本数据的-layernorm)
    - [自适应学习 SwitchableNorm](#自适应学习-switchablenorm)
    - [对 BatchNorm 需要注意的几点](#对-batchnorm-需要注意的几点)
  - [PyTorch 示例](#pytorch-示例)
- [卷积网络相关层](#卷积网络相关层)
  - [模型层 API](#模型层-api-1)
  - [卷积层](#卷积层)
    - [普通卷积](#普通卷积)
    - [空洞卷积](#空洞卷积)
    - [分组卷积](#分组卷积)
    - [深度可分离卷积](#深度可分离卷积)
    - [转置卷积](#转置卷积)
  - [上采样层](#上采样层)
  - [PyTorch 示例](#pytorch-示例-1)
    - [卷积输出尺寸关系演示](#卷积输出尺寸关系演示)
    - [卷积层参数数量演示](#卷积层参数数量演示)
    - [上采样层](#上采样层-1)
- [循环网络相关层](#循环网络相关层)
  - [模型层 API](#模型层-api-2)
  - [RNN](#rnn)
  - [LSTM](#lstm)
    - [LSTM 结构解析](#lstm-结构解析)
    - [LSTM 数学表示](#lstm-数学表示)
    - [LSTM 解释](#lstm-解释)
  - [GRU](#gru)
    - [GRU 结构解析](#gru-结构解析)
    - [GRU 数学表示](#gru-数学表示)
    - [GRU 解释](#gru-解释)
  - [参考](#参考)
  - [PyTorch 示例](#pytorch-示例-2)
- [Transformer 相关层](#transformer-相关层)
  - [模型层 API](#模型层-api-3)
  - [Transformer 结构解析](#transformer-结构解析)
  - [Transformer 数学表示](#transformer-数学表示)
  - [Transformer 要点问题](#transformer-要点问题)
  - [PyTorch 示例](#pytorch-示例-3)
  - [参考](#参考-1)
- [自定义模型层](#自定义模型层)
  - [PyTorch 的 torch.nn.Linear 层源码](#pytorch-的-torchnnlinear-层源码)
- [functional 和 Module](#functional-和-module)
  - [functional](#functional)
  - [Module](#module)
    - [使用 Module 管理参数](#使用-module-管理参数)
    - [使用 Module 管理子模块](#使用-module-管理子模块)
  - [参考](#参考-2)
</p></details><p></p>

# 模型层简介

深度学习模型一般由各种模型层组合而称。`torch.nn` 中内置了非常丰富的各种模型层，
它们都属于 `torch.nn.Module` 的子类，具备参数管理功能

如果这些内置的模型层不能满足需求，也可以通过继承 `torch.nn.Module` 基类构建自定义的模型层。
实际上 PyTorch 不区分模型和模型层，都是通过继承 `torch.nn.Module` 进行构建，
因此，只要继承 `torch.nn.Module` 基类并实现 `forward` 方法即可自定义模型层

# 基础层

## 模型层 API

* `nn.Linear`: 全连接层
    - 参数个数 = 输入层特征数 × 输出层特征数(weight) ＋ 输出层特征数(bias)
* `nn.Embedding`: 嵌入层
    - 一种比 One-Hot 更加有效的对离散特征进行编码的方法
    - 一般用于将输入中的单词映射为稠密向量，嵌入层的参数需要学习
* `nn.Flatten`: 压平层
    - 用于将多维张量样本压成一维张量样本
* Normalization
    - `nn.GroupNorm`: 组归一化
        - 一种替代批归一化的方法，将通道分成若干组进行归一，不受 batch 大小限制
    - `nn.LayerNorm`: 层归一化
        - 常用于 NLP 领域，不受序列长度不一致影响
    - `nn.InstanceNorm2d`: 样本归一化
        - 一般在图像风格迁移任务中效果较好
    - BatchNormalization
        - `nn.BatchNorm1d`: 一维批标准化层
            - 通过线性变换将输入批次缩放平移到稳定的均值和标准差。
              可以增强模型对输入不同分布的适应性，加快模型训练速度，有轻微正则化效果
            - 一般在激活函数之前使用。可以用 `affine` 参数设置该层是否含有可以训练的参数
        - `nn.BatchNorm2d`: 二维批标准化层
            - 常用于 CV 领域
        - `nn.BatchNorm3d`: 三维批标准化层
* Dropout
    - `nn.Dropout`: 一维随机丢弃层
        - 一种正则化手段
    - `nn.Dropout2d`: 二维随机丢弃层
    - `nn.Dropout3d`: 三维随机丢弃层
* Padding
    - `nn.ConstantPad2d`: 二维常数填充层
        - 对二维张量样本填充常数扩展长度
    - `nn.ReplicationPad1d`: 一维复制填充层
        - 对一维张量样本通过复制边缘值填充扩展长度
    - `nn.ZeroPad2d`: 二维零值填充层
        - 对二维张量样本在边缘填充0值
* `nn.Threshold`: 限幅层
    - 当输入大于或小于阈值范围时，截断之

## 归一化层

重点说说各种归一化层:

`$$y = \frac{x - \mathrm{E}[x]}{ \sqrt{\mathrm{Var}[x] + \epsilon}} \cdot \gamma + \beta$$`

### 结构化数据的 BatchNorm1d

* 结构化数据的主要区分度来自每个样本特征在全体样本中的排序，
  将全部样本的某个特征都进行相同的放大缩小平移操作，
  样本间的区分度基本保持不变，所以结构化数据可以做 BatchNorm，
  但 LayerNorm 会打乱全体样本根据某个特征的排序关系，引起区分度下降

![](https://tva1.sinaimg.cn/large/e6c9d24egy1h5mbd2ill5j20a808z0ta.jpg)

### 图片数据的 BatchNorm2d

* 图片数据的主要区分度来自图片中的纹理结构，
  所以图片数据的归一化一定要在图片的宽高方向上操作以保持纹理结构，
  此外在 Batch 维度上操作还能够引入少许的正则化，对提升精度有进一步的帮助

![](https://tva1.sinaimg.cn/large/e6c9d24egy1h5m92dtnd0j20tn07ztab.jpg)

### 文本数据的 LayerNorm

- 文本数据的主要区分度来自于词向量(Embedding 向量)的方向，
  所以文本数据的归一化一定要在特征(通道)维度上操作以保持词向量方向不变。
  此外文本数据还有一个重要的特点是不同样本的序列长度往往不一样，
  所以不可以在 Sequence 和 Batch 维度上做归一化，
  否则将不可避免地让 padding 位置对应的向量变成非零向量

<img src="https://tva1.sinaimg.cn/large/e6c9d24egy1h5m903lv0nj20jc0iawfx.jpg" width=50% height="50%" />

### 自适应学习 SwitchableNorm

此外，有论文提出了一种可自适应学习的归一化: `SwitchableNorm`，
可应用于各种场景且有一定的效果提升

`SwitchableNorm` 是将 BatchNorm、LayerNorm、InstanceNorm 结合，赋予权重，
让网络自己去学习归一化层应该使用什么方法

### 对 BatchNorm 需要注意的几点

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

## PyTorch 示例

```python
import torch
from torch import nn

batch_size, channel, height, width = 32, 16, 128, 128
tensor = torch.arange(0, 32 * 16 * 128 * 128) \
    .view(32, 16, 128, 128) \
    .float()
bn = nn.BatchNorm2d(num_features = channel, affine = False)
bn_out = bn(tensor)

channel_mean = torch.mean(bn_out[:, 0, :, :])
channel_std = torch.std(bn_out[:, 0, :, :])
print(f"channel mean: {channel_mean.item()}")
print(f"channel std: {channel_std.item()}")
```

```python
import torch
from torch import nn

batch_size, sequence, features = 32, 100, 2048
tensor = torch.arange(0, 32 * 100 * 2048) \
    .view(32, 100, 2048) \
    .float()
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

# 卷积网络相关层

## 模型层 API

* 卷积层
    - `nn.Conv1d`: 普通一维卷积，常用于文本
        - 参数个数 = 输入通道数 × 卷积核尺寸(如3) × 卷积核个数 + 卷积核尺寸(如3)
    - `nn.Conv2d`: 普通二维卷积，常用于图像
        - 参数个数 = 输入通道数 × 卷积核尺寸(如 3x3) × 卷积核个数 + 卷积核尺寸(如 3x3) 
        - 通过调整 `dilation` 参数大于 1，可以变成空洞卷积，增加感受野
        - 通过调整 `groups` 参数不为 1，可以变成分组卷积。
          分组卷积中每个卷积核仅对其对应的一个分组进行操作。
          当 `groups` 参数数量等于输入通道数时，
          相当于 TensorFlow 中的二维深度卷积层 `tf.keras.layers.DepthwiseConv2D`
        - 利用分组卷积和 `$1 \times 1$` 卷积的组合操作，
          可以构造相当于 TensorFlow 中的二维深度可分离卷积层 `tf.keras.layers.SeparableConv2D`
    - `nn.Conv3d`: 普通三维卷积，常用于视频
        - 参数个数 = 输入通道数 × 卷积核尺寸(如 3x3x3) × 卷积核个数 + 卷积核尺寸(如 3x3x3)
* 池化层
    - `nn.MaxPool1d`: 一维最大池化
    - `nn.MaxPool2d`: 二维最大池化
        - 一种下采样方式，没有需要训练的参数
    - `nn.MaxPool3d`: 三维最大池化
    - `nn.AdaptiveMaxPool2d`: 二维自适应最大池化
        - 无论输入图像的尺寸如何变化，输出的图像尺寸是固定的。该函数的实现原理，
          大概是通过输入图像的尺寸和要得到的输出图像的尺寸来反向推算池化算子的 `padding`、`stride` 等参数
    - `nn.FractionalMaxPool2d`: 二维分数最大池化
        - 普通最大池化通常输入尺寸是输出的整数倍，而分数最大池化则可以不必是整数
        - 分数最大池化使用了一些随机采样策略，有一定的正则效果，可以用它来代替普通最大池化和 Dropout 层
    - `nn.AvgPool2d`: 二维平均池化
    - `nn.AdaptiveAvgPool2d`: 二维自适应平均池化
        - 无论输入的维度如何变化，输出的维度是固定的
* 其他
    - `nn.ConvTranspose2d`: 二维卷积转置层，俗称反卷积层
        - 并非卷积的逆操作，但在卷积核相同的情况下，当其输入尺寸是卷积操作输出尺寸的情况下，
          卷积转置的输出尺寸恰好是卷积操作的输入尺寸。在语义分割中可用于上采样
    - `nn.Upsample`: 上采样层，操作效果和池化相反。可以通过 `mode` 参数控制上采样策略为：
        - `"nearest"`：最邻近策略
        - `"linear"`：线性插值策略
    - `nn.Unfold`: 滑动窗口提取层
        - 其参数和卷积操作 `nn.Conv2d` 相同
        - 实际上，卷积操作可以等价于 `nn.Unfold` 和 `nn.Linear` 以及 `nn.Fold` 的一个组合。
          其中 `nn.Unfold` 操作可以从输入中提取各个滑动窗口的数值矩阵，并将其压平成一维。
          利用 `nn.Linear` 将 `nn.Unfold` 的输出和卷积核做乘法后，
          再使用 `nn.Fold` 操作将结果转换成输出图片形状
    - `nn.Fold`: 逆滑动窗口提取层

## 卷积层

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

### 普通卷积

普通卷积的操作分为 3 个维度:

* 在空间维度(Height 和 Width 维度)，是共享卷积核权重，滑窗相乘求和
    - 融合空间信息
* 在输入通道维度，是每一个通道使用不同的卷积核参数，并对输入通道维度求和
    - 融合通道信息
* 在输出通道维度，操作方式是并行堆叠(多种)，有多少个卷积核就有多少个输出通道

> 普通卷积层的参数数量 = 输入通道数 × 卷积核尺寸(如 3x3) × 输出通道数(即卷积核个数) + 输出通道数(考虑偏置时）

![](https://tva1.sinaimg.cn/large/e6c9d24egy1h5nhe0lsutg20az0aln03.gif)

### 空洞卷积

和普通卷积相比，空洞卷积可以在保持较小参数规模的条件下增大感受野，
常用于图像分割领域

其缺点是可能产生网格效应，即有些像素被空洞漏过无法利用到，
可以通过使用不同膨胀因子的空洞卷积的组合来克服该问题

![](https://tva1.sinaimg.cn/large/e6c9d24egy1h5nhe0y7x1g20az0al0vu.gif)

* [Dilated Convolution(空洞卷积、膨胀卷积)详解](https://developer.orbbec.com.cn/v/blog_detail/892)

### 分组卷积

和普通卷积相比，分组卷积将输入通道分成 g 组，卷积核也分成对应的 g 组，
每个卷积核只在其对应的那组输入通道上做卷积，最后将 g 组结果堆叠拼接

由于每个卷积核只需要在全部输入通道的 1/g 个通道上做卷积，参数量降低为普通卷积的 1/g。
分组卷积要求输入通道和输出通道数都是 g 的整数倍

![](https://tva1.sinaimg.cn/large/e6c9d24egy1h5npy1zyalj20ie0erwf8.jpg)

* [理解分组卷积和深度可分离卷积如何降低参数量](https://zhuanlan.zhihu.com/p/65377955)

### 深度可分离卷积

深度可分离卷积的思想是将融合空间信息和融合通道信息的操作在卷积中分成独立的两步完成。
深度可分离卷积的思想是先用 `$g=m$`(输入通道数)的分组卷积逐通道作用融合空间信息，
再用 `$n$`(输出通道数)个 `$1 \times 1$`卷积融合通道信息。
其参数量为 `$(m \times k \times k)+ n \times m$`, 
相比普通卷积的参数量 `$m \times n \times k \times k$` 显著减小。
同时，由于深度可分离卷积融合空间信息与融合通道信息相互分离，往往还能比普通卷积取得更好的效果

![](https://tva1.sinaimg.cn/large/e6c9d24egy1h5npbiuvzvj20uq0e7dge.jpg)

### 转置卷积

一般的卷积操作后会让特征图尺寸变小，但转置卷积(也被称为反卷积)可以实现相反的效果，即放大特征图尺寸

两种方式理解转置卷积:

* 第一种方式是转置卷积是一种特殊的卷积，通过设置合适的 padding 的大小来恢复特征图尺寸
* 第二种理解基于卷积运算的矩阵乘法表示方法，转置卷积相当于将卷积核对应的表示矩阵做转置，
  然后乘上输出特征图压平的一维向量，即可恢复原始输入特征图的大小

![](https://tva1.sinaimg.cn/large/e6c9d24egy1h5ns98iiamj20v70u075e.jpg)

* [转置卷积(Transpose Convolution)](https://zhuanlan.zhihu.com/p/115070523)

## 上采样层

除了使用转置卷积进行上采样外，在图像分割领域更多的时候一般是使用双线性插值的方式进行上采样，
该方法没有需要学习的参数，通常效果也更好，除了双线性插值之外，
还可以使用最邻近插值的方式进行上采样，但使用较少。此外，还有一种上采样方法是反池化。使用也不多

![](https://tva1.sinaimg.cn/large/e6c9d24egy1h5nsi5pt4eg20na0co74k.gif)

## PyTorch 示例

### 卷积输出尺寸关系演示

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


### 卷积层参数数量演示

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

### 上采样层

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

# 循环网络相关层

## 模型层 API

* `nn.RNN`: 简单循环网络层(支持多层)
    - 容易存在梯度消失，不能够适用长期依赖问题。一般较少使用
* `nn.LSTM`: 长短记忆循环网络层(支持多层)
    - 最普遍使用的循环网络层。具有携带轨道，遗忘门，更新门，输出门，
      可以较为有效地缓解梯度消失问题，从而能够适用长期依赖问题
    - 设置 `bidirectional = True` 时可以得到双向 LSTM
    - 需要注意的时，默认的输入和输出形状是 `(seq, batch, feature)`, 
      如果需要将 `batch` 维度放在第 `0` 维，则要设置 `batch_first` 参数设置为 `True`
* `nn.GRU`: 门控循环网络层(支持多层)
    - LSTM 的低配版，不具有携带轨道，参数数量少于 LSTM，训练速度更快
* `nn.RNNCell`: 简单循环网络单元
    - 和 `nn.RNN` 在整个序列上迭代相比，它仅在序列上迭代一步。一般较少使用
* `nn.LSTMCell`: 长短记忆循环网络单元
    - 和 `nn.LSTM` 在整个序列上迭代相比，它仅在序列上迭代一步。一般较少使用
* `nn.GRUCell`: 门控循环网络单元
    - 和 `nn.GRU` 在整个序列上迭代相比，它仅在序列上迭代一步。一般较少使用

## RNN

一般地，各种 RNN 序列模型层(RNN, GRU, LSTM 等)可以用函数表示如下:

`$$h_t = f(h_{t-1},i_t)$$`

这个公式的含义是: `$t$` 时刻循环神经网络的输出向量 `$h_t$`，
由 `$t-1$` 时刻的输出向量 `$h_{t-1}$` 和 `$t$` 时刻的输入 `$i_t$` 变换而来

## LSTM

### LSTM 结构解析

![](https://tva1.sinaimg.cn/large/e6c9d24egy1h5o1d0limwj21n80qg0vb.jpg)

### LSTM 数学表示

`$$\begin{align}
i_{t}=\sigma\left(W_{i} x_{t}+U_{i} h_{t-1}+b_{i}\right) \tag{1} \\
f_{t}=\sigma\left(W_{f} x_{t}+U_{f} h_{t-1}+b_{f}\right) \tag{2} \\
o_{t}=\sigma\left(W_{o} x_{t}+U_{o} h_{t-1}+b_{o}\right) \tag{3} \\
\tilde{c}_{t}=\tanh \left(W_{c} x_{t}+U_{c} h_{t-1}+b_{c}\right) \tag{4} \\
c_{t}=f_{t} \odot c_{t-1}+i_{t} \odot \tilde{c}_{t} \tag{5} \\
h_{t}=o_{t} \odot \tanh \left(c_{t}\right) \tag{6}
\end{align}$$`

### LSTM 解释

* LSTM 通过引入了三个门来控制信息的传递，分别是遗忘门、输入门、输出门。三个门的作用为: 
    - 遗忘门: 遗忘门 `$f_t$` 控制上一时刻的内部状态，需要遗忘多少信息
    - 输入门: 输入门 `$i_t$` 控制当前时刻的候选状态，有多少信息需要保存
    - 输出门: 输出门 `$o_t$` 控制当前时刻的内部状态，有多少信息需要输出给外部状态

## GRU

### GRU 结构解析

![](https://tva1.sinaimg.cn/large/e6c9d24egy1h5ohjoivv7j21nj0u0q5n.jpg)

### GRU 数学表示

`$$\begin{align}
z_{t}=\sigma\left(W_{z} x_{t}+U_{z} h_{t-1}+b_{z}\right)\tag{1} \\
r_{t}=\sigma\left(W_{r} x_{t}+U_{r} h_{t-1}+b_{r}\right) \tag{2}\\
\tilde{h}_{t}=\tanh \left(W_{h} x_{t}+U_{h}\left(r_{t} \odot h_{t-1}\right)+b_{h}\right) \tag{3}\\
h_{t}=\left(1-z_{t}\right) \odot \tilde{h}_{t} + z_{t} \odot h_{t-1} \tag{4}
\end{align}$$`

其中:

* 小圈表示哈达玛积，也就是两个向量逐位相乘
* (1) 式和 (2) 式计算的是更新门 `$u_t$` 和重置门 `$r_t$`，是两个长度和 `$h_t$` 相同的向量
* 注意到 (4) 式，实际上和 ResNet 的残差结构是相似的，都是 `$f(x) = x + g(x)$` 的形式，
  可以有效地防止长序列学习反向传播过程中梯度消失问题

### GRU 解释

* GRU 的结构比 LSTM 更为简单一些，GRU 只有两个门，更新门和重置门:
    - 更新门: 更新门用于控制每一步 `$h_t$` 被更新的比例，更新门越大，`$h_t$` 更新幅度越大
    - 重置门: 重置门用于控制更新候选向量 `$\tilde{h}_{t}$` 中前一步的状态 `$h_{t-1}$` 被重新放入的比例，
      重置门越大，更新候选向量中 `$h_{t-1}$` 被重新放进来的比例越大
* GRU 的参数数量为 LSTM 的 3/4

## 参考

- [LSTM与GRU的原理](https://zhuanlan.zhihu.com/p/184937263)

## PyTorch 示例

```python
import torch 
from torch import nn 

inputs = torch.randn(8,200,64) #batch_size, seq_length, features

gru = nn.GRU(
    input_size = 64,
    hidden_size = 32,
    num_layers = 1,
    batch_first = True
)
gru_output,gru_hn = gru(inputs)
print("--GRU--")
print("gru_output.shape:",gru_output.shape)
print("gru_hn.shape:",gru_hn.shape)
print("\n")


print("--LSTM--")
lstm = nn.LSTM(
    input_size = 64,
    hidden_size = 32,
    num_layers = 1,
    batch_first = True
)
lstm_output,(lstm_hn,lstm_cn) = lstm(inputs)
print("lstm_output.shape:",lstm_output.shape)
print("lstm_hn.shape:",lstm_hn.shape)
print("lstm_cn.shape:",lstm_cn.shape)


from torchkeras import summary
summary(gru,input_data=inputs);
summary(lstm,input_data=inputs);
```


# Transformer 相关层

## 模型层 API

* `nn.Transformer`: Transformer 网络结构
    - Transformer 网络结构是替代循环网络的一种结构，解决了循环网络难以并行，
      难以捕捉长期依赖的缺陷。它是目前 NLP 任务的主流模型的主要构成部分
* `nn.TransformerEncoder`: Transformer 编码器结构
    - 由多个 `nn.TransformerEncoderLayer` 编码器层组成
* `nn.TransformerDecoder`: Transformer 解码器结构
    - 由多个 `nn.TransformerDecoderLayer` 解码器层组成
* `nn.TransformerEncoderLayer`: Transformer 的编码器层
    - 主要由 Multi-Head self-Attention, Feed-Forward 前馈网络, LayerNorm 归一化层, 以及残差连接层组成
* `nn.TransformerDecoderLayer`: Transformer 的解码器层
    - 主要由 Masked Multi-Head self-Attention, Multi-Head cross-Attention, Feed-Forward 前馈网络, 
      LayerNorm 归一化层, 以及残差连接层组成
* `nn.MultiheadAttention`: 多头注意力层
    - 用于在序列方向上融合特征
    - 使用的是 Scaled Dot Production Attention，并引入了多个注意力头

## Transformer 结构解析

![img](images/Transformer.jpg)

## Transformer 数学表示

`$$\operatorname{Attention}(Q, K, V)=\operatorname{softmax}\left(\frac{Q K^{T}}{\sqrt{d_{k}}}\right) V$$`

`$$\begin{aligned}
\operatorname{MultiHead}(Q, K, V) &=\operatorname{Concat}\left(\operatorname{head}_{1}, \ldots, \text { head }_{\mathrm{h}}\right) W^{O} \\
\text { where }\, head_{i} &=\operatorname{Attention}\left(Q W_{i}^{Q}, K W_{i}^{K}, V W_{i}^{V}\right)
\end{aligned}$$`

## Transformer 要点问题

1. Transformer 是如何解决长距离依赖的问题的？
    - Transformer 是通过引入 Scale-Dot-Product 注意力机制来融合序列上不同位置的信息，从而解决长距离依赖问题。
      以文本数据为例，在循环神经网络 LSTM 结构中，输入序列上相距很远的两个单词无法直接发生交互，
      只能通过隐藏层输出或者细胞状态按照时间步骤一个一个向后进行传递。
      对于两个在序列上相距非常远的单词，中间经过的其它单词让隐藏层输出和细胞状态混入了太多的信息，
      很难有效地捕捉这种长距离依赖特征。但是在 Scale-Dot-Product 注意力机制中，
      序列上的每个单词都会和其它所有单词做一次点积计算注意力得分，
      这种注意力机制中单词之间的交互是强制的不受距离影响的，所以可以解决长距离依赖问题
2. Transformer 在训练和测试阶段可以在时间(序列)维度上进行并行吗？
    - 在训练阶段，Encoder 和 Decoder 在时间(序列)维度都是并行的；
      在测试阶段，Encoder 在序列维度是并行的，Decoder 是串行的
    - 首先，Encoder 部分在训练阶段和预测阶段都可以并行比较好理解，
      无论在训练还是预测阶段，它干的事情都是把已知的完整输入编码成 memory，
      在序列维度可以并行
    - 对于 Decoder 部分有些微妙。在预测阶段 Decoder 肯定是不能并行的，因为 Decoder 实际上是一个自回归，
      它前面 `$k-1$` 位置的输出会变成第 `$k$` 位的输入的。前面没有计算完，后面是拿不到输入的，肯定不可以并行。
      那么训练阶段能否并行呢？虽然训练阶段知道了全部的解码结果，但是训练阶段要和预测阶段一致啊，
      前面的解码输出不能受到后面解码结果的影响啊。
      但 Transformer 通过在 Decoder 中巧妙地引入 Mask 技巧，使得在用 Attention 机制做序列特征融合的时候，
      每个单词对位于它之后的单词的注意力得分都为 0，这样就保证了前面的解码输出不会受到后面解码结果的影响，
      因此 Decoder 在训练阶段可以在序列维度做并行
3. Scaled-Dot Product Attention 为什么要除以 `$\sqrt{d_k}$`?
    - 为了避免 `$d_k$` 变得很大时 softmax 函数的梯度趋于 0。
      假设 Q 和 K 中的取出的两个向量 `$q$` 和 `$k$` 的每个元素值都是正态随机分布，
      数学上可以证明两个独立的正态随机变量的积依然是一个正态随机变量，
      那么两个向量做点积，会得到 `$d_k$` 个正态随机变量的和，
      数学上 `$d_k$` 个正态随机变量的和依然是一个正态随机变量，
      其方差是原来的 `$d_k$` 倍，标准差是原来的 `$\sqrt{d_k}$` 倍。
      如果不做 scale, 当 `$d_k$` 很大时，求得的 `$QK^T$` 元素的绝对值容易很大，
      导致落在 softmax 的极端区域(趋于 0 或者 1)，极端区域 softmax 函数的梯度值趋于 0，
      不利于模型学习。除以 `$\sqrt{d_k}$`，恰好做了归一，不受 `$d_k$` 变化影响
4. MultiHeadAttention 的参数数量和 head 数量有何关系?
    - MultiHeadAttention 的参数数量和 head 数量无关。
      多头注意力的参数来自对 QKV 的三个变换矩阵以及多头结果 concat 后的输出变换矩阵。
      假设嵌入向量的长度是 d_model, 一共有 h 个 head. 对每个 head，
      `$W_{i}^{Q},W_{i}^{K},W_{i}^{V}$` 这三个变换矩阵的尺寸都是 `$d_model \times (d_model/h)$`，
      所以 h 个 head 总的参数数量就是 `$3 \times d_model \times (d_model/h) \times h = 3 \times d_model \times d_model$`。
      它们的输出向量长度都变成 `$d_model/h$`，经过 attention 作用后向量长度保持，
      h 个 head 的输出拼接到一起后向量长度还是 d_model，
      所以最后输出变换矩阵的尺寸是 `$d_model×d_model$`。
      因此，MultiHeadAttention 的参数数量为 `$4 \times d_model \times d_model$`，和 head 数量无关
5. Transformer 有什么缺点？
    - Transformer 主要的缺点有两个，一个是注意力机制相对序列长度的复杂度是 `$O(n^2)$`，第二个是对位置信息的
        - 第一，Transformer 在用 Attention 机制做序列特征融合的时候，
          每两个单词之间都要计算点积获得注意力得分，这个计算复杂度和序列的长度平方成正比，
          对于一些特别长的序列，可能存在着性能瓶颈，有一些针对这个问题的改进方案如 Linformer
        - 第二个是 Transformer 通过引入注意力机制两两位置做点乘来融合序列特征，
          而不是像循环神经网络那样由先到后地处理序列中的数据，导致丢失了单词之间的位置信息关系，
          通过在输入中引入正余弦函数构造的位置编码 PositionEncoding 一定程度上补充了位置信息，
          但还是不如循环神经网络那样自然和高效


## PyTorch 示例

```python
import torch 
from torch import nn 

#验证MultiheadAttention和head数量无关
inputs = torch.randn(8,200,64) #batch_size, seq_length, features

attention_h8 = nn.MultiheadAttention(
    embed_dim = 64,
    num_heads = 8,
    bias=True,
    batch_first=True
)

attention_h16 = nn.MultiheadAttention(
    embed_dim = 64,
    num_heads = 16,
    bias=True,
    batch_first=True
)


out_h8 = attention_h8(inputs,inputs,inputs)
out_h16 = attention_h16(inputs,inputs,inputs)

from torchkeras import summary 
summary(attention_h8,input_data_args=(inputs,inputs,inputs));

summary(attention_h16,input_data_args=(inputs,inputs,inputs));
```


```python
import torch 
from torch import nn 
from copy import deepcopy

#多头注意力的一种简洁实现

class ScaledDotProductAttention(nn.Module):
    "Compute 'Scaled Dot Product Attention'"
    def __init__(self):
        super(ScaledDotProductAttention, self).__init__()

    def forward(self,query, key, value, mask=None, dropout=None):
        d_k = query.size(-1)
        scores = query@key.transpose(-2,-1) / d_k**0.5     
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e20)
        p_attn = F.softmax(scores, dim = -1)
        if dropout is not None:
            p_attn = dropout(p_attn)
        return p_attn@value, p_attn
    
class MultiHeadAttention(nn.Module):
    def __init__(self, h, d_model, dropout=0.1):
        "Take in model size and number of heads."
        super(MultiHeadAttention, self).__init__()
        assert d_model % h == 0
        # We assume d_v always equals d_k
        self.d_k = d_model // h
        self.h = h
        self.linears = nn.ModuleList([deepcopy(nn.Linear(d_model, d_model)) for _ in range(4)])
        
        self.attn = None
        self.dropout = nn.Dropout(p=dropout)
        self.attention = ScaledDotProductAttention()
        
    def forward(self, query, key, value, mask=None):
        "Implements Figure 2"
        if mask is not None:
            # Same mask applied to all h heads.
            mask = mask.unsqueeze(1)
        nbatches = query.size(0)
        
        # 1) Do all the linear projections in batch from d_model => h x d_k 
        query, key, value = \
            [l(x).view(nbatches, -1, self.h, self.d_k).transpose(1, 2)
             for l, x in zip(self.linears, (query, key, value))]
        
        # 2) Apply attention on all the projected vectors in batch. 
        x, self.attn = self.attention(query, key, value, mask=mask, 
                                 dropout=self.dropout)
        
        # 3) "Concat" using a view and apply a final linear. 
        x = x.transpose(1, 2).contiguous() \
             .view(nbatches, -1, self.h * self.d_k)
        return self.linears[-1](x)
```

## 参考

- [Transformer知乎原理讲解](https://zhuanlan.zhihu.com/p/48508221)
- [Transformer哈佛博客代码讲解](http://nlp.seas.harvard.edu/annotated-transformer/)

# 自定义模型层

如果这些内置的模型层不能满足需求，也可以通过继承 `torch.nn.Module` 基类构建自定义的模型层。
实际上 PyTorch 不区分模型和模型层，都是通过继承 `torch.nn.Module` 进行构建，
因此，只要继承 `torch.nn.Module` 基类并实现 `forward` 方法即可自定义模型层

## PyTorch 的 torch.nn.Linear 层源码

可以仿照 `torch.nn.Linear` 自定义模型层

```python
import math

import torch
from torch import nn
import torch.nn.functional as F

class Linear(nn.Module):
    __constants__ = ["in_features", "out_features"]

    def __init__(self, in_features, out_features, bias = True):
        super(Linear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(
            torch.Tensor(out_features, in_features)
        )
        if bias:
            self.bias = nn.Parameter(torch.Tensor(out_features))
        else:
            self.register_parameter("bias", None)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.weight, a = math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = nn.init.calculate_fan_in_and_fan_out(
                self.weight
            )
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.bias, -bound, bound)
    
    def forward(self, input):
        return F.linear(input, self.weight, self.bias)

    def extra_repr(self):
        return f"in_features={self.in_features}, \
                 out_features={self.out_features}, \
                 bias={self.bias is not None}"
```

# functional 和 Module

PyTorch 和神经网络相关的功能组件大多都封装在 `torch.nn` 模块下，
这些功能组件的绝大部分既有函数形式实现，也有类形式实现

* torch.nn.functional
* torch.nn.Module

## functional

`torch.nn.functional` 有各种功能组件的函数实现:

```python
import torch.nn.functional as F
```

* 激活函数
    - `F.relu`
    - `F.sigmoid`
    - `F.tanh`
    - `F.softmax`
* 模型层
    - `F.linear`
    - `F.conv2d`
    - `F.max_pool2d`
    - `F.dropout2d`
    - `F.embedding`
* 损失函数
    - `F.binary_cross_entropy`
    - `F.mse_loss`
    - `F.cross_entropy`

示例：

```python
import torch
import torch.nn.functional as F

torch.relu(torch.tensor(-1.0))
F.relu(torch.tensor(-1.0))
```

```
tensor(0.)
tensor(0.)
```

## Module

为了便于对参数进行管理，一般通过继承 `torch.nn.Module` 转换称为类的实现形式，
并直接封装在 `torch.nn` 模块下

```python
from torch import nn
```

* 激活函数
    - `nn.ReLU`
    - `nn.Sigmoid`
    - `nn.Tanh`
    - `nn.Softmax`
* 模型层
    - `nn.Linear`
    - `nn.Conv2d`
    - `nn.MaxPool2d`
    - `nn.Dropout2d`
    - `nn.Embedding`
* 损失函数
    - `nn.BCELoss`
    - `nn.MSELoss`
    - `nn.CrossEntropyLoss`

实际上，`torch.nn.Module` 除了可以管理其引用的各种参数，
还可以管理其引用的子模块，功能十分强大

### 使用 Module 管理参数

在 PyTorch 中，模型的参数是需要被优化器训练的，
因此，通常要设置参数为 `requires_grad = True` 的张量。
同时，在一个模型中，往往有许多的参数，要手动管理这些参数并不是一件容易的事情

PyTorch 一般将参数用 `nn.Parameter` 来表示，
并且用 `nn.Module` 来管理其结构下的所有参数

* 载入 Python 依赖

```python
import torch
from torch import nn
import torch.nn.functional as F
```

* 设置参数为 `requires_grad = True` 的张量

```python
torch.randn(2, 2, requires_grad = True)
```

* `nn.Parameter()` 具有 `require_grad = True` 属性

```python
w = nn.Parameter(torch.randn(2, 2))
print(w)
print(w.requires_grad)
```

* `nn.ParameterList()` 可以将多个 `nn.Parameter()` 组成一个列表

```python
params_list = nn.ParameterList([
    nn.Parameter(torch.rand(8, i))
    for i in range(1, 3)
])
print(params_list)
print(params_list[0].requires_grad)
```

* `nn.ParameterDict()` 可以将多个 `nn.Parameter()` 组成一个字典

```python
params_dict = nn.ParameterDict({
    "a": nn.Parameter(torch.rand(2, 2)),
    "b": nn.Parameter(torch.zeros(2)),
})
print(params_dict)
print(params_dict["a"].requires_grad)
```

* 用 `nn.Module()` 将 `nn.Parameter`、`nn.ParameterList()`、`nn.ParameterDict()` 管理起来

```python
module = nn.Module()
module.w = nn.Parameter(
    torch.randn(2, 2)
)
module.params_list = nn.ParameterList([
    nn.Parameter(torch.rand(8, i))
    for i in range(1, 3)
])
module.param_dict = nn.ParameterDict({
    "a": nn.Parameter(torch.rand(2, 2)),
    "b": nn.Parameter(torch.zeros(2)),
})

num_param = 0
for param in module.named_parameters():
    print(param, "\n")
    num_param = num_param + 1
print(f"Number of Parameters = {num_param}")
```

* 实践当中，一般通过继承 `nn.Module` 来构建模块类，并将所有含有需要学习的部分放在构造函数中

```python
class Linear(nn.Module):
    __constants__ = ["in_features", "out_features"]

    def __init__(self, in_features, out_features, bias = True):
        super(Linear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(
            torch.Tensor(out_features, in_features)
        )
        if bias:
            self.bias = nn.Parameter(torch.Tensor(out_features))
        else:
            self.register_parameter("bias", None)
        
    def forward(self, input):
        return F.linear(input, self.weight, self.bias)
```


### 使用 Module 管理子模块

一般情况下，很少直接使用 `nn.Parameter` 来定义参数构建模型，而是通过一些拼装一些常用的模型层来构造模型。
这些模型层也是继承自 `nn.Module` 的对象，本身也包括参数，属于要定义的模块的子模块

`nn.Module` 提供了一些方法可以管理这些子模块:

* `children()`: 返回生成器，包括模块下的所有子模块
* `named_children()`: 返回一个生成器，包括模块下的所有子模块，以及它们的名字
* `modules()`: 返回一个生成器，包括模块下的所有各个层级的模块，包括模块本身
* `named_modules()`: 返回一个生成器，包括模块下的所有各个层级的模块以及它们的名字，包括模块本身

其中:

* `children()` 和 `named_children()` 方法较多使用
* `modules()` 和 `named_modules()` 方法较少使用，其功能可以通过多个 `named_children()` 的嵌套使用实现

```python
class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()

        self.embedding = nn.Embedding(
            num_embedding = 10000, 
            embedding_dim = 3, 
            padding_idx = 1
        )

        self.conv = nn.Sequential()
        self.conv.add_module(
            "conv_1", 
            nn.Conv1d(in_channels = 3, out_channels = 16, kernel_size = 5),
        )
        self.conv.add_module(
            "pool_1",
            nn.MaxPool1d(kernel_size = 2),
        )
        self.conv.add_module(
            "relu",
            nn.ReLU(),
        )
        self.conv.add_module(
            "conv_2",
            nn.Conv1d(in_channels = 16, out_channels = 128, kernel_size = 2),
        )
        self.conv.add_module(
            "pool_2",
            nn.MaxPool1d(kernel_size = 2),
        )
        self.conv.add_module(
            "relu_2",
            nn.ReLU(),
        )

        self.dense = nn.Sequential()
        self.dense.add_module("flatten", nn.Flatten())
        self.dense.add_module("linear", nn.Linear(6144, 1))
    
    def forward(self, x):
        x = self.embedding(x).transpose(1, 2)
        x = self.conv(x)
        y = self.dense(x)
        return y

net = Net()
```

```python
i = 0
for child in net.children():
    i += 1
    print(child, "\n")
print("child number", i)
```

```
Embedding(10000, 3, padding_idx=1) 

Sequential(
  (conv_1): Conv1d(3, 16, kernel_size=(5,), stride=(1,))
  (pool_1): MaxPool1d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
  (relu_1): ReLU()
  (conv_2): Conv1d(16, 128, kernel_size=(2,), stride=(1,))
  (pool_2): MaxPool1d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
  (relu_2): ReLU()
) 

Sequential(
  (flatten): Flatten(start_dim=1, end_dim=-1)
  (linear): Linear(in_features=6144, out_features=1, bias=True)
) 

child number 3
```

```python
i = 0
for name, child in net.named_children():
    i += 1
    print(name, ":", child, "\n")
print("child number", i)
```

```
embedding : Embedding(10000, 3, padding_idx=1) 

conv : Sequential(
  (conv_1): Conv1d(3, 16, kernel_size=(5,), stride=(1,))
  (pool_1): MaxPool1d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
  (relu_1): ReLU()
  (conv_2): Conv1d(16, 128, kernel_size=(2,), stride=(1,))
  (pool_2): MaxPool1d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
  (relu_2): ReLU()
) 

dense : Sequential(
  (flatten): Flatten(start_dim=1, end_dim=-1)
  (linear): Linear(in_features=6144, out_features=1, bias=True)
) 

child number 3
```

```python
i = 0
for module in net.modules():
    i += 1
    print(module)
print("module number:", i)
```

```
Net(
  (embedding): Embedding(10000, 3, padding_idx=1)
  (conv): Sequential(
    (conv_1): Conv1d(3, 16, kernel_size=(5,), stride=(1,))
    (pool_1): MaxPool1d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
    (relu_1): ReLU()
    (conv_2): Conv1d(16, 128, kernel_size=(2,), stride=(1,))
    (pool_2): MaxPool1d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
    (relu_2): ReLU()
  )
  (dense): Sequential(
    (flatten): Flatten(start_dim=1, end_dim=-1)
    (linear): Linear(in_features=6144, out_features=1, bias=True)
  )
)
Embedding(10000, 3, padding_idx=1)
Sequential(
  (conv_1): Conv1d(3, 16, kernel_size=(5,), stride=(1,))
  (pool_1): MaxPool1d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
  (relu_1): ReLU()
  (conv_2): Conv1d(16, 128, kernel_size=(2,), stride=(1,))
  (pool_2): MaxPool1d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
  (relu_2): ReLU()
)
Conv1d(3, 16, kernel_size=(5,), stride=(1,))
MaxPool1d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
ReLU()
Conv1d(16, 128, kernel_size=(2,), stride=(1,))
MaxPool1d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
ReLU()
Sequential(
  (flatten): Flatten(start_dim=1, end_dim=-1)
  (linear): Linear(in_features=6144, out_features=1, bias=True)
)
Flatten(start_dim=1, end_dim=-1)
Linear(in_features=6144, out_features=1, bias=True)
module number: 12
```

* 通过 `named_children` 方法找到 `embedding` 层，并将其参数设置为不可训练，相当于冻结 embedding 层

```python
children_dict = {
    name: module for name, module in net.named_children()
}
print(children_dict)

embedding = children_dict["embedding"]
embedding.requires_grad_(False)
```

```
{'embedding': Embedding(10000, 3, padding_idx=1), 'conv': Sequential(
  (conv_1): Conv1d(3, 16, kernel_size=(5,), stride=(1,))
  (pool_1): MaxPool1d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
  (relu_1): ReLU()
  (conv_2): Conv1d(16, 128, kernel_size=(2,), stride=(1,))
  (pool_2): MaxPool1d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
  (relu_2): ReLU()
), 'dense': Sequential(
  (flatten): Flatten(start_dim=1, end_dim=-1)
  (linear): Linear(in_features=6144, out_features=1, bias=True)
)}
Embedding(10000, 3, padding_idx=1)
```

```python
# 第一层的参数已经不可以被训练
for param in embedding.parameters():
    print(param.requires_grad)
    print(param.numel())
```

```
False
30000
```

## 参考

* [Differentiable Learning-To-Normalize Via Switchable Normalization](https://arxiv.org/pdf/1806.10779.pdf)