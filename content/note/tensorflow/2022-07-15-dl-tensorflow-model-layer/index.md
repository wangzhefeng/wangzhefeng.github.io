---
title: TensorFlow 网络层
author: 王哲峰
date: '2022-07-15'
slug: dl-tensorflow-model-layer
categories:
  - deeplearning
  - tensorflow
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

- [内置模型层](#内置模型层)
  - [模型层共有的方法](#模型层共有的方法)
  - [常用内置模型层](#常用内置模型层)
    - [核心层](#核心层)
    - [Convolutional Layers](#convolutional-layers)
    - [Pooling Layers](#pooling-layers)
    - [Locally-connected Layers](#locally-connected-layers)
    - [Recurrent Layers](#recurrent-layers)
    - [Embedding Layers](#embedding-layers)
    - [Merge Layers](#merge-layers)
    - [Advanced Activations Layers](#advanced-activations-layers)
    - [Normalization Layers](#normalization-layers)
    - [Nosise Layers](#nosise-layers)
    - [Others Layers](#others-layers)
- [自定义模型层](#自定义模型层)
  - [编写 tf.keras.Lambda 层](#编写-tfkeraslambda-层)
  - [继承 tf.keras.layers.Layer 基类](#继承-tfkeraslayerslayer-基类)
  - [线性层示例](#线性层示例)
- [模型层配置](#模型层配置)
  - [输入](#输入)
  - [输出](#输出)
  - [激活函数](#激活函数)
    - [模型可用的激活函数](#模型可用的激活函数)
    - [在模型中使用激活函数](#在模型中使用激活函数)
  - [参数初始化](#参数初始化)
    - [Initializers 的使用方法](#initializers-的使用方法)
    - [常数初始化器](#常数初始化器)
    - [分布初始化器](#分布初始化器)
    - [矩阵初始化器](#矩阵初始化器)
    - [LeCun 分布初始化器](#lecun-分布初始化器)
    - [Glorot 分布初始化器](#glorot-分布初始化器)
    - [He 正态分布和均匀分布初始化器](#he-正态分布和均匀分布初始化器)
    - [自定义初始化器](#自定义初始化器)
  - [正则化](#正则化)
    - [Regularizers 的使用方法](#regularizers-的使用方法)
    - [可用的 Regularizers](#可用的-regularizers)
    - [自定义的 Regularizer](#自定义的-regularizer)
  - [约束](#约束)
    - [Constraints 的使用方法](#constraints-的使用方法)
    - [可用的 Constraints](#可用的-constraints)
</p></details><p></p>

深度学习模型一般由各种模型层组合而成，`tf.keras.layers` 内置了非常丰富的各种功能的模型层，
如果这些内置模型层不能够满足需求，
可以通过编写 `tf.keras.Lambda` 匿名模型层或继承 `tf.keras.layers.Layer` 基类构建自定义的模型层，
其中，`tf.keras.Lambda` 匿名模型层只适用于构造没有学习参数的模型层

# 内置模型层

## 模型层共有的方法

- layer`.get_weights()`
- layer`.set_weights(weights)`
- layer`.get_config()`
   - `tf.keras.layer.Dense.from_config(config)`
   - `tf.keras.layer.deserialize({"class_name": , "config": config})`
- 如果 Layer 是单个节点(不是共享 layer), 可以使用以下方式获取 layer 的属性:
   - layer.`input`
   - layer.`output`
   - layer.`input_shape`
   - layer.`output_shape`
- 如果 Layer 具有多个节点(共享 layer), 可以使用以下方式获取 layer 的属性:
   - layer.`getinputat(note_index)`
   - layer.`getoutputat(note_index)`
   - layer.`getinputshapeat(noteindex)`
   - layer.`getoutputshaepat(noteindex)`

## 常用内置模型层

### 核心层

- `Dense`
- `Drop`
- `Flatten`
- `Input`
- `Reshape`
- `Permute`
- `RepeatVector`
- `Lambda`
- `Masking`
- `SpatialDropout1D`
- `SpatialDropout2D`
- `SpatialDropout3D`
- `Activation`
- `ActivityRegularization`

### Convolutional Layers

- 卷积层
    - `Conv1D`
    - `Conv2D`
    - `Conv3D`
    - `SeparableConv1D`
    - `SeparableConv2D`
    - `DepthwiseConv3D`
- Transpose
    - `Conv2DTranspose`
    - `Conv3DTranspose`
- Cropping
    - `Cropping1D`
    - `Cropping2D`
    - `Cropping3D`
- UnSampling
    - `UnSampling1D`
    - `UnSampling2D`
    - `UnSampling3D`
- ZeroPadding
    - `ZeroPadding1D`
    - `ZeroPadding2D`
    - `ZeroPadding3D`

### Pooling Layers

- 最大池化
    - `MaxPolling1D()`
    - `MaxPolling2D()`
    - `MaxPolling3D()`
    - `GlobalMaxPolling1D()`
    - `GlobalMaxPolling2D()`
    - `GlobalMaxPolling3D()`
- 平均池化
    - `AveragePolling1D()`
    - `AveragePolling2D()`
    - `AveragePolling3D()`
    - `GlobalAveragePolling1D()`
    - `GlobalAveragePolling2D()`
    - `GlobalAveragePolling3D()`

### Locally-connected Layers

- `LocallyConnected1D()`
- `LocallyConnected2D()`

### Recurrent Layers

- RNN
    - `RNN()`
    - `SimpleRNN()`
    - `SimpleRNNCell()`
- GRU
    - `GRU()`
    - `GRUCell()`
- LSTM
    - `LSTM()`
    - `LSTMCell()`
    - `ConvLSTM2D()`
    - `ConvLSTM2DCell()`
- CuDNN
    - `CuDNNGRU()`
    - `CuDNNLSTM()`

### Embedding Layers

- `Embedding()`

### Merge Layers

- `Add()`
- `Subtract()`
- `Multiply()`
- `Average()`
- `Maximum()`
- `Minimum()`
- `Concatenate()`
- `Dot()`


- `add()`
- `subtract()`
- `multiply()`
- `average()`
- `maximum()`
- `minimum()`
- `concatenate()`
- `dot()`
  
### Advanced Activations Layers

- `LeakyReLU()`
- `PReLU()`
- `ELU()`
- `ThresholdedReLU()`
- `Softmax()`
- `ReLU()`
- Activation Functions

### Normalization Layers

- `BatchNormalization()`

### Nosise Layers

- `GaussianNoise()`
- `GaussianDropout()`
- `AlphaDropout()`

### Others Layers

- Layer wrapper
    - `TimeDistributed()`
    - `Bidirectional()`
- Writting Customilize Keras Layers
    - `build(input_shape)`
    - `call(x)`
    - `compute_output_shape(input_shape)`

# 自定义模型层

## 编写 tf.keras.Lambda 层

## 继承 tf.keras.layers.Layer 基类

自定义层需要继承 `tf.keras.layers.Layers` 类, 并重写 `__init__`、`build`、`call` 三个方法

```python
import numpy as np
import tensorflow as tf

class MyLayer(tf.keras.layers.Layer):
    def __init__(self):
        super().__init__()
        # 初始化代码
    
    def build(self, input_shape): # input_shape 是一个 TensorShape 类型对象, 提供输入的形状
        # 在第一次使用该层的时候调用该部分代码, 在这里创建变量可以使得变量的形状自适应输入的形状
        # 而不需要使用者额外指定变量形状
        # 如果已经可以完全确定变量的形状, 也可以在 __init__ 部分创建变量
        self.variable_0 = self.add_weight(...)
        self.variable_1 = self.add_weight(...)
    
    def call(self, inputs):
        # 模型调用的代码(处理输入并返回输出)
        return output
```

## 线性层示例

```python
import numpy as np
import tensorflow as tf

class LinearLayer(tf.keras.layers.Layer):
    def __init__(self, units):
        super.__init__()
        self.units = units
    
    def build(self, input_shape):
        self.w = self.add_variable(
            name = "w", 
            shape = [input_shape[-1], self.units],  # [n, 1]
            initializer = tf.zeros_initializer()
        )
        self.b = self.add_variable(
            name = "b",
            shape = [self.units],                   # [1]
            initializer = tf.zeros_initializer()
        )
    
    def call(self, inputs):
        y_pred = tf.matmul(inputs, self.w) + self.b
        return y_pred

class LinearModel(tf.keras.Model):
    def __init__(self):
        super().__init__()
        self.layer = LinearLayer(untis = 1)
    
    def call(self, inputs):
        output = self.layer(inputs)
        return output
```

# 模型层配置

```python
model.add(Layer(
    # 输出、输出
    output_dim,
    input_dim,
    # 参数初始化
    kernel_initializer,
    bias_initializer,
    # 参数正则化
    kernel_regularizer,
    activity_regularizer,
    # 参数约束
    kernel_constraint,
    bias_constraint,
    # 层激活函数
    activation,
))

# 输出
input_s = Input()

# 激活函数
model.add(Activation)
```

## 输入


## 输出


## 激活函数

### 模型可用的激活函数

- softmax: Softmax activation function
   - x =>
   - `keras.activatons.softmax(x, axis = 1)`
- relu: Rectified Linear Unit
   - x => max(x, 0)
   - `keras.activations.relu(x, alpha = 0.0, max_value = None, threshold = 0.0)`
- tanh: Hyperbolic tangent activation function
   - `keras.activations.tanh(x)`
- sigmoid: Sigmoid activation function
   - x => 1/(1 + exp(-x))
   - `keras.activations.sigmoid(x)`
- linear: Linear activation function
   - x => x
   - `keras.activations.linear(x)`

### 在模型中使用激活函数

在 TensorFlow Keras 模型中使用激活函数一般有两种方式

* 作为某系层的 `activation` 参数指定
* 显式添加 `tf.keras.layers.Activation` 激活层

```python
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import layers, models

tf.keras.backend.clear_session()

model = models.Sequential()
model.add(layers.Dense(32, input_shape = (None, 16), activation = tf.nn.relu))
model.add(layers.Dense(10))
model.add(layers.Activation(tf.nn.softmax))
model.summary()
```

## 参数初始化

### Initializers 的使用方法

初始化定义了设置 Keras Layer 权重随机初始的方法

- `kernel_initializer` param
   - "random_uniform"
- `bias_initializer` param

### 常数初始化器

- keras.initializers.Initializer()
    - 基类
- keras.initializers.Zeros()
    - `0`
- keras.initializers.Ones()
    - `1`
- keras.initializers.Constant()
    - keras.initializers.Constant(value = 0)
        - `0`
    - keras.initializers.Constant(value = 1)
        - `1`

### 分布初始化器

`tf.keras.initializers.RandomNormal(mean = 0.0, stddev = 0.05, seed = None)`
   - 正态分布
`tf.keras.initializers.RandomUniform(minval = 0.05, maxval = 0.05, seed = None)`
   - 均匀分布
`tf.keras.initializers.TruncatedNormal(mean = 0.0, stddev = 0.05, seed = None)`
    - 截尾正态分布:生成的随机值与 `RandomNormal` 生成的类似, 
      但是在距离平均值两个标准差之外的随机值将被丢弃并重新生成。
      这是用来生成神经网络权重和滤波器的推荐初始化器

### 矩阵初始化器

`tf.keras.initializers.VarianveScaling(scale = 1.0, mode = "fan_in", distribution = "normal", seed = None)`
    - 根据权值的尺寸调整其规模
`tf.keras.initializers.Orthogonal(gain = 1.0, seed = None)`
    - [随机正交矩阵](http://arxiv.org/abs/1312.6120)
`tf.keras.initializers.Identity(gain = 1.0)`
    - 生成单位矩阵的初始化器。仅用于 2D 方阵

### LeCun 分布初始化器

`tf.keras.initializers.lecun_normal()`

- LeCun 正态分布初始化器
- 它从以 0 为中心, 标准差为 `stddev = \sqrt{\frac{1}{fanin}}` 的截断正态分布中抽取样本,  
  其中 `fanin` 是权值张量中的输入单位的数量

`tf.keras.initializers.lecun_uniform()`

- LeCun 均匀初始化器
- 它从 `$[-limit, limit]$` 中的均匀分布中抽取样本,  其中 limit 是 `$\sqrt{\frac{3}{fanin}}$`，
  其中 `fanin` 是权值张量中的输入单位的数量

### Glorot 分布初始化器

`tf.keras.initializers.glorot_normal()`

- Glorot 正态分布初始化器, 也称为 Xavier 正态分布初始化器
- 它从以 0 为中心, 标准差为 `$stddev = \sqrt{{2}{fanin + fanout}}$` 的截断正态分布中抽取样本, 
  其中 `fanin` 是权值张量中的输入单位的数量, `fanout` 是权值张量中的输出单位的数量


`tf.keras.initializers.glorot_uniform()`

- Glorot 均匀分布初始化器, 也称为 Xavier 均匀分布初始化器
- 它从 `$[-limit, limit]$` 中的均匀分布中抽取样本,  其中 limit 是 `$\sqrt{\frac{6}{fanin + fanout}}$`， 
  `fanin` 是权值张量中的输入单位的数量, `fanout` 是权值张量中的输出单位的数量

### He 正态分布和均匀分布初始化器

`tf.keras.initializers.he_normal()`: He 正态分布初始化器

* 从以 0 为中心, 标准差为 `$stddev = \sqrt{\frac{2}{fanin}}$` 的截断正态分布中抽取样本, 
  其中 `fanin` 是权值张量中的输入单位的数量

`tf.keras.initializers.he_uniform()`: He 均匀分布方差缩放初始化器

* 它从 `$[-limit, limit]$` 中的均匀分布中抽取样本,  其中 `limit` 是 `$\sqrt{\frac{6}{fan\_in}}$`，
 其中 `fan_in` 是权值张量中的输入单位的数量

### 自定义初始化器


## 正则化

正则化器允许在优化过程中对层的参数或层的激活函数情况进行惩罚, 
并且神经网络优化的损失函数的惩罚项也可以使用

惩罚是以层为对象进行的。具体的 API 因层而异, 但 Dense, Conv1D, Conv2D 和
Conv3D 这些层具有统一的 API

### Regularizers 的使用方法

- [class] keras.regularizers.Regularizer
   - [instance] `kernel_regularizer` param
   - [instance] `bias_regularizer` param
   - [instance] `activity_regularizer` param

### 可用的 Regularizers

- keras.regularizers.l1(0.)
- keras.regularizers.l2(0.)
- keras.regularizers.l1_l2(l1 = 0.01, l2 = 0.01)



### 自定义的 Regularizer


## 约束

`constraints` 模块的函数允许在优化期间对网络参数设置约束(例如非负性)

约束是以层为对象进行的。具体的 API 因层而异, 但 Dense, Conv1D, Conv2D 和
Conv3D 这些层具有统一的 API

### Constraints 的使用方法

* kernel_constraint
* bias_constraint

### 可用的 Constraints

* `tf.keras.constraints.MaxNorm(max_value = 2, axis = 0)`
    - 最大范数权值约束
* `tf.keras.constraints.NonNeg()`
    - 权重非负的约束
* `tf.keras.constraints.UnitNorm()`
    - 映射到每个隐藏单元的权值的约束, 使其具有单位范数
* `tf.keras.constraints.MinMaxNorm(minvalue = 0, maxvalue = 1.0, rate = 1.0, axis = 0)`
    - 最小/最大范数权值约束: 映射到每个隐藏单元的权值的约束, 使其范数在上下界之间

