---
title: TensorFlow Keras 网络层
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

- [自定义层](#自定义层)
- [Keras Layers 共有的方法](#keras-layers-共有的方法)
  - [Keras Layers](#keras-layers)
- [Keras Layers 配置](#keras-layers-配置)
  - [Activation Function](#activation-function)
  - [可用的 activations](#可用的-activations)
  - [Keras 参数初始化(Initializers)](#keras-参数初始化initializers)
  - [Keras 正则化(Regularizers)](#keras-正则化regularizers)
  - [Keras 约束(Constraints)](#keras-约束constraints)
</p></details><p></p>

# 自定义层

- 自定义层需要继承 `tf.keras.layers.Layers` 类, 并重写 `__init__`、`build`、`call` 三个方法

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

- 线性层示例

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

# Keras Layers 共有的方法

```python
from keras import layers
```

- layer.get_weights()
- layer.set_weights(weights)
- layer.get_config()
   - keras.layer.Dense.from_config(config)
   - keras.layer.deserialize({"class_name": , "config": config})
- 如果 Layer 是单个节点(不是共享 layer), 可以使用以下方式获取 layer
   的属性:
   - layer.input
   - layer.output
   - layer.input_shape
   - layer.output_shape
- 如果 Layer 具有多个节点(共享 layer), 可以使用以下方式获取 layer
   的属性:
   - layer.getinputat(note_index)
   - layer.getoutputat(note_index)
   - layer.getinputshapeat(noteindex)
   - layer.getoutputshaepat(noteindex)

## Keras Layers

- **Core Layers**
   - Dense
   - Activation
   - Drop
   - Flatten
   - Input
   - Reshape
      - `keras.layers.Reshape(target_shape)`
   - Permute
   - RepeatVector
   - Lambda
   - ActivityRegularization
   - Masking
   - SpatialDropout1D
   - SpatialDropout2D
   - SpatialDropout3D`
- **Convolutional Layers**
   - 卷积层
      - Conv1D
      - Conv2D
      - Conv3D
      - SeparableConv1D
         - `keras.layers.SeparableConv1D(rate)`
      - SeparableConv2D
      - DepthwiseConv3D
   - Transpose
      - Conv2DTranspose
      - Conv3DTranspose
   - Cropping
      - Cropping1D
      - Cropping2D
      - Cropping3D
   - UnSampling
      - UnSampling1D
      - UnSampling2D
      - UnSampling3D
   - ZeroPadding
      - ZeroPadding1D
      - ZeroPadding2D
      - ZeroPadding3D
- **Pooling Layers**
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
- **Locally-connected Layers**
   - `LocallyConnected1D()`
   - `LocallyConnected2D()`
- **Recurrent Layers**
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
- **Embedding Layers**
   - `Embedding()`
- **Merge Layers**

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
- **Advanced Activations Layers**
   - `LeakyReLU()`
   - `PReLU()`
   - `ELU()`
   - `ThresholdedReLU()`
   - `Softmax()`
   - `ReLU()`
   - Activation Functions
- **Normalization Layers**
   - `BatchNormalization()`
- **Nosise Layers**
   - `GaussianNoise()`
   - `GaussianDropout()`
   - `AlphaDropout()`
- **Others**
   - Layer wrapper
      - `TimeDistributed()`
      - `Bidirectional()`
   - Writting Customilize Keras Layers
      - `build(input_shape)`
      - `call(x)`
      - `compute_output_shape(input_shape)`


# Keras Layers 配置

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

## Activation Function

- Keras Activations
   - `Activation` layer
   - `activation` argument supported by all forward layers

**调用方法:**

```python
from keras.layers import Activation, Dense
from keras import backend as K

# method 1
model.add(Dense(64))
model.add(Activation("tanh"))

# method 2
model.add(Dense(64, activation = "tanh"))

# method 3
model.add(Dense(64, activation = K.tanh))
```

## 可用的 activations

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


## Keras 参数初始化(Initializers)

**Initializers 的使用方法:**

   初始化定义了设置 Keras Layer 权重随机初始的方法

- `kernel_initializer` param

   - "random_uniform"

- `bias_initializer` param

**可用的 Initializers:**

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
- keras.initializers.RandomNormal(mean = 0.0, stddev = 0.05, seed =
   None)
   - 正态分布
- keras.initializers.RandomUniform(minval = 0.05, maxval = 0.05, seed =
   None)
   - 均匀分布
- keras.initializers.TruncatedNormal(mean = 0.0, stddev = 0.05, seed =
   None)
   - 截尾正态分布:生成的随机值与 `RandomNormal`
      生成的类似, 但是在距离平均值两个标准差之外的随机值将被丢弃并重新生成。这是用来生成神经网络权重和滤波器的推荐初始化器
- keras.initializers.VarianveScaling(scale = 1.0, mode = "fan_in",
   distribution = "normal", seed = None)
   - 根据权值的尺寸调整其规模
- keras.initializers.Orthogonal(gain = 1.0, seed = None)
   - `随机正交矩阵 <http://arxiv.org/abs/1312.6120>`__
- keras.initializers.Identity(gain = 1.0)
   - 生成单位矩阵的初始化器。仅用于 2D 方阵
- keras.initializers.lecun_normal()
   - LeCun 正态分布初始化器
   - 它从以 0 为中心, 标准差为 stddev = sqrt(1 / fanin)
      的截断正态分布中抽取样本,  其中 fanin
      是权值张量中的输入单位的数量
- keras.initializers.lecun_uniform()
   - LeCun 均匀初始化器
   - 它从 [-limit, limit] 中的均匀分布中抽取样本,  其中 limit 是 sqrt(3
      / fanin),  fanin 是权值张量中的输入单位的数量
- keras.initializers.glorot_normal()
   - Glorot 正态分布初始化器, 也称为 Xavier 正态分布初始化器
   - 它从以 0 为中心, 标准差为 stddev = sqrt(2 / (fan*in + fanout))
      的截断正态分布中抽取样本,  其中 fanin
      是权值张量中的输入单位的数量,  fanout
      是权值张量中的输出单位的数量
- keras.initializers.glorot_uniform()
   - Glorot 均匀分布初始化器, 也称为 Xavier 均匀分布初始化器
   - 它从 [-limit, limit] 中的均匀分布中抽取样本,  其中 limit 是 sqrt(6
      / (fan*in + fanout)),  fanin 是权值张量中的输入单位的数量, 
      fanout 是权值张量中的输出单位的数量
- keras.initializers.he_normal()
   - He 正态分布初始化器
   - 它从以 0 为中心, 标准差为 stddev = sqrt(2 / fanin)
      的截断正态分布中抽取样本,  其中 fanin
      是权值张量中的输入单位的数量
- keras.initializers.he_uniform()
   - He 均匀分布方差缩放初始化器
   - 它从 :math:`[-limit, limit]` 中的均匀分布中抽取样本,  其中
      :math:`limit` 是 :math:`sqrt(6 / fan_in)`\ ,  其中 fan_in
      是权值张量中的输入单位的数量
- 自定义 Initializer

```python
from keras import backend as K

def my_init(shape, dtype = None):
      return K.random_normal(shape, dtype = dtype)

model.add(Dense(64, kernel_initializer = my_init))
```

## Keras 正则化(Regularizers)

正则化器允许在优化过程中对\ `层的参数`\ 或\ `层的激活函数`\ 情况进行惩罚, 并且神经网络优化的损失函数的惩罚项也可以使用

惩罚是以层为对象进行的。具体的 API 因层而异, 但 Dense, Conv1D, Conv2D 和
Conv3D 这些层具有统一的 API

**Regularizers 的使用方法:**

- [class] keras.regularizers.Regularizer
   - [instance] `kernel_regularizer` param
   - [instance] `bias_regularizer` param
   - [instance] `activity_regularizer` param

**可用的 Regularizers:**

- keras.regularizers.l1(0.)
- keras.regularizers.l2(0.)
- keras.regularizers.l1_l2(l1 = 0.01, l2 = 0.01)
- 自定义的 Regularizer:
   - `def l1_reg: pass`

## Keras 约束(Constraints)

`constraints` 模块的函数允许在优化期间对网络参数设置约束(例如非负性)。

约束是以层为对象进行的。具体的 API 因层而异, 但 Dense, Conv1D, Conv2D 和
Conv3D 这些层具有统一的 API

**Constraints 的使用方法:**

- kernel_constraint
- bias_constraint

**可用的 Constraints:**

- keras.constraints.MaxNorm(max_value = 2, axis = 0)
   - 最大范数权值约束
- keras.constraints.NonNeg()
   - 权重非负的约束
- keras.constraints.UnitNorm()
   - 映射到每个隐藏单元的权值的约束, 使其具有单位范数
- keras.constraints.MinMaxNorm(minvalue = 0, maxvalue = 1.0, rate
   = 1.0, axis = 0)
   - 最小/最大范数权值约束:映射到每个隐藏单元的权值的约束, 使其范数在上下界之间


