---
title: TensorFlow 网络层
author: wangzf
date: '2022-07-15'
slug: dl-tensorflow-model-layer
categories:
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
img {
    pointer-events: none;
}
</style>

<details><summary>目录</summary><p>

- [内置模型层](#内置模型层)
  - [模型层共有的方法](#模型层共有的方法)
    - [权重设置与获取](#权重设置与获取)
    - [层属性获取](#层属性获取)
  - [常用内置模型层](#常用内置模型层)
    - [常用核心层](#常用核心层)
    - [卷积网络相关层](#卷积网络相关层)
    - [循环网络相关层](#循环网络相关层)
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
  - [正则化](#正则化)
    - [Regularizers 的使用方法](#regularizers-的使用方法)
    - [可用的 Regularizers](#可用的-regularizers)
    - [自定义的 Regularizer](#自定义的-regularizer)
  - [约束](#约束)
    - [Constraints 的使用方法](#constraints-的使用方法)
    - [可用的 Constraints](#可用的-constraints)
</p></details><p></p>

深度学习模型一般由各种模型层组合而成，`tf.keras.layers` 内置了非常丰富的各种功能的模型层

如果这些内置模型层不能够满足需求，可以通过编写 `tf.keras.Lambda` 匿名模型层，
或继承 `tf.keras.layers.Layer` 基类构建自定义的模型层

# 内置模型层

## 模型层共有的方法

### 权重设置与获取

- `.get_weights()`
- `.set_weights(weights)`
- `.get_config()`
    - `tf.keras.layer.Dense.from_config(config)`
    - `tf.keras.layer.deserialize({"class_name": class_name, "config": config})`

### 层属性获取

- 如果 Layer 是单个节点(不是共享 layer), 可以使用以下方式获取 layer 的属性:
    - `.input`
    - `.output`
    - `.input_shape`
    - `.output_shape`
- 如果 Layer 具有多个节点(共享 layer), 可以使用以下方式获取 layer 的属性:
    - `.getinputat(note_index)`
    - `.getoutputat(note_index)`
    - `.getinputshapeat(noteindex)`
    - `.getoutputshaepat(noteindex)`

## 常用内置模型层

### 常用核心层

* `Dense()`: 密集连接层
    - 参数个数 = 输入层特征数 `$\times$` 输出层特征数(weight) + 输出层特征数(bias)
* `Input()`: 输入层
    - 通常使用 Functional API 方式构建模型时作为第一层
* `Flatten()`: 压平层
    - 用于将多维张量压成一维
* `Reshape()`: 形状重塑层
    - 改变输入张量的形状
* `Activation()`: 激活函数层
    - 一般放在 `Dense` 层后面，等价于在 `Dense` 层中指定 activation
* 正则化层
    -  `Dropout()`: 随机置零层
        - 训练期间以一定概率将输入置为 0，一种正则化手段
    - `SpatialDropout2D()`: 空间随机置零层
        - 训练期间以一定概率将整个特征图置 0，一种正则化手段
        - 有利于避免特征图之间过高的相关性
* `DenseFeature()`: 特征列接入层
    - 用于接收一个特征列列表并产生一个密集链接层
* 合并层
    - `Add()`: 加法层
    - `Subtract()`: 减法层
    - `Multiply()`: 乘法层
    - `Average()`: 取平均层
    - `Maximum()`: 取最大值层
    - `Minimum()`: 取最小值层
    - `Concatenate()`: 拼接层
        - 将多个张量在某个维度上拼接
    - `Dot()`: 点积层
* 标准化层
    - `BatchNormalization()`: 批标准化层
        - 通过线性变换将输入批次缩放平移到稳定的均值和标准差
        - 可以增强模型对输入不同分布的适应性，加快模型训练速度，有轻微正则化效果
        - 一般在激活函数之前使用
* 高级激活函数层
    - `LeakyReLU()`
    - `PReLU()`
    - `ELU()`
    - `ThresholdedReLU()`
    - `Softmax()`
    - `ReLU()`
* 噪声层
    - `GaussianNoise()`
    - `GaussianDropout()`
    - `AlphaDropout()`
* 自定义层
    - `Lambda()`: 编写 `tf.keras.Lambda` 层
    - 继承 `tf.keras.layers.Layer` 基类自定义层
        - `build(input_shape)`
        - `call(x)`
        - `compute_output_shape(input_shape)`
* 其他层
    - `Permute()`
    - `RepeatVector()`
    - `Masking()`

### 卷积网络相关层

* 卷积层
    - `Conv1D`: 普通一维卷积
        - 常用于文本，参数个数 = 输入通道数 `$\times$` 卷积核尺寸(如 3) `$\times$` 卷积核个数
    - `Conv2D`: 普通二维卷积
        - 常用于图像，参数个数 = 输入通道数 `$\times$` 卷积核尺寸(如 3 `$\times$` 3) `$\times$` 卷积核个数
    - `Conv3D`: 普通三维卷积
        - 常用于视频，参数个数 = 输入通道数 `$\times$` 卷积核尺寸(如 3 `$\times$` 3 `$\times$` 3) `$\times$` 卷积核个数
    - `SeparableConv2D`: 二维深度可分离卷积层
        - 不同于普通卷积同时对区域和通道操作，深度可分离卷积先操作区域，
          再操作通道。即先对每个通道做独立卷积操作区域，再用1乘1卷积跨通道组合操作通道。
        - 参数个数 = 输入通道数×卷积核尺寸 + 输入通道数×1×1×输出通道数
        - 深度可分离卷积的参数数量一般远小于普通卷积，效果一般也更好
    - `DepthwiseConv2D`: 二维深度卷积层
        - 仅有 `SeparableConv2D` 前半部分操作，即只操作区域，不操作通道，
          一般输出通道数和输入通道数相同，但也可以通过设置 `depth_multiplier` 让输出通道为输入通道的若干倍数
        - 输出通道数 = 输入通道数 × depth_multiplier
        - 参数个数 = 输入通道数 × 卷积核尺寸 × depth_multiplier
* Transpose
    - `Conv2DTranspose`: 二维卷积转置层
        - 俗称反卷积层。并非卷积的逆操作，但在卷积核相同的情况下，
          当其输入尺寸是卷积操作输出尺寸的情况下，
          卷积转置的输出尺寸恰好是卷积操作的输入尺寸
    - `Conv3DTranspose`
* Cropping
    - `Cropping1D`
    - `Cropping2D`
    - `Cropping3D`
* UnSampling
    - `UnSampling1D`
    - `UnSampling2D`
    - `UnSampling3D`
* ZeroPadding
    - `ZeroPadding1D`
    - `ZeroPadding2D`
    - `ZeroPadding3D`
* Locally-connected Layers
    - `LocallyConnected2D()`: 二维局部连接层
        - 类似 `Conv2D`，唯一的差别是没有空间上的权值共享，所以其参数个数远高于二维卷积
* Pooling Layers
    - 最大池化
        - `MaxPolling1D()`
        - `MaxPolling2D()`: 二维最大池化层，也称作下采样层
            - 池化层无可训练参数，主要作用是降维
        - `MaxPolling3D()`
        - `GlobalMaxPolling1D()`
        - `GlobalMaxPolling2D()`: 全局最大池化层
            - 每个通道仅保留一个值。一般从卷积层过渡到全连接层时使用，是 `Flatten` 的替代方案
        - `GlobalMaxPolling3D()`
    - 平均池化
        - `AveragePolling1D()`
        - `AveragePolling2D()`: 二维平均池化层
        - `AveragePolling3D()`
        - `GlobalAveragePolling1D()`
        - `GlobalAveragePolling2D()`: 全局平均池化层
            - 每个通道仅保留一个值
        - `GlobalAveragePolling3D()`

### 循环网络相关层

* RNN
    - `RNN()`: RNN基本层
        - 接受一个循环网络单元或一个循环单元列表，通过调用 `tf.keras.backend.rnn` 函数在序列上进行迭代从而转换成循环网络层
    - `SimpleRNN()`: 简单循环网络层
        - 容易存在梯度消失，不能够适用长期依赖问题，一般较少使用
    - `SimpleRNNCell()`: SimpleRNN 单元
        - 和 SimpleRNN 在整个序列上迭代相比，它仅在序列上迭代一步
    - `AbstractRNNCell`: 抽象 RNN 单元
        - 通过对它的子类化用户可以自定义 RNN 单元，再通过 RNN 基本层的包裹实现用户自定义循环网络层
* `Embedding()`: 嵌入层
    - 一种比 one-hot 更加有效的对离散特征进行编码的方法
    - 一般用于将输入中的单词映射为稠密向量
    - 嵌入层的参数需要学习
* LSTM
    - `LSTM()`: 长短记忆循环网络层，最普遍使用的循环网络层
        - 具有携带轨道，遗忘门，更新门，输出门
        - 可以较为有效地缓解梯度消失问题，从而能够适用长期依赖问题
        - 设置 `return_sequences = True` 时可以返回各个中间步骤输出，否则只返回最终输出
    - `LSTMCell()`: LSTM 单元
        - 和 LSTM 在整个序列上迭代相比，它仅在序列上迭代一步。可以简单理解 LSTM 即 RNN 基本层包裹 `LSTMCell`
    - `ConvLSTM2D()`: 卷积长短记忆循环网络层
        - 结构上类似 `LSTM()`，但对输入的转换操作和对状态的转换操作都是卷积运算
    - `ConvLSTM2DCell()`
* GRU
    - `GRU()`: 门控循环网络层
        - `LSTM()` 的低配版，不具有携带轨道，参数数量少于 `LSTM()`，训练速度更快
    - `GRUCell()`: GRU 单元
        - 和 GRU 在整个序列上迭代相比，它仅在序列上迭代一步
* Attention
    - `Attention`: Dot-product类型注意力机制层
        - 可以用于构建注意力模型
    - `AdditiveAttention`: Additive 类型注意力机制层
        - 可以用于构建注意力模型
- CuDNN
    - `CuDNNGRU()`
    - `CuDNNLSTM()`
- Layer wrapper
    - `TimeDistributed()`: 时间分布包装器
        - 包装后可以将 Dense、Conv2D 等作用到每一个时间片段上
    - `Bidirectional()`: 双向循环网络包装器

# 自定义模型层

## 编写 tf.keras.Lambda 层

如果自定义模型层没有需要被训练的参数，一般推荐使用 `Lambda` 层实现。
`Lambda` 层由于没有需要被训练的参数，只需要定义正向传播逻辑即可，使用比 `Layer` 基类子类化更加简单。
`Lambda` 层的正向逻辑可以使用 Python 的 `lambda` 函数来表达，也可以使用 `def` 关键字定义函数来表达

```python
import tensorflow as tf
from tensorflow.keras import layers, models, regularizers

mypower = layers.Lambda(lambda x: tf.math.pow(x, 2))
mypower(tf.range(5))
```

## 继承 tf.keras.layers.Layer 基类

如果自定义模型层有需要被训练的参数，则可以通过对 `Layer` 基类子类化实现。
通过 `Layer` 的子类化自定义层一般需要继承 `tf.keras.layers.Layers` 类, 
并重写 `__init__`、`build`、`call` 三个方法

```python
import numpy as np
import tensorflow as tf

class MyLayer(tf.keras.layers.Layer):
    def __init__(self):
        super().__init__()
        # 初始化代码
    
    def build(self, input_shape):
        """
        input_shape 是一个 TensorShape 类型对象, 提供输入的形状
        """
        # 在第一次使用该层的时候调用该部分代码, 
        # 在这里创建变量可以使得变量的形状自适应输入的形状
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

class Linear(tf.keras.layers.Layer):
    def __init__(self, units = 32, **kwargs):
        super(Linear, self).__init__(**kwargs)
        self.units = units
    
    def build(self, input_shape):
        """
        build 方法一般定义 Layer 需要被训练的参数
        """
        self.w = self.add_weight(
            name = "w", 
            shape = (input_shape[-1], self.units),  # [n, 1]
            initializer = tf.zeros_initializer(),
            trainable = True,
        )
        self.b = self.add_weight(
            name = "b",
            shape = (self.units),  # [1]
            initializer = tf.zeros_initializer(),
            trainable = True,
        )
        # 相当于设置 self.built = True
        super(Linear, self).build(input_shape)
    
    @tf.function
    def call(self, inputs):
        """
        call 方法一般定义正向传播运算逻辑，__call__ 方法调用了它
        """
        y_pred = tf.matmul(inputs, self.w) + self.b
        return y_pred
    
    def get_config(self):
        """
        如果要让自定义的 Layer 通过 Function API 
        组成模型时可以被保存成 h5 模型
        需要自定义 get_config 方法
        """
        config = super(Linear, self).get_config()
        config.update({
            "units": self.units,
        })
        return config


# ------------------------------
# 
# ------------------------------
# 模型实例化
linear = Linear(units = 8)
print(linear.built)

# 指定 input_shape，显式调用 build 方法，第 0 维代表样本数量，用 None 填充
linear.build(input_shape = (None, 16))
print(linear.built)
print(linear.compute_output_shape(input_shape = (None, 16)))


# ------------------------------
# 
# ------------------------------
linear = Linar(units = 16)
print(linear.built)

# 如果 built = False，调用 __call__ 时会先调用 build 方法，再调用 call 方法
linear(tf.random.uniform((100, 64)))
print(linear.built)

config = linear.get_config()
print(config)



# ------------------------------
# 模型构建
# ------------------------------
tf.keras.backend.clear_session()

model = models.Sequential()
model.add(
    Linear(units = 1, input_shape = (2,))
)
print(f"model.input_shape: {model.input_shape}")
print(f"model.output_shape: {model.output.shape}")
model.summary()

# ------------------------------
# 模型训练
# ------------------------------
model.compile(optimizer = "sgd", loss = "mse", metrics = ["mae"])
predictions = model.predict(tf.constant(
    [[3.0, 2.0], 
     [4.0, 5.0]]
))
print(predictions)

# ------------------------------
# 保存模型为 h5 模型
# ------------------------------
model.save("./models/linear_model.h5", save_format = "h5")
model_loaded_keras = tf.keras.models.load_model(
    "./models/linear_model.h5", 
    custom_objects = {"Linear": Linear},
)
predictions = model_loaded_keras.predict(tf.constant(
    [[3.0, 2.0], 
     [4.0, 5.0]]
))
# ------------------------------
# 保存模型成 tf 模型
# ------------------------------
model.save("./models/linear_model", save_format = "tf")
model_loaded_tf = tf.keras.models.load_model(
    "./models/linear_model"
)
predictions = model_loaded_tf.predict(tf.constant(
    [[3.0, 2.0], 
     [4.0, 5.0]]
))
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

# 输入
inputs = Input()

# 激活函数
model.add(Activation)
```

## 输入


## 输出


## 激活函数

### 模型可用的激活函数

* Softmax:
    - `tf.nn.softmax`
* ReLU:
    - `tf.nn.relu`
* tanh:
    - `tf.nn.tanh`
* Sigmoid:
    - `tf.nn.sigmoid`
* Linear: 
    - `tf.nn.linear`
* Leaky ReLU:
    - `tf.nn.leaky_relu`
* ELU:
    - `tf.nn.elu`
* SELU:
    - `tf.nn.selu`
* swish:
    - `tf.nn.swish`
* GELU:
    - TODO

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
model.add(layers.Dense(
    32, 
    input_shape = (None, 16), 
    activation = tf.nn.relu
))
model.add(layers.Dense(10))
model.add(layers.Activation(tf.nn.softmax))
model.summary()
```

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

* `kernel_constraint` 参数
* `bias_constraint` 参数

### 可用的 Constraints

* `tf.keras.constraints.MaxNorm(max_value = 2, axis = 0)`
    - 最大范数权值约束
* `tf.keras.constraints.NonNeg()`
    - 权重非负的约束
* `tf.keras.constraints.UnitNorm()`
    - 映射到每个隐藏单元的权值的约束, 使其具有单位范数
* `tf.keras.constraints.MinMaxNorm(minvalue = 0, maxvalue = 1.0, rate = 1.0, axis = 0)`
    - 最小/最大范数权值约束: 映射到每个隐藏单元的权值的约束, 使其范数在上下界之间

