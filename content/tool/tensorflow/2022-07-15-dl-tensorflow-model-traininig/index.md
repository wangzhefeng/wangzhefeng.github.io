---
title: TensorFlow 模型编译和训练
author: 王哲峰
date: '2022-07-15'
slug: dl-tensorflow-model-training
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

- [回调函数](#回调函数)
- [模型训练](#模型训练)
  - [使用 GPU 训练模型](#使用-gpu-训练模型)
    - [使用单 GPU 训练模型](#使用单-gpu-训练模型)
    - [使用多 GPU 训练模型](#使用多-gpu-训练模型)
  - [使用 TPU 训练模型](#使用-tpu-训练模型)
- [损失函数](#损失函数)
  - [内置损失函数](#内置损失函数)
    - [内置损失函数的两种形式](#内置损失函数的两种形式)
    - [回归损失](#回归损失)
    - [二分类损失](#二分类损失)
    - [多分类损失](#多分类损失)
    - [概率损失](#概率损失)
  - [创建自定义损失函数](#创建自定义损失函数)
    - [类形式的损失函数](#类形式的损失函数)
    - [函数形式的损失函数](#函数形式的损失函数)
  - [损失函数的使用—compile() \& fit()](#损失函数的使用compile--fit)
    - [通过实例化一个损失类创建损失函数](#通过实例化一个损失类创建损失函数)
    - [直接使用损失函数](#直接使用损失函数)
  - [损失函数的使用—单独使用](#损失函数的使用单独使用)
- [参考资料](#参考资料)
</p></details><p></p>

# 回调函数

# 模型训练

TensorFlow 训练模型通常有 3 种方法:

* 内置 `fit()` 方法
* 内置 `train_on_batch()` 方法
* 自定义训练循环

## 使用 GPU 训练模型

### 使用单 GPU 训练模型

### 使用多 GPU 训练模型

## 使用 TPU 训练模型

# 损失函数

一般来说，监督学习的目标函数由损失函数和正则化项组成，

`$$Objective = Loss + Regularization$$`

对于 Keras 模型：

* 目标函数中的正则化项一般在各层中指定
    - 例如使用 `Dense` 的 `kernel_regularizer` 和 `bias_regularizer` 等参数指定权重使用 L1 或者 L2 正则化项，
      此外还可以用 `kernel_constraint` 和 `bias_constraint` 等参数约束权重的取值范围，这也是一种正则化手段
* 损失函数在模型编译时候指定
    - 对于回归模型，通常使用的损失函数是均方损失函数 `mean_squared_error`
    - 对于二分类模型，通常使用的是二元交叉熵损失函数 `binary_crossentropy`
    - 对于多分类模型
        - 如果 `label` 是 one-hot 编码的，则使用类别交叉熵损失函数 `categorical_crossentropy`。
        - 如果 `label` 是类别序号编码的，则需要使用稀疏类别交叉熵损失函数 `sparse_categorical_crossentropy`

如果有需要，也可以自定义损失函数，自定义损失函数需要接收两个张量 `y_true`，`y_pred` 作为输入参数，
并输出一个标量作为损失函数值

## 内置损失函数

### 内置损失函数的两种形式

内置的损失函数一般有类的实现和函数的实现两种形式，如

* 二分类损失函数
    - `BinaryCrossentropy` 和 `binary_crossentropy`
* 二分类、多分类
    - 类别交叉熵损失函数：`CategoricalCrossentropy` 和 `categorical_crossentropy`
    - 稀疏类别交叉熵损失函数:`SparseCategoricalCrossentropy` 和 `sparse_categorical_crossentropy`

* 语法

```python
tf.keras.losses.Class(
    from_loits = False，
    label_smoothing = 0，
    reduction = "auto"，
    name = ""
)
```

* 示例

```python
# data
y_ture = [[0.，1.]，[0.，0.]]
y_pred = [[0.6，0.4]，[0.4，0.6]]

# reduction="auto" or "sum_over_batch_size"
bce = tf.keras.losses.BinaryCrossentropy()
bce(y_true，y_pred).numpy()

# reduction=sample_weight
bce = tf.keras.losses.BinaryCrossentropy()
bce(y_true，y_pred，sample_weight = [1，0]).numpy()

# reduction=sum
bce = tf.keras.losses.BinaryCrossentropy(
    reduction = tf.keras.losses.Reduction.SUM
)
bce(y_true，y_pred).numpy()

# reduction=none
bce = tf.keras.losses.BinaryCrossentropy(
    reduction = tf.keras.losses.Reduction.NONE
)
bce(y_true，y_pred).numpy()
```

### 回归损失

* `MeanSquaredError` 类、`mean_squared_error` 函数，MSE
    - 均方误差损失，mse
* `Huber` 类
    - Huber 损失
    - 介于 mse 和 mae 之间，对异常值比较鲁棒，相对 mse 有一定优势
* `MeanAbsoluteError` 类、`mean_absolute_error` 函数，MAE
    - 平均绝对值误差损失，mae
* `MeanAbsolutePercentageError` 类、`mean_absolute_percentage_error` 函数
    - 平均百分比误差损失，mape

### 二分类损失

* `BinaryCrossentropy` 类、`binary_crossentropy()` 函数
    - 二元交叉熵损失
* `Hinge` 类、`hinge` 函数
    - 合页损失
    - 最著名的应用是支持向量机的损失函数

### 多分类损失

* `CategoricalCrossentropy` 类、`categorical_crossentropy()` 函数
    - 类别交叉熵
    - 要求 label 为 one-hot 编码
* `SparseCategoricalCrossentropy` 类、`sparse_categorical_crossentropy()` 函数
    - 稀疏类别交叉熵
    - 多分类
    - 要求 label 为序号编码形式
* `CosineSimilarity` 类、`cosine_similarity` 函数
    - 余弦相似度

### 概率损失

* `KLDivergence` 类、`kl_divergence()` 函数，KLD
    - 相对熵损失，KL 散度
    - 常用于最大期望算法 EM 的损失函数，两个概率分布差异的一种信息度量

## 创建自定义损失函数

如果有需要，也可以自定义损失函数，自定义损失函数需要接收两个张量 `y_true`，`y_pred` 作为输入参数，
并输出一个标量作为损失函数值

### 类形式的损失函数

自定义损失函数需要继承 `tf.keras.losses.Loss` 类，重写 `call` 方法即可，
输入真实值 `y_true` 和模型预测值 `y_pred`，输出模型预测值和真实值之间通
过自定义的损失函数计算出的损失值

Focal Loss 是一种对 `binary_crossentropy` 的改进损失函数形式。
它在样本不均衡和存在较多易分类的样本时相比 `binary_crossentropy` 具有明显的优势。
它有两个可调参数，alpha 参数和 gamma 参数。其中 alpha 参数主要用于衰减负样本的权重，
gamma 参数主要用于衰减容易训练样本的权重。从而让模型更加聚焦在正样本和困难样本上。
这就是为什么这个损失函数叫做 Focal Loss，其数学表达式如下：

`$$focal\_loss(y,p) = \begin{cases}
-\alpha  (1-p)^{\gamma}\log(p) &
\text{if y = 1}\\
-(1-\alpha) p^{\gamma}\log(1-p) &
\text{if y = 0}
\end{cases} $$`

```python  
import tensorflow as tf
from tensorflow.keras import losses

class FocalLoss(losses.Loss):

    def __init__(self，
                 gamma = 2.0，
                 alpha = 0.75，
                 name = "focal_loss"):
        self.gamma = gamma
        self.alpha = alpha
    
    def call(self，y_true，y_pred):
        bce = tf.losses.binary_crossentropy(y_true，y_pred)
        p_t = (y_true * y_pred) + ((1 - y_true) * (1 - y_pred))
        alpha_factor = y_true * self.alpha + (1 - y_true) * (1 - self.alpha)
        modulating_factor = tf.pow(1.0 - p_t，self.gamma)
        loss = tf.reduce_sum(
            alpha_factor * modulating_factor * bce，
            axis = -1
        )
        return loss
```

### 函数形式的损失函数

```python
import tensorflow as tf
from tensorflow.keras import losses

def focal_loss(gamma = 2.0，alpha = 0.75):
    def focal_loss_fixed(y_true，y_pred):
        bce = tf.losses.binary_crossentropy(y_true，y_pred)
        p_t = (y_true * y_pred) + ((1- y_true) * (1 - y_pred))
        alpha_factor = y_true * alpha + (1 - y_true) * (1 - alpha)
        modulating_factor = tf.pow(1.0 - p_t，gamma)
        loss = tf.reduce_sum(
            alpha_factor * modulating_factor * bce，
            axis = -1
        )
        
    return focal_loss_fixed
```

## 损失函数的使用—compile() & fit()

### 通过实例化一个损失类创建损失函数

* 可以传递配置参数

```python
from tensorflow import keras
from tensorflow.keras import layers，losses

# 模型构建
model = keras.Sequential()
model.add(
    layers.Dense(
        64，
        kernel_initializer = "uniform"，
        input_shape = (10,)
    )
)
model.add(layers.Activation("softmax"))

# 模型编译
model.compile(
    loss = losses.SparseCategoricalCrossentropy(from_logits = True)，
    optimizer = "adam"，
    metrics = ["acc"]
)
```

### 直接使用损失函数

```python
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.losses import sparse_categorical_crossentropy

# 模型构建
model = keras.Sequential()
model.add(
    layers.Dense(
        64，
        kernel_initializer = "uniform"，
        input_shape = (10,)
    )
)
model.add(layers.Activation("softmax"))

# 模型编译
model.compile(
    loss = "sparse_categorical_crossentropy"，
    optimizer = "adam"，
    metrics = ["acc"]
)
```

## 损失函数的使用—单独使用

```python
tf.keras.losses.mean_squared_error(tf.ones((2，2))，tf.zeros((2，2)))
loss_fn = tf.keras.losses.MeanSquaredError(resuction = "sum_over_batch_size")
loss_fn(tf.ones((2，2))，tf.zeros((2，2)))

loss_fn = tf.keras.losses.MeanSquaredError(reduction = "sum")
loss_fn(tf.ones((2，2))，tf.zeros((2，2)))

loss_fn = tf.keras.losses.MeanSquaredError(reduction = "none")
loss_fn(tf.ones((2，2))，tf.zeros((2，2)))

loss_fn = tf.keras.losses.mean_squared_error
loss_fn(tf.ones((2，2,))，tf.zeros((2，2)))

loss_fn = tf.keras.losses.MeanSquaredError()
loss_fn(tf.ones((2，2))，tf.zeros((2，2)))
```

# 参考资料

* [Focal Loss](https://zhuanlan.zhihu.com/p/80594704)
