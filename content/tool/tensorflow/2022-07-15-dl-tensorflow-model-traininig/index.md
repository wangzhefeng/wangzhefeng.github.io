---
title: TensorFlow 模型编译和训练
author: wangzf
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
- [TensorFlow 优化器](#tensorflow-优化器)
    - [TensorFlow 内置优化器](#tensorflow-内置优化器)
        - [SGD](#sgd)
        - [SGDM(TODO)](#sgdmtodo)
        - [NAG(TOOD)](#nagtood)
        - [RMSprop](#rmsprop)
        - [NAG](#nag)
        - [Adagrad](#adagrad)
        - [Adadelta](#adadelta)
        - [Adam](#adam)
        - [Adamax(TODO)](#adamaxtodo)
        - [Nadam](#nadam)
    - [TensorFlow 优化器使用方法](#tensorflow-优化器使用方法)
        - [模型编译和拟合](#模型编译和拟合)
        - [自定义迭代训练](#自定义迭代训练)
        - [学习率衰减和调度](#学习率衰减和调度)
    - [TensorFlow 优化器使用示例](#tensorflow-优化器使用示例)
        - [optimizer.apply\_gradients](#optimizerapply_gradients)
        - [optimizer.minimize](#optimizerminimize)
        - [model.compile 和 model.fit](#modelcompile-和-modelfit)
    - [TensorFlow 优化器核心 API](#tensorflow-优化器核心-api)
        - [apply\_gradients](#apply_gradients)
        - [weights\_property](#weights_property)
        - [get\_weights](#get_weights)
        - [set\_weights](#set_weights)
    - [TensorFlow 优化器共有参数](#tensorflow-优化器共有参数)
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

# TensorFlow 优化器

模型优化算法的选择直接关系到最终模型的性能，有时候效果不好，
未必是特征的问题或者模型设计的问题，很可能就是优化算法的问题

深度学习优化算法大概经历了以下的发展历程:

`SGD -> SGDM -> NAG -> Adagrad -> Adadelta(RMSprop) -> Adam -> Nadam`

对于一般的新手，优化器直接使用 Adam，并使用默认参数就可以了。
如果要追求评价指标效果，可能会偏爱前期使用 Adam 优化器快速下降，
后期使用 SGD 并精调优化器参数得到更好的结果

此外，目前也有一些前沿的优化算法，据称效果比 Adam 更好，
例如: LazyAdam、Look-ahead、RAdam、Ranger 等

## TensorFlow 内置优化器

在 `tf.keras.optimizers` 子模块中，深度学习优化算法基本都有对应的类实现

```python
from tensorflow.keras import optimizers
```

### SGD

SGD，默认参数为纯 SGD

* 设置 `momentum` 参数不为 0 实际上就变成了 SGDM，考虑了一阶动量
* 设置 `nesterov` 为 `True` 后变成了 NAG(Nesterov Acceleration Gradient)

在计算梯度时计算的是向前走一步所在位置的梯度

```python
sgd = optimizers.SGD(lr = 0.01)
model.compile(loss, optimizer = sgd)
# or
model.compile(loss, optimizer = "sgd")
```

### SGDM(TODO)

```python
sgdm = optimizers.SGD(lr = 0.01, momentum = 0.1)
model.compile(loss, optimizer = sgdm)
# or
model.compile(loss, optimizer = "sgdm")
```

### NAG(TOOD)

```python
nag = optimizers.SGD(lr = 0.01, nesterov = True)
model.compile(loss, optimizer = nag)
# or
model.compile(loss, optimizer = "nag")
```

### RMSprop

```python
rmsprop = optimizers.RMSprop(lr = 0.001)
model.compile(loss, optimizer = rmsprop)
# or
model.compile(loss, optimizer = "rmsprop")
```

### NAG


### Adagrad

```python
adagrad = optimizers.Adagrad(lr = 0.01)
model.compile(loss, optimizer = adagrad)
# or
model.compile(loss, optimizer = "adagrad")
```

### Adadelta

```python
adadelta = optimizers.Adadelta(lr = 1.0)
model.compile(loss, optimizer = adadelta)
# or
model.compile(loss, optimizer = "adadelta")
```

### Adam

```python
adam = optimizers.Adam(lr = 0.001)
model.compile(loss, optimizer = adam)
# or
model.compile(loss, optimizer = "adam")
```

### Adamax(TODO)

```python
adamax = optimizers.Adamax(lr = 0.02)
model.compile(loss, optimizer = adamax)
# or
model.compile(loss, optimizer = "adamax")
```

### Nadam

```python
nadam = optimizers.Nadam(lr = 0.002)
model.compile(loss, optimizer = nadam)
# or
model.compile(loss, optimizer = "nadam")
```

## TensorFlow 优化器使用方法

* `optimizer.apply_gradients`
    - TensorFlow 优化器主要使用 `optimizer.apply_gradients` 方法传入变量和对应梯度，
      从而对给定的变量进行迭代
* `optimizer.minimize`
    - 直接使用 `optimizer.minimize` 方法对目标函数进行迭代优化
* `model.fit()`
    - 最常用的方式是在编译时将优化器传入 Keras 的 Model，通过调用 `model.fit()` 实现对 Loss 的迭代优化

初始化优化器时会创建一个变量 `optimizer.iterations` 用于记录迭代的次数。
因此优化器和 `tf.Variable` 一样，一般需要在 `@tf.function` 外创建

### 模型编译和拟合

```python
from tensorflow import keras
from tensorflow.keras import layers

# model
model = keras.Sequential()
model.add(layers.Dense(64, kernel_initializer = "uniform", input_shape = (10,)))
model.add(layers.Activate("softmax"))

# model compile
opt = keras.optimizers.Adam(learning_rate = 0.01)
model.compile(loss = "categorical_crossentropy", optimizer = opt)
# or
# model.compile(loss = "categorical_crossentropy", optimizer = "adam")
```

### 自定义迭代训练

```python
# Instantiate an optimizer
optimizer = tf.keras.optimizer.Adam()

# Iterate over the batches of a dataset
for x, y in dataset:
    # open a GradientTape
    with tf.GradientTape() as tape:
        # Forward pass
        logits = model(x)
        
        # Loss value for this batch
        loss_value = loss_fn(y, logits)

    # Get gradients of loss wrt the weights
    gradients = tape.gradient(loss_value, model.trainable_weights)

    # Update the weights of the model
    optimizer.apply_gradients(zip(gradients, model.trainable_weights))
```

### 学习率衰减和调度

* 可以使用学习率时间表来调整优化器的学习率如何随时间变化
* ExponentialDecay: 指数衰减
* PiecewiseConstantDecay: 
* PolynomialDecay: 多项式衰减
* InverseTimeDecay: 逆时间衰减

```python
lr_schedule = keras.optimizers.schedules.ExponentialDecay(
    initial_learning_rate = 1e-2,
    decay_steps = 10000,
    decay_rate = 0.9
)
optimizer = keras.optimizers.SGD(learning_rate = lr_schedule)
```

## TensorFlow 优化器使用示例

### optimizer.apply_gradients

求 `$f(x) = a \times x^{2} + b \times x + c$` 的最小值

```python
import tensorflow as tf
from tensorflow.keras import optimizer

# 自变量
x = tf.Variable(0.0, name = "x", dtype = tf.float32)

# 优化器
optimizer = optimizer.SGD(learning_rate = 0.01)

@tf.function
def minimizef():
    a = tf.constant(1.0)
    b = tf.constant(-2.0)
    c = tf.constant(1.0)
    while tf.constant(True):
        with tf.GradientTape() as tape:
            y = a * tf.pow(x, 2) + b * x + c
        dy_dx = tape.gradient(y, x)
        optimizer.apply_gradient(grads_and_vars = [(dy_dx, x)])

        # 迭代终止条件
        if tf.abs(dy_dx) < tf.constant(0.00001):
            break
        
        # 每 100 次迭代打印日志
        if tf.math.mod(optimizer.iterations, 100) == 0:
            printbar()
            tf.print(f"step = {optimizer.iterations}")
            tf.print(f"x = {x}")
            tf.print("")
    y = a * tf.pow(x, 2) + b * x + c

    return y

tf.print(f"y = {minimizef()}")
tf.print(f"x = {x}")
```

### optimizer.minimize

求 `$f(x) = a \times x^{2} + b \times x + c$` 的最小值

```python
import tensorflow as tf
from tensorflow.keras import optimizer

# 自变量
x = tf.Variable(0.0, name = "x", dtype = tf.float32)

# 优化器
optimizer = optimizer.SGD(learning_rate = 0.01)

def f():
    a = tf.constant(1.0)
    b = tf.constant(-2.0)
    c = tf.constant(1.0)
    y = a * tf.pow(x, 2) + b * x + c

@tf.function
def train(epoch = 1000):
    for _ in tf.range(epoch):
        optimizer.minimize(f, [x])
    tf.print(f"epoch = {optimizer.iterations}")
    return f()

train(1000)
tf.print(f"y = {f()}")
tf.print(f"x = {x}")
```

### model.compile 和 model.fit

求 `$f(x) = a \times x^{2} + b \times x + c$` 的最小值

```python
import tensorflow as tf
from tensorflow.keras import optimizers

tf.keras.backend.clear_session()


# 模型构建
class FakeModel(tf.keras.models.Model):

    def __init__(self, a, b, c):
        super(FakeModel, self).__init__()
        self.a = a
        self.b = b
        self.c = c
    
    def build(self):
        self.x = tf.Variable(0.0, name = "x")
        self.built = True
    
    def call(self, features):
        loss = self.a * (self.x) ** 2 + self.b * (self.x) + self.c
        return (tf.ones_like(features) * loss)

model = FakeMOdel(
    tf.constant(1.0),
    tf.constant(-2.0),
    tf.constant(1.0)
)
model.build()
model.summary()

# 损失函数
def myloss(y_true, y_pred):
    return tf.reduce_mean(y_pred)

# 优化器(SGDM)
sgd = optimizers.SGD(
    lr = 0.01, 
    decay = 1e-6, 
    momentum = 0.9, 
    nesterov = True,
)

# 编译模型
model.compile(
    loss = myloss, 
    optimizer = sgd,
)

# 模型训练
history = model.fit(
    tf.zeros((100, 2)),
    tf.ones(100),
    batch_size = 1, 
    epochs = 1000,
)

# 模型结果
tf.print(f"x = {model.x}")
tf.print(f"loss = {model(tf.constant(0.0))}")
```



## TensorFlow 优化器核心 API

* apply_gradients
* weights_property
* get_weights
* set_weights

### apply_gradients

- 语法

```python
Optimizer.apply_gradients(
    grads_and_vars, name=None, experimental_aggregate_gradients=True
)
```

- 参数
  - grads_and_vars: 梯度、变量对的列表
  - name: 返回的操作的名称
  - experimental_aggregate_gradients: 

- 示例

```python
grads = tape.gradient(loss, vars)
grads = tf.distribute.get_replica_context().all_reduce("sum", grads)

# Processing aggregated gradients.
optimizer.apply_gradients(zip(grad, vars), experimental_aggregate_gradients = False)
```

### weights_property

- 语法

```python
import tensorflow as tf

tf.keras.optimizers.Optimizer.weights
```

### get_weights

- 语法

```python
Optimizer.get_weights()
```

- 示例

```python
# 模型优化器
opt = tf.keras.optimizers.RMSprop()

# 模型构建、编译
m = tf.keras.models.Sequential()
m.add(tf.keras.layers.Dense(10))
m.compile(opt, loss = "mse")

# 数据
data = np.arange(100).reshape(5, 20)
labels = np.zeros(5)

# 模型训练
print("Training")
results = m.fit(data, labels)
print(opt.get_weights)
```

### set_weights

- 语法

```python
Optimizer.set_weights(weights)
```

- 示例

```python
# 模型优化器
opt = tf.keras.optimizers.RMSprop()

# 模型构建、编译
m = tf.keras.models.Sequential([tf.keras.layers.Dense(10)])
m.compile(opt, loss = "mse")

# 数据        
data = np.arange(100).reshape(5, 20)
labels = np.zeros(5)

# 模型训练
print("Training")
results = m.fit(data, labels)

# 优化器新权重
new_weights = [
    np.array(10),       # 优化器的迭代次数
    np.ones([20, 10]),  # 优化器的状态变量
    np.zeros([10])      # 优化器的状态变量
]
opt.set_weights(new_weights)
opt.iteration
```


## TensorFlow 优化器共有参数

control gradient clipping

- `clipnorm`
- `clipvalue`

```python
from tensorflow.keras import optimizers

# All parameter gradients will be clipped to
# a maximum norm of 1.
sgd = optimizers.SGD(lr = 0.01, clipnorm = 1)

# All parameter gradients will be clipped to
# a maximum value of 0.5 and
# a minimum value of -0.5.
sgd = optimizers.SGD(lr = 0.01, clipvalue = 0.5)
```

# 参考资料

* [Focal Loss](https://zhuanlan.zhihu.com/p/80594704)
