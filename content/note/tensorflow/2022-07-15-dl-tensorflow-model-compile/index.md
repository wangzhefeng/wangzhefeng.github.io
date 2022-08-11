---
title: TensorFlow 模型编译
author: 王哲峰
date: '2022-07-15'
slug: dl-tensorflow-model-compile
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

- [损失函数](#损失函数)
  - [常用损失函数](#常用损失函数)
  - [损失函数的使用——compile() & fit()](#损失函数的使用compile--fit)
  - [损失函数的使用——单独使用](#损失函数的使用单独使用)
  - [创建自定义损失函数](#创建自定义损失函数)
- [评价指标](#评价指标)
  - [metrics](#metrics)
  - [Accuracy metrics](#accuracy-metrics)
  - [Probabilistic metrics](#probabilistic-metrics)
  - [Regression metrics](#regression-metrics)
  - [Classification metrics based on True/False positives & negatives](#classification-metrics-based-on-truefalse-positives--negatives)
  - [image segmentation metrics](#image-segmentation-metrics)
  - [Hinge metrics for "maximum-margin" Classification](#hinge-metrics-for-maximum-margin-classification)
  - [评价指标的使用——compile() & fit()](#评价指标的使用compile--fit)
  - [评价指标的使用——单独使用](#评价指标的使用单独使用)
  - [自定义评估指标](#自定义评估指标)
- [优化器](#优化器)
  - [Optimizers](#optimizers)
  - [optimizder 的使用方式](#optimizder-的使用方式)
  - [optimizers 的共有参数](#optimizers-的共有参数)
  - [优化器的使用](#优化器的使用)
  - [优化算法核心 API](#优化算法核心-api)
</p></details><p></p>


```python
model.compile(loss, optimizer, metrics)
```

# 损失函数

   - Loss Function
   - Objective Function
   - Optimization score Function

**回归:**

```python
from keras import losses

# 回归 
from keras.losses import mean_squared_error
from keras.losses import mean_absolute_error
from keras.losses import mean_absolute_percentage_error
from keras.losses import mean_squared_logarithmic_error
from keras.losses import squared_hinge
from keras.losses import hinge
from keras.losses import categorical_hinge
from keras.losses import logcosh

model.Compile(loss = ["mse", "MSE", mean_squared_error], 
            optimizer, 
            metircs)
model.Compile(loss = ["mae", "MAE", mean_absolute_error], 
            optimizer, 
            metircs)
model.Compile(loss = ["mape", "MAPE", mean_absolute_percentage_error], 
            optimizer, 
            metircs)
model.Compile(loss = ["msle", "MLSE", mean_squared_logarithmic_error], 
            optimizer, 
            metircs)
```

**分类:**

```python
# 分类
from keras.losses import categorical_crossentropy
from keras.losses import sparse_categorical_crossentropy
from keras.losses import binary_crossentropy
from keras.losses import kullback_leibler_divergence
from keras.losses import poisson
from keras.losses import cosine_proximity

model.Compile(loss = ["kld", "KLD", kullback_leibler_divergence], 
            optimizer, 
            metircs)
model.Compile(loss = ["cosine", cosine_proximity], 
            optimizer, 
            metircs)
```



The purpose of loss functions is to compute the quantity that a model 
should seek to minimize during training.

## 常用损失函数

- class handle
    - 可以传递配置参数
- function handle

1. 概率损失(Probabilistic losses)

- `BinaryCrossentropy` class
    - `binary_crossentropy()` function
- `CategoricalCrossentropy` class
    - `categorical_crossentropy()` function
- `SparseCategoricalCrossentropy` class
    - `sparse_categorical_crossentropy()` function
- `Possion` class
    - `possion()` function
- `KLDivergence` class
    - `kl_divergence()` function

class & function() 使用方法

- 作用
- 二分类损失函数
    - BinaryCrossentropy & binary_crossentropy
    - Computes the cross-entropy loss between true labels and predicted labels.
- 二分类、多分类
    - CategoricalCrossentropy & categorical_crossentropy
    - SparseCategoricalCrossentropy & sparse_categorical_crossentropy
- 其他
- 语法

```python
tf.keras.losses.Class(
    from_loits = False, 
    label_smoothing = 0, 
    reduction = "auto", 
    name = ""
)
```

- 示例

```python
# data
y_ture = [[0., 1.], [0., 0.]]
y_pred = [[0.6, 0.4], [0.4, 0.6]]

# reduction="auto" or "sum_over_batch_size"
bce = tf.keras.losses.BinaryCrossentropy()
bce(y_true, y_pred).numpy()

# reduction=sample_weight
bce = tf.keras.losses.BinaryCrossentropy()
bce(y_true, y_pred, sample_weight = [1, 0]).numpy()

# reduction=sum
bce = tf.keras.losses.BinaryCrossentropy(reduction = tf.keras.losses.Reduction.SUM)
bce(y_true, y_pred).numpy()

# reduction=none
bce = tf.keras.losses.BinaryCrossentropy(reduction = tf.keras.losses.Reduction.NONE)
bce(y_true, y_pred).numpy()
```

2. 回归损失(Regression losses)

- `MeanSquaredError` class
    - `mean_squared_error` function 
- `MeanAbsoluteError` class
    - `mean_absolute_error` function
- `MeanAbsolutePercentageError` class
    - `mean_absolute_percentage_error` function
- `MeanSquaredLogarithmicError` class
    - `mean_squared_logarithmic_error` function
- `CosineSimilarity` class
    - `cosine_similarity` function
- `Huber` class
    - `huber` function
- `LogCosh` class
    - `log_cosh` function

3. Hinge losses for "maximum-margin" classification

- `Hinge` class
    - `hinge` function
- `SquaredHinge` class
    - `squared_hinge` function
- `CategoricalHinge` class
    - `categorical_hinge` function

## 损失函数的使用——compile() & fit()

- 通过实例化一个损失类创建损失函数, 可以传递配置参数

```python
from tensorflow import keras
from tensorflow.keras import layers

model = keras.Sequential()
model.add(layers.Dense(64, kernel_initializer = "uniform", input_shape = (10,)))
model.add(layers.Activation("softmax"))

model.compile(
    loss = keras.losses.SparseCategoricalCrossentropy(from_logits = True), 
    optimizer = "adam", 
    metrics = ["acc"]
)
```

- 直接使用损失函数

```python
from tensorflow.keras.losses import sparse_categorical_crossentropy

model.compile(
    loss = "sparse_categorical_crossentropy", 
    optimizer = "adam", 
    metrics = ["acc"]
)
```

## 损失函数的使用——单独使用

```python
tf.keras.losses.mean_squared_error(tf.ones((2, 2)), tf.zeros((2, 2)))
loss_fn = tf.keras.losses.MeanSquaredError(resuction = "sum_over_batch_size")
loss_fn(tf.ones((2, 2)), tf.zeros((2, 2)))

loss_fn = tf.keras.losses.MeanSquaredError(reduction = "sum")
loss_fn(tf.ones((2, 2)), tf.zeros((2, 2)))

loss_fn = tf.keras.losses.MeanSquaredError(reduction = "none")
loss_fn(tf.ones((2, 2)), tf.zeros((2, 2)))

loss_fn = tf.keras.losses.mean_squared_error
loss_fn(tf.ones((2, 2,)), tf.zeros((2, 2)))

loss_fn = tf.keras.losses.MeanSquaredError()
loss_fn(tf.ones((2, 2)), tf.zeros((2, 2)))
```

## 创建自定义损失函数

- Any callable with the signature `loss_fn(y_true, y_pred)` that returns an array of 
    losses (one of sample in the input batch) can be passed to compile() as a loss. 
- Note that sample weighting is automatically supported for any such loss.

示例:

```python
def my_loss_fn(y_true, y_pred):
    squared_difference = tf.square(y_true - y_pred)
    return tf.reduce_mean(squared_difference, axis = -1)

model.compile(optimizer = "adam", loss = my_loss_fn)
```

1. `add_loss()` API

```python
from tensorflow.keras.layers import Layer

class MyActivityRegularizer(Layer):
    """Layer that creates an activity sparsity regularization loss."""

    def __init__(self, rate = 1e-2):
        super(MyActivityRegularizer, self).__init__()
        self.rate = rate

    def call(self, inputs):
        self.add_loss(self.rate * tf.reduce_sum(tf.square(inputs)))

        return inputs

from tensorflow.keras import layers

class SparseMLP(Layer):
    """Stack of Linear layers with a sparsity regularization loss."""

    def __init__(self, output_dim):
        super(SparseMLP, self).__init__()
        self.dense_1 = layers.Dense(32, activation=tf.nn.relu)
        self.regularization = MyActivityRegularizer(1e-2)
        self.dense_2 = layers.Dense(output_dim)

    def call(self, inputs):
        x = self.dense_1(inputs)
        x = self.regularization(x)
        return self.dense_2(x)

mlp = SparseMLP(1)
y = mlp(tf.ones((10, 10)))

print(mlp.losses)  # List containing one float32 scalar

mlp = SparseMLP(1)
mlp(tf.ones((10, 10)))
assert len(mlp.losses) == 1
mlp(tf.ones((10, 10)))
assert len(mlp.losses) == 1  # No accumulation.
```

- 自定义损失函数需要继承 `tf.keras.losses.Loss` 类, 重写 `call` 方法即可, 
  输入真实值 `y_true` 和模型预测值 `y_pred`, 输出模型预测值和真实值之间通
  过自定义的损失函数计算出的损失值

```python  
import numpy as np
import tensorflow as tf

class MeanSquaredError(tf.keras.losses.Loss):
    def call(self, y_true, y_pred):
        return tf.reduce_mean(tf.square(y_pred - y_true))
```

# 评价指标

- Metric 是一个评估模型表现的函数
- Metric 函数类似于一个损失函数, 只不过模型评估返回的 metric
  不用来训练模型, 因此, 可以使用任何损失函数当做一个 metric 函数使用

## metrics

API:

```python
from keras import metrics
from keras.metrics import binary_accuracy
from keras.metrics import categorical_accuracy
from keras.metrics import sparse_categorical_accuracy
from keras.metrics import top_k_categorical_accuracy
from keras.metrics import sparse_top_k_categorical_accuracy
from keras.metrics import mae

from keras.losses import mean_squared_error
from keras.losses import mean_absolute_error
from keras.losses import mean_absolute_percentage_error
from keras.losses import mean_squared_logarithmic_error
from keras.losses import squared_hinge
from keras.losses import hinge
from keras.losses import categorical_hinge
from keras.losses import logcosh
from keras.losses import categorical_crossentropy
from keras.losses import sparse_categorical_crossentropy
from keras.losses import binary_crossentropy
from keras.losses import kullback_leibler_divergence
from keras.losses import poisson
from keras.losses import cosine_proximity
```

Metrics Name:

```python
metrics = ["acc", "accuracy"]
```

## Accuracy metrics

- Accuracy class
- BinaryAccuracy class
- CategoricalAccuracy class
- TopKCategoricalAccuracy class
- SparseTopKCategoricalAccuracy class

## Probabilistic metrics

- BinaryCrossentropy class
- CategoricalCrossentropy class
- SparseCategoricalCrossentropy class
- KLDivergence class
- Poisson class

## Regression metrics

- MeanSquaredError class
- RootMeanSquaredError class
- MeanAbsoluteError class
- MeanAbsolutePercentageError class
- CosineSimilarity class
- LogCoshError class

## Classification metrics based on True/False positives & negatives

- AUC class
- Precision class
- Recall class
- TurePositives class
- TrueNegatives class
- FalsePositives class
- FalseNegatives class
- PrecisionAtRecall class
- SensitivityAtSpecificity class
- SpecificityAtSensitivity class

## image segmentation metrics

- MeanIoU class

## Hinge metrics for "maximum-margin" Classification

- Hinge class
- SquaredHinge class
- CategoricalHinge class

## 评价指标的使用——compile() & fit()


## 评价指标的使用——单独使用

## 自定义评估指标

- 自定义评估指标需要继承 `tf.keras.metrics.Metric` 类, 
  并重写 `__init__`、`update_state`、`result` 三个方法

```python
import numpy as np
import tensorflow as tf

class SparseCategoricalAccuracy(tf.keras.metrics.Metric):
    def __init__(self):
        super().__init__()
        self.total = self.add_weight(name = "total", dtype = tf.int32, initializer = tf.zeros_initializer())
        self.count = self.add_weight(name = "total", dtype = tf.int32, initializer = tf.zeros_initializer())

    def update_state(self, y_true, y_pred, sample_weight = None):
        values = tf.cast(tf.equal(y_true, tf.argmax(y_pred, axis = 1, output_type = tf.int32)), tf.int32)
        self.total.assign_add(tf.shape(y_true)[0])
        self.count.assign_add(tf.reduce_sum(values))

    def result(self):
        return self.count / self.total
```


```python

import keras.backend as K

def mean_pred(y_true, y_pred):
   return K.mean(y_pred)

model.compile(optimizers = "rmsprop",
            loss = "binary_accuracy",
            metrics = ["accuracy", mean_pred])
```

# 优化器

## Optimizers

- SGD
- RMSprop
- Adagrad
- Adadelta
- Adam
- Adamax
- Nadam

```python
from keras import optimizers

sgd = optimizers.SGD(lr = 0.01)
model.compile(loss, optimizer = sgd)
# or
model.compile(loss, optimizer = "sgd")

rmsprop = optimizers.RMSprop(lr = 0.001)
model.compile(loss, optimizer = rmsprop)
# or
model.compile(loss, optimizer = "rmsprop")

adagrad = optimizers.Adagrad(lr = 0.01)
model.compile(loss, optimizer = adagrad)
# or
model.compile(loss, optimizer = "adagrad")

adadelta = optimizers.Adadelta(lr = 1.0)
model.compile(loss, optimizer = adadelta)
# or
model.compile(loss, optimizer = "adadelta")

adam = optimizers.Adam(lr = 0.001)
model.compile(loss, optimizer = adam)
# or
model.compile(loss, optimizer = "adam")

adamax = optimizers.Adamax(lr = 0.02)
model.compile(loss, optimizer = adamax)
# or
model.compile(loss, optimizer = "adamax")

nadam = optimizers.Nadam(lr = 0.002)
model.compile(loss, optimizer = nadam)
# or
model.compile(loss, optimizer = "nadam")
```

## optimizder 的使用方式

(1) ``keras.optimizers`` 和 ``optimizer`` 参数

```python
from keras import optimizers

# 编译模型
sgd = optimizers.SGD(lr = 0.01, decay = 1e-6, momentum = 0.9, nesterov = True)
model.compile(loss = "mean_squared_error", optimizer = sgd)
```

(2) ``optimizer`` 参数

```python
# 编译模型
model.compile(loss = "mean_squared_error", optimizer = "sgd")
```

## optimizers 的共有参数

- control gradient clipping
   - ``clipnorm``
   - ``clipvalue``

```python
from keras import optimizers

# All parameter gradients will be clipped to
# a maximum norm of 1.
sgd = optimizers.SGD(lr = 0.01, clipnorm = 1)

# All parameter gradients will be clipped to
# a maximum value of 0.5 and
# a minimum value of -0.5.
sgd = optimizers.SGD(lr = 0.01, clipvalue = 0.5)
```

## 优化器的使用

1. 模型编译(compile)和拟合(fit)

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
# model.compile(loss = "categorical_crossentropy", optimizer = "adam")
```

2. 自定义迭代训练

```python
# Instantiate an optimizer
optimizer = tf.keras.optimizer.Adam()

# Iterate over the batches of a dataset.
for x, y in dataset:
# open a GradientTape
with tf.GradientTape() as tape:
    # Forward pass.
    logits = model(x)
    
    # Loss value for this batch
    loss_value = loss_fn(y, logits)

# Get gradients of loss wrt the weights
gradients = tape.gradient(loss_value, model.trainable_weights)

# Update the weights of the model
optimizer.apply_gradients(zip(gradients, model.trainable_weights))
```

3. 学习率衰减(decay)、调度(sheduling)

- 可以使用学习率时间表来调整优化器的学习率如何随时间变化
- ExponentialDecay: 指数衰减
- PiecewiseConstantDecay: 
- PolynomialDecay: 多项式衰减
- InverseTimeDecay: 逆时间衰减

```python
lr_schedule = keras.optimizers.schedules.ExponentialDecay(
initial_learning_rate = 1e-2,
decay_steps = 10000,
decay_rate = 0.9
)
optimizer = keras.optimizers.SGD(learning_rate = lr_schedule)
```

## 优化算法核心 API

- apply_gradients
- weights_property
- get_weights
- set_weights

1. apply_gradients

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

2. weights_property

- 语法

```python
import tensorflow as tf

tf.keras.optimizers.Optimizer.weights
```

3. get_weights

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

4. set_weights

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
