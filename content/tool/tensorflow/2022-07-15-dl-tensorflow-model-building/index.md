---
title: TensorFlow 模型构建
author: 王哲峰
date: '2022-07-15'
slug: dl-tensorflow-model-building
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

- [模型共有的方法和属性](#模型共有的方法和属性)
- [Sequential API](#sequential-api)
- [Functional API](#functional-api)
- [Subclassing API](#subclassing-api)
- [回调函数-Callbacks](#回调函数-callbacks)
</p></details><p></p>

使用 Keras 接口有以下 3 种方式构建模型:

* 使用 Sequential(Sequential API) 按层顺序构建模型
* 使用函数式 API(Functional API) 构建任意结构模型
* 继承 Model 基类(Subclassing API)构建自定义模型

# 模型共有的方法和属性

```python
from tf.keras.model import Model
from tf.keras.model import model_from_json, model_from_yaml
```

- model.layers
- model.inputs
- model.outputs
- model.summary()
- Config
   - model.get_config()
      - Model.from_config()
      - Sequential.from_config()
- Weights
   - model.get_weights()
      - *to Numpy arrays*
   - model.set_weights(weights)
      - *from Numpy arrays*
   - model.save_weights(filepath)
      - *to HDF5 file*
   - model.loadweights(filepath, byname = False)
      - *from HDF5 file*
- Save or Load
   - model.to_json()
      - modelfromjson()
   - modeltoyaml()
      - modelfromyaml()

# Sequential API

Sequential 模型是层(layers)的线性堆叠

```python
from tensorflow import keras
from tensorflow.keras import layers, models
from tensorflow.keras.datasets import mnist

# data
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0

# model
model = models.Sequential()
model.add(layers.Dense(units = 64, activation = "relu"))
model.add(layers.Dense(units = 10, activation = "softmax"))
model.compile(
   loss = "categorical_crossentropy",
   optimizer = "sgd",
   metrics = ["accuracy"]
)
model.fit(x_train, y_train, epochs = 5, batch_size = 32)
loss_and_metrics = model.evaluate(x_test, y_test, batch_size = 128)
classes = model.predict(x_test, batch_size = 128)
```

# Functional API

* Keras 函数式 API 是定义复杂模型的方法
* Keras 函数式 API 可以重用经过训练的模型, 可以通过在张量上调用任何模型并将其视为一个层(layers)
  - 调用模型的结构
  - 调用模型的权重

函数式 API 特点

* 所有模型都像层(layer)一样可以调用
* 多输入和多输出模型
* 共享图层
* "层节点"概念


```python
inputs = tf.keras.Input(shape = (28, 28, 1))
x = tf.keras.layers.Flatten()(inputs)
x = tf.keras.layers.Dense(units = 100, activation = tf.nn.relu)(x)
x = tf.keras.layers.Dense(units = 10)(x)
outputs = tf.keras.layers.Softmax()(x)
model = tf.keras.Model(inputs = inputs, outputs = outputs)
```

# Subclassing API

- 使用 Subclassing API 建立模型, 即对 `tf.keras.Model` 类进行扩展以定义自己的新模型
- 实现 forward pass in the ``call`` method
- 模型的 layers 定义在 ``__init__(self, ...)`` 中
- 模型的前向传播定义在 ``call(self, inputs)`` 中
- 可以通过调用制定的自定义损失函数 ``self.add_loss(loss_tensor)``
- 在 subclassing 模型中, 模型的拓扑结构被定义为 Python 代码, 而不是 layers 的静态图, 
  因此无法检查或序列化模型的拓扑结构, 即以下方法不适用于 subclassing 模型:
   - model.inputs
   - model.outputs
   - model.to_yaml()
   - model.to_json()
   - model.get_config()
   - model.save()
- 模型(keras.model.Model)子类的 API 可以为实现更加复杂的模型提供了灵活性, 但是是有代价的, 除了以上的功能不能使用, 并且模型更复杂, 更容易出错


```python
import tensorflow as tf

class MyModel(tf.keras.Model):
    
    def __init__(self):
        super(MyModel, self).__init__()
        
        # 此处添加初始化的代码(包含call方法中会用到的层)例如:
        self.layer1 = tf.keras.layers.BuildInLayer()
        self.layer2 = MyCustomLayer(...)

    def call(self, input):
        # 此处添加模型调用的代码(处理输入并返回输出), 例如:
        x = layer1(input)
        self.output = layer2(x)
        return output

model = MyModel()

with tf.GradientTape() as tape:
    logits = model(images)
    loss_value = loss(logits, labels)
grads = tape.gradient(loss_value, model.trainable_variables)
optimizer.apply(zip(grads, model.trainable_variables))
```

# 回调函数-Callbacks

- 回调函数是一个函数的集合, 会在训练的阶段使用
- 可以使用回调函数查看训练模型的内在状态和统计。
  也可以传递一个列表的回调函数(作为 `callbacks` 关键字参数)到 `Sequential` 或 `Model` 类型的 `.fit()` 方法。
  在训练时, 相应的回调函数的方法会被在各自的阶段被调用

回调函数API:

- keras.callbacks.Callback()
   - 用来创建新的回调函数的抽象基类
   - `.params`
   - `.model`
- keras.callbacks.BaseLogger(stateful_metrics = None)
   - 基类训练 epoch 评估值的均值
- keras.callbacks.TerminateOnNaN()
   - 当遇到损失为 `NaN` 停止训练
- keras.callbacks.ProgbarLogger()
- keras.callbacks.History()
   - 所有事件都记录到 History 对象
- keras.callbacks.ModelCheckpoint()
   - 在每个训练期之后保存模型
- keras.callbacks.EarlyStopping()
- keras.callbacks.RemoteMonitor()
- keras.callbacks.LearningRateScheduler(schedule, verbose = 0)
- keras.callbacks.TensorBoard()
- keras.callbacks.ReduceLROnPlateau()
- keras.callbacks.CSVLogger()
- keras.callbacks.LambdaCallback()

创建回调函数:

```python
from keras.layers import Dense, Activation
from keras.models import Sequential
from keras.callbacks import ModelCheckpoint

# 模型建立
model = Sequenital()
model.add(Dense(10, input_dim = 784, kernel_initializer = "uniform"))
model.add(Activation("softmax"))

# 模型编译
model.compile(loss = "categorical_crossentropy", optimizer = "rmsporp")

# 模型训练
# 在训练时, 保存批量损失值
class LossHistory(keras.callbacks.Callback):
      def on_train_begin(self, logs = {}):
         self.losses = []

      def on_batch_end(self, batch, logs = {}):
         self.losses.append(logs.get("loss"))
history = LossHistory()

# 如果验证集损失下降, 在每个训练 epoch 后保存模型
checkpointer = ModelCheckpoint(filepath = "/tmp/weight.hdf5",
                               verbose = 1,
                               save_best_only = True)
model.fit(x_train, 
         y_train, 
         batch_size = 128, 
         epochs = 20, 
         verbose = 0,
         validation_data = (x_test, y_test), 
         callbacks = [history, checkpointer]
)

# 模型结果输出
print(history.losses)
```


