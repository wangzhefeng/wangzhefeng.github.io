---
title: TensorFlow 工具集
author: 王哲峰
date: '2022-07-15'
slug: dl-tensorflow-utils
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

- [模型可视化](#模型可视化)
  - [plot\_model()](#plot_model)
  - [model\_to\_dot()](#model_to_dot)
- [序列化工具(Serialization utilities)](#序列化工具serialization-utilities)
  - [CustomObjectScope class](#customobjectscope-class)
  - [get\_custom\_objects()](#get_custom_objects)
  - [register\_keras\_serializable()](#register_keras_serializable)
  - [serialize\_keras\_object()](#serialize_keras_object)
  - [daserialize\_keras\_object()](#daserialize_keras_object)
- [Python \& Numpy utilities](#python--numpy-utilities)
  - [to\_categorical()](#to_categorical)
  - [normalize()](#normalize)
  - [get\_file()](#get_file)
  - [Progbar class](#progbar-class)
  - [Sequence class](#sequence-class)
</p></details><p></p>

# 模型可视化

## plot_model()

- Converts a Keras model to dot format and save to a file.

```python

import tensorflow as tf

tf.keras.utils.plot_model(
    model,
    to_file = "model.png",
    show_shapes = False,
    show_dtype = False,
    show_layer_names = True,
    rankdir = "TB",
    expand_nested = False,
    dpi = 96,
)
```

## model_to_dot()

- Convert a Keras model to dot format.

```python
import tensorflow as tf

tf.keras.utils.model_to_dot(
    model,
    show_shapes = False,
    show_dtype = False,
    show_layer_names = True,
    rankdir = "TB",             # "TB": a vertical plot; "LR": a horizontal plot
    expand_nested = False,
    dpi = 96,
    subgraph = False,
)
```

# 序列化工具(Serialization utilities)

- custom_object_scope()
- get_custom_objects()
- register_keras_serializable()
- serialize_keras_object()
- daserialize_keras_object()

## CustomObjectScope class

- 作用

    - 将自定义类/函数 暴露给 Keras 反序列化内部组件
    - 在范围 `with custom_object_scope(object_dict)`, Keras 方法将能够反序列化已保存的配置引用的任何自定义对象

- 语法

```python
import tensorflow as tf

tf.keras.utils.custom_object_scope(*args)
```

- 示例

```python
# 一个自定义的正则化器 `my_regularizer`
my_regularizer = None

# a layer
layer = Dense(3, kernel_regularizer = my_regularizer)

# Config contains a reference to "my_regularizer"
config = layer.get_config()
...

# Later
with custom_object_scope({"my_regularizer": my_regularizer}):
    layer = Dense.from_config(config)
```

## get_custom_objects()

- 作用

    - 额, 下次一定

- 语法

```python
import tensorflow as tf

tf.keras.utils.get_custom_objects()
```

- 示例

```python
get_custom_objects().clear()
get_custom_objects()["MyObject"] = MyObject
```

## register_keras_serializable()

- 作用

    - 额, 下次一定

- 语法

```python
import tensorflow as tf

tf.keras.utils.register_keras.serializable(package = "Custom", name = None)
```

## serialize_keras_object()

- 作用

    - 将 Keras 对象序列化为 Json 兼容的表示形式

- 语法

```python
import tensorflow as tf

tf.keras.utils.serialize_keras_object(instance)
```

## daserialize_keras_object()

- 作用

    - 将 Keras 对象的序列化形式转换回实际对象

- 语法

```python
import tensorflow as tf

tf.keras.utils.deserialize_keras_object(
    identifier, 
    module_objects = None,
    custom_objects = None,
    printable_module_name = "object"
)
```

# Python & Numpy utilities

## to_categorical()

- 作用

    - 将一个类别型向量(整数)转换为 二元类别矩阵
    - 类似于 one-hot

- 语法

```python
import tensorflow as tf

utils.to_categorical(y,
                    num_classes = None,
                    dtypes = "float32")
```

- 示例

```python
# example 1
a = tf.keras.utils.to_categorical([0, 1, 2, 3], num_classes = 4)
a = tf.constant(a, shape = [4, 4])
print(a)

# example 2
b = tf.constant([.9, .04, .03, .03,
                    .3, .45, .15, .13,
                    .04, .01, .94, .05,
                    .12, .21, .5, .17],
                    shape = [4, 4])
loss = tf.keras.backend.categorical_crossentropy(a, b)
print(np.around(loss, 5))

# example 3
loss = tf.keras.backend.categorical_crossentropy(a, a)
print(np.around(loss, 5))
```


## normalize()

- 作用
    
    - 标准化一个 Numpy 数组

- 语法

```python
import tensorflow as tf

tf.keras.utils.normalize(x, axis = -1, order = 2)
```

## get_file()

- 作用

    - Downloads a file from a URL if it not already in the cache.
    - By default the file at the url `origin` is downloaded to the cache_dir `~/.keras`, 
      placed in the cache_subdir datasets, and given the filename `fname`. 
      The final location of a file `example.txt` would therefore be `~/.keras/datasets/example.txt`.
    - Files in tar, tar.gz, tar.bz, and zip formats can also be extracted. 
      Passing a hash will verify the file after download. 
      The command line programs shasum and sha256sum can compute the hash.

- 语法

```python

tf.keras.utils.get_file(
    fname,
    origin,
    untar=False,
    md5_hash=None,
    file_hash=None,
    cache_subdir="datasets",
    hash_algorithm="auto",
    extract=False,
    archive_format="auto",
    cache_dir=None,
)
```

- 示例

```python

import tensorflow

path_to_downloaded_file = tf.keras.utils.get_file(
    "flower_photos",
    "https://storage.googleapis.com/download.tensorflow.org/example_images/flower_photos.tgz",
    untar = True
)
```

## Progbar class

- 作用

    - 显示进度条

- 语法

```python
import tensorflow as tf

tf.keras.utils.Progbar(
    target, 
    width = 30, 
    verbose = 1, 
    interval = 0.05, 
    stateful_metrics = None, 
    unit_name = "step"
)
```

## Sequence class

- 作用
    - 用于拟合数据序列(如数据集)的基础对象
    - 每个人都Sequence必须实现__getitem__和__len__方法。如果您想在各个时期之间修改数据集, 则可以实现 on_epoch_end。该方法__getitem__应返回完整的批次
    - Sequence是进行多处理的更安全方法。这种结构保证了网络在每个时期的每个样本上只会训练一次, 而生成器则不会
- 语法

```python
import tensorflow as tf
tf.keras.utils.Sequence()
```

- 示例

```python
from skimage.io import imread
from skimage.transform import resize
import numpy as np
import math

# Here, `x_set` is list of path to the images
# and `y_set` are the associated classes.

class CIFAR10Sequence(Sequence):

    def __init__(self, x_set, y_set, batch_size):
        self.x, self.y = x_set, y_set
        self.batch_size = batch_size

    def __len__(self):
        return math.ceil(len(self.x) / self.batch_size)

    def __getitem__(self, idx):
        batch_x = self.x[idx * self.batch_size:(idx + 1) *
        self.batch_size]
        batch_y = self.y[idx * self.batch_size:(idx + 1) *
        self.batch_size]

        return np.array([
            resize(imread(file_name), (200, 200))
            for file_name in batch_x]), np.array(batch_y)
```
