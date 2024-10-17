---
title: TensorFlow 模型保存和加载
author: wangzf
date: '2022-07-15'
slug: dl-tensorflow-model-save-load
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

- [tf.train.Checkpoint: 变量的保存与恢复](#tftraincheckpoint-变量的保存与恢复)
  - [tf.train.Checkpoint 介绍](#tftraincheckpoint-介绍)
  - [tf.train.Checkpoint 代码框架](#tftraincheckpoint-代码框架)
- [使用 SaveModel 完整导出模型](#使用-savemodel-完整导出模型)
- [Keras 自有的模型导出格式](#keras-自有的模型导出格式)
</p></details><p></p>



# tf.train.Checkpoint: 变量的保存与恢复

很多时候, 希望在模型训练完成后能将训练好的参数(变量)保存起来, 这样在需要使用模型的其他地方载入模型和参数, 
就能直接得到训练好的模型, 保存模型有很多中方式:

- Python 的序列化模块 `pickle` 存储 `model.variables`

    - 然而, TensorFlow 的变量类型 `ResourceVariable` 并不能被序列化
    - 语法:

```python
import pickle
```

## tf.train.Checkpoint 介绍

- `tf.train.Checkpoint` 简介

TensorFlow 提供了 `tf.train.Checkpoint` 这一强大的变量保存与恢复类, 提供的方法可以保存和恢复 TensorFlow 中的大部分对象, 
比如下面类的实例都可以被保存: 

- `tf.keras.optimizer`
- `tf.Variable`
- `tf.keras.Layer`
- `tf.keras.Model`
- Checkpointable State 的对象


- `tf.train.Checkpoint` 使用方法

- 方法:

    - `save()`
    - `restore()`

- 语法:

```python
# 保存训练好的模型, 先声明一个 Checkpoint
model = TrainedModel()
checkpoint = tf.train.Checkpoint(myAwesomeModel = model, myAwesomeOptimizer = optimizer)
checkpoint.save(save_path_with_prefix)

# 载入保存的训练模型
model_to_be_restored = MyModel()  # 待恢复参数的同一模型
checkpoint = tf.train.Checkpoint(myAwesomeModel = model_to_be_restored)
checkpoint.restore(save_path_with_prefix_and_index)

# 为了载入最近的一个模型文件, 返回目录下最近一次检查点的文件名
tf.train.latest_checkpoint(save_path)
```

.. note:: 

- 参数:

    - `myAwesomeModel`: 待保存的模型 model 所取的任意键名, 在恢复变量时还将使用这一键名
    - `myAwesomeOptimizer`: 待保存的模型 optimizer 所取的任意键名, 在恢复变量时还将使用这一键名 
    - `save_path_with_prefix`: 保存文件的目录+前缀
    - `save_path_with_prefix_and_index`: 之前保存的文件目录+前缀+序号

- `checkpoint.save("./model_save/model.ckpt")`: 会在模型保存的文件夹中生成三个文件:

    - `checkpoint`
    - `model.ckpt-1.index`
    - `model.ckpt-1.data-00000-of-00001`

- `checkpoint.restore("./model/save/model.ckpt-1")`

    - 载入前缀为 `model.ckpt`、序号为 `1` 的文件来恢复模型


## tf.train.Checkpoint 代码框架

1.train.py 模型训练阶段

```python

# 训练好的模型
model = MyModel()

# 实例化 Checkpoint, 指定保存对象为 model(如果需要保存 Optimizer 的参数也可以加入)
checkpoint = tf.train.Checkpoint(myModel = model)
manager = tf.train.CheckpointManager(checkpoint, directory = "./save", checkpoint_name = "model.ckpt", max_to_keep = 10)

# ...(模型训练代码)

# 模型训练完毕后将参数保存到文件(也可以在模型训练过程中每隔一段时间就保存一次)
if manager:
    manager.save(checkpoint_number = 100)
else:
    checkpoint.save("./save/model.ckpt")
```

2.test.py 模型使用阶段

```python

# 要使用的模型
model = MyModel()

# 实例化 Checkpoint, 指定恢复对象为 model
checkpoint = tf.train.Checkpoint(myModel = model)

# 从文件恢复模型参数
checkpoint.restore(tf.train.latest_checkpoint("./save))

# ...(模型使用代码)
```

.. note:: 

- `tf.train.Checkpoint` (检查点)只保存模型的参数, 不保存模型的计算过程, 
    因此一般用于在具有的模型源码时恢复之前训练好的模型参数。如果需要导出模型(无须源代码也能运行模型)。

# 使用 SaveModel 完整导出模型

作为模型导出格式的 `SaveModel` 包含了一个 TensorFlow 程序的完整信息: 不仅包含参数的权值, 还包含计算的流程(计算图)。
当模型导出为 SaveModel 文件时, 无须模型的源代码即可再次运行模型, 这使得 `SaveModel` 尤其适用于模型的分享和部署。

Keras 模型均可以方便地导出为 `SaveModel` 格式。不过需要注意的是, 因为 `SaveModel` 基于计算图, 
所以对于通过继承 `tf.keras.Model` 类建立的 Keras 模型来说, 需要导出为 `SaveModel` 格式的方法(比如 call) 都需要
使用 `@tf.function` 修饰。


语法:

```python
# 保存
tf.saved_model.save(model, "保存的目标文件夹名称")

# 载入
model = tf.saved_model.load("保存的目标文件夹名称")
```


示例:

```python

```







# Keras 自有的模型导出格式

示例:

```bash
curl -LO https://raw.githubcontent.com/keras-team/keras/master/examples/mnist_cnn.py
```

```python
model.save("mnist_cnn.h5")
```


```python

import keras

keras.models.load_model("mnist_cnn.h5")
```
