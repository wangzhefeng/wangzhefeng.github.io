---
title: TensorFlow & Keras
author: 王哲峰
date: '2022-07-15'
slug: dl-keras-tensorflow
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

- [需要掌握的内容](#需要掌握的内容)
  - [TensorFlow 新手](#tensorflow-新手)
  - [TensorFlow 专家](#tensorflow-专家)
  - [TensorFlow 库和扩展程序](#tensorflow-库和扩展程序)
- [TODO](#todo)
  - [多输入模型](#多输入模型)
  - [多输出模型](#多输出模型)
  - [经验总结](#经验总结)
    - [机器、深度学习任务问题](#机器深度学习任务问题)
    - [回归问题](#回归问题)
    - [二分类问题](#二分类问题)
    - [数据预处理问题](#数据预处理问题)
    - [样本量问题](#样本量问题)
    - [网络结构选择问题](#网络结构选择问题)
    - [优化器](#优化器)
- [相关资料](#相关资料)
</p></details><p></p>

# 需要掌握的内容

## TensorFlow 新手

* [x] 快速入门
* [ ] Keras 机器学习基础知识
* [ ] 加载和预处理数据

## TensorFlow 专家

* [x] 快速入门
* [ ] 自定义层、训练循环
* [ ] 分布式训练

## TensorFlow 库和扩展程序

* [ ] TensorBoard
* [ ] TensorFLow Hub
* [ ] 数据集
* [ ] 模型优化
* [ ] 概率
* [ ] XLA
* [ ] TFX
* [ ] ...

# TODO

- Keras Sequential 
- 模型假设, 网路只有一个输入和一个输出, 而且网络是层的线性堆叠; 
- 有些网络需要多个独立的输入, 有些网络则需要多个输出, 而有些网络在层与层之间具有内部分支, 这样的网络看起来像是层构成的图(graph), 而不是层的线性堆叠; 
- 多模态(multimodal)输入
- 元数据
- 文本描述
- 图片
- 预测输入数据的多个目标属性
- 类别
- 连续值
- 非线性地网络拓扑结构, 网络结构是有向无环图
- Inception 系列网络
   - 输入被多个并行的卷积分支所处理, 然后将这些分支的输出合并为单个张量; 
- ResNet 系列网络
   - 向模型中添加残差连接(residual connection), 将前面的输出张量与后面的输出张量相加, 
      从而将前面的表示重新注入下游数据流中, 这有助于防止信息处理流程中的信息损失; 

## 多输入模型

- Keras 函数式 API
   - 可以构建具有多个输入的模型, 通常情况下, 这种模型会在某一时刻用一个可以组合多个张量的层将不同输入分支合并, 张量组合方式可能是相加, 连接等, 比如:
- `keras.layers.add`
- `keras.layers.concatenate`
  
- 问答模型:


- 输入:
   - 自然语言描述的问题
   - 文本片段, 提供用于回答问题的信息
- 输出
   - 一个回答, 在最简单的情况下, 这个回答只包含一个词, 可以通过对某个预定义的词表做softmax得到; 

```python
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import *

from tensorflow.keras.model import Model
# from keras.model import Model
from tensorflow.keras import layers, Input
# from keras import layers, Input

# =========================================================================
# 构建模型
# =========================================================================
text_vocabulary_size = 10000
question_vocabulary_size = 10000
answer_vocabulary_size = 500
# 文本片段
text_input = Input(
   shape = (None,), 
   dtype = "int32", 
   name = "text"
)
embedded_text = layers.Embedding(text_vocabulary_size, 64)(text_input)
encoded_text = layres.LSTM(32)(embedded_text)
# 自然语言描述的问题
question_input = Input(
   shape = (None,),
   dtype = "int32",
   name = "question"
)
embedded_question = layers.Embedding(question_vocabulary_size, 32)(question_input)
encoded_question = layers.LSTM(16)(embedded_question)

concatenated = layers.concatenate([encoded_text, encoded_question], axis = -1)

answer = layers.Dense(answer_vocabulary_size, activation = "softmax")(concatenated)

model = Model(inputs = [text_input, question_input], outputs = answer)

model.compile(
   optimizer = "rmsprop",
   loss = "categorical_crossentropy",
   metrics = ["acc"]
)

# =========================================================================
# 训练模型
# =========================================================================
import numpy as np
num_samples = 1000
max_length = 100
text = np.random.randint(1, text_vocabulary_size, size = (num_samples, max_length))
question = np.random.randint(1, question_vocabulary_size, size = (num_samples, max_length))
answers = np.random.randint(answer_vocabulary_size, size = (num_samples))

answers = keras.utils.to_categorical(answers, answer_vocabulary_size)

model.fit([text, question], answers, epochs = 10, batch_size = 128)
model.fit(
   {
      "text": text,
      "question": question,
   },
   answers,
   epochs = 10,
   batch_size = 128
)
```


## 多输出模型

网络同时预测数据的不同性质

```python
from keras import layers, Input
from keras.models import Model

vocabulary_size = 50000
num_income_groups = 10

# 输入层
posts_input = Input(shape = (None,), dtype = "int32", name = "posts")
embedded_posts = layers.Embedding(256, vocabulary_size)(posts_input)
# 隐藏层
x = layers.Conv1D(128, 5, activation = "relu")(embedded_posts)
x = layers.MaxPooling1D(5)(x)
x = layers.Conv1D(256, 5, activation = "relu")(x)
x = layers.Conv1D(256, 5, activation = "relu")(x)
x = layers.MaxPooling1D(5)(x)
x = layers.Conv1D(256, 5, activation = "relu")(x)
x = layers.Conv1D(256, 5, activation = "relu")(x)
x = layers.GlobalMaxPooling1D()(x)
x = layers.Dense(128, activation = "relu")(x)
# 输出层
age_prediction = layers.Dense(1, name = "age")(x)
income_prediction = layers.Dense(num_income_groups, activation = "softmax", name = "income")(x)
gender_prediction = layers.Dense(1, activation = "sigmoid", name = "gender")(x)
# 构建模型
model = Model(posts_input, [age_prediction, income_prediction, gender_prediction])

model.compile(optimizer = "rmsprop", loss = ["mse", "categorical_crossentropy", "binary_crossentropy"])
model.compile(
   optimizer = "rmsprop",
   loss = {
      "age": "mse",
      "income": "categorical_crossentropy",
      "gender": "binary_crossentropy"
   }
)
```

## 经验总结

### 机器、深度学习任务问题

- 二分类
- 多分类
- 标量回归

### 回归问题

- 回归问题使用的损失函数
   - 均方误差(MSE)
- 回归问题使用的评估指标
   - 平均绝对误差(MAE)
- 回归问题网络的最后一层只有一个单元, 没有激活, 是一个线性层, 这是回归的典型设置, 添加激活函数会限制输出范围

### 二分类问题

- 二分类问题使用的损失函数
   - 对于二分类问题的 sigmoid 标量输出, `binary_crossentropy`
- 对于二分类问题, 网络的最后一层应该是只有一个单元并使用 sigmoid 激活的 Dense 层, 网络输出应该是 0~1 范围内的标量, 表示概率值

### 数据预处理问题

- 在将原始数据输入神经网络之前, 通常需要对其进行预处理
   - 结构化数据
   - 图像数据
   - 文本数据
- 将取值范围差异很大的数据输入到神经网络中是有问题的
   - 网路可能会自动适应这种取值范围不同的数据, 但学习肯定变得更加困难
   - 对于这种数据, 普遍采用的最佳实践是对每个特征做标准化, 即对于输入数据的每个特征(输入数据矩阵中的列), 
      减去特征平均值, 再除以标准差, 这样得到的特征平均值为 0, 标准差为 1
   - 用于测试数据标准化的均值和标准差都是在训练数据上计算得到的。在工作流程中, 不能使用测试数据上计算得到的任何结果, 
      即使是像数据标准化这么简单的事情也不行
- 如果输入数据的特征具有不同的取值范围, 应该首先进行预处理, 对每个特征单独进行缩放

### 样本量问题

- 如果可用的数据很少, 使用 K 折交叉验证可以可靠地评估模型
- 如果可用的训练数据很少, 最好使用隐藏层较少(通常只有一到两个)的小型模型, 以避免严重的过拟合
   - 较小的网络可以降低过拟合

### 网络结构选择问题

- 如果可用的训练数据很少, 最好使用隐藏层较少(通常只有一到两个)的小型模型, 以避免严重的过拟合
- 如果数据被分为多个类别, 那么中间层过小可能会导致信息瓶颈

### 优化器

- 无论你的问题是什么, `rmsprop` 优化器通常都是足够好的选择

# 相关资料

* 数据
    - [MNIST 数据集主页](http://yann.lecun.com/exdb/mnist/)
* 网络论文
    - []()
