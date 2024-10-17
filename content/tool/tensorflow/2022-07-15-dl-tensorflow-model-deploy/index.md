---
title: TensorFlow 模型部署
author: wangzf
date: '2022-07-15'
slug: dl-tensorflow-model-deploy
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

- [TensorFlow Serving 模型部署介绍](#tensorflow-serving-模型部署介绍)
- [准备 protobuf 模型文件](#准备-protobuf-模型文件)
- [安装 TensorFlow Serving](#安装-tensorflow-serving)
  - [安装 Docker](#安装-docker)
  - [加载 tensorflow/serving 镜像到 Docker 中](#加载-tensorflowserving-镜像到-docker-中)
- [启动 TensorFlow Serving 服务](#启动-tensorflow-serving-服务)
- [向 API 服务发送请求](#向-api-服务发送请求)
  - [Linux 的 curl 命令发送请求](#linux-的-curl-命令发送请求)
  - [Python 的 requests 库发送请求](#python-的-requests-库发送请求)
</p></details><p></p>


# TensorFlow Serving 模型部署介绍

TensorFlow 训练好的模型以 TensorFlow 原生方式保存成 protobuf 文件后可以用很多种方式部署运行

* 通过 TensorFlow.js 可以用 JavaScript 脚本加载模型并在浏览器中运行 TensorFlow 模型
* 通过 TensorFlow-Lite 可以在移动和嵌入式设备上加载并运行 TensorFlow 模型
* 通过 TensorFlow Serving 可以加载模型后提供网络接口 API 服务，通过任意编程语言发送网络请求都可以获取模型预测结果
* 通过 TensorFlow for Java 接口可以在 Java 或者 Spark(scala) 中调用 TensorFlow 模型进行预测

使用 TensorFlow Serving 部署模型要完成以下步骤:

1. 准备 protobuf 模型文件
2. 安装 TensorFlow Serving 
3. 启动 TensorFlow Serving 服务
4. 向 API 服务发送请求，获取预测结果

# 准备 protobuf 模型文件

使用 tf.keras 训练一个简单的线性回归模型，并保存成 protobuf 文件

```python
# %tensorflow_version 2.x
import tensorflow as tf
from tensorflow.keras import models, layers, optimizers
print(f"tf.__version__ = {tf.__version__}")

# 样本数量
num_samples = 800

# 数据集
X = tf.random.uniform([num_samples, 2], minval = -10, maxval = 10)
w0 = tf.constant([[2.0], [-1.0]])
b0 = tf.constant(3.0)
Y = X@w0 + b0 + tf.random.normal([n, 1], mean = 0.0, stddev = 2.0)

# 构建模型
tf.keras.backend().clear_session()
inputs = layers.Input(shape = (2,), name = "inputs")
outputs = layers.Dense(1, name = "outputs")(inputs)
linear = models.Model(inputs = inputs, outputs = outputs)
linear.summary()

# 训练模型
linear.compile(
    optimizer = "rmsprop",
    loss = "mse",
    metrics = ["mae"],
)
linear.fit(X, Y, batch_size = 8, outputs = outputs)
tf.print(f"w = {linear.layers[1].kernel}")
tf.print(f"b = {linear.layers[1].bias}")

# 保存模型
export_path = "./data/linear_model/"
version = "1"  # 后续可以通过版本号进行模型版本迭代与管理
linear.save(export_path + version, save_format = "tf")

# 查看保存的模型文件
# !ls {export_path + version} 

# 查看模型文件相关信息
# !saved_model_cli show --dir {export_path = str(version)} --al
```

# 安装 TensorFlow Serving

安装 TensorFlow Serving 有两种主要方式: 

* 通过 Docker 镜像安装
* 通过 apt 安装

通过 Docker 镜像安装是最简单、最直接的方法，推荐采用

## 安装 Docker

* TODO

## 加载 tensorflow/serving 镜像到 Docker 中

```bash
$ docker pull tensorflow/serving
```

# 启动 TensorFlow Serving 服务

```bash
$ docker run -t --rm -p 8501:8501 \
    -v "/Users/.../data/linear_model/" \
    -e MODEL_NAME=linear_model \
    tensorflow/serving & >server.log 2>&1
```

# 向 API 服务发送请求

TensorFlow Serving 支持使用 gRPC 方法和 RESTful API 方法调用以 TensorFlow Serving 部署的模型

RESTful API 以标准的 HTTP POST 方法进行交互, 请求和回复均为 JSON 对象。
为了调用服务器端的模型, 在客户端向服务器发送以下格式的请求.

- 服务器 URI: ``http://服务器地址:端口号/v1/models/模型名:predict``
- 请求内容

```json
{
    "signature_name": "需要调用的函数签名(Sequential模式不需要)",
    "instances": "输入数据",
}
```

- 回复:

```json
{
    "predictions": "返回值"
}
```

## Linux 的 curl 命令发送请求

```bash
$ curl -d '{"instances": [[1.0, 2.0], [5.0, 7.0]]}' \
    -X POST http://localhost:8501/v1/models/linear_model:predict
```

```
{
    "predictions": [[3.06546211], [6.02843142]]
}
```

## Python 的 requests 库发送请求

```python
import json, requests

data = json.dumps({
    "signature_name": "serving_default",
    "instances": [[1.0, 2.0], [5.0, 7.0]],
})
headers = {
    "content-type": "application/json",
}
json_response = requests.post(
    "http://localhost:8501/v1/models/linear_model:predict",
    data = data,
    headers = headers,
)
predictions = json.loads(json_response.text)["predictions"]
print(predictions)

[[3.06546211], [6.02843142]] 
```
