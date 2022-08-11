---
title: TensorFlow Serving
author: 王哲峰
date: '2022-08-12'
slug: dl-tensorflow-serving
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

- [TensorFLow Serving 安装](#tensorflow-serving-安装)
- [TensorFLow Serving 模型部署](#tensorflow-serving-模型部署)
- [在客户端调用以 TensorFLow  Serving 部署的模型](#在客户端调用以-tensorflow--serving-部署的模型)
</p></details><p></p>

# TensorFLow Serving 安装


# TensorFLow Serving 模型部署



# 在客户端调用以 TensorFLow  Serving 部署的模型

TensorFLow Serving 支持使用 gRPC 方法和 RESTful API 方法调用以 
TensorFLow Serving 部署的模型。

RESTful API 以标准的 HTTP POST 方法进行交互, 请求和回复均为 JSON 对象。为了调用服务器端的模型, 在客户端向服务器发送以下格式的请求.

- 服务器 URI: ``http://服务器地址:端口号/v1/models/模型名:predict``
- 请求内容

```json
{
    "signature_name": "需要调用的函数签名(Sequential模式不需要)",
    "instances": "输入数据"
}
```

- 回复:

```json
{
    "predictions": "返回值"
}
```
