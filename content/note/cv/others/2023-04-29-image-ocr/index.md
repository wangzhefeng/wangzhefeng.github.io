---
title: 光学字符识别
subtitle: Text Detection OCR
author: wangzf
date: '2023-04-29'
slug: image-ocr
categories:
  - computer vision
tags:
  - model
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

- [OCR 模型](#ocr-模型)
  - [CRNN](#crnn)
  - [CRAFT](#craft)
- [OCR 任务](#ocr-任务)
  - [从图像中删除文本](#从图像中删除文本)
  - [车牌识别](#车牌识别)
- [工具](#工具)
  - [Keras OCR](#keras-ocr)
- [参考](#参考)
</p></details><p></p>

# OCR 模型

## CRNN


## CRAFT


# OCR 任务




## 从图像中删除文本

## 车牌识别




# 工具

## Keras OCR

Keras-OCR 提供现成的 OCR 模型和端到端训练管道，以构建新的 OCR 模型

在这种情况下，使用预训练的模型，它对任务非常有效。Keras-OCR 将自动下载探测器和识别器的预训练权重。
当通过 Keras-OCR 传递图像时，它将返回一个 `(word，box)` 元组，其中框(`box`)包含四个角的坐标 `(x，y)`

# 参考

* [使用CV2和Keras OCR从图像中删除文本](https://mp.weixin.qq.com/s/I1_2xGMGxBkUK7gMyrE9gQ)
* [keras-ocr](https://keras-ocr.readthedocs.io/en/latest/index.html)
