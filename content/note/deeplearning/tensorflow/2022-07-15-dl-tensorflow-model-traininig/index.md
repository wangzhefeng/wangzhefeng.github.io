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
</style>

<details><summary>目录</summary><p>

- [回调函数](#回调函数)
- [模型训练](#模型训练)
  - [使用 GPU 训练模型](#使用-gpu-训练模型)
    - [使用单 GPU 训练模型](#使用单-gpu-训练模型)
    - [使用多 GPU 训练模型](#使用多-gpu-训练模型)
  - [使用 TPU 训练模型](#使用-tpu-训练模型)
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

# 参考资料

* [Focal Loss](https://zhuanlan.zhihu.com/p/80594704)
