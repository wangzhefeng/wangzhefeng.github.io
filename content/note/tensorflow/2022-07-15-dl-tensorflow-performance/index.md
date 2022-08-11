---
title: TensorFlow Performance
author: 王哲峰
date: '2022-07-15'
slug: dl-tensorflow-performance
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

- [使用 tf.function 提升性能](#使用-tffunction-提升性能)
  - [@tf.funciton: 图执行模式](#tffunciton-图执行模式)
  - [@tf.function 基础使用方法](#tffunction-基础使用方法)
  - [@tf.function 内在机制](#tffunction-内在机制)
  - [AutoGraph: 将 Python 控制流转化为 TensorFlow 计算图](#autograph-将-python-控制流转化为-tensorflow-计算图)
  - [使用传统的 tf.Session](#使用传统的-tfsession)
- [分析 TenforFlow 的性能](#分析-tenforflow-的性能)
- [图优化](#图优化)
- [混合精度](#混合精度)
</p></details><p></p>

# 使用 tf.function 提升性能

## @tf.funciton: 图执行模式

虽然目前 TensorFlow 默认的即时执行模式具有灵活及易调试的特性, 但在特定的场合, 
例如追求高性能或部署模型时, 依然希望使用图执行模式, 将模型转换为高效的 TensorFlow 图模型。

TensorFlow 2 提供了 ``bashtf.function` 模块, 结合 AutoGraph 机制, 使得我们仅需加入一个简单的
`@tf.function` 修饰符, 就能轻松将模型以图执行模式运行。

## @tf.function 基础使用方法


`@tf.function` 的基础使用非常简单, 只需要将我们希望以图执行模式运行的代码封装在一个函数内, 
并在函数前面加上 `@tf.function` 即可.


## @tf.function 内在机制






## AutoGraph: 将 Python 控制流转化为 TensorFlow 计算图




## 使用传统的 tf.Session


# 分析 TenforFlow 的性能


# 图优化


# 混合精度
