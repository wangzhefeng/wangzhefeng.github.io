---
title: TensorFlow 张量和计算图
author: 王哲峰
date: '2022-09-10'
slug: dl-tensorflow-tensor
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

- [张量](#张量)
  - [张量数据结构](#张量数据结构)
  - [张量数据操作](#张量数据操作)
  - [张量数学运算](#张量数学运算)
- [计算图](#计算图)
  - [静态计算图](#静态计算图)
  - [动态计算图](#动态计算图)
  - [AutoGraph](#autograph)
    - [AutoGraph 使用规范](#autograph-使用规范)
    - [AutoGraph 机制原理](#autograph-机制原理)
    - [AutoGraph 和 tf.Module](#autograph-和-tfmodule)
</p></details><p></p>

张量和计算图是 TensorFlow 的核心概念

# 张量

TensorFlow 的基本数据结构是张量。张量即多维数组，TensorFlow 的张量和 Numpy 中的 array 很类似。
从行为特性来开，有两种类型的张量:
    
* 常量 Constant
    - 常量的值在计算图中不可以被重新赋值
* 变量 Variable
    - 变量可以在计算图中用 `assign` 等算子重新赋值

## 张量数据结构


## 张量数据操作


## 张量数学运算









# 计算图

计算图由节点(nodes)和线(edges)组成:

* 节点表示操作符 Operation，或者称之为算子
* 线表示计算间的依赖

实现表示有数据的传递依赖，传递的数据即张量，虚线通常可以表示控制依赖，即执行先后顺序

TensorFlow 有三种计算图的构建方式

* 静态计算图
* 动态计算图
* AutoGraph

## 静态计算图

TensorFlow 1.0 采用的是静态计算图，需要先使用 TensorFlow 的各种算子创建计算图，
然后再开启一个会话 Session，显式执行计算图

在 TensorFlow 1.0 中，使用静态计算图分两步:

1. 定义计算图
2. 在会话中执行计算图



## 动态计算图

TensorFlow 2.0 采用的是动态计算图，即每使用一个算子后，
该算子会被动态加入到隐含的默认计算图中立即执行得到结果，
而无需开启 Session

* 使用动态计算图(Eager Excution)的好处是方便调试程序
    - 动态计算图会让 TensorFlow 代码的表现和 Python 原生代码的表现一样，
      写起来就像 Numpy 一样，各种日志打印，控制流全部都是可以使用的
* 使用动态图的缺点是运行效率相对会低一点
    - 因为使用动态图会有许多词 Python 进程和 TensorFlow 的 C++ 进程之间的通信
    - 而静态计算图构建完成之后几乎全部在 TensorFlow 内核上使用 C++ 代码执行，效率更高。
      此外，静态图会对计算步骤进行一定的优化，去除和结果无关的计算步骤 

## AutoGraph

如果需要在 TensorFlow 中使用静态图，
可以使用 `@tf.function` 装饰器将普通 Python 函数转换成对应的 TensorFlow 计算图构建代码。
运行该函数就相当于在 TensorFlow 1.0 中用 Session 执行代码。使用 `@tf.function` 构建静态图的方式叫做 AutoGraph


### AutoGraph 使用规范


### AutoGraph 机制原理


### AutoGraph 和 tf.Module


