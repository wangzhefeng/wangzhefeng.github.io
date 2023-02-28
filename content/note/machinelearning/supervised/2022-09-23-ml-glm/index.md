---
title: 广义线性模型
author: 王哲峰
date: '2022-09-23'
slug: ml-glm
categories:
  - machinelearning
tags:
  - model
---

<style>
h1 {
    background-color: #2B90B6;
    background-image: linear-gradient(45deg, #4EC5D4 10%, #146b8c 20%);
    background-size: 100%;
    -webkit-background-clip: text;
    -moz-background-clip: text;
    -webkit-text-fill-color: transparent;
    -moz-text-fill-color: transparent;
}
h2 {
    background-color: #2B90B6;
    background-image: linear-gradient(45deg, #4EC5D4 10%, #146b8c 20%);
    background-size: 100%;
    -webkit-background-clip: text;
    -moz-background-clip: text;
    -webkit-text-fill-color: transparent;
    -moz-text-fill-color: transparent;
}
h3 {
    background-color: #2B90B6;
    background-image: linear-gradient(45deg, #4EC5D4 10%, #146b8c 20%);
    background-size: 100%;
    -webkit-background-clip: text;
    -moz-background-clip: text;
    -webkit-text-fill-color: transparent;
    -moz-text-fill-color: transparent;
}
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

- [广义线性模型概览](#广义线性模型概览)
  - [模型介绍](#模型介绍)
- [广义线性模型原理](#广义线性模型原理)
  - [一般线性模型](#一般线性模型)
  - [线性回归](#线性回归)
  - [Logistic 回归](#logistic-回归)
  - [Probit 回归](#probit-回归)
  - [Possion 回归](#possion-回归)
- [参考](#参考)
</p></details><p></p>

# 广义线性模型概览

## 模型介绍

在统计学上，广义线性模型(Generalized Linear Model, GLM)是一种应用灵活的线性回归模型。
该模型允许因变量的偏差分布有除了正态分布之外的其他分布。
此模型假设实验者所测量的随机变量的分布函数与实验中系统性效应（即非随机的效应）可经由链接函数（link function）建立可解释其相关性的函数

# 广义线性模型原理

## 一般线性模型

有些人可能会把一般线性模式和广义线性模式给弄混了。一般线性模式可视为广义线性模式的一个链接函数为恒等的特例。
一般线性模式有着悠长的发展历史。广义线性模式具非恒等链接函数者有着渐近一致的结果

## 线性回归


## Logistic 回归

## Probit 回归


## Possion 回归



# 参考

* [wiki-广义线性模型](https://zh.wikipedia.org/zh-cn/%E5%BB%A3%E7%BE%A9%E7%B7%9A%E6%80%A7%E6%A8%A1%E5%9E%8B)
* https://zhangzhenhu.github.io/blog/glm/source/%E5%B9%BF%E4%B9%89%E7%BA%BF%E6%80%A7%E6%A8%A1%E5%9E%8B/content.html
* https://zhuanlan.zhihu.com/p/22876460
* https://xg1990.com/blog/archives/304
* https://andrewwang.rbind.io/courses/bayesian_statistics/notes/Ch9_h.pdf

