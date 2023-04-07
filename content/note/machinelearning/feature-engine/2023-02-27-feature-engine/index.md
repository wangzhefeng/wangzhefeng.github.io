---
title: 特征工程概览
author: 王哲峰
date: '2023-02-27'
slug: feature-engine
categories:
  - feature engine
tags:
  - machinelearning
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

- [文章](#文章)
- [特征工程](#特征工程)
- [结构化数据](#结构化数据)
- [非结构化数据](#非结构化数据)
- [样本相关](#样本相关)
</p></details><p></p>

# 文章

* [全网写特征工程最通透的](https://mp.weixin.qq.com/s/SVjZvbeJUaBag-NiELGDVw)
* [使用sklearn做单机特征工程](https://www.cnblogs.com/jasonfreak/p/5448385.html)

数据建模就是从数据中学习到洞见(insights)的过程，这个过程其实是很曲折的，要经过数据的表达、模型的学习两部。其中：数据的表达就是原始数据经过 clean 和 transformer 得到 feaaures 的过程，即为特征工程

# 特征工程

在机器学习中，所有数据最终都会转化为数值型特征，所有特征工程都会归结为某种数值型特征工程技术。
特征工程，顾名思义，是对原始数据进行一系列工程处理，将其提炼为特征，作为输入供算法和模型使用

模型和特征工程是一个整体，二者相辅相成，我们在设计模型的时候希望模型可以充分地吸收数据，
从数据集中自动挖掘出与我们标签相关的信息。尽可能做到在不需要任何特征工程的情况下拿到好的效果。
但从目前的情况来看，还没有一个模型可以做到这些，所以依然需要通过借助专家特征工程的方式对数据进行某种方式的组合和变化，
将数据处理为模型易于吸收的形式。在模型的弱势区，即学习不好的地方对其进行帮助，从而使得我们模型可以获得更好的效果

换言之，特征工程很大程度上是在帮助模型学习，在模型学习不好的地方或者难以学习的地方，
采用特征工程的方式帮助其学习，通过人为筛选、人为构建组合特征让模型原本很难学好的东西可以更加容易地进行学习、
进而拿到更好的效果

下面是一些业内关于特征工程在机器学习邻域的箴言：

> * Garbage in, garbage out.
> * 对于一个机器学习问题，数据和特征往往决定了结果的上限，
>   而模型、算法的选择及优化则是在逐步接近这个上限。
> * 没有最好的模型, 只有最合适的模型。
> * 一个模型所能提供的信息一般来源于两个方面：一是训练数据中蕴含的信息；
>   二是在模型的形成过程中(包括构造、学习、推理等)，人们提供的先验信息。

# 结构化数据

* **数值型特征**
    - 特征合理性检查
        - 量级
        - 正负
    - 特征尺度
        - 尺度: 
            - 最大值, 最小值
            - 是否横跨多个数量级
    - 特征分布
        - 对数变换
        - Box-Cox变换
    - 特征组合
        - 交互特征
        - 多项式特征
    - 特征选择
        - PCA
* **类别型特征**
    - 分类任务目标变量
    - 类别特征
* **时间序列数据**
    - 时间序列插值
    - 时间序列降采样
    - 时间序列聚合计算
    - 时间序列平滑

# 非结构化数据

* **文本数据**
    - 扁平化
    - 过滤
    - 分块
* **图像数据**
* **音频数据**
* **视屏数据**

# 样本相关

* **样本筛选（采样）**
    - 欠采样
    - 过采样
    - 过采样和欠采样结合
* **样本组织**

