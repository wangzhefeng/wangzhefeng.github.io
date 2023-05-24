---
title: TODS 异常检测
author: 王哲峰
date: '2023-03-05'
slug: timeseries-lib-tods
categories:
  - timeseries
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

- [TODS 异常检测简介](#tods-异常检测简介)
  - [TODS 特点](#tods-特点)
  - [TODS 异常检测模块](#tods-异常检测模块)
  - [TODS 异常检测场景](#tods-异常检测场景)
- [TODS 使用](#tods-使用)
  - [安装](#安装)
- [参考](#参考)
</p></details><p></p>

# TODS 异常检测简介

TODS 是一个全栈的自动化机器学习系统，主要针对多变量时间序列数据的异常检测。
该系统可以处理三种常见的时间序列异常检测场景：点的异常检测(异常是时间点)、模式的异常检测(异常是子序列)、
系统的异常检测(异常是时间序列的集合)。TODS 提供了一系列相应的算法

## TODS 特点

TODS 具有如下特点：

* 全栈式机器学习系统
    - 支持从数据预处理、特征提取、到检测算法和人为规则每一个步骤并提供相应的接口
* 广泛的算法支持
    - 包括 PyOD 提供的点的异常检测算法、最先进的模式的异常检测算法(例如 DeepLog，Telemanon)，
      以及用于系统的异常检测的集合算法
* 自动化的机器学习
    - 旨在提供无需专业知识的过程，通过自动搜索所有现有模块中的最佳组合，基于给定数据构造最优管道

## TODS 异常检测模块

TODS 提供了详尽的用于构建基于机器学习的异常检测系统的模块，它们包括：

![img](images/tods.png)

* 数据处理(data processing)
    - data loading
    - data filtering
    - data validation
    - data binarization
    - timestamp transformation
* 时间序列处理(time series processing)
    - seasonality/trend decomposition
    - timeseries transformation
    - timeseries scaling
    - timeseries smoothing
* 特征分析(feature analysis)
    - time domain
    - frequency domain
    - laten factor models
* 检测算法(detection algorithms)
    - 传统方法
        - IForest
        - AutoRegression
    - 启发式方法
        - HotSax algorithm
        - Matrix Profile
    - 深度学习方法
        - RNN-LSTM
        - GAN
        - VAE
    - ensemble 方法
* 强化模块( reinforcement module)
    - rule-based filtering
    - active learning based methods

这些模块所提供的功能包括常见的数据预处理、时间序列数据的平滑或变换、从时域或频域中抽取特征、多种多样的检测算法以及让人类专家来校准系统。
通过这些模块提供的功能包括：通用数据预处理、时间序列数据平滑/转换、从时域/频域中提取特征、各种检测算法，以及涉及人类专业知识来校准系统

## TODS 异常检测场景

![img](images/tranditional_type.png)

![img](images/tranditional_type3.png)

TODS 的这些模块功能可以让时间序列数据执行三种常见的异常值检测场景：

* 逐点检测(时间点作为异常值)
    - 当时间序列中存在潜在的系统故障或小故障时，通常会出现逐点异常值。
      这种异常值存在于全局(与整个时间序列中的数据点相比)或局部(与相邻点相比)的单个数据点上
* 模式检测(子序列作为异常值)
    - 局部异常值通常出现在特定上下文中，具有相同值的数据点如果不在特定上下文中显示，则不会被识别为异常值。
      检测局部异常值的常用策略是识别上下文(通过季节性趋势分解、自相关)，
      然后应用统计/机器学习方法(例如 AutoRegression、IsolationForest、OneClassSVM)来检测异常值
* 系统检测(时间序列集作为异常值)
    - 全局异常值通常很明显，检测全局异常值的常见做法是获取数据集的统计值(例如，最小值/最大值/平均值/标准偏差)并设置检测异常点的阈值

当数据中存在异常行为时，通常会出现模式异常值。模式异常值是指与其他子序列相比其行为异常的时间序列数据的子序列(连续点)。
检测模式异常值的常见做法，包括不和谐分析(例如，矩阵配置文件、HotSAX)和子序列聚类

Discords 分析利用滑动窗口将时间序列分割成多个子序列，并计算子序列之间的距离(例如，欧几里德距离)以找到时间序列数据中的不一致。
子序列聚类也将子序列分割应用于时间序列数据，并采用子序列作为每个时间点的特征，其中滑动窗口的大小为特征的数量。
然后，采用无监督机器学习方法，例如聚类（例如，KMeans、PCA）或逐点异常值检测算法来检测模式异常值

当许多系统之一处于异常状态时，系统异常值会不断发生，其中系统被定义为多元时间序列数据。
检测系统异常值的目标是从许多类似的系统中找出处于异常状态的系统。例如，从具有多条生产线的工厂检测异常生产线。
检测这种异常值的常用方法是执行逐点和模式异常值检测以获得每个时间点/子序列的异常值分数，
然后采用集成技术为每个系统生成整体异常值分数以进行比较和检测

# TODS 使用

## 安装

```bash
$ pip install tods
```

```bash
$ pip install -e git+https://github.com/datamllab/tods.git@dev#egg=tods 
```



# 参考

* [TODS：功能强大的多元时间序列异常检测工具](https://mp.weixin.qq.com/s/seAk389JPZccD24iljzmXg)
* [TODS GitHub](https://github.com/datamllab/tods/tree/benchmark)
* [TODS Doc](https://tods-doc.github.io/)

