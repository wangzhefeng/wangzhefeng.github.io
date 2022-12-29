---
title: Lib Kats
author: 王哲峰
date: '2022-04-29'
slug: timeseries-lib-kats
categories:
  - timeseries
tags:
  - ml
---

> Kats is a toolkit to analyze time series data, a lightweight, easy-to-use, and generalizable framework to perform time series analysis. Time series analysis is an essential component of Data Science and Engineering work at industry, from understanding the key statistics and characteristics, detecting regressions and anomalies, to forecasting future trends. Kats aims to provide the one-stop shop for time series analysis, including detection, forecasting, feature extraction/embedding, multivariate analysis, etc. 

* [Doc](https://facebookresearch.github.io/Kats/)

Kats 是 Facebook 布的一个专门为了时间序列服务的工具库。它作为一个 Toolkit 包，提供了四种简易且轻量化的 API:

* 预测(forecasting)
    - 封装了 10+models，主要是传统的时间序列模型，这些模型有支持 ensemble 的 API，结合时间序列特征的功能甚至可以做 meta-learning
* 检测(detection)
    - 官方叫做 detection，大致是对时间序列做类似于异常检测，change point detection 之类的检测
* 时间序列特征(TSFeature)
    - API 十分简单，可以得到 65 个时间序列相关的 features
* 模拟 simulator
    - 可以按照某些时序特征比如 seasonality 去创造时间序列来方便实验


# forecasting



# Detection


# TSFeatures


# Utilities


# 参考

* [Facebook时序工具库Kats](https://zhuanlan.zhihu.com/p/394686861)

