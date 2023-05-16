---
title: Sktime 
subtitle: 机器学习
author: 王哲峰
date: '2022-05-01'
slug: timeseries-lib-sktime
categories:
  - timeseries
tags:
  - machinelearning
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

- [安装](#安装)
  - [pip](#pip)
  - [conda](#conda)
- [时间序列数据类型](#时间序列数据类型)
- [时间序列任务](#时间序列任务)
  - [时间序列预测](#时间序列预测)
  - [时间序列分类](#时间序列分类)
  - [时间序列回归](#时间序列回归)
  - [时间序列聚类](#时间序列聚类)
  - [时间序列标注](#时间序列标注)
- [参考](#参考)
</p></details><p></p>

> A unified framework for machine learning with time series.

# 安装

## pip

```bash
$ pip install sktime
$ pip install sktime[all_extras]
```

## conda

```bash
$ conda install -c conda-forge sktime
$ conda install -c conda-forge sktime-all-extras
```

# 时间序列数据类型

在 sktime 时间序列中，数据可以指单变量、多变量或面板数据，
不同之处在于时间序列变量之间的数量和相互关系，以及观察每个变量的实例数

* 单变量时间序列数据是指随时间跟踪单个变量的数据
* 多变量时间序列数据是指针对同一实例随时间跟踪多个变量的数据。例如，一个国家/地区的多个季度经济指标或来自同一台机器的多个传感器读数
* 面板时间序列数据是指针对多个实例跟踪变量（单变量或多变量）的数据。例如，多个国家/地区的多个季度经济指标或多台机器的多个传感器读数




# 时间序列任务

## 时间序列预测


## 时间序列分类


## 时间序列回归


## 时间序列聚类


## 时间序列标注

> time series annotation，时间序列标注是指异常值检测、变化点检测和分割







# 参考

* [Doc](http://www.sktime.net/en/latest/)
* [GitHub](https://github.com/sktime/sktime)

