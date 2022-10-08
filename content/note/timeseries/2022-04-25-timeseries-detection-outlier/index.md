---
title: 时间序列异常值检测与处理
author: 王哲峰
date: '2022-04-25'
slug: timeseries-outlier-detection
categories:
  - timeseries
tags:
  - ml
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

- [异常值检测算法](#异常值检测算法)
</p></details><p></p>




异常检测(Outlier Detection)也称为异常挖掘或异常检测, 
是从大量数据中提取隐含在其中的人们事先不知道的但又是潜在的有用的信息和知识的过程

异常检测需要解决两个主要问题: 

* 在给定的数据集合中定义什么样的数据是异常的
* 找到一个有效的方法来检测这样的异常数据
  
按异常在时间序列中的不同表现形式, 时间序列异常可以分为3种: 

* 序列异常
* 点异常
* 模式异常

时间序列异常检测方法主要包括: 

1. 基于窗口的方法
2. 基于距离的方法
3. 基于密度的方法
4. 基于支持向量机的方法
5. 基于聚类的方法

## 异常值检测算法

