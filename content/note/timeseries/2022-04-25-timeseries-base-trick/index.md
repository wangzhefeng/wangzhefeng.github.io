---
title: 时间序列分析-技巧
author: 王哲峰
date: '2022-04-25'
slug: timeseries-base-trick
categories:
  - timeseries
tags:
  - ml
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

- [时间序列元特征](#时间序列元特征)
  - [元特征抽取](#元特征抽取)
  - [预测](#预测)
  - [示例](#示例)
</p></details><p></p>



# 时间序列元特征

在时间序列等相关的问题中，除了许多传统的时间序列相关的统计特征之外，
还有一类非常重要的特征，这类特征并不是基于手工挖掘的，而是由机器学习模型产出的，
但更为重要的是，它往往能为模型带来巨大的提升

对时间序列抽取元特征，一共需要进行两个步骤：

* 抽取元特征
* 将元特征拼接到一起重新训练预测得到最终预测的结果

## 元特征抽取

元特征抽取部分，操作如下：

* 先把数据按时间序列分为两块
* 使用时间序列的第一块数据训练模型得到模型 1
* 使用时间序列的第二块数据训练模型得到模型 2
* 使用模型 1 对第二块的数据进行预测得到第二块数据的元特征
* 使用模型 2 对测试集进行预测得到测试集的元特征

![img](images/meta-feature.jpg)


## 预测

将元特征作为新的特征，与原来的数据进行拼接，重新训练新的模型，
并用新训练的模型预测得到最终的结果

![img](images/meta-feature-forecast.jpg)

## 示例

* [ ] TODO