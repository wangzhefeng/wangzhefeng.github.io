---
title: NLP-文本分类
author: wangzf
date: '2022-04-05'
slug: nlp-ml-text-classification
categories:
  - nlp
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

- [TODO](#todo)
- [文本分类任务简介](#文本分类任务简介)
- [朴素贝叶斯分类模型](#朴素贝叶斯分类模型)
  - [理论](#理论)
  - [实现](#实现)
</p></details><p></p>

# TODO

* [文本分类算法综述](https://mp.weixin.qq.com/s/nwYDRoMKYZ0xal9bRXCfHw)
* [一文总结文本分类必备经典模型](https://mp.weixin.qq.com/s/f5SkoWD4BY_HDWfPi5R5ng)

# 文本分类任务简介

一般, 文本分类大致分为以下几个步骤:

1. 定义阶段
    - 定义数据以及分类体系, 具体分为哪些类别, 需要哪些数据
2. 数据预处理
    - 对文档做分词、去停用词等准备工作
3. 数据提取特征
    - 对文档矩阵进行降维, 提取训练集中最有用的特征
4. 模型训练阶段
    - 选择具体分类模型以及算法, 训练出文本分类器
5. 评测阶段
    - 在测试集上测试并评价分类器性能
6. 应用阶段
    - 应用性能最高的分类模型对待分类文档进行分类

# 朴素贝叶斯分类模型

## 理论


## 实现

