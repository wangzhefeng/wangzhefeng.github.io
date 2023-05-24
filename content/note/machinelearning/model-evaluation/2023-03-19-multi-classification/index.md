---
title: 多分类评价指标
author: 王哲峰
date: '2022-11-22'
slug: multi-classification
categories:
  - machinelearning
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

- [Categorization Accuracy](#categorization-accuracy)
- [Multi Class Log Loss](#multi-class-log-loss)
- [MAP-Mean Average Precision](#map-mean-average-precision)
- [Mean F1](#mean-f1)
- [Average Jaccard Index](#average-jaccard-index)
- [参考](#参考)
</p></details><p></p>


# Categorization Accuracy

> Categorization Accuracy，分类准确率

定义：

`$$logloss = -\frac{1}{N}\sum_{i=1}^{N}I(y_{i} = p_{i})$$`

其中：

* `$N$` 为测试样本的个数
* `$y_{i}$` 为第 `$i$` 个样本的类别标签
* `$p_{i}$` 为预测的第 `$i$` 个样本的类别



`$$Loss = -\frac{1}{N}\sum_{i=1}^{N}\sum_{j=1}^{K} y_{i,k}log(p_{i,k})$$`

其中：

* `$y_{i,k}$` 表示第 `$i$` 个样本标签为 `$k$` 的情况，
  如果标签为 `$k$` 则是 1，反之为 0
* `$p_{i,k}$` 表示模型预测第 `$i$` 个样本属于 `$k$` 的概率

# Multi Class Log Loss

定义：

`$$logloss = -\frac{1}{N}\sum_{i=1}^{N}\sum_{i=1}^{N}\sum_{i=1}^{M}y_{i,j}log(p_{i,j})$$`

其中：

* `$N$` 为测试样本的个数
* `$M$` 为类标签的个数

针对准确率问题，目前常采用的损失函数为 Multiclass Log Loss，其数学形式如下：

`$$logloss = -\frac{1}{N}\sum_{i=1}^{N}\sum_{i=1}^{N}\sum_{i=1}^{M}y_{i,j}log(p_{i,j})$$`

其中：

* `$N$` 为测试样本的个数
* `$M$` 为类标签的个数

# MAP-Mean Average Precision

定义：

`$$MAP = \frac{1}{|U|}\sum_{u=1}^{|U|}\frac{1}{min(A, m)}\sum_{k=1}^{min(n, A)}P(k)$$`

其中：

* `$|U|$` 为用户的个数
* `$P(k)$` 为在截止点 `$k$` 处的精度(Precision)
* `$n$` 是预测物品的数量
* `$M$` 是给定用户购买物品的数量，如果 `$M=0$`，则精度定义为 0

使用 Sigmoid Cross Entropy，注意与其它常用的多分类损失函数的区别

# Mean F1

定义：

`$F1 = \frac{2pr}{p+r}$`

其中：

* `$p = \frac{tp}{tp+fp}$`
* `$r = \frac{tp}{tp+fn}$`

Mean Square Loss

# Average Jaccard Index

两个区域 A 和 B 的 Jaccard Index 可以表示为：

`$$Jaccard = \frac{TP}{TP + FP + FN} = \frac{A \cap B}{A \cup B} = \frac{|A \cap B|}{|A| + |B| - |A \cap B|}$$`

其中：

* `$TP$` 表示 True Positive 的面积
* `$FP$` 表示 False Positive 的面积
* `$FN$` 表示 False Negative 的面积

# 参考

* [多分类相关指标优化​](https://mp.weixin.qq.com/s?__biz=Mzk0NDE5Nzg1Ng==&mid=2247492485&idx=1&sn=440c944d690f4df4dd4279aea07d2cfc&chksm=c32afa0af45d731cf4af9bc6dd848dcd38d724c57cd9bacad16dd8d5db19b925ac7ea3ae4d89&scene=21#wechat_redirect)

