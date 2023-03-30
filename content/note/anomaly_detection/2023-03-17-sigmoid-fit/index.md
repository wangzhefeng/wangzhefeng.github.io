---
title: Sigmoid 曲线拟合
author: 王哲峰
date: '2023-03-17'
slug: sigmoid-fit
categories:
  - timeseries
tags:
  - model
---


Sigmoid 参数曲线：

`$$f(x) = \frac{a}{b + e^{-(dx - c)}}$$`

损失函数：

`$$S(p) = \sum_{i=1}^{m}(y_{i} - f(x_{i}, p))$$`




# 参考

* [风电异常数据识别与清洗竞赛冠军方案分享](https://mp.weixin.qq.com/s?__biz=Mzk0NDE5Nzg1Ng==&mid=2247490892&idx=1&sn=bdd9aea219596e172636cccee4c6dc84&chksm=c32904c3f45e8dd56024943db0e3dc49efcfe21b493d8ab14b8085df7c490d751a0af5b2e618&scene=21#wechat_redirect)

