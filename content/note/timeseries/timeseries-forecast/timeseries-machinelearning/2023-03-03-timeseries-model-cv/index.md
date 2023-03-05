---
title: 时间序列交叉验证
author: 王哲峰
date: '2023-03-03'
slug: timeseries-model-cv
categories:
  - timeseries
tags:
  - model
---

在时序问题上，需要特别注意不能做随机 split，而需要在时间维度上做前后的 split，
以保证与实际预测应用时的情况一致


# 参考

* [时间序列交叉验证](https://lonepatient.top/2018/06/10/time-series-nested-cross-validation.html)
* [时间序列交叉验证](https://lonepatient.top/2018/06/10/time-series-nested-cross-validation.html)
* [9个时间序列交叉验证方法的介绍和对比](https://mp.weixin.qq.com/s/JpZV2E102FU94_aj-b-sOA)
* [样本组织](https://mp.weixin.qq.com/s?__biz=Mzk0NDE5Nzg1Ng==&mid=2247492305&idx=1&sn=c4c9783ee3ab85a8f7a813e803f15177&chksm=c32afb5ef45d7248d539aca50cff13a840ff53bb2400166ea146256675b08b93419be3f8fadc&scene=21#wechat_redirect)

