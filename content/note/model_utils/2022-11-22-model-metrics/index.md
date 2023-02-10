---
title: 模型评价指标
author: 王哲峰
date: '2022-11-22'
slug: model-metrics
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
</style>

<details><summary>目录</summary><p>

- [二分类](#二分类)
  - [准确率 Accuracy](#准确率-accuracy)
    - [准确率](#准确率)
    - [损失函数](#损失函数)
  - [Precision 和 recall 以及 F1](#precision-和-recall-以及-f1)
    - [举例](#举例)
    - [损失函数](#损失函数-1)
  - [Log Loss](#log-loss)
    - [定义](#定义)
    - [损失函数](#损失函数-2)
  - [混淆矩阵 以及 ROC 和 AUC](#混淆矩阵-以及-roc-和-auc)
    - [损失函数](#损失函数-3)
  - [Normalized Gini Coefficient](#normalized-gini-coefficient)
    - [定义](#定义-1)
    - [损失函数](#损失函数-4)
    - [实现](#实现)
- [多分类](#多分类)
  - [Categorization Accuracy](#categorization-accuracy)
    - [准确率](#准确率-1)
    - [损失函数](#损失函数-5)
  - [Multi Class Log Loss](#multi-class-log-loss)
    - [Log Loss](#log-loss-1)
    - [损失函数](#损失函数-6)
  - [MAP-Mean Average Precision](#map-mean-average-precision)
    - [MAP](#map)
    - [损失函数](#损失函数-7)
  - [Mean F1](#mean-f1)
    - [Mean F1](#mean-f1-1)
    - [损失函数](#损失函数-8)
  - [Average Jaccard Index](#average-jaccard-index)
    - [Jaccard Index](#jaccard-index)
    - [损失函数](#损失函数-9)
- [回归](#回归)
  - [RMSE](#rmse)
  - [MSE](#mse)
  - [MAE](#mae)
  - [RMSLE](#rmsle)
  - [MAPE](#mape)
- [时间序列预测](#时间序列预测)
  - [MAPE](#mape-1)
  - [WMAPE](#wmape)
  - [SMAPE](#smape)
- [排序](#排序)
- [聚类](#聚类)
  - [Rank Index](#rank-index)
  - [Mutual Information based scores](#mutual-information-based-scores)
  - [Homogeneity, completeness and V-measure](#homogeneity-completeness-and-v-measure)
  - [Fowlkes-Mallows scores](#fowlkes-mallows-scores)
  - [Silhouette Coefficient](#silhouette-coefficient)
  - [Calinski-Harabasz Index](#calinski-harabasz-index)
  - [Davies-BouIdin Index](#davies-bouidin-index)
  - [Contingency Matrix](#contingency-matrix)
  - [Pair Confusion Matrix](#pair-confusion-matrix)
- [参考](#参考)
</p></details><p></p>

# 二分类

## 准确率 Accuracy

### 准确率

定义：

`$$Accuracy = \frac{1}{N}\sum_{i=1}^{N}I(y_{i} = \bar{y}_{i})$$`

其中：

* `$N$` 为测试样本个数
* `$y_{i}\in \{0, 1\}$` 表示样本 `$i$` 对应的真实标签
* `$\bar{y}_{i}\in \{0, 1\}$` 表示样本 `$i$` 对应的预测结果

### 损失函数

针对准确率问题，目前常采用的损失函数为 Binary Log Loss/Binary Cross Entropy，其数学形式如下：

`$$LogLoss = -\frac{1}{N}\sum_{i=1}^{N}\Big(y_{i}log p_{i} + (1 - y_{i})log(1 - p_{i})\Big)$$`

其中：

* `$p_{i}$` 为模型对第 `$i$` 个样本的预测概率

## Precision 和 recall 以及 F1

一般来说：

* precision 是检索出来的条目(文档、网页)有多少是准确的(准确的定义为所关心的正好被检索出来)

`$$precision = \frac{某类被正确分类的关系实例个数}{被判定为某类的关系实例总数}$$`

* recall 就是所有准确的条目有多少被检索出来了

`$$recall = \frac{某类被正确分类的关系实例个数}{测试集中某类的关系实例总数}$$`

* F-Measure
    - Precision 和 Recall 指标有时候会出现的矛盾的情况，这样就需要综合考虑他们，
      最常见的方法就是 F-Measure(又称为 F-Score)。F-Measure 是 Precision 和 Recall 加权调和评价：

`$$F = \frac{(\alpha^{2}+1)Precision \times Recall}{\alpha^{2}(Precision + Recall)}$$`

* F1
    - 当 `$\alpha = 1$` 时，就是最常见的 F1，因此，F1 综合了 precision 和 recall 的结果，
      当 F1 较高时，则能说明试验方法比较有效

`$$F1 = \frac{2 \cdot precision \cdot recall}{precision + recall}$$`

我们总是希望检索结果 precision 越高越好，同时 recall 也越高越好，但事实上这两者在某些情况下有矛盾的。
比如，极端情况下：只搜索出了一个结果，且是准确的，那么 precision 就是 100%，但是 recall 却很低；
如果把所有结果都返回，那么比如 recall 是 100%，但是 precision 就会很低。
因此 ，在不同的场合中需要自己判断希望 precision 比较高或是 recall 比较高
如果是做实验研究，可以绘制 precision-recall 曲线来帮助分析

### 举例

举个例子，某池塘有 1400 条鲤鱼，300 只虾，300 只鳖。
现在以捕鲤鱼为目的，撒一大网，逮着 700 条鲤鱼，200 只虾，100 只鳖，
那么 precision, recall, F1 分别如下：

`$$precision = \frac{700}{700 + 200 + 100}=70\%$$`

`$$recall = \frac{700}{1400} = 50\%$$`

`$$F1 = \frac{2 \times 70\% \times 50\%}{70\% + 50\%} = 58.3\%$$`

不妨看看如果把池子里的所有的鲤鱼、虾和鳖都一网打尽，这些指标又有何变化：

`$$precision = \frac{1400}{1400 + 300 + 300}=70\%$$`

`$$recall = \frac{1400}{1400} = 100\%$$`

`$$F1 = \frac{2 \times 70\% \times 100\%}{70\% + 100\%} = 82.35\%$$`

由此可见，准确率(precision)是评估捕获的成果(被判定为关心的类别样本)中目标成果(关心的类别样本)所占的比例；
召回率(recall)就是从关注领域(关心的类别样本)中，召回目标类别(被判定为关心的类别中关心的类别)的比例；
F1 值则综合这两者指标的评估指标，用于综合反映整体领域的指标

### 损失函数

和准确率指标优化类似，此处使用 Binary Cross Entropy 进行优化即可，
不同之处在于，在得到最终的预测概率之后，需要通过一些策略寻找最优阈值

还有的时候会对损失函数进行加权优化；例如标签为 1 的样本的权重就设置大一些等

## Log Loss

### 定义

`$$LogLoss = -\frac{1}{N}\sum_{i=1}^{N}\Big(y_{i}log p_{i} + (1 - y_{i})log(1 - p_{i})\Big)$$`

其中：

* `$p_{i}$` 为模型对第 `$i$` 个样本的预测概率

### 损失函数

因为 Logloss 是可以直接进行优化的函数，一般我们会直接基于 LogLoss 函数进行优化

## 混淆矩阵 以及 ROC 和 AUC

AUC(Area Under Curve)被定义为 ROC 曲线下与坐标轴围成的面积，
一般以 TPR 为 y 轴，以 FPR 为 x 轴，就可以得到 ROC 曲线。
AUC 的数值都不会大于 1。又由于 ROC 曲线一般都处于 `$y=x$` 这条直线的上方，
所以 AUC 的取值范围在 0.5 和 1 之间。AUC 越接近 1.0，检测方法真实性越高。
等于 0.5 时，一般就无太多应用价值了。

其中关于：FPR(False Positive Rate)以及 TPR(True Positive Rate)的数学计算公式为

`$$$$`

### 损失函数

最为常见的还是基于 LogLoss 函数的优化

## Normalized Gini Coefficient

### 定义

`$$Gini = 2 \times AUC - 1$$`

### 损失函数

LogLoss

### 实现

```python
from sklearn.metrics import roc_auc_score
import numpy as np 


def gini(actual, pred, cmpcol = 0, sortcol = 1):
    assert (len(actual) == len(pred))
    all = np.asarray(np.c_[actual, pred, np.arange(len(actual))], dtype=np.float)
    all = all[np.lexsort((all[:, 2], -1 * all[:, 1]))]
    totalLosses = all[:, 0].sum()
    giniSum = all[:, 0].cumsum().sum() / totalLosses

    giniSum -= (len(actual) + 1) / 2.
    return giniSum / len(actual)


def gini_normalized(actual, pred):
    return gini(actual, pred) / gini(actual, actual)

p = np.array([0.9, 0.3, 0.8, 0.75, 0.65, 0.6, 0.78, 0.7, 0.05, 0.4, 0.4, 0.05, 0.5, 0.1, 0.1]) 
y = np.array([1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0])

2 * roc_auc_score(y_score = p, y_true = y) - 1
gini_normalized(actual = y, pred = p)
```

# 多分类

## Categorization Accuracy

### 准确率

定义：

`$$logloss = -\frac{1}{N}\sum_{i=1}^{N}I(y_{i} = p_{i})$$`

其中：

* `$N$` 为测试样本的个数
* `$y_{i}$` 为第 `$i$` 个样本的类别标签
* `$p_{i}$` 为预测的第 `$i$` 个样本的类别

### 损失函数

`$$Loss = -\frac{1}{N}\sum_{i=1}^{N}\sum_{j=1}^{K} y_{i,k}log(p_{i,k})$$`

其中：

* `$y_{i,k}$` 表示第 `$i$` 个样本标签为 `$k$` 的情况，
  如果标签为 `$k$` 则是 1，反之为 0
* `$p_{i,k}$` 表示模型预测第 `$i$` 个样本属于 `$k$` 的概率

## Multi Class Log Loss

### Log Loss

定义：

`$$logloss = -\frac{1}{N}\sum_{i=1}^{N}\sum_{i=1}^{N}\sum_{i=1}^{M}y_{i,j}log(p_{i,j})$$`

其中：

* `$N$` 为测试样本的个数
* `$M$` 为类标签的个数

### 损失函数

针对准确率问题，目前常采用的损失函数为 Multiclass Log Loss，其数学形式如下：

`$$logloss = -\frac{1}{N}\sum_{i=1}^{N}\sum_{i=1}^{N}\sum_{i=1}^{M}y_{i,j}log(p_{i,j})$$`

其中：

* `$N$` 为测试样本的个数
* `$M$` 为类标签的个数

## MAP-Mean Average Precision

### MAP

定义：

`$$MAP = \frac{1}{|U|}\sum_{u=1}^{|U|}\frac{1}{min(A, m)}\sum_{k=1}^{min(n, A)}P(k)$$`

其中：

* `$|U|$` 为用户的个数
* `$P(k)$` 为在截止点 `$k$` 处的精度(Precision)
* `$n$` 是预测物品的数量
* `$M$` 是给定用户购买物品的数量，如果 `$M=0$`，则精度定义为 0

### 损失函数

使用 Sigmoid Cross Entropy，注意与其它常用的多分类损失函数的区别

## Mean F1

### Mean F1

定义：

`$F1 = \frac{2pr}{p+r}$`

其中：

* `$p = \frac{tp}{tp+fp}$`
* `$r = \frac{tp}{tp+fn}$`

### 损失函数

Mean Square Loss


## Average Jaccard Index

### Jaccard Index

两个区域 A 和 B 的 Jaccard Index 可以表示为：

`$$Jaccard = \frac{TP}{TP + FP + FN} = \frac{A \cap B}{A \cup B} = \frac{|A \cap B|}{|A| + |B| - |A \cap B|}$$`

其中：

* `$TP$` 表示 True Positive 的面积
* `$FP$` 表示 False Positive 的面积
* `$FN$` 表示 False Negative 的面积

### 损失函数

Sigmoid

# 回归

## RMSE

> Root Mean Square Error

RMSE 可以直接优化的函数，一般默认选用平方损失函数进行优化即可，很多工具包里面也称之为 L2 损失

## MSE

> Mean Square Error

MSE 是可以直接优化的函数，所以直接默认选用平方损失函数进行优化即可，很多工具包里面也称之为 L2 损失

## MAE

> Mean Absolute Error

MAE 在诸多工具包中也已经有对应的优化函数，直接使用即可，有些包中也会称之为 L1 损失函数

## RMSLE

> Root Mean Squared Logarithmic Error

先对数据做 log1p 转化，然后使用 L2 损失函数直接求解即可

## MAPE

> Mean Absolute Percentage Error

如果采用神经网络对此类问题进行优化，可以直接自己定义 MAPE 的 Loss

# 时间序列预测

在指标方面，作为一个回归问题，可以使用 MAE，MSE 等方式来计算。
但这类 metric 受到具体预测数值区间范围不同，
展现出来的具体误差值区间也会波动很大。比如预测销量可能是几万到百万，
而预测车流量可能是几十到几百的范围，那么这两者预测问题的 MAE 可能就差距很大，
很难做多个任务间的横向比较。所以实际问题中，
经常会使用对数值量纲不敏感的一些 metric，尤其是 SMAPE 和 WMAPE 这两种

这类误差计算方法在各类不同的问题上都会落在 0~1 的区间范围内，
方便来进行跨序列的横向比较，十分方便。在实际项目中还会经常发现，
很多真实世界的时序预测目标，如销量，客流等，
都会形成一个类似 tweedie 或 poisson 分布的情况。
如果用 WMAPE 作为指标，模型优化目标基本可以等价为 MAE（优化目标为中位数），
则整体的预测就会比平均值小（偏保守）。在很多业务问题中，
预测偏少跟预测偏多造成的影响是不同的，所以实际在做优化时，
可能还会考察整体的预测偏差（总量偏大或偏小），
进而使用一些非对称 loss 来进行具体的优化

## MAPE


## WMAPE

`$$WMAPE = \frac{\sum_{t=1}^{n}|A_{t} - F_{t}|}{\sum_{t=1}^{n}|A_{t}|}$$`



## SMAPE




# 排序

# 聚类

## Rank Index

## Mutual Information based scores

## Homogeneity, completeness and V-measure

## Fowlkes-Mallows scores

## Silhouette Coefficient

## Calinski-Harabasz Index

## Davies-BouIdin Index

## Contingency Matrix

## Pair Confusion Matrix

# 参考

* [准确率(Precision)、召回率(Recall)、F值(F-Measure)的简要说明](https://blog.csdn.net/huacha__/article/details/80329707?spm=1001.2014.3001.5502)
* [损失函数softmax_cross_entropy、binary_cross_entropy、sigmoid_cross_entropy之间的区别与联系](https://blog.csdn.net/sjyttkl/article/details/103958639)
* https://github.com/nagadomi/kaggle-coupon-purchase-prediction
* https://github.com/viig99/stackexchange-transfer-learning
* https://deepsense.io/deep-learning-for-satellite-imagery-via-image-segmentation/
* https://arxiv.org/pdf/1505.04597.pdf
* https://github.com/toshi-k/kaggle-satellite-imagery-feature-detection
* [多分类相关指标优化​](https://mp.weixin.qq.com/s?__biz=Mzk0NDE5Nzg1Ng==&mid=2247492485&idx=1&sn=440c944d690f4df4dd4279aea07d2cfc&chksm=c32afa0af45d731cf4af9bc6dd848dcd38d724c57cd9bacad16dd8d5db19b925ac7ea3ae4d89&scene=21#wechat_redirect)
* [Intuitive Explanation of the Gini Coefficient](https://theblog.github.io/post/gini-coefficient-intuitive-explanation/)
* [Optimizing probabilities for best MCC](https://www.kaggle.com/cpmpml/optimizing-probabilities-for-best-mcc)
* [Choosing the correct error metric: MAPE vs. sMAPE](https://towardsdatascience.com/choosing-the-correct-error-metric-mape-vs-smape-5328dec53fac)
* [What is the different MAE, MAPE, MSE, and RMSE](https://www.kaggle.com/learn-forum/52081)
* [mape和smape，基于mae的回归评价指标](https://zhuanlan.zhihu.com/p/259662864)
* [Model Fit Metrics](https://www.kaggle.com/residentmario/model-fit-metrics)
