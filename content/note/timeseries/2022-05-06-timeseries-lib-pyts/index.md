---
title: Lib pyts 时间序列分类
author: 王哲峰
date: '2022-05-07'
slug: timeseries-lib-pyts
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
</style>

<details><summary>目录</summary><p>

- [时间序列分类介绍](#时间序列分类介绍)
- [时间序列分类算法](#时间序列分类算法)
  - [传统方法](#传统方法)
    - [DTW(Dynamic time warping) 和 KNN](#dtwdynamic-time-warping-和-knn)
    - [SAXVSM](#saxvsm)
    - [BOSSVS](#bossvs)
    - [Learning Shapelets](#learning-shapelets)
  - [基于特征的方法](#基于特征的方法)
    - [Time Series Forest](#time-series-forest)
    - [Time Series Bag-of-Features](#time-series-bag-of-features)
  - [深度学习方法](#深度学习方法)
    - [MLP & FCN & ResNet](#mlp--fcn--resnet)
    - [LSTM_FCN & BiGRU-CNN](#lstm_fcn--bigru-cnn)
    - [MC-CNN(multi-channel CNN) & MCNN(multi-scale CNN)](#mc-cnnmulti-channel-cnn--mcnnmulti-scale-cnn)
- [时间序列库 pyts](#时间序列库-pyts)
  - [pyts Demo](#pyts-demo)
  - [pyts 安装](#pyts-安装)
  - [特征提取](#特征提取)
    - [Shapelet Transform](#shapelet-transform)
    - [](#)
- [参考](#参考)
</p></details><p></p>


# 时间序列分类介绍



# 时间序列分类算法

## 传统方法

### DTW(Dynamic time warping) 和 KNN

在深度学习大量使用之前，在时间序列分类方面，DTW和KNN的结合是一种很有用的方法。
在做两个不同的时间序列匹配的时候，虽然我们可以通过肉眼发现它们之间存在着很高的相似性，
但是由于部分位置的拉伸或者压缩，或者沿着时间轴上存在着平移，对两者直接计算欧氏距离效果一般不是很好，
而定义规则进行对齐又无法适用实际应用中的多种情况，而DTW是在1970年提出了，
其思想在一定程度上解决了该问题，通过动态规划的方法，来计算损失（即距离），最小的作为输出结果


```python
from pyts.classification import KNeighborsClassifier
from pyts.datasets import load_gunpoint

X_train, X_test, y_train, y_test = load_gunpoint(return_X_y = True)

clf = KNeighborsClassifier(metric = "dtw")
clf.fit(X_train, y_train)
clf.score(X_test, y_test)
```

### SAXVSM


```python
from pyts.classification import SAXVSM
from pyts.datasets import load_gunpoint

X_train, X_test, y_train, y_test = load_gunpoint(return_X_y = True)

clf = SAXVSM(window_size = 34, sublinear_tf = False, use_idf = False)
clf.fit(X_train, y_train)
clf.score(X_test, y_test)
```


### BOSSVS

```python
from pyts.classification import BOSSVS
from pyts.datasets import load_gunpoint

X_train, X_test, y_train, y_test = load_gunpoint(return_X_y = True)

clf = BOSSVS(window_size = 28)
clf.fit(X_train, y_train)
clf.score(X_test, y_test)
```

### Learning Shapelets


```python
from pyts.classification import LearningShapelets
from pyts.datasets import load_gunpoint

X_train, X_test, y_train, y_test = load_gunpoint(return_X_y = True)

clf = LearningShapelets(random_state = 42, tol = 0.01)
clf.fit(X_train, y_train)
clf.score(X_test, y_test)
```

## 基于特征的方法

这一类的方法都是一些通过某种度量关系来提取相关特征的方法，如词袋法，
通过找到该时间序列中是否有符合已有词袋中的特征（序列的样子），
将一个序列用词来表示，再对词进行分类。而其他的基于特征的方法都是利用了类似的方法，
如提取统计量，基于规则等，再通过分类模型进行分类


### Time Series Forest

```python
from pyts.classification import TimeSeriesForest
from pyts.datasets import load_gunpoint

X_train, X_test, y_train, y_test = load_gunpoint(return_X_y = True)

clf = TimeSeriesForest(random_state = 43)
clf.fit(X_train, y_train)
clf.score(X_test, y_test)
```

### Time Series Bag-of-Features

```python
from pyts.classification import TimeSeriesForest
from pyts.datasets import load_gunpoint

X_train, X_test, y_train, y_test = load_gunpoint(return_X_y = True)

clf = TimeSeriesForest(random_state = 43)
clf.fit(X_train, y_train)
clf.score(X_test, y_test)
```

## 深度学习方法

### MLP & FCN & ResNet


### LSTM_FCN & BiGRU-CNN


### MC-CNN(multi-channel CNN) & MCNN(multi-scale CNN)








# 时间序列库 pyts

* [GitHub](https://github.com/johannfaouzi/pyts)
* [Doc](https://pyts.readthedocs.io/en/latest/)

## pyts Demo

```python
from pyts.classification import BOSSVS
from pyts.datasets import load_gunpoint

X_train, X_test, y_train, y_test = load_gunpoint(return_X_y = True)
clf = BOSSVS(window_size = 28)
clf.fit(X_train, y_train)
clf.score(X_test, y_test)
```

## pyts 安装

```bash
$ pip install pyts
```

## 特征提取

### Shapelet Transform


### 



# 参考

* [[1](Time Series Classifification from Scratch with Deep Neural Networks: A Strong Baseline]()
* [[2] DTW](https://blog.csdn.net/raym0ndkwan/article/details/45614813)
* [[3] DTW & KNN](https://nbviewer.jupyter.org/github/markdregan/K-Nearest-Neighbors-with-Dynamic-Time-Warping/blob/master/K_Nearest_Neighbor_Dynamic_Time_Warping.ipynb)
* [[4] BiGRU CNN](http://www.doc88.com/p-0334856528441.html)
* [[5] LSTM Fully Convolutional Networks for Time Series Classification]()
