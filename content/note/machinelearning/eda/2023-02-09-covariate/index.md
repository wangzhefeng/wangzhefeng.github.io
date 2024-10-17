---
title: 交叉变量探索分析
author: wangzf
date: '2023-02-09'
slug: covariate
categories:
  - data analysis
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
img {
    pointer-events: none;
}
</style>

<details><summary>目录</summary><p>

- [特征和标签关系分析](#特征和标签关系分析)
  - [标签为数值变量](#标签为数值变量)
    - [类别特征](#类别特征)
      - [pandas 绘图](#pandas-绘图)
      - [boxplot 绘图](#boxplot-绘图)
    - [数值特征](#数值特征)
      - [person 相关系数](#person-相关系数)
      - [scatter 绘图](#scatter-绘图)
      - [regplot 绘图](#regplot-绘图)
    - [时间特征](#时间特征)
      - [plot 绘图](#plot-绘图)
      - [statsmodels 分解](#statsmodels-分解)
  - [标签为二元变量](#标签为二元变量)
    - [类别特征](#类别特征-1)
    - [数值特征](#数值特征-1)
  - [标签为多元变量](#标签为多元变量)
    - [类别特征](#类别特征-2)
    - [数值特征](#数值特征-2)
- [特征和特征关系分析](#特征和特征关系分析)
- [参考](#参考)
</p></details><p></p>


在数据竞赛或者建模的时候，将变量的交叉分析划分为下面的三大块：

* 特征和标签的关系分析：重点探讨特征与标签的关系，特征与标签是否强相关等等
* 特征和特征的关系分析：重点观察特征之间的冗余关系，是否是衍生关系等等
* 可视化的技巧：特征之间的分析是做不完的，很多情况下一般也就只会看到三阶左右的特征关系，
  但是当数据特征字段上百的时候，即使是二阶交叉分析也需要进行好几天的分析，
  一般选取感兴趣的进行观测探索

# 特征和标签关系分析

特征与标签的关系分析是我们最为关注的，希望可以了解到我们每个特征与标签之间的关系，依据我们的经验，

很多时候如果特征与标签存在较强的关系，那么大概率该特征是存在非常大的价值的，也就是我们说的重要的特征。
那么我们一般需要观察哪些指标呢？

## 标签为数值变量

> 回归

### 类别特征

关于类别变量与数值标签的关系，一般会观察下面的结果：

* 每个类别情况下对应的均值，这个可以直接使用 pandas 进行绘制
* 均值反映的信息并不十分详细，如果希望得到更加具体的分布，可以使用 boxplot 进行绘制

如果不同类别之间的标签分布相差较大，则说明该类别信息是非常有有价值的，
如果所有类别的标签都是一样的分布，则该类别信息的区分度相对较低

#### pandas 绘图

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
%matplotlib inline

df = pd.read_csv('./data/house_train.csv') 

var = 'MSZoning'
label = 'SalePrice'
df.groupby(var)[label].mean().plot(kind = 'bar')
```

#### boxplot 绘图

```python
import matplotlib.pyplot  as plt
import seaborn as sns
%matplotlib inline

plt.figure(figsize = [10,6])

var = 'MSZoning'
sns.boxplot(x = var, y = label, data = df)
```

### 数值特征

关于数值变量与数值标签的关系，一般会观察下面的结果：

* 数值特征与数值标签的 pearson 相关系数；如果该数值的绝对值越大，
  往往说明该特征能为模型带来非常大的帮助
* 观察数值特征与数值标签的时候，一般采用散点图即可，也可以使用 regplot 绘制出拟合的曲线

#### person 相关系数

```python
df[["LotArea", label]].corr("pearson")
```

#### scatter 绘图

```python
import matplotlib.pyplot  as plt
%matplotlib inline

plt.figure(figsize = [10, 6])
plt.scatter(x = "LotArea", y = label, data = df)
plt.xlabel("LotArea")
plt.ylabel(label)
```

#### regplot 绘图

```python
import matplotlib.pyplot  as plt
import seaborn as sns
%matplotlib inline

plt.figure(figsize = [10, 6])
sns.regplot(x = df["LotArea"], y = df[label])
```

### 时间特征

关于时间与数值标签，主要希望随着时间变化，数据是否表现除了某些特殊的模式(周期性等)，
以及是否出现了明显的异常现象等等。时间变量与数值变量的可视化直接使用 plot 函数即可

#### plot 绘图

```python
plt.figure(figsize = [10,6])

# 用 index 模拟时间
plt.plot(
    df.index, df[label].values, 
    color = 'black', 
    linestyle = '--', 
    linewidth = '1', 
    label = label
)
plt.xlabel('Time', fontsize = 14)   
plt.ylabel(label, fontsize = 16)    
```

#### statsmodels 分解

```python
from statsmodels.tsa.seasonal import seasonal_decompose

decomposition = seasonal_decompose(ts, freq, two_sided = False)
trend = decomposition.trend
seasonal = decomposition.seasonal
residual = decomposition.resid
decomposition.plot()
plt.show()
```

## 标签为二元变量

> 二分类

### 类别特征

关于类别变量与二元标签的关系可以直接通过 barplot 函数进行可视化，
如果不同类之间的分布差不大，那说明该类别变量大概率是意义不大的

```python
df = pd.read_csv('./data/titanic_train.csv') 

var = 'Pclass'
label = 'Survived'
df.groupby(var)[label].mean()

plt.figure(figsize = [10, 6])
sns.barplot(x = bar, y = label, data = df)
```

### 数值特征

数值变量与二元标签的关系一般可以通过下面的两种方式分析：

* 对数值变量进行分桶，然后基于类别变量与二元标签的关系进行分析
* 使用 boxplot 函数，观测在不同标签下，数值特征的分布差异

```python
plt.figure(figsize = [10, 6])
sns.boxplot(df[label], df['Fare'])
```

## 标签为多元变量

> 多分类

在观测标签为多分类的问题时，因为标签是多个类别的，可以通过两种策略对其进行观察：

* 将多分类转化为多个二分类然后进行观测，这么做最大的问题是需要分析量大大增加了，会使得问题变得更加繁琐
* 采用和数值变量与二元标签的策略对模型进行观测

```python
multi_label = 'y_generated'
df['y_generated'] = np.random.randint(low = 0.5, high = 3.5, size = df.shape[0])
```

### 类别特征

直接使用 countplot，将 hue 的位置设置为标签的名称即可

```python
plt.figure(figsize = [10,6])
sns.countplot(x = 'Pclass', hue = multi_label, data = df)
plt.ylabel('Number of Occurrences', fontsize = 12)
plt.xlabel(multi_label, fontsize = 12)
plt.show()
```

### 数值特征

直接使用 boxplot 即可，观测在每个类处数值变量的分布情况。
如果所有类处的数值变量分布都类似，那可能该数值变量带来的影响会相对较小，反之影响较大

```python
plt.figure(figsize = [10,6])
sns.boxplot(x = multi_label, y = 'Age', data = df)
plt.ylabel('Age distribution', fontsize = 12)
plt.xlabel(multi_label, fontsize = 12)
plt.show()
```

# 特征和特征关系分析









# 参考

* [Analyzing Relationships Between Variables]()
* [Correlation and regression]()
* [Comprehensive data exploration with Python]()
* [Python 预测[周期性时间序列]]()
* [Data analysis and feature extraction with Python]()
* [pandas.DataFrame.corr]()
* [How To... Calculate Spearman's Rank Correlation Coefficient (By Hand)]()
* [spearman相关系数]()
* [皮尔逊相关系数]()
* [Pytanic]()
* [Tutorial: Python Subplots]()
* [Simple Exploration Notebook - 2σ Connect]()
* [数据探索分析-变量交叉分析](https://mp.weixin.qq.com/s?__biz=Mzk0NDE5Nzg1Ng==&mid=2247493301&idx=1&sn=9e0b8719083510c8625d37facb8c691d&chksm=c32aff3af45d762c27cee8b53099ce93877228c8646477c25b249c351f3d9a4cef8680defc51&scene=21#wechat_redirect)

