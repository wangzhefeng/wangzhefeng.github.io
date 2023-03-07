---
title: 时间序列预测-机器学习
author: 王哲峰
date: '2022-04-25'
slug: timeseries-forecast_ml
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

- [传统时序预测方法的问题](#传统时序预测方法的问题)
- [时间序列的机器学习预测](#时间序列的机器学习预测)
  - [时间序列初始数据集](#时间序列初始数据集)
  - [时间序列数据集处理](#时间序列数据集处理)
    - [截面数据](#截面数据)
    - [时间序列数据](#时间序列数据)
  - [时间序列模型构建](#时间序列模型构建)
- [时间序列预测方法](#时间序列预测方法)
  - [单步预测](#单步预测)
  - [多步预测](#多步预测)
    - [递归预测](#递归预测)
    - [多目标回归](#多目标回归)
    - [递归多目标回归](#递归多目标回归)
  - [多变量预测](#多变量预测)
  - [递归预测](#递归预测-1)
  - [直接预测 Direct Forecasting](#直接预测-direct-forecasting)
  - [堆叠预测 Stacking Forecasting](#堆叠预测-stacking-forecasting)
  - [修正预测 Rectified Forecasting](#修正预测-rectified-forecasting)
- [参考文章](#参考文章)
</p></details><p></p>

# 传统时序预测方法的问题

1. 对于时序本身有一些性质上的要求，需要结合预处理来做拟合，不是端到端的优化
2. 需要对每条序列做拟合预测，性能开销大，数据利用率和泛化能力堪忧，无法做模型复用
3. 较难引入外部变量，例如影响销量的除了历史销量，还可能有价格，促销，业绩目标，天气等等
4. 通常来说多步预测能力比较差

正因为这些问题，实际项目中一般只会用传统方法来做一些 baseline，
主流的应用还是属于下面要介绍的机器学习方法

# 时间序列的机器学习预测

做时间序列预测，传统模型最简便，比如 Exponential Smoothing 和 ARIMA。
但这些模型一次只能对一组时间序列做预测，比如预测某个品牌下某家店的未来销售额。
而现实中需要面对的任务更多是：预测某个品牌下每家店的未来销售额。
也就是说，如果这个品牌在某个地区一共有 100 家店，那我们就需要给出这100家店分别对应的销售额预测值。
这个时候如果再用传统模型，显然不合适，毕竟有多少家店就要建多少个模型，太累。
而且在大数据时代，我们面对的数据往往都是高维的，如果仅使用这些传统方法，很多额外的有用信息可能会错过

所以，如果能用机器学习算法对这 “100家店” 一起建模，那么整个预测过程就会高效很多。
但是，用机器学习算法做时间序列预测，处理的数据会变得很需要技巧。
对于普通的截面数据，在构建特征和分割数据（比如做 K-fold CV）的时候不需要考虑时间窗口。
而对于时间序列，时间必须考虑在内，否则模型基本无效。因此在数据处理上，后者的复杂度比前者要大

## 时间序列初始数据集

![img](images/data.png)

对于时间序列数据来说，训练集即为历史数据，测试集即为新数据。历史数据对应的时间均在时间分割点之前，
新数据对应的时间均在分割点之后。历史数据和新数据均包含 `$N$` 维信息（如某品牌每家店的地理位置、销售的商品信息等），
但前者比后者多一列数据：预测目标变量(Target/Label)，即要预测的对象(如销售额)

基于给出的数据，预测任务是：根据已有数据，预测测试集的 Target（如，
根据某品牌每家店 2018 年以前的历史销售情况，预测每家店 2018 年 1 月份头 15 天的销售额）

## 时间序列数据集处理

在构建预测特征上，截面数据和时间序列数据遵循的逻辑截然不同

### 截面数据

首先来看针对截面数据的数据处理思路：

![img](images/cross_section.png)

对于截面数据来说，训练集数据和测试集数据在时间维度上没有区别，
二者唯一的区别是前者包含要预测的目标变量，而后者没有该目标变量。
一般来说，在做完数据清洗之后，我们用 “N维数据” 来分别给训练集、测试集构建 M 维预测特征（维度相同），
然后用机器学习算法在训练集的预测特征和 Target 上训练模型，
最后通过训练出的模型和测试集的预测特征来计算预测结果（测试集的 Target）。
此外，为了给模型调优，我们一般还需要从训练集里面随机分割一部分出来做验证集

### 时间序列数据

时间序列的处理思路则有所不同，时间序列预测的核心思想是：用过去时间里的数据预测未来时间里的 Target：

![img](images/ts.png)

所以，在构建模型的时候，首先，将所有过去时间里的数据，即训练集里的 N 维数据和 Target 都应该拿来构建预测特征。
而新数据本身的 N 维数据也应该拿来构建预测特征。前者是历史特征（对应图上的预测特征 A），后者是未来特征（对应图上的预测特征 B），
二者合起来构成总预测特征集合。最后，用预测模型和这个总的预测特征集合来预测未来 Target

看到这里，一个问题就产生了：既然所有的数据都拿来构建预测特征了，那预测模型从哪里来？没有 Target 数据，模型该怎么构建？

你可能会说，那就去找 Target 呗。对，没有错。但这里需要注意，我们要找的不是未来时间下的 Target（毕竟未来的事还没发生，根本无从找起），
而是从过去时间里构造 “未来的” Target，从而完成模型的构建。这是在处理时间序列上，逻辑最绕的地方

## 时间序列模型构建

用机器学习算法构造时间序列预测模型，关键的思路在于，通过时间滑窗，人为地构造 “未来” Target，来给算法进行学习

![img](images/ml.png)

和之前一样，从时间的角度上来看，有历史数据和新数据。但这里，不能简单地把历史数据作为训练集、把新数据作为测试集

1. 首先，在历史数据上，通过截取不同时间窗口的数据来构造一组或几组数据
    - 比如，历史数据是 2017 年 1 月到 12 月每家店每天的销售数据，那么可以截取 3 组数据（见上图的深绿、浅绿部分）：
        - 2017 年 1 月到 10 月的数据
        - 2017 年 2 月到 11 月的数据
        - 2017 年 3 月到 12 月的数据
2. 然后，人为地给每组数据划分历史窗口（对应上图的深绿色部分）和未来窗口（对应上图的浅绿色部分）
    - 比如，对于 2017 年 1 月到 10 月的数据，把 1 月到 9 月作为历史窗口，10 月作为未来窗口，以此类推
3. 接着，分别给每组数据构建预测特征，包括历史特征（预测特征 A）和未来特征（预测特征 B）。
   而此时，每组数据还有预测 Target

这个时候，把得到的所有预测特征（例子里是三组预测特征）都合并起来作为训练集特征、
把所有预测 Target（例子里是三组预测 Target）合并起来作为训练集 Target，之后就可以构建机器学习模型了

有了训练集和训练模型，还差测试集。测试集的构建遵循之前的数据处理逻辑，拿历史数据构建历史特征，
拿新数据构建未来特征，然后把这些特征加入到从训练集上训练出的预测模型中去，即可得到任务需要的最终预测值。
这里需要注意，划多少个时间窗口因数据而异。此外，数据的历史窗口（图上深绿部分）和未来窗口（图上浅绿部分）可以是定长也可以是变长，
看具体情况

# 时间序列预测方法

机器学习方法处理时间序列问题的基本思路就是吧时间序列切分成一段历史训练窗口和未来的预测窗口，
对于预测窗口中的每一条样本，基于训练窗口的信息来构建特征，转化为一个表格类预测问题来求解

实际场景中，一般需要确定几个参数：

1. 历史窗口的大小
    - 即预测未来时，要参考过去多少时间的信息作为输入。太少可能信息量不充分，
      太多则会引入早期不相关的信息(比如疫情前的信息可能目前就不太适用了)
2. 预测点 gap 的大小
    - 即预测未来时，是从 T+1 开始预测，还是 T+2，T+3，这与现实的业务场景有关，
      例如像补货场景，预测 T+1 的销量，可能已经来不及下单补货了，
      所以需要扩大这个提前量，做 T+3 甚至更多提前时间的预测
3. 预测窗口的大小
    - 即需要连续预测多长的未来值。比如从 T+1 开始一直到 T+14 都需要预测输出。
      这一点也跟实际的业务应用场景有关

也会看到一些额外加入时序预处理步骤的方法，比如先做 STL 分解再做建模预测。
尝试下来这类方法总体来说效果并不明显，但对于整个 pipeline 的复杂度却有较大的增加，
对于 AutoML、模型解释等工作都造成了一定的困扰，所以实际项目中应用的也比较少

## 单步预测

在时间序列预测中的标准做法是使用滞后的观测值 `$x_{t-1}$` 作为输入变量来预测当前的时间的观测值 `$x_{t}$`

## 多步预测

使用过去的观测序列 `$\{\ldots, x_{t-2}, x_{t-1}\}$` 预测未来的观测序列 `$\{x_{t}, x_{t+1}, \ldots\}$`

大多数预测问题都被定义为单步预测，根据最近发生的事件预测系列的下一个值。
时间序列多步预测需要预测未来多个值， 提前预测许多步骤具有重要的实际优势，
多步预测减少了长期的不确定性。但模型试图预测更远的未来时，模型的误差也会逐渐增加

### 递归预测

> Recursive Forecasting

多步预测最简单的方法是递归形式，训练单个模型进行单步预测，
然后将模型与其先前的预测结果作为输入得到后续的输出

```python
from sklearn.linear_model import LinearRegression

# using a linear regression for simplicity. any regression will do.
recursive = LinearRegression()
# training it to predict the next value of the series (t+1)
recursive.fit(X_tr, Y_tr['t+1'])
# setting up the prediction data structure
predictions = pd.DataFrame(np.zeros(Y_ts.shape), columns=Y_ts.columns)

# making predictions for t+1
yh = recursive.predict(X_ts)
predictions['t+1'] = yh

# iterating the model with its own predictions for multi-step forecasting
X_ts_aux = X_ts.copy()
for i in range(2, Y_tr.shape[1] + 1):
    X_ts_aux.iloc[:, :-1] = X_ts_aux.iloc[:, 1:].values
    X_ts_aux['t-0'] = yh

    yh = recursive.predict(X_ts_aux)

    predictions[f't+{i}'] = yh
```

上述代码逻辑在 sktime 中也可以找到相应的实现：[sktime.RecursiveTimeSeriesRegressionForecaster](https://www.sktime.org/en/stable/api_reference/auto_generated/sktime.forecasting.compose.RecursiveTimeSeriesRegressionForecaster.html)

递归方法只需要一个模型即可完成整个预测范围，且无需事先确定预测范围

但此种方法用自己的预测作为输入，这导致误差逐渐累计，对长期预测的预测性能较差

### 多目标回归

多目标回归为每一个预测结果构建一个模型，如下是一个使用案例：

```python
from sklearn.multioutput import MultiOutputRegressor

direct = MultiOutputRegressor(LinearRegression())
direct.fit(X_tr, Y_tr)
direct.predict(X_ts)
```

scikit-learn 的 MultiOutputRegressor 为每个目标变量复制了一个学习算法。
在这种情况下，预测方法是 LinearRegression

此种方法避免了递归方式中错误传播，但多目标预测需要更多的计算资源。
此外多目标预测假设每个点是独立的，这是违背了时序数据的特点

### 递归多目标回归

递归多目标回归结合了多目标和递归的思想。为每个点建立一个模型。
但是在每一步的输入数据都会随着前一个模型的预测而增加

```python
from sklearn.multioutput import RegressorChain

dirrec = RegressorChain(LinearRegression())
dirrec.fit(X_tr, Y_tr)
dirrec.predict(X_ts)
```

这种方法在机器学习文献中被称为 chaining。scikit-learn 通过 RegressorChain 类为其提供了一个实现

## 多变量预测

多元时间序列，即每个时间有多个观测值：

`$$\{X_{t} = (x_{t}^{a}, x_{t}^{b}, x_{t}^{c}, \ldots)\}_{t}^{T}$$`

这意味着通过不同的测量手段得到了多种观测值，并且希望预测其中的一个或几个值。
例如，可能有两组时间序列观测值 `$\{x_{t-1}^{a}, x_{t-2}^{a}, \ldots\}$`，
`$\{x_{t-1}^{b}, x_{t-2}^{b}, \ldots\}$`，希望分析这组多元时间序列来预测 `$x_{t}^{a}$` 

基于以上场景，许多监督学习的方法可以应用在时间序列的预测中，
在运用机器学习模型时，可以把时间序列模型当成一个回归问题来解决，
比如 svm/xgboost/逻辑回归/回归树/...

## 递归预测 

## 直接预测 Direct Forecasting

## 堆叠预测 Stacking Forecasting

## 修正预测 Rectified Forecasting



# 参考文章

* [Time Series Forecasting as Supervised Learning](https://machinelearningmastery.com/time-series-forecasting-supervised-learning/)
* [How to Convert a Time Series to a Supervised Learning Problem in Python](https://machinelearningmastery.com/convert-time-series-supervised-learning-problem-python/)
* [Machine Learning Strategies for Time Series Forecasting](https://link.springer.com/chapter/10.1007%2F978-3-642-36318-4_3)
* [slide](http://di.ulb.ac.be/map/gbonte/ftp/time_ser.pdf)
* [Machine Learning for Sequential Data: A Review](http://web.engr.oregonstate.edu/~tgd/publications/mlsd-ssspr.pdf)
* [如何用Python将时间序列转换为监督学习问题](https://cloud.tencent.com/developer/article/1042809)
* [How To Resample and Interpolate Your Time Series Data With Python](https://machinelearningmastery.com/resample-interpolate-time-series-data-python/)
* [时间序列预测](https://mp.weixin.qq.com/s?__biz=Mzg3NDUwNTM3MA==&mid=2247484974&idx=1&sn=d841c644fd9289ad5ec8c52a443463a5&chksm=cecef3dbf9b97acd8a9ededc069851afc00db422cb9be4d155cb2c2a9614b2ee2050dc7ab4d7&scene=21#wechat_redirect)
* [机器学习与时间序列预测](https://www.jianshu.com/p/e81ab6846214)

