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

- [时间序列回归](#时间序列回归)
- [时间序列预测方式](#时间序列预测方式)
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
- [时间序列模型选择](#时间序列模型选择)
  - [模型](#模型)
  - [local 模型和 global 模型](#local-模型和-global-模型)
- [参考文章](#参考文章)
</p></details><p></p>

传统时序预测方法的一些问题：

1. 对于时序本身有一些性质上的要求，需要结合预处理来做拟合，不是端到端的优化
2. 需要对每条序列做拟合预测，性能开销大，数据利用率和泛化能力堪忧，无法做模型复用
3. 较难引入外部变量，例如影响销量的除了历史销量，还可能有价格，促销，业绩目标，天气等等
4. 通常来说多步预测能力比较差

正因为这些问题，实际项目中一般只会用传统方法来做一些 baseline，
主流的应用还是属于下面要介绍的机器学习方法

# 时间序列回归

时序预测模型与回归预测模型不同，时序预测模型依赖于数值在时间上的先后顺序，是回归模型中的一部分。
简单来说，时间序列的回归分析需要分析历史数据，找到历史数据演化中的模式和特征，
其主要分为线性回归分析和非线性回归分析两种类型

回归分析多采用机器学习方法，我们首先需要明确机器学习(或深度学习)模型构建与验证的主体思路：

1. 分析数据构建数据特征，将数据转化为特征样本集合
2. 明确样本与标签，划分训练集与测试集
3. 比较不同模型在相同的训练集中的效果，或是相同模型的不同参数在同一个训练集中拟合的效果
4. 在验证样本集中验证模型的准确度，通过相关的结果评估公式选择表现最好同时没有过拟合的模型

近年来时间序列预测方法，多采用机器学习方式。机器学习的方法，主要是构建样本数据集，
采用“时间特征”到“样本值”的方式，通过有监督学习，学习特征与标签之前的关联关系，
从而实现时间序列预测。常用的场景有以下几种

# 时间序列预测方式

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

上述代码逻辑在sktime中也可以找到相应的实现：[sktime.RecursiveTimeSeriesRegressionForecaster](https://www.sktime.org/en/stable/api_reference/auto_generated/sktime.forecasting.compose.RecursiveTimeSeriesRegressionForecaster.html)

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



# 时间序列模型选择

## 模型

模型这块，基本上没有什么花样，大家的主流选择基本都是 GBDT 和 NN。
个人最常使用的选择是 LightGBM 和 fastai，然后选择好时序验证方式，
做自动参数优化就可以了（比如使用 Optuna 或 FLAML）。
lgbm 的训练速度快，而且在某些业务特征比较重要的情况下，往往能达到比神经网络更好更稳定的效果。
而 NN 的主要优势在类别变量的表达学习上，理论上可以达到更好的 embedding 表示。
此外 NN 的 loss 设计上也会比较灵活，相对来说 lgbm 的 loss 或者多目标学习限制条件就比较多了。
总体来说，目前最常见的选择仍然是树模型一族

* lightgbm
* fastai

## local 模型和 global 模型

有一个值得注意的考量点在于 local 模型与 global 模型的取舍。
前面提到的经典时序方法中都属于 local 模型，即每一个序列都要构建一个单独的模型来训练预测；
而提到的把所有数据都放在一起训练则是 global 模型的方式。实际场景中，
可能需要预测的时序天然就会有很不一样的规律表现，比如科技类股票，跟石油能源类股票的走势，
波动都非常不一样，直接放在一起训练反而可能导致整体效果下降。
所以很多时候我们要综合权衡这两种方式，在适当的层级做模型的拆分训练。
深度学习领域有一些工作如 DeepFactor 和 FastPoint 也在自动适配方面做了些尝试

# 参考文章

* [Time Series Forecasting as Supervised Learning](https://machinelearningmastery.com/time-series-forecasting-supervised-learning/)
* [How to Convert a Time Series to a Supervised Learning Problem in Python](https://machinelearningmastery.com/convert-time-series-supervised-learning-problem-python/)
* [Machine Learning Strategies for Time Series Forecasting](https://link.springer.com/chapter/10.1007%2F978-3-642-36318-4_3)
* [slide](http://di.ulb.ac.be/map/gbonte/ftp/time_ser.pdf)
* [Machine Learning for Sequential Data: A Review](http://web.engr.oregonstate.edu/~tgd/publications/mlsd-ssspr.pdf)
* [如何用Python将时间序列转换为监督学习问题](https://cloud.tencent.com/developer/article/1042809)
* [How To Resample and Interpolate Your Time Series Data With Python](https://machinelearningmastery.com/resample-interpolate-time-series-data-python/)
* [时间序列预测](https://mp.weixin.qq.com/s?__biz=Mzg3NDUwNTM3MA==&mid=2247484974&idx=1&sn=d841c644fd9289ad5ec8c52a443463a5&chksm=cecef3dbf9b97acd8a9ededc069851afc00db422cb9be4d155cb2c2a9614b2ee2050dc7ab4d7&scene=21#wechat_redirect)

