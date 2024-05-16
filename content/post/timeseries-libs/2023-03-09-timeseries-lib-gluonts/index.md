---
title: GluonTS
subtitle: 深度学习、概率预测
author: 王哲峰
date: '2023-03-09'
slug: timeseries-lib-gluonts
categories:
  - timeseries
tags:
  - tool
---

GluonTS is a Python package for probabilistic time series modeling, focusing on deep learning based models, based on PyTorch and MXNet.

# 安装

```bash
$ pip install "gluonts[mxnet,pro]"
$ pip install "gluonts[torch,pro]"
```

其他：

```bash
$ pip install "gluonts[torch]"
$ pip install "gluonts[mxnet]"
$ R -e 'install.packages(c("forecast", "nnfor"), repos="https://cloud.r-project.org")'
$ pip install "gluonts[R]"
$ pip install "gluonts[prophet]"
$ pip install orjson
$ pip install ujson
$ pip install "gluonts[arrow]"
$ pip install "gluonts[shell]"
```


# 示例

```python
import pandas as pd
import matplotlib.pyplot as plt
from gluonts.dataset.pandas import PandasDataset
from gluonts.dataset.split import split
from gluonts.dataset.util import to_pandas
from gluonts.dataset.repository.datasets import get_dataset
from gluonts.mx import DeepAREstimator, Trainer


# model
dataset = get_dataset("airpassengers")

# model
deepar = DeepAREstimator(
    prediction_length = 12,
    freq = "M",
    trainer = Trainer(epochs = 5),
)
model = deepar.train(dataset.train)

# model predict
true_values = to_pandas(list(dataset.test)[0])
true_values.to_timestamp().plot(color = "k")

prediction_input = PandasDataset([
    true_values[:-36],
    true_values[:-24],
    true_values[:-12],
])
predictions = model.predict(prediction_input)

# plotting
for color, prediction in zip(["green", "blue", "purple"], predictions):
    prediction.plot(color = f"tab:{color}")
plt.legend(["True values"], loc = "upper left", fontsize = "xx-large")
plt.show()
```

# 概念

## 时间序列预测

预测不可预测的事情是不可能的，预测的前提是生成时间序列值的潜在因素在未来不会发生根本性变化。
它是一种预测平凡而非意外的工具

## 目标和特征

* 静态特征
* 动态特征

## 概率预测

一个直观的看待这个问题的方法是想象预测一个时间序列 100 次，它返回 100 个不同的时间序列样本，
这些样本围绕它们形成一个分布——除了我们可以直接发出这些分布然后从中抽取样本

分布提供的好处是它们提供了一系列可能的值。想象一下，作为一名餐馆老板，想知道要买多少食材；
如果我们买得太少，我们将无法满足客户的需求，但买太多会产生浪费。因此，当我们预测需求时，
如果模型可以告诉我们可能有 50 道菜的需求，但不太可能超过 60 道，那将是有价值的

## 局部和全局模型

在 GluonTS 中，我们使用局部和全局模型的概念

局部模型适用于单个时间序列并用于对该时间序列进行预测，而全局模型在多个时间序列中进行训练，
并且单个全局模型用于对数据集的所有时间序列进行预测

训练一个全局模型可能会花费很多时间：长达数小时，但有时甚至数天。因此，将模型作为预测请求的一部分进行训练是不可行的，
它作为单独的“离线”步骤发生。相比之下，拟合局部模型通常要快得多，并且作为预测的一部分“在线”完成

在 GluonTS 中，局部模型可直接用作预测器，而全局模型可用作估计器，需要先对其进行训练

## 估计器和预测器

Estimator 将和 Predictor 分成两类的原因是许多模型需要专门的训练步骤来生成全局模型。
这个全局模型只训练一次，但用于对所有时间序列进行预测

这与局部模型形成对比，局部模型适用于单个时间序列，因此试图捕捉每个时间序列的特征，而不是整个数据集

在 GluonTS 中，局部模型可直接用作预测器，而全局模型可用作估计器，需要先对其进行训练

```python
# global DeepAR model
estimator = DeepAREstimator(prediction_length=24, freq="H")
predictor = estimator.train(train_data)

# local Prophet model
predictor = ProphetPredictor(prediction_length=24)
```

## 数据集

任何可以发出字典的东西都可以充当 `Dataset`

```python
DataEntry = dict[str, Any]

class Dataset(Protocol):
    def __iter__(self) -> Iterator[DataEntry]:
        ...

    def __len__(self) -> int:
        ...
```

# 深度模型列表

* https://ts.gluon.ai/stable/getting_started/models.html


# 参考

* [Doc](https://ts.gluon.ai/stable/index.html)

