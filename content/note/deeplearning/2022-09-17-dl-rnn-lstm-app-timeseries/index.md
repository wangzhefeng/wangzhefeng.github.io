---
title: LSTM 时间序列预测
author: 王哲峰
date: '2022-09-17'
slug: dl-rnn-lstm-app-timeseries
categories:
  - deeplearning
tags:
  - model
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

- [时间序列分析](#时间序列分析)
  - [单变量时间序列](#单变量时间序列)
  - [多元时间序列](#多元时间序列)
- [LSTM](#lstm)
  - [LSTM 简介](#lstm-简介)
  - [LSTM 实现时间序列](#lstm-实现时间序列)
    - [Python 依赖](#python-依赖)
    - [数据](#数据)
    - [训练数据分割](#训练数据分割)
    - [数据规范化](#数据规范化)
    - [数据处理](#数据处理)
    - [超参数调优](#超参数调优)
    - [模型预测](#模型预测)
</p></details><p></p>

# 时间序列分析

时间序列分析表示基于时间顺序的一系列数据。它可以是秒、分钟、小时、天、周、月、年。
未来的数据将取决于它以前的值

## 单变量时间序列

对于单变量时间序列数据，将使用单列进行预测，因此即将到来的未来值将仅取决于它之前的值

## 多元时间序列

在多元变量中将有多个列来对目标值进行预测，目标列的预测值不仅取决于它以前的值，还取决于其他特征。
因此，要预测即将到来的目标值，必须包括目标在内的所有列对目标值进行预测

# LSTM 

## LSTM 简介


## LSTM 实现时间序列

目标:

* CSV 文件中包含了谷歌从 2001-01-25 到 2021-09-29 的股票数据，数据是按照天数频率的
* 预测 Open 列的未来值


### Python 依赖

```python
import numpy as np
import pandas as pd
from matplotlib.pyplot as plt

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from sklearn.preprocessing import MinMaxScaler
from keras.wrappers.scikit_learn import KerasRegressor
from sklearn.model_selection import GridSearchCV
```

### 数据

```python
df = pd.read_csv("train.csv", parse_dates = ["Date"], index_col = [0])
df.head()
df.tail()
df.shape
```

### 训练数据分割

```python
test_split = round(len(df) * 0.2)

df_train = df[:-test_split]
df_test = df[-test_split:]

print(df_train)
print(df_test)
```

### 数据规范化

```python
scaler = MinMaxScaler(feature_range = (0, 1))

df_train_scaled = scaler.fit_transform(df_train)
df_test_scaled = scaler.transform(df_test)

print(df_train_scaled)
```

### 数据处理

```python
def create_x_y(data, n_past):
    data_x = []
    data_y = []
    for i in range(n_past, len(data)):  # range(30, 4162), range(30, 1041)
        data_x.append(data[(i - n_past):i, 0:data.shape[1]])
        data_y.append(data[i, 0])

    return np.array(data_x), np.array(data_y)

train_x, train_y = create_x_y(df_train_scaled, 30)
test_x, test_y = create_x_y(df_test_scaled, 30)
print(f"train_x shape: {train_x.shape}")
print(f"train_y shape: {train_y.shape}")

print(f"test_x shape: {test_x.shape}")
print(f"test_y shape: {test_y.shape}")

print(f"train_x[0]: {train_x[0]}")
print(f"train_y[0]: {train_y[0]}")
```

```
(4132, 30, 5)
(4132,)

(1011, 30, 5)
(1011,)


```

* `n_past` 是在预测下一个目标值时，使用的过去的数据样本数，
    - 使用过去的 `n_past` 个样本值(包括目标列在内的所有特性)来预测第 `n_past + 1` 个目标值
* `train_x` -- `train_y`
    - `[0:30, 0:5]` -- `[30, 0]`
    - `[1:31, 0:5]` -- `[31, 0]`
    - `[2:32, 0:5]` -- `[32, 0]` 
    - ...

### 超参数调优

```python
def build_model(optimizer):
    grid_model = Sequential()
    grid_model.add(LSTM(50), return_sequences = True, input_shape = (30, 5))
    grid_model.add(LSTM(50))
    grid_model.add(Dropout(0.2))
    grid_model.add(Dense(1))

    grid_model.compile(loss = "mse", optimizer = optimizer)
    
    return grid_model
```

```python
grid_model = KerasRegressor(
    build_fn = build_model, 
    verbose = 1,
    validation_data = (tset_x, test_y)
)

parameters = {
    "batch_size": [16, 20],
    "epochs": [8, 10],
    "optimizer": ["adam", "Adadelta"],
}

grid_search = GridSearchCV(
    estimator = grid_model,
    param_grid = parameters,
    cv = 2,
)

grid_search = grid_search.fit(train_x, train_y)

# 最优参数
grid_search.best_params_
```

### 模型预测

```python
# 训练好的模型
my_model = grid_search.best_estimator_.model

# 模型预测
prediction = my_model.predict(test_x)
print(f"prediction:\n {prediction}")
print(f"\nPrediction shape: {prediction.shape}")
```

```python
prediction_copy_array = np.repeat(prediction, 5, axis = -1)
pred = scaler.inver_transform(prediction_copy_array)
```



- https://mp.weixin.qq.com/s?__biz=MzU1MjYzNjQwOQ==&mid=2247499754&idx=1&sn=183c8aa1156023a19b061c27a0be8407&chksm=fbfda57ccc8a2c6a60f630b2cd9b2d587d345ea002c8af81b5059154c97398589a6c86e15bfd&scene=132#wechat_redirect
- https://github.com/sksujan58/Multivariate-time-series-forecasting-using-LSTM