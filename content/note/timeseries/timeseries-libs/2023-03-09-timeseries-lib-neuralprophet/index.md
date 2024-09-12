---
title: NeuralProphet
subtitle: Prophet & AR-Net
author: 王哲峰
date: '2023-03-09'
slug: timeseries-lib-neuralprophet
categories:
  - timeseries
tags:
  - tool
---

Based on Neural Networks, inspired by Facebook Prophet and AR-Net, built on Pytorch.

# 安装

```bash
$ pip install neuralprophet
$ pip install "neuralprophet[live]"
```


# 示例

```python
import pandas as pd

from neuralprophet import NeuralProphet
from neuralprophet import set_random_seed
set_random_seed(0)

# data
data_location = "https://raw.githubusercontent.com/ourownstory/neuralprophet-data/main/datasets/"
df = pd.read_csv(data_location + "wp_log_peyton_manning.csv")
print(df.head())
print(df.shape)

# model
m = NeuralProphet()

# train and test data
df_train, df_test = m.split_df(df, valid_p = 0.2)

# model train
metrics = m.fit(df_train, validation_df = df_test)

# model validation
# m = NeuralProphet()
# train_metrics = m.fit(df_trian)
# test_metrics = m.fit(df_test)

# model validation metrics
with pd.option_context("display.max_rows", None, "display.max_columns", None):
    print(metrics)

# model predict
predicted = m.predict(df)
forecast = m.predict(df)
print(predicted)
print(forecast)

# plotting
forecast_plot = m.plot(forecast)
fig_comp = m.plot_components((forecast))
fig_param = m.plot_parameters()
```



# 参考

* [Doc](https://neuralprophet.com/)

