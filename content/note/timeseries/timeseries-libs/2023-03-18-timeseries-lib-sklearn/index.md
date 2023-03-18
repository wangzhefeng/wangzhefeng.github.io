---
title: SKlearn
subtitle: 相关特征工程
author: 王哲峰
date: '2023-03-18'
slug: timeseries-lib-sklearn
categories:
  - timeseries
tags:
  - tool
---

# 时间相关特征工程

## 数据

```python
from sklearn.datasets import fetch_openml

bike_sharing = fetch_openml(
    "Bike_Sharing_Demand", 
    version = 2,
    as_frame = True,
    parser = "pandas"
)
df = bike_sharing.frame
```

# 时间序列数据分割




# 参考

* [sklearn.model_selection.TimeSeriesSplit](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.TimeSeriesSplit.html#sklearn.model_selection.TimeSeriesSplit)
* [Time-related feature engineering](https://scikit-learn.org/stable/auto_examples/applications/plot_cyclical_feature_engineering.html#concluding-remarks)

