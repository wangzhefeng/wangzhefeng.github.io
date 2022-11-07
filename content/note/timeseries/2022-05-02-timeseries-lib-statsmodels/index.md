---
title: Lib Statsmodels
author: 王哲峰
date: '2022-05-02'
slug: timeseries-lib-statsmodels
categories:
  - timeseries
tags:
  - ml
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

- [自相关系数和偏自相关系数计算](#自相关系数和偏自相关系数计算)
- [自相关图和偏自相关图绘制](#自相关图和偏自相关图绘制)
</p></details><p></p>


# 自相关系数和偏自相关系数计算

```python
import statsmodels.api as sm

X = [2, 3, 4, 3, 8, 7]
print(sm.tsa.stattools.acf(X, nlags = 1, adjusted = True))
[1, 0.3559322]
```

# 自相关图和偏自相关图绘制

```python
import numpy as np
import pandas as pd
import akshare as ak
from matplotlib import pyplot as plt
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

np.random.seed(123)

# -------------- 准备数据 --------------
# 白噪声
white_noise = np.random.standard_normal(size=1000)

# 随机游走
x = np.random.standard_normal(size=1000)
random_walk = np.cumsum(x)

# GDP
df = ak.macro_china_gdp()
df = df.set_index('季度')
df.index = pd.to_datetime(df.index)
gdp = df['国内生产总值-绝对值'][::-1].astype('float')

# GDP DIFF
gdp_diff = gdp.diff(4).dropna()

# -------------- 绘制图形 --------------
fig, ax = plt.subplots(4, 2)
fig.subplots_adjust(hspace = 0.5)

plot_acf(white_noise, ax = ax[0][0])
ax[0][0].set_title('ACF(white_noise)')
plot_pacf(white_noise, ax = ax[0][1])
ax[0][1].set_title('PACF(white_noise)')

plot_acf(random_walk, ax = ax[1][0])
ax[1][0].set_title('ACF(random_walk)')
plot_pacf(random_walk, ax = ax[1][1])
ax[1][1].set_title('PACF(random_walk)')

plot_acf(gdp, ax = ax[2][0])
ax[2][0].set_title('ACF(gdp)')
plot_pacf(gdp, ax = ax[2][1])
ax[2][1].set_title('PACF(gdp)')

plot_acf(gdp_diff, ax = ax[3][0])
ax[3][0].set_title('ACF(gdp_diff)')
plot_pacf(gdp_diff, ax = ax[3][1])
ax[3][1].set_title('PACF(gdp_diff)')

plt.show()
```