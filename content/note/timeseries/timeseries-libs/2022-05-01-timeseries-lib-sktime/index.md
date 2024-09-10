---
title: Sktime 
subtitle: 机器学习
author: 王哲峰
date: '2022-05-01'
slug: timeseries-lib-sktime
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
img {
    pointer-events: none;
}
</style>

<details><summary>目录</summary><p>

- [安装](#安装)
  - [pip](#pip)
  - [conda](#conda)
- [时间序列数据类型](#时间序列数据类型)
- [时间序列任务](#时间序列任务)
  - [时间序列预测](#时间序列预测)
  - [时间序列分类](#时间序列分类)
  - [时间序列回归](#时间序列回归)
  - [时间序列聚类](#时间序列聚类)
  - [时间序列标注](#时间序列标注)
  - [时间序列交叉验证](#时间序列交叉验证)
    - [单个窗口拆分](#单个窗口拆分)
      - [Single Window 分割](#single-window-分割)
    - [滑动窗口拆分](#滑动窗口拆分)
      - [不指定初始窗口](#不指定初始窗口)
      - [指定初始窗口](#指定初始窗口)
    - [扩展窗口拆分](#扩展窗口拆分)
    - [指定分割点多次分割](#指定分割点多次分割)
    - [模型选择](#模型选择)
- [参考](#参考)
</p></details><p></p>

> A unified framework for machine learning with time series.

# 安装

## pip

```bash
$ pip install sktime
$ pip install sktime[all_extras]
```

## conda

```bash
$ conda install -c conda-forge sktime
$ conda install -c conda-forge sktime-all-extras
```

# 时间序列数据类型

在 sktime 时间序列中，数据可以指单变量、多变量或面板数据，
不同之处在于时间序列变量之间的数量和相互关系，以及观察每个变量的实例数

* 单变量时间序列数据是指随时间跟踪单个变量的数据
* 多变量时间序列数据是指针对同一实例随时间跟踪多个变量的数据。例如，一个国家/地区的多个季度经济指标或来自同一台机器的多个传感器读数
* 面板时间序列数据是指针对多个实例跟踪变量（单变量或多变量）的数据。例如，多个国家/地区的多个季度经济指标或多台机器的多个传感器读数




# 时间序列任务

## 时间序列预测


## 时间序列分类


## 时间序列回归


## 时间序列聚类


## 时间序列标注

> time series annotation，时间序列标注是指异常值检测、变化点检测和分割

## 时间序列交叉验证

Sktime 提供了相应的类“窗口拆分器”，窗口拆分器有两个可配置的参数：

* `window_length`：每个折的训练窗口长度
* `fh`：预测范围(forecasting horizon, fh)。指定训练窗口后要包含在测试数据中的值。
  可以是整数、整数列表或 Sktime `ForecastingHorizon` 对象
* `initial_window`：第一个折的训练窗口长度。如果未设置，`window_length` 将用作第一个折的长度
* `step_length`：折之间的步长。默认值为 1 步

导入 Python 依赖库：

```python
from warnings import simplefilter

import numpy as np
import seaborn as sns
from matplotlib.ticker import MaxNLocator

from sktime.datasets import load_airline
from sktime.forecasting.base import ForecastingHorizon
from sktime.forecasting.model_selection import (
    CutoffSplitter,
    ExpandingWindowSplitter,
    SingleWindowSplitter,
    SlidingWindowSplitter,
    temporal_train_test_split,
)
from sktime.utils.plotting import plot_series

from plot_windows import plot_windows
from plot_windows import get_windows
```

工具函数：

```python
def get_windows(y, cv):
    """
    Generate windows

    Args:
        y (_type_): _description_
        cv (_type_): _description_
    """
    train_windows = []
    test_windows = []
    for i, (train, test) in enumerate(cv.split(y)):
        train_windows.append(train)
        test_windows.append(test)
    
    return train_windows, test_windows


def plot_windows(y, train_windows, test_windows, title = ""):
    """
    Visualize training and test windows
    """
    simplefilter("ignore", category = UserWarning)

    def get_y(length, split):
        """
        Create a constant vecotr based on the split for y-axis.
        """
        return np.ones(length) * split
    
    # split params
    n_splits = len(train_windows)
    n_timepoints = len(y)
    len_test = len(test_windows[0])

    # plot params
    train_color, test_color = sns.color_palette("colorblind")[:2]
    fig, ax = plt.subplots(figsize = plt.figaspect(0.3))

    for i in range(n_splits):
        train = train_windows[i]
        test = test_windows[i]

        ax.plot(np.arange(n_timepoints), get_y(n_timepoints, i), marker = "o", c = "lightgray")
        ax.plot(train, get_y(len(train), i), marker = "o", c = train_color, label = "Window")
        ax.plot(test, get_y(len_test, i), marker = "o", c = test_color, label = "Forecasting horizon")

        ax.invert_yaxis()
        ax.yaxis.set_major_locator(MaxNLocator(integer = True))
        ax.set(
            title = title,
            ylabel = "Window number",
            xlabel = "Time",
            xticklabels = y.index,
        )
        # remove duplicate labels/handlers
        handles, labels = [(leg[:2]) for leg in ax.get_legend_handles_labels()]
        ax.legend(handles, labels);
```

数据：

```python
y = load_airline().iloc[:30]
print(y.head())
print(y.shape)
print(y.name)
fig, ax = plot_series(y)
```

```
Period
1949-01    112.0
1949-02    118.0
1949-03    132.0
1949-04    129.0
1949-05    121.0
Freq: M, Name: Number of airline passengers, dtype: float64

(30,)

Number of airline passengers
```

![img](images/data.png)

### 单个窗口拆分

> single train-test split

初始化后，窗口拆分器可以与 KFold 验证类相同的方式使用，为每个数据拆分提供训练和测试索引：

```python
y_train, y_test = temporal_train_test_split(y = y, test_size = 0.25)
fig, ax = plot_series(y_train, y_test, labels = ["y_train", "y_test"])
```

![img](images/single_1.png)


```python
fh = ForecastingHorizon([1, 2, 3, 4, 5])
y_train, y_test = temporal_train_test_split(y, fh = fh)
fig, ax = plot_series(y_train, y_test, labels  = ["y_train", "y_test"])
```

![img](images/single_2.png)

#### Single Window 分割

```python
# splitter parameters
window_length = 5
fh = ForecastingHorizon([1, 2, 3])
# split
cv = SingleWindowSplitter(window_length = window_length, fh = fh)
n_splits = cv.get_n_splits(y)
print(f"Number of Folds: {n_splits}")

# split windows and split datasets
train_windows, test_windows = get_windows(y, cv)
print(f"train windows:\n {train_windows} \ntrain data:\n {y.iloc[train_windows[0]]}")
print(f"test windows:\n {test_windows} \ntest data:\n {y.iloc[test_windows[0]]}")

# plotting windows
plot_windows(y, train_windows, test_windows)
```

```
Number of Folds: 1
train windows:
 [array([22, 23, 24, 25, 26])] 
train data:
 1950-11    114.0
1950-12    140.0
1951-01    145.0
1951-02    150.0
1951-03    178.0
Freq: M, Name: Number of airline passengers, dtype: float64
test windows:
 [array([27, 28, 29])] 
test data:
 1951-04    163.0
1951-05    172.0
1951-06    178.0
Freq: M, Name: Number of airline passengers, dtype: float64
```

![img](images/single_window.png)

### 滑动窗口拆分

此拆分器会随着时间的推移在滑动窗口上生成折。每个折的训练数据和测试数据的大小是恒定的

#### 不指定初始窗口

```python
# splitter parameters
window_length = 5
fh = ForecastingHorizon([1, 2, 3])

# split
cv = SlidingWindowSplitter(window_length = window_length, fh = fh)
n_splits = cv.get_n_splits(y)
print(f"Number of Folds: {n_splits}")

# split windows and split datasets
train_windows, test_windows = get_windows(y, cv)
print(f"train windows:\n {train_windows}")
print(f"test windows:\n {test_windows}")

# plotting windows
plot_windows(y, train_windows, test_windows)
```

```
Number of Folds: 23
train windows:
 [array([0, 1, 2, 3, 4]), array([1, 2, 3, 4, 5]), array([2, 3, 4, 5, 6]), array([3, 4, 5, 6, 7]), array([4, 5, 6, 7, 8]), array([5, 6, 7, 8, 9]), array([ 6,  7,  8,  9, 10]), array([ 7,  8,  9, 10, 11]), array([ 8,  9, 10, 11, 12]), array([ 9, 10, 11, 12, 13]), array([10, 11, 12, 13, 14]), array([11, 12, 13, 14, 15]), array([12, 13, 14, 15, 16]), array([13, 14, 15, 16, 17]), array([14, 15, 16, 17, 18]), array([15, 16, 17, 18, 19]), array([16, 17, 18, 19, 20]), array([17, 18, 19, 20, 21]), array([18, 19, 20, 21, 22]), array([19, 20, 21, 22, 23]), array([20, 21, 22, 23, 24]), array([21, 22, 23, 24, 25]), array([22, 23, 24, 25, 26])]
test windows:
 [array([5, 6, 7]), array([6, 7, 8]), array([7, 8, 9]), array([ 8,  9, 10]), array([ 9, 10, 11]), array([10, 11, 12]), array([11, 12, 13]), array([12, 13, 14]), array([13, 14, 15]), array([14, 15, 16]), array([15, 16, 17]), array([16, 17, 18]), array([17, 18, 19]), array([18, 19, 20]), array([19, 20, 21]), array([20, 21, 22]), array([21, 22, 23]), array([22, 23, 24]), array([23, 24, 25]), array([24, 25, 26]), array([25, 26, 27]), array([26, 27, 28]), array([27, 28, 29])]
```

![img](images/sliding_split1.png)

#### 指定初始窗口

```python
# splitter parameters
window_length = 5
fh = ForecastingHorizon([1, 2, 3])
initial_window = 10

# split
cv = SlidingWindowSplitter(window_length = window_length, fh = fh, initial_window = initial_window)
n_splits = cv.get_n_splits(y)
print(f"Number of Folds: {n_splits}")

# split windows and split datasets
train_windows, test_windows = get_windows(y, cv)
print(f"train windows:\n {train_windows}")
print(f"test windows:\n {test_windows}")

# plotting windows
plot_windows(y, train_windows, test_windows)
```

```
Number of Folds: 18
train windows:
 [array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9]), array([ 6,  7,  8,  9, 10]), array([ 7,  8,  9, 10, 11]), array([ 8,  9, 10, 11, 12]), array([ 9, 10, 11, 12, 13]), array([10, 11, 12, 13, 14]), array([11, 12, 13, 14, 15]), array([12, 13, 14, 15, 16]), array([13, 14, 15, 16, 17]), array([14, 15, 16, 17, 18]), array([15, 16, 17, 18, 19]), array([16, 17, 18, 19, 20]), array([17, 18, 19, 20, 21]), array([18, 19, 20, 21, 22]), array([19, 20, 21, 22, 23]), array([20, 21, 22, 23, 24]), array([21, 22, 23, 24, 25]), array([22, 23, 24, 25, 26])]
test windows:
 [array([10, 11, 12]), array([11, 12, 13]), array([12, 13, 14]), array([13, 14, 15]), array([14, 15, 16]), array([15, 16, 17]), array([16, 17, 18]), array([17, 18, 19]), array([18, 19, 20]), array([19, 20, 21]), array([20, 21, 22]), array([21, 22, 23]), array([22, 23, 24]), array([23, 24, 25]), array([24, 25, 26]), array([25, 26, 27]), array([26, 27, 28]), array([27, 28, 29])]
```

![img](images/sliding_split2.png)

### 扩展窗口拆分

与滑动窗口拆分器一样，扩展窗口拆分会随着时间的推移在滑动窗口上生成折。
但是，训练序列的长度会随着时间的推移而增长，每个后续折都会保留完整序列历史。
每个折的测试序列长度是恒定的

```python
# splitter parameters
initial_window = 5
fh = ForecastingHorizon([1, 2, 3])

# split
cv = ExpandingWindowSplitter(initial_window = initial_window, fh = fh)
n_splits = cv.get_n_splits(y)
print(f"Number of Folds: {n_splits}")

# split windows and split datasets
train_windows, test_windows = get_windows(y, cv)
print(f"train windows:\n {train_windows}")
print(f"test windows:\n {test_windows}")

# plotting windows
plot_windows(y, train_windows, test_windows)
```

```
Number of Folds: 23
train windows:
 [array([0, 1, 2, 3, 4]), array([0, 1, 2, 3, 4, 5]), array([0, 1, 2, 3, 4, 5, 6]), array([0, 1, 2, 3, 4, 5, 6, 7]), array([0, 1, 2, 3, 4, 5, 6, 7, 8]), array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9]), array([ 0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10]), array([ 0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11]), array([ 0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12]), array([ 0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13]), array([ 0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14]), array([ 0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14, 15]), array([ 0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14, 15, 16]), array([ 0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14, 15, 16,
       17]), array([ 0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14, 15, 16,
       17, 18]), array([ 0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14, 15, 16,
       17, 18, 19]), array([ 0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14, 15, 16,
       17, 18, 19, 20]), array([ 0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14, 15, 16,
       17, 18, 19, 20, 21]), array([ 0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14, 15, 16,
       17, 18, 19, 20, 21, 22]), array([ 0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14, 15, 16,
       17, 18, 19, 20, 21, 22, 23]), array([ 0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14, 15, 16,
       17, 18, 19, 20, 21, 22, 23, 24]), array([ 0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14, 15, 16,
       17, 18, 19, 20, 21, 22, 23, 24, 25]), array([ 0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14, 15, 16,
       17, 18, 19, 20, 21, 22, 23, 24, 25, 26])]
test windows:
 [array([5, 6, 7]), array([6, 7, 8]), array([7, 8, 9]), array([ 8,  9, 10]), array([ 9, 10, 11]), array([10, 11, 12]), array([11, 12, 13]), array([12, 13, 14]), array([13, 14, 15]), array([14, 15, 16]), array([15, 16, 17]), array([16, 17, 18]), array([17, 18, 19]), array([18, 19, 20]), array([19, 20, 21]), array([20, 21, 22]), array([21, 22, 23]), array([22, 23, 24]), array([23, 24, 25]), array([24, 25, 26]), array([25, 26, 27]), array([26, 27, 28]), array([27, 28, 29])]
```


![img](images/expanding_split.png)

### 指定分割点多次分割

```python
# splitter parameters
window_length = 5
fh = ForecastingHorizon([1, 2, 3])
cutoffs = np.array([10, 13, 15, 25])  # Specify cutoff points (by array index).

cv = CutoffSplitter(cutoffs = cutoffs, window_length = window_length, fh = fh)

n_splits = cv.get_n_splits(y)
print(f"Number of Folds = {n_splits}")

# split windows and split datasets
train_windows, test_windows = get_windows(y, cv)
print(f"train windows:\n {train_windows}")
print(f"test windows:\n {test_windows}")

# plotting windows
plot_windows(y, train_windows, test_windows)
```

```
Number of Folds = 4
train windows:
 [array([ 6,  7,  8,  9, 10]), array([ 9, 10, 11, 12, 13]), array([11, 12, 13, 14, 15]), array([21, 22, 23, 24, 25])]
test windows:
 [array([11, 12, 13]), array([14, 15, 16]), array([16, 17, 18]), array([26, 27, 28])]
```

![img](images/cutoff.png)


### 模型选择

Sktime 提供了两个类，它们使用交叉验证来搜索预测模型的最佳参数：

* `ForecastingGridSearchCV`： 评估所有可能的参数组合
* `ForecastingGrandomizedSearchCV`：随机选择要评估的超参数

这些类通过反复拟合和评估同一个模型来工作。这两个类类似于 scikit learn 中的交叉验证方法，并遵循类似的界面

* 要调整的预测器
* 交叉验证构造函数（例如Sliding Window Splitter）
* 参数网格（例如{'window_length'：[1,2,3]}）
* 参数
* 评估指标（可选）

在下面的示例中，跨时间滑动窗口使用带交叉验证的网格搜索来选择最佳模型参数。
参数网格指定模型参数 `sp`（季节周期数）和 `seasonal`（季节分量类型）的哪些值

预测器拟合 60 个时间步长初始窗口的数据。后续窗口的长度为 20。预测范围设置为 1，
这意味着测试窗口仅包含在训练窗口之后出现的单个值

```python
from sktime.forecasting.exp_smoothing import ExponentialSmoothing
from sktime.forecasting.model_selection import ForecastingGridSearchCV, SlidingWindowSplitter

# model
forecaster = ExponentialSmoothing()

# params
param_grid = {
    "sp": [1, 6, 12], 
    'seasonal': ['add', 'mul']
}

# cv
cv = SlidingWindowSplitter(
    initial_window = 60, 
    window_length = 20, 
    fh = 1
)

# model selection
gscv = ForecastingGridSearchCV(
    forecaster, 
    strategy = "refit", 
    cv = cv, 
    param_grid = param_grid
)

cscv.fit(y_train)
y_pred = gscv.predict([1, 2])

print(gscv.best_params_)
print(gscv.best_forecaster_)
```





# 参考

* [Doc](http://www.sktime.net/en/latest/)
* [GitHub](https://github.com/sktime/sktime)

