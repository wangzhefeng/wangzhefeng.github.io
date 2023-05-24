---
title: sklearn 数据分割
author: 王哲峰
date: '2023-03-11'
slug: model-validation-sklearn
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
img {
    pointer-events: none;
}
</style>

<details><summary>目录</summary><p>

- [时间序列分割](#时间序列分割)
  - [Time Series Split](#time-series-split)
    - [介绍](#介绍)
    - [优缺点](#优缺点)
    - [应用](#应用)
  - [带间隙的 Time Series Split](#带间隙的-time-series-split)
    - [介绍](#介绍-1)
    - [应用](#应用-1)
  - [滑动 Time Series Split](#滑动-time-series-split)
    - [介绍](#介绍-2)
    - [应用](#应用-2)
- [蒙特卡洛交叉验证](#蒙特卡洛交叉验证)
  - [MonteCarloCV 介绍](#montecarlocv-介绍)
  - [MonteCarloCV 实现](#montecarlocv-实现)
  - [MonteCarloCV 使用](#montecarlocv-使用)
</p></details><p></p>

# 时间序列分割

## Time Series Split

### 介绍

对时间序列进行多次拆分是个好主意，这样做可以在时间序列数据的不同部分上测试模型。
在时间序列分割中，时间序列被分成 `$k+1$` 个连续的大小相等的数据块，前 `$k$` 个块为训练数据，
第 `$k+1$` 个块为测试数据

下面是该技术的可视化描述:

![img](images/timeseriessplit.png)

### 优缺点

使用时间序列分割的主要好处如下:

* 保持了观察的顺序
    - 这个问题在有序数据集(如时间序列)中非常重要
* 生成了很多拆分，几次拆分后可以获得更稳健的评估
    - 如果数据集不大，这一点尤其重要

主要缺点是跨折叠的训练样本量是不一致的：

* 假设将该方法应用于上图所示的 5 次分折。在第一次迭代中，所有可用观测值的 20% 用于训练，
  但是，这个数字在最后一次迭代中是 80%。因此，初始迭代可能不能代表完整的时间序列，这个问题会影响性能估计。
  可以使用蒙特卡洛交叉验证结局这个问题

### 应用

时间序列分割就是 scikit-learn 中 `TimeSeriesSplit` 实现

```python
import numpy as np
from sklearn.model_selection import TimeSeriesSplit


X = np.array([
    [1, 2],
    [3, 4],
    [1, 2],
    [3, 4],
    [1, 2],
    [3, 4],
])
y = np.array([1, 2, 3, 4, 5, 6])

tscv = TimeSeriesSplit(n_splits = 5, max_train_size = None, test_size = None, gap = 0)

for i, (train_index, test_index) in enumerate(tscv.split(X)):
    print(f"Fold {i}:")
    print(f"  Train: index={train_index}")
    print(f"  Test: index={test_index}")
```

```
Fold 0:
  Train: index=[0]
  Test:  index=[1]
Fold 1:
  Train: index=[0 1]
  Test:  index=[2]
Fold 2:
  Train: index=[0 1 2]
  Test:  index=[3]
Fold 3:
  Train: index=[0 1 2 3]
  Test:  index=[4]
Fold 4:
  Train: index=[0 1 2 3 4]
  Test:  index=[5]
```

## 带间隙的 Time Series Split

### 介绍

可以在上述技术中增加训练和验证之间的间隙。这有助于增加两个样本之间的独立性

![img](images/timeseriessplit_gap.png)

### 应用

使用 `TimeSeriesSplit` 类中的 `gap` 参数引入这个间隙

```python
import numpy as np
from sklearn.model_selection import TimeSeriesSplit


X = np.array([
    [1, 2],
    [3, 4],
    [1, 2],
    [3, 4],
    [1, 2],
    [3, 4],
])
y = np.array([1, 2, 3, 4, 5, 6])

tscv = TimeSeriesSplit(n_splits = 3, test_size = 2, gap = 2)

for i, (train_index, test_index) in enumerate(tscv.split(X)):
    print(f"Fold {i}:")
    print(f"  Train: index={train_index}")
    print(f"  Test: index={test_index}")
```

```
Fold 0:
  Train: index=[0 1 2 3]
  Test:  index=[6 7]
Fold 1:
  Train: index=[0 1 2 3 4 5]
  Test:  index=[8 9]
Fold 2:
  Train: index=[0 1 2 3 4 5 6 7]
  Test:  index=[10 11]
```

## 滑动 Time Series Split 

### 介绍

另一种应用时间序列分割的方法是滑动窗口，即在迭代之后老的数据块被丢弃。
这种方法可能在两种情况下有用：

* 数据量巨大
* 旧的观察已经过时了

这种变体也可以应用于训练样本和验证样本之间的间隙

![img](images/timeseriessplit_sliding.png)

### 应用

```python
# TODO
```

# 蒙特卡洛交叉验证

## MonteCarloCV 介绍

蒙特卡罗交叉验证(MonteCarloCV)是一种可以用于时间序列的方法。与时间序列分割(Time Series Split)不同，
每个迭代中的验证原点是随机选择的，即在不同的随机起点来获取一个时间周期的数据

下图是这种技术的直观图示：

![img](images/montecarlocv.png)

像 TimeSeriesSplit 一样，MonteCarloCV 也保留了观测的时间顺序。它还会保留多次重复估计过程。

MonteCarloCV 与 TimeSeriesSplit 的区别主要有两个方面：

* 对于训练和验证样本量，使用 TimeSeriesSplit 时训练集的大小会增加。在 MonteCarloCV 中，
  训练集的大小在每次迭代过程中都是固定的，这样可以防止训练规模不能代表整个数据
* 随机的分折，在 MonteCarloCV 中，验证原点是随机选择的。这个原点标志着训练集的结束和验证的开始。
  在 TimeSeriesSplit 的情况下，这个点是确定的。它是根据迭代次数预先定义的

经过详细研究 MonteCarloCV，这包括与 TimeSeriesSplit 等其他方法的比较。
MonteCarloCV 可以获得更好的估计

## MonteCarloCV 实现

不幸的是，scikit-learn 不提供 MonteCarloCV 的实现。所以，需要手动实现它：

```python
from typing import List, Generator
import numpy as np
from sklearn.model_selection._split import _BaseKFold
from sklearn.utils.validation import indexable, _num_samples

 
class MonteCarloCV(_BaseKFold):
 
    def __init__(self, 
                 n_splits: int, 
                 train_size: float, 
                 test_size: float, 
                 gap: int = 0):
        """
        Monte Carlo Cross-Validation
 
        Holdout applied in multiple testing periods
        Testing origin (time-step where testing begins) is randomly chosen according to a monte carlo simulation
 
        Parameters
        ----------
        n_splits: (int) Number of monte carlo repetitions in the procedure
        train_size: (float) Train size, in terms of ratio of the total length of the series
        test_size: (float) Test size, in terms of ratio of the total length of the series
        gap: (int) Number of samples to exclude from the end of each train set before the test set.
        """
        self.n_splits = n_splits
        self.n_samples = -1
        self.gap = gap
        self.train_size = train_size
        self.test_size = test_size
        self.train_n_samples = 0
        self.test_n_samples = 0 
        self.mc_origins = []
 
    def split(self, X, y = None, groups = None) -> Generator:
        """
        Generate indices to split data into training and test set.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training data, where `n_samples` is the number of samples
            and `n_features` is the number of features.
        y : array-like of shape (n_samples,)
            Always ignored, exists for compatibility.
        groups : array-like of shape (n_samples,)
            Always ignored, exists for compatibility.
        
        Yields
        ------
        train : ndarray
            The training set indices for that split.
        test : ndarray
            The testing set indices for that split.
        """
        X, y, groups = indexable(X, y, groups)
        self.n_samples = _num_samples(X)
 
        self.train_n_samples = int(self.n_samples * self.train_size) - 1
        self.test_n_samples = int(self.n_samples * self.test_size) - 1
 
        # Make sure we have enough samples for the given split parameters
        if self.n_splits > self.n_samples:
            raise ValueError(
                f'Cannot have number of folds={self.n_splits} greater'
                f' than the number of samples={self.n_samples}.'
            )
        if self.train_n_samples - self.gap <= 0:
            raise ValueError(
                f'The gap={self.gap} is too big for number of training samples'
                f'={self.train_n_samples} with testing samples={self.test_n_samples} and gap={self.gap}.'
            )
 
        indices = np.arange(self.n_samples)
        selection_range = np.arange(self.train_n_samples + 1, self.n_samples - self.test_n_samples - 1)
 
        self.mc_origins = np.random.choice(
            a = selection_range,
            size = self.n_splits,
            replace = True
        )
        for origin in self.mc_origins:
            if self.gap > 0:
                train_end = origin - self.gap + 1
            else:
                train_end = origin - self.gap
            train_start = origin - self.train_n_samples - 1
            test_end = origin + self.test_n_samples

            yield (
                indices[train_start:train_end],
                indices[origin:test_end],
            )
 
    def get_origins(self) -> List[int]:
        return self.mc_origins
```

MonteCarloCV 接受四个参数：

* `n_splitting`：分折或迭代的次数。这个值趋向于 10
* `training_size`：每次迭代时训练集的大小与时间序列大小的比值
* `test_size`：类似于 `training_size`，但用于验证集
* `gap`：分离训练集和验证集的观察数。与 TimeSeriesSplits 一样，此参数的值默认为 0(无间隙)

每次迭代的训练和验证大小取决于输入数据。发现一个 0.6/0.1 的分区工作得很好。
也就是说，在每次迭代中，60% 的数据被用于训练，10% 的观察结果用于验证

## MonteCarloCV 使用

```python
from sklearn.datasets import make_regression
from src.mccv import MonteCarloCV

X, y = make_regression(n_samples = 120)

mccv = MonteCarloCV(
    n_splits = 5, 
    train_size = 0.6, 
    test_size = 0.1, 
    gap = 0
)

for train_index, test_index in mccv.split(X):
    print("TRAIN:", train_index, "TEST:", test_index)
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]
```

该实现也与 scikit-learn 兼容。以下是如何结合 GridSearchCV：

```python
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV

model = RandomForestRegressor()
param_search = {
    'n_estimators': [10, 100]
}

gsearch = GridSearchCV(
    estimator = model, 
    cv = mccv, 
    param_grid = param_search
)
gsearch.fit(X, y)
```
