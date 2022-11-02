---
title: Lib tslearn 时间序列预测
author: 王哲峰
date: '2022-05-04'
slug: timeseries-lib-tslearn
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

- [tslearn 安装](#tslearn-安装)
  - [conda](#conda)
    - [requirements](#requirements)
    - [installation](#installation)
  - [PyPI](#pypi)
    - [requirements](#requirements-1)
    - [installation](#installation-1)
  - [Latest GitHub-hosted Version](#latest-github-hosted-version)
    - [requirements](#requirements-2)
    - [installation](#installation-2)
  - [Requirements](#requirements-3)
- [tslearn 快速开始](#tslearn-快速开始)
  - [时间序列格式](#时间序列格式)
    - [tslearn.utils.to_time_series](#tslearnutilsto_time_series)
    - [tslearn.utils.to_time_series_dataset](#tslearnutilsto_time_series_dataset)
  - [载入标准时间序列数据集](#载入标准时间序列数据集)
    - [tslearn datasets](#tslearn-datasets)
    - [text 格式数据](#text-格式数据)
  - [使用示例](#使用示例)
- [tslearn 任务](#tslearn-任务)
  - [Clustering](#clustering)
  - [Classification](#classification)
  - [Regression](#regression)
- [tslearn 和其他包](#tslearn-和其他包)
  - [cesium](#cesium)
</p></details><p></p>

> `tslearn` is a Python package that provides 
> machine learning tools for the analysis of time series. 

# tslearn 安装

## conda

### requirements

* numpy
* scikit-learn
* scipy

### installation

```bash
$ conda install -c conda-forge tslearn
```

## PyPI

### requirements

* numpy
* scikit-learn
* scipy
* cython
* C++ build tools

### installation

```bash
$ python -m pip install tslearn
```

## Latest GitHub-hosted Version

### requirements

* numpy
* scikit-learn
* scipy
* cython

```bash
$ python -m pip install cython
```

* C++ build tools

### installation

```bash
$ python -m pip install https://github.com/tslearn-team/tslearn/archive/main.zip
```

## Requirements

* scikit-learn
* numpy
* scipy
    - scipy>=1.3.0: 使用 `tslearn.datasets.UCR_UEA_datasets` 从 UCR/UEA archive 载入多变量数据集
* tensorflow(v2)
    - `tslearn.shapelets`
* h5py
    - 读写 hdf5 文件格式

# tslearn 快速开始

## 时间序列格式

### tslearn.utils.to_time_series

```python
from tslearn.utils import to_time_series

my_first_time_series = [1, 3, 4, 2]
formatted_time_series = to_time_series(my_first_time_series)
print(formatted_time_series.shape)  # (4, 1)
```

### tslearn.utils.to_time_series_dataset

```python
from tslearn.utils import to_time_series_dataset

my_first_time_series = [1, 3, 4, 2]
my_second_time_series = [1, 2, 4, 2]
formatted_dataset = to_time_series_dataset([
    my_first_time_series, 
    my_second_time_series
])
print(formatted_dataset.shape)  ## (2, 4, 1)

my_third_time_series = [1, 2 4, 2, 2]
formatted_dataset = to_time_series_dataset([
    my_first_time_series,
    my_second_time_series,
    my_third_time_series,
])
print(formatted_dataset.shape)  # (3, 5, 1)
```

## 载入标准时间序列数据集

### tslearn datasets

```python
from tslearn.datasets import UCR_UEA_datasets

X_train, y_train, X_test, y_test = UCR_UEA_datasets().load_dataset("TwoPatterns")
print(X_train.shape)
print(y_train.shape)
```

### text 格式数据

text 格式数据示例：

```
1.0 0.0 2.5|3.0 2.0 1.0
1.0 2.0|4.333 2.12
```

text 格式数据读写：

```python
from tslearn.utils import save_time_series_txt, load_time_series_txt

time_series_dataset = load_time_series_txt("path/to/your/file.txt")
save_time_series_txt("path/to/another/file.txt", dataset_to_be_saved)
```

## 使用示例

```python
from tslearn.clustering import TimeSeriesKMeans

km = TimeSeriesKMeans(n_clusters = 3, metric = "dtw")
km = fit(X_train)
```



# tslearn 任务

## Clustering



## Classification


## Regression



# tslearn 和其他包




## cesium



```python
from tslearn.utils import from_cesium_dataset, to_cesium_dataset
from cesium.data_management import Timeseries

from_cesium_dataset([
    TimeSeries(m = [1, 2])],
    TimeSeries(m = [1, 4, 3])
)

len(to_cesium_dataset(
    [[[1],
      [2],
      [None]],
     [[1],
      [4],
      [3]]]  
))
```