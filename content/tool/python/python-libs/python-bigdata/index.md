---
title: Python 大数据集处理
author: 王哲峰
date: '2022-07-24'
slug: python-bigdata
categories:
  - Python
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

- [Python 主流数据处理工具](#python-主流数据处理工具)
  - [pandas](#pandas)
  - [Dask](#dask)
  - [datatable](#datatable)
  - [rapids](#rapids)
  - [参考资料](#参考资料)
- [Python 主流数据存储格式](#python-主流数据存储格式)
  - [csv](#csv)
  - [csv 格式转换为 pickle/feather/parquet/jay/h5](#csv-格式转换为-picklefeatherparquetjayh5)
  - [pickle](#pickle)
  - [feather](#feather)
  - [parquet](#parquet)
  - [jay](#jay)
  - [参考资料](#参考资料-1)
- [pandas](#pandas-1)
  - [参考资料](#参考资料-2)
- [datatable](#datatable-1)
  - [安装](#安装)
  - [核心概念](#核心概念)
  - [核心方法](#核心方法)
  - [最佳实践](#最佳实践)
  - [参考资料](#参考资料-3)
- [Dask](#dask-1)
  - [安装](#安装-1)
  - [参考资料](#参考资料-4)
- [RAPIDS](#rapids-1)
- [tqdm](#tqdm)
- [PySpark](#pyspark)
</p></details><p></p>


# Python 主流数据处理工具

如何在有限的 RAM 下快速地读取数据，使用更少的 disk 来存储数据是我们在处理大型数据时需要特别考虑的

* pandas
    - 具有非常丰富的数据处理工具，能非常方便地处理数据
    - 在处理大数据的时候，使用pandas会带来不必要的大内存的使用
* Dask
    - Dask 将 Pandas 扩展为一个并行处理的框架，所以读取数据的速度非常快，
      但是 Dask 如果需要自定义很多函数处理问题的时候可能会遇到很多问题。
    - 除了pandas，Dask 还扩展了 NumPy 工作流，支持地球科学、卫星图像、基因组学、
      生物医学应用和机器学习算法中的多维数据分析；
    - Dask-ML 则扩展了 scikit-learn 和 XGBoost 等机器学习 api，
      支持在大型模型和大型数据集上进行 scalable 的训练和预测
* datatable
    - 非常快地读取大规模的数据集
    - 支持out-of-memory的数据集
    - 多线程数据处理
    - 灵活的API
* RAPIDS
    - Rapids 将数据的处理迁移到了 GPU 上面，所以在速度上得到了大大的提升

## pandas

```python
# jupyter lab/notebook
import pandas as pd

%%time
dtypes = {
    "row_id": "int64",
    "timestamp": "int64",
    "user_id": "int32",
    "content_id": "int16",
    "content_type_id": "boolean",
    "task_container_id": "int16",
    "user_answer": "int8",
    "answered_correctly": "int8",
    "prior_question_elapsed_time": "float32", 
    "prior_question_had_explanation": "boolean"
}

df = pd.read_csv("data/train.csv", dtype = dtypes)
print("Train size:", df.shape)
df.head()
```

## Dask

```python
# jupyter lab/notebook
import dask.dataframe as dd

%%time
dtypes = {
    "row_id": "int64",
    "timestamp": "int64",
    "user_id": "int32",
    "content_id": "int16",
    "content_type_id": "boolean",
    "task_container_id": "int16",
    "user_answer": "int8",
    "answered_correctly": "int8",
    "prior_question_elapsed_time": "float32", 
    "prior_question_had_explanation": "boolean"
}

df = dd.read_csv("data/train.csv", dtype = dtypes).compute()
print("Train size:", data.shape)
df.head()
```

## datatable

```python
# jupyter lab/notebook
import datatable as dt

%%time
df = dt.fread("data/train.csv") 
print("Train size:", df.shape)
df.head()
```

## rapids

```python
# jupyter lab/notebook

# rapids installation (make sure to turn on GPU)
import sys
!cp ../input/rapids/rapids.0.16.0 /opt/conda/envs/rapids.tar.gz
!cd /opt/conda/envs/ && tar -xzvf rapids.tar.gz > /dev/null
sys.path = ["/opt/conda/envs/rapids/lib/python3.7/site-packages"] + sys.path
sys.path = ["/opt/conda/envs/rapids/lib/python3.7"] + sys.path
sys.path = ["/opt/conda/envs/rapids/lib"] + sys.path

import cudf

%%time
df = cudf.read_csv("data/train.csv") 
print("Train size:", df.shape)
df.head()
```

## 参考资料

- https://www.kaggle.com/code/rohanrao/tutorial-on-reading-large-datasets/notebook

# Python 主流数据存储格式

* csv
    - csv格式是使用最多的一个存储格式，但是其存储和读取的速度会略慢
* feather
    - feather 是一种可移植的文件格式，用于存储 Arrow 表或数据帧(来自 Python 或 R 等语言)，
      它在内部使用 Arrow-IPC 格式。Feather 是在 Arrow 项目早期创建的，
      作为 Python(pandas)和 R 的快速、语言无关的数据帧存储的概念证明 
* hdf5
    - hdf5 设计用于快速 I/O 处理和存储，它是一个高性能的数据管理套件，
      可以用于存储、管理和处理大型复杂数据
* jay
    - datatable 使用 `.jay`(二进制)格式，这使得读取数据集的速度非常快
* parquet
    * 在 Hadoop 生态系统中，parquet 被广泛用作表格数据集的主要文件格式，
      Parquet 使 Hadoop 生态系统中的任何项目都可以使用压缩的、高效的列数据表示的优势。
      现在 parquet 与 Spark 一起广泛使用。
      这些年来，它变得更容易获得和更有效，也得到了 pandas 的支持
* pickle
    - pickle 模块实现二进制协议，用于序列化和反序列化 Python 对象结构。
      Python 对象可以以 pickle 文件的形式存储，pandas 可以直接读取 pickle 文件。
      注意，pickle 模块不安全。最好只 unpickle 你信任的数据

## csv

```python
# jupyter lab/notebook
import pandas as pd

%%time
train_df = pd.read_csv("data/train.csv")
train_df.info
```

## csv 格式转换为 pickle/feather/parquet/jay/h5

```python
import pandas as pd
import datatable as dt

# train_df = dt.fread("data/train.csv").to_pandas()
train_df.to_csv("data/train.csv", index = False)
train_df.to_pickle("data/train.pkl.gzip")
train_df.to_feather("data/train.feather")
train_df.to_parquet("data/train.parquet")
dt.Frame(train_df).to_jay("data/train.jay")
train_df.to_hdf("data/train.h5", "train")
dt.Frame(train_df).to_jay("data/train.jay")
```

## pickle

```python
# jupyter lab/notebook
import pandas as pd

%%time
train_pickle = pd.read_pickle("data/train.pkl.gzip")
train_pickle.info()
```

## feather

```python
# jupyter lab/notebook
import pandas as pd

%%time
train_feather = pd.read_feather("data/train.feather")
train_feather.info()
```

## parquet

```python
# jupyter lab/notebook
import pandas as pd

%%time
train_parquet = pd.read_parquet("data/train.parquet")
train_parquet.info()
```

## jay

```python
# jupyter lab/notebook
import pandas as pd

%%time
train_jay = dt.fread("data/train.jay")
train.jay.shape
```


## 参考资料

- https://www.kaggle.com/code/pedrocouto39/fast-reading-w-pickle-feather-parquet-jay











# pandas


## 参考资料

* https://www.kaggle.com/code/sohier/competition-api-detailed-introduction/notebook


# datatable

## 安装

```bash
$ pip install datatable
```

```python
import datatable as dt
print(dt.__version__)
```

## 核心概念

* `f.`
* `g`


## 核心方法

* 数据加载
    - 创建数据结构
    - 转换数据结构
* 数据属性
    - `DT.shape`
    - `DT.names`
    - `DT.stypes`
* 数据操作
    - 统计方法
        - DT.sum()
        - DT.max()
        - DT.min()
        - DT.mean()
        - DT.sd()
        - DT.mode()
        - DT.nmodal()
        - DT.nunique()
    - 筛选、索引行列
        - `DT[:]`
        - `del DT[:]`
    - 数据合并
        - `DT.rbind()`
        - `DT.cbind()`
    - Groupby/Join
        - `DT[:, ,stat(), by()]`
* 数据卸载
    - `.csv`
    - `.jay`
    - `.`


## 最佳实践



## 参考资料

* https://datatable.readthedocs.io/en/latest/index.html
* https://github.com/parulnith/An-Overview-of-Python-Datatable-package

# Dask

## 安装

```bash
$ pip install 'dask[complete]'  # Install everything

$ pip install dask  # Install only core parts of dask

$ pip install 'dask[array]'  # Install requirements for dask array
$ pip install 'dask[dataframe]'  # Install requirements for dask dataframe
$ pip install 'dask[diagnostics]'  # Install requirements for dask diagnostics
$ pip install 'dask[distributed]'  # Install requirements for distributed dask
```


## 参考资料

* https://www.dask.org/

# RAPIDS


# tqdm




# PySpark