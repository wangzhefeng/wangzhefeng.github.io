---
title: 探索性数据分析概述
author: wangzf
date: '2023-02-27'
slug: eda
categories:
  - data analysis
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

- [探索性数据分析](#探索性数据分析)
    - [探索性数据分析简介](#探索性数据分析简介)
    - [探索性数据分析模块](#探索性数据分析模块)
- [自动探索性数据分析](#自动探索性数据分析)
    - [D-Tale](#d-tale)
    - [Pandas-Profiling](#pandas-profiling)
    - [Sweetviz](#sweetviz)
    - [AutoViz](#autoviz)
    - [Dataprep](#dataprep)
    - [Klib](#klib)
    - [Dabl](#dabl)
    - [Speedml](#speedml)
    - [DataTile](#datatile)
    - [edaviz](#edaviz)
- [参考](#参考)
</p></details><p></p>

# 探索性数据分析

## 探索性数据分析简介

探索性数据分析是上世纪六十年代提出，其方法由美国统计学家 John Tukey 提出的。

探索性数据分析(Exploratory Data Analysis，简称 EDA)，
是指对已有的数据（特别是调查或观察得来的原始数据）在尽量少的先验假定下进行探索，
通过作图、制表、方程拟合、计算特征量等手段探索数据的结构和规律的一种数据分析方法。
特别是对面对大数据的时候，各种杂乱的“脏数据”，往往不知所措，
不知道从哪里开始了解目前拿到手上的数据时候，探索性数据分析就非常有效。

## 探索性数据分析模块

探索性数据分析基本可以汇总到下面的四个模块：

![img](images/eda.jpeg)

1. 全局数据分析
    - 通过数据全局分析，可以了解到数据的整体情况，包括数据类型、大小、质量等
2. 单变量数据分析
    - 单独对每个变量进行观测，包括类别变量、连续变量、文本变量等
3. 交叉特征分析
    - 每个自变量与因变量之间的关系
    - 每个自变量与其他自变量之间的关系
4. 训练集、测试集分布分析：
    - 关于训练集和测试集的分布探索，在目前数据挖掘中最为核心的模块之一，
      训练集和测试集的分布不一致也是导致线上和线下不一致的重要原因之一

# 自动探索性数据分析

Dataprep 是最常用的 EDA 包，AutoViz 和 D-Tale 也是不错的选择，
如果需要定制化分析可以使用 Klib，SpeedML 整合的东西比较多，
单独使用它啊进行 EDA 分析不是特别的适用，其他的包可以根据个人喜好选择，
其实都还是很好用的，最后 edaviz 就不要考虑了，因为已经不开源了。

## D-Tale

D-Tale 使用 Flask 作为后端、React 作为前端并且可以与 ipython notebook 和终端无缝集成。

D-Tale 可以支持 Pandas 的 DataFrame、Series、MultiIndex、DatetimeIndex、RangeIndex。

D-Tale 库用一行代码就可以生成一个报告，其中包含数据集、相关性、图表和热图的总体总结，
并突出显示缺失的值等。D-Tale 还可以为报告中的每个图表进行分析。

```python
import dtale
import pandas as pd

dtale.show(pd.read_csv("titanic.csv"))
```

## Pandas-Profiling

Pandas-Profiling 可以生成 Pandas DataFrame 的概要报告。
Panda-Profiling 扩展了 Pandas DataFrame `df.profile_report()`，
并且在大型数据集上工作得非常好，它可以在几秒钟内创建报告。

```python
import pandas as pd
from pandas_profiling import ProfileReport

profile = ProfileReport(pd.read_csv("titanic.csv"), explorative = True)
profile.to_file("output.html")
```

## Sweetviz

Sweetviz 是一个开源的 Python 库，只需要两行 Python 代码就可以生成漂亮的可视化图，
将 EDA(探索性数据分析)作为一个 HTML 应用程序启动。
Sweetviz 包是围绕快速可视化目标值和比较数据集构建的。

Sweetviz 库生成的报告包含数据集、相关性、分类和数字特征关联等的总体总结。

```python
import pandas as pd
import sweetviz as sv

sweet_report = sv.analyze(pd.read_csv("titanic.csv"))
sweet_report.show_html("sweet_report.html")
```

## AutoViz

Autoviz 包可以用一行代码自动可视化任何大小的数据集，
并自动生成 HTML、bokeh 等报告。用户可以与 AutoViz 包生成的 HTML 报告进行交互。

```python
import pandas as pd
from autoviz.AutoViz_Class import AutoViz_Class

autoviz = AutoViz_Class().AutoViz("train.csv")
```

## Dataprep

Dataprep 是一个用于分析、准备和处理数据的开源 Python 包。
DataPrep 构建在 Pandas 和 Dask DataFrame 之上，可以很容易地与其他 Python 库集成。
DataPrep 的运行速度这 10 个包中最快的，他在几秒钟内就可以为 Pandas/Dask DataFrame 生成报告。

```python
from dataprep.datasets import load_dataset
from dataprep.eda import create_report

df = load_dataset("titanic.csv")
create_report(df).show_browser()
```

## Klib

klib 是一个用于导入、清理、分析和预处理数据的 Python 库。

klibe 虽然提供了很多的分析函数，但是对于每一个分析需要我们手动的编写代码，
所以只能说是半自动化的操作，但是如果我们需要更定制化的分析，他是非常方便的。

```python
import klib
import pandas as pd

df = pd.read_csv("DATASET.csv")
klib.missingval_plot(df)
klib.corr_plot(df_cleaned, annot = False)
klib.dist_plot(df_cleaned["Win_Prob"])
klib.cat_plot(df, figsize=(50,15))
```

## Dabl

Dabl 不太关注单个列的统计度量，而是更多地关注通过可视化提供快速概述，
以及方便的机器学习预处理和模型搜索。

Dabl 中的 `Plot()` 函数可以通过绘制各种图来实现可视化，包括:

* 目标分布图
* 散点图
* 线性判别分析

```python
import pandas as pd
import dabl

df = pd.read_csv("titanic.csv")
dabl.plot(df, target_col="Survived")
```

## Speedml

SpeedML 是用于快速启动机器学习管道的 Python 包。
SpeedML 整合了一些常用的 ML 包，包括 Pandas，Numpy，Sklearn，Xgboost 和 Matplotlib，
所以说其实 SpeedML 不仅仅包含自动化 EDA 的功能。

```python
from speedml import Speedml

sml = Speedml(
    '../input/train.csv', 
    '../input/test.csv',
    target = 'Survived', 
    uid = 'PassengerId'
)
sml.train.head()

sml.plot.correlate()
sml.plot.distribute()
sml.plot.ordinal('Parch')
sml.plot.ordinal('SibSp')
sml.plot.continuous('Age')
```

## DataTile

DataTile（以前称为Pandas-Summary）是一个开源的 Python 软件包，负责管理，
汇总和可视化数据。DataTile 基本上是 Pandas DataFrame `describe()` 函数的扩展。

```python
import pandas as pd
from datatile.summary.df import DataFrameSummary

df = pd.read_csv("titanic.csv")
dfs = DataFrameSummary(df)
dfs.summary()
```

## edaviz

edaviz 是一个可以在 Jupyter Notebook 和 Jupyter Lab 中进行数据探索和可视化的 python 库，
他本来是非常好用的，但是后来被砖厂 (Databricks) 收购并且整合到 bamboolib 中。

# 参考

* [Exploratory Data Analysis](https://www.stat.cmu.edu/~hseltman/309/Book/chapter4.pdf)
* [Tidy Data](https://tomaugspurger.net/posts/modern-5-tidy/)
