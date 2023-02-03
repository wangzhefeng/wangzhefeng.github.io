---
title: Pandas
author: 王哲峰
date: '2022-10-03'
slug: python-pandas
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
</style>

<details><summary>目录</summary><p>

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
- [最佳实践](#最佳实践)
  - [pipe](#pipe)
  - [assign](#assign)
  - [query](#query)
  - [resample](#resample)
  - [groupby 和 transform](#groupby-和-transform)
  - [向量化计算](#向量化计算)
  - [assign 和 numpy select](#assign-和-numpy-select)
  - [timeseries](#timeseries)
</p></details><p></p>

# 自动探索性数据分析

Dataprep 是我最常用的 EDA 包，AutoViz 和 D-Tale 也是不错的选择，
如果你需要定制化分析可以使用 Klib，SpeedML 整合的东西比较多，
单独使用它啊进行 EDA 分析不是特别的适用，其他的包可以根据个人喜好选择，
其实都还是很好用的，最后 edaviz 就不要考虑了，因为已经不开源了

## D-Tale

D-Tale 使用 Flask 作为后端、React 作为前端并且可以与 ipython notebook 和终端无缝集成

D-Tale 可以支持 Pandas 的 DataFrame、Series、MultiIndex、DatetimeIndex、RangeIndex

D-Tale 库用一行代码就可以生成一个报告，其中包含数据集、相关性、图表和热图的总体总结，
并突出显示缺失的值等。D-Tale 还可以为报告中的每个图表进行分析


```python
import dtale
import pandas as pd

dtale.show(pd.read_csv("titanic.csv"))
```


## Pandas-Profiling

Pandas-Profiling 可以生成 Pandas DataFrame 的概要报告。
panda-profiling 扩展了 pandas DataFrame `df.profile_report()`，
并且在大型数据集上工作得非常好，它可以在几秒钟内创建报告

```python
import pandas as pd
from pandas_profiling import ProfileReport

profile = ProfileReport(pd.read_csv("titanic.csv"), explorative = True)
profile.to_file("output.html")
```

## Sweetviz

Sweetviz 是一个开源的 Python 库，只需要两行 Python 代码就可以生成漂亮的可视化图，
将 EDA(探索性数据分析)作为一个 HTML 应用程序启动。
Sweetviz 包是围绕快速可视化目标值和比较数据集构建的

Sweetviz 库生成的报告包含数据集、相关性、分类和数字特征关联等的总体总结

```python
import pandas as pd
import sweetviz as sv

sweet_report = sv.analyze(pd.read_csv("titanic.csv"))
sweet_report.show_html("sweet_report.html")
```

## AutoViz

Autoviz 包可以用一行代码自动可视化任何大小的数据集，
并自动生成 HTML、bokeh 等报告。用户可以与 AutoViz 包生成的 HTML 报告进行交互

```python
import pandas as pd
from autoviz.AutoViz_Class import AutoViz_Class

autoviz = AutoViz_Class().AutoViz("train.csv")
```

## Dataprep

Dataprep 是一个用于分析、准备和处理数据的开源 Python 包。
DataPrep 构建在 Pandas 和 Dask DataFrame 之上，可以很容易地与其他 Python 库集成。
DataPrep 的运行速度这 10 个包中最快的，他在几秒钟内就可以为 Pandas/Dask DataFrame 生成报告

```python
from dataprep.datasets import load_dataset
from dataprep.eda import create_report

df = load_dataset("titanic.csv")
create_report(df).show_browser()
```

## Klib

klib 是一个用于导入、清理、分析和预处理数据的 Python 库

klibe 虽然提供了很多的分析函数，但是对于每一个分析需要我们手动的编写代码，
所以只能说是半自动化的操作，但是如果我们需要更定制化的分析，他是非常方便的

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
以及方便的机器学习预处理和模型搜索

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
所以说其实 SpeedML 不仅仅包含自动化 EDA 的功能

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
汇总和可视化数据。DataTile 基本上是 Pandas DataFrame `describe()` 函数的扩展

```python
import pandas as pd
from datatile.summary.df import DataFrameSummary

df = pd.read_csv("titanic.csv")
dfs = DataFrameSummary(df)
dfs.summary()
```

## edaviz

edaviz 是一个可以在 Jupyter Notebook 和 Jupyter Lab 中进行数据探索和可视化的 python 库，
他本来是非常好用的，但是后来被砖厂 (Databricks) 收购并且整合到 bamboolib 中


# 最佳实践

## pipe

`pipe` 即为管道, 把前一项输出的 df，作为后一项输入的 df,
同时把 df 操作函数对象作为第一参数，它所需的参数以 `args` 和 `kwargs` 形式传入。
这样能避免产生中间的 df。当参数复杂(比如是巨大的 dictionary，或者是一连串函数计算后的结果)、
或高阶方法多时，比直接 chaining 可读性高

在每次分析工作完成后，把琐碎的数据清理工作以 pipe 的形式放在数据导入后的下一步

```python
dtype_mapping = {
    "a": np.float32,
    "b": np.uint8,
    "c": np.float64,
    "d": np.int64,
    "e": str,
}
df_cleaned = (df
    .pipe(pd.DataFrame.sort_index, ascending = False)  # 按索引排序
    .pipe(pd.DataFrame.fillna, value = 0, method = "ffill")  # 缺失值处理
    .pipe(pd.DataFrame.astype, dtype_mapping)  # 数据类型转换
    .pipe(pd.DataFrame.clip, lower = -0.05, upper = 0.05)  # 极端值处理
)
```

封装成一个函数：

```python
def clean_data(df):
    df_cleaned = (df
        .pipe(pd.DataFrame.sort_index, ascending = False)  # 按索引排序
        .pipe(pd.DataFrame.fillna, value = 0, method = "ffill")  # 缺失值处理
        .pipe(pd.DataFrame.astype, dtype_mapping)  # 数据类型转换
        .pipe(pd.DataFrame.clip, lower = -0.05, upper = 0.05)  # 极端值处理
    )
    return df_cleaned
```

## assign

可以使用 `assign` 方法，把一些列生成操作集中在一起。
和直接用 `df['x'] = ...` 不同的是 `assign` 方法会生成一个新的 df，
原始的 df 不会变 ，不会有 setting with copy warning，
还有一个好处，就是不会因为生成新的操作而打断函数 chaining

```python
df = pd.DataFrame(
    data = 25 + 5 * np.random.randn(10),
    columns = ["temp_c"]
)

df_new = df.assign(
    temp_f = lambda x: x["temp_c"] * 9 / 5 + 32,
    temp_k = lambda x: (x["temp_f"] + 459.67) * 5 / 9,
)
```

## query

用 `query` 可以解决很多条件的筛选问题，明显 `query` 方法简洁，而且条件越多，
逻辑判断越多，可读性优势就越明显(前提是单个条件都是简单的判断)

```python
df = pd.DataFrame(
    data = np.random.randn(10, 3),
    columns = list("abc")
)

# 普通方法
df.loc[((df["a"] > 0) & (df["b"] < 0.05)) | (df["c"] > df["b"])]

# query
df.query("(a > 0 and b < 0.05) or c > b")
```


## resample



## groupby 和 transform





## 向量化计算



## assign 和 numpy select

在下面 df 上成列 c，如果同行列 a 的值 >0.5 并且 <0.9，那么列 c 同行的值等于列 b，
否则为 None

```
a          b
2007-01-08  0.786667        270
2007-01-09  0.853333        280
2007-01-10  0.866667        282
2007-01-11  0.880000        277
2007-01-12  0.880000        266
2007-01-15  0.866667        279
```

用 `df.where` 是最直接的解法，但是 where 有缺点，就是一次只能处理一个条件。
就是 condition1 满足，赋值 v1。不满足则 other 

还有一种用 numpy 的 select。这个方法的好处是可以给定任意条件，并匹配对应的值。
满足条件 1，赋值 v1；满足条件 2，赋值 v2...。如果条件多了，也能一次完成赋值操作

普通方法：

```python
def abcd_to_e(x):
    if x['a']>1:
        return 1
    elif x['b']<0:
        return x['b']
    elif x['a']>0.1 and x['b']<10:
        return 10
    elif ...:
        return 1000
    elif ...
    
    else: 
        return x['c']

df.apply(abcd_to_e, axis = 1)
```

numpy select 方法：

```python
np.random.seed(123)

df = pd.DataFrame(
    np.random.randn(10, 2),
    columns = list("ab")
)

df.assign(c = np.select(
    [(df.a > 0.5) & (df.b < 0.9)], 
    [df.b], 
    default = np.nan
))
```


## timeseries

一年的时间序列数据，读取每月第一天的数据

```python
index = pd.date_range("01/01/2021", "12/31/2021")
df = pd.DataFrame(
    data = np.random.randn(index.size, 3),
    index = index,
    columns = list("abc")
)
# 删除所有单数月份的1号
df_droped = df.drop(
    labels = [datetime.datetime(2021, i, 1) for i in range(1, 13, 2)]
)
```

方法 1:

```python
df_droped.loc[df_droped.index.day == 1]
```

方法 2:

```python
df_droped.resample("MS").first()
df_droped.resample("MS").agg("first")
```

方法 3:

```python
df_droped.asfreq("MS")
```

方法 4:

```python
df_droped.groupby(pd.Grouper(freq = "MS")).first()
```

