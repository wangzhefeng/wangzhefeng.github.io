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

- [Pyjanitor](#pyjanitor)
  - [coalesc](#coalesc)
  - [concatenate\_columns 和 deconcatenate\_column](#concatenate_columns-和-deconcatenate_column)
  - [take\_first](#take_first)
  - [自定义 janitor](#自定义-janitor)
- [最佳实践](#最佳实践)
  - [pipe](#pipe)
  - [assign](#assign)
  - [query](#query)
  - [resample](#resample)
  - [groupby 和 transform](#groupby-和-transform)
  - [向量化计算](#向量化计算)
  - [assign 和 numpy select](#assign-和-numpy-select)
  - [timeseries](#timeseries)
- [参考](#参考)
</p></details><p></p>


# Pyjanitor

`pyjanitor` 库的灵感来自于 R 语言的 `janitor` 包，英文单词即为清洁工之意，也就是通常用来进行数据处理或清洗数据。
`pyjanitor` 脱胎于 Pandas 生态圈，其使用的核心也是围绕着链式展开，可以使得我们更加专注于每一步操作的动作或谓词（Verbs）

`pyjanitor` 的 API 文档并不复杂，大多数 API 都是围绕着通用的清洗任务而设计。这主要涉及为几部分：

* 操作列的方法（Modify columns）
* 操作值的方法（Modify values）
* 用于筛选的方法（Filtering）
* 用于数据预处理的方法（Preprocessing），主要是机器学习特征处理的一些方法
* 其他方法

需要注意的是，尽管 `pyjanitor` 库名称带有 py 二字，但是在导入时则是输入 `janitor`；
就像 `Beautifulsoup4` 库在导入时写为 `bs4` 一样，以免无法导入而报错

## coalesc

示例：比如我数据中有两个字段 a 和 b，但是两个字段或多或少都有缺失值。
需要定义一个新的字段 c，它由两个字段构建而来。如果第一个字段中存在缺失值，
则取第二个字段中的值，反之亦可；如果两者都为缺失，则保留缺失值

示例数据：

```python
import pandas as pd
import numpy as np

df = pd.DataFrame({
    "a": [None, 2, None, None, 5, 6],
    "b": [1, None, None, 4, None, 6],
})
df
```

pandas 加 apply 实现：

```python
import pandas as pd

def get_valid_value(col_x, col_y):
    if not pd.isna(col_x) and pd.isna(col_y):
        return col_x
    elif pd.isna(col_x) and not pd.isna(col_y):
        return col_y
    elif not (pd.isna(col_x) or pd.isna(col_y)):
        return col_x
    else:
        return np.nan

df["c"] = df.apply(lambda x: get_valid_value(x["a"], x["b"]), axis = 1)
df
```

pyjanitor coalesc 实现：

```python
import janitor

df.coalesc(
    column_names = ["a", "b"],
    new_column_name = "c",
    delete_columns = False,
)
```

## concatenate_columns 和 deconcatenate_column

* `concatenate_columns()` 将多个列根据某个分隔符合并成一个新列
* `deconcatnate_column()` 将单个列拆分成多个列

```python
import pandas as pd
import janitor

df = pd.DataFrame({
    "date_time": [
        "2020-02-01 11:00:00",
        "2020-02-03 12:10:11",
        "2020-03-24 13:24:31"
    ]
})

(
    df
        .deconcatenate_column(
            column_name = "date_time",
            new_column_names = ["date", "time"],
            sep = " ",
            preserve_position = False,
        )
        .deconcatenate_column(
            column_name = "date",
            new_column_names = ["year", "month", "day"],
            sep = "-",
            preserve_position = True,
        )
        .concatenate_columns(
            column_names = ["year", "month", "day"],
            new_column_name = "new_date",
            sep = "-",
        )
)
```

## take_first

有的时候，会 `groupby()` 某个字段并对一些数值列进行操作、倒序排列，
最后每组取最大的数即倒序后的第一行。在 R 语言中可以很轻易直接这么实现：

```r
library(dplyr)

df <- data.frame(
    a = c("x", "x", "y", "y", "y"),
    b = c(1, 3, 2, 5, 4)
)

df %>% 
    group_by(a) %>%
    arrange(desc(b)) %>%
    slice(1) %>%
    ungroup()
```

在没使用 pyjanitor 之前，我往往都是通过 Pandas 这么实现的：

```python
import pandas as pd

df = pd.DataFrame({
    "a": ["x", "x", "y", "y", "y"],
    "b": [1, 3, 2, 5, 4]
})

(
    df
        .groupby("a")
        .apply(lambda grp: grp.sort_values(by = "b", ascending = False).head(1))
        .reset_index(drop = True)
)
```

这里利用了 groupby 之后的生成的 DataFrameGroupBy 对象再进行多余的降序取第一个的操作，
最后将分组后产生的索引值删除。现在可以直接使用 pyjanitor 中的 take_first 方法直接一步到位：

```python
import pandas as pd
import janitor

df = pd.DataFrame({
    "a": ["x", "x", "y", "y", "y"],
    "b": [1, 3, 2, 5, 4]
})

df.take_first(subset = "a", by = "b", ascending = False)
```

## 自定义 janitor

pyjanitor 中的方法仅仅只是一些通用的实现方法，不同的人在使用过程中可能也会有不同的需要。
但好在我们也可以实现自己的 janitor 方法。pyjanitor 得益于 pandas-flavor 库的加持得以轻松实现链式方法

pandas-flavor 提供了能让使用者简单且快速地编写出带有 Pandas 味儿的方法：

* 第一步，只需要在你编写的函数、方法或类中添加对应的装饰器即可
* 第二步，确保最后返回的是 DataFrame 或 Series 类的对象即可

本质上来说，pandas-flavor 库中提供的装饰器就等价于重写或新增了 DataFrame 类的方法，
在使用过程中如果方法有报错，那就需要还原加载 pandas 库之后再重新写入

关于 pandas-flavor 装饰器的用法，详见项目的 Github（https://github.com/Zsailer/pandas_flavor）

比如我们写一个简单清理数据字段或变量名称多余空格的方法：

```python
import pandas as pd
import pandas_flavor as pf

@pf.register_dataframe_method
def strip_names(df):
    import re

    colnames = df.columns.tolist()
    colnames = list(map(lambda col: '_'.join(re.findall(r"\w+", col)), colnames))
    df.columns = colnames
    return df
```

```python
data = pd.DataFrame({
    " a ": [1, 1],
    "  b  zz  ": [2, 1],
})
data
dta.strip_names()
```




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


# 参考

* [Python 自动探索性数据分析神库](https://mp.weixin.qq.com/s/F9Ixe9_d4XDxK-MJOMaVqQ)
* [用 Pyjanitor 更好地进行数据清洗与处理](https://mp.weixin.qq.com/s/9AasPTO-7Caku3_5CMj5kQ)

