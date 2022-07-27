---
title: PySpark 快速入门
author: 王哲峰
date: '2022-07-27'
slug: spark-pyspark-quickstart
categories:
  - spark
tags:
  - tool
---

<style>
h1 {
  background-color: #2B90B6;
  background-image: linear-gradient(45deg, #4EC5D4 10%, #146b8c 20%);
  background-size: 100%;
  -webkit-background-clip: text;
  -moz-background-clip: text;
  -webkit-text-fill-color: transparent;
  -moz-text-fill-color: transparent;
}
h2 {
  background-color: #2B90B6;
  background-image: linear-gradient(45deg, #4EC5D4 10%, #146b8c 20%);
  background-size: 100%;
  -webkit-background-clip: text;
  -moz-background-clip: text;
  -webkit-text-fill-color: transparent;
  -moz-text-fill-color: transparent;
}


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

- [DataFrame 创建](#dataframe-创建)
- [DataFrame 数据查看](#dataframe-数据查看)
- [筛选数据](#筛选数据)
- [Apply](#apply)
- [Grouping](#grouping)
- [Getting Data in/out](#getting-data-inout)
- [Working with SQL](#working-with-sql)
</p></details><p></p>


```
from pyspark.sql import SparkSession
spark = SparkSession.builder.getOrCreate()
```

# DataFrame 创建

```python
from datetime import datetime, date
import pandas as pd
from pyspark.sql import Row

df = spark.createDataFrame([
    Row(a = 1, b = 2., c = "string1", d = date(2000, 1, 1), e = datetime(2000, 1, 1, 12, 0)),
    Row(a = 2, b = 3., c = "String2", d = date(2000, 2, 1), e = datetime(2000, 1, 2, 12, 0)),
    Row(a = 4, b = 5., c = "String3", d = date(2000, 3, 1), e = datetime(2000, 1, 3, 12, 0)),
])
df

df = spark.createDataFrame([
    (1, 2., 'string1', date(2000, 1, 1), datetime(2000, 1, 1, 12, 0)),
    (2, 3., 'string2', date(2000, 2, 1), datetime(2000, 1, 2, 12, 0)),
    (3, 4., 'string3', date(2000, 3, 1), datetime(2000, 1, 3, 12, 0))
], schema = 'a long, b double, c string, d date, e timestamp')
df

pandas_df = pd.DataFrame({
    'a': [1, 2, 3],
    'b': [2., 3., 4.],
    'c': ['string1', 'string2', 'string3'],
    'd': [date(2000, 1, 1), date(2000, 2, 1), date(2000, 3, 1)],
    'e': [datetime(2000, 1, 1, 12, 0), datetime(2000, 1, 2, 12, 0), datetime(2000, 1, 3, 12, 0)]
})
df = spark.createDataFrame(pandas_df)
df
```



# DataFrame 数据查看

# 筛选数据

# Apply

# Grouping

# Getting Data in/out

# Working with SQL



