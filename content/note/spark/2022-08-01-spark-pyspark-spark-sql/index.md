---
title: PySpark Spark SQL
subtitle: Apache Arrow in PySpark
author: 王哲峰
date: '2022-08-01'
slug: spark-pyspark-spark-sql
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

- [Apache Arrow in PySpark](#apache-arrow-in-pyspark)
  - [Apache Arrow](#apache-arrow)
  - [PyArrow](#pyarrow)
    - [推荐安装版本](#推荐安装版本)
    - [安装方式](#安装方式)
- [启用与 Pandas 之间的转换](#启用与-pandas-之间的转换)
- [Pandas UDFs](#pandas-udfs)
- [Pandas Function APIs](#pandas-function-apis)
</p></details><p></p>

# Apache Arrow in PySpark

Apache Arrow is an in-memory columnar data format that is used 
in Spark to efficiently transfer data between JVM and Python processes

Its usage is not automatic and might require some minor changes 
to configuration or code to take full advantage and ensure compatibility

## Apache Arrow

https://arrow.apache.org/docs/index.html

## PyArrow

### 推荐安装版本

* pyarrow==1.0.0
* pandas>=1.0.5

### 安装方式

* 如果 PySpark 是通过 pip 安装的，那么 PyArrow 可以通过如下命名作为 SQL 模块的依赖项自动安装

```bash
$ pip install "pyspark[sql]"
```

* [其他](https://arrow.apache.org/docs/python/install.html)

# 启用与 Pandas 之间的转换

在使用 `DataFrame.toPandas()` 将 Spark DataFrame 转换为 Pandas DataFrame，
以及使用 `SparkSession.createDataFrame()` 从 Pandas DataFrame 创建 Spark DataFrame 时，
Arrow 可以作为一个优化器。

要在执行这两种转换操作时使用 Array，
用户首先需要在 Spark 配置项中将 `spark.sql.execution.arrow.pyspark.enable` 设置为 `true`，
默认情况下这个配置项为 `false`

此外，如果在 Spark 实际计算之前发生错误，
那么通过 `spark.sql.execution.arrow.pyspark.enable` 启用的优化
可以通过 `spark.sql.execution.arrow.pyspark.fallback.enable` 配置项
自动回退到非 Arrow 优化实现

```python
import numpy as np
import pandas as pd

# Enable Arrow-based columnar data transfers
spark.conf.set("spark.sql.execution.arrow.pyspark.enable", "true")

# Pandas DataFrame
pdf = pd.DataFrame(np.random.rand(100, 3))

# Pandas DataFrame -> Spark DataFrame using Arrow
df = spark.createDataFrame(pdf)

# Spark DataFrame -> Pandas DataFrame using Arrow
result_pdf = df.select("*").toPandas()

print(f"Pandas DataFrame result statistics:\n{str(result_pdf.descrive())}\n")
```

# Pandas UDFs



# Pandas Function APIs

