---
title: Spark 环境
author: 王哲峰
date: '2022-07-27'
slug: spark-env
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

- [安装 Spark](#安装-spark)
- [安装 PySpark](#安装-pyspark)
  - [普通安装](#普通安装)
  - [支持 Spark SQL](#支持-spark-sql)
  - [pandas API on Spark](#pandas-api-on-spark)
  - [针对特定的 Hadoop 版本](#针对特定的-hadoop-版本)
  - [pyspark 依赖库](#pyspark-依赖库)
- [ERROR](#error)
  - [pyspark 版本问题](#pyspark-版本问题)
- [PySpark 资料](#pyspark-资料)
- [TODO](#todo)
</p></details><p></p>

# 安装 Spark

* 官方安装文档
    - https://spark.apache.org/downloads.html
    ![spark_org_download](images/spark_org_download.png)

# 安装 PySpark

## 普通安装

```shell
pip install pyspark
```

![pyspark_install](images/pyspark_install.png)

## 支持 Spark SQL

```shell
pip install spark[sql]
```

## pandas API on Spark

```shell
pip install pyspark[pandas_on_spark] plotly
```

## 针对特定的 Hadoop 版本

- 默认使用 Hadoop 3.2 和 Hive 3.2

```shell
# 设置 Hadoop 版本
$ PYSPARK_HADOOP_VERSION=2.7 pip install pyspark
# 设置镜像及 Hadoop 版本
$ PYSPARK_RELEASE_MIRROR=http://mirror.apache-kr.org PYSPARK_HADOOP_VERSION=2.7 pip install
# 跟踪安装和下载状态
$ PYSPARK_HADOOP_VERSION=2.7 pip install pyspark -v
```

## pyspark 依赖库

- `py4j`: https://www.py4j.org/

![py4j](images/py4j.png)

# ERROR

## pyspark 版本问题

- 报错信息：

```
py4j.protocol.Py4JError: org.apache.spark.api.python.PythonUtils.getEncryptionEnabled does not exist in the JVM
```

- 解决方法：

```shell
pip install pyspark=="2.4.7"
```

# PySpark 资料

- 官方文档 https://spark.apache.org/docs/latest/api/python/index.html

# TODO

- 通过 Docker 构建集群环境
- 构建 Python 包

