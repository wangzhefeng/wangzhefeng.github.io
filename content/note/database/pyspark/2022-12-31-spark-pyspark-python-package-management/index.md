---
title: PySpark Python 包管理
subtitle: PySpark Python Package Management
author: 王哲峰
date: '2022-12-31'
slug: spark-pyspark-python-package-management
categories:
  - spark
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

- [PySpark Python 包管理介绍](#pyspark-python-包管理介绍)
  - [PySpark App 示例](#pyspark-app-示例)
  - [PySpark 原生功能](#pyspark-原生功能)
- [Conda](#conda)
  - [Tools](#tools)
  - [创建、打包环境](#创建打包环境)
  - [传输程序脚本及存档文件](#传输程序脚本及存档文件)
    - [spark-submit script](#spark-submit-script)
    - [Python shells 或 notebook](#python-shells-或-notebook)
    - [pyspark shell](#pyspark-shell)
- [Virtualenv](#virtualenv)
  - [Tools](#tools-1)
  - [创建、打包环境](#创建打包环境-1)
  - [传输程序脚本及存档文件](#传输程序脚本及存档文件-1)
    - [spark-submit script](#spark-submit-script-1)
    - [Python shells 或 notebook](#python-shells-或-notebook-1)
    - [pyspark shell](#pyspark-shell-1)
- [PEX](#pex)
  - [Tools](#tools-2)
  - [创建、打包环境](#创建打包环境-2)
  - [传输程序脚本及存档文件](#传输程序脚本及存档文件-2)
    - [spark-submit script](#spark-submit-script-2)
    - [Python shells 或 notebooks](#python-shells-或-notebooks)
    - [pyspark shell](#pyspark-shell-2)
- [TODO](#todo)
</p></details><p></p>

# PySpark Python 包管理介绍

当需要在 YARN、Kubernetes、Mesos 集群管理器上运行 PySpark App 时，
需要确定 Python 程序及其运行所需的依赖库已经安装在执行器(executor)上

## PySpark App 示例

* 一个将要在集群上运行的应用程序

```python
# app.py

import pandas as pd
from pyspark.sql.functions import pandas_udf
from pyspark.sql import SparkSession


def main(spark):
    df = spark.createDataFrame([
        (1, 1.0), 
        (1, 2.0), 
        (2, 3.0), 
        (2, 5.0), 
        (2, 10.0)
    ], schema = ("id", "v"))

    @pandas_udf("double")
    def main_udf(v: pd.Series) -> float:
        return v.mean()
    
    print(df.groupby("id").agg(mean_udf(df["v"])).collect())

if __name__ == "__main__":
    main(SparkSession.builer.getOrCreate())
```

## PySpark 原生功能

PySpark 允许通过以下方式加载 Python 文件(`.py`)、压缩的 Python 包(`.zip`)、Egg 文件(`.egg`) 到执行器(executors)

* 在配置项中设置 `spark.submit.pyFiles`
* 在 Spark 脚本中设置 `--py-files` 选项
* 直接在应用程序中调用 [`pyspark.SparkContext.addPyFile()`](https://spark.apache.org/docs/latest/api/python/reference/api/pyspark.SparkContext.addPyFile.html#pyspark.SparkContext.addPyFile)

这是一种将自定义 Python 代码发送到集群的简单方法。
可以只添加单个文件或压缩整个包并上传它们。
使用 `pyspark.SparkContext.addPyFile()` 允许即使在开始工作后上传代码

但是，它不允许添加构建为 Wheels 的包，因此不允许包含与本机代码的依赖项

# Conda

## Tools

* [conda](https://docs.conda.io/en/latest/)
* [conda-pack](https://conda.github.io/conda-pack/spark.html)

## 创建、打包环境

下面的示例创建了一个用于驱动器(driver)和执行器(executor)的 Conda 环境，
并将其打包到一个存档文件(archive)中。
此存档文件(archive)捕获 Python 的 Conda 环境并存储 Python 解释器及其所有相关依赖项

```bash
$ conda create -y -n pyspark_conda_env -c conda-forge pyarrow pandas conda-pack
$ conda activate pyspark_conda_env
$ conda pack -f -o pyspark_conda_env.tar.gz
```

## 传输程序脚本及存档文件

使用如下方法，将 archive 文件与应用程序代码一起传输到集群中，
它会在执行器上自动解压 archive 文件

* 在 `spark-submit` 命令中使用 `--archives` 选项
* 设置 `spark.archives` 配置项(`spark.yarn.dist.archives` in YARN)

### spark-submit script

```bash
$ export PYSPARK_DRIVER_PYTHON=python  # Do not set in cluster modes.
$ export PYSPARK_PYTHON=./environment/bin/python
$ spark-submit --archives pyspark_conda_env.tar.gz#environment app.py
```

### Python shells 或 notebook

```python
import os
from pyspark.sql import SparkSession
from app import main

os.environ["PYSPARK_PYTHON"] = "./environment/bin/python"
spark = SparkSession.builder.config(
    "spark.archives",  # 'spark.yarn.dist.archives' in YARN.
    "pyspark_conda_env.tar.gz#envionment"
).getOrCreate()

main(spark)
```

### pyspark shell

```bash
$ export PYSPARK_DRIVER_PYTHON=python
$ export PYSPARK_PYTHON=./environment/bin/python
$ pyspark --archives pyspark_conda_env.tar.gz#environment
```

# Virtualenv

## Tools

* [virtualenv](https://virtualenv.pypa.io/en/latest/)
* [venv](https://docs.python.org/3/library/venv.html)
* [venv-pack](https://jcristharif.com/venv-pack/index.html)

## 创建、打包环境

将虚拟环境打包为 archive 文件，archive 文件包含了 Python 解释器和依赖。
要求集群中所有节点都安装相同的 Python 解释器因为 venv-pack 将 Python 解释器打包为符号链接

```bash
$ python -m venv pyspark_venv
$ source pyspark_venv/bin/activate
$ pip install pyarrow pandas venv-pack
$ venv-pack -o pyspark_venv.tar.gz
```

## 传输程序脚本及存档文件

使用如下方法，可以直接将 archive 文件传递和解包，并在执行器(executors)上启动环境

* 在 `spark-submit` 命令中使用 `--archives` 选项
* 设置 `spark.archives` 配置项(`spark.yarn.dist.archives` in YARN)

### spark-submit script

```bash
$ export PYSPARK_DRIVER_PYTHON=python  # Do not set in cluster modes.
$ export PYSPARK_PYTHON=./environment/bin/python
$ spark-submit --archives pyspark_venv.tar.gz#environment app.py
```

### Python shells 或 notebook

```python
import os
from pyspark.sql import SparkSession
from app import main

os.environ["PYSPARK_PYTHON"] = "./environment/bin/python"
spark = SparkSession.builder.config(
    "spark.archives",  # 'spark.yarn.dist.archives' in YARN.
    "pyspark_env.tar.gz#environment"
).getOrCreate()
main(spark)
```

### pyspark shell

```bash
$ export PYSPARK_DRIVER_PYTHON=pyhton
$ export PYSPARK_PYTHON=./environment/bin/python
$ pyspark --archives pyspark_venv.tar.gz#environment
```

# PEX

## Tools

* [PEX](https://github.com/pantsbuild/pex)


## 创建、打包环境

```bash
$ pip install pyarrow pandas pex
$ pex pyspark pyarrow pandas -o pyspark_pex_env.pex
```

`.pex` 文件的行为与 Python 解释器很类似，但是 `.pex` 文件本身并不包含 Python 解释器，
因此，集群中所有节点都应该安装相同的 Python 解释器

```bash
$ ./pyspark_pex_env.pex -c "import pandas; print(pandas.__version__)"
1.1.5
```

## 传输程序脚本及存档文件

为了在集群中使用 `.pex` 文件，应该通过以下方法来将它传输到集群

*  在 `spark-submit` 命令中使用 `--files` 选项
* `spark.files` 配置(YARN 中的 `spark.yarn.dist.files`)

### spark-submit script

```bash
$ export PYSPARK_DRIVER_PYTHON=python  # Do not set in cluseter modes
$ export PYSPARK_PYTHON=./pyspark_pex_env.pex
$ spark-submit --files pyspark_pex_env.pex app.py
```

### Python shells 或 notebooks

```python
import os
from pyspark.sql import SparkSession
from app import main

os.envircon["PYSPARK_PYTHON"] = "./pyspark_pex_env.pex"
spark = SparkSession.builder.config(
    "spark.files",  # "spark.yarn.dist.files" in YARN
    "pyspark_pex_env.pex"
), getOrCreate()
main(spark)
```

### pyspark shell

```bash
$ export PYSPARK_DRIVER_PYTHON=python
$ export PYSPARK_PYTHON=./pyspark_pex_env.pex
$ pyspark --files pyspark_pex_env.pex
```

# TODO

- [Docker example with Spark on S3 storage](https://github.com/criteo/cluster-pack/blob/master/examples/spark-with-S3/README.md)