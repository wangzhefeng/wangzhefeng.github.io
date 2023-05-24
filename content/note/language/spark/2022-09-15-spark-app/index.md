---
title: 5-Spark 应用程序
author: 王哲峰
date: '2022-09-15'
slug: spark-app
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
img {
    pointer-events: none;
}
</style>

<details><summary>目录</summary><p>

- [Spark 应用程序](#spark-应用程序)
- [Spark Run on cluster](#spark-run-on-cluster)
  - [Spark APP 的体系结构](#spark-app-的体系结构)
  - [Spark APP 内、外部生命周期](#spark-app-内外部生命周期)
  - [Spark 重要的底层执行属性](#spark-重要的底层执行属性)
- [开发 Spark 应用程序](#开发-spark-应用程序)
  - [Spark App](#spark-app)
    - [Scala App](#scala-app)
    - [Python App](#python-app)
    - [Java App](#java-app)
  - [Testing Spark App](#testing-spark-app)
  - [Configuring Spark App](#configuring-spark-app)
- [部署 Spark 应用程序](#部署-spark-应用程序)
- [Spark 应用程序监控和 Debug](#spark-应用程序监控和-debug)
- [Spark 应用程序性能调优](#spark-应用程序性能调优)
</p></details><p></p>

# Spark 应用程序

Spark 可以通过内置的命令行工具 ``spark-submit`` 轻松地将测试级别的交互程序转化为生产级别的应用程序.

通过修改 ``spark-submit`` 的 ``master`` 参数, 可以将将应用程序代码发送到一个集群并在那里执行, 
应用程序将一直运行, 直到正确退出或遇到错误。应用程序需要在集群管理器的支持下进行, 常见的集群管理器有 
Standalone, Mesos 和 YARN 等.

- 示例 1

```bash
./bin/spark-submit \
--class org.apache.spark.examples.SparkPi \      # 运行的类 
--master local \                                 # 在本地机器上运行程序
./examples/jars/spark-examples_2.11-2.2.0.jar 10 # 运行的 JAR 包
```

- 示例 2

```bash
./bin/spark-submit \
-- master local \
./examples/src/main/python/pi.py 10
```

# Spark Run on cluster

- Spark APP 的体系结构、组件   
- Spark APP 内部的生命周期
- Spark APP 外部的生命周期
- Spark 重要的底层执行属性, 例如, 流水线处理
- 运行一个 Spark APP 需要什么

## Spark APP 的体系结构

- Spark APP 的体系结构包含三个基本组件:
    - Spark 驱动器
        - Spark 驱动器是控制应用程序的进程。它负责控制整个 Spark 应用程序的执行并且维护着 Spark 集群的状态, 
          即执行器的任务和状态, 它必须与集群管理器交互才能获得物理资源并启动执行器。简而言之, 
          它只是一个物理机器上的一个进程, 负责维护集群上运行的应用程序状态。
    - Spark 执行器
    - 集群管理器
- Spark APP 选择执行模式。当在运行 Spark APP 之前, 通过选择执行模式将能够确定计算资源的物理位置。Spark 有三种模式可供选择:
    - 集群模式
    - 客户端模式
    - 本地模式

## Spark APP 内、外部生命周期

## Spark 重要的底层执行属性


# 开发 Spark 应用程序

Spark 应用程序：

- a Spark cluster
- application code

## Spark App

### Scala App

Build applications using Java Virtual Machine(JVM) based build tools:

- sbt
- Apache Maven

**1.Build applications using sbt**

- Configure an sbt build for Scala application with a ``build.sbt`` file to manage the package information:
   - Project metadata(package name, package versioning information, etc.)
   - Where to resolve dependencies
   - Dependencies needed for your library

```bash
// build.stb

name := "example"
organization := "com.databricks"
scalaVersion := "2.11.8"

// Spark Information
val sparkVersion = "2.2.0"

// allows us to include spark packages
resolvers += "bintray-spark-packages" at 
"https://dl.bintray.com/spark-package/maven/"

resolvers += "Typesafe Simple Repository" at
"http://repo.typesafe.com/typesafe/simple/maven-releases/"

resolvers += "MavenRepository" at
"https://mvnrepository.com/"

libraryDependencies ++= Seq(
// Spark core
"org.apache.spark" %% "spark-core" % sparkVersion,
"org.apache.spark" %% "spark-sql" % sparkVersion,
// the rest of the file is omitted for brevity
)
```

**2.Build the Project directories using standard Scala project structure**

```bash
src/
main/
    resources/
        <files to include in main jar here>
    scala/
        <main Scala sources>
    java/
        <main Java sources>
test/
    resources/
        <files to include in test jar here>
    scala/
        <test Scala sources>
    java/
        <test Java sources>
```

**3.Put the source code in the Scala and Java directories**

```scala
// in Scala
// src/main/scala/DataFrameExample.scala

import org.apache.spark.sql.SparkSession

object DataFrameExample extends Seriallizable {
    def main(args: Array[String]) = {

        // data source path
        val pathToDataFolder = args(0)

        // start up the SparkSession along with explicitly setting a given config
        val spark = SparkSession
            .builder()
            .appName("Spark Example")
            .config("spark.sql.warehouse.dir", "/user/hive/warehouse")
            .getOrCreate()

        // udf registration
        spark.udf.register(
            "myUDF", someUDF(_: String): String
        )

        // create DataFrame
        val df = spark
            .read
            .format("json")
            .option("path", pathToDataFolder + "data.json")

        // DataFrame transformations an actions
        val manipulated = df
            .groupBy(expr("myUDF(group"))
            .sum()
            .collect()
            .foreach(x => println(x))
    }
}
```

**4.Build Project**

- (1) run ``sbt assemble``
    - build an ``uber-jar`` or ``fat-jar`` that contains all of the dependencies in one JAR
    - Simple
    - cause complications(especially dependency conflicts) for others
- (2) run ``sbt package``
    - gather all of dependencies into the target folder
    - not package all of them into one big JAR

**5.Run the application**

```bash
# in Shell
$ SPARK_HOME/bin/spark-submit \
    --class com.databricks.example.DataFrameExample\
    --master local \
    target/scala-2.11/example_2.11-0.1-SNAPSHOT.jar "hello"
```

### Python App

- build Python scripts;
- package multiple Python files into egg or ZIP files of Spark code;
- use the ``--py-files`` argument of ``spark-submit`` to add ``.py, .zip, .egg`` files to be distributed with application;

**1.Build Python scripts of Spark code**

```python

# in python
# pyspark_template/main.py

from __future__ import print_function

if __name__ == "__main__":
  from pyspark.sql import SparkSession
  spark = SparkSession \
     .builder \
     .master("local") \
     .appName("Word Count") \
     .config("spark.some.config.option", "some-value") \
     .getOrCreate()

  result = spark \
     .range(5000) \
     .where("id > 500") \
     .selectExpr("sum(id)") \
     .collect()
  print(result)
```

**2.Running the application**

```bash
# in Shell
$SPARK_HOME/bin/spark-submit --master local pyspark_template/main.py
```

### Java App

**1.Build applications using mvn**

```xml
<!-- pom.xml -->
<!-- in XML -->
<dependencies>
<dependency>
    <groupId>org.apache.spark</groupId>
    <artifactId>spark-core_2.11</artifactId>
    <version>2.1.0</version>
</dependency>
<dependency>
    <groupId>org.apahce.spark</groupId>
    <artifactId>spark-sql_2.11</artifactId>
    <version>2.1.0</version>
</dependency>
<dependency>
    <groupId>org.apache.spark</groupId>
    <artifactId>graphframes</artifactId>
    <version>0.4.0-spark2.1-s_2.11</version>
</dependency>
</dependencies>
<repositories>
<!-- list of other repositores -->
<repository>
    <id>SparkPackageRepo</id>
    <url>http://dl.bintray.com/spark-packages/maven</url>
</repository>
</repositories>
```

**2.Build the Project directories using standard Scala project structure**

```
src/
main/
    resources/
        <files to include in main jar here>
    scala/
        <main Scala sources>
    java/
        <main Java sources>
test/
    resources/
        <files to include in test jar here>
    scala/
        <test Scala sources>
    java/
        <test Java sources>
```

**3.Put the source code in the Scala and Java directories**

```java
// in Java
import org.apache.spark.sql.SparkSession;
public class SimpleExample {
    public static void main(String[] args) {
        SparkSession spark = SparkSession
            .builder()
            .getOrCreate();
        spark.range(1, 2000).count();
    }
}
```

**4.Build Project**

- Package the source code by using ``mvn`` package;

**5.Running the application**

```bash
# in Shell
$SPARK_HOME/bin/spark-submit \
    --class com.databricks.example.SimpleExample \
    --master local \
    target/spark-example-0.1-SNAPSHOT.jar "Hello"
```

## Testing Spark App

- Strategic Principles
- Tactial Takeaways
- Connecting to Unit Testing Frameworks
- Connecting to Data Source

## Configuring Spark App


# 部署 Spark 应用程序



# Spark 应用程序监控和 Debug


# Spark 应用程序性能调优


