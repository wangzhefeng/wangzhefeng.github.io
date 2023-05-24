---
title: Spark DataSet
author: 王哲峰
date: '2022-09-15'
slug: spark-dataset
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

- [Dataset 介绍](#dataset-介绍)
- [Create DataSet](#create-dataset)
  - [Java: Encoders](#java-encoders)
  - [Scala: case class](#scala-case-class)
- [Actions](#actions)
- [Transformations](#transformations)
  - [DataFrame 上的 Transformation 操作](#dataframe-上的-transformation-操作)
  - [DataSet 特有的 Transformation 操作](#dataset-特有的-transformation-操作)
- [Joins](#joins)
- [Grouping and Aggregations](#grouping-and-aggregations)
</p></details><p></p>


# Dataset 介绍

- Dataset 是 Spark 结构化 API 的基本类型, 而DataFrame 是 Row 类型的 Dataset. 
- Dataset 具有严格的 Java 虚拟机(JVM)语言特性, 仅与 Scala 和 Java 一起使用, 
  可以定义 Dataset 中每一行所包含的对象, 在 Scala 中就是一个 case 类对象, 
  它实质上定义了一种模式 schema, 而在 Java 中就是 Java Bean.
- Dataset 在编译时检查类型, DataFrame 在运行时检查类型.
- 使用 DataFrame API 时, 不需要创建字符串或整数, Spark 就可以通过操作 Row 对象来处理数据。
  如果使用 Scala 或 Java, 则所有 DataFrame 实际上都是 Row 类型的 Dataset. 
  为了有效地支持特定领域的对象, 需要一个称为”编码器(Encoder)“的特殊概念, 
  编码器将特定领域类型 T 映射为 Spark 的内部类型.
- 当使用 Dataset API 时, 将 Spark Row 格式的每一行转换为指定的特定领域类型的对象(case 类或 Java 类), 
  这种转换会减慢操作速度, 但可以提供更大的灵活性.

DataSet 的使用场景:

- 1.当有些业务逻辑要执行的操作无法使用 DataFrame 操作表示
    - 有些操作不能使用结构化 API 来表示, 比如有些业务逻辑想用特定的函数而非 SQL 或 DataFrame 来实现, 就需要用到 Dataset
- 2.如果需要类型安全, 并且愿意牺牲一定性能来实现它
    - 因为 Dataset API 是类型安全的, 对于其类型无效的操作（例如, 两个字符串类型相减）将在编译时出错, 
      而不是在运行时失败, 如果正确性和防御性代码（bulletproof code）是更需要考虑的事情, 
      所以牺牲一些性能或许是最好的选择, 这不能保证不接受格式错误的数据, 但可以更方便地处理
- 3.在单节点作业和 Spark 作业之间重用对行的各种转换代码
    - Spark 的 API 包含了 Scala Sequence 类型, 它们以分布式方式运行, 
      因此使用 Dataset 的一个优点是, 如果你将所有数据和转换定义为 case 类, 
      那么在分布和单机作业中使用它们没什么区别, 此外, 当你在本地磁盘存储 DataFrame 时, 
      它们一定是正确的类和类型, 这使进一步的操作更容易

Dataset 最常用的应用场景可能是 **先用 DataFrame 再用 Dataset**, 
这可以手动在性能和类型安全之间进行权衡. 比如：当基于 DataFrame 执行 ETL 转换作业之后, 
想将数据送入驱动器并使用单机库操作, 或者当需要在 Spark SQL 中执行过滤和进一步操作前, 
进行每行分析的预处理转换操作的时候.

# Create DataSet

创建一个 DataSet 是一个纯手工操作, 需要事先知道并且定义数据的 schema.

## Java: Encoders

- Java 编码器相当简单, 只需指定类, 然后在需要 DataFrame（即 `Dataset<Row>` 类型）的时候对该类进行编码：

```java
// in Java
import org.apache.spark.sql.Encoders;

// 定义 Flight 类
public class Flight implements Serializable{
   String DEST_COUNTRY_NAME;
   String ORIGIN_COUNTRY_NAME;
   Long DEST_COUNTRY_NAME;
}

// 创建 DataFrame(Dataset<Flight>)
DataSet<Flight> flights = spark
  .read
  .parquet("/data/flight-data/parquet/2010-summary.parquet/")
  .as(Encoders.bean(Flight.class));
```

## Scala: case class

在 Scala 中创建 Dataset, 要定义 Scala 的 case 类, Scala `case class` 具有以下特征:

- 不可变(Immutable)
- 通过模式匹配可分解(Decomposable through pattern matching), 来获取类属性
- 允许基于结构的比较, 而不是基于引用进行比较(Allows for comparision based on structrue instead of reference)
- 易用、易操作(Easy to use and manipulate)

```scala
// 定义 DataSet Flight 的 schema
case class Flight(
   DEST_COUNTRY_NAME: String, 
   ORIGIN_COUNTRY_NAME: String, 
   count: BigInt
)

// 创建 DataFrame
val flightsDF = spark
  .read
  .parquet("/data/flight-data/parquet/2010-summary.parquet/")

val flights = flightsDF.as[Flight]
```

# Actions

DataFrame 上的 Action 操作也对 DataSet 有效;

```scala
flights.show(2)
flights.collect()
flights.take()
flights.count()

flights.first.DEST_COUNTRY_NAME
```

# Transformations

- DataFrame 上的 Transformation 操作也对 DataSet 有效
- 除了 DataFrame 上的 Transformation, Dataset 上也有更加复杂和强类型的 Transformation 操作, 
  因为, 操作 Dataset 相当于操作的是原始的 Java Virtual Machine (JVM) 类型

## DataFrame 上的 Transformation 操作


## DataSet 特有的 Transformation 操作

- Filtering

```scala
def originIsDestination(flight_row: Flight): Boolean = {
    return flight_row.ORIGIN_COUNTRY_NAME == flight_row.DEST_COUNTRY_NAME
}

flights
   .filter(flight_row => originIsDestination(flight_row))
   .first()
```

- Mapping

```scala
val destinations = flights.map(f => f.DEST_COUNTRY_NAME)
val localDestinations = destinations.take(5)
```

# Joins

```scala
case class FlightMetadata(
   count: BigInt, 
   randomData: BigInt
)

val flightsMeta = spark
   .range(500)
   .map(x => (x, scala.unit.Random.nextLong))
   .withColumnRenamed("_1", "count")
   .withColumnRenamed("_2", "randomData")
   .as[FlightMetadata]

val flights2 = flights
   .joinWith(flightsMeta, flights.col("count") === flightsMeta.col("count"))
```

```scala
flights2.selectExpr("_1.DEST_COUNTRY_NAME")
flights2.take(2)
val flights2 = flights.join(flightsMeta, Seq("count"))
val flights2 = flights.join(flightsMeta.toDF(), Seq("count"))
val flights2 = flights.join(flightsMeta.toDF(), Seq("count"))
```

# Grouping and Aggregations

- DataSet 中的 Grouping 和 Aggregation 跟 DataFrame 中的 Grouping 和 Aggregation 一样的用法, 
  因此, `groupBy`, `rollup` 和 `cube` 对 DataSet 依然有效, 只不过不再返回 DataFrame, 
  而是返回 DataSet, 实际上是丢弃了 type 信息.
- 如果想要保留 type 信息, 有一些方法可以实现, 比如: `groupByKey`, 
  `groupByKey` 可以通过 group 一个特殊的 DataSet key, 然后返回带有 type 信息的 DataSet；
  但是 `groupByKey` 不再接受一个具体的 column 名字, 而是一个函数, 
  这样使得可以使用一些更加特殊的聚合函数来对数据进行聚合。
  但是这样做虽然灵活, 却失去了性能上的优势。

```scala
// in Scala
// groupBy
flights.groupBy("DEST_COUNTRY_NAME").count()

// groupByKey
flights.groupByKey(x => x.DEST_COUNTRY_NAME).count()
flights.groupByKey(x => x.DEST_COUNTRY_NAME).count().explain
```

```scala
def grpSum(countryName: String, values: Iterator[Flight]) = {
   values.dropWhile(_.count < 5).map(x => (countryName, x))
}

flights
   .groupByKey(x => x.DEST_COUNTRY_NAME)
   .flatMapGroups(grpSum)
   .show(5)
```

```scala
def grpSum2(f: Flight): Integer = {
   1
}
flights2
   .groupByKey(x => x.DEST_COUNTRY_NAME)
   .mapValues(grpSum2)
   .count()
   .take(5)
```

```scala
// in Scala
// 创建新的操作并定义如何执行 reduceGroups 聚合
def sum2(left: Flight, right: Flight) = {
   Flight(left.DEST_COUNTRY_NAME, null, left.count + right.count)
}

flights
   .groupByKey(x => x.DEST_COUNTRY_NAME)
   .reduceGroups((l, r) => sum2(l, r))

// 这是一个比扫描后立即聚合(直接调用groupBy)更耗时的过程, 但得到的是相同的结果
flights.groupBy("DEST_COUNTRY_NAME").count().explain
```

