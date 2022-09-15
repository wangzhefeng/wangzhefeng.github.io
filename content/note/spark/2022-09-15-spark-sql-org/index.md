---
title: Spark SQL Org
author: 王哲峰
date: '2022-09-15'
slug: spark-sql-org
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

- [SparkSession](#sparksession)
  - [创建 SparkSession](#创建-sparksession)
  - [SparkSession 类及其方法](#sparksession-类及其方法)
  - [SparkSession in Spark 2.0 新特性](#sparksession-in-spark-20-新特性)
- [DataFrame](#dataframe)
  - [DataFrame创建](#dataframe创建)
    - [从一个 RDD 创建 DataFrame](#从一个-rdd-创建-dataframe)
    - [从 Hive Table 创建 DataFrame](#从-hive-table-创建-dataframe)
    - [从 Spark 数据源创建 DataFrame](#从-spark-数据源创建-dataframe)
  - [弱类型 Dataset 操作-DataFrame 操作](#弱类型-dataset-操作-dataframe-操作)
  - [程序化运行 SQL 查询](#程序化运行-sql-查询)
- [Spark SQL](#spark-sql)
  - [Spark SQL 创建表](#spark-sql-创建表)
  - [Spark SQL 创建外部表](#spark-sql-创建外部表)
  - [Spark SQL 插入表](#spark-sql-插入表)
  - [Spark SQL Describing 表 Matadata](#spark-sql-describing-表-matadata)
  - [Spark SQL Refreshing 表 Matadata](#spark-sql-refreshing-表-matadata)
  - [Spark SQL 删除表](#spark-sql-删除表)
- [Dataset](#dataset)
  - [Dataset 创建](#dataset-创建)
  - [Dataset 和 RDD](#dataset-和-rdd)
  - [Dataset](#dataset-1)
- [聚合(Aggregations)](#聚合aggregations)
  - [Untype 用户自定义的聚合函数](#untype-用户自定义的聚合函数)
  - [Type-Safe 用户自定义聚合函数](#type-safe-用户自定义聚合函数)
- [性能调优](#性能调优)
  - [将数据缓存到内存中](#将数据缓存到内存中)
- [分布式 SQL引擎](#分布式-sql引擎)
  - [Thrift JDBC/ODBC 服务](#thrift-jdbcodbc-服务)
  - [Spark SQL CLI](#spark-sql-cli)
</p></details><p></p>

# SparkSession

使用 DataFrame 和 Dataset API 进行 Spark 编程的程序入口 SparkSession

## 创建 SparkSession

(1)在命令行模式下,进入spark-shell后SparkSession会默认创建为`spark`

```bash
$ spark-shell
```

- Spark context Web UI: http://192.168.0.111:4040
- Spark context: `sc (master = local[*], app id = local-1556297140303)`
- Spark session: `spark`

(2)编程模式下需要使用SparkSession类手动创建

```scala
import org.apache.spark.sql.SparkSession

val spark = SparkSession
    .builder()
    .master("local")
    .appName("Spark SQL DataFrame and DataSet")
    .config("spark.some.config.option", "some-value")
    .getOrCreate()
```

## SparkSession 类及其方法

```
class SparkSession extends Serializable with Closeable with Logging
```

```scala
spark.version()
spark.newSession()
spark.sparkContext()
spark.catalog()
spark.conf()
spark.listenerManager()
spark.sessionState()
spark.sharedState()

spark.read()
spark.baseRelationToDataFrame()
spark.createDataFrame()
spark.createDataset()
spark.emptyDataFrame()
spark.emptyDataset()
spark.experimental()
spark.implicits()
spark.range()
spark.readStream()
spqrk.sql()
spark.sqlContext()
spark.streams()
spark.table()
spark.time()
spark.udf()
spark.stop()
spark.close()
```

## SparkSession in Spark 2.0 新特性

SparkSession in Spark 2.0 provides builtin support for Hive features
including the ability to write queries using HiveQL, access to Hive
UDFs, and the ability to read data from Hive tables. To use these
features, you do not need to have an existing Hive setup.

# DataFrame

## DataFrame创建

创建 DataFrame 的方式：

1. 从现有的RDD创建；
   - spark.SparkContext.makeRDD().map().toDF()
2. 从Hive表创建；
3. 从任何下面介绍的Spark数据源创建；

### 从一个 RDD 创建 DataFrame


### 从 Hive Table 创建 DataFrame


### 从 Spark 数据源创建 DataFrame

在最简单的模式下, 默认的数据源是 `parquet` 文件, 除非对默认的数据源进行配置

```
org.apache.spark.sql.source.default
```

- 一般的 `Load/Save` 函数
- `parquet` 文件
- `ORC` 文件
- `JSON` 文件
- `Hive Table`
- JDBC To Other Databases
- `Avro` 文件
- Troubleshooting


1.Load/Save函数
2.parquet文件

parquet 是一种柱状文件格式, 许多其他的数据处理系统都支持它。
Spark SQL 支持读取和写入 parquet 文件, 这些文件自动保留原始数据的模式(schema).
在编写 parquet 文件时, 为了保持兼容性, 所有的列都会自动转换为可为空(nullable).

(1)以编程方式加载数据：

```scala
import org.apache.spark.sql.SparkSession
import spark.implicits._

object parquetFileData {
    def main(args: Array[String]): Unit = {
        // build the SparkSession as spark
        val spark = SparkSession.
            .builder()
            .master("local")
            .appName("Parquet File Data Read/Write")
            .config("spark-config-option", "spark-config-value")
            .getOrReplace()
        
        // Loading data programmatically
        val peopleDF = spark.read.json("/home/wangzhefeng/project/bigdata/data/examples/src/main/resources/people.json")
        
        // Save the DataFrame as Parquet files, maintaing the schema infomation
        peopleDF.write.parquet("people.parquet")
        
        // Read the parquet file to a DataFrame
        val parquetFileDF = spark.read.parquet("/home/wangzhefeng/project/bigdata/data/spark_data/people.parquet")
        
        // parquet files used to create a temporary view and then used in SQL statements
        parquetFileDF.createOrReplaceTempView("parquetFile")
        val namesDF = spark.sql("SELECT name FROM parquetFile WHERE age BETWEEN 13 AND 19")
        namesDF.map(attributes => "Name:" + attributes(0)).show()
        
        // stop the SparkSession
        spark.stop()
    }
}
```

(2)数据分区发现(partition discover)

- 表分区是Hive等系统中常用的优化方法。在分区表中, 数据通常存储在不同的目录中, 
  分区列值在每个分区目录的路径中编码。所有内置文件源(包括Text/CSV/JSON/ORC/Parquet)都能够自动发现和推断分区信息。
  例如可以使用以下目录结构将所有以前使用的填充数据存储在分区表中, 
  并将两个额外的列(性别和国家/地区)作为分区列。

```
path
└── to
   └── table
       ├── gender=male
       │   ├── ...
       │   │
       │   ├── country=US
       │   │   └── data.parquet
       │   ├── country=CN
       │   │   └── data.parquet
       │   └── ...
       └── gender=female
           ├── ...
           │
           ├── country=US
           │   └── data.parquet
           ├── country=CN
           │   └── data.parquet
           └── ...
```

- 将 `path/to/table` 传给 `SparkSession.read.parquet` 或者 `SparkSession.read.load`, 
  Spark SQL 将自动从路径中检索分区信息：

```
root
    |-- name: string (nullable = true)
    |-- age: long (nullable = true)
    |-- gender: string (nullable = true)
    |-- country: string (nullable = true)
```

- 分区列的数据类型是自动推断的, 目前支持数字, 日期, 时间戳, 字符类型。
  如果不希望自动推断分区列的数据类型可以通过下面的配置设置, 
  禁止类型推断时, 字符串类型将用于分区列：

```
spark.sql.sources.partitionColumnTypeInference.enabled = false
```


- 从 Spark 1.6.0 开始, 分区发现默认只查找给定路径下的分区。
   - 对于上面的示例, 如果用户将 `path/to/table/gender=male` 传递给 `SparkSession.read.parquet` 或 `SparkSession.read.load`, 
     则不会将性别视为分区列
   - 如果用户需要指定分区发现应该开始的基本路径, 则可以在数据源选项中设置 `basePath`。
     例如, 当 `path/to/table/gender=male` 是数据的路径并且用户将 `basePath` 设置为 `path/to/table/` 时, `gender` 将是分区列

(3) 模式合并(schema merging)

- 与Protocol Buffer, Avro和Thrift一样, Parquet也支持模式演变。
  用户可以从简单模式开始, 并根据需要逐渐向模式添加更多列。
  通过这种方式, 用户可能最终得到具有不同但相互兼容的模式的多个Parquet文件。
  Parquet数据源现在能够自动检测这种情况并合并所有这些文件的模式
    - 由于模式合并是一项相对昂贵的操作, 并且在大多数情况下不是必需的, 
      因此默认从1.5.0开始关闭它。您可以启用它
    - 在读取Parquet文件时将数据源选项 `mergeSchema` 设置为 `true`(如下面的示例所示), 
      或将全局 SQL 选项 `spark.sql.parquet.mergeSchema` 设置为 `true`

```scala
import spark.implicits._
import org.apache.spark.sql.SparkSession

object parquetSchemaMerging {
    def main(args: Array[String]): Unit = {
        val spark = SparkSession
            .builder()
            .master("local")
            .appName("parquet schema merging")
            .config("spark-config-option", "spark-config-value")
            .getOrCreate()
        
        // squares DataFrame
        val squaresDF = spark.sparkContext
            .makeRDD(1 to 5)
            .map(i => (i, i * i))
            .toDF("value", "square")
        squaresDF.write.parquet("/home/wangzhefeng/project/bigdata/data/spark_data/partition_data/test_table/key=1")
        
        // cubes DataFrame
        val cubesDF = spark.sparkContext
            .makeRDD(6 to 10)
            .map(i => (i, i * i * i))
            .toDF("value", "cube")
        cubesDF.write.parquet("/home/wangzhefeng/project/bigdata/data/spark_data/partition_data/test_table/key=2")
        
        // Read the partitioned table
        val mergedDF = spark.read.option("mergeSchema", "true")
            .parquet("/home/wangzhefeng/project/bigdata/data/spark_data/partition_data/test_table")
        mergedDF.printSchema()
        mergedDF.show()
        
        // other operations
        mergedDF.createOrReplaceTempView("mergedDFs")
        val orderedMergedDF = spark.sql("SELECT * FROM mergedDFs ORDER BY value")
        mergedDF.show()
        
        spark.stop()
    }
}
```

输出结果：

```
root
  |-- value: integer (nullable = true)
  |-- square: integer (nullable = true)
  |-- cube: integer (nullable = true)
  |-- key: integer (nullable = true)

+-----+------+----+---+
|value|square|cube|key|
+-----+------+----+---+
|    4|    16|null|  1|
|    5|    25|null|  1|
|    9|  null| 729|  2|
|   10|  null|1000|  2|
|    1|     1|null|  1|
|    2|     4|null|  1|
|    3|     9|null|  1|
|    6|  null| 216|  2|
|    8|  null| 512|  2|
|    7|  null| 343|  2|
+-----+------+----+---+


+-----+------+----+---+
|value|square|cube|key|
+-----+------+----+---+
|    1|     1|null|  1|
|    2|     4|null|  1|
|    3|     9|null|  1|
|    4|    16|null|  1|
|    5|    25|null|  1|
|    6|  null| 216|  2|
|    7|  null| 343|  2|
|    8|  null| 512|  2|
|    9|  null| 729|  2|
|   10|  null|1000|  2|
+-----+------+----+---+
```

(4) Hive Metastore Parquet表转换

- 在读取和写入 Hive Metastore Parquet 表时, 
  Spark SQL 将尝试使用自己的 Parquet 支持而不是 Hive SerDe 来获得更好的性能。
  此行为由 `spark.sql.hive.convertMetastoreParquet` 配置控制, 默认情况下处于打开状态.
- Hive / Parquet Schema Reconciliation
    - 从表模式处理的角度来看, Hive 和 Parquet 之间存在两个主要区别:
        - Hive 区分大小写, 而 Parquet 则不区分大小写
        - Hive 认为所有列都可以为空, 而 Parquet 中的可空性很重要
    - 由于这个原因, 在将 Hive Metastore Parquet 表转换为 Spark SQL Parquet表时, 
      我们必须将 Hive Metastore 模式与 Parquet 模式进行协调。对帐规则是：
        - 两个模式中具有相同名称的字段必须具有相同的数据类型, 而不管是否为空。
           协调字段应具有 Parquet 端的数据类型, 以便遵循可为空性
        - 协调的模式恰好包含Hive Metastore模式中定义的那些字段
            - 仅出现在Parquet模式中的任何字段都将放入已协调的模式中
            - 仅出现在 Hive Metastore 模式中的任何字段都将在协调模式中添加为可空字段
- 元数据刷新
    - Spark SQL 缓存 Parquet 元数据以获得更好的性能。启用 Hive Metastore Parquet 表转换后, 
      还会缓存这些转换表的元数据。如果这些表由 Hive 或其他外部工具更新, 则需要手动刷新它们以确保元数据一致。

```scala
spark.catalog.refreshTable("my_table")
```

(5) 配置(Configuration)

- 其他一些 Parquet 生成系统, 特别是 Impala, Hive 和旧版本的 Spark SQL, 在写出 Parquet 模式时不区分二进制数据和字符串。
  此标志告诉 Spark SQL 将二进制数据解释为字符串, 以提供与这些系统的兼容性
- 一些 Parquet 生产系统, 特别是 Impala 和 Hive, 将时间戳存储到 INT96 中。
  此标志告诉 Spark SQL 将 INT96 数据解释为时间戳, 以提供与这些系统的兼容性
- 设置编写 Parquet 文件时使用的压缩编解码器。如果在特定于表的选项/属性中指定了“compression”或“parquet.compression”, 
  则优先级为“compression”, “parquet.compression”, “spark.sql.parquet.compression.codec”。
  可接受的值包括：none, uncompressed, snappy, gzip, lzo, brotli, lz4, zstd。请注意, 
  `zstd` 需要在 Hadoop 2.9.0 之前安装 `ZStandardCodec`, `brotli` 需要安装 `BrotliCodec`
- 设置为 true 时启用 Parquet 过滤器下推优化
- 设置为 false 时, Spark SQL 将使用 Hive SerDe 作为 parquet 而不是内置支持
- 如果为 true, 则 Parquet 数据源合并从所有数据文件收集的模式, 
  否则, 如果没有可用的摘要文件, 则从摘要文件或随机数据文件中选取模式
- 如果为 true, 则数据将以 Spark 1.4 及更早版本的方式写入。
  例如, 十进制值将以 Apache Parquet 的固定长度字节数组格式写入, 
  其他系统(如Apache Hive 和 ApacheImpala)也使用该格式。 
  如果为 false, 将使用 Parquet 中的较新格式。例如, 小数将以基于 int 的格式写入。
  如果 Parquet 输出旨在用于不支持此较新格式的系统, 请设置为 true。

```
spark.setConf("spark.sql.parquet.OPTIONS", "true")

SET spark.sql.parquet.binaryAsString=true
```

| 配置项                                  | 默认值   | 配置项解释                                                          |
|----------------------------------------|---------|--------------------------------------------------------------------|
| spark.sql.parquet.binaryAsString       | false   | Spark SQL将二进制数据解释为字符串 |
| spark.sql.parquet.int96AsTimestamp     | true    | Spark SQL将INT96数据解释为时间戳 |
| spark.sql.parquet.com pression.codec   | snappy  | 设置为true时启用Parquet过滤器下推优化 |
| spark.sql.parquet.filterPushdown        | true    | 设置为true时启用Parquet过滤器下推优化 |
| spark.sql.hive.convertMetastoreParquet | true    | 设置为 false 时, Spark SQL 将使用 HiveSerDe 作为 parquet 而不是内置支持 |
| spark.sql.parquet.mergeSchema          | false   | 如果为 true, 则Parquet数据源合并从所有数据文件收集的模式 |
| spark.sql.parquet.writeLegacyFormat    | false   | 如果为 true, 则数据将以 Spark 1.4 及更早版本的方式写入 |

4.JSON文件

Spark SQL 可以自动推断 JSON 数据集的模式(Schema), 并将其加载为 `Dataset[Row]`, 转换方式:

- SparkSession.read.json()
- Dataset[String]
- JSON 文件

```scala
import spark.implicits._

object readJSON {
    val spark = SparkSession
        .builter()
        .appName("JSON")
        .master("local")
        .config("config-option", "config-value")
        .getOrCreate()
    
    // json file or a directory storing text files
    val path = "/home/wangzhefeng/bigdata/data/examples/src/main/resoureces/people.json"
    val peopleDF = spark.read.json(path)
    peopleDF.printSchema()
    peopleDF.createOrReplaceTempView("people")
    val teenagerNamesDF = spark.sql("SELECT name FROM people WHERE age BETWEEN 13 AND 19")
    teenagerNamesDF.show()  
}
```

```scala
import spark.implicits._

object readJSON {
   val spark = SparkSession
       .builter()
       .appName("JSON")
       .master("local")
       .config("config-option", "config-value")
       .getOrCreate()
   
   // 
   val otherPeopleDataset = spark.createDataset(
       """{"name": "Yin", "address": {"city": "Columnbus", "state": "Ohio"}}"""::Nil
   )
   val otherPeople = spark.read.json(otherPeopleDataset)
   otherPeople.show()   
}
```

## 弱类型 Dataset 操作-DataFrame 操作

```scala
import spark.implicits._
import org.apache.spark.sql.SparkSession

object DataFrameOperation {
   def main(args: Array[String]): Unit = {
       val spark = SparkSession
           .builder()
           .master("local")
           .appName("SQL Query Programmatically")
           .config("config-option", "config-value")
           .getOrCreate()
       // Create a DataFrame
       df = spark.read.json("/home/wangzhefeng/project/bigdata/data/examples/src/main/resources/people.json")
       
       df.apply()
       df.col("salary")
       df.colRegex("^s")
       df.select("name").show()
       df.select($"name").show()
       df.select($"name", $"age" + 1).show()
       df.selectExpr("name", "age as age2", "abs(salary)")
       df.select(expr("name"), expr("age as age2"), expr("abs(salary)"))
       df.filter($"age" > 21).show()
       // cube
       df.cube("department", "group").avg()
       df.cube($"department", $"gender").agg(Map(
         "salary" -> "avg",
         "age" -> "max"
       ))
       // drop
       df.drop("age")
       // groupBy, agg
       df.agg(max($"age"), avg($"salary"))
       df.groupby().agg(max($"age"), avg($"salary"))
       df.agg(Map("age" -> "max", "salary" -> "avg"))
       df.groupBy().agg(Map("age" -> "max", "salary" -> "avg"))
       df.groupBy("age").count().show()
       df.groupBy($"department", $"gender").agg(Map("salary" -> "avg", "age" -> "max"))
       // join
       import org.apache.spark.sql.functions._
       df1.join(right = df2, joinExprs, joinType = "full")
       df1.join(df2, joinExprs, joinType = "outer")
       df1.join(df2, joinExprs, joinType = "full_outer")
       df1.join(df2, joinExprs, joinType = "inner")
       df1.join(df2, joinExprs)
       df1.join(df2).where()
       df1.join(df2, joinExprs, joinType = "left")
       df1.join(df2, joinExprs, joinType = "left_out")
       df1.join(df2, joinExprs, joinType = "right")
       df1.join(df2, joinExprs, joinType = "right_out")
       df1.join(df2, joinExprs, joinType = "left_semi")
       df1.join(df2, joinExprs, joinType = "left_anti")
       df1.join(df2, joinExprs, joinType = "corss")
       df1.join(df2, usingColumns = "", join_type = "")
       df1.join(df2, usingColumn = "")
       df1.join(df2)
       df.crossJoin(df2)
       // na
       df.na.drop()
       // rollup
       df.rollup("department", "group").avg()
       df.rollup($"department", $"group").agg(Map(
         "salary" -> "avg",
         "age" -> "max"
       ))
       // DataFrame 统计函数
       df1.stat.freqItems(Seq("a"))
       // withColumn
       df1.withColumn(colName = "", col = "")
       df1.withColumnRenamed(existingName = "", newName = "")
       
       spark.stop()
   }
}
```

## 程序化运行 SQL 查询

```scala
import spark.implicits._
import org.apache.spark.sql.SparkSesssion

object SQLQueryProgrammatically {
   def main(args: Array[String]): Unit = {
       val spark = SparkSession
           .builder()
           .master("local")
           .appName("SQL Query Programmatically")
           .config("config-option", "config-value")
           .getOrCreate()
       
       // Create a DataFrame
       df = spark.read.json("/home/wangzhefeng/project/bigdata/data/examples/src/main/resources/people.json")
       
       // Register the DataFrame as a SQL temporary view
       df.createOrReplaceTempView("people")
       val sqlDF_1 = spark.sql("SELECT * FROM people")
       sqlDF_1.show()    
       
       // Register the DataFrame as a global temporary view
       df.createGlobalTempView("people_Global")
       val sqlDF_2 = spark.sql("SELECT * FROM people_Global")
       sqlDF_2.show()
       
       spark.stop()
   }
}
```

# Spark SQL 

## Spark SQL 创建表

## Spark SQL 创建外部表

## Spark SQL 插入表

## Spark SQL Describing 表 Matadata

## Spark SQL Refreshing 表 Matadata

## Spark SQL 删除表


# Dataset

## Dataset 创建

- Dataset 与 RDD 类似, 但是 Dataset 不使用 Java 序列化或 kryo, 
  而是使用一种特殊的 Encoder 来序列化对象以便网络进行处理或传输. 
  虽然 Encoder 和标准序列化都是将对象转换为字节, 但是 Encoder 是动态生成的代码, 
  并使用一种格式允许 Spark 执行许多操作, 如 filtering, sorting, hashing 而无需将字节反序列化为对象;

```scala
// 使用scala样例类创建Dataset
case class Person(name: String, age: Long)
val caseClassDS = Seq(Person("Andy", 32)).toDS()
caseClassDS.show()
```

```scala
// 使用一般的scala数据结构创建Dataset
val primitiveDS = Seq(1, 2, 3).toDS()
primitiveDS.map(_ + 1).collect()
primitiveDS.show()
```

```scala
// 通过将DataFrame转换为Dataset
val path = "/home/wangzhefeng/project/bigdata/data/examples/src/main/resources/people.json"
val peopleDS = spark.read.json(path).as[Person]
peopleDS.show()
```

## Dataset 和 RDD

- 将现有的RDD转换为Dataset
    - 使用反射来推断包含特定类型对象的RDD模式
    - 通过编程接口,构建模式,将模式应用于现有的RDD

```scala
import spark.implicits._

case class Person(name: String, age: Long)

// 从一个文本文件创建一个名为Person对象的RDD,将RDD转换为DataFrame
val peopleDF = spark.sparkContext
   .textFile("/home/wangzhefeng/bigdata/data/examples/src/main/resources/people.txt")
   .map(_.split(","))
   .map(attributes => Person(attributes(0), attributes(1).trim.toInt))
   .toDF()

peopleDF.createOrReplaceTempView("people")
val teenagersDF = saprk.sql("SELECT name, age FROM people WHERE age BETWEEN 13 AND 19")
teenagersDF.map(teenager => "Name: " + teenager(0)).show()
teenagersDF.map(teenager => "Name: " + teenager.getAs[String]("name")).show()
implicit val mapEncoder = org.apache.spark.sql.Encoders.kryo[Map[String, Any]]
teenagerDF.map(teenager => teenager.getValuesMap[Any](List("name", "age"))).collect()
```

## Dataset

# 聚合(Aggregations)

DataFrame内置聚合函数

- approx\ *count*\ distinct()
- avg()
- collect_list()
- collect_set()
- corr()
- count()
- countDistinct()
- covar_pop()
- covar_samp()
- first()
- grouping()
- grouping_id()
- kurtosis()
- last()
- max()
- mean()
- min()
- skewness()
- stddev()
- stddev_pop()
- stddev_samp()
- sum()
- sumDistinct()
- var_pop()
- var_samp()
- variance()

## Untype 用户自定义的聚合函数

用户必须 `extends UserDefinedAggregateFunction` 抽象类来实现一个自定义的 untype 聚合函数;

```scala
import org.apache.spark.sql.{Row, SparkSession}
import org.apache.spark.sql.expressions.MutableAggregationBuffer
import org.apache.spark.sql.expressions.UserDefinedAggregateFunction
import org.apache.spark.sql.types._

object MyAverage extends UserDefinedAggregateFunction {
   def inputSchema: StructType = StructType(StructField("inputColumn", LongType) :: Nil)
   
}
```

## Type-Safe 用户自定义聚合函数

```scala
import org.apache.spark.sql.{Encoder, Encoders, SparkSession}
import org.apache.spark.sql.expression.Aggregator

case class Employee(name: String, salary: Long)
case class Average(var sum: Long, var count: Long)

object MyAverage extends Aggregator[Employee, Average, Double] {
    def zero: Average = Average(0L, 0L)
    
    def reduce(buffer: Average, employee: Employee): Average = {
        buffer.sum += employee.salary
        buffer.count += 1
        buffer
    }
    
    def merge(b1: Average, b2: Average): Average = {
        b1.sum += b2.sum
        b1.count += b2.count
        b1
    }
    
    def finish(reduction: Average): Double = reduction.sum.toDouble / reduction.count
    def bufferEncoder: Encoder[Average] = Encoders.product
    def outputEncoder: Encoder[Double] = Encoders.scalaDouble
}


object Computer {
    def main(args: Arrar[String]) = {
        val spark = SparkSession
            .builder()
            .appName("User defined agg")
            .master("local")
            .config("config-option", "config-value")
            .getOrCreate()
        
        val ds = spark.read.json("/home/wangzhefeng/project/bigdata/data/examples/src/main/resources/empolyees.json").as[Employee]
        ds.show()
        val averageSalary = MyAverage.toColumn.name("average_salary")
        val result = ds.select(averageSalary)
        result.show()
    }
}
```

# 性能调优

## 将数据缓存到内存中

缓存数据:

```scala
spark.catalog.cacheTable("tableName")
dataFrame.cache()
```

删除内存中的缓存数据:

```scala
spark.catalog.uncacheTable("tableName")
```

# 分布式 SQL引擎

Spark SQL 能够作为一个分布式查询引擎, 在这种模式下, 
用户或者客户端可以通过直接运行 SQL 查询语句与 Spark SQL 进行交互, 
而不需要写任何代码. 有两种方式运行这种模式:

- JDBC/ODBC
- CLI(command-line interface)

## Thrift JDBC/ODBC 服务

启动 JDBC/ODBC 服务:

```bash
./sbin/start-thriftserver.sh
```

## Spark SQL CLI

- Spark SQL CLI 是一个能够在本地模式以并从命令行输入查询时运行 Hive metastore 服务方便的工具；
   - 需要配置好 Hive:
      - hive-site.xml
      - core-site.xml
      - hdfs-site.xml
   - 注意：Spark SQL CLI 不能与 Thrift JDBC 服务器进行通信；

```bash
# 启动Spark SQL CLI
./bin/spark-sql

# 查看spark-sql可用的完整列表
./bin/spark-sql --help
```

