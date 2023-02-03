---
title: Spark Data Sources
author: 王哲峰
date: '2022-09-06'
slug: spark-data-source
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

- [Spark 数据源 API](#spark-数据源-api)
  - [Read API Structure](#read-api-structure)
  - [Write API Structure](#write-api-structure)
- [Spark 读取 CSV 文件](#spark-读取-csv-文件)
  - [CSV Read/Write Options](#csv-readwrite-options)
  - [Spark Reading CSV 文件](#spark-reading-csv-文件)
  - [Spark Writing CSV 文件](#spark-writing-csv-文件)
- [Spark 读取 JSON 文件](#spark-读取-json-文件)
  - [JSON Read/Write Options](#json-readwrite-options)
  - [Spark Reading JSON 文件](#spark-reading-json-文件)
  - [Spark Writing JSON 文件](#spark-writing-json-文件)
- [Spark 读取 Parquet 文件](#spark-读取-parquet-文件)
  - [Parquet Read/Write Options](#parquet-readwrite-options)
  - [Spark Reading Parquet 文件](#spark-reading-parquet-文件)
  - [Spark Writing Parquet 文件](#spark-writing-parquet-文件)
- [Spark 读取 ORC 文件](#spark-读取-orc-文件)
  - [Spark Reading ORC 文件](#spark-reading-orc-文件)
  - [Spark Writing ORC 文件](#spark-writing-orc-文件)
- [Spark 读取 SQL Database](#spark-读取-sql-database)
  - [SQLite](#sqlite)
  - [JDBC 数据源 Options](#jdbc-数据源-options)
  - [Spark Reading From SQL Database](#spark-reading-from-sql-database)
  - [查询下推](#查询下推)
    - [并行读取数据库](#并行读取数据库)
    - [基于滑动窗口的分区](#基于滑动窗口的分区)
  - [Spark Writing To SQL Database](#spark-writing-to-sql-database)
- [Spark 读取 Text 文件](#spark-读取-text-文件)
  - [Spark Reading Text 文件](#spark-reading-text-文件)
  - [Spark Writing Text 文件](#spark-writing-text-文件)
- [高级 I/O](#高级-io)
  - [可分割的文件类型和压缩](#可分割的文件类型和压缩)
  - [并行读数据](#并行读数据)
  - [并行写数据](#并行写数据)
    - [数据划分](#数据划分)
    - [数据分桶](#数据分桶)
  - [写入复杂类型](#写入复杂类型)
  - [管理文件大小](#管理文件大小)
  - [Cassandra Connector](#cassandra-connector)
</p></details><p></p>

Spark 核心数据源:

- CSV
- JSON
- Parquet
- ORC
- JDBC/ODBC connection
- Plain-text file

其他数据源(Spark Community):

- Cassandra
- HBase
- MongoDB
- AWS Redshift
- XML
- others...

# Spark 数据源 API

## Read API Structure

**(1) 核心结构:**

Spark 读取数据的核心结构如下:

``` 
DataFrameReader.format().option("key", "value").schema().load()
```

其中:

- `format`: 是可选的,因为 Spark 默认读取 Parquet 格式的文件;
- `option`: 可以设置很多自定义配置参数,不同的数据源格式都有自己的一些可选配置参数可以手动设置;
    - `mode`: ;
    - `inferSchema`: 是否启用 schema 推断;
    - `path`: ;
- `schema`: 是可选的,有些数据文件提供了schema,可以使用模式推断(schema inference) 进行自动推断;也可以设置自定义的 schema 配置;

**(2) Spark 读取数据的基本操作:**

Spark 读取数据的操作的示例:

```scala
// in Scala

spark.read.format("parquet")
   .option("mode", "FAILFAST")
   .option("inferSchema", "true")
   .option("path", "path/to/file(s)")
   .schema(someSchema)
   .load()
```

其中:

- `spark.read`: DataFrameReader 是 Spark 读取数据的基本接口,可以通过 `SparkSession` 的 `read` 属性获取;
- `format`: 
- `schema`: 
- `option`: 选项的设置可以通过创建一个配置项的映射结构来设置;
- read modes: 从外部源读取数据很容易会遇到错误格式的数据, 尤其是在处理半结构化数据时. 读取模式指定当 Spark 遇到错误格式的记录时应采取什么操作;
   - `option("mode", "permissive")`
      - 默认选项, 当遇到错误格式的记录时, 将所有字段设置为 `null` 并将所有错误格式的记录放在名为 `_corrupt_record` 字符串中;
   - `option("mode", "dropMalformed")`
      - 删除包含错误格式记录的行;
   - `option("mode", "failFast")`
      - 遇到错误格式的记录后立即返回失败;


***
**Note:**

* `format`, `option` 和 `schema` 都会返回一个 `DataFrameReader`,它可以进行进一步的转换,并且都是可选的.每个数据源都有一组特定的选项,用于设置如何将数据读入 Spark.
***



## Write API Structure

**(1) 核心结构:**

Spark 写数据的核心结构如下:

```
DataFrameWriter.format().option().partitionBy().bucketBy().sortBy().save()
```

其中:

- `format`: 是可选的,因为 Spark 默认会将数据保存为 Parquet 文件格式;
- `option`: 可以设置很多自定义配置参数,不同的数据源格式都有自己的一些可选配置参数可以手动设置;
- `partitionBy`, `bucketBy`, `sortBy`: 只对文件格式的数据起作用,可以通过设置这些配置对文件在目标位置存放数据的结构进行配置;

**(2) Spark 读取数据的基本操作:**

Spark 写数据的操作的示例:

```scala
// in Scala

dataframe.write.format("parquet")
   .option("mode", "OVERWRITE")
   .option("dataFormat", "yyyy-MM-dd")
   .option("path", "path/to/file(s)")
   .save()
```

其中:

- `dataframe.write`: DataFrameWriter 是 Spark 写出数据的基本接口,可以通过 `DataFrame` 的 `write` 属性来获取;
- `format`: ;
- `option` 选项的设置还可以通过创建一个配置项的映射结构来设置;
- write modes:
  - `option("mode", "errorIfExists")`
     - 默认选项, 如果目标路径已经存在数据或文件,则抛出错误并返回写入操作失败;
  - `option("mode", "append")`
     - 将输出文件追加到目标路径已经存在的文件上或目录的文件列表;
  - `option("mode", "overwrite")`
     - 将完全覆盖目标路径中已经存在的任何数据;
  - `option("mode", "ignore")`
     - 如果目标路径已经存在数据或文件,则不执行任何操作;

# Spark 读取 CSV 文件

CSV (comma-separated values), 是一种常见的文本文件格式, 其中每行表示一条记录, 用逗号分隔记录中的每个字段. 
虽然 CSV 文件看起来结构良好, 实际上它存在各种各样的问题, 是最难处理的文件格式之一,
这是因为实际应用场景中遇到的数据内容或数据结构并不会那么规范. 因此, CSV 读取程序包含大量选项, 
通过这些选项可以帮助你解决像解决忽略特定字符等的这种问题,比如当一列的内容也以逗号分隔时, 
需要识别出该逗号是列中的内容,还是列间分隔符.

## CSV Read/Write Options

+------------+------------------------------+-----------------------------+----------------------------------+------------------------------------------------------------------------------------------------------------------------------------+
| Read/Write | Key                          | Potential values            | Default                          | Description                                                                                                                        |
+============+==============================+=============================+==================================+====================================================================================================================================+
| Read/Write | `sep`                      | Any single string character | `,`                            | This single character that is used as separator for each field and value.                                                          |
+------------+------------------------------+-----------------------------+----------------------------------+------------------------------------------------------------------------------------------------------------------------------------+
| Read/Write | `header`                   | `true/false`              | `false`                        | A Boolean flag that declares whether the first line in the file(s) are the names of the columns.                                   |
+------------+------------------------------+-----------------------------+----------------------------------+------------------------------------------------------------------------------------------------------------------------------------+
| Read       | `escape`                   | Any string character        | `\`                            | The character Spark should use to escape other characters in the file.                                                             |
+------------+------------------------------+-----------------------------+----------------------------------+------------------------------------------------------------------------------------------------------------------------------------+
| Read       | `inferSchema`              | `true/false`              | `false`                        | Specifies whether Spark should infer column types when reading the file.                                                           |
+------------+------------------------------+-----------------------------+----------------------------------+------------------------------------------------------------------------------------------------------------------------------------+
| Read       | `ignoreLeadingWhiteSpace`  | `true/false`              | `false`                        | Declares whether leading spaces from value being read should be skipped.                                                           |
+------------+------------------------------+-----------------------------+----------------------------------+------------------------------------------------------------------------------------------------------------------------------------+
| Read       | `ignoreTrailingWhiteSpace` | `true/false`              | `false`                        | Declares whether trailing spaces from value being read should be skipped.                                                          |
+------------+------------------------------+-----------------------------+----------------------------------+------------------------------------------------------------------------------------------------------------------------------------+
| Read/Write | `nullValue`                | Any string character        | `""`                           | Declares what character represents a `null` value in the file.                                                                   |
+------------+------------------------------+-----------------------------+----------------------------------+------------------------------------------------------------------------------------------------------------------------------------+
| Read/Write | `nanValue`                 | Any string character        | `NaN`                          | Declares what character represents a `NaN`  or missing character in the CSV file.                                                |
+------------+------------------------------+-----------------------------+----------------------------------+------------------------------------------------------------------------------------------------------------------------------------+
| Read/Write | `positiveInf`              | Any string or character     | `Inf`                          | Declares what character(s) represent a positive infinite value.                                                                    |
+------------+------------------------------+-----------------------------+----------------------------------+------------------------------------------------------------------------------------------------------------------------------------+
| Read/Write | `negativeInf`              | Any String or character     | `-Inf`                         | Declares what character(s) represent a positive negative infinite value.                                                           |
+------------+------------------------------+-----------------------------+----------------------------------+------------------------------------------------------------------------------------------------------------------------------------+
| Read/Write | `compression/codec`        | None, uncom pressed, bzip2, | `none`                         | Declares  what compression codec Spark should use to read or write the file.                                                       |
|            |                              | deflate,gzip,lz4, or snappy |                                  |                                                                                                                                    |
+------------+------------------------------+-----------------------------+----------------------------------+------------------------------------------------------------------------------------------------------------------------------------+
| Read/Write | `dateFormat`               | Any string or character that| `yyyy-MM-dd`                   | Declares the date format for any columns that are date type.                                                                       |
|            |                              | conform to java's           |                                  |                                                                                                                                    |
|            |                              | SimpleDataFormat            |                                  |                                                                                                                                    |
+------------+------------------------------+-----------------------------+----------------------------------+------------------------------------------------------------------------------------------------------------------------------------+
| Read/Write | `timestampFormat`          | Any string or character that| `yyyy-MM-dd'T' HH:mm:ss.SSSZZ` | Declares the timestamp format for any columns that are timestamp type                                                              |
|            |                              | conform to java's           |                                  |                                                                                                                                    |
|            |                              | SimpleDataFormat            |                                  |                                                                                                                                    |
+------------+------------------------------+-----------------------------+----------------------------------+------------------------------------------------------------------------------------------------------------------------------------+
| Read       | `maxColumns`               | Any integer                 | `20480`                        | Declares the maximum number for columns in the file.                                                                               |
+------------+------------------------------+-----------------------------+----------------------------------+------------------------------------------------------------------------------------------------------------------------------------+
| Read       | `maxCharsPerColumn`        | Any integer                 | `1000000`                      | Declares the maximum number of character in a column.                                                                              |
+------------+------------------------------+-----------------------------+----------------------------------+------------------------------------------------------------------------------------------------------------------------------------+
| Read       | `escapeQuotes`             | `true/false`              | `true`                         | Declares whether Spark should escape quotes that are found in lines.                                                               |
+------------+------------------------------+-----------------------------+----------------------------------+------------------------------------------------------------------------------------------------------------------------------------+
| Read       |`maxMalformeLogPerPartition`| Any integer                 | `10`                           | Sets the maximum number of malformed rows Spark will log for each partition. Malformed records beyond this number will be ignore.  |
+------------+------------------------------+-----------------------------+----------------------------------+------------------------------------------------------------------------------------------------------------------------------------+
| Read       | `multiLine`                | `true/false`              | `false`                        | This option allows you read multiline CSV files where each logical row in the CSV file might span multipe rows in the file itself. |
+------------+------------------------------+-----------------------------+----------------------------------+------------------------------------------------------------------------------------------------------------------------------------+
| Write      | `quoteAll`                 | `true/false`              | `false`                        | Specifies whether all values should be enclosed in quotes, as opposed to just escaping values that have a quote character.         |
+------------+------------------------------+-----------------------------+----------------------------------+------------------------------------------------------------------------------------------------------------------------------------+



## Spark Reading CSV 文件

示例 1:

```scala  
// in Scala

val csvFile = spark.read.format("csv")
   .option("header", "true")
   .option("mode", "FAILFAST")
   .option("inferSchema", "true")
   .load("some/path/to/file.csv")
   .show()
```

示例 2:

```python
# in Python

csvFile = spark.read.format("csv") \
   .option("header", "true") \
   .option("mode", "FAILFAST") \
   .option("inferSchema", "true") \
   .load("/data/flight-data/csv/2010-summary.csv")
   .show()
```


示例 3:

```scala  
// in Scala

import org.apache.spark.sql.types.{StructType, StructField, StringType, LongType}

val myManualSchema = new StructType(Array(
   new StructField("DEST_COUNTRY_NAME", StringType, true),
   new StructField("ORIGIN_COUNTRY_NAME", StringType, true),
   new StructField("count", LongType, false)
))

val csvFile = spark.read.format("csv")
   .option("header", "true")
   .option("mode", "FAILFAST")
   .schema(myManualSchema)
   .load("/data/flight-data/csv/2010-summary.csv")
   .show()
```

***
**Note:**

* 通常, Spark 只会在作业执行而不是 DataFrame 定义时发生失败.
***
   

## Spark Writing CSV 文件

示例 1:

```scala
// in Scala

csvFile.write.format("csv")
   // .mode("overwrite")
   .option("mode", "overwrite")
   .option("sep", "\t")
   .save("/tmp/my-tsv-file.tsv")
```

示例 2:

```python
# in Python

csvFile.write.format("csv") \
   # .mode("overwrite") \
   .option("mode", "overwrite") \
   .option("sep", "\t") \
   .save("/tmp/my-tsv-file.tsv")
```


# Spark 读取 JSON 文件

JSON (JavaScript Object Notation). 在 Spark 中, 提及的 `JSON 文件` 指 `换行符分隔的 JSON`, 
每行必须包含一个单独的、独立的有效 JSON 对象, 这与包含大的 JSON 对象或数组的文件是有区别的.

换行符分隔 JSON 对象还是一个对象可以跨越多行, 这个可以由 `multiLine` 选项控制, 当 `multiLine` 为 `true` 时, 
则可以将整个文件作为一个 JSON 对象读取, 并且 Spark 将其解析为 DataFrame. 换行符分隔的 JSON 实际上是一种更稳定的格式, 
因为它可以在文件末尾追加新纪录(而不是必须读入整个文件然后再写出).

换行符分隔的 JSON 格式流行的另一个关键原因是 JSON 对象具有结构化信息, 并且(基于 JSON的) JavaScript 也支持基本类型, 这使得它更易用, Spark 可以替我们完成很多对结构化数据的操作. 
由于 JSON 结构化对象封装的原因, 导致 JSON 文件选项比 CSV 的要少很多.

## JSON Read/Write Options

+------------+-------------------------------+-----------------------------+-----------------------------------------------------+-----------------------------------------------------------------------------------------+
| Read/Write | Key                           | Potential values            | Default                                             | Description                                                                             |
+============+===============================+=============================+=====================================================+=========================================================================================+
| Read/Write | `compression/codec`         | None, uncom pressed, bzip2, | `none`                                            | Declares  what compression codec Spark should use to read or write the file.            |
|            |                               | deflate,gzip,lz4, or snappy |                                                     |                                                                                         |
+------------+-------------------------------+-----------------------------+-----------------------------------------------------+-----------------------------------------------------------------------------------------+
| Read/Write | `dateFormat`                | Any string or character that| `yyyy-MM-dd`                                      | Declares the date format for any columns that are date type.                            |
|            |                               | conform to Java's           |                                                     |                                                                                         |
|            |                               | SimpleDataFormat            |                                                     |                                                                                         |
+------------+-------------------------------+-----------------------------+-----------------------------------------------------+-----------------------------------------------------------------------------------------+
| Read/Write | `timestampFormat`           | Any string or character that| `yyyy-MM-dd'T' HH:mm:ss.SSSZZ`                    | Declares the timestamp format for any columns that are timestamp type                   |
|            |                               | conform to java's           |                                                     |                                                                                         |
|            |                               | SimpleDataFormat            |                                                     |                                                                                         |
+------------+-------------------------------+-----------------------------+-----------------------------------------------------+-----------------------------------------------------------------------------------------+
| Read       | `primitiveAsString`         | `true/false`              | `false`                                           | Infers all primitive values as string type.                                             |
+------------+-------------------------------+-----------------------------+-----------------------------------------------------+-----------------------------------------------------------------------------------------+
| Read       | `allowComments`             | `true/false`              | `false`                                           | Ignores Java/C++ style comment in JSON  records.                                        |
+------------+-------------------------------+-----------------------------+-----------------------------------------------------+-----------------------------------------------------------------------------------------+
| Read       | `allowUnquotedFieldNames`   | `true/false`              | `false`                                           | Allows unquotes JSON field names.                                                       |
+------------+-------------------------------+-----------------------------+-----------------------------------------------------+-----------------------------------------------------------------------------------------+
| Read       | `allowSingleQuotes`         | `true/false`              | `true`                                            | Allows single quotes in addition to double quotes.                                      |
+------------+-------------------------------+-----------------------------+-----------------------------------------------------+-----------------------------------------------------------------------------------------+
| Read       | `allNumericLeadingZeros`    | `true/false`              | `false`                                           | Allows leading zeroes in number (e.g., 00012).                                          |
+------------+-------------------------------+-----------------------------+-----------------------------------------------------+-----------------------------------------------------------------------------------------+
| Read       | `allowBackslashEscAPIngAny` | `true/false`              | `false`                                           | Allows accepting quoting of all characters using backlash quoting mechanism.            |
+------------+-------------------------------+-----------------------------+-----------------------------------------------------+-----------------------------------------------------------------------------------------+
| Read       | `columnNameOfCorruptRecord` | Any string                  | Value of `spark.sql.column & NameOfCorruptRecord` | Allows renaming the new field having a malformed string created by `permissive` mode. |
|            |                               |                             |                                                     | This will override the configuration value.                                             |
+------------+-------------------------------+-----------------------------+-----------------------------------------------------+-----------------------------------------------------------------------------------------+
| Read       | `multiLine`                 | `true/false`              | `false`                                           | Allows for reading in non-line-delimited JSON files.                                    |
+------------+-------------------------------+-----------------------------+-----------------------------------------------------+-----------------------------------------------------------------------------------------+

## Spark Reading JSON 文件

示例 1:

```scala
// in Scala

import org.apache.spark.sql.types.{StructType, StructField, StringType, LongType}

val myManualSchema = new StructType(Array(
   new StructField("DEST_COUNTRY_NAME", StringType, true),
   new StructField("ORIGIN_COUNTRY_NAME", StringType, true),
   new StructField("count", LongType, false)
))

spark.read.format("json")
   .option("mode", "FAILFAST")
   .schema("myManualSchema")
   .load("/data/flight-data/json/2010-summary.json")
   .show(5)
```

示例 2:

```python
# in Python

spark.read.format("json") \
   .option("mode", "FAILFAST") \
   .option("inferSchema", "true") \
   .load("/data/flight-data/json/2010-summary.json") \
   .show(5)
```

## Spark Writing JSON 文件

示例 1:

```scala
// in Scala

csvFile.write.format("json")
   .mode("overwrite")
   .save("/tmp/my-json-file.json")
```

示例 2:

```python
# in Python

csvFile.write.format("json") \
   .mode("overwrite") \
   .save("/tmp/my-json-file.json")
```

***
**Note:**


* 每个数据分片作为一个文件写出, 而整个 DataFrame 将输出到一个文件夹. 文件中每行仍然代表一个 JSON 对象:

```bash
$ ls /tmp/my-json-file.json/
/tmp/my-json-file.json/part-0000-tid-543....json
```
***

   


# Spark 读取 Parquet 文件

Parquet 是一种开源的面向列的数据存储格式,它提供了各种存储优化,尤其适合数据分析.
Parquet 提供列压缩从而可以节省空间,而且它支持按列读取而非整个文件地读取.

作为一种文件格式,Parquet 与 Apache Spark 配合得很好,而且实际上也是 Spark 的默认文件格式.
建议将数据写到 Parquet 以便长期存储,因为从 Parquet 文件读取始终比从 JSON 文件或 CSV 文件效率更高.

Parquet 的另一个优点是它支持复杂类型,也就是说如果一个数组(CSV 文件无法存储组列)、map 映射或 struct 结构体,
仍可以正常读取和写入,不会出现任何问题.


## Parquet Read/Write Options

由于 Parquet 含有明确定义且与 Spark 概念密切一致的规范,所以它只有很少的可选项,实际上只有两个.
Parqute 的可选项很少,因为它在存储数据时执行本身的 schema.

虽然只有两个选项,但是如果使用的是不兼容的 Parquet 文件,仍然会遇到问题. 
并且,当使用不同版本的 Spark 写入 Parquet 文件时要小心,因为这可能会导致让人头疼的问题.

+------------+------------------------------+-----------------------------+-----------------------------------+------------------------------------------------------------------------------+
| Read/Write | Key                          | Potential values            | Default                           | Description                                                                  |
+============+==============================+=============================+===================================+==============================================================================+
| Write      | `compression` or `codec` | None, uncom pressed, bzip2, | `none`                          | Declares  what compression codec Spark should use to read or write the file. |
|            |                              | deflate,gzip,lz4, or snappy |                                   |                                                                              |
+------------+------------------------------+-----------------------------+-----------------------------------+------------------------------------------------------------------------------+
| Read       | `merge Schema`             | `true/false`              | Value of the configuration        | You can incrementally add columns to newly written Parquet files             |
|            |                              |                             | `spark.sql.parquet.mergeSchema` | in the same table/folder. Use this option to enable or disable this feature. |
+------------+------------------------------+-----------------------------+-----------------------------------+------------------------------------------------------------------------------+

## Spark Reading Parquet 文件

示例 1:

```scala
// in Scala

spark.read.format("parquet")
   .load("/data/flight-data/parquet/2010-summary.parquet")
   .show(5)
```

示例 2:

```python
# in Python

spark.read.format("parquet") \
   .load("/data/flight-data/parquet/2010-summary.parquet") \
   . show(5)
```

## Spark Writing Parquet 文件

示例 1:

```scala
// in Scala

csvFile.write.format("parquet")
   .mode("overwrite")
   .save("/tmp/my-parquet-file.parquet")
```

示例 2:

```python
# in Python

csvFile.write.format("parquet") \
   .mode("overwrite") \
   .save("/tmp/my-parquet-file.parquet")
```



# Spark 读取 ORC 文件

ORC 是为 Hadoop 作业而设计的自描述、类型感知的列存储文件格式.它针对大型流式数据读取进行优化, 
但集成了对快速查找所需行的相关支持.

实际上,读取 ORC 文件数据时没有可选项,这是因为 Spark 非常了解该文件格式.

ORC 和 Parquet 有什么区别？在大多数情况下,他们非常相似,本质区别是, 
Parquet 针对 Spark 进行了优化,而 ORC 则是针对 Hive 进行了优化.

## Spark Reading ORC 文件

示例 1:

```scala
// in Scala

spark.read.format("orc")
   .load("/data/flight-data/orc/2010-summary.orc")
   .show()
```

示例 2:

```python
# in Python

spark.read.format("orc") \
   .load("/data/flight-data/orc/2010-summary.orc") \
   .show()
```

## Spark Writing ORC 文件

示例 1:

```scala
// in Scala

csvFile.write.format("orc")
   .mode("overwrite")
   .save("/tmp/my-json-file.orc")
```

示例 2:

```python
# in Python

csvFile.write.format("orc") \
   .mode("overwrite") \
   .save("/tmp/my-json-file.orc")
```


# Spark 读取 SQL Database

数据库不仅仅是一些数据文件,而是一个系统,有许多连接数据库的方式可供选择.
需要确定 Spark 集群网络是否容易连接到数据库系统所在的网络上

读写数据库中的文件需要两步;

1. 在 Spark 类路径中为指定的数据库包含 Java Database Connectivity(JDBC) 驱动;
2. 为连接驱动器提供合适的 JAR 包;

示例:

```bash
./bin/spark-shell \
--driver-class-path postgresql-9.4.1207.jar \
--jars postgresql-9.4.1207.jar
```

## SQLite

SQLite 可以在本地计算机以最简配置工作,但在分布式环境中不行,如果想在分布式环境中运行这里的示例,则需要连接到其他数据库

SQLite 是目前使用最多的数据库引擎,它功能强大、速度快且易于理解,只是因为 SQLite 数据库只是一个文件.


## JDBC 数据源 Options

- `Url`: 要连接的 JDBC URL
- `dbtable`: 表示要读取的 JDBC 表
- `dirver`: 用于连接到此 URL 的 JDBC 驱动器的类名
- `partitionColumn`, lowerBound, upperBound: 描述了如何在从多个 worker 并行读取时对表格进行划分
- `numPartitions`: 在读取和写入数据表时,数据表可用于并行的最大分区数,这也决定了并发 JDBC 连接的最大数目
- `fetchsize`: 表示 JDBC 每次读取多少条记录
- `batchsize`: 表示 JDBC 批处理的大小,用于指定每次写入多少条记录
- `isolationLevel`: 表示数据库的事务隔离级别(适用于当前连接)
- `truncate`: 
- `createTableOptions`: 
- `createTableColumnTypes`: 表示创建表时使用的数据库列数据类型,而不使用默认值



## Spark Reading From SQL Database

示例 1:

```scala
// in Scala

import java.sql.DirverManager

// 指定格式和选项
val dirver = "org.sqlite.JDBC"
val path = "/data/flight-data/jdbc/my-sqlite.db"
val url = s"jdbc::sqlite:/${path}"
val tablename = "flight_info"

// 测试连接
val connection = DirverManager.getConnection(url)
connection.isClosed()
connection.close()

// 从 SQL 表中读取 DataFrame
val dbDataFrame = spark
   .read
   .format("jdbc")
   .option("url", url)
   .option("dbtable", tablename)
   .option("driver", driver)
   .load()
```

示例 2:

```python
# in Python

# 指定格式和选项
driver = "org.sqlite.JDBC"
path = "/data/flight-data/jdbc/my-sqlite.db"
url = "jdbc:sqlite:"  + path
tablename = "flight_info"

# 从 SQL 表中读取 DataFrame
dbDataFrame = spark \
   .read \
   .format("jdbc") \
   .option("url", url) \
   .option("dbtable", tablename) \
   .option("driver", driver)
   .load() 
```

示例 3:

```scala
// in Scala

// 指定格式和选项
val driver = "org.postgresql.Driver"
val url = "jdbc:postgresql://database_server"
val tablename = "schema.tablename"
val username = "username"
val password = "my-secret-password"

// 从 SQL 表中读取 DataFrame
val pgDF = spark
   .read
   .format("jdbc")
   .option("driver", driver)
   .option("url", url)
   .option("dbtable", tablename)
   .option("user", username)
   .option("password", password)
   .load()
```

示例 4:

```python
# in Python

# 指定格式和选项
driver = "org.postgresql.Driver"
url = "jdbc:postgresql://database_server"
tablename = "schema.tablename"
username = "username"
password = "my-secret-password"

# 从 SQL 表中读取 DataFrame
pgDF = spark \
   .read \
   .format("jdbc") \
   .option("driver", driver) \
   .option("url", url) \
   .option("dbtable", tablename) \
   .option("user", username) \
   .option("password", password) \
   .load()
```

结果查询:

```scala
dbDataFrame.select("DEST_COUNTRY_NAME").distinct().show(5)
```

## 查询下推

在创建 DataFrame 之前,Spark 会尽力过滤数据库中的数据.例如,在查询中,从查询计划可以看到它从表中只选择相关的列名.

```scala
// in Scala

dbDataFrame.select("DEST_COUNTRY_NAME").distinct().explain()
```

在某些查询中,Spark 实际上可以做得更好,例如,如果在DataFrame上指定一个 `filter`,Spark 就会将过滤器函数下推到数据库端, 在下面的解释计划中可以看到 PushedFilters 的操作.

```scala
// in Scala

dbDataFrame.filter("DEST_COUNTRY_NAME in ('Anguilla', 'Sweden')").explain
```

```python
dbDataFrame.filter("DEST_COUNTRY_NAME in ('Anguilla', 'Sweden')").explain
```

### 并行读取数据库

Spark 有一个底层算法,可以将多个文件放入一个数据分片,或者反过来将一个文件划分到多个数据分片,这取决于文件大小及文件类型和压缩格式是否允许划分.

SQL 数据库中也存在与文件一样的分片灵活性,但必须手动配置它,可以通过指定最大分区数量来限制并行读写的最大数量.

```scala
// in Scala

// 指定格式和选项
val driver = "org.postgresql.Driver"
val url = "jdbc:postgresql://database_server"
val tablename = "schema.tablename"

val dbDataFrame = spark
   .read
   .format("jdbc")
   .option("url", url)
   .option("dbtable", tablename)
   .option("driver", driver)
   .option("numPartitions", 10)
   .load()
```

```python
# in Python

// 指定格式和选项
driver = "org.postgresql.Driver"
url = "jdbc:postgresql://database_server"
tablename = "schema.tablename"

dbDataFrame = spark
   .read \
   .format("jdbc") \
   .option("url", url) \
   .option("dbtable", tablename) \
   .option("driver", driver) \
   .option("numPartitions", 10) \
   .load()
```

可以通过在连接中显式地将谓词下推到 SQl 数据库中执行,这有利于通过制定谓词来控制分区数据的物理存放位置.

```scala
// in Scala

val props = new java.util.Properties
props.setProperty("driver", "org.sqlite.JDBC")
val predicates = Array(
   "DEST_COUNTRY_NAME = 'Sweden' OR ORIGIN_COUNTRY_NAME = 'Sweden'",
   "DEST_COUNTRY_NAME = 'Anguilla' OR ORIGIN_COUNTRY_NAME = 'Anguilla'"
)
spark.read.jdbc(url, tablename, predicates, props).show()
spark.read.jdbc(url, tablename, predicates, props).rdd.getNumPartitions // 2
```

```python
# in Python

props. = {"driver": "org.sqlite.JDBC"}
predicates = [
   "DEST_COUNTRY_NAME = 'Sweden' OR ORIGIN_COUNTRY_NAME = 'Sweden'",
   "DEST_COUNTRY_NAME = 'Anguilla' OR ORIGIN_COUNTRY_NAME = 'Anguilla'"
]
spark.read.jdbc(url, tablename, predicates = predicates, properties = props).show()
spark.read.jdbc(url, tablename, predicates = predicates, properties = props).rdd.getNumPartitions // 2
```



### 基于滑动窗口的分区

如何基于谓词进行分区？

在下面的例子中,将基于数值型的 count 列进行分区.在这里,为第一个分区和最后一个分区分别制定一个最小值和一个最大值,
超出该范围的数据将存放到第一个分区和最后一个分区;接下来,指定分区总数(这是为了并行操作的).然后 Spark 会并行查询数据库,
并返回 numPartitions 个分区.只需修改 count 列数值的上界和下界,即可将数据相应地存放到各个分区中.


```scala
// in Scala

val = colName = "count"
val lowerBound = 0L
val upperBound = 348113L // 这是数据集最大行数
val numPartitions = 10

// 根据 count 列数值从小到大均匀划分 10 个间隔区间的数据,之后每个区间数据被分配到一个分区
spark.read.jdbc(url, tablename, colName, lowerBound, upperBound, numPartitions, props).count() // 255
```

```python
# Python

colName = "count"
lowerBound = 0L
upperBound = 348113L  # 这是数据集最大行数
numPartitions = 10

# 根据 count 列数值从小到大均匀划分 10 个间隔区间的数据,之后每个区间数据被分配到一个分区
spark.read.jdbc(url, 
                  tablename, 
                  column = colName, 
                  properties = props, 
                  lowerBound = lowerBound, 
                  upperBound = upperBound, 
                  numPartitions = numPartitions).count() # 255
```




## Spark Writing To SQL Database

写入 SQL 数据库只需指定 URI 并指定写入模式来写入数据库即可.

示例 1:

```scala
// in Scala

val newPath = "jdbc:sqlite://tmp/my-sqlite.db"

csvFile
   .write
   .mode("overwrite")
   .jdbc(newPath, tablename, props)
```

示例 2:

```python
# in Python

newPath = "jdbc:sqlite://tmp/my-sqlite.db"

csvFile
   .write
   .jdbc(newPath, tablename, mode = "overwrite", properites = props)
```

示例 3:

```scala
// in Scala

val newPath = "jdbc:sqlite://tmp/my-sqlite.db"

csvFile
   .write
   .mode("append")
   .jdbc(newPath, tablename, props)
```

示例 4:

```python
# in Python

newPath = "jdbc:sqlite://tmp/my-sqlite.db"

csvFile
   .write
   .jdbc(newPath, tablename, mode = "append", properites = props)
```

查看结果:

示例 1:

```scala
// in Scala

spark
   .read
   .jdbc(newPath, tablename, props)
   .count() // 255
```

示例 2:

```python
# in Python

spark
   .read
   .jdbc(newPath, tablename, properites = props)
   .count() # 255
```

示例 3:

```scala
// in Scala

spark
   .read
   .jdbc(newPath, tablename, props)
   .count() // 765
```

示例 4:

```python
# in Python

spark
   .read
   .jdbc(newPath, tablename, properites = props)
   .count() # 765
```




# Spark 读取 Text 文件

Spark 还支持读取纯文本文件,文件中每一行将被解析为 DataFrame 中的一条记录,然后根据要求进行转换.

假设需要将某些 Apache 日志文件解析为结构化的格式,或是想解析一些纯文本以进行自然语言处理,这些都需要操作文本文件.
由于文本文件能够充分利用原生(native tye)的灵活性,因此它很适合作为 Dataset API 的输入.

## Spark Reading Text 文件

读取文本文件非常简单, 只需指定类型为 `textFile` 即可:

- 如果使用 `textFile`, 分区目录名将被忽略. 
- 如果要根据分区读取和写入文本文件, 应该使用 `text`,它会在读写时考虑分区.

示例 1:

```scala
spark.read.textFile("/data/flight-data/csv/2010-summary.csv")
   .selectExpr("split(value, ',') as rows")
   .show()
```

## Spark Writing Text 文件

当写文本文件时, 需确保仅有一个字符串类型的列写出; 否则, 写操作将失败.

示例 1:

```scala
// in Scala

csvFile
   .select("DEST_COUNTRY_NAME")
   .write.text("/tmp/simple-text-file.txt")
```

示例 2:

```python
# in Python

csvFile \
   .limit(10) \
   .select("DEST_COUNTRY_NAME", "count") \
   .write.partitionBy("count") \
   .text("/tmp/five-csv-file2py.csv")
```



# 高级 I/O

可以通过在写入之前控制数据分片来控制写入文件的并行度,还可以通过控制数据分桶(bucketing)和数据划分(partition)来控制特定的数据布局方式

## 可分割的文件类型和压缩

某些文件格式是“可分割的”,因此 Spark 可以只获取该文件中满足查询条件的某一个部分,无需读取整个文件,从而提高读取效率.
此外,假设你使用的是 Hadoop 分布式文件系统(HDFS),则如果该文件包含多个文件夹,分割文件则可进一步优化提高性能.
与此同时需要进行压缩管理,并非所有的压缩格式都是可分割的.存储数据的方式对 Spark 作业稳定运行至关重要,
我们推荐采用 gzip 压缩格式的 Parquet 文件格式.


## 并行读数据

多个执行器不能同时读取统一文件,但可以同时读取不同的文件.
通常,这意味着当你从包含多个文件的文件夹中读取时,每个文件都将被视为 DataFrame 的一个分片,
并由执行器并行读取,多余的额文件会进入读取队列等候.

## 并行写数据

写数据涉及的文件数量取决于 DataFrame 的分区数.默认情况是每个数据分片都还有一定的数据写入,
这意味着虽然我们指定的是一个“文件”,但实际上它是由一个文件夹中的多个文件组成,每个文件对应着一个数据分片.

示例:

```scala
csvFile
   .repartition(5)
   .write
   .format("csv")
   .save("/tmp/multiple.csv")
```

***
**Note:**

* 它会生成包含 5 个文件的 文件夹,调用 `ls` 命令就可以查看到:

```bash
$ ls /tmp/multiple.csv
```
***

   

   



### 数据划分

数据划分工具支持你在写入数据时控制存储数据以及存储数据的位置.
将文件写出时,可以将列编码为文件夹,这使得你在之后读取时可跳过大量数据,
只读入与问题相关的列数据而不必扫描整个数据集.所有基于文件的数据源都支持这些;


```scala
// in Scala

csvFile.limit(10).write.mode("overwrite").partitionBy("DEST_COUNTRY_NAME").save("/tmp/partitioned-files.parquet")
```


```python
# in python

csvFile.limit(10).write.mode("overwrite").partitionBy("DEST_COUNTRY_NAME").save("/tmp/partitioned-files.parquet")
```

写操作完成后,Parquet “文件” 中就会有一个文件夹列表;

```bash
$ ls /tmp/partitioned-files.parquet
```

其中每一个都将包含 Parquet 文件,这些文件包含文件夹名称中谓词为 true 的数据;

```bash
$ ls /tmp/partitioned-files.parquet/DEST_COUNTRY_NAME=Senegal/
```

***
**Note:**

* 读取程序对某表执行操作之前经常执行过滤操作,这时数据划分就是最简单的优化.例如,基于日期来划分数据最常见,
  因为通常我们只想查看前一周(而不是扫描所有日期数据),这个优化可以极大提升读取程序的速度.
***


### 数据分桶

数据分桶是另一种文件组织方法,可以使用该方法控制写入每个文件的数据.
具有相同桶 ID (哈希分桶的 ID) 的数据将放置到一个物理分区中,
这样就可以避免在稍后读取数据时进行洗牌(shuffle).根据你之后希望如何使用该数据来对数据进行预分区,
就可以避免连接或聚合操作时执行代价很大的 shuffle 操作.

与其根据某列进行数据划分,不如考虑对数据进行分桶,因为某列如果存在很多不同的值,就可能写出一大堆目录.这将创建一定数量的文件,数据也可以按照组织起来放置到这些“桶”中;

```scala
// in Scala 

val numberBuckets = 10
val columnToBucketBy = "count"

csvFile
   .write
   .format("parquet")
   .mode("overwrite")
   .bucketBy(numberBuckets, columnToBucketBy)
   .save("bucketedFiles")
```

***
**Note:**

* 数据分桶仅支持 Spark 管理的表.有关数据分桶和数据划分的更多信息, 
  请参阅 2017 年 Spark Summit 的演讲(https://spark-summit.org/2017/event/why-you-should-care-about-data-layout-in-the-filesystem/).
***

## 写入复杂类型

Spark 具有多种不同的内部类型.尽管 Spark 可以使用所有这些类型,但并不是每种数据文件格式都支持这些内部类型.
例如,CSV 文件不支持复杂类型,而 Parquet 和 ORC 文件则支持复杂类型.

## 管理文件大小

管理文件大小对数据写入不那么重要,但对之后的读取很重要.当你写入大量的小文件时,由于管理所有的这些小文件而产生很大的元数据开销.
许多文件系统(如 HDFS)都不能很好的处理大量的小文件,而 Spark 特别不适合处理小文件.
你可能听过“小文件问题”,反之亦然,你也不希望文件太大,因为你只需要其中几行时,必须读取整个数据块就会使效率低下.

Spark 2.2 中引入了一种更自动化地控制文件大小的新方法.之前介绍了输出文件数量与写入时数据分片数量以及选取的划分列有关.
现在,则可以利用另一个工具来限制输出文件大小,从而可以选出最优的文件大小.
可以使用 `maxRecordsPerFile` 选项来指定每个文件的最大记录数,这使得你可以通过控制写入每个文件的记录数来控制文件大小.
例如,如果你将程序(writer)的选项设置为 `df.write.option("maxRecordsPerFile", 5000)`,Spark 将确保每个文件最多包含5000条记录.


## Cassandra Connector

- [Cassandra Connector](https://github.com/datastax/spark-cassandra-connector)

***
**Note:**

*  有很多方法可以用于实现自定义的数据源, 但由于 API 正在不断演化发展(为了更好地支持结构化流式处理).
***

