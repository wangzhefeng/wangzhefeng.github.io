---
title: Hive
author: 王哲峰
date: '2022-09-12'
slug: database-hive
categories:
  - database
tags:
  - tool
  - sql
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

- [Hive 简介](#hive-简介)
  - [简介](#简介)
  - [优点](#优点)
  - [缺点](#缺点)
  - [架构](#架构)
  - [与 RDBMS 比较](#与-rdbms-比较)
- [Hive 安装](#hive-安装)
- [Hive 基本操作](#hive-基本操作)
  - [Hive 常用命令](#hive-常用命令)
  - [本地文件导入 Hive 表中](#本地文件导入-hive-表中)
  - [Hive 其他操作](#hive-其他操作)
- [Hive 常见属性配置](#hive-常见属性配置)
  - [数据仓库位置](#数据仓库位置)
  - [查询信息显示配置](#查询信息显示配置)
  - [参数配置方式](#参数配置方式)
- [Hive 数据类型](#hive-数据类型)
  - [基本数据类型](#基本数据类型)
  - [集合数据类型](#集合数据类型)
  - [数据类型转化](#数据类型转化)
- [Hive 文件格式](#hive-文件格式)
- [Hive 数据库](#hive-数据库)
  - [查询现有数据库](#查询现有数据库)
  - [创建新的数据库](#创建新的数据库)
  - [输出数据库的信息](#输出数据库的信息)
  - [切换用户当前工作数据库](#切换用户当前工作数据库)
  - [修改数据库](#修改数据库)
  - [删除数据库](#删除数据库)
- [Hive 表](#hive-表)
  - [创建管理表](#创建管理表)
  - [创建分区管理表](#创建分区管理表)
  - [创建外部表](#创建外部表)
  - [复制表模式](#复制表模式)
  - [列举表的属性信息](#列举表的属性信息)
  - [列举数据库中的所有表](#列举数据库中的所有表)
  - [查看表的详细表结构信息](#查看表的详细表结构信息)
  - [删除表](#删除表)
  - [修改表](#修改表)
  - [向管理表中装载数据](#向管理表中装载数据)
- [Hive 视图](#hive-视图)
  - [创建视图](#创建视图)
  - [复制视图](#复制视图)
  - [删除视图](#删除视图)
  - [查看视图](#查看视图)
  - [显示视图的元数据信息](#显示视图的元数据信息)
  - [视图是只读的](#视图是只读的)
- [Hive 索引](#hive-索引)
  - [建立索引](#建立索引)
  - [Bitmap 索引](#bitmap-索引)
  - [重建索引](#重建索引)
  - [显示索引](#显示索引)
  - [删除索引](#删除索引)
  - [查看查询语句是否用到了索引](#查看查询语句是否用到了索引)
  - [定制化索引](#定制化索引)
- [Hive 函数](#hive-函数)
  - [函数基础](#函数基础)
  - [内置函数](#内置函数)
    - [数学函数](#数学函数)
    - [集合函数](#集合函数)
    - [类型转换函数](#类型转换函数)
    - [日期函数](#日期函数)
    - [条件函数](#条件函数)
    - [字符串函数](#字符串函数)
    - [聚合函数](#聚合函数)
    - [表生成函数](#表生成函数)
  - [用户自定义函数](#用户自定义函数)
- [参考资料](#参考资料)
</p></details><p></p>

# Hive 简介

## 简介

Hive 是基于 Hadoop 的一个数据仓库工具。Hive 的计算基于 Hadoop 实现的一个特别的计算模型 MapReduce，
它可以将计算任务分割成多个处理单元，然后分散到一群家用或服务器级别的硬件机器上，降低成本并提高水平扩展性

大数据主要解决海量数据的三大问题：传输问题、存储问题、计算问题。而 Hive 主要解决存储和计算问题。
Hive 是由 Facebook 开源的基于 Hadoop 的数据仓库工具，用于解决海量结构化日志的数据统计

Hive 的数据存储在 Hadoop 一个分布式文件系统上，即 HDFS，运行在 Yarn 上，
但它可以将结构化的数据文件映射为一张表，并提供类 SQL 的查询功能，称之为 Hive-SQL，简称 HQL。
简单来说，Hive 是在 Hadoop 上封装了一层 HQL 的接口，这样开发人员和数据分析人员就可以使用 HQL 来进行数据的分析，
而无需关注底层的 MapReduce 的编程开发。所以 Hive 的本质是将 HQL 转换成 MapReduce 程序。
Hive 把表和字段转换成 HDFS 中的文件夹和文件，并将这些元数据保持在关系型数据库中，
如 Derby 或 MySQL。Hive 适合做离线数据分析，如：批量处理和延时要求不高场景

## 优点

需明确的是，Hive 作为数仓应用工具，对比 RDBMS(关系型数据库)：

* Hive 不支持记录级别的增删改操作，但是可以通过查询创建新表来将结果导入到文件中；Hive 2.3.2 版本支持记录级别的插入操作
* Hive 延迟较高，不适用于实时分析
* Hive 不支持事务，因为没有增删改，所以主要用来做 OLAP（联机分析处理），而不是 OLTP（联机事务处理）
* Hive 自动生成的 MapReduce 作业，通常情况下不够智能

## 缺点

* Hive 封装了一层接口，并提供类 SQL 的查询功能，避免去写 MapReduce，减少了开发人员的学习成本
* Hive 支持用户自定义函数，可以根据自己的需求来实现自己的函数
* 适合处理大数据
* 可扩展性强：可以自由扩展集群的规模，不需要重启服务而进行横向扩展
* 容错性强：可以保障即使有节点出现问题，SQL 语句也可以完成执行

## 架构

Hive 架构图：

![img](images/hive.png)

如上图所示：

* Hadoop 使用 HDFS 进行存储，并使用 MapReduce 进行计算
* Diver 中包含解释器（Interpreter）、编译器（Compiler）、优化器（Optimizer）和执行器（Executor）：
    - 解释器：利用第三方工具将 HQL 查询语句转换成抽象语法树 AST，并对 AST 进行语法分析，
      比如说表是否存在、字段是否存在、SQL 语义是否有误
    - 编译器：将 AST 编译生成逻辑执行计划
    - 优化器：多逻辑执行单元进行优化
    - 执行器：把逻辑执行单元转换成可以运行的物理计划，如 MapReduce、Spark
* Hive 提供了 CLI（hive shell）、JDBC/ODBC（Java 访问 hive）、WebGUI 接口（浏览器访问 hive）
* Hive 中有一个元数据存储（Metastore），通常是存储在关系数据库中如 MySQL、Derby 等。
  元数据包括表名、表所在数据库、表的列名、分区及属性、表的属性、表的数据所在的目录等
* Thrift Server 为 Facebook 开发的一个软件框架，可以用来进行可扩展且跨语言的服务开发，
  Hive 通过集成了该服务能够让不同编程语言调用 Hive 的接口

所以 Hive 查询的大致流程为：通过用户交互接口接收到 HQL 的指令后，经过 Driver 结合元数据进行类型检测和语法分析，
并生成一个逻辑方法，通过进行优化后生成 MapReduce，并提交到 Hadoop 中执行，并把执行的结果返回给用户交互接口

## 与 RDBMS 比较

Hive 采用类 SQL 的查询语句，所以很容易将 Hive 与关系型数据库（RDBMS）进行对比。
但其实 Hive 除了拥有类似 SQL 的查询语句外，再无类似之处。
需要明白的是：数据库可以用做 online 应用；而 Hive 是为数据仓库设计的

|         | Hive                   | RDBMS |
|---------|------------------------|----|
| 查询语言 | HQL                    | SQL |
| 数据存储 | HDFS                   | 本地文件系统中 |
| 数据更新 | 读多写少（不建议改写）     | 增删改查 |
| 数据操作 | 覆盖追加                 | 行级别更新删除 |
| 索引    | 0.8 版本后引入 bitmap 索引 | 建立索引 |
| 执行    |	MapReduce               | Executor |
| 执行延迟 | 延迟较高                 | 延迟较低 |
| 可扩展性 | 可扩展性高               | 可扩展性低 |
| 数据规模 | 很大                    | 较小 |
| 分区    |	支持                    | 支持 |

总的来说，Hive 只具备 SQL 的外表，但应用场景完全不同。Hive 只适合用来做海量离线数据统计分析，也就是数据仓库。
清楚这一点，有助于从应用角度理解 Hive 的特性

# Hive 安装

* TODO

# Hive 基本操作

## Hive 常用命令

Hive 命令帮助：

```bash
$ hive -help

usage: hive
 -d,--define <key=value>          Variable substitution to apply to Hive
                                  commands. e.g. -d A=B or --define A=B
    --database <databasename>     Specify the database to use
 -e <quoted-query-string>         SQL from command line
 -f <filename>                    SQL from files
 -H,--help                        Print help information
    --hiveconf <property=value>   Use value for given property
    --hivevar <key=value>         Variable substitution to apply to Hive
                                  commands. e.g. --hivevar A=B
 -i <filename>                    Initialization SQL file
 -S,--silent                      Silent mode in interactive shell
 -v,--verbose                     Verbose mode (echo executed SQL to the
                                  console)
```

常用的两个命令是 `"-e"` 和 `"-f"`：

* `-e` 表示不进入 hive cli 直接执行 SQL 语句

```bash
$ hive -e "select * from teacher;"
```

* `-f` 表示执行 SQL 语句的脚本，方便用 Crontab 进行定时调度

```bash
$ hive -f /opt/module/datas/hivef.sql
```

## 本地文件导入 Hive 表中

1. 首先需要创建一张表

```sql
CREATE TABLE student (
    id int,
    name string
)
ROW FORMAT DELIMITED FIELDS TERMINATED BY '\t';
```

> 简单介绍下字段：
> 
> * `ROW FORMAT DELIMITED`：分隔符的设置的开始语句
> * `FIELDS TERMINATED BY`：设置每一行字段与字段之间的分隔符，我们这是用 `'\t'` 进行划分
> 
> 除此之外，还有其他的分割符设定：
> 
> * `COLLECTION ITEMS TERMINATED BY`：设置一个复杂类型（array/struct）字段的各个 item 之间的分隔符
> * `MAP KEYS TERMINATED BY`：设置一个复杂类型（Map）字段的 key-value 之间的分隔符
> * `LINES TERMINATED BY`：设置行与行之间的分隔符
> 
> 这里需要注意的是 `ROW FORMAT DELIMITED` 必须在其它分隔设置之前；
> `LINES TERMINATED BY` 必须在其它分隔设置之后，否则会报错

2. 然后，我们需要准备一个文件

```
# stu.txt
1 Xiao_ming
2 xiao_hong
3 xiao_hao
```

需要注意，每行内的字段需要用 `'\t'` 进行分割

3. 接着需要使用 `load` 语法加载本地文件，`load` 语法为

```bash
hive> load data [local] inpath 'filepath' [overwrite] 
      into table tablename [partition (partcol1=val1,partcol2=val2...)]
```

其中：

* `local` 用来控制选择本地文件还是 HDFS 文件
* `overwrite` 可以选择是否覆盖原来数据
* `partition` 可以制定分区

```bash
hive> load data local inpath '/Users/***/Desktop/stu1.txt' into table student;
```

最后查看下数据：

```bash
hive> select * from student;
OK
1 Xiao_ming
2 xiao_hong
3 xiao_hao
Time taken: 1.373 seconds, Fetched: 3 row(s)
```

## Hive 其他操作

Hive 退出 hive cli 模式，不过这种区别只是在旧版本中有，两者在新版本已经没有区别了

* quit：不提交数据退出
* exit：先隐性提交数据，再退出

在 hive cli 中可以用以下命令查看 HDFS 文件系统和本地文件系统：

```bash
$ dfs -ls /;  # 查看 hdfs 文件系统
$ ! ls ./;  # 查看本地文件系统
```

用户根目录下有一个隐藏文件记录着 hive 输入的所有历史命令：

```bash
$ cat ./hivehistory
```

注意：hive 语句不区分大小写

# Hive 常见属性配置

## 数据仓库位置



## 查询信息显示配置


## 参数配置方式





# Hive 数据类型

## 基本数据类型


## 集合数据类型

* 整数
    - TINYINT
    - SMALLINT
    - INT
    - BIGINT
* 布尔型
    - BOOLEAN
* 浮点型
    - FLOAT
    - DOUBLE
* 字符型
    - STRING
* 日期时间
    - TIMESTAMP
* 其他
    - BINARY
    - STRUCT
    - MAP
    - ARRAY


## 数据类型转化




# Hive 文件格式

* csv
* tsv

```
--root
    -- /001
    -- /002
    -- /003
```

# Hive 数据库

* 查询
* 创建
* 修改
* 删除

## 查询现有数据库

```sql
SHOW DATABASES;
SHOW DATABASES LIKE 'h.*'; | -- 正则表达式
```

## 创建新的数据库

```sql
-- 数据库所在的目录: /usr/hive/warehouse/financials.db
CREATE DATABASE financials;
CREATE DATABASE [IF NOT EXISTS] financials;

-- 数据库所在的目录: /my/preperred/directory.db
CREATE DATABASE financials
LOCATION '/my/preperred/directory';

-- 增加数据库描述信息
CREATE DATABASE financials
COMMENT 'Holds all financial tables';
WITH DBPROPERTIES  ('creator' = 'wangzhefeng', 'date': '2018-08-11');
```

## 输出数据库的信息

```sql
DESCRIBE DATABASE financials;
DESCRIBE DATABASE [EXTENDED] financials;
```

## 切换用户当前工作数据库

```sql
USE financials;

set hiv.cli.print.current.db = true;
set hiv.cli.print.current.db = true;
```

## 修改数据库

```sql
ALTER DATABASE financials 
SET DBPROPERTIES ('edited-by' = 'wangzhefeng');
```

## 删除数据库

```sql
DROP DATABASE IF EXISTS financials;
DROP DATABASE IF EXISTS financials RESTRICT;
DROP DATABASE IF EXISTS financials CASCADE;
```

# Hive 表

* 创建
* 删除
* 查询
* 修改

## 创建管理表

* 管理表也叫内部表

```sql
CREATE TABLE IF NOT EXISTS mydb.employees (
    name STRING COMMENT 'Employees name',
    salary FLOAT COMMENT 'Employee salary',
    subordinates ARRAY<STRING> COMMENT 'Names of subordinates',
    deductions MAP<STRING, FLOAT> COMMENT 'Keys are deductions names, values are percentages',
    address STRUCT<street:STRING, city:STRING, state:STRING, zip:INT> COMMENT 'Home address'
)
COMMENT 'Description of the table'  -- 表注释
TBLPROPERTIES (  -- 表属性
    'creator' = 'wangzhefeng', 
    'create-at' = "2018-08-11 17:56:00", 
    ...)
LOCATION '/usr/hive/warehouse/mydb.db/employees';
```

## 创建分区管理表

```sql
CREATE TABLE IF NOT EXISTS employees (
    name STRING,
    salary FLOAT,
    subordinates ARRAY<STRING>,
    deductions MAP<STRING, FLOAT>,
    address STRUCT<street:STRING, city:STRING, state:STRING, zip:INT>
)
PATRITION BY (country STRING, state STRING);

-- set hive.mapred.mode = strict;
-- set hive.mapred.mode = nonstrict;
```

## 创建外部表

```sql
CREATE EXTERNAL TABLE IF NOT EXISTS stocks (
    exchange STRING,
    symbol STRING,
    ymd STRING,
    price_open FLOAT,
    price_high FLOAT,
    price_low FLOAT,
    price_open FLOAT,
    volume INT,
    price_adj_close FLOAT
)
ROW FORMAT DELIMITED 
FIELDS TERMINATED BY ','
LOCATION '/data/stocks';
```

## 复制表模式

复制另一张表的模式, 数据不会复制

```sql
-- 管理表
CREATE TABLE [IF NOT EXISTS] mydb.employees2
LIKE mydb.employees;

-- 外部表
CREATE EXTERNAL TABLE [IF NOT EXISTS] mydb.employees3;
LIKE mydb.employees
LOCATION '/path/to/data';
```

## 列举表的属性信息

```sql
SHOW TBLPROPERTIES mydb.employees;
SHOW PARTITIONS employees;
SHOW PARTITIONS employees PARTITION (country = 'US');
SHOW PARTITIONS employees PARTITION (country = 'US', state = 'AK');
```

## 列举数据库中的所有表

```sql
USE mydb;

SHOW TABLES;
SHOW TABLES IN mydb;
SHOW TABLES 'empl.*';
```

## 查看表的详细表结构信息

```sql
DESCRIBE EXTENDED mydb.employees;
DESCRIBE FORMATED mydb.employees;
DESCRIBE mydb.employees.salary;
```

## 删除表

* 管理表: 元数据信息和表内数据都会删除
* 外部表: 删除元数据信息

```sql
DROP TABLE [IF EXISTS] employees;
```

## 修改表

* 只修改元数据信息

表重命名

```sql 
ALTER TABLE table1 
RENAME TO table2;
```

增加, 修改, 删除分区表

```sql
ALTER TABLE table_name
ADD [IF NOT EXISTS] 
PARTITION ()
PARTITION ()
PARTITION ();

ALTER TABLE table_name 
PARTITION ()
SET LOCATION '';

ALTER TABLE table_name
DROP [IF EXISTS]
PARTITION ();
```

修改列

```sql
ALTER TABLE table_name
CHANGE COLUMN hms hours_minutes_seconds INT 
COMMENT ""
AFTER severity; -- FIRST
```

增加列

```sql
ALTER TABLE table_name
ADD COLUMNS (
    app_name STRING COMMENT '',
    session_id INT COMMENT '',
);
```

删除、替换列

```sql
ALTER TABLE table_name 
REPLACE COLUMNS (
    hours_minutes_seconds INT COMMENT '',
    severity STRING COMMENT '',
    message STRING COMMENT ''
)
```

修改表属性

```sql
ALTER TABLE table_name
SET TBLPROPERTIES ('notes' = '');
```

修改存储属性

```sql
ALTER TABLE table_name
PARTITION ()
SET FILEFORMAT SEQUENCEFILE;

ALTER TABLE table_name
SET SERDE 'com.example.JSONSerDe'
WITH SERDEPROPERTIES (
    'prop3' = 'value3',
    'prop4' = 'value4'
);
```

## 向管理表中装载数据

```sql
LOAD DATA LOCAL INPATH '${env:HOME}/california-employees'
OVERWRITE INTO TABLE employees
PARTITION (country = 'US', state = 'CA');
```

* overwrite

```sql
INSERT OVERWRITE TABLE employees
PARTITION (country = 'US', state = 'OR')
SELECT *
FROM staged_employees se
WHERE se.cnty = 'US' AND se.st = 'OR';
```

* append

```sql
INSERT [INTO] TABLE employees
PARTITION (country = 'US', state = 'OR')
SELECT *
FROM staged_employees se
WHERE se.cnty = 'US' AND se.st = 'OR';
```

# Hive 视图

## 创建视图

```sql
CREATE VIEW [IF NOT EXISTS] shorter_join(field1, field2, field3, ...) AS
SELECT *
FROM people 
JOIN cart ON (cart.people_id = people.id)
WHERE firstname = 'john';

CREATE VIEW [IF NOT EXISTS] shipments(time, part)
COMMENT 'Time and parts for shipments.'
TBLPROPERTIES ('creator' = 'me') AS 
SELECT *
FROM people 
JOIN cart ON (cart.people_id = people.id)
WHERE firstname = 'john';
```

## 复制视图

```sql
CREATE VIEW shipments2
LIKE shipments;
```

## 删除视图

```sql
DROP VIEW [IF EXISTS] shipments;
```

## 查看视图

```sql
SHOW TABLES;
```

## 显示视图的元数据信息

```sql
DESCRIBE shipments;
DESCRIBE EXTENDED shipments;
```

## 视图是只读的

* 视图不能进行 `insert`, `load`
* 视图是只读的，只允许修改元数据信息

```sql
ALTER VIEW shipments 
SET TBLPROPERTIES ('create' = 'me', 'create_at' = '2018-11-10');
```

# Hive 索引

```sql
-- 建表employees
CREATE TABLE employees (
    name STRING,
    salary FLOAT,
    subordinates ARRAY<STRING>,
    deductions MAP<STRING, FLOAT>,
    address STRUCT<street: STRING, city: STRING, state: STRING, zip: INT>
)
PARTITION BY (country STRING, state STRING);
```

## 建立索引

```sql
CREATE INDEX employees_index 
ON TABLE employees (country)
AS 'org.apache.hadoop.hive.ql.index.compact.CompactIndexHandler'
WITH DEFERRED REBUILD 
IDXPROPERTIES ('creator' = 'wangzhefeng', 'create_at' = '2018-11-09')
IN TABLE employees_index_table
PARTITIONED BY (country, name)
COMMENT 'Employees indexed by country and name';
```

## Bitmap 索引

```sql
CREATE INDEX employees_index 
ON TABLE employees (country)
AS 'BITMAP'
WITH DEFERRED REBUILD 
IDXPROPERTIES ('creator' = 'wangzhefeng', 'create_at' = '2018-11-09')
IN TABLE employees_index_table
PARTITIONED BY (country, name)
COMMENT 'Employees indexed by country and name';
```

## 重建索引

```sql
ALTER INDEX employees_index
ON TABLE employees
PARTITION (country = 'US')
REBUILD;
```

## 显示索引

```sql
SHOW [FORMATTED] INDEX ON employees;
SHOW [FORMATTED] INDEXES ON employees;
```

## 删除索引

```sql
DROP INDEX [IF EXISTS] employees_index 
ON TABLE employees;
```

## 查看查询语句是否用到了索引

```sql
EXPLAIN ...;
```

## 定制化索引

* https://cwiki.apache.org/confluence/display/Hive/IndexDe#CREATE_INDEX

# Hive 函数

## 函数基础

函数类型: 

* 内置函数
* 用户自定义函数(加载)

列举当前会话中所加载的所有函数名称：

```sql
SHOW FUNCTIONS;
```

查看函数文档：

```sql
DESCRIBE FUNCTION concat;
DESCRIBE FUNCTION EXTENDED concat;
```

调用函数：

```sql
SELECT 
    concat(column1, column2) AS x 
FROM table_name;
```

## 内置函数

Hive 的内置函数包括：

* 数学函数（Mathematical Functions）
* 集合函数（Collection Functions）
* 类型转换函数（Type Conversion Functions）
* 日期函数（Date Functions）
* 条件函数（Conditional Functions）
* 字符串函数（String Functions）
* 聚合函数（Aggregate Functions）
* 表生成函数（Table-Generating Functions）

当然，Hive 还在一直更新，有需要的话，可以去官网去查看最新的函数

### 数学函数

| 函数 | 描述 |
|----|----|
| round(DOUBLE a) | 返回对 a 四舍五入的 BIGINT 值 |
| round(DOUBLE a, INT d) | 返回 DOUBLE 型 d 的保留 n 位小数的 DOUBLE 型的近似值 |
| bround(DOUBLE a) | 银行家舍入法（1~4：舍，6~9：进，5->前位数是偶：舍，5->前位数是奇：进） |
| bround(DOUBLE a, INT d) | 银行家舍入法,保留 d 位小数 |
| floor(DOUBLE a) | 向下取整，最数轴上最接近要求的值的左边的值  如：6.10->6  -3.4->-4 |
| ceil(DOUBLE a), ceiling(DOUBLE a) | 求其不小于小给定实数的最小整数如：ceil(6) = ceil(6.1)= ceil(6.9) = 6 |
| rand(), rand(INT seed) | 每行返回一个 DOUBLE 型随机数 seed 是随机因子 |
| exp(DOUBLE a), exp(DECIMAL a) | 返回 e 的 a 幂次方，a 可为小数 |
| ln(DOUBLE a), ln(DECIMAL a) | 以自然数为底 d 的对数，a 可为小数 |
| log10(DOUBLE a), log10(DECIMAL a) | 以 10 为底 d 的对数，a 可为小数 |
| log2(DOUBLE a), log2(DECIMAL a) | 以 2 为底数 d 的对数，a 可为小数 |
| log(DOUBLE base, DOUBLE a)log(DECIMAL base, DECIMAL a) | 以 base 为底的对数，base 与 a 都是 DOUBLE 类型 |
| pow(DOUBLE a, DOUBLE p), power(DOUBLE a, DOUBLE p) | 计算 a 的 p 次幂 |
| sqrt(DOUBLE a), sqrt(DECIMAL a) | 计算 a 的平方根 |
| bin(BIGINT a) | 计算二进制 a 的 STRING 类型，a 为 BIGINT 类型 |
| hex(BIGINT a) hex(STRING a) hex(BINARY a) | 计算十六进制 a 的 STRING 类型，如果 a 为 STRING 类型就转换成字符相对应的十六进制 |
| unhex(STRING a) | hex 的逆方法 |
| conv(BIGINT num, INT from_base, INT to_base), conv(STRING num, INT from_base, INT to_base) | 将 BIGINT/STRING 类型的 num 从 from_base 进制转换成 to_base 进制 |
| abs(DOUBLE a) | 计算 a 的绝对值 |
| pmod(INT a, INT b), pmod(DOUBLE a, DOUBLE b) | a 对 b 取模 |
| sin(DOUBLE a), sin(DECIMAL a) | 求 a 的正弦值 |
| asin(DOUBLE a), asin(DECIMAL a) | 求反正弦值 |
| cos(DOUBLE a), cos(DECIMAL a) | 求余弦值 |
| acos(DOUBLE a), acos(DECIMAL a) | 求反余弦值 |
| tan(DOUBLE a), tan(DECIMAL a) | 求正切值 |
| atan(DOUBLE a), atan(DECIMAL a) | 求反正切值 |
| degrees(DOUBLE a), degrees(DECIMAL a) | 奖弧度值转换角度值 |
| radians(DOUBLE a), radians(DOUBLE a) | 将角度值转换成弧度值 |
| positive(INT a), positive(DOUBLE a) | 返回 a |
| negative(INT a), negative(DOUBLE a) | 返回 a 的相反数 |
| sign(DOUBLE a), sign(DECIMAL a) | 如果 a 是正数则返回 1.0，是负数则返回 -1.0，否则返回 0.0 |
| e() | 数学常数 e |
| pi() | 数学常数 pi |
| factorial(INT a) | 求 a 的阶乘 |
| cbrt(DOUBLE a) | 求 a 的立方根 |
| shiftleft(TINYINT|SMALLINT|INT a, INT b)shiftleft(BIGINT a, INT b) | 按位左移 |
| shiftright(TINYINT|SMALLINT|INT a, INTb)shiftright(BIGINT a, INT b) | 按拉右移 |
| shiftrightunsigned(TINYINT|SMALLINT|INTa, INT b),shiftrightunsigned(BIGINT a, INT b) | 无符号按位右移（<<<） |
| greatest(T v1, T v2, ...) | 求最大值 |
| least(T v1, T v2, ...) | 求最小值 |

### 集合函数

| 函数                          | 描述         |
|------------------------------|--------------|
| size(Map<K.V>)               | 求 map 的长度 |
| size(Array)                  | 求数组的长度   |
| map_keys(Map<K.V>)           | 返回 map 中的所有 key |
| map_values(Map<K.V>)         | 返回 map 中的所有 value |
| array_contains(Array, value) | 如该数组 Array 包含 value 返回 true，否则返回 false |
| sort_array(Array)            | 按自然顺序对数组进行排序并返回 |

### 类型转换函数

| 函数 | 描述 |
|----|----|
| binary(string\|binary) | 将输入的值转换成二进制 |
| cast(expr as) | 将 expr 转换成 type 类型 如：cast("1" as BIGINT) 将字符串 1 转换成了 BIGINT 类型，如果转换失败将返回 NULL |

### 日期函数

| 函数 | 描述 |
|----------------------|----|
| from_unixtime(bigint unixtime[, string format]) | 将时间的秒值转换成 format 格式（format 可为“yyyy-MM-dd hh:mm:ss”,“yyyy-MM-dd hh”,“yyyy-MM-dd hh:mm”等等）如 from_unixtime(1250111000,"yyyy-MM-dd") 得到2009-03-12 |
| unix_timestamp() | 获取本地时区下的时间戳 |
| unix_timestamp(string date) | 将格式为 yyyy-MM-dd HH:mm:ss 的时间字符串转换成时间戳  如 unix_timestamp('2009-03-20 11:30:01') = 1237573801 |
| unix_timestamp(string date, string pattern) | | 将指定时间字符串格式字符串转换成 Unix 时间戳，如果格式不对返回 0 如：unix_timestamp('2009-03-20', 'yyyy-MM-dd') = 1237532400 |
| to_date(string timestamp) | 返回时间字符串的日期部分 |
| year(string date) | 返回时间字符串的年份部分 |
| quarter(date/timestamp/string) | 返回当前时间属性哪个季度，如quarter('2015-04-08') = 2 |
| month(string date) | 返回时间字符串的月份部分 |
| day(string date) dayofmonth(date) | 返回时间字符串的天 |
| hour(string date) | 返回时间字符串的小时 |
| minute(string date) | | 返回时间字符串的分钟 |
| second(string date) | | 返回时间字符串的秒 |
| weekofyear(string date) | 返回时间字符串位于一年中的第几个周内  如weekofyear("1970-11-01 00:00:00") = 44, weekofyear("1970-11-01") = 44 |
| datediff(string enddate, string startdate) | 计算开始时间 startdate 到结束时间 enddate 相差的天数 |
| date_add(string startdate, int days) | 从开始时间 startdate 加上 days |
| date_sub(string startdate, int days) | 从开始时间 startdate 减去 days |
| from_utc_timestamp(timestamp, string timezone) | 如果给定的时间戳并非 UTC，则将其转化成指定的时区下时间戳 |
| to_utc_timestamp(timestamp, string timezone) | 如果给定的时间戳指定的时区下时间戳，则将其转化成 UTC 下的时间戳 |
| current_date | 返回当前时间日期 |
| current_timestamp | 返回当前时间戳 |
| add_months(string start_date, int num_months) | 返回当前时间下再增加 num_months 个月的日期 |
| last_day(string date) | 返回这个月的最后一天的日期，忽略时分秒部分（HH:mm:ss） |
| next_day(string start_date, string day_of_week) | | 返回当前时间的下一个星期 X 所对应的日期 如：next_day('2015-01-14', 'TU') = 2015-01-20  以2015-01-14 为开始时间，其下一个星期二所对应的日期为2015-01-20 |
| trunc(string date, string format) | 返回时间的最开始年份或月份  如 trunc("2016-06-26",“MM”)=2016-06-01  trunc("2016-06-26",“YY”)=2016-01-01  注意所支持的格式为 MONTH/MON/MM, YEAR/YYYY/YY |
| months_between(date1, date2) | 返回 date1 与 date2 之间相差的月份，如 date1>date2，则返回正，如果date1<date2,则返回负，否则返回 0.0  如：months_between('1997-02-28 10:30:00', '1996-10-30') = 3.94959677  1997-02-28 10:30:00与1996-10-30 相差 3.94959677 个月 |
| date_format(date/timestamp/string ts, string fmt) | 按指定格式返回时间 date  如：date_format("2016-06-22","MM-dd")=06-22 |

### 条件函数

| 函数                                                        | 描述 |
|------------------------------------------------------------|------|
| if(boolean testCondition, T valueTure, T valueFalseOrNull) | 如果 testCondition 为 true 就返回valueTrue，否则返回 valueFalseOrNull ，（valueTrue，valueFalseOrNull 为泛型） |
| nvl(T value, T default_value) | 如果 value 值为 NULL 就返回 default_value,否则返回 value |
| COALESCE(T v1, T v2, ...) | 返回第一非 NULL 的值，如果全部都为 NULL 就返回 NULL  如：COALESCE (NULL,44,55)=44 |
| CASE a WHEN b THEN c [WHEN d THEN e] * [ELSE f] END | 如果 a=b 就返回 c,a=d 就返回 e，否则返回 f。如 CASE 4 WHEN 5  THEN 5 WHEN 4 THEN 4 ELSE 3 END 将返回 4 |
| CASE WHEN a THEN b [WHEN c THEN d] * [ELSE e] END | 如果 a=ture 就返回 b,c=ture 就返回 d，否则返回 e  如：CASE WHEN  5>0  THEN 5 WHEN 4>0 THEN 4 ELSE 0 END 将返回5；CASE WHEN  5<0  THEN 5 WHEN 4<0 THEN 4 ELSE 0 END 将返回0 |
| isnull(a) | 如果 a 为 null 就返回 true，否则返回 false |
| isnotnull(a) | 如果 a 为非 null 就返回 true，否则返回 false |

### 字符串函数

| 函数 | 描述 |
|----|----|
| ascii(string str) | 返回 str 中首个 ASCII 字符串的整数值 |
| base64(binary bin) | 将二进制 bin 转换成 64 位的字符串 |
| concat(string|binary A, string|binary B...) | 对二进制字节码或字符串按次序进行拼接 |
| context_ngrams(array<array>, array, int K, int pf) | 与 ngram 类似，但context_ngram()允许你预算指定上下文(数组)来去查找子序列，具体看StatisticsAndDataMining |
| concat_ws(string SEP, string A, string B...) | 与 concat()类似，但使用指定的分隔符喜进行分隔 |
| concat_ws(string SEP, array) | 拼接 Array 中的元素并用指定分隔符进行分隔 |
| decode(binary bin, string charset) | 使用指定的字符集 charset 将二进制值 bin 解码成字符串，支持的字符集有：'US-ASCII', 'ISO-8859-1', 'UTF-8', 'UTF-16BE', 'UTF-16LE', 'UTF-16'，如果任意输入参数为 NULL 都将返回 NULL |
| encode(string src, string charset) | 使用指定的字符集 charset 将字符串编码成二进制值，支持的字符集有：'US-ASCII', 'ISO-8859-1', 'UTF-8', 'UTF-16BE', 'UTF-16LE', 'UTF-16'，如果任一输入参数为 NULL 都将返回 NULL |
| find_in_set(string str, string strList) | 返回以逗号分隔的字符串中 str 出现的位置，如果参数 str 为逗号或查找失败将返回 0，如果任一参数为 NULL 将返回 NULL |
| format_number(number x, int d) | 将数值 X 转换成"#,###,###.##"格式字符串，并保留 d 位小数，如果 d 为 0，将进行四舍五入且不保留小数 |
| get_json_object(string json_string, string path) | 从指定路径上的 JSON 字符串抽取出 JSON 对象，并返回这个对象的 JSON 格式，如果输入的 JSON 是非法的将返回 NULL,注意此路径上 JSON 字符串只能由数字 字母 下划线组成且不能有大写字母和特殊字符，且 key 不能由数字开头，这是由于 Hive 对列名的限制 |
| in_file(string str, string filename) | 如果文件名为 filename 的文件中有一行数据与字符串 str 匹配成功就返回 true |
| instr(string str, string substr) | 查找字符串 str 中子字符串 substr 出现的位置，如果查找失败将返回 0，如果任一参数为 NULL 将返回 NULL，注意位置为从 1 开始的 |
| length(string A) | 返回字符串的长度 |
| locate(string substr, string str[, int pos]) | 查找字符串 str 中的 pos 位置后字符串 substr 第一次出现的位置 |
| lower(string A) lcase(string A) | 将字符串 A 的所有字母转换成小写字母 |
| lpad(string str, int len, string pad) | 从左边开始对字符串 str 使用字符串 pad 填充，最终 len 长度为止，如果字符串 str 本身长度比 len 大的话，将去掉多余的部分 |
| ltrim(string A) | 去掉字符串 A 前面的空格 |
| ngrams(array<array>, int N, int K, int pf) | 返回出现次数 TOP K 的的子序列,n 表示子序列的长度，具体看StatisticsAndDataMining |
| parse_url(string urlString, string partToExtract [, string keyToExtract]) | 返回从 URL 中抽取指定部分的内容，参数 urlString 是 URL 字符串，而参数 partToExtract 是要抽取的部分，这个参数包含(HOST, PATH, QUERY, REF, PROTOCOL, AUTHORITY, FILE, and USERINFO,例如：parse_url('http://facebook.com/path1/p.php?k1=v1&k2=v2#Ref1', 'HOST') ='facebook.com'，如果参数 partToExtract 值为 QUERY 则必须指定第三个参数 key  如：parse_url('http://facebook.com/path1/p.php?k1=v1&k2=v2#Ref1', 'QUERY', 'k1') =‘v1’ |
| printf(String format, Obj... args) | 按照 printf 风格格式输出字符串 | 
| regexp_extract(string subject, string pattern, int index) | 抽取字符串 subject 中符合正则表达式 pattern 的第 index 个部分的子字符串，注意些预定义字符的使用，如第二个参数如果使用'\s'将被匹配到s,'\s'才是匹配空格 |
| regexp_replace(string INITIAL_STRING, string PATTERN, string REPLACEMENT) | 按照 Java 正则表达式 PATTERN 将字符串 INTIAL_STRING 中符合条件的部分成 REPLACEMENT 所指定的字符串，如里 REPLACEMENT 这空的话，抽符合正则的部分将被去掉  如：regexp_replace("foobar", "oo|ar", "") = 'fb.' 注意些预定义字符的使用，如第二个参数如果使用'\s'将被匹配到s,'\s'才是匹配空格 |
| repeat(string str, int n) | 重复输出 n 次字符串 str |
| reverse(string A) | 反转字符串 |
| rpad(string str, int len, string pad) | 从右边开始对字符串 str 使用字符串 pad 填充，最终 len 长度为止，如果字符串 str 本身长度比 len 大的话，将去掉多余的部分 |
| rtrim(string A) | | 去掉字符串后面出现的空格 |
| sentences(string str, string lang, string locale) | 字符串 str 将被转换成单词数组，如：sentences('Hello there! How are you?') =( ("Hello", "there"), ("How", "are", "you") ) |
| space(int n) | 返回 n 个空格 |
| split(string str, string pat) | 按照正则表达式 pat 来分割字符串 str,并将分割后的数组字符串的形式返回 |
| str_to_map(text[, delimiter1, delimiter2])| 将字符串 str 按照指定分隔符转换成 Map，第一个参数是需要转换字符串，第二个参数是键值对之间的分隔符，默认为逗号;第三个参数是键值之间的分隔符，默认为"=" |
| substr(string|binary A, int start) substring(string|binary A, int start) | 对于字符串 A,从 start 位置开始截取字符串并返回 |
| substr(string|binary A, int start, int len) substring(string|binary A, int start, int len) | 对于二进制/字符串 A,从 start 位置开始截取长度为 length 的字符串并返回 |
| substring_index(string A, string delim, int count) | 截取第 count 分隔符之前的字符串，如 count 为正则从左边开始截取，如果为负则从右边开始截取 |
| translate(string|char|varchar input, string|char|varchar from, string|char|varchar to) | 将 input 出现在 from 中的字符串替换成 to 中的字符串 如：translate("MOBIN","BIN","M")="MOM" |
| trim(string A) | 将字符串 A 前后出现的空格去掉 |
| unbase64(string str) | 将 64 位的字符串转换二进制值 |
| upper(string A) ucase(string A) | 将字符串 A 中的字母转换成大写字母 |
| initcap(string A) | 将字符串 A 转换第一个字母大写其余字母的字符串 |
| levenshtein(string A, string B) | 计算两个字符串之间的差异大小  如：levenshtein('kitten', 'sitting') = 3 |
| soundex(string A) | 将普通字符串转换成 soundex 字符串 |

### 聚合函数

| 函数 | 描述 |
|--------------------------------------------------------|-----------------------------------------------------------------------------------------------------------|
| count(*), count(expr), count(DISTINCT expr[, expr...]) | 统计总行数，包括含有 NULL 值的行；统计提供非 NULL 的 expr 表达式值的行数；统计提供非 NULL 且去重后的 expr 表达式值的行数 |
| sum(col), sum(DISTINCT col) | | sum(col),表示求指定列的和，sum(DISTINCT col)表示求去重后的列的和 |
| avg(col), avg(DISTINCT col) | avg(col),表示求指定列的平均值，avg(DISTINCT col)表示求去重后的列的平均值 |
| min(col) | 求指定列的最小值 |
| max(col) | 求指定列的最大值 |
| variance(col), var_pop(col) | 求指定列数值的方差 |
| var_samp(col) | 求指定列数值的样本方差 |
| stddev_pop(col) | | 求指定列数值的标准偏差 |
| stddev_samp(col) | 求指定列数值的样本标准偏差 |
| covar_pop(col1, col2) | 求指定列数值的协方差 |
| covar_samp(col1, col2) | 求指定列数值的样本协方差 |
| corr(col1, col2) | 返回两列数值的相关系数 |
| percentile(BIGINT col, p) | 返回 col 的 p% 分位数 |

### 表生成函数

| 函数 | 描述 |
|----|----|
| explode(ARRAY<TYPE> a) | 对于 a 中的每个元素，将生成一行且包含该元素 |
| explode(ARRAY) | 每行对应数组中的一个元素 |
| explode(MAP) | 每行对应每个 map 键-值，其中一个字段是 map 的键，另一个字段是 map 的值 |
| posexplode(ARRAY) | 与 explode 类似，不同的是还返回各元素在数组中的位置 |
| stack(INT n, v_1, v_2, ..., v_k) | 把 M 列转换成 N 行，每行有 M/N 个字段，其中 n 必须是个常数 |
| json_tuple(jsonStr, k1, k2, ...) | 从一个 JSON 字符串中获取多个键并作为一个元组返回，与 get_json_object 不同的是此函数能一次获取多个键值 |
| parse_url_tuple(url, p1, p2, ...) | 返回从 URL 中抽取指定 N 部分的内容，参数 url 是 URL 字符串，而参数 p1,p2,....是要抽取的部分，这个参数包含 HOST, PATH, QUERY, REF, PROTOCOL, AUTHORITY, FILE, USERINFO, QUERY: |
| inline(ARRAY<STRUCT[STRUCT]>) | 将结构体数组提取出来并插入到表中 |

## 用户自定义函数


# 参考资料

* [Hive Cheet Sheet](http://hortonworks.com/wp-content/uploads/2016/05/Hortonworks.CheatSheet.SQLtoHive.pdf)
* [Hive 的内置函数](https://mp.weixin.qq.com/s?__biz=MzIwMDIzNDI2Ng==&mid=2247485016&idx=1&sn=57d5a9f9643ec98286b111a924f54208&chksm=9681022da1f68b3bbfc25658add897e3103f534bf1f4eaa3f7cf91f8277e71eba2bccae0f85f&cur_album_id=1474891417461293058&scene=189#wechat_redirect)

