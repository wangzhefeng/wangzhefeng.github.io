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
- [Hive 安装](#hive-安装)
- [Hive 数据类型](#hive-数据类型)
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
- [参考资料](#参考资料)
</p></details><p></p>

# Hive 简介

Hive 是基于 Hadoop 的一个数据仓库工具。Hive 的计算基于 Hadoop 实现的一个特别的计算模型 MapReduce，
它可以将计算任务分割成多个处理单元，然后分散到一群家用或服务器级别的硬件机器上，降低成本并提高水平扩展性

Hive 的数据存储在 Hadoop 一个分布式文件系统上，即 HDFS，运行在 Yarn 上。
Hive 把表和字段转换成 HDFS 中的文件夹和文件，并将这些元数据保持在关系型数据库中，
如 derby 或 MySQL。Hive 适合做离线数据分析，如：批量处理和延时要求不高场景

需明确的是，Hive 作为数仓应用工具，对比 RDBMS(关系型数据库) 有三个“不能”：

* 不能像 RDBMS 一般实时响应，Hive 查询延时大
* 不能像 RDBMS 做事务型查询，Hive 没有事务机制
* 不能像 RDBMS 做行级别的变更操作(包括插入、更新、删除)

另外，Hive 相比 RDBMS 是一个更“宽松”的世界，比如：

* Hive 没有定长的 `VARCHAR`  这种类型，字符串都是 `STRING`
* Hive 是读时模式，它在保存表数据时不会对数据进行校验，
  而是在读数据时校验不符合格式的数据设置为 `NULL`

# Hive 安装

* TODO

# Hive 数据类型

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
SHOW DATABASES LIKE 'h.*';	-- 正则表达式
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

函数类型: 

* 内置函数
* 用户自定义函数(加载)

列举当前会话中所加载的所有函数名称

```sql
SHOW FUNCTIONS;
```

查看函数文档

```sql
DESCRIBE FUNCTION concat;
DESCRIBE FUNCTION EXTENDED concat;
```

调用函数

```sql
SELECT concat(column1, column2) AS x 
FROM table_name;
```

# 参考资料

* [Hive Cheet Sheet](http://hortonworks.com/wp-content/uploads/2016/05/Hortonworks.CheatSheet.SQLtoHive.pdf)

