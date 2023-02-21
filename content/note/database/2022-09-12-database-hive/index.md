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
  - [操作](#操作)
    - [DDL 操作](#ddl-操作)
    - [DML 操作](#dml-操作)
- [Hive 安装](#hive-安装)
  - [环境准备](#环境准备)
  - [下载安装包](#下载安装包)
  - [解压安装包](#解压安装包)
  - [核心配置](#核心配置)
  - [连接 MySQL](#连接-mysql)
  - [验证安装](#验证安装)
  - [使用 Hive](#使用-hive)
  - [使用 beeline](#使用-beeline)
  - [安装过程的问题](#安装过程的问题)
    - [问题 1](#问题-1)
    - [问题 2](#问题-2)
    - [问题 3](#问题-3)
- [Hive 基本操作](#hive-基本操作)
  - [常用命令](#常用命令)
  - [本地文件导入表](#本地文件导入表)
  - [其他操作](#其他操作)
- [Hive 属性配置](#hive-属性配置)
  - [数据仓库位置](#数据仓库位置)
  - [查询信息显示配置](#查询信息显示配置)
  - [参数配置方式](#参数配置方式)
- [Hive 数据类型](#hive-数据类型)
  - [基本数据类型](#基本数据类型)
  - [集合数据类型](#集合数据类型)
    - [数据类型介绍](#数据类型介绍)
    - [案例实操](#案例实操)
  - [数据类型转换](#数据类型转换)
- [Hive 数据组织](#hive-数据组织)
  - [存储结构](#存储结构)
  - [数据格式](#数据格式)
  - [解析数据](#解析数据)
  - [数据模型](#数据模型)
  - [元数据](#元数据)
  - [表类型](#表类型)
- [Hive 数据库](#hive-数据库)
  - [创建数据库](#创建数据库)
    - [语法](#语法)
    - [示例](#示例)
  - [删除数据库](#删除数据库)
    - [语法](#语法-1)
    - [示例](#示例-1)
  - [修改数据库](#修改数据库)
    - [语法](#语法-2)
    - [示例](#示例-2)
  - [列举数据库](#列举数据库)
  - [描述数据库](#描述数据库)
  - [切换数据库](#切换数据库)
- [Hive 表](#hive-表)
  - [创建表](#创建表)
    - [一般语法](#一般语法)
    - [CTAS](#ctas)
    - [复制表模式](#复制表模式)
  - [删除表](#删除表)
  - [清空表](#清空表)
  - [修改表和分区](#修改表和分区)
    - [修改表](#修改表)
    - [修改列](#修改列)
    - [修改分区](#修改分区)
  - [列举表信息](#列举表信息)
    - [列举数据库中的所有表](#列举数据库中的所有表)
    - [列举表的属性信息](#列举表的属性信息)
    - [列举表或分区扩展](#列举表或分区扩展)
  - [描述表信息](#描述表信息)
    - [描述表](#描述表)
    - [描述列统计](#描述列统计)
    - [描述分区](#描述分区)
- [Hive 视图](#hive-视图)
  - [创建视图](#创建视图)
  - [修改视图](#修改视图)
  - [复制视图](#复制视图)
  - [删除视图](#删除视图)
  - [列举视图](#列举视图)
  - [描述视图](#描述视图)
- [Hive 索引](#hive-索引)
  - [建立索引](#建立索引)
  - [删除索引](#删除索引)
  - [显示索引](#显示索引)
  - [修改索引](#修改索引)
  - [查看查询语句是否用到了索引](#查看查询语句是否用到了索引)
  - [定制化索引](#定制化索引)
- [Hive 宏命令](#hive-宏命令)
  - [创建宏](#创建宏)
  - [删除宏](#删除宏)
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
    - [临时函数](#临时函数)
    - [持久化函数](#持久化函数)
  - [列举函数](#列举函数)
- [Hive 操作](#hive-操作)
  - [Load Data](#load-data)
  - [Insert Data](#insert-data)
  - [Export Data](#export-data)
  - [Insert Values](#insert-values)
  - [Update](#update)
  - [Delete](#delete)
  - [Merge](#merge)
- [Hive SQL 优化](#hive-sql-优化)
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

## 操作

SQL 语言分为四大类：

* 数据查询语言 DQL：基本结构由 SELECT、FROM、WEHERE 子句构成查询块
* 数据操纵语言 DML：包括插入、更新、删除
* 数据定义语言 DDL：包括创建数据库中的对象——表、视图、索引等
* 数据控制语言 DCL：授予或者收回数据库的权限，控制或者操纵事务发生的时间及效果、对数据库进行监视等

而 HQL 中，分类如下（以 Hive 的 wiki 分类为准）

### DDL 操作

HQL DDL 语法包括：

* 创建：
    - CREATE DATABASE/SCHEMA
    - CREATE TABLE
    - CREATE VIEW
    - CREATE FUNCTION
    - CREATE INDEX
* 删除：
    - DROP DATABASE/SCHEMA
    - DROP TABLE
    - DROP VIEW
    - DROP INDEX
* 替代：
    - ALTER DATABASE/SCHEMA
    - ALTER TABLE
    - ALTER VIEW
* 清空：
    - TRUNCATE TABLE
* 修复：
    - MSCK REPAIR TABLE 
    - ALTER TABLE RECOVER PARTITIONS
* 展示：
    - SHOW DATABASES/SCHEMAS
    - SHOW TABLES
    - SHOW TBLPROPERTIES
    - SHOW VIEWS
    - SHOW PARTITIONS
    - SHOW FUNCTIONS
    - SHOW INDEX[ES]
    - SHOW COLUMNS
    - SHOW CREATE TABLE
* 描述：
    - DESCRIBE DATABASE/SCHEMA
    - DESCRIBE table_name
    - DESCRIBE view_name
    - DESCRIBE materialized_view_name

### DML 操作

HQL DML 语法包括：

* 导入：Load file to table
* 导出：Writing data into thie filesystem from queries
* 插入：Inserting data into table from queries/ SQL
* 更新：Update
* 删除：Delete
* 合并：Merge

# Hive 安装

## 环境准备

1. [搭建好 Hadoop 集群，并启动](https://blog.csdn.net/u011109589/article/details/124852278 )
2. 安装 Java，运行环境 JDK
3. 在 hadoop1 虚拟机节点上[安装 MySQL 服务](https://blog.csdn.net/Ayue1220/article/details/105284569?spm=1001.2101.3001.6650.1&utm_medium=distribute.pc_relevant.none-task-blog-2~default~CTRLIST~default-1.pc_relevant_default&depth_1-utm_source=distribute.pc_relevant.none-task-blog-2~default~CTRLIST~default-1.pc_relevant_default&utm_relevant_index=2)

## 下载安装包

```bash
$ cd /opt/module
$ wget https://mirrors.tuna.tsinghua.edu.cn/apache/hive/hive-3.1.2/apache-hive-3.1.3-bin.tar.gz
```

## 解压安装包

```bash
$ tar -zxvf apache-hive-3.1.3-bin.tar.gz
$ mv apache-hive-3.1.3-bin hive
```

## 核心配置

添加 Hive 核心配置，选择远程 MySQL 模式

```bash
$ cd /opt/module/hive/conf
$ vim hive-site.xml
```

```xml
<property>
    <name>javax.jdo.option.ConnectionURL</name>
    <value>jdbc:mysql://hadoop1:3306/hivedb?createDatabaseIfNotExist=true&amp;characterEncoding=UTF-8&amp;useSSL=false&amp;serverTimezone=GMT</value>
</property>

<property>
    <name>javax.jdo.option.ConnectionDriverName</name>
    <value>com.mysql.cj.jdbc.Driver</value>
</property>

<!-- 修改为你自己的Mysql账号 -->
<property>
    <name>javax.jdo.option.ConnectionUserName</name>
    <value>root</value>
</property>

<!-- 修改为你自己的Mysql密码 -->
<property>
    <name>javax.jdo.option.ConnectionPassword</name>
    <value>123456</value>
</property>

<!-- 忽略HIVE 元数据库版本的校验，如果非要校验就得进入MYSQL升级版本 -->
<property>
    <name>hive.metastore.schema.verification</name>
    <value>false</value>
</property>

<property> 
    <name>hive.cli.print.current.db</name>
    <value>true</value>
</property>

<property> 
    <name>hive.cli.print.header</name>
    <value>true</value>
</property>

<!-- hiveserver2 -->
<property>
    <name>hive.server2.thrift.port</name>
    <value>10000</value>
</property>

<property>
	<name>hive.server2.thrift.bind.host</name>
	<value>hadoop1</value>
</property>
```

## 连接 MySQL

下载连接 MySQL 的 JDBC 驱动包到 Hive 的 `lib` 目录下

```bash
$ cd /opt/module
# 下载 MySQL 驱动包
wget https://repo1.maven.org/maven2/mysql/mysql-connector-java/8.0.17/mysql-connector-java-8.0.17.jar
```

在 MySQL 上创建 Hive 的元数据存储库

```bash
$ create database hivedb;
```

执行 Hive 的初始化工作

```bash
$ cd /opt/module/hive/bin
$ ./schematool -initSchema -dbType mysql
```

## 验证安装

初始化完成后，在 MySQL 的 `hivedb` 数据库中查看是否初始化成功

```bash
# 若展示多个数据表，即代表初始化成功
mysql> show tables;
```

## 使用 Hive

```bash
$ cd /opt/module/hive/bin
$ ./hive
```

```bash
# 建表
create table student(id int, name string);

# 插入数据
insert into table student(1, 'abc');

# 插入成功后，查询
select * from student;
```

## 使用 beeline

1. 首先启动 `hiveserver2` 服务：

```bash
$ nohup ./bin/hiveserver2>> hiveserver2.log 2>&1 &
```

2. `hiveserver2` 服务启动后，使用 `beeline` 客户端访问 `hiveserver2` 服务：

```bash
$ cd /opt/module/hive

# 进入 beeline 客户端
$ bin/beeline

# 执行连接 hiveserver2 操作
beeline> !connect jdbc:hive2://hadoop1:10000/default

# 或者
$ bin/beeline -u jdbc:hive2://hadoop1:10000/default -n root
```

注意：Hive 的默认引擎为 MR

至此，就完成了Hive的安装。

## 安装过程的问题

### 问题 1

```
com.google.common.base.Preconditions.checkArgument(ZLjava/lang/String;Ljava/lang/Object;)
```

* 错误原因：系统找不到这个类所在的 jar 包或者 jar 包的版本不一样系统不知道使用哪个。Hive 启动报错的原因是后者
* 解决办法：
    - (1) `com.google.common.base.Preconditions.checkArgument` 这个类所在的 jar 包为：`guava.jar`
    - (2) `hadoop-3.1.3`（路径：hadoop/share/hadoop/common/lib）中该 jar 包为  `guava-27.0-jre.jar`；
      而 `hive-3.1.3`(路径：hive/lib)中该 jar 包为 `guava-19.0.jar`
    - (3）将 jar 包变成一致的版本：删除 Hive 中低版本 jar 包，将 Hadoop 中高版本的复制到 Hive 的 lib 中。
      再次启动问题得到解决

### 问题 2

```
Error: Could not open client transport with JDBC Uri: jdbc:hive2://hadoop1:10000/default: Failed to open new session: java.lang.RuntimeException: org.apache.hadoop.ipc.RemoteException(org.apache.hadoop.security.authorize.AuthorizationException): User: root is not allowed to impersonate anonymous (state=08S01,code=0)
```

* 解决办法：
    - (1）修改 Hadoop 配置文件 `/opt/module/hadoop/etc/hadoop/core-site.xml`。添加如下内容，
      然后将 `core-site.xml` 分发到集群的其他节点

    ```xml 
    <property>     
        <name>hadoop.proxyuser.root.hosts</name>     
        <value>*</value> </property> 
    <property>     
        <name>hadoop.proxyuser.root.groups</name>     
        <value>*</value> 
    </property>
    ```
    
    - (2）重启 Hadoop 集群

### 问题 3

用 `beeline` 操作 Hive 时，如何关闭打印的 info 日志信息？

* 解决办法：
    - (1）在使用 beeline 时加入以下设置即可 `hiveconf hive.server2.logging.operation.level=NONE`
    - (2）在 `hive-site.xml` 中修改如下配置也可以禁用在 `beeline` 中显示额外信息

    ```xml
    <property>
        <name>hive.server2.logging.operation.level</name>
        <value>NONE</value>
        <description>
          Expects one of [none, execution, performance, verbose].
          HS2 operation logging mode available to clients to be set at session level.
          For this to work, hive.server2.logging.operation.enabled should be set to true.
            NONE: Ignore any logging
            EXECUTION: Log completion of tasks
            PERFORMANCE: Execution + Performance logs 
            VERBOSE: All logs
        </description>
    </property>
    ```

# Hive 基本操作

## 常用命令

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

## 本地文件导入表

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

## 其他操作

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

# Hive 属性配置

## 数据仓库位置

`Default` 的数据仓库原始位置是在 HDFS 上的 `/user/hive/warehouses` 路径下，
如果某张表属于 `Default` 数据库，那么会直接在数据仓库目录创建一个文件夹。
以刚刚创建的表 `student` 为例，来查询其所在集群的位置：

```bash
hive> desc formatted student;
OK
# col_name           data_type            comment
id                   int
name                 string

# Detailed Table Information
Database:            default
OwnerType:           USER
Owner:               **
CreateTime:          Fri Jul 17 08:59:14 CST 2020
LastAccessTime:      UNKNOWN
Retention:           0
Location:            hdfs://localhost:9000/user/hive/warehouse/student
Table Type:          MANAGED_TABLE
Table Parameters:
 bucketing_version    2
 numFiles             1
 numRows              0
 rawDataSize          0
 totalSize            34
 transient_lastDdlTime 1594948899

# Storage Information
SerDe Library:       org.apache.hadoop.hive.serde2.lazy.LazySimpleSerDe
InputFormat:         org.apache.hadoop.mapred.TextInputFormat
OutputFormat:        org.apache.hadoop.hive.ql.io.HiveIgnoreKeyTextOutputFormat
Compressed:          No
Num Buckets:         -1
Bucket Columns:      []
Sort Columns:        []
Storage Desc Params:
 field.delim          \t
 serialization.format \t
Time taken: 0.099 seconds, Fetched: 32 row(s)
```

可以看到，`Table Information` 里面有一个 `Location`，表示当前表所在的位置，
因为 `student` 是 `Default` 数据仓库的，所以会在 `/user/hive/warehouse/` 路径下

如果想要修改 `Default` 数据仓库的原始位置，需要在 `hive-site.xml`(可以来自 `hive-default.xml.template`)文件下加入如下配置信息，
并修改其 `value`：

```xml
<property> 
  <name>hive.metastore.warehouse.dir</name> 
  <value>/user/hive/warehouse</value> 
  <description>location of default database for the warehouse</description> 
</property>
```

同时也需要给修改的路径配置相应的权限：

```bash
$ hdfs dfs -chmod g+w /user/hive/warehouse
```

## 查询信息显示配置

可以在 `hive-site.xml` 中配置如下信息，便可以实现显示当前数据库以及查询表的头信息：

```xml
<property> 
  <name>hive.cli.print.header</name> 
  <value>true</value> 
</property>

<property> 
  <name>hive.cli.print.current.db</name> 
  <value>true</value> 
</property>
```

当然也可以通过 `set` 命令来设置：

```bash
$ set hive.cli.print.header=true;  # 显示表头
$ set hive.cli.print.current.db=true;  # 显示当前数据库
```

看下前后对比：

```bash
# 前
hive> select * from student;
OK
1 Xiao_ming
2 xiao_hong
3 xiao_hao
Time taken: 0.231 seconds, Fetched: 3 row(s)

# 后
hive (default)>  select * from student;
OK
student.id student.name
1 Xiao_ming
2 xiao_hong
3 xiao_hao
Time taken: 0.202 seconds, Fetched: 3 row(s)
```

## 参数配置方式

可以用 `set` 查看当前所有参数配置信息，但是一般不这么做，会显示很多信息：

```bash
hive> set
```

通常配置文件有三种方式：

1. 配置文件方式
    - 默认配置文件：`hive-default.xml`
    - 用户自定义配置文件：`hive-site.xml`
    - 用户自定义配置会覆盖默认配置，配置文件的设定对本机启动的所有 Hive 进程都有效
    - 另外，Hive 也会读入 Hadoop 的配置，因为 Hive 是作为 Hadoop 的客户端启动的，Hive 的配置会覆盖 Hadoop 的配置
2. 命令行参数方式
    - 启动 Hive 时，可以在命令行添加 `-hiveconf param=value` 来设定参数，这样设置也是仅对本次 Hive 启动有效
      比如：设置 reduce 个数：

    ```bash
    $ hive -hiveconf mapred.reduce.tasks=100;
    ```

3. 参数声明方式
    - 可以在 hive cli 中通过 `set` 关键字设定参数，这样设置也是仅对本次 Hive 启动有效

    ```bash
    hive (default)> set mapred.reduce.tasks=100;
    ```

上述三种设定方式的优先级依次递增。即：配置文件<命令行参数<参数声明。
注意某些系统级的参数，例如 `log4j` 相关的设定，必须用前两种方式设定，
因为那些参数的读取在会话建立以前已经完成了

# Hive 数据类型

## 基本数据类型

| 数据类型 | Java 数据类型 | 长度 |
|----|----|----|
| TINYINT | byte | 1 byte 有符号整数 |
| SMALINT | short | 2 byte 有符号整数 |
| INT | int | 4 byte 有符号整数 |
| BIGINT | long | 8 byte 有符号整数 |
| BOOLEAN | boolean | 布尔类型，true 或者 false |
| FLOAT | float | 单精度浮点数 |
| DOUBLE | double | 双精度浮点数 |
| STRING | string | 字符系列。可以指定字符集，可以使用单引号或者双引号 |
| TIMESTAMP | | 时间类型 |
| BINARY | | 字节数组 |

Hive 的 `STRING` 类型相当于关系型数据库的 `VARCHAR` 类型，该类型是一个可变的字符串，
不过它不能声明其中最多能存储多少个字符，理论上它可以存储 2GB 的字符数

## 集合数据类型

### 数据类型介绍

| 数据类型 | 描述 | 长度 |
|----|----|----|
| STRUCT | 和 c 语言中的 struct 类似，都可以通过“点”符号访问元素内容。例如，如果某个列的数据类型是 STRUCT{first STRING, last STRING}，那么第 1 个元素可以通过字段.first 来引用 | struct() |
| MAP | MAP 是一组键-值对元组集合，使用数组表示法可以访问数据。例如，如果某个列的数据类型是 MAP，其中键->值对 是 ’first’->’John’ 和 ’last’->’Doe’，那么可以通过字段名 [‘last’] 获取最后一个元素 | map() |
| ARRAY | 数组是一组具有相同类型和名称的变量的集合。这些变量称为数组的元素，每个数组元素都有一个编号，编号从零开始。例如，数组值为 [‘John’, ‘Doe’]， 那么第 2 个元素可以通过数组名 [1] 进行引用 | Array() |

Hive 有三种复杂数据类型 `ARRAY`、`MAP`、`STRUCT`。
`ARRAY` 和 `MAP` 与 Java 中的 `Array` 和 `Map` 类似，
而 `STRUCT` 与 C 语言中的 `Struct` 类似，它封装了一个命名字段集合，
复杂数据类型允许任意层次的嵌套

### 案例实操

1. 假设某表有如下一行，我们用 JSON 格式来表示其数据结构。在 Hive 下访问的格式为：

```json
{
  "name": "songsong",
  "friends": ["bingbing" , "lili"] , //列表 Array, 
  "children": {   //键值 Map,
    "xiao song": 18 ,
    "xiaoxiao song": 19 
  },
  "address": {  //结构 Struct,
    "street": "hui long guan" ,
    "city": "beijing" 
  }
}
```

2. 基于上述数据结构，我们在 Hive 里创建对应的表，并导入数据。创建本地测试文件 `text.txt`：

```
songsong,bingbing_lili,xiao song:18_xiaoxiao song:19,hui long guan_beijing yangyang,caicai_susu,xiao yang:18_xiaoxiao yang:19,chao yang_beijing
```

注意：`MAP`，`STRUCT` 和 `ARRAY` 里的元素间关系都可以用同一个字符表示，这里用 `_`

3. Hive 上创建测试表 `test`：

```sql
create table test( 
  name string, 
  friends array<string>, 
  children map<string, int>, 
  address struct<street:string, city:string> 
)
row format delimited fields terminated by ',' 
collection items terminated by '_' 
map keys terminated by ':'
lines terminated by '\n';
```

字段解释：

* `row format delimited fields terminated by ','`：列分隔符
* `collection items terminated by '_'`：MAP STRUCT 和 ARRAY 的分隔符(数据分割符号)
* `map keys terminated by ':'`：MAP 中的 key 与 value 的分隔符
* `lines terminated by '\n'`：行分隔符

4. 导入文本数据到测试表中：

```bash
hive (default)> load data local inpath '/Users/chenze/Desktop/test.txt' into table test;
```

5. 访问三种集合列里的数据：

先查看下数据：

```bash
hive (default)> select * from test;
OK
test.name test.friends test.children test.address
songsong ["bingbing","lili"] {"xiao song":18,"xiaoxiao song":19} {"street":"hui long guan","city":"beijing yangyang"}
Time taken: 0.113 seconds, Fetched: 1 row(s)
```

查看 `ARRAY`，`MAP`，`STRUCT` 的访问方式：

```bash
hive (default)> select friends[1],children['xiao song'],address.city from test where name="songsong";
OK
_c0 _c1 city
lili 18 beijing yangyang
Time taken: 0.527 seconds, Fetched: 1 row(s)
```

## 数据类型转换

Hive 的原子数据类型是可以进行隐式转换的，类似于 Java 的类型转换，
例如某表达式使用 INT 类型，TINYINT 会自动转换为 INT 类型，
但是 Hive 不会进行反向转化，例如，某表达式使用 TINYINT 类型，
INT 不会自动转换为 TINYINT 类型，它会返回错误，除非使用 CAST 操作

1. 隐式类型转换规则如下
    - 任何整数类型都可以隐式地转换为一个范围更广的类型，如 TINYINT 可以转换 成 INT，INT 可以转换成 BIGINT
    - 所有整数类型、FLOAT 和 STRING 类型都可以隐式地转换成 DOUBLE
    - TINYINT、SMALLINT、INT 都可以转换为 FLOAT
    - BOOLEAN 类型不可以转换为任何其它的类型
2. 可以使用 CAST 操作显示进行数据类型转换
    - 例如 `CAST('1' AS INT)` 将把字符串 `'1'` 转换成整数 `1`；
      如果强制类型转换失败，如执行 `CAST('X' AS INT)`，表达式返回空值 `NULL`

# Hive 数据组织

## 存储结构

Hive 的存储结构包括：

* 数据库
* 表
* 视图
* 分区
* 表数据
* 等

数据库，表，分区等等都对 应 HDFS 上的一个目录。表数据对应 HDFS 对应目录下的文件

## 数据格式

Hive 中所有的数据都存储在 HDFS 中，没有专门的数据存储格式，
因为 Hive 是读模式（Schema On Read），
可支持 TextFile，SequenceFile，RCFile 或者自定义格式等

- TextFile：默认格式，存储方式为行存储。数据不做压缩，磁盘开销大，数据解析开销大
- SequenceFile：Hadoop API 提供的一种二进制文件支持，其具有使用方便、可分割、可压缩的特点。
  SequenceFile 支持三种压缩选择：NONE, RECORD, BLOCK。Record 压缩率低，一般建议使用 BLOCK 压缩
- RCFile：一种行列存储相结合的存储方式
- ORCFile：数据按照行分块，每个块按照列存储，其中每个块都存储有一个索引。
  Hive 给出的新格式，属于 RCFILE 的升级版，性能有大幅度提升，而且数据可以压缩存储，压缩快，且可以快速列存取
- Parquet：一种行式存储，同时具有很好的压缩性能；同时可以减少大量的表扫描和反序列化的时间

## 解析数据

只需要在创建表的时候告诉 Hive 数据中的列分隔符和行分隔符，Hive 就可以解析数据

* Hive 的默认列分隔符：控制符 `Ctrl + A`，`\x01 Hive` 的
* Hive 的默认行分隔符：换行符 `\n`

## 数据模型

Hive 中包含以下数据模型：

* database：在 HDFS 中表现为 `${hive.metastore.warehouse.dir}` 目录下一个文件夹
* table：在 HDFS 中表现所属 database 目录下一个文件夹
* external table：与 table 类似，不过其数据存放位置可以指定任意 HDFS 目录路径
* partition：在 HDFS 中表现为 table 目录下的子目录
* bucket：在 HDFS 中表现为同一个表目录或者分区目录下根据某个字段的值进行 hash 散列之后的多个文件
* view：与传统数据库类似，只读，基于基本表创建

## 元数据

Hive 的元数据存储在 RDBMS 中，除元数据外的其它所有数据都基于 HDFS 存储。
默认情 况下，Hive 元数据保存在内嵌的 Derby 数据库中，只能允许一个会话连接，
只适合简单的 测试。实际生产环境中不适用，为了支持多用户会话，则需要一个独立的元数据库，
使用 MySQL 作为元数据库，Hive 内部对 MySQL 提供了很好的支持

## 表类型

Hive 中的表分为内部表、外部表、分区表和 Bucket 表

* 内部表和外部表的区别：
   - 创建内部表时，会将数据移动到数据仓库指向的路径；创建外部表时，
     仅记录数据所在路径，不对数据的位置做出改变；
   - 删除内部表时，删除表元数据和数据；删除外部表时，删除元数据，不删除数据。
     所以外部表相对来说更加安全些，数据组织也更加灵活，方便共享源数据
   - 内部表数据由 Hive 自身管理，外部表数据由 HDFS 管理
   - 未被 external 修饰的是内部表，被 external 修饰的为外部表
   - 对内部表的修改会直接同步到元数据，而对外部表的表结构和分区进行修改，
     则需要修改 `'MSCK REPAIR TABLE [table_name]'`
* 内部表和外部表的使用选择：
   - 大多数情况，他们的区别不明显，如果数据的所有处理都在 Hive 中进行，
     那么倾向于选择内部表；但是如果 Hive 和其他工具要针对相同的数据集进行处理，外部表更合适
   - 使用外部表访问存储在 HDFS 上的初始数据，然后通过 Hive 转换数据并存到内部表中
   - 使用外部表的场景是针对一个数据集有多个不同的 Schema
   - 通过外部表和内部表的区别和使用选择的对比可以看出来，
     hive 其实仅仅只是对存储在 HDFS 上的数据提供了一种新的抽象，而不是管理存储在 HDFS 上的数据。
     所以不管创建内部表还是外部表，都可以对 hive 表的数据存储目录中的数据进行增删操作

> 使用外部表的场景是针对一个数据集有多个不同的 Schema
> 
> 通过外部表和内部表的区别和使用选择的对比可以看出来，
> hive 其实仅仅只是对存储在 HDFS 上的数据提供了一种新的抽象。
> 而不是管理存储在 HDFS 上的数据。所以不管创建内部 表还是外部表，
> 都可以对 hive 表的数据存储目录中的数据进行增删操作

* 分区表和分桶表的区别：
    - Hive 数据表可以根据某些字段进行分区操作，细化数据管理，可以让部分查询更快。
      同时表和分区也可以进一步被划分为 Buckets，分桶表的原理和 MapReduce 编程中的 HashPartitioner 的原理类似
    - 分区和分桶都是细化数据管理，但是分区表是手动添加区分，由于 Hive 是读模式，所以对添加进分区的数据不做模式校验，
      分桶表中的数据是按照某些分桶字段进行 hash 散列形成的多个文件，所以数据的准确性也高很多

# Hive 数据库

* 创建：
    - CREATE DATABASE/SCHEMA
* 删除：
    - DROP DATABASE/SCHEMA
* 替代：
    - ALTER DATABASE/SCHEMA
* 展示：
    - SHOW DATABASES/SCHEMAS
* 描述：
    - DESCRIBE DATABASE/SCHEMA

## 创建数据库

### 语法

```sql
CREATE (DATABASE|SCHEMA) [IF NOT EXISTS] database_name
  [COMMENT database_comment]
  [LOCATION hdfs_path]
  [MANAGEDLOCATION hdfs_path]
  [WITH DBPROPERTIES (property_name=property_value, ...)];
```

* `SCHEMA` 和 `DATABASE` 的用法是可互换，因为含义相同
* `IF NOT EXISTS` 最好加上，防止冲突
* `LOCATION hdfs_path` 加载 hdfs 上的数据
* `MANAGEDLOCATION` 出现在 Hive 4.0 中，指外部表的默认目录
* `WITH DBPROPERTIES` 可以设置属性和值，会存储在 MySQL 中的元数据库中

### 示例

```sql
-- 数据库所在的目录(默认数据库): /usr/hive/warehouse/financials.db
CREATE DATABASE financials;

CREATE DATABASE [IF NOT EXISTS] financials;
```

```sql
-- 数据库所在的目录: /my/preperred/directory.db
CREATE DATABASE financials
LOCATION '/my/preperred/directory';
```

```sql
-- 增加数据库描述信息
-- 数据库所在的目录: /my/preperred/directory.db
CREATE DATABASE financials
LOCATION '/my/preperred/directory'
COMMENT 'Holds all financial tables';
WITH DBPROPERTIES  ('creator'='wangzhefeng', 'date'='2018-08-11');
```

## 删除数据库

### 语法

```sql
DROP (DATABASE|SCHEMA)[IF EXISTS] database_name [RESTRICT|CASCADE];
```

* 删除数据库的默认行为是 `RESTRICT`
* 如果数据库不为空，需要添加 `CASCADE` 进行级联删除

### 示例

```sql
DROP DATABASE IF EXISTS financials;
DROP DATABASE IF EXISTS financials RESTRICT;
DROP DATABASE IF EXISTS financials CASCADE;
```

## 修改数据库

### 语法

可以修改数据库的

* 属性（property）
* 所属人（owner）
* 位置（location）
* 外部表位置（managed location）

修改位置时，并不会将数据库的当前目录的内容移动到新的位置，只是更改了默认的父目录，
在该目录中为此数据库添加新表。数据库的其他元素无法进行更改

```sql
-- 属性
ALTER (DATABASE|SCHEMA) database_name 
SET DBPROPERTIES (property_name=property_value, ...);   -- (Note: SCHEMA added in Hive 0.14.0)

-- 所属人
ALTER (DATABASE|SCHEMA) database_name 
SET OWNER [USER|ROLE] user_or_role;   -- (Note: Hive 0.13.0 and later; SCHEMA added in Hive 0.14.0)

-- 位置
ALTER (DATABASE|SCHEMA) database_name 
SET LOCATION hdfs_path; -- (Note: Hive 2.2.1, 2.4.0 and later)
 
-- 外部表位置
ALTER (DATABASE|SCHEMA) database_name 
SET MANAGEDLOCATION hdfs_path; -- (Note: Hive 4.0.0 and later)
```

### 示例

```sql
ALTER DATABASE financials 
SET DBPROPERTIES ('edited-by'='wangzhefeng');
```

## 列举数据库

Show 操作可以利用正则表达式进行过滤，而正则表达式中的通配符只能是 “*” 或 “|” 供选择

```sql
SHOW (DATABASE|SCHEMAS) [LIKE ``identifier_with_wildcards``];
```

将列出了元存储中定义的所有数据库

## 描述数据库

描述数据库，包括数据库名、注释、位置等。`EXTENDED` 还会显示了数据库属性

```sql
DESCRIBE DATABASE [EXTENDED] db_name;
DESCRIBE SCHEMA [EXTENDED] db_name;  -- (Note: Hive ``1.1``.``0` `and later)
```

## 切换数据库

切换用户当前工作数据库

```sql
USE database_name;
USE DEFAULT;
```

# Hive 表

* 创建：
    - CREATE TABLE
* 删除：
    - DROP TABLE
* 替代：
    - ALTER TABLE
* 清空：
    - TRUNCATE TABLE
* 修复：
    - MSCK REPAIR TABLE 
    - ALTER TABLE RECOVER PARTITIONS
* 展示：
    - SHOW TABLES
* 描述：
    - DESCRIBE table_name

## 创建表

### 一般语法

```sql
CREATE [TEMPORARY] [EXTERNAL] TABLE [IF NOT EXISTS] [db_name.]table_name  -- (Note: TEMPORARY available in Hive 0.14.0 and later)
  [(col_name data_type [column_constraint_specification] [COMMENT col_comment], ... [constraint_specification])]
  [COMMENT table_comment]
  [PARTITIONED BY (col_name data_type [COMMENT col_comment], ...)]
  [CLUSTERED BY (col_name, col_name, ...) [SORTED BY (col_name [ASC|DESC], ...)] INTO num_buckets BUCKETS]
  [SKEWED BY (col_name, col_name, ...)                  -- (Note: Available in Hive 0.10.0 and later)]
     ON ((col_value, col_value, ...), (col_value, col_value, ...), ...)
     [STORED AS DIRECTORIES]
  [
   [ROW FORMAT row_format] 
   [STORED AS file_format]
     | STORED BY 'storage.handler.class.name' [WITH SERDEPROPERTIES (...)]  -- (Note: Available in Hive 0.6.0 and later)
  ]
  [LOCATION hdfs_path]
  [TBLPROPERTIES (property_name=property_value, ...)]   -- (Note: Available in Hive 0.6.0 and later)
  [AS select_statement];   -- (Note: Available in Hive 0.5.0 and later; not supported for external tables)
```

* Hive 表名和列名不区分大小写，但 SerDe（序列化/反序列化） 和属性名称是区分大小写的
* `TEMPORARY`：临时表只对此次 session 有效，退出后自动删除
* `EXTERNAL`：由 HDFS 托管的外部表，不加则为由 Hive 管理的内部表
* `PARTITIONED`：分区，可以用一个或多个字段进行分区，分区的好处在于只需要针对分区进行查询，而不必全表扫描
* `CLUSTERED`：分桶，并非所有的数据集都可以形成合理的分区。
  可以对表和分区进一步细分成桶，桶是对数据进行更细粒度的划分。
  Hive 默认采用对某一列的数据进行 Hash 分桶。
  分桶实际上和 MapReduce 中的分区是一样的。分桶数和 Reduce 数对应
* `SKEWED`：数据倾斜，通过制定经常出现的值（严重倾斜），Hive 会在元数据中记录这些倾斜的列名和值，
  在 join 时能够进行优化。若是指定了 `STORED AS DIRECTORIES`，
  也就是使用列表桶（ListBucketing），Hive 会对倾斜的值建立子目录，查询会更加得到优化
* `STORED AS file_format`：文件存储类型
* `LOCATION hdfs_path`：HDFS 的位置
* `TBLPROPERTIES`：表的属性和值
* `AS select_statement`：可以设置一个代号，不支持外部表

### CTAS

CTAS：Create table as select，用查询结果来创建和填充。
CTAS 有些限制：目标表不能是分区表、不能是外部表、不能是列表桶表

语法：

```sql
CREATE [TEMPORARY] [EXTERNAL] TABLE [IF NOT EXISTS] [db_name.]table_name
SELECT *
FROM table_name_other;
```

### 复制表模式

可以从已有的数据中进行复制，使用 `LIKE` 字段。但只会复制另一张表的模式, 数据不会复制

语法：

```sql
CREATE [TEMPORARY] [EXTERNAL] TABLE [IF NOT EXISTS] [db_name.]table_name
LIKE existing_table_or_view_name
[LOCATION hdfs_path];
```

## 删除表

删除表的元数据和数据

* 管理表: 元数据信息和表内数据都会删除
* 外部表: 删除元数据信息

如果加 `PURGE` 字段，则数据不会转移到 `.Trash/Current` 目录下，因此，误操作后将无法恢复

```sql
DROP TABLE [IF EXISTS] table_name [PURGE];  
-- (Note: PURGE available in Hive 0.14.0 and later)
```

## 清空表

清空表或分区（一个或多个分区）的所有行

```sql
TRUNCATE [TABLE] table_name [PARTITION partition_spec];
-- partition_spec: (partition_column = partition_col_value, partition_column = partition_col_value, ...)
```

## 修改表和分区

只修改元数据信息

### 修改表

修改表名：

```sql 
ALTER TABLE table_name
RENAME TO new_table_name;
```

修改表属性：

```sql
ALTER TABLE table_name
SET TBLPROPERTIES table_properties;
-- table_properties: (property_name = property_value, property_name = property_value, ... )
```

修改表注释：

```sql
ALTER TABLE table_name
SET TBLPROPERTIES('comment' = new_comment);
```

添加 SerDe 属性：

```sql
ALTER TABLE table_name [PARTITION partition_spec] 
SET SERDE serde_class_name [WITH SERDEPROPERTIES serde_properties];

ALTER TABLE table_name [PARTITION partition_spec] 
SET SERDEPROPERTIES serde_properties;
-- serde_properties: (property_name = property_value, property_name = property_value, ... )

--- Hive 4.0 支持删除 SerDe 属性：
ALTER TABLE table_name [PARTITION partition_spec] UNSET SERDEPROPERTIES (property_name, ... );
```

更改表存储属性：

```sql
ALTER TABLE table_name CLUSTERED BY (col_name, col_name, ...) [SORTED BY (col_name, ...)]
  INTO num_buckets BUCKETS;
```

修改表倾斜：

```sql
ALTER TABLE table_name SKEWED BY (col_name1, col_name2, ...)
  ON ([(col_name1_value, col_name2_value, ...) [, (col_name1_value, col_name2_value), ...]
  [STORED AS DIRECTORIES];
```

更改表不倾斜：

```sql
ALTER TABLE table_name NOT SKEWED;
```

更改表未存储为目录：

```sql
ALTER TABLE table_name NOT STORED AS DIRECTORIES;
```

更改表的约束：

```sql
ALTER TABLE table_name 
ADD CONSTRAINT constraint_name PRIMARY KEY (column, ...) 
DISABLE NOVALIDATE;

ALTER TABLE table_name 
ADD CONSTRAINT constraint_name FOREIGN KEY (column, ...) 
REFERENCES table_name(column, ...) DISABLE NOVALIDATE RELY;

ALTER TABLE table_name 
ADD CONSTRAINT constraint_name UNIQUE (column, ...) 
DISABLE NOVALIDATE;

ALTER TABLE table_name 
CHANGE COLUMN column_name column_name data_type 
CONSTRAINT constraint_name NOT NULL ENABLE;

ALTER TABLE table_name 
CHANGE COLUMN column_name column_name data_type 
CONSTRAINT constraint_name DEFAULT default_value ENABLE;

ALTER TABLE table_name 
CHANGE COLUMN column_name column_name data_type 
CONSTRAINT constraint_name CHECK check_expression ENABLE;
 
ALTER TABLE table_name 
DROP CONSTRAINT constraint_name;
```

### 修改列

更改列名称/类型/位置/注释：

```sql
ALTER TABLE table_name 
  [PARTITION partition_spec] 
  CHANGE [COLUMN] col_old_name col_new_name column_type
  [COMMENT col_comment] [FIRST|AFTER column_name] [CASCADE|RESTRICT];
```

添加/替换列：

```sql
ALTER TABLE table_name 
  [PARTITION partition_spec]  -- (Note: Hive 0.14.0 and later)
  ADD|REPLACE COLUMNS (col_name data_type [COMMENT col_comment], ...)
  [CASCADE|RESTRICT]  -- (Note: Hive 1.1.0 and later)
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

### 修改分区

添加分区：

```sql
ALTER TABLE table_name ADD [IF NOT EXISTS] 
PARTITION partition_spec 
[LOCATION 'location'][, PARTITION partition_spec [LOCATION 'location'], ...];
 
-- partition_spec: (partition_column = partition_col_value, partition_column = partition_col_value, ...)
```

重命名分区：

```sql
ALTER TABLE table_name 
PARTITION partition_spec RENAME TO PARTITION partition_spec;
```

交换分区：

```sql
-- Move partition from table_name_1 to table_name_2
ALTER TABLE table_name_2 EXCHANGE 
PARTITION (partition_spec) WITH TABLE table_name_1;

-- multiple partitions
ALTER TABLE table_name_2 EXCHANGE 
PARTITION (partition_spec, partition_spec2, ...) WITH TABLE table_name_1;
```

恢复分区：

```sql
MSCK [REPAIR] TABLE table_name [ADD/DROP/SYNC PARTITIONS];
```

如果新的分区被直接加入到 HDFS（比如 hadoop fs -put），或从 HDFS 移除，metastore 并将不知道这些变化，
除非用户在分区表上每次新添或删除分区时分别运行 ALTER TABLE table_name ADD/DROP PARTITION 命令。
可以运行恢复分区来进行维修

删除分区：

```sql
ALTER TABLE table_name 
DROP [IF EXISTS] PARTITION partition_spec[, PARTITION partition_spec, ...]
[IGNORE PROTECTION] [PURGE];
```

## 列举表信息

### 列举数据库中的所有表

Show 操作可以利用正则表达式进行过滤，而正则表达式中的通配符只能是 “*”或 “|” 供选择

```sql
SHOW TABLES [IN database_name] 
[LIKE ``'identifier_with_wildcards'``];
```

### 列举表的属性信息

```sql
SHOW TBLPROPERTIES tblname;
SHOW TBLPROPERTIES tblname(``"foo"``);
```

### 列举表或分区扩展

```sql
SHOW TABLE EXTENDED [IN|FROM database_name] 
LIKE ``'identifier_with_wildcards'`` 
[PARTITION(partition_spec)];
```

## 描述表信息

### 描述表

```sql
DESCRIBE [EXTENDED|FORMATTED] 
table_name[.col_name ( [.field_name] | [.``'$elem$'``] | [.``'$key$'``] | [.``'$value$'``])*];
```

### 描述列统计

```sql
DESCRIBE FORMATTED [db_name.]table_name column_name;
DESCRIBE FORMATTED [db_name.]table_name column_name PARTITION (partition_spec);
```

### 描述分区

```sql
DESCRIBE [EXTENDED|FORMATTED] table_name[.column_name] PARTITION partition_spec;
```


# Hive 视图

* 创建：
    - CREATE VIEW
* 删除：
    - DROP VIEW
* 替代：
    - ALTER VIEW
* 展示：
    - SHOW VIEWS
* 描述：
    - DESCRIBE view_name

## 创建视图

* 视图是纯逻辑对象，没有相关的存储
* 如果视图的定义 `SELECT` 表达式无效，则 `CREATE VIEW` 语句将失败
* 视图只读，不能用作 `LOAD/INSERT/ALTER` 的目标
* 视图可能包含 `ORDER BY` 和 `LIMIT` 子句

```sql
CREATE VIEW [IF NOT EXISTS] [db_name.]view_name 
[(column_name [COMMENT column_comment], ...)]
[COMMENT view_comment]
[TBLPROPERTIES (property_name = property_value, ...)]
AS SELECT ...;
```

## 修改视图

* 视图是只读的，视图不能进行 `insert`, `load`
* 视图是只读的，只允许修改元数据信息

```sql
ALTER VIEW [db_name.]view_name 
SET TBLPROPERTIES table_properties;
-- table_properties: (property_name = property_value, propert
```

## 复制视图

```sql
CREATE VIEW [db_name.]view_name
LIKE table_name;
```

```sql
ALTER VIEW [db_name.].view_name 
AS select_statement;
```

## 删除视图

```sql
DROP VIEW [IF EXISTS] [db_name.]view_name;
```

## 列举视图

```sql
SHOW VIEWS [IN/FROM database_name] 
[LIKE ``'pattern_with_wildcards'``];
```

## 描述视图

显示视图的元数据信息

```sql
DESCRIBE [EXTENDED|FORMATTED] 
table_name[.col_name ([.field_name] | [.``'$elem$'``] | [.``'$key$'``] | [.``'$value$'``])*];
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

使用给定的列作为键在表上创建索引

```sql
CREATE INDEX index_name
ON TABLE base_table_name (col_name, ...)
AS index_type
[WITH DEFERRED REBUILD]
[IDXPROPERTIES (property_name=property_value, ...)]
[IN TABLE index_table_name]
[
    [ ROW FORMAT ...] STORED AS ...
    | STORED BY ...
]
[LOCATION hdfs_path]
[TBLPROPERTIES (...)]
[COMMENT "index comment"];
```

## 删除索引

```sql
DROP INDEX [IF EXISTS] index_name
ON table_name;
```

## 显示索引

```sql
SHOW [FORMATTED] INDEX ON table_name;
SHOW [FORMATTED] INDEXES ON table_name;
```

## 修改索引

`REBUILD` 为使用 `WITH DEFERRED REBUILD` 子句的索引建立索引或重建先前建立的索引。
如果指定分区，那么只有该分区重建

```sql
ALTER INDEX index_name
ON table_name
[PARTITION partition_spec] REBUILD;
```

## 查看查询语句是否用到了索引

```sql
EXPLAIN ...;
```

## 定制化索引

* https://cwiki.apache.org/confluence/display/Hive/IndexDe#CREATE_INDEX

# Hive 宏命令

宏命令，与 Java 中的宏一致

## 创建宏

```sql
CREATE TEMPORARY MACRO macro_name([col_name col_type, ...]) expression;
```

举个例子：

```sql
CREATE TEMPORARY MACRO fixed_number() 42;
CREATE TEMPORARY MACRO string_len_plus_two(x string) length(x) + 2;
CREATE TEMPORARY MACRO simple_add (x int, y int) x + y;
```

宏的有效期存在于该 Session 内

## 删除宏

```sql
DROP TEMPORARY MACRO [IF EXISTS] macro_name;
```

# Hive 函数

## 函数基础

函数类型: 

* 内置函数
* 用户自定义函数(加载)

列举当前会话中所加载的所有函数名称：

```sql
SHOW FUNCTIONS [LIKE ``"<pattern>"``];
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

### 临时函数

创建和删除临时函数

```sql
CREATE TEMPORARY FUNCTION function_name AS class_name;
DROP TEMPORARY FUNCTION [IF EXISTS] function_name;
```

### 持久化函数

在 Hive 0.13 或更高版本中，函数可以注册到 metastore，
这样就可以在每次查询中进行引用，而不需要每次都创建临时函数

创建和删除永久函数：

```sql
CREATE FUNCTION [db_name.]function_name AS class_name
``[USING JAR|FILE|ARCHIVE ``'file_uri'` `[, JAR|FILE|ARCHIVE ``'file_uri'``] ];
 
DROP FUNCTION [IF EXISTS] function_name;
```

重载函数：

```sql
RELOAD (FUNCTIONS|FUNCTION);
```

## 列举函数

```sql
SHOW FUNCTIONS [LIKE ``"<pattern>"``];
```

# Hive 操作

## Load Data

在将数据加载到表中时，Hive 不执行任何转换。Load 操作是纯复制/移动操作，
仅将数据文件移动到与 Hive 表对应的位置

```sql
LOAD DATA [LOCAL] INPATH ``'filepath'` `[OVERWRITE] INTO TABLE tablename [PARTITION (partcol1=val1, partcol2=val2 ...)]

LOAD DATA [LOCAL] INPATH ``'filepath'` `[OVERWRITE] INTO TABLE tablename [PARTITION (partcol1=val1, partcol2=val2 ...)] [INPUTFORMAT ``'inputformat'` `SERDE ``'serde'``] (``3.0` `or later)
```

* `filepath` 可以是绝对路径也可以是相对路径，也可以是一个 URI
* 加载到目标可以是一个表或一个分区。如果是分区表，则必须制定所有分区列的值来确定加载特定分区；
* `filepath` 可以是文件，也可以是目录
* 制定 LOCAL 可以加载本地文件系统，否则默认为 HDFS；
* 如果使用了 OVERWRITE，则原内容将被删除；否则，将直接追加数据。

Hive 3.0 开始支持 Load 操作。

举例子：

```sql
CREATE TABLE tab1 (col1 ``int``, col2 ``int``) 
PARTITIONED BY (col3 ``int``) STORED AS ORC;
LOAD DATA LOCAL INPATH ``'filepath'` `INTO TABLE tab1;
```

## Insert Data

将查询数据插入到 Hive 表中

```sql
-- 标准语法:
INSERT OVERWRITE TABLE tablename1 [PARTITION (partcol1=val1, partcol2=val2 ...) [IF NOT EXISTS]] select_statement1 FROM from_statement;
INSERT INTO TABLE tablename1 [PARTITION (partcol1=val1, partcol2=val2 ...)] select_statement1 FROM from_statement;

-- Hive 扩展(多表插入模式):
FROM from_statement
INSERT OVERWRITE TABLE tablename1 [PARTITION (partcol1=val1, partcol2=val2 ...) [IF NOT EXISTS]] select_statement1
[INSERT OVERWRITE TABLE tablename2 [PARTITION ... [IF NOT EXISTS]] select_statement2]
[INSERT INTO TABLE tablename2 [PARTITION ...] select_statement2] ...;

FROM from_statement
INSERT INTO TABLE tablename1 [PARTITION (partcol1=val1, partcol2=val2 ...)] select_statement1
[INSERT INTO TABLE tablename2 [PARTITION ...] select_statement2]
[INSERT OVERWRITE TABLE tablename2 [PARTITION ... [IF NOT EXISTS]] select_statement2] ...;

-- Hive 扩展 (动态分区插入模式):
INSERT OVERWRITE TABLE tablename PARTITION (partcol1[=val1], partcol2[=val2] ...) select_statement FROM from_statement;
INSERT INTO TABLE tablename PARTITION (partcol1[=val1], partcol2[=val2] ...) select_statement FROM from_statement;
```

* INSERT OVERWRITE 将覆盖在表或分区的任何现有数据
* INSERT INTO将追加到表或分区，保留原有数据不变
* 插入目标可以是一个表或分区。如果是分区表，则必须由设定所有分区列的值来指定表的特定分区；
* 可以在同一个查询中指定多个INSERT子句(也称为多表插入)。多表插入可使数据扫描所需的次数最小化。
  通过对输入数据只扫描一次(并应用不同的查询操作符)，Hive可以将数据插入多个表中
* 如果给出分区列值，我们将其称为静态分区，否则就是动态分区

## Export Data

将查询数据写入到文件系统中

```sql
-- 标准语法:
INSERT OVERWRITE [LOCAL] DIRECTORY directory1
  [ROW FORMAT row_format] [STORED AS file_format] (Note: Only available starting with Hive 0.11.0)
  SELECT ... FROM ...

-- Hive 扩展 (多表插入):
FROM from_statement
INSERT OVERWRITE [LOCAL] DIRECTORY directory1 select_statement1
[INSERT OVERWRITE [LOCAL] DIRECTORY directory2 select_statement2] ...

row_format
  : DELIMITED [FIELDS TERMINATED BY char [ESCAPED BY char]] [COLLECTION ITEMS TERMINATED BY char]
        [MAP KEYS TERMINATED BY char] [LINES TERMINATED BY char]
        [NULL DEFINED AS char] (Note: Only available starting with Hive 0.13)
```

* 目录可以是一个完整的 URI
* 使用 LOCAL，可以将数据写入到本地文件系统的目录上
* 写入文件系统的数据被序列化为由 ^A 做列分割符，换行做行分隔符的文本。
  如果任何列都不是原始类型（而是 MAP、ARRAY、STRUCT、UNION），
  则这些列被序列化为 JSON 格式
* 可以在同一查询中，INSERT OVERWRITE 到目录，到本地目录和到表（或分区）
* INSERT OVERWRITE 语句是 Hive 提取大量数据到 HDFS 文件目录的最佳方式。
  Hive 可以从 map-reduce 作业中的并行写入 HDFS 目录

## Insert Values

直接从 SQL 将数据插入到表中

```sql
--标准语法
-- 此处的 values_row is: (value [, value ...])
-- 此处的 value 或者是 NULL 或者是任何有效的 sql 表达式
INSERT INTO TABLE tablename 
[PARTITION (partcol1[=val1], partcol2[=val2] ...)] 
VALUES values_row [, values_row ...];
```

* 在 `VALUES` 子句中列出的每一行插入到表 `tablename` 中
* 以 `INSERT ... SELECT` 同样的方式，来支持动态分区
* 不支持 `INSERT INTO VALUES` 子句将数据插入复杂的数据类型（数组、映射、结构、联合）列中

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

## Update

```sql
UPDATE tablename SET column = value [, column = value ...] [WHERE expression]
```

* 被引用的列必须是被更新表中的列
* 设置的值必须是 Hive Select 子句中支持的表达式。算术运算符，UDF，转换，文字等，是支持的，子查询是不支持的
* 只有符合 WHERE 子句的行才会被更新
* 分区列不能被更新
* 分桶列不能被更新

## Delete

只有符合 `WHERE` 子句的行会被删除

```sql
DELETE FROM tablename [WHERE expression]
```

## Merge

* Merge 允许根据与源表 Join 的结果对目标表执行操作
* on 语句会对源与目标进行检查，此计算开销很大

```sql
-- 标准语法
MERGE INTO <target table> AS T USING <source expression/table> AS S
ON <boolean expression1>
WHEN MATCHED [AND <boolean expression2>] THEN UPDATE SET <set clause list>
WHEN MATCHED [AND <boolean expression3>] THEN DELETE
WHEN NOT MATCHED [AND <boolean expression4>] THEN INSERT VALUES<value list>
```


# Hive SQL 优化

* [Hive SQL 优化](https://zhuanlan.zhihu.com/p/320515172)

# 参考资料

* [Apache Hive](https://hive.apache.org/)
* [Hive Cheet Sheet](http://hortonworks.com/wp-content/uploads/2016/05/Hortonworks.CheatSheet.SQLtoHive.pdf)
* [Hive 的基本认识](https://mp.weixin.qq.com/s?__biz=MzIwMDIzNDI2Ng==&mid=2247484971&idx=1&sn=c1f5da9d37a4754c7dbbd7a8589ca6a9&chksm=9681025ea1f68b48595d7632cb5051f3cee016ac3719e973b82de118146b039150136e58806b&scene=178&cur_album_id=1474891417461293058#rd)
* [Hive 的内置函数](https://mp.weixin.qq.com/s?__biz=MzIwMDIzNDI2Ng==&mid=2247485016&idx=1&sn=57d5a9f9643ec98286b111a924f54208&chksm=9681022da1f68b3bbfc25658add897e3103f534bf1f4eaa3f7cf91f8277e71eba2bccae0f85f&cur_album_id=1474891417461293058&scene=189#wechat_redirect)
* [Hive DDL 与 DML 操作](https://mp.weixin.qq.com/s?__biz=MzIwMDIzNDI2Ng==&mid=2247485012&idx=1&sn=8e65bf804c926cf1a913300473149c85&chksm=96810221a1f68b37db8acceea5feac7e8079ed222d34c3a24205ec3d05675c889b163cd8fe6a&scene=178&cur_album_id=1528340767222628353#rd)

