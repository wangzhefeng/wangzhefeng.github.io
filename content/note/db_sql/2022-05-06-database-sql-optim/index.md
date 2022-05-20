---
title: 数据库操作优化
author: 王哲峰
date: '2022-05-06'
slug: database-sql-optim
categories:
  - database
tags:
  - sql
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

- [数据库](#数据库)
- [数据库优化](#数据库优化)
  - [数据库引擎优化](#数据库引擎优化)
  - [表结构优化](#表结构优化)
    - [常识](#常识)
    - [优化](#优化)
  - [SQL语句优化](#sql语句优化)
    - [常识](#常识-1)
    - [优化](#优化-1)
</p></details><p></p>


# 数据库

- MySQL
- Oracle
- Sql Server

# 数据库优化

1. 数据库引擎优化
2. 表结构优化
   - 索引
   - 游标
   - 临时表
3. SQL语句优化

## 数据库引擎优化



## 表结构优化


### 常识

### 优化

1. 在 `where`, `order by` 涉及的字段上建立索引
2. 索引并不是越多越好, 索引固然可以提高相应的 select 的效率, 
   但同时也降低了 insert 及 update 的效率, 因为 insert 或 update 时有可能会重建索引, 
   所以怎样建索引需要慎重考虑, 视具体情况而定. 一个表的索引数最好不要超过 6 个, 
   若太多则应考虑一些不常使用到的列上建的索引是否有必要

## SQL语句优化

### 常识

* 定长字段中 NULL 也占空间

### 优化

* WHERE 语句
   1. 尽量避免全表扫描
      - 措施: 用 `where` 语句进行筛选
   2. 尽量避免在 `where` 子句中对字段进行 `is null` 判断
      - 后果: 将导致引擎放弃使用索引而进行全表扫描; 
      - 措施: 最好不要给数据库留NULL, 尽可能使用NOT NULL填充数据库; 
   3. 尽量避免在 `where` 子句中使用 `!=` 操作符
      - 后果: 将导致引擎放弃使用索引而进行全表扫描; 
   4. 尽量避免在 `where` 子句中使用 `or` 连接条件
      - 后果: 如果一个字段有索引, 一个字段没有索引, 将导致引擎放弃使用索引而进行全表扫描; 
      - 措施: 使用 `union all`
   5. 尽量避免在 `where` 子句中使用 `in` 和 `not in`
      - 后果: 将导致引擎放弃使用索引而进行全表扫描; 
      - 措施: 
         - 连续值使用 `>, <, >=, <=` 或者 `between` ; 
         - 用 `exists` 代替 `in` , 或用 `not exists` 代替 `not in` ; 
   6. 尽量避免在 `where` 子句中使用参数
      - 后果: 全表扫描; 
      - 措施: 强制查询使用索引;  `select * from t with(index(索引名)) where num = @num` ; 
   7. 尽量避免在 `where` 子句中 `=` 左边对字段进行函数、算数运算、或其他表达式操作
      - 后果: 将导致引擎放弃使用索引而进行全表扫描; 
      - 措施: 对表达式进行转换;  `where num / 2 = 100` => `where num = 100 * 2` ; 
   8. 尽量避免在 `where` 子句中对字段进行函数操作
      - 后果: 将导致引擎放弃使用索引而进行全表扫描; 
      - 措施: 使用其他筛选条件
- JOIN
   1. 对多张表进行 `join` 操作, 要先分页再join
- 字段类型
   1. 尽可能的使用 varchar/nvarchar 代替 char/nchar, 因为首先变长字段存储空间小, 
      可以节省存储空间, 其次对于查询来说, 在一个相对较小的字段内搜索效率显然要高些
   2. 尽量使用数字型字段, 若只含数值信息的字段尽量不要设计为字符型, 这会降低查询和连接的性能, 并会增加存储开销. 这是因为引擎在处理查询和连
      接时会逐个比较字符串中每一个字符, 而对于数字型而言只需要比较一次就够了
- 索引使用
   1. 在使用索引字段作为条件时, 如果该索引是复合索引, 
      那么必须使用到该索引中的第一个字段作为条件时才能保证系统使用该索引, 
      否则该索引将不会被使用, 并且应尽可能的让字段顺序与索引顺序相一致
   2. 应尽可能的避免更新 clustered 索引数据列, 因为 clustered 索引数据列的顺序就是表记录的物理存储顺序, 
      一旦该列值改变将导致整个表记录的顺序的调整, 会耗费相当大的资源. 若应用系统需要频繁更新 clustered 索引数据列, 
      那么需要考虑是否应将该索引建为 clustered 索引
- 游标使用
   1. 尽量避免使用游标, 因为游标的效率较差
- 临时表使用
   1. 如果使用到了临时表, 在存储过程的最后务必将所有的临时表显式删除, 先 truncate table, 
      然后 drop table , 这样可以避免系统表的较长时间锁定
   2. 在新建临时表时, 如果一次性插入数据量很大, 那么可以使用 select into
      代替 create table, 避免造成大量 log, 以提高速度; 如果数据量不大, 
      为了缓和系统表的资源, 应先 create table, 然后 insert
   3. 避免频繁创建和删除临时表, 以减少系统表资源的消耗. 临时表并不是不可使用, 
      适当地使用它们可以使某些例程更有效, 例如, 当需要重复引用大型表或常用表中的某个数据集时. 
      但是, 对于一次性事件, 最好使用导出表
- 表变量使用
   1. test
- 存储过程
   1. test

