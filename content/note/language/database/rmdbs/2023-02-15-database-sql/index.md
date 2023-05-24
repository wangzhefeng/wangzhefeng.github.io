---
title: SQL 技术
author: 王哲峰
date: '2023-02-15'
slug: database-sql
categories:
  - database
tags:
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
img {
    pointer-events: none;
}
</style>

<details><summary>目录</summary><p>

- [MySQL 基本操作](#mysql-基本操作)
  - [实验描述](#实验描述)
  - [具体操作](#具体操作)
  - [操作总结](#操作总结)
- [SQL 执行顺序](#sql-执行顺序)
- [SQL 常用模型](#sql-常用模型)
  - [CASE WHEN](#case-when)
    - [常用形式](#常用形式)
    - [行列转换](#行列转换)
    - [多指标数据](#多指标数据)
    - [多条件统计](#多条件统计)
  - [DISTINCT](#distinct)
  - [GROUP BY](#group-by)
  - [HAVING](#having)
  - [ORDER BY](#order-by)
  - [聚合函数](#聚合函数)
    - [SUM 与 CASE WHEN 的结合](#sum-与-case-when-的结合)
    - [COUNT 与 CASE WHEN 的结合](#count-与-case-when-的结合)
  - [topN](#topn)
    - [每组最大的 N 条记录](#每组最大的-n-条记录)
  - [SQL 优化](#sql-优化)
- [SQL 函数](#sql-函数)
  - [Oracle](#oracle)
    - [NVL](#nvl)
    - [链接符](#链接符)
    - [参考](#参考)
  - [MySQL](#mysql)
- [SQL 题目](#sql-题目)
  - [活动运营](#活动运营)
  - [用户行为分析](#用户行为分析)
  - [用户新增留存分析](#用户新增留存分析)
  - [SQL 求分位数](#sql-求分位数)
- [参考](#参考-1)
</p></details><p></p>

# MySQL 基本操作

## 实验描述

1. 创建数据库 `Market`
2. 创建表 `customers`，结构如下
3. 将 `c_contact` 字段插入到 `c_birth` 后面
4. 将 `c_name` 字段类型修改为 `varchar(70)`
5. 将 `c_contact` 字段改名 `c_phone`
6. 增加 `c_gender` 字段，数据类型 `char(1)`
7. 将表名修改为 `customers_info`
8. 删除字段 `c_city`
9. 修改数据表的存储引擎为 `mysiam`
10. 创建另一表格 `others`，关联 `customers_info` 的 `c_num`
11. 删除 `others` 表格

## 具体操作

登录 MySQL：

```bash
$ mysql -uroot -p123456
```

查看数据库：

```bash
mysql> show databases;
+--------------------+
| Database           |
+--------------------+
| information_schema |
| Market             |
| mysite_db          |
| mysql              |
| performance_schema |
| sys                |
| wangzf             |
+--------------------+
7 rows in set (0.00 sec)
```

创建数据库 `Market`：

```bash
mysql> create database Market;
Query OK, 1 row affected (0.01 sec)

mysql> use Market;
Database change
```

创建表 `customers`：

```bash
mysql> create table customers (
           c_num int(11) primary key auto_increment not null,
           c_name varchar(50) default null,
           c_contact varchar(50) default null,
           c_city varchar(50) default null,
           c_birth datetime not null
       );
Query OK, 0 rows affected, 1 warning (0.01 sec)

mysql> desc customers;
+-----------+-------------+------+-----+---------+----------------+
| Field     | Type        | Null | Key | Default | Extra          |
+-----------+-------------+------+-----+---------+----------------+
| c_num     | int(11)     | NO   | PRI | NULL    | auto_increment |
| c_name    | varchar(50) | YES  |     | NULL    |                |
| c_contact | varchar(50) | YES  |     | NULL    |                |
| c_city    | varchar(50) | YES  |     | NULL    |                |
| c_birth   | datetime    | NO   |     | NULL    |                |
+-----------+-------------+------+-----+---------+----------------+
4 rows in set (0.01 sec)
```

将 `c_contact` 字段插入到 `c_birth` 后面

```bash
mysql> alter table customers modify c_contact varchar(50) after c_birth;
Query OK, 0 rows affected (0.01 sec)
Records: 0  Duplicates: 0  Warnings: 0
```

将 `c_name` 字段类型修改为 `varchar(70)`：

```bash
mysql> alter table customers modify c_name varchar(70);
Query OK, 0 rows affected (0.02 sec)
Records: 0  Duplicates: 0  Warnings: 0
```

将 `c_contact` 字段改名 `c_phone`：

```bash
mysql> alter table customers change c_contact c_phone varchar(50);
Query OK, 0 rows affected (0.01 sec)
Records: 0  Duplicates: 0  Warnings: 0
```

增加 `c_gender` 字段，数据类型 `char(1)`：

```bash
mysql> alter table customers add c_gender char(1);
Query OK, 0 rows affected (0.01 sec)
Records: 0  Duplicates: 0  Warnings: 0
```

将表名修改为 `customers_info`

```bash
mysql> alter table customers rename customers_info;
Query OK, 0 rows affected (0.00 sec)
```

删除字段 `c_city`：

```bash
mysql> alter table customers_info drop c_city;
Query OK, 0 rows affected (0.01 sec)
Records: 0  Duplicates: 0  Warnings: 0
```

创建另一表格 `others`，关联 `customers_info` 的 `c_num`：

```bash
mysql> create table others (id int(11), name varchar(40), foreign key (id) references customers_info(c_num));
Query OK, 0 rows affected, 1 warning (0.01 sec)

mysql> desc others;
+-------+-------------+------+-----+---------+-------+
| Field | Type        | Null | Key | Default | Extra |
+-------+-------------+------+-----+---------+-------+
| id    | int(11)     | YES  | MUL | NULL    |       |
| name  | varchar(40) | YES  |     | NULL    |       |
+-------+-------------+------+-----+---------+-------+
2 rows in set (0.00 sec)
```

修改数据表的存储引擎为 `innodb`：

```bash
mysql> alter table customers_info engine=innodb;
Query OK, 0 rows affected (0.02 sec)
Records: 0  Duplicates: 0  Warnings: 0
```

删除 `others` 表格：

```bash
mysql> drop table others;
```

## 操作总结

* 修改表名用 `rename`
* 修改字段数据类型用 `modify`
* 调整字段顺序也用 `modify`
* 同时修改字段名和字段类型用 `change`
    - 修改字段名
    - 同时修改字段名和字段类型 

```bash
$ Alter table <表名> change <旧字段名><新字段名><新数据类型>;
```

* 添加字段用 `add`，默认排在最后一行
    - `first`：添加字段在第一行
    - `after`：在指定列后加上新列
* 删除列，用 `drop`，只能一个个删除

# SQL 执行顺序

一般执行顺序：

1. FROM
    - 先选择一个表，或者说源头，构成一个结果集
2. WHERE
    - 对结果集进行筛选，筛选出需要的信息形成新的结果集
3. GROUP BY
    - 对新的结果集分组
4. HAVING
    - 筛选出想要的分组
5. SELECT
    - 选择列
6. DISTINCT
    - 去重
7. ORDER BY
    - 当所有的操作都结束，最后排序
8. LIMIT

加上多表连接后，执行顺序：先执行子查询，再执行主查询

1. FROM 和 JOIN ON
    - (1) 首先对 `FROM` 子句中的前两个表执行笛卡尔积运算，运算结果形成一个结果集合
    - (2) 按 `ON` 中的条件，对上边的结果集进行筛选，形成新的结果集
    - (3) 以 LEFT JOIN ON 为例，如果主表中存在未匹配到的行，
      把主表中的这几行以外部行的形式加到上边结果集中形成新的结果集
    - (4) 如果存在多张表，重复上面步骤 (1)~(3)
2. WHERE
    - 对结果集进行筛选，筛选出需要的信息形成新的结果集
3. GROUP BY
    - 对新的结果集分组
4. HAVING
    - 筛选出想要的分组
5. SELECT
    - 选择列
6. DISTINCT
    - 去重
7. ORDER BY
    - 当所有的操作都结束，最后排序
8. LIMIT

# SQL 常用模型

* `CASE WHEN` 的用法：不管偏不偏，你可能真没见过这种写法
* 内连接 VS 左连接：80% 的业务代码都与之相关
* `DISTINCT` 的用法：你可能真的错怪 `DISTINCT` 了
* `ORDER BY` 的注意事项：`ORDER BY` 一般放在主查询后，子查询无效
* `GROUP BY`：新手小白，总是 `GROUP BY` 时报错
* `HAVING`：`HAVING` 有时真的很牛逼
* `topN` 问题：分组取最大，分组取前几
* 标准 SQL 与基于 Hive 的 SQL 最常见的区别
* 不得不知的聚合函数：数据库中的聚合函数真的比 Excel 快很多

## CASE WHEN

### 常用形式

数据：

```sql
CREATE TABLE employees (
    LAST_NAME VARCHAR(50),
    JOB_ID VARCHAR(50),
    SALARY FLOAT
);

INSERT INIO employees (LAST_NAME, JOB_ID, SALARY) VALUES ('OConnell', 'SH_CLERK', 2600.00);
INSERT INIO employees (LAST_NAME, JOB_ID, SALARY) VALUES ('Grant', 'SH_CLERK', 2600.00);
INSERT INIO employees (LAST_NAME, JOB_ID, SALARY) VALUES ('Whalen', 'AD_ASST', 4400.00);
INSERT INIO employees (LAST_NAME, JOB_ID, SALARY) VALUES ('Hartstein', 'MK_MAN', 13000.00);
INSERT INIO employees (LAST_NAME, JOB_ID, SALARY) VALUES ('Fay', 'MK_ERP', 6000.00);
INSERT INIO employees (LAST_NAME, JOB_ID, SALARY) VALUES ('Mavris', 'HR_ERP', 6500.00);
INSERT INIO employees (LAST_NAME, JOB_ID, SALARY) VALUES ('Baer', 'PR_REP', 10000.00);
INSERT INIO employees (LAST_NAME, JOB_ID, SALARY) VALUES ('Higgins', 'AC_MGR', 12008.00);
INSERT INIO employees (LAST_NAME, JOB_ID, SALARY) VALUES ('Gietz', 'AC_ACCOUNT', 8300.00);
INSERT INIO employees (LAST_NAME, JOB_ID, SALARY) VALUES ('King', 'AD_PRES', 24000.00);
```

```sql
SELECT 
    last_name,
    job_id,
    salary,
    CASE job_id
        WHEN 'IT_PROJ' THEN 1.1 * salary
        WHEN 'ST_CLERK' THEN 1.15 * salary
        WHEN 'SA_REP' THEN 1.2 * salary
        ELSE salary
    END AS salary_new
FROM employees;
```

### 行列转换

数据：

```sql
CREATE TABLE EMP (
    DEPTNO INT,
    JOB VARCHAR(50)
);

INSERT INTO EMP (DEPTNO, JOB) VALUES (20, 'CLERK');
INSERT INTO EMP (DEPTNO, JOB) VALUES (30, 'SALESMAN');
INSERT INTO EMP (DEPTNO, JOB) VALUES (20, 'MANAGER');
INSERT INTO EMP (DEPTNO, JOB) VALUES (30, 'CLERK');
INSERT INTO EMP (DEPTNO, JOB) VALUES (10, 'PRESIDENT');
INSERT INTO EMP (DEPTNO, JOB) VALUES (30, 'MANAGER');
INSERT INTO EMP (DEPTNO, JOB) VALUES (10, 'CLERK');
INSERT INTO EMP (DEPTNO, JOB) VALUES (10, 'MANAGER');
INSERT INTO EMP (DEPTNO, JOB) VALUES (20, 'ANALYST');
```

```sql
SELECT
    deptno,
    SUM(clerk) AS clerk,
    SUM(salesman) AS salesman,
    SUM(manager) AS manager,
    SUM(analyst) AS analyst,
    SUM(president) AS president,
FROM (
    SELECT
        deptno,
        CASE job WHEN 'CLERK' THEN SAL END AS clerk,
        CASE job WHEN 'SALESMAN' THEN SAL END AS salesman, 
        CASE job WHEN 'MANAGER' THEN SAL END AS manager,
        CASE job WHEN 'ANALYST' THEN SAL END AS analyst, 
        CASE job WHEN 'PRESIDENT' THEN SAL END AS president 
    FROM EMP)
GROUP BY 
    deptno;
```

### 多指标数据

当想得到多个指标的数据，又不想写多条语句

```sql
CREATE TABLE qiuzhiliao AS 
SELECT 
    week_begin_date_id, 
    u_user
    (CASE 
        WHEN channel = 'oppokeke' THEN 'oppokeke'
        WHEN channel = 'huawei' THEN 'huawei'
        WHEN channel = 'xiaomi' THEN 'xiaomi'
        WHEN channel = 'yingyongbao' THEN 'yingyongbao'
        WHEN channel = 'yingyongbaozx' THEN 'yingyongbaozx'
        WHEN channel = 'AppStore' THEN 'AppStore'
        WHEN channel = 'baidu' THEN 'baidu'
        WHEN channel = '360zhushou' THEN channel ='360zhushou'
        WHEN channel = 'wandoujia' THEN 'wandoujia'
        WHEN channel LIKE 'bdsem%' THEN 'bdsem'
        WHEN channel LIKE 'sgsem%' THEN 'sgsem'
        WHEN channel LIKE 'smsem%' THEN 'smsem'
        ELSE '360zhushou'
        END 
    ) AS channel
FROM tmp.qzl_1
WHERE 
    (channel IN ('oppokeke', 'huawei', 'xiaomi', 'yingyongbao', 
                    'yingyongbaozx', 'AppStore', 'baidu', '360zhushou', 'wandoujia')
    OR channel LIKE 'bdsem%'
    OR channel LIKE 'sgsem%'
    OR channel LIKE 'smsem%')


SELECT 
    week_begin_date_id,
    channel,
    COUNT(DISTINCT u_user) u_user
FROM qiuzhiliao
GROUP BY 
    week_begin_date_id,
    channel 
ORDER BY 
    week_begin_date_id,
    channel
```

### 多条件统计

```sql
SELECT 
    zc.stab,
    SUM(CASE WHEN toppop > 10000 THEN 1 ELSE 0 END) AS num_10000,
    SUM(CASE WHEN toppop > 1000 THEN 1 ELSE 0 END) as num_1000,
    SUM(CASE WHEN toppop > 100 THEN 1 ELSE 0 END) AS num_100
FROM ZipCensus AS zc
GROUP BY zc.stab
```

## DISTINCT

* `DISTINCT` 有去除重复值的效果，但查询字段多于两个时，
  就是对这两个字段联合去重(两个字段同时相同，才会被当做重复值)
* `DISTINCT` 只能放在首列，否则会报错
* 去除重复值最好用 `GROUP BY`，`DISTINCT` 很多时候出现在 `COUNT(DISTINCT 字段)` 中用于统计去除重复值后的条数

## GROUP BY

* `GROUP BY` + 聚合建
* `GROUP BY` 后只能出现聚合键 或聚合函数，不能出现其他字段，否则报错

## HAVING

* `HAVING` 只能跟在 `GROUP BY` 后，不能单独使用
* `HAVING` 是对 `GROUP BY` 分组后的数据进行筛选 判断

## ORDER BY

* `ORDER BY` 放在语句的最后，同时执行顺序也是最后
* `ORDER BY` 放在子查询中会失效，一般放在主查询中最后执行

## 聚合函数

### SUM 与 CASE WHEN 的结合

```sql
SELECT 
    a.课程号, 
    b.课程名称,
    SUM(CASE WHEN 成绩 BETWEEN 85 AND 100 THEN 1 ELSE 0 END) AS '[100-85]',
    SUM(CASE WHEN 成绩 >= 70 AND 成绩 < 85 THEN 1 ELSE 0 END) AS '[85-70]',
    SUM(CASE WHEN 成绩 >= 60 AND 成绩 < 70  THEN 1 ELSE 0 END) AS '[70-60]',
    SUM(CASE WHEN 成绩 < 60 THEN 1 ELSE 0 END) AS '[<60]'
FROM score AS a 
RIGHT JOIN course AS b 
    ON a.课程号 = b.课程号
GROUP BY 
    a.课程号,
    b.课程名称;
```

### COUNT 与 CASE WHEN 的结合

```sql
-- 产品路径统计可以这么写
-- 春节活动入口统计

SELECT 
    day,
    COUNT(CASE WHEN x2_1.event_key = 'initApp' THEN x2_1.u_user ELSE NULL END) AS ad_pv,  --开屏曝光的pv
    COUNT(DISTINCT CASE WHEN x2_1.event_key = 'initApp' THEN x2_1.u_user ELSE NULL END) AS ad_uv,  --开屏曝光的uv
FROM (
    SELECT 
        FROM_UNIXTIME(
            UNIX_TIMESTAMP(CAST(day as string), 'yyyyMMdd'), 
            'yyyy-MM-dd'
        ) as day,
        event_key,
        u_user,
        status,
        button,
        device
    FROM table
    WHERE 
        FROM_UNIXTIME(
            UNIX_TIMESTAMP(CAST(day as string), 'yyyyMMdd'), 
            'yyyy-MM-dd'
        ) BETWEEN '2019-03-28' AND DATE_SUB(FROM_UNIXTIME(UNIX_TIMESTAMP(),'yyyy-MM-dd'), 1)
        AND event_key = 'initApp'
GROUP BY 
    day 
ORDER BY 
    day
```

## topN

1. 分组取最大、最小、平均值，但无法的大聚合键之外的数据，这时可以使用关联子查询
    - `GROUP BY` + 聚合函数
2. 求得每组前两名数据
    - `LIMIT` + `UNION ALL`

```sql
SELECT
    课程号,
    MAX(成绩) AS 最大成绩,
FROM score
GROUP BY
    课程号;
```

3. 分组取最大值，但得不到聚合键之外的数据，使用关联子查询

```sql
SELECT * 
FROM score AS a
WHERE 成绩 = (
    SELECT 
        MAX(成绩)
    FROM score AS b
    WHERE b.课程号 = a.课程号
);
```

### 每组最大的 N 条记录

1. 先求出最大记录所在的组

```sql
select 
    课程号,
    MAX(成绩) AS 最大成绩
FROM score 
GROUP BY 课程号;
```

2. `UNION` 连接

```sql
(
    SELECT * 
    FROM score 
    WHERE 课程号 = '0001' 
    ORDER BY 成绩 DESC 
    LIMIT 2
)
UNION ALL
(
    SELECT * 
    FROM score 
    WHERE 课程号 = '0002' 
    ORDER BY 成绩 DESC 
    LIMIT 2
)
UNION ALL
(
    SELECT * 
    FROM score 
    WHERE 课程号 = '0003' 
    ORDER BY 成绩 DESC 
    LIMIT 2
);
```

## SQL 优化

1. Hive SQL 中多表 JOIN 时，需要将小表写在左边，Hive 会将小表数据存放内存中，
   以实现 MapReduce 中 `map join` 的效果
2. SparkSQL 中，`join` 时大小表的顺序对执行效率几乎没有影响
3. `count(*)` 会返回包含 `null` 的行数；`count(字段)返回不包括 null 的行数
4. Spark 支持非等值连接，并且查询速度高于 Hive。一般先在 Spark 上创建表格，然后在 Hive 中查询并导出查询结果

<p align='right'><a href='#top'><返回顶端></a></p>

# SQL 函数

## Oracle

### NVL

`NULL` 是没有，不参与计算，可以将 `NULL` 转换为 0 参与计算

* `NVL(字段, 0)`：字段非空显式本身，`NULL` 补为 0
* `NVL(字段, 1, 0)`：字段非空替换为 1，`NULL` 补为 0
    - 类似二进制编码 
* `''` 是空字符串，空也是字符

### 链接符

在转义特殊字符的时候通常使用单引号，但这种转义方式很不直观。
在 Oracle 中使用 `q'` 来对特殊字符进行转义。
`q'` 转义符通常后面使用 `!`、`[]`、`{}`、`()`、`<>` 等转义符号，
也可以使用 `\`，也可以用字母、数字、=、+、-、*、&、$、%、# 等，不可以使用空格

在 Oracle中，`||` 运算符可以将两个或两个以上的字符串连接在一起

如果链接符 `q` 与 `||` 一块使用：

* 链接符 `q` 需要在 `||` 后使用
* 除了起别名，会用到 `""`，在其他任何地方，都使用 `''`

```sql
# 正确用法
SELECT first_name || '\\' FROM employees;
SELECT first_name || q'[\\]' FROM employees;  -- 与上面的等价

# 以下用法错误
SELECT first_name || '[\\]' FROM employees;
SELECT first_name || "abc" FROM employees;
```

### 参考

* [SQL函数](https://zhuanlan.zhihu.com/p/72252028)

## MySQL

* TODO

<p align='right'><a href='#top'><返回顶端></a></p>

# SQL 题目

## 活动运营

> 题目：活动运营数据分析
> 数据表：
> * 表 1：订单表 `orders`，字段：`user_id`(用户编号)、`order_pay`(订单金额)、`order_time`(下单时间)
> * 表 2：活动报名报 `act_apply`，字段：`act_id`(活动编号)、`user_id`(报名用户)、`act_time`(报名时间)
> 需求：
> 1. 统计每个活动对应所有用户在报名后产生的总订单金额、总订单数。
>    (每个用户限报一个活动,题干默认用户报名后产生的订单均为参加活动的订单）
> 2. 统计每个活动从开始后到当天（考试日）平均每天产生的订单数，
>    活动开始时间定义为最早有用户报名的时间。（涉及到时间的数据类型均为：datetime）

需求 1 思路：

| 活动             | 总订单金额      | 总订单数             |
|-----------------|----------------|--------------------|
| group by act_id | sum(order_pay) | count(order_time)  |
| 表 2             | 表 1           | 表 1               |

```sql
SELECT 
    a.act_id,
    SUM(o.order_pay) AS act_pay,
    COUNT(o.order_time) AS act_orders
FROM act_apply AS a
INNER JOIN orders AS o
    ON a.user_id = o.user_id
WHERE a.act_time <= o.order_time
GROUP BY 
    a.act_id
```

需求 2 思路：

| 活动             | 平均每天订单数              |
|-----------------|---------------------------|
| group by act_id | count(order_time)/时间间隔  |
| 表 2            | 表 1                       |

```sql
SELECT 
    a.act_id,
    COUNT(o.order_time) / DATEDIFF(NOW(), MIN(a.begin_time)) AS avg_orders
FROM (
    SELECT
        act_id,
        user_id,
        act_time,
        MIN(act_time) OVER(PARTITION BY act_id) AS begin_time
    FROM 
        act_apply
) AS a
INNER JOIN orders AS o
    ON a.user_id = o.user_id
WHERE
    (a.act_time >= a.begin_time AND a.act_time <= NOW())
    AND a.act_time <= o.order_time
GROUP BY 
    a.act_id
```

## 用户行为分析

> 题目：用户行为分析
> * 表 1：用户行为表 `tracking_log`，字段：`user_id`(用户编号)、`opr_id`(操作编号)、`log_time`(操作时间)
> 需求：
> 1. 计算每天的访客数和他们的平均操作次数
> 2. 统计每天符合以下条件的用户数：A 操作之后是 B 操作，AB操作必须相邻

需求 1 思路：

| 日期 | 访客数 | 平均操作次数 |
|----|----|----|
| group by date(log_time) | count(distinct user_id | avg(count(opr_id) |

```SQL
SELECT 
    DATE(t.log_time) AS date_,
    COUNT(DISTINCT t.user_id) AS user_num,
    AVG(t.opr_num) AS opr_avg
FROM (
    SELECT 
        DATE(log_time) as log_time,
        user_id,
        COUNT(opr_id) AS opr_num
    FROM 
        tracking_log
    GROUP BY user_id, DATE(log_time)
) AS t
GROUP BY DATE(t.log_time)
```

需求 2 思路：

```sql
SELECt
    DATE(log_time) as log_time,
    COUNT(DISTINCT user_id) as user_id
FROM (
    SELECT 
        user_id,
        DATE(log_time) as log_time,
        opr_id,
        LEAD(opr_id, 1) OVER(PARTITION BY user_id ORDER BY log_time) as opr_id_2
    FROM tracking_log
)
WHERE 
    opr_id = 'A' 
    AND opr_id_2 = 'B'
GROUP BY DATE(log_time)
```

## 用户新增留存分析

> 题目：用户新增留存分析
> * 表 1：用户登录表 `user_log`，字段：`user_id`(用户编号)、`log_time`(登录时间)
> 需求：
> * 每天新增用户数，以及他们第 2 天、30 天的回访比例

需求思路：

| 日期 | 新增用户数 | 第 2 天回访用户 | 第 30 天回访用户 |
|-----|-----------|---------------|----------------|
| group by date(log_time) | count(distinct user_id) | count(distinct user_id) | count(distinct user_id) |


* 如何定义新增用户：用户登陆表中最早的登陆时间所在的用户数为当天新增用户数
* 第 2 天回访用户数：第一天登陆的用户中，第二天依旧登陆的用户为次日留存率
* 第 30 天的回访用户数：第一天登陆用户中，第 30 天依旧登陆的用户

```sql
SELECT 
    DATE(t1.user_begin) AS date_user_begin,
    COUNT(DISTINCT t1.user_id) AS '新增用户',
    COUNT(DISTINCT t2.user_id) AS '第二日留存用户',
    COUNT(DISTINCT t3.user_id) AS '第30日留存用户'
FROM (
    SELECT
        user_id,
        MIN(log_time) as user_begin
    FROM user_log
    GROUP BY user_id
) AS t1
LEFT JOIN user_log AS t2
    ON t1.user_id = t2.user_id
       AND DATE(t2.log_time) = DATE(t1.user_begin) + 1
LEFT JOIN user_log AS t3
    ON t1.user_id = t3.user_id
       AND DATE(t3.log_time) = DATE(t1.user_begin) + 29
GROUP BY DATE(t1.user_begin)
```

## SQL 求分位数

中位数：

```sql
SELECT
    feature1 * (1 - float_part) + next_feature1 * (float_part - 0) AS median
FROM (
    SELECT
        feature1,
        ROW_NUMBER() OVER(ORDER BY feature1 ASC) AS rank,
        LEAD(feature1, 1) OVER(ORDER BY feature1 ASC) AS next_feature1
    FROM iris
    ) a
INNER JOIN (
    SELECT
        CAST((COUNT(feature1) + 1) / 2 AS INT) AS int_part,
        (COUNT(feature1) + 1) / 2 % 1 AS float_part
    FROM iris
) b ON a.rank = b.int_part;
```

分位数：

```sql
select
    feature1 * (1 - float_part) + next_feature1 * (float_part-0) AS q1
from (
    SELECT
        feature1,
        ROW_NUMBER() OVER(ORDER BY feature1 ASC) AS rank,
        LEAD(feature1, 1) OVER(ORDER BY feature1 ASC) AS next_feature1
    FROM iris) a
INNER JOIN (
    SELECT
        CAST((COUNT(feature1) + 1) * 0.25 AS INT) AS int_part,
        (COUNT(feature1) + 1) * 0.25 % 1 AS float_part
    FROM iris
) b ON a.rank = b.int_part
```



<p align="right"><a href="#top"><返回顶端></a></p>


# 参考

* [SQL 常用模型](https://zhuanlan.zhihu.com/p/71258400)

<p align='right'><a href='#top'><返回顶端></a></p>
