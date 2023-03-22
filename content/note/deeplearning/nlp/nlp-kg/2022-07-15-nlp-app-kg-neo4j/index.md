---
title: NLP-Neo4j
author: 王哲峰
date: '2022-04-05'
slug: nlp-app-kg-neo4j
categories:
  - nlp
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

- [Neo4j 介绍](#neo4j-介绍)
  - [Neo4j 是什么](#neo4j-是什么)
  - [Neo4j 的特点](#neo4j-的特点)
  - [Neo4j 的优点](#neo4j-的优点)
  - [Neo4j 的缺点、限制](#neo4j-的缺点限制)
  - [Neo4j 构建模块](#neo4j-构建模块)
  - [Neo4j 概念详解](#neo4j-概念详解)
    - [Neo4j 关系](#neo4j-关系)
  - [Neo4j 安装、使用](#neo4j-安装使用)
    - [Neo4j 安装](#neo4j-安装)
    - [Neo4j 使用](#neo4j-使用)
- [Neo4j CQL](#neo4j-cql)
  - [CQL 简介](#cql-简介)
  - [CQL 命令关键字](#cql-命令关键字)
  - [CQL 函数](#cql-函数)
  - [CQL 数据类型](#cql-数据类型)
  - [CQL 命令](#cql-命令)
    - [CREATE](#create)
    - [CREATE...MATCH...RETURN](#creatematchreturn)
    - [WHERE](#where)
    - [DELETE](#delete)
    - [REMOVE](#remove)
    - [SET](#set)
    - [ORDER BY](#order-by)
    - [Sorting](#sorting)
    - [UNION](#union)
    - [LIMIT、SKIP](#limitskip)
    - [MERGE](#merge)
    - [NULL](#null)
    - [IN](#in)
    - [图形字体](#图形字体)
    - [ID 属性](#id-属性)
    - [Caption](#caption)
  - [CQL 函数](#cql-函数-1)
    - [Neo4j CQL 字符串函数](#neo4j-cql-字符串函数)
    - [Neo4j CQL AGGREGATION 聚合函数](#neo4j-cql-aggregation-聚合函数)
    - [Neo4j CQL 关系函数](#neo4j-cql-关系函数)
  - [Admin 管理员](#admin-管理员)
    - [Neo4j 数据库备份和恢复](#neo4j-数据库备份和恢复)
    - [Neo4j CQL 索引](#neo4j-cql-索引)
    - [Neo4j CQL UNIQUE 约束](#neo4j-cql-unique-约束)
    - [Neo4j CQL DROP UNIQUE](#neo4j-cql-drop-unique)
  - [Neo4j 实战](#neo4j-实战)
- [py2neo](#py2neo)
  - [py2neo.database](#py2neodatabase)
    - [连接](#连接)
- [参考](#参考)
</p></details><p></p>

# Neo4j 介绍

## Neo4j 是什么

- Neo4j 是一个世界领先的开源图形数据库。它是由 Neo 技术使用 Java 语言完全开发的。
- Neo4j是一个高性能的 NOSQL 图形数据库, 它将结构化数据存储在网络上而不是表中。
  它是一个嵌入式的、基于磁盘的、具备完全的事务特性的 Java 持久化引擎, 但是它将
  结构化数据存储在网络(从数学角度叫做图)上而不是表中。
- Neo4j 也可以被看作是一个高性能的图引擎, 该引擎具有成熟数据库的所有特性。程序员
  工作在一个面向对象的、灵活的网络结构下而不是严格、静态的表中——但是他们可以享受到
  具备完全的事务特性、企业级的数据库的所有好处。
- Neo4j 管网
    - http://www.neo4j.org

## Neo4j 的特点

- Neo4j 是开源的
- Neo4j 无 Schema
- Neo4j 的查询语言是 Neo4j CQL, 类似 SQL
- Neo4j 是一种图形数据库
    - 图形数据库是以图形结构的形式存储数据的数据库。它以节点、关系、属性的形式存储应用程序的数据, 
      正如 RDBMS 以表的“行、列”的形式存储数据, GDBMS 以"图形"的形式存储数据。
- Neo4j 遵循 **属性图数据模型** 存储和管理数据
    - 属性图模型规则如下:
        - 图形是 **一组节点** 和 **连接这些节点的关系**
        - **节点** 和 **关系** 包含表示数据的 **属性**
            - 关系连接节点
                - 关系具有方向:单向、双向
                    - 在属性图数据模型中, 关系应该是定向的, 如果尝试创建没有方向的关系, 那么将抛出一个错误
                    - 在 Neo4j 中, 关系也应该是有方向性的, 如果尝试创建没有方向的关系, 那么将抛出一个错误
                - 每个关系包含:
                    - 开始节点 或 从节点
                    - 结束节点 或 到节点 
            - 属性是用于表示数据的键值对
            - 节点用圆圈表示, 关系用方向键表示
    - Neo4j 是一个流行的图数据库。其他图形数据库有: 
        - Oracel NoSQL
        - OrientDB
        - HypherGraphDB
        - GraphBase
        - InfiniteGraph
        - AllegroGraph
- Neo4j 通过使用 Apache Lucence 支持索引
- Neo4j 支持 UNIQUE 约束
- Neo4j 包含一个用于执行 CQL 命令的 UI:Neo4j 数据浏览器
- Neo4j 支持完整的 ACID(原子性, 一致性, 隔离性和持久性)规则
- Neo4j 采用原生图形库与本地 GPE(图形处理引擎)
- Neo4j 支持查询的数据导出到 JSON 和 XLS 格式
- Neo4j 提供了 REST API, 可以被任何编程语言 (如Java, Spring, Scala等) 访问
- Neo4j 提供了可以通过任何 UI MVC 框架 (如Node JS) 访问的 Java 脚本 
- Neo4j 支持两种 Java API: Cypher API 和 Native Java API 来开发 Java 应用程序

## Neo4j 的优点

- 很容易表示连接的数据
- 检索/遍历/导航更多的连接数据是非常容易和快速的
- 非常容易地表示半结构化数据
- Neo4j CQL 查询语言命令是人性化的可读格式, 非常容易学习
- 使用简单而强大的数据模型
- 不需要复杂的连接来检索连接的/相关的数据, 因为它很容易检索它的相邻节点或关系细节没有连接或索引

## Neo4j 的缺点、限制

- 节点数、关系、属性的限制
- 不支持 Sharding

## Neo4j 构建模块

- 节点
- 属性
- 关系
- 标签
    - Label将一个公共名称与一组节点或关系相关联。 节点或关系可以包含一个或多个标签。 
      我们可以为现有节点或关系创建新标签。 我们可以从现有节点或关系中删除现有标签。
- 数据浏览器
    - http://localhost:7474/browser/

## Neo4j 概念详解

### Neo4j 关系

Neo4j 图数据库遵循属性图模型来存储和管理其数据。根据属性图模型, 关系应该是定向的, 
否则, Neo4j 将抛出一个错误消息。基于方向性, Neo4j 关系被分为两种类型:

- 单向关系
- 双向关系

1.使用新节点创建关系

- 语法

```
CREATE (<node1-name>:<node1-label-name>)-[<relationship-name>:<relationship-label-name>]->(<node2-name>:<node2-label-name>)
CREATE (<node1-name>:<node1-label-name>)<-[<relationship-name>:<relationship-label-name>]-(<node2-name>:<node2-label-name>)
CREATE (<node1-name>:<node1-label-name>)<-[<relationship-name>:<relationship-label-name>]->(<node2-name>:<node2-label-name>)
```

- 示例

```
CREATE (e:Employee)-[r:DemoRelation]->(c:Employee)
CREATE (e:Employee)<-[r:DemoRelation]-(c:Employee)
CREATE (e:Employee)<-[r:DemoRelation]->(c:Employee)
```

2.使用已知节点创建带属性的关系

- 语法

```
MATCH (<node1-name>:<node1-label-name>),(<node2-name>:<node2-label-name>)
CREATE
    (<node1-name>)-[<relationship-name>:<relationship-label-name>{<define-properties-list>}]->(<node2-name>)
RETURN <relationship-name>
```

- 示例

```
MATCH (cust:Customer), (cc.CreditCard)
CREATE (cust)-[r:DO_SHOPPING_WITH{shopdate:"12/12/2014", price:55000}]->(cc)
RETURN r
```

3.检索关系节点的详细信息

- 语法

```
MATCH (<node1-name>)-[<relationship-name>:<relationship-label-name>]->(<node2-name>)
RETURN <node1-name>, <relationship-name>, <node2-name>
```

- 示例

```
MATCH (cust)-[r:DO_SHOPPING_WITH]->(cc)
RETURN cust, cc, r
```

4.示例

- 目标:演示如何使用属性和创建两个节点、两个节点的关系
- 需求: 创建两个节点:客户节点(Customer)和信用卡节点(CreditCard)
    - Customer 节点包含:id, name, dob 属性
    - CreditCard 节点包含:id, number, cvv, expiredate 属性
    - Customer 与 CreditCard 关系:DO_SHOPPING_WITH
    - CreditCard 到 Customer 关系:ASSOCIATED_WITH
- 步骤:
    - 创建 Customer 节点
    - 创建 CreditCard 节点
    - 观察先前创建的两个节点: Customer 和 CreditCard
    - 创建 Customer 和 CreditCard 节点之间的关系
    - 查看新创建的关系详细信息
    - 详细查看每个节点和关系属性

```
CREATE (e:Customer{id:"1001", name:"Abc", dob:"01/10/1982"})
CREATE (cc:CreditCard{id:"5001", number:"1234567890", cvv:"888", expiredate:"20/17"})

MATCH (cust:Customer), (cc:CreditCard)
CREATE (cust)-[r1:DO_SHOPPING_WITH{shopdate:"12/12/2014", price:55000}]->(cc)
RETURN r1

MATCH (cust:Customer), (cc:CreditCard)
CREATE (e)<-[r2:ASSOCIATED_WITH]-(cc)
RETURN r2

MATCH (cc:CreditCard) RETURN cc.id, cc.number, cc.cvv, cc.expiredate
```

## Neo4j 安装、使用

### Neo4j 安装

- 首先在 https://neo4j.com/download/ 下载 Neo4j。Neo4j 分为社区版和企业版, 
  企业版在横向扩展、权限控制、运行性能、HA 等方面都比社区版好, 适合正式的生产环境, 
  普通的学习和开发采用免费社区版就好。
- 在 Mac 或者 Linux 中, 安装好 jdk 后, 直接解压下载好的 Neo4J 包, 
  运行 `bin/neo4j start` 即可

### Neo4j 使用

- Neo4j 提供了一个用户友好的 web 界面, 可以进行各项配置、写入、查询等操作, 并且提供了可视化功能。
  类似 ElasticSearch 一样。
- 打开浏览器, 输入 http://127.0.0.1:7474/browser/, 界面最上方就是交互的输入框。

# Neo4j CQL

## CQL 简介

Cypher 是 Neo4j 的声明式图形查询语言, 允许用户不必编写图形结构的遍历代码, 
就可以对图形数据进行高效的查询。Cypher 的设计目的类似 SQL, 适合于开发者以
及在数据库上做点对点模式(ad-hoc)查询的专业操作人员。其具备的能力包括:
    
- 创建、更新、删除节点和关系 
- 通过模式匹配来查询和修改节点和关系 
- 管理索引和约束等

Neo4j CQL 代表 Neo4j Cypher Query Language

- CQL 是 Neo4j 图形数据库的查询语言
- CQL 是一种声明性模式匹配语言
- CQL 遵循 SQL 语法
- CQL 的语法非常简单且人性化、可读性强

类似 MySQL 一样, 在实际的生产应用中, 除了简单的查询操作会在 Neo4j 的 web 页面进行外, 
一般还是使用 Python、Java 等的 driver 来在程序中实现

## CQL 命令关键字

| CQL命令 |    用法 |
|--------|---------------------|
| CREATE   |  创建节点、关系、属性 |
| MATCH    |  检索节点、关系、属性数据 |
| RETURN   |  返回查询结果 |
| WHERE    |  提供条件过滤检索数据 |
| DELETE   |  删除节点、关系 |
| REMOVE   |  删除节点、关系的属性 |
| ORDER BY |  排序检索数据 |
| SET      |  添加或更新标签 |

## CQL 函数

| 定制列表功能     | 用法  |
|--------|---------------------|
| String       |   用于使用 String 字面量 |
| Aggregation  |   用于对 CQL 查询结果执行一些聚合操作 |
| Relationshop |   用于获取关系的细节, startnode, endnode等 |

## CQL 数据类型

| CQL 数据类型   |  用法 |
|--------|---------------------|
| boolean   |      用于表示布尔文字: true, false |
| byte      |      用于表示8位整数 |
| short     |      用于表示16位整数 |
| int       |      用于表示32位整数 |
| long      |      用于表示64位整数 |
| float     |      I用于表示32位浮点数 |
| double    |      用于表示64位浮点数 |
| char      |      用于表示16位字符 |
| String    |      用于表示字符串 |

## CQL 命令

### CREATE

1.创建没有属性的节点

创建 **节点标签名称**, 相当于 MySQL 数据库中的表名

- 语法

```
CREATE (<node-name>:<node-label-name>)
```

- 说明
    - `<node-name>`: 要创建的节点名称
    - `<node-label-name>`: 要创建的节点标签
    - Neo4j 数据库服务器使用 `<node-name>` 将此节点详细信息存储在 Database.As 中作为 Neo4j DBA 或 Developer, 
      不能使用它来访问节点详细信息。
    - Neo4j 数据库服务器创建一个 `<node-label-name>` 作为内部节点名称的别名, 
      作为 Neo4j DBA 或 Developer, 应该使用此标签名称来访问节点详细信息。
- 示例

```
CREATE (emp:Employee)

CREATE (dept:Dept)
```

2.创建具有属性的节点

创建一个具有一些属性(键-值对)的节点来存储数据

- 语法

```
CREATE (
    <node-name>:<node-label-name> {
        <Property1-name>:<Property1-Value>,
        <Property2-name>:<Property2-Value>,
        ...,
        <Propertyn-name>:<Propertyn-Value>
    }
)
```

- 示例

```
CREATE (
    emp:Employee {
        id: 123,
        name: "Lokesh",
        sal: 35000,
        deptno: 10
    }
)

CREATE (
    dept:Dept {
        deptno: 10,
        dname: "Accounting",
        location: "Hyderabad"
    }
)
```

3.创建多个标签的节点

- 语法

```
CREATE (<node-name>:<node-label-name1>:<label-name2>...:<label-namen>)
```

- 示例

```
CREATE (m:Movie:Cinema:Film:Picture)
```

### CREATE...MATCH...RETURN

- Neo4j CQL `MATCH` 命令用于:
    - 从数据库获取有关节点和属性的数据
    - 从数据库获取有关节点、关系和属性的数据
    - 不能单独使用 MATCH 命令从数据库检索数据。 如果单独使用它,将 InvalidSyntax 错误
- Neo4j CQL `RETURN` 用于:
    - 检索节点的某些属性
    - 检索节点的所有属性
    - 检索节点和关联关系的某些属性
    - 检索节点和关联关系的所有属性
    
- 语法

```
MATCH (<node-name>:<node-label-name>)
RETURN
    <node-name>.<property1-name>,
    <node-name>.<property1-name>,
    ...,
    <node-name>.<property1-name>
```

- 示例 1:

```
MATCH (dept:Dept) // 会报错
MATCH (dept:Dept) RETURN dept
MATCH (dept:Dept) RETURN dept.deptno, dept.dname, dept.location
MATCH (e:Employee) RETURN e
MATCH (p:Employee {id:123, name:"Lokesh"}) RETURN p
MATCH (p:Employee) WHERE p.name = "Lokesh" RETURN p
```

### WHERE

Neo4j CQL `WHERE` 过滤 `MATCH` 查询的结果

- 语法

```
WHERE <condition>
WHERE <property-name> <comparison-operator> <value>
WHERE <condition> <boolean-operator> <condition>
```

其中：

- `<comparison-operator>`:
  - `=`
  - `<>`
  - `<`
  - `>`
  - `<=`
  - `>=`
- `<boolean-operator>`
  - AND
  - OR
  - NOT
  - XOR

使用 WHERE 子句创建关系

- 语法

```
MATCH (<node1-name>:<node1-label-name>),(<node2-name>:<node2-label-name>)
WHERE <condition>
CREATE (<node1-name>)-[<relationship-name>:<relationship-label-name>{<relationship-properties>}]->(<node2-name>)
```

- 示例

```
MATCH (cust:Customer)
RETURN cust.id, cust.name, cust.dob

MATCH (cc:CreditCard)
RETURN cc.id, cc.number, cc.expiredate, cc.cvv

MATCH (cust:Customer), (cc.CreditCard)
WHERE cust.id = "1001" AND cc.id = "5001"
CREATE (cust)-[r:DO_SHOPPING_WITH{shopdate:"12/12/2014", price:55000}]->(cc)
RETURN r
```

- 示例

```
MATCH (emp:Employee)
RETURN emp.empid, emp.name, emp.salary, emp.deptno

MATCH (emp:Employee)
WHERE emp.name = "Abc"
RETURN emp

MATCH (emp:Employee)
WHERE emp.name = 'Abc' OR emp.name = 'Xyz'
RETURN emp

MATCH (cust:Customer), (cc:CreditCard)
WHERE cust.id = '1001' AND cc.id = '5001'
CREATE (cust)-[r:DO_SHOPPING_WITH{shopdate:"12/12/2014", price:55000}]->(cc)
RETURN r

MATCH p = (m:Bot{id:123})<-[:BotRelation]->(:Bot) 
RETURN p
```

### DELETE

Neo4j 使用 CQL `DELETE` 用来:

- 删除节点
- 删除节点及相关节点和关系

1.删除节点

- 语法

```
DELETE <node-name-list>

DELETE <node1-name>, <node2-name>, <relationship-name>
```

- 示例

```
MATCH (e:Employee) 
RETURN e

MATCH (e:Employee) 
DELETE e

MATCH (e:Employee) 
RETURN e

MATCH (cc:CreditCard)-[r]-(c:Customer)
RETURN r 

MATCH (cc:CreditCard)-[rel]-(c:Customer)
DELETE cc, c, rel
```

2.删除节点和关系

- 语法

```
DELETE <node-name1>, <node-name2>, <relationship-name>
```

- 示例

```
MATCH (cc:CreditCard)-[rel]-(c:Customer)
DELETE cc, c,rel
```

### REMOVE

Neo4j CQL `REMOVE` 删除节点或关系的现有属性

- 语法

```
// 删除节点/关系的属性
REMOVE <property-name-list>

REMOVE
    <node-name>.<property1-name>,
    <node-name>.<property2-name>, 
    .... 
    <node-name>.<propertyn-name> 

// 删除节点/关系的标签
REMOVE <label-name-list>

REMOVE 
    <node-name>:<label2-name>, 
    .... 
    <node-name>:<labeln-name> 
```

- 示例1

```
CREATE (
    book:Book {
        id:122,
        title:"Neo4j Tutorial",
        pages:340,
        price:250
    }
)

MATCH (book:Book)
RETURN book

MATCH (book {id:122})
REMOVE book.price
RETURN book

MATCH (dc:DebitCard)
RETURN dc

MATCH (dc:DebitCard)
REMOVE dc.cvv
RETURN dc
```

- 示例2

```
MATCH (m:Movie)
RETURN m

MATCH (m:Movie) 
REMOVE m:Picture

MATCH (m:Movie) 
RETURN m
```

### SET

Neo4j CQL 已提供 SET 子句来执行以下操作:

- 向现有节点或关系添加新属性
- 添加或更新属性值

- 语法

```
SET <property-name-list>

SET
    <node-label-name>.<property1-name>,
    <node-label-name>.<property2-name>, 
    .... 
    <node-label-name>.<propertyn-name> 
```

- 示例

```
MATCH (book:Book)
RETURN book

MATCH (book:Book)
SET book.title = "superstar" 
RETURN book
```

### ORDER BY

Neo4j CQL `ORDER BY` 对 MATCH 查询返回的结果进行排序

- 语法

```
ORDER BY <property-name-list> [DESC]

ORDDR BY
    <node-label-name>.<property1-name> [DESC],
    <node-label-name>.<property2-name> [DESC], 
    .... 
    <node-label-name>.<propertyn-name> [DESC]
```

- 示例

```
MATCH (emp:Employee) 
RETURN emp.empid, emp.name, emp.salary, emp.deptno

MATCH (emp:Employee) 
RETURN emp.empid, emp.name, emp.salary, emp.deptno
ORDER BY emp.name
```

### Sorting


### UNION

Neo4j CQL 与 SQL 一样, 有两个语句可以将两个不同的结果合并成一组结果

`UNION`

- 将两组结果中的公共行组合并返回到一组结果中, 会进行去重
- 限制
    - 结果列的 名称 和 类型 和来自两组结果的名称、类型必须匹配
- 语法

```
MATCH Command
UNION/UNION ALL
MATCH Command
```

- 示例

```
MATCH (cc.CreditCard) RETURN cc.id, cc.number
UNION/UNION ALL
MATCH (dc.DebitCard) RETURN dc.id, dc.number

MATCH (cc.CreditCard)
RETURN cc.id as id, cc.number as number, cc.name as name, cc.valid_from as valid_from, cc.valid_to as valid_to
UNION/UNION ALL
MATCH (dc.DebitCard)
RETURN dc.id as id, dc.number as number, dc.name as name, dc.valid_from as valid_from, dc.valid_to as valid_to
```

`UNION ALL`

- 将两组结果中的公共行组合并返回到一组结果中, 不会进行去重
- 限制
    - 结果列的 名称 和 类型 和来自两组结果的名称、类型必须匹配
- 语法
    - 同 `UNION`
- 示例
    - 同 `UNION`

### LIMIT、SKIP

LIMIT

- Neo4j CQL 提供 `LIMIT` 子句来过滤或限制查询返回的行数, 它修剪CQL查询结果集底部的结果
- 语法

```
LIMIT <number>
```

- 示例

```
MATCH (emp:Employee) 
RETURN emp

MATCH (emp:Employee) 
RETURN emp
LIMIT 2
```

SKIP

- Neo4j CQL 提供 `SKIP` 来过滤或限制查询返回的行数, 它修整了CQL查询结果集顶部的结果
- 语法

```
SKIP <number>
```

- 示例

```
MATCH (emp:Employee) 
RETURN emp

MATCH (emp:Employee) 
RETURN emp
SKIP 2
```

### MERGE

- Neo4j CQL `MERGE` 命令:
    - 创建节点, 关系和属性
    - 为从数据库检索数据
    - MERGE 命令是 CREATE 命令和 MATCH 命令的组合
    - Neo4j CQL MERGE 命令在图中搜索给定模式, 如果存在则返回结果; 如果不存在于图中, 则创建新的节点/关系并返回结果
- 语法

```
MERGE (<node-name>:<label-name> {
    <Property1-name>:<Pro<rty1-Value>
    .....
    <Propertyn-name>:<Propertyn-Value>
})

CREATE (gp1:GoogleProfile1 {Id:201401, Name:"Apple"})
CREATE (gp1:GoogleProfile1 {Id:201401, Name:"Apple"})
MATCH  (gp1:GoogleProfile1) 
RETURN gp1.Id, gp1.Name

MERGE (gp2:GoogleProfile2 {Id:201402, Name:"Nokia"})
MERGE (gp2:GoogleProfile2 {Id:201402, Name:"Nokia"})
MATCH  (gp2:GoogleProfile2) 
RETURN gp2.Id,g p2.Name
```

### NULL

- Neo4j CQL 将空值视为对节点或关系的属性的缺失值或未定义值
    - 当创建一个具有现有节点标签名称但未指定其属性值的节点时, 它将创建一个具有 NULL 属性值的新节点
- 示例

```
MATCH (e:Employee) 
RETURN e.id, e.name, e.sal, e.deptno

CREATE (e:Employee)

MATCH (e:Employee) 
RETURN e.id, e.name, e.sal, e.deptno

MATCH (e:Employee) 
WHERE e.id IS NOT NULL
RETURN e.id, e.name, e.sal, e.deptno

MATCH (e:Employee) 
WHERE e.id IS NULL
RETURN e.id, e.name, e.sal, e.deptno
```

### IN 

- Neo4j CQL 提供了一个 IN 运算符, 以便为 CQL 命令提供值的集合
- 语法

```
IN [<Collection-of-values>]
```

- 示例

```
MATCH (e:Employee) 
RETURN e.id, e.name, e.sal, e.deptno

MATCH (e:Employee) 
WHERE e.id IN [123,124]
RETURN e.id, e.name, e.sal, e.deptno
```

### 图形字体

- 使用 Neo4j 数据浏览器来执行和查看 Neo4j CQL 命令或查询的结果, 
  Neo4j 数据浏览器包含两种视图来显示查询结果:
    - UI查看
    - 网格视图
- 默认情况下, Neo4j数据浏览器以小字体显示节点或关系图, 并在UI视图中显示默认颜色

### ID 属性

在 Neo4j 中, `Id` 是节点和关系的默认内部属性. 这意味着, 
当我们创建一个新的节点或关系时, Neo4j 数据库服务器将为内部使用分配一个数字。它会自动递增

以相同的方式, Neo4j 数据库服务器为关系分配一个默认 Id 属性

- 节点的 Id 属性的最大值约为 35 亿
- 关系的 Id 属性的最大值约为 35 亿

### Caption

在 Neo4j 数据中, 当我们在 Neo4j DATA 浏览器中执行 `MATCH + RETURN` 命令以查看 UI 视图中的数据时, 
通过使用它们的 Id 属性显示节点和/或关系结果。它被称为 `CAPTION` 的 `id` 属性

## CQL 函数

### Neo4j CQL 字符串函数

Neo4J CQL 提供了一组 String 函数, 用于在 CQL 查询中获取所需的结果

- UPPER
- LOWER
- SUBSTRING
- REPLACE

### Neo4j CQL AGGREGATION 聚合函数

Neo4j CQL 提供了一些在 RETURN 子句中使用的聚合函数

- COUNT
- MAX
- MIN
- SUM
- AVG

### Neo4j CQL 关系函数

Neo4j CQL 提供了一组关系函数, 以在获取开始节点, 结束节点等细节时知道关系的细节

- STARTNODE
- ENDNODE
- ID
- TYPE

## Admin 管理员

### Neo4j 数据库备份和恢复

### Neo4j CQL 索引

### Neo4j CQL UNIQUE 约束

### Neo4j CQL DROP UNIQUE

## Neo4j 实战


本文通过一个实际的案例来一步一步教你使用 Cypher 来操作 Neo4j。
案例的节点主要包括人物和城市两类, 人物和人物之间有朋友、夫妻等关系, 
人物和城市之间有出生地的关系

1. 首先, 我们删除数据库中以往的图, 确保一个空白的环境进行操作

```
MATCH (n) DETACH DELETE n
```

2. 创建人物节点

```
CREATE (n:Person {name:"John"}) RETURN n
CREATE (n:Person {name:'Sally'}) RETURN n
CREATE (n:Person {name:'Steve'}) RETURN n
CREATE (n:Person {name:'Mike'}) RETURN n
CREATE (n:Person {name:'Liz'}) RETURN n
CREATE (n:Person {name:'Shawn'}) RETURN n
```

> * node-name: n
> * node-label-name: Person
> * <id>: 0
> * name: "John"

3. 创建地区节点

```
CREATE (n:Location {city:'Miami', state:'FL'})
CREATE (n:Location {city:'Boston', state:'MA'})
CREATE (n:Location {city:'Lynn', state:'MA'})
CREATE (n:Location {city:'Portland', state:'ME'})
CREATE (n:Location {city:'San Francisco', state:'CA'})
```

4. 创建人物之间的关系

```
MATCH (a:Person {name:"Shawn"}), (b:Person {name:"Sally"})
MERGE (a)-[:FRIENDS {since:2001}]->(b)

MATCH (a:Person {name:'Shawn'}), (b:Person {name:'John'}) 
MERGE (a)-[:FRIENDS {since:2012}]->(b)

MATCH (a:Person {name:'Mike'}), (b:Person {name:'Shawn'}) 
MERGE (a)-[:FRIENDS {since:2006}]->(b)

MATCH (a:Person {name:'Sally'}), (b:Person {name:'Steve'}) 
MERGE (a)-[:FRIENDS {since:2006}]->(b)

MATCH (a:Person {name:'Liz'}), (b:Person {name:'John'}) 
MERGE (a)-[:MARRIED {since:1998}]->(b)
```

5. 创建人物-地区之间的关系


```
MATCH (a:Person {name:"John"}), (b:Location {city:"Boston"})
MERGE (a)-[:BORN_IN {year:1978}]->(b)

MATCH (a:Person {name:'Liz'}), (b:Location {city:'Boston'}) 
MERGE (a)-[:BORN_IN {year:1981}]->(b)

MATCH (a:Person {name:'Mike'}), (b:Location {city:'San Francisco'}) 
MERGE (a)-[:BORN_IN {year:1960}]->(b)

MATCH (a:Person {name:'Shawn'}), (b:Location {city:'Miami'}) 
MERGE (a)-[:BORN_IN {year:1960}]->(b)

MATCH (a:Person {name:'Steve'}), (b:Location {city:'Lynn'}) 
MERGE (a)-[:BORN_IN {year:1970}]->(b)
```

6. 查询-所有在 Boston 出生的人物

```
MATCH (a:Person)-[:BORN_IN]->(b:Location {city:"Boston"})
RETURN a, b
```

7. 查询所有对外有关系的节点

```
MATCH (a)-->()
RETURN a
```

8. 查询所有有关系的节点

```
MATCH (a)--() 
RETURN a
```

9. 查询所有对外有关系的节点, 以及关系类型

```
MATCH (a)-[r]->()
RETURN a.name, type(r)
```

10. 查询所有有结婚关系的节点

```
MATCH (n)-[:MARRIED]-()
RETURN n
```

11. 创建节点的时候就建好关系

```
CREATE (a:Person {name:"Todd"})-[r:FRIENDS]->(b:Person {name:"Carlos"})
```

12. 查询某人的朋友的朋友

```
MATCH (a:Person {name:"Mike"})-[r1:FRIENDS]-()-[r2:FRIENDS]-(friend_of_a_friend)
RETURN friend_of_a_friend.name AS fofName
```

13. 增加、修改节点的属性

```
MATCH (a:Person {name:'Liz'}) SET a.age=34
MATCH (a:Person {name:'Shawn'}) SET a.age=32
MATCH (a:Person {name:'John'}) SET a.age=44
MATCH (a:Person {name:'Mike'}) SET a.age=25
```

14. 删除节点的属性

```
MATCH (a:Person {name:"Mike"})
SET a.test = "test"

MATCH (a:Person {name:"Mike"})
REMOVE a.test
```

15. 删除节点

```
MATCH (a:Location {city:"Portland"})
DELETE a
```

16. 删除有关系的节点

```
MATCH (a:Person {name:"Todd"})-[rel]-(b:Person)
DELETE a, b, rel
```

17. 查询所有节点、关系

```
MATCH (n) 
RETURN n
LIMIT 25
```

# py2neo

安装

```bash
$ pip install --upgrade py2neo
```

使用

```python
from py2neo import Graph
```

核心 API

- `Graph` class
    - `Subgraph` class
        - `Node` object
        - `Relationship` object

## py2neo.database

```python
from py2neo import Graph

graph = Graph(password = "password")
graph.run("UNWIND range(1, 3) AS n RETURN n, n * n as n_sq").to_table()
```

### 连接

- GraphService objects
- Graph
    - auto
    - begin
    - call
    - create
    - delete
    - delete_all()
    - evaluate
    - exists
    - match
    - match_one
    - merge
    - name
    - nodes
    - play
    - pull
    - push(subgraph)
    - `read(cypher, parameters = None, **kwargs)`
    - relationships
    - `run(cypher, parameters = None, **kwargs)`
    - schema
    - separate
    - service
- SystemGraph objects
- Schema objects
- GraphService objects
- ProcedureLibrary objects
- Procedure objects
    - class py2neo.database.Procedure(graph, name)

# 参考

* [py2neo](https://py2neo.readthedocs.io/en/latest/) 

