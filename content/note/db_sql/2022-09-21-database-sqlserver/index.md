---
title: SQL Server
author: 王哲峰
date: '2022-09-21'
slug: database-sqlserver
categories:
  - database
tags:
  - sql
---




```bash
$ sqlcmd
$ sqlcmd -S localhost -U SA -P Alvin123
```


```sql
-- 查看当前用户下的所有数据库
SELECT Name FROM SYS.DATABASES
GO

-- 新建数据库TestDB
CREATE DATABASE TestDB

-- 选中数据库TestDB
USE TestDB

-- 新建表Inventory
CREATE TABLE Inventory (id INT, name NVARCHAR(50), quantity INT)

-- 插入数据
INSERT INTO Inventory VALUES (1, 'banana', 150); 
INSERT INTO Inventory VALUES (2, 'orange', 154);
GO

-- 查询数据
SELECT * FROM Inventory WHERE quantity > 152;
GO
```

