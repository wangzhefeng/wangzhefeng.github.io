---
title: Python 连接数据库
author: 王哲峰
date: '2022-10-03'
slug: python-database
categories:
  - Python
  - database
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

- [MySQL](#mysql)
  - [mysqlclient](#mysqlclient)
    - [安装](#安装)
    - [使用](#使用)
  - [mysql-connector-python](#mysql-connector-python)
    - [安装](#安装-1)
    - [使用](#使用-1)
  - [PyMySQL](#pymysql)
    - [安装](#安装-2)
    - [使用](#使用-2)
  - [aiomysql](#aiomysql)
    - [安装](#安装-3)
    - [使用](#使用-3)
</p></details><p></p>

# MySQL

## mysqlclient

mysqlclient 包是用于 MySQL 的最流行的 Python 包之一。它包含 MySQLdb 模块，
一个提供 Python 数据库 API 的 MySQL 接口

### 安装

Linux:

```bash
$ sudo apt-get install python3-dev default-libmysqlclient-dev build-essential
$ pip install mysqlclient
```

Windows:

```bash
$ pip install mysqlclient
```

### 使用

`.env` 文件

```
HOST
USERNAME
PASSWORD
DATABASE
VERIFY_IDENTITY
SSL_CERT
```

```python
import os

import MySQLdb

# 访问 .env 文件中的数据库凭证(SSL模式)
from dotenv import load_dotenv
load_dotenv()


# Create the connection object
connection = MySQLdb.connect(
    host = os.getenv("HOST"),
    user = os.getenv("USERNAME"),
    passwd = os.getenv("PASSWORD"),
    db = os.getenv("DATABASE"),
    ssl_mode = "VERIFY_IDENTITY",
    ssl = {
        'ca': os.getenv("SSL_CERT")
    }
)

# Create cursor and use it to execute SQL command
cursor = connection.cursor()
cursor.execute("select @@version")
version = cursor.fetchone()

if version:
    print('Running version: ', version)
else:
    print('Not connected.')
```

## mysql-connector-python

MySQL connector/Python 模块是 Oracle 支持的官方驱动，
用于通过 Python 连接 MySQL。该连接器完全是 Python 语言，
而 mysqlclient 是用 C 语言编写的。它也是独立的，
意味着它不需要 MySQL 客户端库或标准库以外的任何 Python 模块

注意，MySQL Connector/Python 不支持旧的 MySQL 服务器认证方法，
这意味着 4.1 之前的 MySQL 版本不能工作

### 安装

```bash
$ pip install mysql-connector-python
```

### 使用

```python
import os
from dotenv import load_dotenv
from mysql.connector import Error
import mysql.connector

load_dotenv()

connection = mysql.connector.connect(
host=os.getenv("HOST"),
database=os.getenv("DATABASE"),
user=os.getenv("USERNAME"),
password=os.getenv("PASSWORD"),
ssl_ca=os.getenv("SSL_CERT")
)

try:
    if connection.is_connected():
        cursor = connection.cursor()
    cursor.execute("select @@version ")
    version = cursor.fetchone()
    if version:
        print('Running version: ', version)
    else:
        print('Not connected.')
except Error as e:
    print("Error while connecting to MySQL", e)
finally:
    connection.close()
```

## PyMySQL

PyMySQL 包是另一个连接器，你可以用它来连接 Python 和 MySQL。
如果你追求速度，这是一个很好的选择，因为它比 mysql-connector-python 快


### 安装

```bash
$ pip install PyMySQL
```

### 使用

```python
from dotenv import load_dotenv
import pymysql
import os
load_dotenv()
connection = pymysql.connect(
    host=os.getenv("HOST"),
    database=os.getenv("DATABASE"),
    user=os.getenv("USERNAME"),
    password=os.getenv("PASSWORD"),
    ssl_ca=os.getenv("SSL_CERT")
)
cursor = connection.cursor()
cursor.execute("select @@version ")
version = cursor.fetchone()
if version:
    print('Running version: ', version)
else:
    print('Not connected.')
connection.close()
```

## aiomysql

aiomysql 库用于从 asyncio 框架访问 MySQL 数据库。除了是异步的特性之外，
连接代码与 PyMySQL 相似。注意，使用 aiomysql 需要 Python 3.7 以上版本和 PyMySQL

### 安装

```bash
$ pip install asyncio
$ pip install aiomysql
```

### 使用

```python
import os
import asyncio
import aiomysql
import ssl

from dotenv import load_dotenv
load_dotenv()

ctx = ssl.SSLContext(ssl.PROTOCOL_TLS_CLIENT)
ctx.load_verify_locations(cafile=os.getenv("SSL_CERT"))

loop = asyncio.get_event_loop()

async def connect_db():
   connection = await aiomysql.connect(
       host=os.getenv("HOST"),
       port=3306,
       user=os.getenv("USERNAME"),
       password=os.getenv("PASSWORD"),
       db=os.getenv("DATABASE"),
       loop=loop,
       ssl=ctx
   )
   cursor = await connection.cursor()
   await cursor.execute("select @@version")
   version = await cursor.fetchall()
   print('Running version: ', version)
   await cursor.close()
   connection.close()
loop.run_until_complete(connect_db())
```