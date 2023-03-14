---
title: Java 连接 MySQL
author: 王哲峰
date: '2020-10-01'
slug: java-mysql
categories:
  - java
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

- [环境](#环境)
- [Java 连接 MySQL](#java-连接-mysql)
  - [创建测试数据](#创建测试数据)
  - [Java 连接数据库](#java-连接数据库)
</p></details><p></p>

Java 使用 JDBC 连接 MySQL 数据库

# 环境

- Java 连接 MySQL 需要驱动包，最新版下载地址为：http://dev.mysql.com/downloads/connector/j/，
  解压后得到 jar 库文件，然后在对应的项目中导入该库文件。

# Java 连接 MySQL

## 创建测试数据

在 MySQL 中创建 `RUNOOB` 数据库，并创建 `websites` 数据表，表结构如下：

```sql
CREATE TABLE `websites` (
    `id` int(11) NOT NULL AUTO_INCREMENT,
    `name` char(20) NOT NULL DEFAULT '' COMMENT '站点名称',
    `url` varchar(255) NOT NULL DEFAULT '',
    `alexa` int(11) NOT NULL DEFAULT '0' COMMENT 'Alexa 排名',
    `country` char(10) NOT NULL DEFAULT '' COMMENT '国家',
    PRIMARY KEY (`id`)
) ENGINE=InnoDB AUTO_INCREMENT=10 DEFAULT CHARSET=utf8;
```


```sql
INSERT INTO `websites` VALUES 
    ('1', 'Google', 'https://www.google.cm/', '1', 'USA'), 
    ('2', '淘宝', 'https://www.taobao.com/', '13', 'CN'), 
    ('3', '菜鸟教程', 'http://www.runoob.com', '5892', ''), 
    ('4', '微博', 'http://weibo.com/', '20', 'CN'), 
    ('5', 'Facebook', 'https://www.facebook.com/', '3', 'USA');
```

## Java 连接数据库

使用 JDBC 连接 MySQL 数据库.

```java
/* MySQLDemo.java */

package com.runoob.test;

import java.sql.*;

public class MySQLDemo {
    // ==========================================
    // MySQL JDBC驱动、数据库URL
    // ==========================================
    // MySQL 8.0 以下版本 - JDBC 驱动名及数据库 URL
    static final String JDBC_DRIVER = "com.mysql.jdbc.Driver";
    static final String DB_URL = "jdbc:mysql://localhost:3306/RUNOOB";

    // MySQL 8.0 以上版本 - JDBC 驱动名及数据库 URL
    // static final String JDBC_DRIVER = "com.mysql.cj.jdbc.Driver";  
    // static final String DB_URL = "jdbc:mysql://localhost:3306/RUNOOB?useSSL=false&allowPublicKeyRetrieval=true&serverTimezone=UTC";
    
    // ==========================================
    // 数据库的用户名与密码，需要根据自己的设置
    // ==========================================
    static final String USER = "root";
    static final String PASS = "123456";

    // ==========================================
    // 一套操作
    // ==========================================
    public static void main(String[] args) {
        Connection conn = null;
        Statement stmt = null;
        try {
            // 注册 JDBC 驱动
            Class.forName(JDBC_DRIVER);

            // 代开连接
            System.out.println("连接数据库...");
            conn = DriverManager.getConnection(DB_URL, USER, PASS);

            // 执行查询
            System.out.println("实例化Statement对象...");
            stmt = conn.createStatement();
            String sql;
            sql = "SELECT id, name, url FROM websites";
            ResultSet rs = stmt.executeQuery(sql);

            // 展开结果数据集数据库
            while(rs.next()) {
                // 通过字段检索
                int id = rs.getInt("id");
                String name = rs.getString("name");
                String url = rs.getString("url");

                // 输出数据
                System.out.print("ID: " + id);
                System.out.print(", 站点名称: " + name);
                System.out.print(", 站点 URL: " + url);
                System.out.print("\n");
            }
            // 完成后关闭
            rs.close();
            stmt.close();
            conn.close();
        }catch(SQLException se) {
            // 处理JDBC错误
            se.printStackTrace();
        }catch(Exception e) {
            // 处理 Class.forName 错误
            e.printStackTrace();
        }finally {
            // 关闭资源 stmt
            try {
                if (stmt != null) {
                    stmt.close();
                }
            }catch(SQLException se2) {
                // 什么都不做
            }

            // 关闭资源 conn
            try {
                if (conn != null) {
                    conn.close();
                }
            }catch(SQLException se) {
                se.printStackTrace();
            }
        }
        System.out.println("Goodbey!");
    }
}
```
