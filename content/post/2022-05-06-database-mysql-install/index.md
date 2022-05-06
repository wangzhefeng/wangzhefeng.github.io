---
title: MySQL 安装--Linux
author: 王哲峰
date: '2022-05-06'
slug: database-mysql-install
categories:
  - database
tags:
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

- [安装过程省略](#安装过程省略)
- [环境变量配置](#环境变量配置)
- [MySQL 服务的启停和状态查看](#mysql-服务的启停和状态查看)
- [启动 MySQL](#启动-mysql)
- [初始化设置](#初始化设置)
- [配置](#配置)
</p></details><p></p>


# 安装过程省略

# 环境变量配置

(1)在终端切换到根目录, 编辑 `~/.zshrc` 或者 `~/.bash_profile`

```bash
cd ~
vim ~/.zshrc
```

(2)在 `~/.zhsrc` 文件中添加配置项

```bash
export PATH=$PATH:/usr/local/mysql/bin
export PATH=$PATH:/usr/local/mysql/support-files
```

(3)保存并启用配置项

```bash
:wq
source ~/.zshrc
echo $PATH
```
# MySQL 服务的启停和状态查看

```bash
# 停止 MySQL 服务
sudo mysql.server stop
# 重启 MySQL 服务
sudo mysql.server restart
# 查看 MySQL 服务状态
sudo mysql.server status
```

# 启动 MySQL

```bash
sudo mysql.server start
mysql -u root -p
```

```bash
Welcome to the MySQL monitor.  Commands end with ; or \g.
Your MySQL connection id is 8
Server version: 8.0.17 MySQL Community Server - GPL

Copyright (c) 2000, 2019, Oracle and/or its affiliates. All rights reserved.

Oracle is a registered trademark of Oracle Corporation and/or its
affiliates. Other names may be trademarks of their respective
owners.

Type 'help;' or '\h' for help. Type '\c' to clear the current input statement.

mysql>
```

# 初始化设置

设置初始化密码, 进入数据库mysql数据库之后执行下面的语句, 设置当前root用户的密码为root. 

```bash
mysql> set password = password("123456");
mysql> exit
```

# 配置

```bash
cd /usr/local/mysql/support-files
cp my-default.cnf /Users/zfwang/Desktop
vim /Users/zfwang/Desktop/my-default.cnf

[mysqld]
default-storage-engine=INNODB
character-set-server=utf8
port = 3306

[client]
default-character-set=utf8

mv /Users/zfwang/Desktop/my-default.cnf /etc
```

```bash
mysql -u root -p
mysql> show variables like '%char%';
```