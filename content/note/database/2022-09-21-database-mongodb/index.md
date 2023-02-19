---
title: MongoDB
author: 王哲峰
date: '2022-09-21'
slug: database-mongodb
categories:
  - database
tags:
  - sql
---



```bash
$ sudo apt-get install mongodb
# $ Exec=bash /usr/local/robo3t-1.1.1-linux-x86_64-c93c6b0/bin/robo3t
$ /usr/local/robo3t-1.1.1-linux-x86_64-c93c6b0/bin/robo3t
```


```sql
show dbs
use db

# select
db.dbs.find().pretty()

# drop
db.dropDatabase()
```
