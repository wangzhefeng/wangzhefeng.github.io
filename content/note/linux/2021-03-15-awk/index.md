---
title: Awk 脚本工具
author: 王哲峰
date: '2021-03-15'
slug: awk
categories:
  - Linux
  - Shell
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

- [Awk 简介](#awk-简介)
- [Awk 语法](#awk-语法)
  - [awk 基本用法](#awk-基本用法)
    - [awk 基本用法形式](#awk-基本用法形式)
    - [awk 基本用法示例](#awk-基本用法示例)
  - [awk 变量](#awk-变量)
  - [函数](#函数)
    - [内置函数](#内置函数)
    - [示例](#示例)
  - [条件](#条件)
    - [条件语法](#条件语法)
    - [条件示例](#条件示例)
  - [if 语句](#if-语句)
    - [if 语句语法](#if-语句语法)
    - [if 语句示例](#if-语句示例)
- [参考资料](#参考资料)
</p></details><p></p>

# Awk 简介

awk 是处理文本文件的一个应用程序，几乎所有的 Linux 系统都自带这个程序。

awk 依次处理文件的每一行，并读取里面的每一个字段。
对于日志、CSV 那样的每行格式相同的文本文件，awk 可能是最方便的工具

awk 其实不仅仅是工具软件，还是一种编程语言

# Awk 语法

## awk 基本用法

### awk 基本用法形式

```bash
# 格式
$ awk 动作 文件名
```

### awk 基本用法示例

示例 1:

```bash
$ awk '{ print $0 }' demo.txt
```

* 示例执行效果就是把每一行原样打印出来
* `demo.txt` 是 awk 要处理的文本文件
* `demo.txt` 前面单引号内部有一个大括号，里面就是每一行的处理动作
* `print $0`
    - `print`: 打印
    - `$0`: 当前行

示例 2:

```bash
$ echo 'this is a test' | awk '{print $0}'
this is a test
```

* 示例执行的效果就是把标准输入 `this is a test` 重新打印一遍

示例 3:

```bash
$ echo 'this is a test' | awk '{print $3}'
a
```

* awk 会根据空格和制表符，将每一行分成若干字段，
  依次用 `$1`、`$2`、`$3` 代表第一个字段、第二个字段、第三个字段
* 这个示例中 `$3` 代表 `this is a test` 的第三个字段 `a`

示例 4:

```
# 把 /etc/passwd 文件保存为 demo.txt

root:x:0:0:root:/root:/usr/bin/zsh
daemon:x:1:1:daemon:/usr/sbin:/usr/sbin/nologin
bin:x:2:2:bin:/bin:/usr/sbin/nologin
sys:x:3:3:sys:/dev:/usr/sbin/nologin
sync:x:4:65534:sync:/bin:/bin/sync
```

```bash
$ awk -F ':' '{ print $1 }' demo.txt
root
daemon
bin
sys
sync 
```

* 这个文件的分隔符号是冒号(`:`)，所以用 `-F` 参数指定分隔符为冒号。
  然后，才能提取到它的第一个字段

## awk 变量

* `$数字`: 表示某个字段
* `NF`: 表示当前行有多少个字段
    - `$NF`: 表示最后一个字段
    - `$(NF-1)`: 表示倒数第二个字段

```bash
$ echo 'this is a test' | awk '{print $NF}'
test
```

```bash
# demo.txt
root:x:0:0:root:/root:/usr/bin/zsh
daemon:x:1:1:daemon:/usr/sbin:/usr/sbin/nologin
bin:x:2:2:bin:/bin:/usr/sbin/nologin
sys:x:3:3:sys:/dev:/usr/sbin/nologin
sync:x:4:65534:sync:/bin:/bin/sync

# print 命令里面的逗号表示输出的时候，两个部分之间使用空格分隔
$ awk -F ':' '{print $1, $(NF-1)}' demo.text
root /root
daemon /usr/sbin
bin /bin
sys /dev
sync /bin
```

* `NR`: 表示当前处理的是第几行

```bash
# print 命令里面，如果原样输出字符，要放在双引号里面
$ awk -F ':' '{print NR ") " $1}' demo.txt
1) root
2) daemon
3) bin
4) sys
5) sync
```

* FILENAME: 当前文件名
* FS: 字段分隔符，默认是空格和制表符
* RS: 行分隔符，用于分隔每一行，默认是换行符
* OFS: 输出字段的分隔符，用于打印时分隔字段，默认是空格
* ORS: 输出记录的分隔符，用于打印时分隔记录，默认为换行符
* OFMT: 数字输出的格式，默认为 `%.6g`

## 函数

### 内置函数

awk 提供了一些内置函数，方便对原始数据处理

* `toupper()`: 用于将字符转换为大写
* `tolower`: 用于将字符转换为小写
* `length()`: 返回字符串长度
* `substr():` 返回子字符串
* `sin()`: 正弦
* `cos()`: 余弦
* `sqrt()`: 平方根
* `rand()`: 随机数

### 示例

```bash
$ awk -F ':' '{print toupper($1)}' demo.txt
ROOT
DAEMON
BIN
SYS
SYNC
```

## 条件

### 条件语法

awk 允许指定输出条件，只输出负荷条件的行。输出条件要写在动作的前面

```bash
$ awk '条件 动作' 文件名
```

### 条件示例

* `/usr/` 为一个正则表达式，只输出包含 `usr` 的行

```bash
$ awk -F ':' '/usr/ {print $1}' demo.txt
root
daemon
bin
sys
```

* 输出奇数行

```bash
$ awk -F ':' 'NR % 2 == 1 {print $1}' demo.txt
root
bin
sync
```

* 输出第三行以后的行

```bash
$ awk -F ':' 'NR > 3 {print $1}' demo.txt
sys
sync
```

* 输出第一个字段等于指定值的行

```bash
$ awk -F ':' '$1 == "root" {print $1}' demo.txt
root

$ awk -F ':' '$1 === "root" || $1 == "bin" {print $1}' demo.txt
root
bin
```

## if 语句

awk 提供了 `if` 结构，用于编写复杂的条件

### if 语句语法

```bash
$ awk '{if (condition) 动作}' 文件名
$ awk '{if (condition) 动作; else 动作}' 文件名
```

### if 语句示例

```bash
$ awk -F ':' '{if ($1 > "m") print $1}' demo.txt
root
sys
sync
```

```bash
$ awk -F ':' '{if ($1 > "m") print $1; else print "---"}' demo.txt
```

# 参考资料

- [用户手册](https://www.gnu.org/software/gawk/manual/html_node/index.html)
- [blog](https://www.ruanyifeng.com/blog/2018/11/awk.html)

