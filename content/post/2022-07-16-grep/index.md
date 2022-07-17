---
title: Shell grep & |
author: 王哲峰
date: '2022-07-16'
slug: grep
categories:
  - Linux
  - Shell
tags:
  - tool
---

# grep 简介

`grep` 命令用于文件内容的搜索，返回所有匹配的行

# grep 语法

```bash
$ grep pattern filename
```

# grep 参数

* `-i` 表示忽略大小写
* `-F` 指定确定的字符串
* `-r/R` 表示搜索某个目录下面的所有文件
* `-v` 过滤包含某个词的行，即 `grep` 的逆操作
* `-l` 只打印出符合条件的文件名而非文件内容字段

# grep 示例

* 找出指定目录下符合 pattern 的文件

```bash
$ grep admin /etc/passwd
_kadmin_admin:*:218:-2:Kerberos Admin Service:/var/empty:/usr/bin/false
_kadmin_changepw:*:219:-2:Kerberos Change Password Service:/var/empty:/usr/bin/false
_krb_kadmin:*:231:-2:Open Directory Kerberos Admin Service:/var/empty:/usr/bin/false
```

* 一般情况下，应该使用 `grep -r` 递归地找出当前目录下符合 `someVar` 的文件

```bash
$ grep -FR 'someVar'
```

* `grep` 默认搜索是大小写敏感的，使用 `-i` 忽略大小写

```bash
$ grep -iR 'someVar'
```

* 使用 `grep -l` 只打印出符合条件的文件名而非文件内容字段

```bash
$ grep -lR 'someVar'
```

* 如果你写的脚本或批处理任务需要上面的输出内容，
  可以使用 `while` 和 `read` 来处理文件名中的空格和其他特殊字符

```bash
$ grep -lR 'someVar' | while IFS= read -r file; do
      head "$file"
  done
```

* 如果你在你的项目里使用了版本控制软件，它通常会在 .svn， .git， .hg 目录下包含一些元数据。
  你也可以很容易地用 `grep -v` 把这些目录移出搜索范围，
  当然得用 `grep -F` 指定一个恰当且确定的字符串，即要移除的目录名

```bash
$ grep -R 'someVar' . | grep -vF '.svn'
```

* 部分版本的 grep 包含了 `—exclude` 和 `—exclude-dir` 选项，这看起来更加易读


- [官方文档](https://www.gnu.org/software/grep/manual/grep.html)
- [blog](https://www.bookstack.cn/read/bash-tutorial/docs-archives-commands-grep.md)