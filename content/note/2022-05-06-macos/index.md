---
title: mac OS
author: 王哲峰
date: '2022-05-06'
slug: macos
categories:
  - macOS
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

- [macOS 内容综述](#macos-内容综述)
- [macOS 系统操作](#macos-系统操作)
  - [查看 macOS 的版本号和编译版本号](#查看-macos-的版本号和编译版本号)
  - [查看系统相关信息](#查看系统相关信息)
  - [查看当前进程](#查看当前进程)
  - [杀进程](#杀进程)
</p></details><p></p>


# macOS 内容综述

1. macOS 背景介绍
2. macOS 系统操作
3. macOS 系统管理
4. macOS Shell
5. macOS 文本操作
6. macOS 服务管理
7. macOS Script
8. macOS 内核

# macOS 系统操作

## 查看 macOS 的版本号和编译版本号

```bash
$ sw_vers
```

## 查看系统相关信息

```bash
# 系统所有相关信息
$ uname -a

# 计算机硬件架构
$ uname -m

# 主机名称
$ uname -n

# 主机处理器类型
$ uname -p

# 内核发行版本号
uname -r

# 内核名称
$ uname -s

# 内核版本
$ uname -v

# Linux 操作系统名称
$ uname -o

# Linux 硬件平台
$ uname -i
```

## 查看当前进程

```bash
$ ps aux | less

# 显示所有进程
$ ps aux -A

# 显示终端中包括其他用户的所有进程
$ ps aux a

# 显示无控制终端的进程
$ ps aux x

# 查看系统中的每个进程
$ ps -A
$ ps -e

# 查看 非 root 运行的进程
$ ps -U root -u root

# 查看具体某个用户运行的进程
$ ps -u user1

# 查看运行中系统动态的视图
$ top
```

## 杀进程

```bash
$ kill
```