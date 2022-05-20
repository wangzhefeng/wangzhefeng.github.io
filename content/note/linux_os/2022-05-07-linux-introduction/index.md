---
title: Linux 介绍
author: 王哲峰
date: '2022-05-07'
slug: linux-introduction
categories:
  - Linux
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

- [什么是 Linux](#什么是-linux)
  - [Linux 有两种含义](#linux-有两种含义)
  - [Linux 的第一印象](#linux-的第一印象)
  - [Linux 环境准备](#linux-环境准备)
- [Linux 的内核版本及常见发行版](#linux-的内核版本及常见发行版)
  - [内核版本](#内核版本)
  - [发行版本](#发行版本)
- [安装 VirtualBox 虚拟机](#安装-virtualbox-虚拟机)
- [在虚拟机中安装 Linux 系统](#在虚拟机中安装-linux-系统)
- [第一次启动 Linux](#第一次启动-linux)
  - [图形页面](#图形页面)
  - [终端页面](#终端页面)
  - [终端的使用](#终端的使用)
  - [常见目录介绍](#常见目录介绍)
</p></details><p></p>


# 什么是 Linux

## Linux 有两种含义

- Linus 编写的开源操作系统的内核
    - Linus Benedict Torvalds(著名的电脑程序员)开发的系统
- 广义的操作系统

## Linux 的第一印象

- 服务端操作系统和客户端操作系统要做的事情不一样
- 命令行操作方式与图形界面的差异


## Linux 环境准备

- 执行环境
    - 云主机
    - 无数据的 PC
    - 虚拟机

# Linux 的内核版本及常见发行版

## 内核版本

- Linus Torvalds 开发的 Linux 内核: https://www.kernel.org/
- 内核版本分为三个部分
- 主版本号、此版本号、末版本号
- 次版本号是奇数为开发版本、偶数为稳定版本

## 发行版本

- RedHat Enterprise Linux        
    - 软件经过专业人员的测试
- Fedora
    - 组件社区免费提供软件
- CentOS
    - 基于 RedHat 编译    
- Debian        
    - 华丽的桌面
- Ubuntu
    - 华丽的桌面


#  安装 VirtualBox 虚拟机

- https://www.virtualbox.org/wik/Downloads

# 在虚拟机中安装 Linux 系统

- http://isoredirect.centos.org/centos/7/isos/x86_64/


# 第一次启动 Linux

## 图形页面

- root 用户账号、密码: 
    - ``root``
    - ``123456``
- 普通用户账号、密码: 
    - ``wangzf``
    - ``Tinker711235813``

## 终端页面

- 进入终端页面: 

```bash
# in Shell

init 3
```

- 登录用户: 
    - root 用户
        - ``root`` 
        - ``123456``
    - 普通用户: 
        - ``wangzf``
        - ``Tinker711235813``
- 退出当前登录用户(切换用户): 

```bash
# in Shell

exit
```

- 关机

```bash
# in Shell

init 0
```

## 终端的使用

- 终端(Shell)
    - 图形终端
    - 命令行终端
        - 服务器维护
    - 远程终端(SSH、VNC)

## 常见目录介绍

- ``/`` 根目录
- ``/root`` root 用户的家目录
- ``/home/username`` 普通用户的家目录
- ``/etc`` 配置文件目录
- ``/bin`` 命令目录
- ``/sbin`` 管理命令目录
- ``/usr/bin`` ``/usr/sbin`` 系统预装的其他命令

使用示例: 

```bash
ls /
ls /root
ls /bin
```