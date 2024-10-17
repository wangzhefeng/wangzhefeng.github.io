---
title: Python 文件
author: wangzf
date: '2022-07-25'
slug: python-data-file
categories:
  - Python
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

- [打开文件](#打开文件)
- [使用文件](#使用文件)
- [文件工具](#文件工具)
  - [Python 3.X 中的文本和二进制文件](#python-3x-中的文本和二进制文件)
  - [在文件中存储并解析 Python 对象](#在文件中存储并解析-python-对象)
  - [用 pickle 存储 Python 原生对象](#用-pickle-存储-python-原生对象)
  - [文件中打包二进制数据的存储与解析](#文件中打包二进制数据的存储与解析)
  - [文件上下文管理](#文件上下文管理)
  - [其他工具](#其他工具)
</p></details><p></p>

内置 `open` 函数会创建一个 **Python 文件对象**, 可以作为计算机上的一个文件链接. 
在调用 `open` 之后, 你可以通过调用返回文件对象的方法来读写相关外部文件. 

文件对象不是数字、序列、对应. 相反, 文件对象只是常见文件处理任务输出模块. 
多数文件方法都与执行外部文件相关的文件对象的输入和输出有关, 
但其他文件方法可查找文件中的新位置、刷新输出缓存等. 

常见文件运算: 

| operation                           | explain                       |
|-------------------------------------|-------------------------------|
| output = open(r"C:\\spam", "w")     | create output file("w":write)  |
| input = open("data", "r")           |                               |
| input = open("data")                |                               |
| aString = input.read()              |                               |
| aString = input.read(N)             |                               |
| aString = input.readline()          |                               |
| aList = input.readlines()           |                               |
| output.write(aString)               |                               |
| output.writelines(aList)            |                               |
| output.close()                      |                               |
| output.flush()                       |                               |
| anyFile.seek(N)                     |                               |
| for line in open("data"): use line  |                               |
| open("f.txt", encoding = "latin-1") |                               |
| open("f.txt", "rb")                 |                               |


# 打开文件

为了打开一个文件, 程序会调用内置 `open` 函数, 
第一个参数是外部名, 接着是处理模式, 第三个是可选参数, 
用来控制输出缓存, 传入 `0` 意味着输出无缓存(写入方法调用时立即传给外部文件):

```python
open("name.file", "mode", 0)
```

模式类型如下:

- `r` 代表为输入打开文件, 默认值
    - `rb`
    - `r+`
- `w` 代表为输出生成并打开文件
    - `wb`
    - `w+`
- `a` 代表为在文件尾部追加内容而打开文件
    - `ab`
    - `a+`
- `*b` 在模式字符串尾部加上 `b` 可以进行二进制数据处理(行末转换和 Python 3.0 Unicode 编码被关闭了)
- `*+` 加上 `+` 意味着同时为输入和输出打开文件, 也就是说, 可以对相同文件对象进行读写, 往往与对文件中的修改的查找操作配合使用


# 使用文件

一旦存在一个文件对象, 就可以调用其他方法来读写相关的外部文件. 
在任何情况下, Python 程序中的文本文件都采用字符串的形式, 
读取文件时会返回字符串形式的文本, 文本作为字符串传递给 `write` 方法.

- 文件迭代器是最好的读取行工具
- 内容是字符串, 不是对象
- close 是通常 选项
- 文件是缓冲的并且是可查找的

# 文件工具

## Python 3.X 中的文本和二进制文件

## 在文件中存储并解析 Python 对象

## 用 pickle 存储 Python 原生对象

## 文件中打包二进制数据的存储与解析

## 文件上下文管理

文件上下文管理比文件自身多了一个异常处理功能, 它允许把文件处理代码包装到一个逻辑层中, 
以确保在退出后可以自动关闭文件, 而不是依赖于垃圾收集上的自动关闭: 

```python

    with open(r"C:\misc\data.txt") as myfile:
        for line in myfile:
            ...use line here...
```

`try/finally` 语句可以提供类似的功能, 但是需要一些额外代码的成本: 

```python
myfile = open(r"C:\misc\data.txt")
try:
    for line in myfile:
        ..use line here...
finally:
    myfile.close()
```

## 其他工具

```python
dir(filename)
help(filename)
```

- seek
    - 能够复位在文件中的当前位置(下次读写将应用在该位置上)
- flush
    - 能够强制性地将缓存输出写入磁盘(文件总会默认进行缓存)
- Pyhton 脚本中通向外部文件的接口: 
    - open
        - open 函数及其返回的文件对象是 Python 脚本中通向外部文件的主要接口
    - 标准流
        - 在 sys 模块中预先打开的文件对象, 例如 sys.stdout
    - os 模块中的描述文件
        - 处理整数文件, 支持诸如文件锁定之类的较低级工具
    - sockets, pipes 和 FIFO 文件
        - 文件类对象, 用于同步进程或者通过网络进行通信
    - 通过键来存取的文件
        - 通过键直接存储的不变的 Python 对象
    - Shell 命令流
        - 像 os.popen 和 subprocess.Popen 这样的工具, 支持产生 shell 命令, 并读取和写入都标准流
    - 第三方开源工具
        - PySerial 扩展中支持与窗口交流
        - pexpect 系统中的交互程序
