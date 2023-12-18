---
title: Python 系统工具
author: 王哲峰
date: '2023-01-09'
slug: system
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

- [os](#os)
  - [os 模块中的工具](#os-模块中的工具)
    - [os 模块总的基本接口](#os-模块总的基本接口)
    - [查看 os 模块的属性](#查看-os-模块的属性)
    - [管理工具](#管理工具)
    - [可移植的常量](#可移植的常量)
  - [os.path 模块中的工具](#ospath-模块中的工具)
  - [在脚本里运行 shell 命令](#在脚本里运行-shell-命令)
  - [os 模块导出的其他工具](#os-模块导出的其他工具)
  - [os 示例](#os-示例)
- [sys](#sys)
  - [sys 示例](#sys-示例)
</p></details><p></p>

# os

`os` 模块包含了在 C 程序和 shell 脚本中经常用的所有操作系统调用. 它的调用涉及目录、进程和 shell 变量等. 

准确地说, 该模块提供了 POSIX 工具, 操作系统调用的跨平台移植标准, 以及不依赖平台的目录处理工具. 如内嵌模块 `os.path`.

在操作实践中, os 基本上作为计算机系统调用的可移植接口来使用: 用 os 和 os.path 编写的脚本通常无需改动即可在其他平台上运行. 在这些平台上, 
还包括了专属该平台的额外工具(比如 Unix 下的底层进程调用). 不过总的来说, 只要是技术上可行的, os 都能做到跨平台. 

## os 模块中的工具

### os 模块总的基本接口

- Shell 变量
    - os.environ
- 运行程序
    - os.system, os.popen, os.execv, os.spawnvv
- 派生进程
    - os.fork, os.pipe, ps.waitpid, os.kill
- 文件描述符, 文件锁
    - os.open, os.read, os.write
- 文件处理
    - os.remove, os.rename, os.mkfifo, os.mkdir, os.rmdir
- 管理工具
    - `os.getcwd`
        - 返回当前的工作目录, 当前的工作目录是脚本所打开的文档应当放置的位置
    - `os.chdir`
        - 改变目录
    - os.chmod
    - `os.getpid`
        - 给出调用的函数的进程 ID, 这是系统为当前运行程序定义的唯一标示符, 可用于进程控制和唯一命名
    - os.listdir, os.access
- 移植工具
    - os.sep, ps.pathsep, os.curdir, ps.path.split, os.path.join
- 路径名工具
    - os.path.exists("path"), os.path.isdir("path"), os.path.getsize("path")

### 查看 os 模块的属性

```python
>>> import os
>>> dir(os)
>>> dir(os.path)
```

### 管理工具

```python
>>> os.getpid()
>>> os.getcwd()
>>> os.chdir(r"C:\Users")
>>> os.getcwd()
```

### 可移植的常量


## os.path 模块中的工具

## 在脚本里运行 shell 命令

## os 模块导出的其他工具

- os.environ
    - 获取和设置 shell 环境变量
- os.fork 
    - 在类 Unix 系统下派生的子进程
- os.pipe
    - 负责程序间通信
- os.execlp
    - 启动新程序
- os.spawnvv
    - 启动带有底层控制的新程序
- os.open 
    - 打开基于底层描述符的文件
- os.mkdir
    - 创建新目录
- os.mkfifo
    - 创建新的命名管道
- os.stat
    - 获取文件底层信息
- os.remove 
    - 根据路径名删除文件
- os.walk
    - 将函数或循环应用于整个目录树的各个部分

说明:

* os 模块提供了一套文件处理调用, 如 open、read 和 write, 但所有这些都涉及底层的文件访问, 它们与用 Python 内建 open 函数
  创建的 stdio 文件截然不同. 通常情况下, 除了特殊的文件处理需求(比如用排他性访问文件锁打开文件), 你应当使用内建的 open 函数, 
  而不是 os 模块, 来处理所有文件. 

## os 示例

```python
#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os

#===========================================================
#                        codeing
#===========================================================
# os
print(dir(os))
print("-" * 100)
print(dir(os.path))
print("-" * 100)
print(os.getpid())
print("-" * 100)
print(os.getcwd())
print("-" * 100)
os.chdir(r"E:\project\projects\python")
print(os.getcwd())
os.chdir(r"E:\project\projects")
print("-" * 100)
print(os.pathsep) # ';'
print(os.sep)	  # '\'
print(os.pardir)  # '..'
print(os.curdir)  # '.'
print(os.linesep) # '\r\n'
print("-" * 100)
#===========================================================
# os.path
print(os.path.isdir(r'E:\project'))
print(os.path.isfile(r'E:\project'))
print(os.path.exists(r'E:\project\projects'))
print(os.path.getsize(r'E:\project\projects\test.py'))
print(os.path.split(r'E:\project\projects\test.py'))
print(os.path.join(r'E:\project', 'test.py'))
name = r'E:\project\test.py'
print(os.path.dirname(name), os.path.basename(name))
print(os.path.splitext(r'E:\project\projects\test.py'))
print(os.sep)
pathname = r'E:\project\projects\test.py'
print(os.path.split(pathname))
print(pathname.split(os.sep))
print(os.sep.join(pathname.split(os.sep)))
print(os.path.join(*pathname.split(os.sep)))
mixed = r'C:\temp\public/files/index.html'
print(mixed)
print(os.path.normpath(mixed))
print(os.path.normpath(r'C:\temp\\sub\.\file.ext'))
os.chdir(r'E:\project')

print(os.getcwd())
print(os.path.abspath(""))
print(os.path.abspath("projects"))
print(os.path.abspath(r"projects\python"))
print(os.path.abspath("."))
print(os.path.abspath(".."))
print(os.path.abspath(r"..\documents"))
print(os.path.abspath(r"E:\project\projects\test.py"))
print(os.path.abspath(r"E:\project\projects"))
print("*" * 100)

#===========================================================
# shell命令
os.chdir(r"E:\project\projects")
os.getcwd()
os.system("dir")
os.system("type sys_code.py")


open("test.py").read()
text = os.popen("type test.py").read()
print(text)
listing = os.popen("dir").readlines()
print(listing)

#===========================================================
# subprocess
import subprocess
subprocess.call("python test.py")
# subprocess.call("cmd /E 'type test.py'")
subprocess.call("type test.py", shell = True)
print('\n' + '-' * 100)

pipe = subprocess.Popen("python test.py", stdout = subprocess.PIPE)
print(pipe.communicate())
print(pipe.returncode)
```

# sys

## sys 示例

```python
#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys

#===========================================================
#                        codeing
#===========================================================

# sys
print(dir(sys))
print("-" * 100)
print(sys.__doc__)
print("-" * 100)
print(help(sys))
print("-" * 100)
print(sys.platform)
if sys.platform[:3] == "win":
    print("hello windows")
print("-" * 100)
print(sys.maxsize)
print("-" * 100)
print(sys.version)
print("-" * 100)
print(sys.path)
sys.path.append(r"E:\project")
print(sys.path)
print("-" * 100)
print(sys.modules)
print(list(sys.modules.keys()))
print(sys)
print(sys.modules["sys"])
print(sys.getrefcount(sys))
print(sys.builtin_module_names)
print("-" * 100)
try:
    raise IndexError
except:
    print(sys.exc_info())
print("-" * 100)

import traceback
def grail(x):
    raise TypeError("already got one")

try:
    grail("authur")
except:
    exc_info = sys.exc_info()
    print(exc_info[0])
    print(exc_info[1])
    print(exc_info[2])
    traceback.print_tb(exc_info[2])
```
