---
title: py2so
author: 王哲峰
date: '2022-07-26'
slug: python-py2so
categories:
  - python
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
img {
    pointer-events: none;
}
</style>


<details><summary>目录</summary><p>

- [安装依赖库](#安装依赖库)
- [写一个测试 demo](#写一个测试-demo)
  - [新建加密脚本和测试脚本](#新建加密脚本和测试脚本)
  - [编译前测试文件夹目录](#编译前测试文件夹目录)
  - [编写测试文件脚本](#编写测试文件脚本)
  - [编译前测试](#编译前测试)
  - [编译前测试输出结果](#编译前测试输出结果)
- [编译加密](#编译加密)
  - [对脚本进行编译加密](#对脚本进行编译加密)
  - [生成文件夹目录](#生成文件夹目录)
- [运行加密后的文件](#运行加密后的文件)
  - [编译后测试](#编译后测试)
  - [编译后测试输出结果](#编译后测试输出结果)
</p></details><p></p>

> 对 Python 源代码进行加密(将`.py`文件编译为`.so`文件)

* 系统环境: Ubuntu 18.04
* Python 环境: Python 3.7.5

# 安装依赖库

```bash
$ sudo apt install python3-dev gcc
$ pip install cython
```

# 写一个测试 demo

## 新建加密脚本和测试脚本

- `script`
    - 项目目录
- `TodayModule.py`
    - 待编译的 `.py` 脚本
- `main.py`
    - 对 `TodayModule.py` 进行调用的脚本
- `setup.py`
    - 对 `TodayModule.py` 执行编译的脚本

```bash
$ mkdir script
$ cd script
$ touch TodayModule.py
$ touch main.py
$ touch setup.py
```

## 编译前测试文件夹目录

```
script
    ├── setup.py
    ├── main.py
    └── ToadyModule.py
```

## 编写测试文件脚本

* 待编译 `.py` 脚本

```python
# TodayModule.py

import datetime

class Today():
    def get_time(self):
        print(datetime.datetime.now())

    def say(self):
        print("Hello World!")
```

* 对 `TodayModule.py` 进行调用的脚本

```python
# main.py

from TodayModule import Today

t = Today()
t.get_time()
t.say()
```

* 对 `TodayModule.py` 执行编译的脚本

```python
# setup.py

from distutils.core import setup
from Cython.Build import cythonize

setup(ext_modules = cythonize(["TodayModule.py"]))
```

## 编译前测试

```bash
$ python3 main.py
```

## 编译前测试输出结果

```
2020-04-10 11:10:41.940473
Hello World!
```

# 编译加密

## 对脚本进行编译加密

```bash
$ cd ./py2so
$ python3 setup.py build_ext
```

## 生成文件夹目录

```
script
    ├── build
    │   ├── lib.linux-x86_64-3.7
    │   │   └── TodayModule.cython-37m-x86_64-linux-gnu.so
    │   └── temp.linux-x86_64-3.7
    │       └── TodayModule.o
    ├── setup.py
    ├── main.py
    ├── TodayModule.c
    └── TodayModule.py
```

# 运行加密后的文件

## 编译后测试

```bash
$ mv ./bulid/lib.lib.linux-x86_64-3.7/TodayModule.cython-37m-x86_64-linux-gnu.so .
$ rm -rf TodayModule.py
$ python3 main.py
```

## 编译后测试输出结果

```
2020-04-10 11:10:43.940473
Hello World!
```

