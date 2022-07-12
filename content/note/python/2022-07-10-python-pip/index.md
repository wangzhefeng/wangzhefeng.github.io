---
title: Python pip
author: 王哲峰
date: '2022-07-10'
slug: pip
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
</style>


<details><summary>目录</summary><p>

- [pip](#pip)
  - [安装 pip](#安装-pip)
    - [Install with `get-pip.py`](#install-with-get-pippy)
    - [使用 Linux Package Managers](#使用-linux-package-managers)
    - [更新 pip](#更新-pip)
  - [安装 package](#安装-package)
    - [Install a package from PyPI](#install-a-package-from-pypi)
    - [Install a package from PyPI or somewhere downloaded `.whl` file](#install-a-package-from-pypi-or-somewhere-downloaded-whl-file)
    - [Show what files were installed](#show-what-files-were-installed)
    - [List what package are outdated](#list-what-package-are-outdated)
    - [Upgrade a package](#upgrade-a-package)
    - [Uninstall a package](#uninstall-a-package)
  - [User Guide](#user-guide)
    - [更改 PyPi pip 源](#更改-pypi-pip-源)
    - [Others](#others)
  - [Reference Guide](#reference-guide)
- [pipdeptree](#pipdeptree)
  - [安装](#安装)
- [使用](#使用)
</p></details><p></p>


# pip

pip - The Python Package Installer

pip is the package install for Python. You can use pip to install 
packages from the Python Package Index(PyPI) and other indexs.

- package installer
    - https://packaging.python.org/guides/tool-recommendations/
- Python Package Index(PyPI)
    - https://pypi.org/

## 安装 pip

### Install with `get-pip.py`

1. 下载 `get-pip.py`
    - 方法一：直接从下面的连接下载
        - [get-pip.py](https://bootstrap.pypa.io/get-pip.py)
    - 方法二：使用 `curl`

```bash
$ curl https://bootstrap.pypa.io/get-pip.py -o get-pip.py
```

2. 安装 `pip`

```bash
$ python get-pip.py
```

> - `get-pip.py` 除了会安装 `pip`，还会安装：
>    - `setuptools`
>    - `wheel`

3. `get-pip.py` 选项
    - `--no-setuptools`
        - 不安装 `setuptools`
    - `--no-wheel`
        - 不安装 `wheel`
    - pip 安装选项
 
```bash
$ python get-pip.py --no-index --find-links=/local/copies

$ python get-pip.py --User

$ python get-pip.py --proxy="http://[user:password@]proxy.server:port"

$ python get-pip.py pip=9.0.2 wheel=0.30.0 setuptools=28.8.0
```

### 使用 Linux Package Managers

See [Installing pip/setuptools/wheel with Linux Package Managers](https://packaging.python.org/guides/installing-using-linux-tools/#installing-pip-setuptools-wheel-with-linux-package-managers) in the [Python Packaging User Guide](https://packaging.python.org/guides/tool-recommendations/).

### 更新 pip

- Linux / macOS

```bash
$ pip install -U pip
```

- Windows

```bash
C:/> python -m pip install -U pip
```

## 安装 package

### Install a package from PyPI

```bash
$ pip install SomePackage
```

### Install a package from PyPI or somewhere downloaded `.whl` file

```bash
$ pip install SomePackage-1.0-py2.py3-none-any.whl
```

### Show what files were installed

```bash
$ pip show --files SomePackage
```

### List what package are outdated

```bash
$ pip list --outdated
```

### Upgrade a package

```bash
$ pip install --upgrade SomePackage
```

### Uninstall a package

```bash
$ pip uninstall SomePackage
```

## User Guide

### 更改 PyPi pip 源

将 pipy 的 pip 源更改为国内 pip 源

* 国内 pip 源列表
    -  阿里云【较快】
        - `Simple Index <http://mirrors.aliyun.com/pypi/simple>`__
    -  清华大学
        - `Simple Index](https://pypi.tuna.tsinghua.edu.cn/simple/>`__
    -  中国科学技术大学
        - `Simple Index](https://pypi.mirrors.ustc.edu.cn/simple/>`__
    -  豆瓣【较快】
        - `Simple Index <http://pypi.douban.com/simple/>`__

* 临时更改 PyPi pip 源

```bash
$ pip3 install *** -i http://mirrors.aliyun.com/pypi/simple/
$ pip3 install *** -i http://mirrors.aliyun.com/pypi/simple/ --trusted-host mirrors.aliyun.com
```

* 永久更改 PyPi pip 源

```bash
$ cd ~
$ mkdir .pip
$ cd .pip
$ vim pip.conf

# [global]
# index-url = http://mirrors.aliyun.com/pypi/simple/
# [install]
# trusted-host = mirrors.aliyun.com
```

### Others

## Reference Guide

- [pip reference guide list](https://pip.pypa.io/en/stable/reference/)



# pipdeptree

pipdeptree 用来产看某个 Python 环境中依赖库之间的依赖关系

## 安装

```bash
$ pip install pipdeptree
```

# 使用

```bash
$ pipdeptree
```

