---
title: Python pip
author: 王哲峰
date: '2022-07-10'
slug: python-pip
categories:
  - python
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

- [TODO](#todo)
- [pip 介绍](#pip-介绍)
- [pip 安装](#pip-安装)
  - [Install with get-pip.py](#install-with-get-pippy)
  - [使用 Linux Package Managers](#使用-linux-package-managers)
  - [更新 pip](#更新-pip)
- [安装 package](#安装-package)
  - [Install a package from PyPI](#install-a-package-from-pypi)
  - [Install a package from PyPI or somewhere downloaded .whl file](#install-a-package-from-pypi-or-somewhere-downloaded-whl-file)
  - [Show what files were installed](#show-what-files-were-installed)
  - [List what package are outdated](#list-what-package-are-outdated)
  - [Upgrade a package](#upgrade-a-package)
  - [Uninstall a package](#uninstall-a-package)
- [更改 PyPi pip 源](#更改-pypi-pip-源)
  - [Windows](#windows)
  - [Linux 和 macOS](#linux-和-macos)
- [pipdeptree](#pipdeptree)
  - [安装](#安装)
  - [使用](#使用)
- [参考](#参考)
</p></details><p></p>

# TODO

* [关于 pip 的 15 个使用小技巧](https://mp.weixin.qq.com/s/2pxwZ15rA9wv9urPiOCuDg)

# pip 介绍

pip - The Python Package Installer

pip is the package install for Python. You can use pip to install 
packages from the Python Package Index(PyPI) and other indexs.

- package installer
    - https://packaging.python.org/guides/tool-recommendations/
- Python Package Index(PyPI)
    - https://pypi.org/

# pip 安装

## Install with get-pip.py

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

## 使用 Linux Package Managers

See [Installing pip/setuptools/wheel with Linux Package Managers](https://packaging.python.org/guides/installing-using-linux-tools/#installing-pip-setuptools-wheel-with-linux-package-managers) in the [Python Packaging User Guide](https://packaging.python.org/guides/tool-recommendations/).

## 更新 pip

- Linux / macOS

```bash
$ pip install -U pip
```

- Windows

```bash
C:/> python -m pip install -U pip
```

# 安装 package

## Install a package from PyPI

```bash
$ pip install SomePackage
```

## Install a package from PyPI or somewhere downloaded .whl file

```bash
$ pip install SomePackage-1.0-py2.py3-none-any.whl
```

## Show what files were installed

```bash
$ pip show --files SomePackage
```

## List what package are outdated

```bash
$ pip list --outdated
```

## Upgrade a package

```bash
$ pip install --upgrade SomePackage
```

## Uninstall a package

```bash
$ pip uninstall SomePackage
```

# 更改 PyPi pip 源

将 PyPi 的 pip 源更改为国内 pip 源，国内常用的 pip 源列表如下：

* 阿里云(较快)
    - http://mirrors.aliyun.com/pypi/simple/
* 清华大学
    - https://pypi.tuna.tsinghua.edu.cn/simple/
* 中国科学技术大学
    - https://pypi.mirrors.ustc.edu.cn/simple/
* 豆瓣(较快)
    - http://pypi.douban.com/simple/


## Windows

临时更改 PyPi pip 源：

```bash
$ pip3 install *** -i http://mirrors.aliyun.com/pypi/simple/
$ pip3 install *** -i http://mirrors.aliyun.com/pypi/simple/ --trusted-host mirrors.aliyun.com
```

永久更改 PyPi pip 源：

1. 打开 `C:\Users\username\AppData\Roaming\`
2. 在上述目录下新建文件夹 `pip`
3. 在 `pip` 文件夹中新建 `pip.ini` 文件
4. 在 `pip.ini` 文件中添加如下内容

```ini
[global]
timeout=6000
index-url=https://pypi.tuna.tsinghua.edu.cn/simple/
trusted-host=pypi.tuna.tsinghua.edu.cn
```

## Linux 和 macOS

临时更改 PyPi pip 源：

```bash
$ pip3 install *** -i http://mirrors.aliyun.com/pypi/simple/
$ pip3 install *** -i http://mirrors.aliyun.com/pypi/simple/ --trusted-host mirrors.aliyun.com
```

永久更改 PyPi pip 源：

1. 进入 `~` 目录

```bash
$ cd ~
```

2. 在 `~` 目录下新建 `.pip` 文件夹

```bash
$ mkdir .pip
```

3. 进入 `.pip` 文件夹

```bash
$ cd .pip
```

4. 创建文件 `pip.conf`

```bash
$ touch pip.conf
```

5. 在 `pip.conf` 中添加如下内容

```ini
[global]
timeout=6000
index-url=https://pypi.tuna.tsinghua.edu.cn/simple/
[install]
trusted-host=pypi.tuna.tsinghua.edu.cn
```

# pipdeptree

pipdeptree 用来产看某个 Python 环境中依赖库之间的依赖关系

## 安装

```bash
$ pip install pipdeptree
```

## 使用

```bash
$ pipdeptree
```

# 参考

* [pip reference guide list](https://pip.pypa.io/en/stable/reference/)

