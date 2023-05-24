---
title: Flask 安装与启动
author: 王哲峰
date: '2022-12-01'
slug: flask-run
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

- [Flask 安装](#flask-安装)
  - [Flask 依赖](#flask-依赖)
  - [PyPI](#pypi)
  - [Flask 配置和惯例](#flask-配置和惯例)
- [Flask 启动](#flask-启动)
  - [示例应用](#示例应用)
  - [macOS, Linux 启动 Flask 应用](#macos-linux-启动-flask-应用)
  - [Windows 启动 Flask 应用](#windows-启动-flask-应用)
  - [启动 Flask 调试模式](#启动-flask-调试模式)
  - [最佳实践](#最佳实践)
</p></details><p></p>

# Flask 安装

## Flask 依赖

* 依赖
    - Werkzeug 用于实现 WSGI ，应用和服务之间的标准 Python 接口。
    - Jinja 用于渲染页面的模板语言。
    - MarkupSafe 与 Jinja 共用，在渲染页面时用于避免不可信的输入，防止注入攻击。
    - ItsDangerous 保证数据完整性的安全标志数据，用于保护 Flask 的 session cookie.
    - Click 是一个命令行应用的框架。用于提供 flask 命令，并允许添加自定义 管理命令。
* 可选依赖
    - Blinker 为 信号 提供支持。
    - SimpleJSON 是一个快速的 JSON 实现，兼容 Python’s json 模块。如果安装 了这个软件，那么会优先使用这个软件来进行 JSON 操作。
    - python-dotenv 当运行 flask 命令时为 通过 dotenv 设置环境变量 提供支持。
    - Watchdog 为开发服务器提供快速高效的重载。


## PyPI

```bash
$ pip install Flask
```

## Flask 配置和惯例

Flask 有许多带有合理缺省值的配置值和惯例。按照惯例，
模板和静态文件存放在应用的 Python 源代码树的子目录中，
名称分别为 templates 和 static。惯例是可以改变的，
但是你大可不必改变，尤其是刚起步的时候




# Flask 启动

## 示例应用

```python
# hello.py

from flask import Flask
app = Flask(__name__)

@app.route("/")
def hello_world():
    return "Hello, World!"
```

## macOS, Linux 启动 Flask 应用

方法 1

```bash
$ export FLASK_APP=hello.py
$ export FLASK_ENV=""
$ flask run
    * Running on http://127.0.0.1:5000/
```

方法 2

```bash
$ export FLASK_APP=hello.py
$ python -m flask run
    * Running on http://127.0.0.1:5000/
```

简单方法

```bash
$ python hello.py
```

## Windows 启动 Flask 应用

Command Prompt

```bash
$ C:\path\to\app>set Flask_APP=hello.py
```

PowerShell

```bash
$ PS C:\path\to\app> $env:FLASK_APP="hello.py"
```

## 启动 Flask 调试模式

1.激活调试器
2.激活自动重载
3.打开 Flask 应用的调试模式

macOS/Linux：

```bash
$ export FLASK_ENV=development
$ flask run
```

Windows：

```bash
$ C:\path\to\app>set FLASK_ENV=development
$ C:\path\to\app>flask run 
```

## 最佳实践

* 系统: macOS

```bash
$ export FLASK_APP=hello.py
$ export FLASK_ENV=development
$ flask run
    * Running on http://127.0.0.1:5000/
```

