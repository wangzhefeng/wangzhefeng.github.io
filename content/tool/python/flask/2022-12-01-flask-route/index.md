---
title: Flask 路由
author: wangzf
date: '2022-12-01'
slug: flask-route
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

- [变量规则](#变量规则)
  - [示例](#示例)
- [唯一 URL、重定向](#唯一-url重定向)
- [URL 构建](#url-构建)
- [HTTP 方法](#http-方法)
</p></details><p></p>

使用 `route()` 装饰器来把函数绑定到 URL:

```python
from flask import Flask

@app.route("/")
def index():
    return "Index Page"

@app.route("/hello")
def hello():
    return "Hello, World"
```

# 变量规则

* 通过把 URL 的一部分标记为 `<variable_name>` 就可以在 URL 中添加变量。
  标记的部分会作为关键字参数传递给函数。
* 通过使用 `<converter:variable_name>`，可以选择性的加上一个转换器，为变量指定规则。转换器类型:

| 类型    | 说明                          |
|--------|-------------------------------|
| string | (缺省值)接受任何不包含斜杠的文本   |
| int    | 接受正整数                      |
| float   | 接受正浮点数                    |
| path   | 类似 string ，但可以包含斜杠     |
| uuid   | 接受 UUID 字符串                |


## 示例
    
```python
from markupsafe import escape

@app.route("/user/<username>")
def show_user_profile(username):
    """
    show the user profile for that user
    """
    return "User %s" % escape(username)

@app.route("/post/<int:post_id>")
def show_post(post_id):
    """
    show the post with the given id, the id is an integer
    """
    return "Post %d" % post_id

@app.route("/path/<path:subpath>")
def show_subpath(subpath):
    """
    show the subpath after /path/
    """
    return "Subpath %s" % escape(subpath)
```

# 唯一 URL、重定向

以下两条规则的不同之处在于是否使用尾部的斜杠

* `projects` 的 URL 是中规中矩的，尾部有一个斜杠，看起来就如同一个文件夹。 
  访问一个没有斜杠结尾的 URL 时 Flask 会自动进行重定向，帮你在尾部加上一个斜杠。
* `about` 的 URL 没有尾部斜杠，因此其行为表现与一个文件类似。
  如果访问这个 URL 时添加了尾部斜杠就会得到一个 404 错误。这样可以保持 URL 唯一，
  并帮助 搜索引擎避免重复索引同一页面。

```python
@app.route('/projects/')
def projects():
    return 'The project page'

@app.route('/about')
def about():
    return 'The about page'
```

# URL 构建

`url_for()` 函数用于构建指定函数的 URL。它把函数名称作为第一个参数。
它可以接受任意个关键字参数，每个关键字参数对应 URL 中的变量。
未知变量将添加到 URL 中作为查询参数。

为什么不在把 URL 写死在模板中，而要使用反转函数 url_for() 动态构建？

1. 反转通常比硬编码 URL 的描述性更好。
2. 你可以只在一个地方改变 URL ，而不用到处乱找。
3. URL 创建会为你处理特殊字符的转义和 Unicode 数据，比较直观。
4. 生产的路径总是绝对路径，可以避免相对路径产生副作用。
5. 如果你的应用是放在 URL 根路径之外的地方（如在 /myapplication 中，不在 / 中），`url_for()` 会为你妥善处理。

```python
from flask import Flask, usl_for
from markupsafe import escape

app = Flask(__name__)

@app.route("/")
def index():
    return 'index'

@app.route("/login")
def login():
    return "login"

@app.route("/user/<username>")
def profile(username):
    return "{}\'s profile".format(escape(username)

with app.test_request_context():
    print(url_for("index"))
    print(url_for("login"))
    print(url_for("login", next = "/"))
    print(url_for("profile", username = "John Doe"))
```

# HTTP 方法

Web 应用使用不同的 HTTP 方法处理 URL 。当你使用 Flask 时，应当熟悉 HTTP 方法。 
缺省情况下，一个路由只回应 GET 请求。 
可以使用 route() 装饰器的 methods 参数来处理不同的 HTTP 方法:

如果当前使用了 GET 方法，Flask 会自动添加 HEAD 方法支持，
并且同时还会 按照 HTTP RFC 来处理 HEAD 请求。
同样，OPTIONS 也会自动实现。

```python
from flask import Flask
from flask import request

app = Flask(__name__)

@app.route("/login", methods = ["GET", "POST"])
def login():
    if request.method == "POST":
        return do_the_login()
    else:
        return show_the_login_form()
```
