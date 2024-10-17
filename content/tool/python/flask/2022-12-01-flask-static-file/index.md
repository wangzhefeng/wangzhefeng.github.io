---
title: Flask 静态文件
author: wangzf
date: '2022-12-01'
slug: flask-static-file
categories:
  - Python
tags:
  - tool
---

动态的 web 应用也需要静态文件，一般是 CSS 和 JavaScript 文件。
理想情况下你的 服务器已经配置好了为你的提供静态文件的服务。
但是在开发过程中， Flask 也能做好 这项工作。
只要在你的包或模块旁边创建一个名为 static 的文件夹就行了。
静态文件位于应用的 ``/static`` 中。

使用特定的 'static' 端点就可以生成相应的 URL.

```python
from flask import url_for

# static/style.css
url_for("static", filename = "style.css")
```


