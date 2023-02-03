---
title: pydantic
author: 王哲峰
date: '2022-12-01'
slug: fastapi-pydantic
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
</style>

<details><summary>目录</summary><p>

- [pydantic 示例](#pydantic-示例)
- [pydantic 特性](#pydantic-特性)
- [pydantic 安装](#pydantic-安装)
  - [pydantic 依赖库](#pydantic-依赖库)
  - [pip 安装](#pip-安装)
  - [conda 安装](#conda-安装)
  - [测试安装](#测试安装)
- [pydantic 使用](#pydantic-使用)
- [Models](#models)
</p></details><p></p>


* Data validation and settings management using python type annotations.
- pydantic enforces type hints at runtime, and provides user friendly errors when data *s invalid.
* Define how data should be in pure, canonical python; validate it with pydantic.

# pydantic 示例

```python
from datetime import datetime
from typing import List, Optional
from pydantic import BaseModel
from pydantic import ValidationError


class User(BaseModel):
    id: int
    name = "John Doe"
    singnup_ts: Optional[datetime] = None
    friends: List[int] = []

external_data = {
    "id": "123",
    "singnup_ts": "2019-06-01 12:22",
    "friends": [1, 2, "3"],
}
user = User(**external_data)
print(user.id)
print(repr(user.singnup_ts))
print(user.friends)
print(user.dict())

try:
    User(signup_ts = "broken", friends = [1, 2, "not number"])
except ValidationError as e:
    print(e.json())
```

# pydantic 特性

* 与 IDE/linter/brain 配合的很好
* Pydantic 的 BaseSettings 类允许在 验证此请求数据、加载系统设置中使用
* 速度快
* 能够验证复杂结构
* 可扩展
* 数据类集成

# pydantic 安装

## pydantic 依赖库
  
* typing-extensions
* dataclasses
* backport(python 3.6)
* [email-validator](https://github.com/JoshData/python-email-validator)
* [python-dotenv](https://pypi.org/project/python-dotenv/)

## pip 安装

```bash
$ pip install pydantic
$ pip install "pydantic[email]"
$ pip install "pydantic[dotenv]"
$ pip install "pydantic[email,dotenv]"
$ pip install email-validation
$ pip install .
```

## conda 安装

```bash
$ codna install pydantic -c conda-forge
```

## 测试安装

```python
import pydantic
print("compiled", pydantic.compiled)
```

# pydantic 使用

# Models

