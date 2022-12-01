---
title: Python 类型提示简介
author: 王哲峰
date: '2022-12-01'
slug: python-type-label
categories:
  - Python
tags:
  - tool
---




# 动机

## 类型提示

```python
def get_full_name(first_name, last_name):
    full_name = first_name.title() + " " + last_name.title()
    return full_name

print(get_full_name("john", "doe"))
```

```python
def get_full_name(first_name: str, last_name: str):
    full_name = first_name.title() + " " + last_name.title()
    return full_name

print(get_full_name("john", "doe"))
```

## 错误检查

```python
def get_name_with_age(name: str, age: int):
    name_with_age = name + "is this old:" + str(age)
    return name_with_age
```

# 声明类型

## 简单类型

* int
* float
* str
* bool
* bytes

```python
def get_items(item_a: str, item_b: int, item_c: float, item_d: bool, item_e: bytes):
    return item_a, item_b, item_c, item_d, item_e
```

## 嵌套类型

使用 `typing` 标准库来声明这些类型以及子类型

* 列表(list)
* 元组和集合(tuple, set)
* 字典(dict)

### List

示例

```python
from typing import List

def process_items(items: List[str]):
    for item in items:
        print(item)
```

### Tuple 和 Set

示例

```python
from typing import Tuple, Set

def process_items(items_t: Tuple[int, int, str], items_s: Set[bytes]):
    return items_t, items_s
```

### Dict

示例

```python
from typing import Dict

def process_items(prices: Dict[str, float]):
    for item_name, item_price in prices.items():
        print(item_name)
        print(item_price)
```

## 类作为类型

示例

```python
class Person:
    def __init__(self, name: str):
        self.name = name

def get_person_name(one_person: Person):
    return one_person.name
```

# Pydantic 模型

- Pydantic 是一个用来执行数据校验的 Python 库
    - 可以将数据的“结构”声明为具有属性的类，每个属性都拥有类型
    - 接着用一些值来创建这个类的实例，这些值会被校验，并被转换为适当的类型(在需要的情况下)，返回一个包含所有数据的对象
- 整个 FastAPI 建立在 Pydantic 的基础之上

示例

```python
class Person:
def __init__(self, name: str):
    self.name = name

def get_person_name(one_person: Person):
    return one_person.name
```

# FastAPI 中的类型提示

使用 FastAPI 时用类型提示声明参数可以获得

* 编辑器支持
* 类型检查
* 定义参数要求
    - 声明对请求路径参数、查询参数、请求头、请求体、依赖等的要求
* 转换数据
    - 将来自请求的数据转换为需要的类型
* 校验数据
    - 对每一个请求，当数据校验失败时自动生成错误信息返回给客户端
* 使用 OpenAPI 记录 API
    - 然后用于自动生成交互式文档的用户界面

