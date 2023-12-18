---
title: Python 数据校验、类型提示
author: 王哲峰
date: '2022-07-25'
slug: python-data-validation
categories:
  - python
tags:
  - tool
---



Python 数据校验、类型提示库: 

- pydantic
- typing
- validators
- voluptuous

# Python 类型提示简介

## 动机

- 类型提示

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

- 错误检查

```python
def get_name_with_age(name: str, age: int):
   name_with_age = name + "is this old:" + str(age)
   return name_with_age
```

## 声明类型

### 简单类型

- int
- float
- str
- bool
- bytes

```python
def get_items(item_a: str, item_b: int, item_c: float, item_d: bool, item_e: bytes):
   return item_a, item_b, item_c, item_d, item_e
```

### 嵌套类型

使用 `typing` 标准库来声明这些类型以及子类型

- 列表(list)
- 元组和集合(tuple, set)
- 字典(dict)

#### List

- 示例

```python
from typing import List

def process_items(items: List[str]):
    for item in items:
        print(item)
```

#### Tuple 和 Set

- 示例

```python

   from typing import Tuple, Set

   def process_items(items_t: Tuple[int, int, str], items_s: Set[bytes]):
      return items_t, items_s

#### Dict

- 示例

```python
from typing import Dict

def process_items(prices: Dict[str, float]):
    for item_name, item_price in prices.items():
        print(item_name)
        print(item_price)
```

## 类作为类型

- 示例

```python   
class Person:
    def __init__(self, name: str):
        self.name = name

def get_person_name(one_person: Person):
    return one_person.name
```

## Pydantic 模型

- Pydantic 是一个用来执行数据校验的 Python 库
   - 可以将数据的“结构”声明为具有属性的类, 每个属性都拥有类型
   - 接着用一些值来创建这个类的实例, 这些值会被校验, 并被转换为适当的类型(在需要的情况下), 返回一个包含所有数据的对象
- 整个 FastAPI 建立在 Pydantic 的基础之上
- 示例

```python
class Person:
    def __init__(self, name: str):
    self.name = name

def get_person_name(one_person: Person):
    return one_person.name
```

## FastAPI 中的类型提示

使用 FastAPI 时用类型提示声明参数可以获得

- 编辑器支持
- 类型检查
- 定义参数要求
   - 声明对请求路径参数、查询参数、请求头、请求体、依赖等的要求
- 转换数据
   - 将来自请求的数据转换为需要的类型
- 校验数据
   - 对每一个请求, 当数据校验失败时自动生成错误信息返回给客户端
- 使用 OpenAPI 记录 API
   - 然后用于自动生成交互式文档的用户界面

# pydantic

   - Data validation and settings management using python type annotations.
   - pydantic enforces type hints at runtime, and provides user friendly errors when data is invalid.
   - Define how data should be in pure, canonical python; validate it with pydantic.

## pydantic 示例

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

## pydantic 特性

- 与 IDE/linter/brain 配合的很好
- Pydantic 的 BaseSettings 类允许在 验证此请求数据、加载系统设置中使用
- 速度快
- 能够验证复杂结构
- 可扩展
- 数据类集成

## pydantic 安装

- pydantic 依赖库
   - typing-extensions
   - dataclasses
   - backport(python 3.6)
   - `email-validator <https://github.com/JoshData/python-email-validator>`_ 
   - `python-dotenv <https://pypi.org/project/python-dotenv/>`_ 
- pip 安装

    ```bash
    $ pip install pydantic
    $ pip install "pydantic[email]"
    $ pip install "pydantic[dotenv]"
    $ pip install "pydantic[email,dotenv]"
    $ pip install email-validation
    $ pip install .
    ```

- conda 安装

    ```bash
    $ codna install pydantic -c conda-forge
    ```

- GitHub 源码安装

    ```bash
    $ pip install git+git://github.com/samuelcolvin/pydantic@master#egg=pydantic
    # or with extras
    $ pip install git+git://github.com/samuelcolvin/pydantic@master#egg=pydantic[email,dotenv]
    ```

- 使用 cython 编译, 使性能提高 30-50%
- 测试安装

    ```python
    import pydantic
    print("compiled", pydantic.compiled)
    ```

## pydantic 使用

### Models


# validators

## 安装 validators

```bash
$ pip install validators
```

## 基础的 validators

在 `validators` 中每一个 `validator` 是一个简单的函数, 
函数参数为要验证的值, 一些函数可能有额外的关键字参数. 
对于每一个函数, 如果验证成功, 则返回 `True`; 
若验证失败, 则返回一个 `ValidationFailure` 对象. 

1. validators.between(value, min = None, max = None)
2. validators.domain(value)

- 验证 value 是否是一个有效域名

3. validators.email(value, whitelist = None)

- 验证是否是合法的邮件地址 

4. validatorss.iban(value)

- 验证是否是合法的国际银行账号号码

5. validators.ip_address.ipv4(value)

- 验证是否是合法的 ipv4 地址

6. validators.ip_address.ipv6(value)

- 验证是否是合法的ipv6地址

7. validators.length(value, min = None, max = None)

- 验证给定的字符串长度是否在指定范围内

8. validators.mac_address(value)

- 验证是否是合法的 mac 地址

9.  validators.slug(value)

- 验证是否是合法的 slug

10. validators.truthy(value)
11. validators.url(value, public = False)

- 验证是否是合法的 url

12. validators.i18n.fi.fi_business_id(business_id)

- 验证 Finnish Business ID

13. validators.i18n.fi.fi_ssn(ssn)

- 验证 Finnish Social Security Number

## 装饰器、自定义验证函数

- API

```python
validators.utils.validator(func, *args, **kwargs)
validators.utils.ValidationFailure(func, args)
```

- 示例

```python
@validator
def is_even(value):
    return not (value % 2)

@validator
def is_positive(value):
    return value > 0

@validator
def is_string(value):
    return isinstance(value, str)

if __name__ == "__main__":
    print is_even(2)
    print is_even(3)
    print is_positive(4)
    print is_positive(0)
    print is_positive(-1)
    print is_string("hello")
    print is_string(3)
```

# validator

## 安装 validator

```bash
$ pip install validator.py
```

## 示例

```python
from validator import validate
from validator import Required, Not, Truthy, Blank, Range, Equals, In

rules = {
    "foo": [Required, Equals(123)],
    "bar": [Required, Truthy()],
    "baz": [In(["spam", "eggs", "bacon"])],
    "qux": [Not(Range(1, 100))]
}
passes = {
    "foo": 123,
    "bar": True,
    "baz": "spam",
    "qux": 101,
}
validate(rules, passes)

fails = {
    "foo": 321,
    "bar": False,
    "baz": "barf",
    "qux": 99
}
validate(rules, fails)
```

## validator 内置验证器

- Equals("")
- Required
- Truthy()
- Range(value1, value2)
- Pattern("")
- In([])
- Not()
- InstanceOf(value)
- SubclassOf(value)
- Length(value, minimum, maximum)

## 条件验证

```python
pet = {
    "name": "whiskers",
    "type": "cat",
}

cat_name_rules = {
    "name": [In("whiskers", "fuzzy", "tiger")]
}
dog_name_rules = {
    "name": [In("spot", "ace", "bandit")]
}
validation = {
    "type": [
        If(Equals("cat"), Then(cat_name_rules)),
        If(Equals("dog"), Then(dog_name_rules))
    ]
}
validate(validation, pet)
```

## 嵌套验证

```python
validator = {
    "foo": [Required, Equals(1)],
    "bar": [
        Required, 
        {
        "baz": [],
        "qux": [
            Required, 
            {
                "quux": [Required, Equals(3)]
            }
        ]
        }
    ]
}
test_case = {
    "foo": 1,
    "bar": {
        "baz": 2,
        "qux": {
        "quux": 3
        }
    }
}
validate(validator, test_case)
```

## 自定义 validator 验证器

```python
dictionary = {
    "foo": "bar"
}
validation = {
    "foo": [lambda x: x == "bar"]
}
validate(validation, dictionary)
```



# voluptuous

## 安装 voluptuous

```bash
$ pip install voluptuous
```

## voluptuous 字典数据验证

### 验证数据类型

1. 先定义一个 schema

```python
import traceback
from voluptuous import Schema, MultipleInvalid

schema = Schema({
    "q": str,
    "per_page": int,
    "page": int,
})
```

2. 待验证数据

```python
data = {
    "q": "hello world",
    "per_page": 20,
    "page": 10,
}
```

3. 验证数据

```python
try:
    schema(data)
except MultipleInvalid as e:
    print(e.errors)
```

### 验证必须字段

```python
from voluptuous import Schema, MultipleInvalid

schema = Schema({
    "q": str,
    "per_page": int,
    "page": int,
})
data = {
    "q": "hello world",
    "page": 10
}
schema(data)
```

```python
from voluptuous import Schema, Required, MultipleInvalid

schema = Schema({
    "q": str,
    Required("per_page"): int,
    "page": int,
})

data = {
    "q": "hello world",
    "page": 10,
}

try:
    schema(data)
except MultipleInvalid as e:
    print(e.errors)
```

### 验证数据长度、数据值范围

```python
from voluptuous import Required, All, Length, Range

schema = Schema({
    Required("q"): All(str, Length(min = 1)),
    Required("per_page", default = 5): All(int, Range(min = 1, max = 20)),
    "page": All(int, Range(min = 0)),
})
```

## voluptuous 验证其他类型数据

### 字面值(Literals)

```python
from voluptuous import Schema

schema = Schema(1)
# success
schema(1)
# error
schema(2)

schema = Schema("a string")
# success
schema("a string")
```

### 类型(types)

```python
from voluptuous import Schema

schema = Schema(int)

# success
schema(1)

# error
schema("one")
```

### ULRs

```python
from voluptuous import Schema, Url

schema = Schema(Url())

# success
schema("http://w3.org")

# error
try:
    schema("one")
    raise AssertionError("MultipleInvalid not raised")
except: MultipleInvalid as e:
    print(e.errors)
```

### Lists

```python
from voluptuous import Schema

schema = Schema([1, "a", "string"])
schema([1])
schema([1, 1, 1])
schema(["a", 1, "string", 1, "string"])
```

```python
from voluptuous import Schema

schema = Schema(list)
schema([])
schema([1, 2])
```

```python
from voluptuous import Schema

schema = Schema([])

# error
try:
    schema([1])
    raise AssertionError("MultipleInvalid not raised")
except MultipleInvalid as e:
    print(e.errors)

# success
schema([])
```

### 自定义函数

```python
from datetime import datetime

def Date(fmt = "%Y-%m-%d"):
    return lambda v: datetime.strptime(v, fmt)

schema = Schema(Date())
schema("2013-03-03")
try:
    schema("2013-03")
    raise AssertionError("MultipleInvalid not raised")
except MultipleInvalid as e:
    print(e.errors)
```

### 字典

- 待验证的数据中每一个键值对需要在字典中已定义, 否则, 验证失败

```python
schema = Schema({
    1: "one",
    2: "two",
})
# success

schema({1: "one"})
```

- 验证数据中有额外的键值对, 并且这种情况下不认为是错误的

```python
from voluptuous import ALLOW_EXTRA

schema = Schema({2: 3}, extra = ALLOW_EXTRA)

# success
schema({1: 2, 2: 3})
```

- 移除额外的键

```python
from voluptuous import Schema, REMOVE_EXTRA

schema = Schema({2: 3}, extra = REMOVE_EXTRA)

# success
schema({1: 2, 2: 3})
```

- 默认情况下, 在字典模式 schema 中定义的 key-value 对, 待验证的数据中不需要完全覆盖

```python
schema = Schema({1: 2, 3: 4})
schema({3: 4})
```

- 完全覆盖

```python
from voluptuous import Schema

schema = Schema({1: 2, 3: 4}, required = True)

# error
try:
    schema({3: 4})
    raise AssertionError("MultipleInvalid not raised")
except MultipleInvalid as e:
    print(e.errors)

# success
schema({1: 2, 3: 4})
```

- 仅设置必须含有其中某一个键

```python
from voluptuous import Schema, Required

schema = Schema({
    Required(1): 2, 
    3: 4
})

# error
try:
    schema({3: 4})
    raise AssertionError("MultipleInvalid not raised")
except MultipleInvalid as e:
    print(e.errors)

# success
schema({1: 2})
```

- 仅对某一个键设置可选择属性

```python
from voluptuous import Schema, Optional

schema = Schema({
    1: 2,
    Optional(3): 4,
}, required = True)

# error
try:
    schema({})
except MultipleInvalid as e:
    print(e.errors)

# success
schema({1: 2})

# error
try:
    schema({1: 2, 4: 5})
    raise AssertionError("MultipleInvalid not raised")
except MultipleInvalid as e:
    print(e.errors)

# success
schema({1: 2, 3: 4})
```

