---
title: FastAPI quick start
author: wangzf
date: '2022-12-01'
slug: fastapi-quickstart
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

- [FastAPI 介绍](#fastapi-介绍)
- [FastAPI 安装](#fastapi-安装)
  - [Requirements](#requirements)
  - [Installation](#installation)
- [FastAPI 示例](#fastapi-示例)
  - [创建一个 `main.py`](#创建一个-mainpy)
  - [运行](#运行)
  - [检查](#检查)
  - [交互式 API 文档](#交互式-api-文档)
- [升级示例](#升级示例)
  - [修改 main.py 文件来从 `PUT` 请求中接收请求体](#修改-mainpy-文件来从-put-请求中接收请求体)
  - [升级交互式 API 文档](#升级交互式-api-文档)
- [OpenAPI](#openapi)
</p></details><p></p>

# FastAPI 介绍

FastAPI 是一个用于构建 API 的现代、快速(高性能)的 web 框架，使用 Python 3.6+ 并基于标准的 Python 类型提示。

* 官方文档: https://fastapi.tiangolo.com
* 源代码: https://github.com/tiangolo/fastapi
* 关键特性:
    - 快速: 可与 NodeJS 和 Go 比肩的极高性能(归功于 Starlette 和 Pydantic)。最快的 Python web 框架之一。
    - 高效编码: 提高功能开发速度约 200％ 至 300％。
    - 更少 bug: 减少约 40％ 的人为(开发者)导致错误。
    - 智能: 极佳的编辑器支持。处处皆可自动补全，减少调试时间。
    - 简单: 设计的易于使用和学习，阅读文档的时间更短。
    - 简短: 使代码重复最小化。通过不同的参数声明实现丰富功能。bug 更少。
    - 健壮: 生产可用级别的代码。还有自动生成的交互式文档。
    - 标准化: 基于(并完全兼容)API 的相关开放标准: OpenAPI (以前被称为 Swagger) 和 JSON Schema。
* 主要内容:
    - Typer, 命令行中的 FastAPI
       - Typer 是 FastAPI 的小同胞。它想要成为命令行中的 FastAPI。
    - FastAPI

# FastAPI 安装

## Requirements

* Python 3.6 +
* FastAPI 站在巨人的肩膀上:
    - `Starlette`: 负责 web 部分
    - `Pydantic`: 负责 data 部分

## Installation

* 安装 fastAPI

```bash
# FastAPI
$ pip install fastapi

# 安装所有依赖
$ pip install "fastapi[all]"
```

* 生产环境中需要一个 ASGI 服务器，
  可以使用 [Uvincorn](http://www.uvicorn.org/) 或者 [Hypercorn](https://gitlab.com/pgjones/hypercorn)

```bash
$ pip install "uvicorn[standard]"
```

# FastAPI 示例

## 创建一个 `main.py`

```python
from typing import Optional
from fastapi import FastAPI

app = FastAPI()

@app.get("/")
def read_root():
    return {
        "Hello": "World"
    }

@app.get("/items/{item_id}")
def read_item(item_id: int, q: Optional[str] = None):
    return {
        "item_id": item_id, 
        "q": q
    }
```

或者使用 `async def...`

```python
from typing import Optional
from fastapi import FastAPI

app = FastAPI()

@app.get("/")
async def read_root():
    return {
        "Hello": "World"
    }

@app.get("/items/{item_id}")
async def read_item(item_id: int, q: Optional[str] = None):
    return {
        "item_id": item_id, 
        "q": q
    }
```

## 运行

```bash
$ uvicorn main:app --reload
```

关于 `uvicorn main:app --reload`:

* `main`: `main.py` 文件(一个 Python 模块)
* `app`: 在 `main.py` 文件中通过 `app = FastAPI()` 创建的对象
* `--reload`: 让服务器在更新代码后重新启动。仅在开发时使用该选项

## 检查

访问 http://127.0.0.1:8000/items/5?q=somequery，可以看到如下的 JSON 响应: 


```json
{
    "item_id": 5, 
    "q": "somequery"
}
```

说明已经创建了一个具有以下功能的 API: 

* 通过路径 `/` 和 `/items/{item_id}` 接受 HTTP 请求
* 以上路径都接受 `GET` 操作(也被称为 HTTP 方法)
* `/items/{item_id}` 路径有一个路径参数 `item_id` 并且应该为 `int` 类型
* `/items/{item_id}` 路径有一个可选的 `str` 类型的查询参数 `q`

## 交互式 API 文档

- 交互式 API 文档
    - 访问 http://127.0.0.1:8000/docs，
      可以看到由 [Swagger UI](https://github.com/swagger-api/swagger-ui) 自动生成的交互式 API 文档.
- 备选 API 文档
    - 访问 http://127.0.0.1:8000/redoc，
      可以看到由 [ReDoc](https://github.com/Rebilly/ReDoc) 生成的交互式 API 文档.

# 升级示例

## 修改 main.py 文件来从 `PUT` 请求中接收请求体

借助 `Pydantic` 来使用标准的 Python 类型声明请求体。

```python
from typing import Optional
from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI()

class Item(BaseModel):
    name: str
    price: float
    is_offer: Optional[bool] = None

@app.get("/")
def read_root():
    return {
        "Hello": "World"
    }

@app.get("/items/{item_id}")
def read_item(item_id: int, q: Optional[str] = None):
    return {
        "item_id": item_id, 
        "q": q
    }

@app.put("/items/{item_id}")
def update_item(item_id: int, item: Item):
    return {
        "item_name": item.name, 
        "item_id": item_id
    }
```

## 升级交互式 API 文档

* 交互式 API 文档
    - 访问 http://127.0.0.1:8000/docs，可以看到由 `Swagger UI` 生成的交互式 API 文档中已经加入新的请求体.
* 备选 API 文档
    - 访问 http://127.0.0.1:8000/redoc，可以看到由 `ReDoc` 生成的交互式 API 文档中已经加入了新的请求参数和请求体.

# OpenAPI

FastAPI 使用定义 API 的 OpenAPI 标准将你的所有 API 转换成[模式]
  
* 模式
    - 模式是对事物的一种定义或描述。它并非具体的实现代码，而只是抽象的描述
* API 模式
    - 在这种场景下，OpenAPI 是一种规定如何定义 API 模式的规范。
      定义的 OpenAPI 模式将包括你的 API 路径，以及它们可能使用的参数等等
* 数据模式
    - 模式这个术语也可能指的是某些数据比如 JSON 的结构。
      在这种情况下，它可以表示 JSON 的属性及其具有的数据类型等等
* OpenAPI 和 JSON Schema
    - OpenAPI 为你的 API 定义 API 模式。
      该模式中包含了你的 API 发送和接收的数据的定义(或称为[模式])，这些定义通过 JSON 数据模式标准 JSON Schema 所生成
* 查看 openapi.json
    - 如果你对原始的 OpenAPI 模式长什么样子感到好奇，其实它只是一个自动生成的包含了所有 API 描述的 JSON
    - 可以直接在 http://127.0.0.1:8000/openapi.json 看到
* OpenAPI 的用途
    - 驱动 FastAPI 内置的 2 个交互式文档系统的正是 OpenAPI 模式
