---
title: FastAPI 部署
author: 王哲峰
date: '2022-12-01'
slug: fastapi-deploy
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

- [FastAPI 版本](#fastapi-版本)
  - [FastAPI 可用版本](#fastapi-可用版本)
  - [FastAPI 版本](#fastapi-版本-1)
  - [更新 FastAPI 版本](#更新-fastapi-版本)
  - [Starlette](#starlette)
  - [Pydantic](#pydantic)
- [HTTPS](#https)
- [Deta 部署 FastAPI](#deta-部署-fastapi)
- [Docker 部署 FastAPI](#docker-部署-fastapi)
  - [tiangolo/uvicorn-gunicorn-fastapi](#tiangolouvicorn-gunicorn-fastapi)
  - [创建一个 Dockerfile](#创建一个-dockerfile)
  - [创建 FastAPI 代码](#创建-fastapi-代码)
  - [构建 Docker 镜像](#构建-docker-镜像)
  - [启动 Docker 容器](#启动-docker-容器)
  - [检查](#检查)
  - [Traefik](#traefik)
  - [具有 Traefik 和 HTTPS 的 Docker Swarm 模式集群](#具有-traefik-和-https-的-docker-swarm-模式集群)
- [手动部署 FastAPI](#手动部署-fastapi)
</p></details><p></p>

# FastAPI 版本

## FastAPI 可用版本

* https://fastapi.tiangolo.com/release-notes/

## FastAPI 版本

* FastAPI 版本

```
fastapi>=0.45.0,<0.46.0
```


* PATCH: 版本号最后一个数字，代表版本更改均用于错误修复和不间断的更改
* MINOR: 版本号中间的数字，代表版本中添加了重大更改和功能

## 更新 FastAPI 版本

1. 首先，应该使用 FastAPI 提供的测试功能测试 APP
2. 然后，更新 FastAPI 到一个比较新的版本，然后在新版本的 FastAPI 中测试 APP
3. 最后，如果一切 OK，或者做了必要的修改后，通过了所有的测试，那么可以将 FastAPI 的版本固定下来

## Starlette

不应该固定 starlette 的版本，因为不同的 FastAPI 版本会使用最新版本的 Starlette

## Pydantic

- Pydantic 的新版本(>=1.0.0)始终与 FastAPI 兼容，
  可以将 Pydantic 固定到高于 1.0.0 但低于 2.0.0 版本的任何版本
- 示例

```
pydantic>=1.2.0,<2.0.0
```

# HTTPS


# Deta 部署 FastAPI

# Docker 部署 FastAPI

- 创建一个具有最佳西能的 FastAPI APP Docker image/container
- HTTPS 知识
- 在服务器上使用自动 HTTPS 设置 Docker Swarm 模式集群
- 使用 Docker Swarm 集群以及 HTTPS 等生成部署完整的 FastAPI APP

## tiangolo/uvicorn-gunicorn-fastapi

- https://github.com/tiangolo/uvicorn-gunicorn-fastapi-docker/blob/master/docker-images/python3.7.dockerfile

## 创建一个 Dockerfile

1. 进入项目目录

```bash
$ mkdir fastapi_demo
$ cd fastapi_demo
```

1. 创建一个 Dockerfile 文件

```
FROM tiangolo/uvicorn-gunicorn-fastapi:python3.7

COPY ./app /app
```

2. 如果创建一个较大的 APP

```
FROM tiangolo/uvicorn-gunicorn-fastapi:python3.7

COPY ./app /app/app
```

3. 如果在 Raspberry Pi(具有 ARM 处理器) 或任何其他体系结构中运行 Docker，
   则 Dockerfile 可以基于 Python 基础镜像(即多体系结构)从头开始 Docker ，
   并单独使用 uvicorn

```
FROM python:3.7

RUN pip install fastapi uvicorn

EXPOSE 80

COPY ./app /app

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "80"]
```

## 创建 FastAPI 代码

1. 创建一个 app 目录
2. 在 app 目录中创建一个 main.py 文件
3. 在 main.py 文件中写入以下内容

```python
from typing import Optional
from fastapi import FastAPI

app = FastAPI()

@app.get("/")
async def read_root():
    return {
        "Hello": "Workld"
    }

@app.get("/items/{item_id}")
asynv def read_item(item_id: int, q: Optional[str] = None):
    return {
        "item_id": item_id,
        "q": q
    }
```

## 构建 Docker 镜像

1. 进入项目目录(Dockerfile 所在目录)

```bash
$ cd fastapi_demo
```

2. 构建 FastAPI 的 Docker 镜像

```bash
$ docker build -t myimage .
```

## 启动 Docker 容器

1. 基于构建的 Docker 镜像启动一个 Docker 容器

```bash
$ docker run -d --name muycontainer -p 80:80 myimage
```

2. 现在在 Docker 容器中有了一个优化的 FastAPI 服务器，并针对当前服务器的 CUP 内核数自动调整性能

## 检查

1. 打开 http://192.168.99.100/items/5?q=somequery 或 http://127.0.0.1/items/5?somequery 可以看到

```json
{"item_id": 5, "q": "somequery"}
```

2. 打开 http://192.168.99.100/docs 或 http://127.0.0.1/docs
3. 可选DOC http://192.168.99.100/redoc 或 http://127.0.0.1/redoc

## Traefik

* Traefik 是高性能的反向代理/负载平衡器，它可以执行 "TLS 终止代理"工作
* Traefik 与 Let's Encrypt 集成，所以 Traefik 可以处理所有的 HTTPS 部分，包括证书获取和更新
* Traefik 还与 Docker 集成，因此，可以在每个 APP 配置中声明域，
  并让其读取这些配置，生成 HTTPS 证书，并自动向 APP 提供 HTTPS，并无需对其配置进行任何更改

## 具有 Traefik 和 HTTPS 的 Docker Swarm 模式集群

* 通过使用 Docker Swarm 模式，可以从一台计算机的集群开始，然后可以根据需要添加更多服务器而增长
* 要使用 Traefik 和 HTTPS 处理设置 Docker Swarm 模式集群，请遵循以下指南: 
    - https://tiangolo.medium.com/docker-swarm-mode-and-traefik-for-a-https-cluster-20328dba6232
* 部署一个 FastAPI APP最简单的方式是使用 [FastAPI 项目生成器](https://fastapi.tiangolo.com/project-generation/)

# 手动部署 FastAPI

手动部署 FastAPI 只需要安装 ASGI 兼容的服务器即可: 

* Uvicorn: a lightning-fast ASGI server, build on uvloop and httptools

```bash
$ pip install uvicorn[standard]
```
 
* Hypercorn: An ASGI server also compatible with HTTP/2

```bash
$ pip install hypercorn
```

(2)运行 APP 

- 开发环境

```bash
$ uvicorn main:app --host 0.0.0.0 --port 80 -reload
```

- 生产环境

```bash
$ uvicorn main:app --host 0.0.0.0 --port 80
```

(3)如果想确保服务器停止后自动重新启动

- [Gunicorn](https://gunicorn.org/)
- [Use Gunicorn as a manager for Uvicorn](https://www.uvicorn.org/#running-with-gunicorn)
