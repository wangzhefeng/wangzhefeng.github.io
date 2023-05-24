---
title: Curl
author: 王哲峰
date: '2022-10-09'
slug: curl
categories:
  - Linux
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

- [使用 Curl 发送 REST API 请求](#使用-curl-发送-rest-api-请求)
- [Curl 选项](#curl-选项)
- [使用 Curl 发送 HTTP GET 请求](#使用-curl-发送-http-get-请求)
- [使用 Curl 发送 HTTP POST 请求](#使用-curl-发送-http-post-请求)
- [使用 Curl 发送 HTTP PUT 请求](#使用-curl-发送-http-put-请求)
- [使用 Curl 发送 HTTP DELETE 请求](#使用-curl-发送-http-delete-请求)
- [使用 Curl 进行身份验证](#使用-curl-进行身份验证)
</p></details><p></p>

# 使用 Curl 发送 REST API 请求

应用程序接口 API 是允许程序相互通信的一种定义和协议。
术语 REST 的意思是表述性状态转移(Representational State Transfer)。
它是一种架构方式，由约束集合组成，可以在创建 Web 服务时使用

RESTful API 是遵循 REST 体系结构的 API。
通常，REST API 使用 HTTP 协议发送和检索数据以及 JSON 格式的响应。
可以使用标准的 HTTP 方法创建、产看、更新或删除资源

API 请求由四个不同部分组成:

* 端点
    - 客户端用于与服务器通信的 URL 
* HTTP 方法
    - 告诉服务器端要执行什么操作。最常见的方法是 GET、POST、PUT、DELETE、PATCH
* headers
    - 用于在服务器和客户端之间传递其他信息，例如: 授权
* body 正文
    - 发送到服务器的数据

Curl 是一个命令行程序，用于从远程服务器或向远程服务器传输数据。
默认情况下，它已安装在大多数 Linux 发行版本上

# Curl 选项

`curl` 命令的语法:

```bash
$ curl [options] [URL...]
```

`curl` 选项:

* `-X`, `--request`: 要使用的 HTTP 方法
* `-i`, `--include`: 包含 headers
* `-d`, `--data`: 要发送的数据
* `-H`, `--header`: 要发送的附加 headers

# 使用 Curl 发送 HTTP GET 请求

GET 方法从服务器请求特定资源。使用 `curl` 进行 HTTP 请求时，GET 是默认方法

API 发出 GET 请求:

```bash
$ curl https://jsonplaceholder.typicode.com/posts
```

要过滤结果，使用 `query` 查询参数:

```bash
$ curl https://jsonplaceholder.typicode.com/posts?userId=1
```

# 使用 Curl 发送 HTTP POST 请求

POST 方法用于在服务器上创建资源，如果资源存在，则将其覆盖

以下命令使用 `-d` 选项指定要发送的数据并发出 POST 请求:

```bash
$ curl -X POST -d "userId=5&title=Hello World&body=Post body." https://jsonplaceholder.typicode.com/posts
```

使用 `Content-Type` header 指定请求正文的类型。 
默认情况下，未指定此 header 标头时，`curl` 使用 `Content-Type: application/x-www-form-urlencoded`。
`-d` 指定要发送 JSON 格式的数据，请将 body 主体正文类型设置为 `application/json`：

```bash
$ curl -X POST \
    -H "Content-Type: application/json" \
    -d '{"name": "linuxize", "email": "freax@myfreax.com"}' \
    https://example/contact
```


# 使用 Curl 发送 HTTP PUT 请求

PUT方法用于更新或替换服务器上的资源。它将指定资源的所有数据替换为请求发送的数据

```bash
$ curl -X PUT \
    -d "userId=5&title=Hello World&body=Post body." \
    https://jsonplaceholder.typicode.com/posts/5
```

PUT 方法也可用于对服务器上的资源进行部分更新:

```bash
$ curl -X PUT \
    -d "title=Hello Universe" \
    https://jsonplaceholder.typicode.com/posts/5
```

# 使用 Curl 发送 HTTP DELETE 请求

DELETE 方法从服务器中删除指定的资源

```bash
$ curl -X DELETE https://jsonplaceholder.typicode.com/posts/5
```

# 使用 Curl 进行身份验证

如果 API 端点需要身份验证，则需要先获取访问密钥。
否则，API 服务器将以 “Access Forbidden” 或 “Unauthorized” 响应消息进行响应

获取访问密钥的过程取决于您使用的 API。获得访问令牌后，您可以在 header 中发送它

```bash
$ curl -X GET \
    -H "Authorization: Bearer {ACCESS_TOKEN}" \
    "https://api.server.io/posts"
```