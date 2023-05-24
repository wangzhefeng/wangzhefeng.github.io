---
title: Dockerfile
author: 王哲峰
date: '2022-08-25'
slug: docker-dockerfile
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

- [Dockerfile 简介](#dockerfile-简介)
- [Dockerfile 详解](#dockerfile-详解)
  - [变量](#变量)
    - [定义](#定义)
    - [转义](#转义)
  - [FROM](#from)
    - [作用](#作用)
    - [格式](#格式)
  - [RUN](#run)
    - [作用](#作用-1)
    - [执行时机](#执行时机)
    - [格式](#格式-1)
    - [使用示例](#使用示例)
  - [CMD](#cmd)
    - [执行时机](#执行时机-1)
    - [格式](#格式-2)
    - [使用示例](#使用示例-1)
  - [LABEL](#label)
    - [作用](#作用-2)
    - [使用示例](#使用示例-2)
  - [EXPOSE](#expose)
    - [作用](#作用-3)
    - [运行时机](#运行时机)
    - [使用示例](#使用示例-3)
  - [ENV](#env)
    - [作用](#作用-4)
    - [使用示例](#使用示例-4)
  - [ADD](#add)
    - [作用](#作用-5)
    - [使用示例](#使用示例-5)
  - [COPY](#copy)
    - [作用](#作用-6)
  - [ENTRYPOINT](#entrypoint)
    - [使用示例](#使用示例-6)
  - [VOLUME](#volume)
    - [作用](#作用-7)
    - [使用示例](#使用示例-7)
  - [ARG](#arg)
    - [作用](#作用-8)
    - [使用示例](#使用示例-8)
  - [ONBUILD](#onbuild)
    - [作用](#作用-9)
    - [使用示例](#使用示例-9)
  - [STOPSIGNAL](#stopsignal)
    - [作用](#作用-10)
    - [使用示例](#使用示例-10)
  - [HEALTHCHECK](#healthcheck)
    - [作用](#作用-11)
    - [格式](#格式-3)
  - [SHELL](#shell)
    - [作用](#作用-12)
    - [使用示例](#使用示例-11)
  - [WORKDIR](#workdir)
    - [作用](#作用-13)
    - [使用示例](#使用示例-12)
  - [USER](#user)
    - [作用](#作用-14)
    - [使用示例](#使用示例-13)
- [Dockerfile 示例](#dockerfile-示例)
- [参考](#参考)
</p></details><p></p>

# Dockerfile 简介

* Docker 可以通过读取 Dockerfile 中的指令自动构建镜像
* Dockerfile 是一个文本文档, 其中包含了用户创建镜像的所有命令和说明

# Dockerfile 详解

## 变量

### 定义

变量用 `$variable_name` 或者 `${variable_name}` 表示:

- `${variable:-word}` 表示如果 `variable` 设置, 则结果将是该值, 如果 `variable` 未设置, `word` 则将是结果
- `${variabel:+word}` 表示如果 `variable` 设置, 则为 `word` 结果, 否则为空字符串

### 转义

变量前加 `\` 可以转义成普通字符串: `\$foo` 或者 `\${foo}` 表示转换为 `$foo` 和 `${foo}` 文本

## FROM

### 作用

初始化一个新的构建阶段，并设置基础镜像

* 单个 Dockerfile 可以多次出现 `FROM`, 以使用之前的构建阶段作为另一个构建阶段的依赖项

### 格式

```bash
FROM [--platform=<platform>] <image> [AS <name>]
FROM [--platform=<platform>] <image>[:<tag>] [AS <name>]
FROM [--platform=<platform>] <image>[@<digest>] [AS <name>]
```

- `--platform`
    - `--platform` 标志可用于在 `FROM` 引用多平台镜像的情况下指定平台. 
      例如, `linux/amd64`、 `linux/arm64`、或 `windows/amd64`
- `image`
- `AS <name>` 
    - 表示为构建阶段命名, 在后续 `FROM` 和 `COPY --from=` 说明中可以使用这个名词, 引用此阶段构建的镜像  
- `tag` 或 `digest` 
    - `tag` 或 `digest` 值是可选的
    - 如果省略其中任何一个, 构建器默认使用一个 `latest` 标签
    - 如果找不到该 `tag` 值, 构建器将返回错误
    - `digest` 其实就是根据镜像内容产生的一个 ID, 只要要镜像的内容不变 `digest` 也不会变

## RUN

### 作用

在当前镜像之上的新层中执行命令

### 执行时机

在 `docker build` 时运行

### 格式

`RUN` 有两种形式:

* `RUN`
    - shell 形式，命令在 shell 中运行，在 Linux 或 macOS上 默认是 `/bin/sh -c`，在 Windows 上默认是 `cmd /S /C`
* `RUN ["executable", "param1", "param2"]`
    - 执行形式

说明:

* 可以使用 `\` 将单个 `RUN` 指令延续到下一行
* `RUN` 在下一次构建期间，指令缓存不会自动失效。可以使用 `--no-cache` 标志使指令缓存无效
* Dockerfile 的指令每执行一次都会在 Docker 上新建一层。
  所以过多无意义的层会造成镜像膨胀过大，可以使用 `&&` 符号链接命令，
  这样执行后，只会创建一层镜像

### 使用示例

```bash
RUN /bin/base -c 'source $HOME/.bashr;'  \
echo  $HOME'
```

## CMD

运行程序

### 执行时机

在 `docker run` 时执行，但是和 `run` 命令不同，`RUN` 是在 `docker build` 时运行

### 格式

支持三种格式:

* `CMD ["executable", "param1", "param2"]` 使用 exec 执行，推荐方式
* `CMD command param1 param2` 在 `/bin/sh` 中执行，提供给需要交互的应用
* `CMD ["param1", "param2"]` 提供给 `ENTRYPOINT` 的默认参数

指定启动容器时执行的命令，每个 Dockerfile 只能有一条 `CMD` 命令。如果指定了多条命令，只有最后一条会被执行

### 使用示例

```bash
FROM ubuntu
CMD ["/usr/bin/wc", "--help"]
```

## LABEL

### 作用

添加元数据

### 使用示例

```bash
LABEL multi.label1="value1" \
      multi.label2="value2" \
      other="value3"
```

## EXPOSE

### 作用

Docker 容器在运行时监听指定的网络端口，可以指定端口是监听 TCP 还是 UDP，如果不指定协议，默认是 TCP。

### 运行时机

`EXPOSE` 指令实际上并未发布端口，要在运行容器时实际发布端口，`docker run -P` 来发布和映射一个或多个端口

### 使用示例

```bash
EXPOSE <port> [<port>/<protocol>...]
EXPOSE 80/udp
```

## ENV

### 作用

设置环境变量，设置的环境变量将持续存在

* 可以使用用 `docker inspect` 查看环境变量
* 可以使用 `docker run --env=` 更改环境变量的值

如果环境变量只在构建期间需要，可以采用如下方式

* 为单个命令设置一个值

```bash
RUN DEBIAN_FRONTEND=noninteractive apt-get update && apt-get install -y ...
```

* 使用 `ARG`，它不会保留在最终的镜像中

```bash
ARG DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y ...
```

### 使用示例

```bash
ENV <key>=<value> ...
```

## ADD

### 作用

复制新文件、目录或远程 URL，并将它们添加到镜像中。
可以指定多个资源，但如果它们是文件或目录，
则它们的路径被解释为相对于构建上下文的源，也就是 `WORKDIR`

### 使用示例

如果文件、目录或远程 URL 中的资源名称中包含通配符，匹配时将使用 Go 的 `filepath.Match` 规则

```bash
ADD hom* /mydir/
ADD hom?.txt /mydir/
```

## COPY

### 作用

语法与 `ADD` 形同，复制拷贝文件。`COPY` 指令和 `ADD` 指令的唯一区别在于：是否支持从远程 URL 获取资源

* `COPY` 指令只能从执行 `docker build` 所在的主机上读取资源并复制到镜像中
* `ADD` 指令还支持通过 URL 从远程服务器读取资源并复制到镜像中

相同需求时，推荐使用 `COPY` 指令，`ADD` 指令更擅长读取本地 `tar` 文件并解压缩

## ENTRYPOINT

`ENTRYPOINT` 和 `CMD` 一样，都是在指定容器启动程序及参数，
不过 `ENTRYPOINT` 不会被 `docker run` 命令行参数指定的指令所覆盖。
如果覆盖的话，需要通过 `docker run --entrypoint` 来指定

### 使用示例

```bash
ENTRYPOINT ["executable", "param1", "param2"]
ENTRYPOINT command param1 param2

# 指定了 ENTRYPOINT 后， CMD 的内容作为参数传给 ENTRYPOINT 指令，实际执行时，将变为
<ENTRYPOIN> <CMD>
```



## VOLUME

### 作用

创建一个具有指定名称的挂载数据卷，它的主要作用是:

* 避免重要的数据因容器重启而丢失
* 避免容器不断变大

### 使用示例

```bash
VOLUME ["/var/log/"]
VOLUME /var/log
```

## ARG

### 作用

定义变量，与 `ENV` 作用相同，不过 `ARG` 变量不会像 `ENV` 变量那样持久化到构建好的镜像中

Docker 有一组预定义的 `ARG` 变量，可以在 Dockerfile 中没有相应指令的情况下使用这些变量

* HTTP_PROXY
* http_proxy
* HTTPS_PROXY
* https_proxy
* FTP_PROXY
* ftp_proxy
* NO_PROXY
* no_proxy

### 使用示例

```bash
ARG <name>[=<default value>]
```

## ONBUILD

### 作用

讲一个触发指令添加到镜像中，以便稍后在镜像用作另一个构建的基础时执行。
也就是另外一个 Dockerfile `FROM` 了这个镜像的时候执行

### 使用示例

```bash
ONBUILD AND . /app/src
ONBUILD RUN /usr/local/bin/python-build --dir /app/src
```

## STOPSIGNAL

### 作用

设置将发送到容器退出的系统调用信号，该信号可以是与内核系统调用表中的位置匹配的有效无符号数，例如 9，
或格式为 SIGNAME 的信号名称，例如 SIGKILL

默认的 stop-signal 是 SIGTERM，在 docker stop 的时候会给容器内 PID 为 1 的进程发送这个 signal，
通过 `--stop-signal` 可以设置自己需要的 signal，主要目的是为了让容器内的应用程序在接收到 signal 之后可以先处理一些事物，
实现容器的平滑退出，如果不做任何处理，容器将在一段时间之后强制退出，会造成业务的强制中断，默认时间是 10s

### 使用示例

```bash
STOPSIGNAL signal
```

## HEALTHCHECK

### 作用

用于指定某个程序或者指令来监控 Docker 容器服务的运行状态

### 格式

`HEALTHCHECK` 指令有两种形式:

* `HEALTHCHECK [OPTIONS] CMD command`
    - 通过在容器内运行命令来检查容器建康状况
* `HEALTHCHECK NONE`
    - 禁用从基础镜像继承的任何建康检查

## SHELL

### 作用

覆盖用于命令的 shell 形式的默认 shell

* Linux 上默认的 shell 是 ["/bash/sh", "-c"]，
* Windows 上是 ["cmd", "/S", "/C"]

### 使用示例

下面的 `SHELL` 指令在 Windows 上特别有用，
因为 Windows 有两种常用且截然不同的本机 shell: cmd 和 powershell，以及可用的备用 shell，包括 sh。
该 `SHELL` 指令可以出现多次，每条 `SHELL` 指令都会覆盖所有先前的 `SHELL` 指令，并影响所有后续指令

```bash
SHELL ["executable", "parameters"]
```

## WORKDIR

### 作用

`WORKDIR` 指工作目录。如果 `WORKDIR` 不存在，即使它没有在后续 Dockerfile 指令中使用，它也会被创建。
`docker build` 构建镜像过程中，每一个 `RUN` 命令都会新建一层，只有通过 `WORKDIR` 创建的目录才会一直存在

### 使用示例

* 可以设置多个 `WORKDIR`，如果提供了相对路径，它将相对于前一条 `WORKDIR` 指令的路径

```bash
WORKDIR /a
WORKDIR b
WORKDIR c
RUN pwd
```

```
/a/b/c
```

* `WORKDIR` 可以解析先前使用的 `ENV`

```bash
ENV DIRPATH=/path
WORKDIR $DIRPATH/$DIRNAME
RUN pwd
```

```
/path/$DIRNAME
```

## USER

### 作用

设置用户名(或 UID) 和可选的用户组(或 GID)

### 使用示例

```bash
USER <user>[:<group>]
# or
USER <UID>[:GID]
```

# Dockerfile 示例










# 参考

- [全面详解 Dockerfile 文件](https://mp.weixin.qq.com/s?__biz=MzAwMjg1NjY3Nw==&mid=2247518419&idx=2&sn=2e25da85a7dcf19fe6ca80484128deb3&chksm=9ac6cb59adb1424f76071ee35d0a01ba9635490b422b239290693851fc2f099bd3414af7792c&scene=132#wechat_redirect)