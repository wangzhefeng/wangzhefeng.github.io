---
title: Docker 制作镜像
author: 王哲峰
date: '2022-07-25'
slug: docker-build-image
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

- [Linux](#linux)
  - [CentOS 7](#centos-7)
    - [安装 docker](#安装-docker)
    - [启动 docker](#启动-docker)
    - [查看 docker 镜像](#查看-docker-镜像)
    - [下载 CentOS 镜像](#下载-centos-镜像)
    - [自定义 Dockerfile 文件](#自定义-dockerfile-文件)
    - [构建镜像](#构建镜像)
    - [创建虚拟机](#创建虚拟机)
    - [验证 ifconfig](#验证-ifconfig)
- [MacOS](#macos)
- [Windows](#windows)
</p></details><p></p>


Docker 是一个开源的应用容器引擎，基于 Go 语言并遵从 Apache2.0 协议开源。
Docker 可以让开发者打包他们的应用以及依赖包到一个轻量级、可移植的容器中，
然后发布到任何流行的 Linux 机器上，也可以实现虚拟化

# Linux

## CentOS 7

### 安装 docker

* 移除旧版本 docker

```bash
$ sudo yum remove docker \
  docker-client \
  docker-client-latest \
  docker-common \
  docker-latest \
  docker-latest-logrotate \
  docker-logrotate \
  docker-engine
```

* 安装 `yum-utils` 包并设置稳定存储库

```bash
$ yum install -y yum-utils
```

* 安装 docker

```bash
$ yum install docker-ce docker-ce-cli containerd.io
```

### 启动 docker

```bash
$ systemctl start docker
```

### 查看 docker 镜像

* 刚安装 docker 是没有镜像的

```bash
$ docker images
```

### 下载 CentOS 镜像

默认 CentOS 镜像没有 `ifconfig`

```bash
$ docker pull centos
```

### 自定义 Dockerfile 文件

```bash
FROM centos
ENV MYPATH /usr/local
WORKDIR $MYPATH
RUN yum install -y net-tools
RUN yum install -y vim
EXPOSE 80
CMD echo $MYPATH
CMD echo "-----end-----"
CMD /bin/bash"
```

### 构建镜像

```bash
$ docker build -f ./dockerfile_chao_centos -t chao.centos:0.1 .
```


### 创建虚拟机

```bash
$ docker run -it --name test chao.centos:0.1 /bin/bash
```

### 验证 ifconfig

```bash
$ ifconfig
```

# MacOS






# Windows
