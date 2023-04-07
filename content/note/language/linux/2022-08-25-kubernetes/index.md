---
title: kubernetes
author: 王哲峰
date: '2022-08-25'
slug: kubernetes
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
</style>

<details><summary>目录</summary><p>

- [Kubernetes 简介](#kubernetes-简介)
  - [Kubernetes 介绍](#kubernetes-介绍)
  - [Kubernetes 特性](#kubernetes-特性)
- [Kubernetes 基础知识](#kubernetes-基础知识)
  - [Kubernetes 可以做什么](#kubernetes-可以做什么)
  - [Kubernetes 基础模块](#kubernetes-基础模块)
    - [创建一个 Kubernetes 集群](#创建一个-kubernetes-集群)
      - [Kubernetes 集群](#kubernetes-集群)
    - [部署应用程序](#部署应用程序)
    - [应用程序探索](#应用程序探索)
    - [应用程序外部可见](#应用程序外部可见)
    - [应用可扩展](#应用可扩展)
    - [应用更新](#应用更新)
- [Mac Kubernetes](#mac-kubernetes)
- [参考](#参考)
</p></details><p></p>

# Kubernetes 简介

## Kubernetes 介绍

Kubernetes 是用于自动部署，扩展和管理容器化应用程序的开源系统。
它将应用程序的容器组合成逻辑单元，以便于管理和服务发现。

Kubernetes 源自 Google 2015 年生产环境的运维经验，
同时聚集了社区的最佳创意和实践

![kubernetes](https://d33wubrfki0l68.cloudfront.net/69e55f968a6f44613384615c6a78b881bfe28bd6/1600c/zh-cn/_common-resources/images/flower.svg)

## Kubernetes 特性

* 自动化上线和回滚
* 服务发现与负载均衡
* 存储编排
* Secret 和配置管理
* 自动装箱
* 批量执行
* IPv4/IPv6 双协议栈
* 水平扩展
* 自我修复
* 为扩展性设计

# Kubernetes 基础知识

## Kubernetes 可以做什么

通过现代的 Web 服务，用户希望应用程序能够 24/7 全天候使用，开发人员希望每天可以多次发布部署新版本的应用程序。 
容器化可以帮助软件包达成这些目标，使应用程序能够以简单快速的方式发布和更新，而无需停机。
Kubernetes 帮助你确保这些容器化的应用程序在你想要的时间和地点运行，并帮助应用程序找到它们需要的资源和工具。
Kubernetes 是一个可用于生产的开源平台，根据 Google 容器集群方面积累的经验，以及来自社区的最佳实践而设计

## Kubernetes 基础模块

### 创建一个 Kubernetes 集群

#### Kubernetes 集群



### 部署应用程序

### 应用程序探索

### 应用程序外部可见

### 应用可扩展

### 应用更新

# Mac Kubernetes

Docker Desktop 包含一个可以在 Mac 上运行的 Kubernetes 服务器, 
因此可以在 Kubernetes 上部署 Docker 工作负载.

Kubernetes 的客户端命令是 `kubectl`

- 将 Kubernetes 指向 docker-desktop:

```bash
kubectl config get-contexts
kubectl config user-context docker-desktop
```



# 参考

- https://kubernetes.io/zh-cn/