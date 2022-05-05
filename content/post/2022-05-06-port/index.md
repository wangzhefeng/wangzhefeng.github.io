---
title: 端口
author: 王哲峰
date: '2022-05-06'
slug: port
categories:
  - web
  - network
tags:
  - tool
---

<style>
h1 {
  background-color: #2B90B6;
  background-image: linear-gradient(45deg, #4EC5D4 10%, #146b8c 20%);
  background-size: 100%;
  -webkit-background-clip: text;
  -moz-background-clip: text;
  -webkit-text-fill-color: transparent;
  -moz-text-fill-color: transparent;
}
h2 {
  background-color: #2B90B6;
  background-image: linear-gradient(45deg, #4EC5D4 10%, #146b8c 20%);
  background-size: 100%;
  -webkit-background-clip: text;
  -moz-background-clip: text;
  -webkit-text-fill-color: transparent;
  -moz-text-fill-color: transparent;
}

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

- [端口简介](#端口简介)
- [硬件端口](#硬件端口)
- [网络端口](#网络端口)
  - [端口含义](#端口含义)
  - [端口分类](#端口分类)
</p></details><p></p>




# 端口简介

端口是英文 "Port" 的意译, 可以认为是设备与外界通讯交流的出口. 
端口可分为虚拟端口和物理端口, 其中: 

- 虚拟端口指计算机内部或交换机路由器内的端口, 不可见. 例如, 
  计算机中的 80 端口、21 端口、23端口等.
- 物理端口又称为接口, 是可见端口, 计算机背板的 RJ45 网口, 
  交换机路由器集线器等 RJ45 端口. 电话使用 RJ11 插口也属于物理端口的范畴.


# 硬件端口



# 网络端口

在网络技术中, 端口(Port)大致有两种意思: 

- 一是物理意义上的端口, 比如, ADSL Modem、集线器、交换机、路由器用于连接其他网络设备的接口, 
  如 RJ-45 端口、SC 端口等等
- 二是逻辑意义上的端口, 一般是指 TCP/IP 协议中的端口, 端口号的范围从 0 到 65535, 
  比如用于浏览网页服务的 80 端口, 用于 FTP 服务的 21 端口等等

## 端口含义



## 端口分类

1. 公认端口(Well Known Ports): 从0到1023, 它们紧密绑定(binding)于一些服务. 通常这些端口的通讯明确表明了某种服务的协议. 例如: 80端口实际上总是HTTP通讯. 
2. 注册端口(Registered Ports): 从1024到49151. 它们松散地绑定于一些服务. 也就是说有许多服务绑定于这些端口, 这些端口同样用于许多其它目的. 例如: 许多系统处理动态端口从1024左右开始. 
3. 动态和/或私有端口(Dynamic and/or Private Ports): 从49152到65535. 理论上, 不应为服务分配这些端口. 实际上, 机器通常从1024起分配动态端口. 但也有例外: SUN的RPC端口从32768开始. 