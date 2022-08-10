---
title: ML 模型部署
author: 王哲峰
date: '2022-07-15'
slug: ml-model-deploy
categories:
  - machinelearning
tags:
  - note
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
h3 {
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

- [ML](#ml)
- [DL](#dl)
  - [TensorFlow](#tensorflow)
    - [TensorFlow Serving-Docker-Tornado](#tensorflow-serving-docker-tornado)
      - [Docker](#docker)
</p></details><p></p>




# ML



# DL

## TensorFlow

### TensorFlow Serving-Docker-Tornado

![](https://pic1.zhimg.com/v2-0cd02fbfa359bfe77397981d1a0e938d_1440w.jpg?source%3D172ae18b)


当我们训练完一个tensorflow(或keras)模型后，需要把它做成一个服务，
让使用者通过某种方式来调用你的模型，而不是直接运行你的代码（因为你的使用者不一定懂怎样安装），
这个过程需要把模型部署到服务器上。常用的做法如使用flask、Django、tornado 等 web 框架创建一个服务器 app，
这个 app 在启动后就会一直挂在后台，然后等待用户使用客户端 POST 一个请求上来（例如上传了一张图片的 url），
app 检测到有请求，就会下载这个 url 的图片，接着调用你的模型，得到推理结果后以 json 的格式把结果返回给用户

这个做法对于简单部署来说代码量不多，对于不熟悉 web 框架的朋友来说随便套用一个模板就能写出来，
但是也会有一些明显的缺点：

1. 需要在服务器上重新安装项目所需的所有依赖
2. 当接收到并发请求的时候，服务器可能要后台启动多个进程进行推理，造成资源紧缺
3. 不同的模型需要启动不同的服务

#### Docker


##