---
title: video-pyav
author: 王哲峰
date: '2023-01-21'
slug: video-pyav
categories:
  - computer vision
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

- [安装](#安装)
- [使用](#使用)
</p></details><p></p>

PyAV 是FFmpeg库的 Pythonic 绑定。我们的目标是提供底层库的所有功能和控制，但尽可能多地管理细节
PyAV 用于通过容器、流、数据包、编解码器和帧直接和精确地访问您的媒体。它公开了该数据的一些转换，并帮助您将数据传入/传出其他包（例如 Numpy 和 Pillow）

# 安装

```bash
$ pip install av
```

# 使用


