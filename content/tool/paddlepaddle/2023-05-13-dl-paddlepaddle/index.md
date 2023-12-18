---
title: PaddlePaddle 概览
author: 王哲峰
date: '2023-05-13'
slug: dl-paddlepaddle
categories:
  - deeplearning
tags:
  - tool
---

# PaddlePaddle 安装和卸载

## macOS

安装：

```bash
$ pip install paddlepaddle==2.4.2
```

卸载：

```bash
$ pip uninstall paddlepaddle
```

验证安装：

```bash
$ python
>>> import paddle
>>> paddle.utils.run_check()
```

```
PaddlePaddle is installed successfully!
```

# PaddlePaddle 组件架构

![img](images/paddle_arc.png)