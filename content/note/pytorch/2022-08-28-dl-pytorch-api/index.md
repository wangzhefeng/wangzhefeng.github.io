---
title: PyTorch API
author: 王哲峰
date: '2022-08-28'
slug: dl-pytorch-api
categories:
  - pytorch
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

- [低阶 API](#低阶-api)
- [中阶 API](#中阶-api)
- [高阶 API](#高阶-api)
- [torch.nn.functional 和 torch.nn.Module](#torchnnfunctional-和-torchnnmodule)
</p></details><p></p>

# 低阶 API

PyTorch 低阶 API 主要包括:

* 张量操作
* 计算图
* 自动微分


# 中阶 API

PyTorch 中阶 API 主要包括

* 模型层
* 损失函数
* 优化器
* 数据管道


# 高阶 API

PyTorch 没有官方的高阶 API，一般需要用户自己实现训练循环、验证循环、预测循环


# torch.nn.functional 和 torch.nn.Module

