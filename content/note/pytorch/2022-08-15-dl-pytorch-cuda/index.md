---
title: PyTorch CUDA
author: 王哲峰
date: '2022-08-15'
slug: dl-pytorch-cuda
categories:
  - deeplearning
  - pytorch
tags:
  - tool
---

# CUDA 语义

`torch.cuda` 用于设置和运行 CUDA 操作，它跟踪当前选择的 GPU，
默认情况下，分配的所有 CUDA 张量都将在该设备上创建。
可以使用 `torch.cuda.device` 上下文管理器更改所选设备





# torch.cuda

* torch.cuda 支持 CUDA tensor 类型
* 延迟初始化的，因此可以随时导入 torch.cuda
* 可以使用 is_available() 查看系统是否支持 CUDA

## 随机数生成


## Communication collectives


## Streams 和 events


## Graphs


## 内存管理


## NVIDIA 工具扩展(NVTX)


## Jiterator

