---
title: Pytorch Distributed Data Parallel
author: wangzf
date: '2025-01-19'
slug: pytorch-distributed-data-parallel
categories:
  - tool
tags:
  - machinelearning
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

- [单机单卡训练](#单机单卡训练)
- [模型分布式训练](#模型分布式训练)
    - [单机多卡训练](#单机多卡训练)
        - [原理](#原理)
        - [示例](#示例)
    - [多机多卡训练](#多机多卡训练)
- [参考](#参考)
</p></details><p></p>

常用的分布式训练框架：

* PyTorch 分布式训练框架 DDP
* DeepSpeed 分布式训练框架

# 单机单卡训练

* 单线程

# 模型分布式训练

## 单机多卡训练

### 原理

**单机多卡模型训练** 就是用多张显卡训练一个模型，对于训练速度的提升有极为明显的效果。
相比较多机多卡，单机多卡实现极其简易，只需要使用 `torch` 自带的通讯通道和端口，
即可快速将单线程的代码转换成多线程。

`torch 1.11.0` 提供了启动命令 `torchrun`，对于 `local rank` 的使用有了优化。
理解了分布式训练的基础原理，遵循着 PyTorch 官方文档，就能写出非常优秀的分布式训练代码。

首先我们要思考一个问题，**分布式训练** 相比较 **单线程训练** 需要做出什么改变呢？

1. 第一个是 **启动的命令行**。以前我们使用 `python train.py` 启动一个线程进行训练，现在我们需要一个新的启动方式，
   从而让机器知道现在我们要启动八个线程了。这八个线程之间的通讯方式完全由 `torch` 帮我们解决。
   而且我们还可以知道，`torch` 会帮助我们产生一个 `local rank` 的标志，如果是八个线程，
   每个线程的 `local rank` 编号会是 `0~7`。这样我们就可以通过 `local rank` 确定当前的线程具体是哪一个线程，
   从而将其绑定至对应的显卡。
2. 第二个是 **模型** 的改变。八个线程如果各自训练各自的模型，那也是白搭。
   所以我们要将模型改成支持分布式的模型。`torch` 为我们提供的解决思路是，
   分布式的 `model` 在多个线程中始终保持参数一致。任意一个线程梯度传播造成的参数改变会影响所有模型中的参数。
3. 第三个是 **数据分配** 的改变。假设我们仍然使用传统的 `dataloader` 去处理数据，会发生什么？
   八个线程，拿到了一样的训练数据集，然后进行训练。虽然这样训练一个周期相当于训练了八个周期，但是总感觉怪怪的。
   我们拿到一个任务，肯定希望它被拆分成多份给不同的工人去做，而不是每个工人都做一遍全部任务。
   所以我们希望 `dataloader` 可以把训练数据集拆成八份，每个线程训练自己获得的那一份数据集。
4. 第四个是 **evaluation**。训练完了模型我们肯定要进行性能测试。由于八个线程上的模型参数是一样的，
   所以我们在任意一个线程上 evaluation 就可以了。这个时候有人可能要说，我可不可以把 evaluation 也变成多线程的呢？
   我觉得可以，但是会有一个问题。我们通常希望获得模型在 test set 上的表现比如 `mae loss`。
   我们去看第三点，分布式的操作会把数据集分成多块，如果 evaluation 变成分布式会让 8 个线程得到 8 个 `mae loss`，
   而这 8 个 `mae loss` 是对于 8 个被拆分的测试数据集的。
   怎么把这 8 个 `mae loss` 结合成对于整体 test set 的 `loss` 是我们需要考虑的问题。
   `mae loss` 还好，直接取平均看起来还比较合理，但是我面对的任务是要去评估 `precision_recall 曲线` 的。
   我没有想好怎么分布式地评估 `P_R 曲线`，同时注意到 test set 相比较 train set 规模小了太多，
   我直接单线程的进行 evaluation 是一个比较方便的做法。当然我相信一定有方法可以分布式地进行 `evaluation`，
   但是我面临的问题不需要这么做。

### 示例

1. 第一步，要让代码支持分布式，也就是引入一些支持分布式通讯的代码。
   具体操作就是需要在代码最前端添加：

```python
import os
import torch.distributed as dist

local_rank = int(os.environ["LOCAL_RANK"])
dist.init_process_group(backend = "gloo|nccl")
```

`local_rank` 是每个进程的标签，对于八个进程，`local_rank` 这个变量最后会被分配 `0~7` 的整数。

2. 第二步，让模型支持分布式

```python
model = torch.nn.parallel.DistributedDataParallel(
    model, 
    device_ids=[local_rank],
    output_device=local_rank,
)
```


## 多机多卡训练


# 参考

* [https://pytorch.org/tutorials/intermediate/ddp_tutorial.html](https://pytorch.org/tutorials/intermediate/ddp_tutorial.html)
* [pytorch单机多卡训练](https://zhuanlan.zhihu.com/p/510718081)