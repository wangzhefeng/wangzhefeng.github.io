---
title: PyTorch TensorBoard 可视化
author: 王哲峰
date: '2022-09-04'
slug: dl-pytorch-tensorboard
categories:
  - deeplearning
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

- [TODO](#TODO)
</p></details><p></p>

在深度学习建模的过程中，如果能够使用丰富的图像来展示模型的结构，
指标的变化，参数的分布，输入的形态等信信息，会提升对问题的洞察力

PyTorch 中利用 TensorBoard 可视化的大概过程如下:

1. 首先，在 PyTorch 中指定一个目录创建一个 `torch.utils.tensorboard.SummaryWriter` 日志写入器
2. 然后，根据需要可视化的信息，利用日志写入器将相应信息日志写入指定的目录
3. 最后，就可以传入日志目录作为参数启动 TensorBoard，然后就可以在 TensorBoard 中看到相应的可视化信息

PyTorch 中利用 TensorBoard 进行信息可视化的方法如下:

* 可视化模型结构: `writer.add_graph`
* 可视化指标变化: `writer.add_scalar`
* 可视化参数分布: `writer.add_histogram`
* 可视化原始图像: `writer.add_image` 或 `writer.add_images`
* 可视化人工绘图: `writer.add_figure`

