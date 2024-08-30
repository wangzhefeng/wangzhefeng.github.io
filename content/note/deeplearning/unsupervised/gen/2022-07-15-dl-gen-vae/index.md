---
title: VAE 深度生成模型
author: 王哲峰
date: '2022-07-15'
slug: dl-gen-vae
categories:
  - deeplearning
tags:
  - model
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

- [TODO](#todo)
- [AutoEncoder 自编码器](#autoencoder-自编码器)
</p></details><p></p>

# TODO

* [深入理解生成模型VAE](https://mp.weixin.qq.com/s/exaBFn4cjedwDVbGKC-UUw)


# AutoEncoder 自编码器

作为一种无监督或者自监督模型, 自编码器本质上是一种数据压缩方法。
从现有情况来看, 无监督学习很有可能是一把决定深度学习未来发展方向的钥匙, 
在缺乏高质量打标数据的监督机器学习时代, 
若是能在无监督学习方向上有所突破对于未来深度学习的发展意义重

自编码器(AutoEncoder, AE)就是一种利用反向传播算法使得输出值等于输入值的神经网络, 
它先将输入压缩成潜在空间表征, 然后将这种表征重构为输出。
所以从本质上来讲, 自编码器是一种数据压缩算法, 
其压缩和解压缩算法都是通过神经网络来实现的。

自编码器有如下三个特点:

- 数据相关性
    - 自编码器只能压缩与自己此前训练数据类似的数据
- 数据有损性
    - 自编码器在解压时得到的输出与原始输入相比会有信息损失, 所以自编码器是一种数据有损的压缩算法
- 自动学习性
    - 自编码器是从数据样本中自动学习的, 这意味着很容易对指定类的输入训练出一种特定的编码器, 而不需要完成任何新的工作
