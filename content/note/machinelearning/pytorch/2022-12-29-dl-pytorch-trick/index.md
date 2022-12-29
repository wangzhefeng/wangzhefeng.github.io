---
title: PyTorch 技巧
author: 王哲峰
date: '2022-12-29'
slug: dl-pytorch-trick
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

- [GPU 相关](#gpu-相关)
  - [指定 GPU 编号](#指定-gpu-编号)
- [模型相关](#模型相关)
  - [查看模型每层输出详情](#查看模型每层输出详情)
- [参考](#参考)
</p></details><p></p>

# GPU 相关

## 指定 GPU 编号

设置当前使用的 GPU 设备仅为 0 号设备，设备名称为 `/gpu:0`

```python
os.environ["CUDA_VISIBLE_DEIVCES"] = "0"
```

设置当前使用的 GPU 设备为 0、1 号两个设备，名称依次为 `/gpu:0`、`/gpu:1`。
根据顺序表示优先使用 0 号设备，然后使用 1 号设备

```python
os.environ["CUDA_VISIBLE_DEVICES"] = "0, 1"
```

# 模型相关


## 查看模型每层输出详情

* https://github.com/sksq96/pytorch-summary

```python
from torchsummary import summary

summary(model, input_size = (channels, H, W))
```


# 参考

* [参考地址](https://mp.weixin.qq.com/s?__biz=MzkyMzI3MTA0Mw==&mid=2247530314&idx=1&sn=0a6a3ff0f81eee0652e8d7d0072badb1&chksm=c1e59da6f69214b032d9482642dfd823f1cae931609af927c48b61a2413e879f219911df683a&scene=132#wechat_redirect)

