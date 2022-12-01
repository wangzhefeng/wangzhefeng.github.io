---
title: PyTorch
subtitle: 
list_pages: true
# order_by: title
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

- [TensorFlow 和 PyTorch](#tensorflow-和-pytorch)
- [PyTorch 技巧](#pytorch-技巧)
  - [指定 GPU 编号](#指定-gpu-编号)
  - [查看模型每层输出详情](#查看模型每层输出详情)
- [文档](#文档)
</p></details><p></p>

# TensorFlow 和 PyTorch

关于选择 TensorFlow 还是 PyTorch，听说来的经验是：

* 如果是工程师，应该优先选择 TensorFlow
* 如果是学生或者研究人员，应该优先选择 PyTorch
* 如果时间足够，最好 TensorFlow 和 PyTorch 都要掌握

理由如下：

1. 在工业界最重要的是模型落地，目前国内的大部分互联网企业只支持 TensorFlow 模型的在线部署，
   不支持 PyTorch。 并且工业界更加注重的是模型的高可用性，
   许多时候使用的都是成熟的模型架构，调试需求并不大
2. 研究人员最重要的是快速迭代发表文章，需要尝试一些较新的模型架构。
   而 PyTorch 在易用性上相比 TensorFlow 有一些优势，更加方便调试。
   并且在 2019 年以来在学术界占领了大半壁江山，能够找到的相应最新研究成果更多
3. TensorFlow 和 PyTorch 实际上整体风格已经非常相似了，学会了其中一个，
   学习另外一个将比较容易。两种框架都掌握的话，能够参考的开源模型案例更多，
   并且可以方便地在两种框架之间切换

# PyTorch 技巧

* [参考地址](https://mp.weixin.qq.com/s?__biz=MzkyMzI3MTA0Mw==&mid=2247530314&idx=1&sn=0a6a3ff0f81eee0652e8d7d0072badb1&chksm=c1e59da6f69214b032d9482642dfd823f1cae931609af927c48b61a2413e879f219911df683a&scene=132#wechat_redirect)


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

## 查看模型每层输出详情

* https://github.com/sksq96/pytorch-summary

```python
from torchsummary import summary

summary(model, input_size = (channels, H, W))
```

# 文档

