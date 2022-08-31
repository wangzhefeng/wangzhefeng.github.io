---
title: TensorFlow
subtitle: TensorFlow Home
list_pages: true
# order_by: title
---


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

# Keras 和 tensorflow.keras

关于 Keras 和 tensorflow.keras，听说的经验：

* Keras 可以看成是一种深度学习框架的高阶接口规范，它帮助用户以更简洁的形式定义和训练深度学习网络模型。
  使用 pip 安装的 Keras 库同时在 TenforFlow、Theano、CNTK 等后端基础上进行了这种高阶接口规范的实现
* tensorflow.keras 是在 TensorFlow 中以 TensorFlow 低阶 API 为基础实现的高阶接口，
  它是 TensorFlow 的一个子模块。tensorflow.keras 绝大部分功能和兼容多种后端的 Keras 库用法完全一样，但并非全部，
  它和 TensorFlow 之间的结合更为紧密

结论：

* 随着谷歌对 Keras 的收购，Keras 库 2.3.0 版本后也将不再进行更新，
  用户应当使用 tensorflow.keras，而不是使用 pip 安装的 Keras

# 文档