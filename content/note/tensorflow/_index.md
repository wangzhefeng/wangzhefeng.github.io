---
title: TensorFlow
subtitle: TensorFlow Home
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
- [Keras 和 tensorflow.keras](#keras-和-tensorflowkeras)
- [经验总结](#经验总结)
  - [机器、深度学习任务问题](#机器深度学习任务问题)
  - [回归问题](#回归问题)
  - [二分类问题](#二分类问题)
  - [数据预处理问题](#数据预处理问题)
  - [样本量问题](#样本量问题)
  - [网络结构选择问题](#网络结构选择问题)
  - [优化器](#优化器)
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


# 经验总结

## 机器、深度学习任务问题

- 二分类
- 多分类
- 标量回归

## 回归问题

- 回归问题使用的损失函数
   - 均方误差(MSE)
- 回归问题使用的评估指标
   - 平均绝对误差(MAE)
- 回归问题网络的最后一层只有一个单元, 没有激活, 是一个线性层, 这是回归的典型设置, 添加激活函数会限制输出范围

## 二分类问题

- 二分类问题使用的损失函数
   - 对于二分类问题的 sigmoid 标量输出, `binary_crossentropy`
- 对于二分类问题, 网络的最后一层应该是只有一个单元并使用 sigmoid 激活的 Dense 层, 网络输出应该是 0~1 范围内的标量, 表示概率值

## 数据预处理问题

- 在将原始数据输入神经网络之前, 通常需要对其进行预处理
   - 结构化数据
   - 图像数据
   - 文本数据
- 将取值范围差异很大的数据输入到神经网络中是有问题的
   - 网路可能会自动适应这种取值范围不同的数据, 但学习肯定变得更加困难
   - 对于这种数据, 普遍采用的最佳实践是对每个特征做标准化, 即对于输入数据的每个特征(输入数据矩阵中的列), 
      减去特征平均值, 再除以标准差, 这样得到的特征平均值为 0, 标准差为 1
   - 用于测试数据标准化的均值和标准差都是在训练数据上计算得到的。在工作流程中, 不能使用测试数据上计算得到的任何结果, 
      即使是像数据标准化这么简单的事情也不行
- 如果输入数据的特征具有不同的取值范围, 应该首先进行预处理, 对每个特征单独进行缩放

## 样本量问题

* 如果可用的数据很少, 使用 K 折交叉验证可以可靠地评估模型
* 如果可用的训练数据很少, 最好使用隐藏层较少(通常只有一到两个)的小型模型, 以避免严重的过拟合
   - 较小的网络可以降低过拟合

## 网络结构选择问题

* 如果可用的训练数据很少, 最好使用隐藏层较少(通常只有一到两个)的小型模型, 以避免严重的过拟合
* 如果数据被分为多个类别, 那么中间层过小可能会导致信息瓶颈

## 优化器

* 无论问题是什么, `rmsprop` 优化器通常都是足够好的选择

# 文档