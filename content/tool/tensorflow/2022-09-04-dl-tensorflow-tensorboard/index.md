---
title: TensorFlow TensorBoard
author: 王哲峰
date: '2022-09-04'
slug: dl-tensorflow-tensorboard
categories:
  - tensorflow
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
img {
    pointer-events: none;
}
</style>

<details><summary>目录</summary><p>

- [实时查看参数变化情况](#实时查看参数变化情况)
  - [TensorBoard 使用介绍](#tensorboard-使用介绍)
  - [TensorBoard 代码框架](#tensorboard-代码框架)
- [查看 Graph 和 Profile 信息](#查看-graph-和-profile-信息)
</p></details><p></p>

# 实时查看参数变化情况

## TensorBoard 使用介绍

1. 首先, 在代码目录下建立一个文件夹, 存放 TensorBoard 的记录文件

```bash
$ mkdir tensorboard
```

2. 在代码中实例化一个记录器

```python
summary_writer =  tf.summary.create_file_writer("./tensorboard")
```

3. 当需要记录训练过程中的参数时, 通过 `with` 语句指定希望使用的记录器, 并对需要记录的参数(一般是标量)运行:

```python
with summary_writer.as_default():
   tf.summary.scalar(name, tensor, step = batch_index)
```

4. 当要对训练过程可视化时, 在代码目录打开终端

```bash
$ tensorboard --logdir=./tensorboard
```

5. 使用浏览器访问命令行程序所输出的网址, 即可访问 TensorBoard 的可视化界面

- `http://计算机名称:6006`

.. note:: 

- 每运行一次 `tf.summary.scalar()`, 记录器就会向记录文件中写入一条记录
- 除了最简单的标量以外, TensorBoard 还可以对其他类型的数据, 如:图像、音频等进行可视化
- 默认情况下, TensorBoard 每 30 秒更新一次数据, 可以点击右上角的刷新按钮手动刷新
- TensorBoard 的使用有以下注意事项:
   - 如果需要重新训练, 那么删除掉记录文件夹内的信息并重启 TensorBoard, 
      或者建立一个新的记录文件夹并开启 TensorBoard, 将 `--logdir` 参数设置为新建里的文件夹
   - 记录文件夹目录许保持全英文

## TensorBoard 代码框架

```python

# (1)实例化一个记录器
summary_writer =  tf.summary.create_file_writer("./tensorboard")

# (2)开始训练模型
for batch_index in range(num_batches):
# ...(训练代码, 将当前 batch 的损失值放入变量 loss 中)

# (3)指定记录器
with summary_writer.as_default():
   tf.summary.scalar("loss", loss, step = batch_index)
   tf.summary.scalar("MyScalar", my_scalar, step = batch_index)
```

# 查看 Graph 和 Profile 信息

在训练时使用 `tf.summary.trace_on` 开启 Trace, 此时 TensorFlow 会将训练时的大量信息, 
如:计算图的结构、每个操作所耗费的时间等, 记录下来。

在训练完成后, 使用 `tf.summary.trace_export` 将记录结果输出到文件。


1.使用 TensorBoard 代码框架对模型信息进行跟踪记录

```python

# (1)实例化一个记录器
summary_writer =  tf.summary.create_file_writer("./tensorboard")

# (2)开启 Trace, 可以记录图结构和 profile 信息
tf.summary.trace_on(graph = True, profiler = True)

# (3)开始训练模型
for batch_index in range(num_batches):
   # (4)...(训练代码, 将当前 batch 的损失值放入变量 loss 中)
   
   # (5)指定记录器, 将当前指标值写入记录器
   with summary_writer.as_default():
      tf.summary.scalar("loss", loss, step = batch_index)
      tf.summary.scalar("MyScalar", my_scalar, step = batch_index)

# (6)保存 Trace 信息到文件
with summary_writer.as_default():
   tf.summary.trace_export(name = "model_trace", step = 0, profiler_outdir = log_dir)
```

2.在 TensorBoard 的菜单中选择 `PROFILE`, 以时间轴方式查看各操作的耗时情况, 
如果使用了 `@tf.function` 建立计算图, 也可以点击 `GRAPHS` 查看图结构



为了将训练好的机器学习模型部署到各个目标平台(如服务器、移动端、嵌入式设备和浏览器等), 
我们的第一步往往是将训练好的整个模型完整导出(序列化)为一系列标准格式的文件。在此基础上, 
我们才可以在不同的平台上使用相对应的部署工具来部署模型文件。

TensorFlow 提供了统一模型导出格式 `SaveModel`, 这是我们在 TensorFlow 2 中主要使用的导出格式。
这样我们可以以这一格式为中介, 将训练好的模型部署到多种平台上. 

同时, 基于历史原因, Keras 的 Sequential 和 Functional 模式也有自有的模型导出格式。

