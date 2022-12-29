---
title: TensorFlow Keras 后端
author: 王哲峰
date: '2022-09-10'
slug: dl-tensorflow-keras-backend
categories:
  - deeplearning
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
</style>

<details><summary>目录</summary><p>

- [什么是 Keras 后端](#什么是-keras-后端)
- [从一个后端切换到另一个后端](#从一个后端切换到另一个后端)
- [keras.json 详细配置](#kerasjson-详细配置)
- [Backend API](#backend-api)
</p></details><p></p>

# 什么是 Keras 后端

Keras 后端:

   Keras 是一个模型级库, 为开发深度学习模型提供了高层次的构建模块。
   它不处理诸如张量乘积和卷积等低级操作。
   
   相反, 它依赖于一个专门的、优化的张量操作库来完成这个操作, 它可以作为 Keras 的「后端引擎」。
   相比单独地选择一个张量库, 而将 Keras 的实现与该库相关联, Keras 以模块方式处理这个问题, 
   并且可以将几个不同的后端引擎无缝嵌入到 Keras 中。

目前可用的 Keras 后端:

   - TensorFlow
   - Theano
   - CNTK

# 从一个后端切换到另一个后端

如果您至少运行过一次 Keras, 您将在以下位置找到 Keras 配置文件. 如果没有, 可以手动创建它.

Keras 配置文件位置:

```bash
# Liunx or Mac
$ vim $HOME/.keras/keras.json

# Windows
$ vim %USERPROFILE%/.keras/keras.json
```

Keras 配置文件创建:

```bash
$ cd ~/.keras
$ sudo subl keras.json
```

也可以定义环境变量 `KERAS_BACKEND`, 不过这会覆盖配置文件 `$HOME/.keras/keras.json` 中定义的内容:

```bash
KERAS_BACKEND=tensorflow python -c "from keras import backend" 
Using TensorFlow backend.
```

当前环境的 Keras 配置文件内容:


```json
{
   "floatx": "float32",
   "epsilon": 1e-07,
   "backend": "tensorflow",
   "image_data_format": "channels_last"
}
```

自定义 Keras 配置文件:

 - 在 Keras 中, 可以加载除了 "tensorflow", "theano" 和 "cntk"
    之外更多的后端。Keras 也可以使用外部后端, 这可以通过更改 keras.json
    配置文件和 "backend" 设置来执行。 假设您有一个名为 my_module 的 Python
    模块, 您希望将其用作外部后端。keras.json 配置文件将更改如下.
    - 必须验证外部后端才能使用, 有效的后端必须具有以下函数:

       - `placeholder`
       - `variable`
       - `function`

    - 如果由于缺少必需的条目而导致外部后端无效, 则会记录错误, 通知缺少哪些条目:

       ```bash
       {
          "image_data_format": "channels_last",
          "epsilon": 1e-07,
          "floatx": "float32",
          "backend": "my_package.my_module"
       }
       ```

# keras.json 详细配置

- `image_data_format`:
   - `"channels_last"`
      - (rows, cols, channels)
      - (conv*dim1, convdim2, conv_dim3, channels)
   - `"channels_first"`
      - (channels, rows, cols)
      - (channels, convdim1, convdim2, conv_dim3)
   - 在程序中返回: `keras.backend.image_data_format()`
- `epsilon`:
   - 浮点数, 用于避免在某些操作中被零除的数字模糊常量
- `floatx`:
   - 字符串: `float16`, `float32`, `float64`\ 。默认浮点精度
- `backend`:
   - 字符串: `tensorflow`, `theano`, `cntk`

# Backend API

* `tf.keras.backend.clear_session()`
* `tf.keras.backend.epsilon()`
    - 返回数字表达式中使用的模糊因子的值
* `tf.keras.backend.floatx()`
    - 返回默认的 float 类型
* `tf.keras.backend.get_uid()`
* `tf.keras.backend.image_data_format()`
    - 返回设置图像数据格式约定的值
* `tf.keras.backend.is_keras_tensor()`
* `tf.keras.backend.reset_uids()`
* `tf.keras.backend.rnn()`
* `tf.keras.backend.set_epsilon()`
    - 设置数字表达式中使用的模糊因子的值
* `tf.keras.backend.set_floatx()`
    - 设置 float 类型
* `tf.keras.backend.set_image_data_format()`
    - 设置图像数据格式约定的值

