---
title: TensorFlow 计算图
author: 王哲峰
date: '2022-07-04'
slug: dl-tensorflow-graph
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
</style>

<details><summary>目录</summary><p>

- [静态计算图](#静态计算图)
  - [TensorFlow 1.0 静态计算图](#tensorflow-10-静态计算图)
  - [TensorFlow 2.0 计算图](#tensorflow-20-计算图)
- [动态计算图](#动态计算图)
  - [普通动态计算图](#普通动态计算图)
  - [动态计算图封装](#动态计算图封装)
- [AutoGraph](#autograph)
  - [AutoGraph 示例](#autograph-示例)
  - [AutoGraph 使用规范](#autograph-使用规范)
    - [使用 TensorFlow 函数](#使用-tensorflow-函数)
    - [避免装饰的函数内部定义 tf.Variable](#避免装饰的函数内部定义-tfvariable)
    - [装饰的函数内部不可修改外部 Python 数据结构](#装饰的函数内部不可修改外部-python-数据结构)
  - [AutoGraph 机制原理](#autograph-机制原理)
    - [创建装饰函数原理](#创建装饰函数原理)
    - [调用装饰函数原理](#调用装饰函数原理)
    - [再次调用装饰函数原理](#再次调用装饰函数原理)
    - [再次调用装饰函数原理](#再次调用装饰函数原理-1)
  - [AutoGraph 和 tf.Module](#autograph-和-tfmodule)
    - [AutoGraph 和 tf.Module 概述](#autograph-和-tfmodule-概述)
    - [应用 tf.Module 封装 AutoGraph](#应用-tfmodule-封装-autograph)
    - [tf.Module 和 tf.keras.Model, tf.keras.layers.Layer](#tfmodule-和-tfkerasmodel-tfkeraslayerslayer)
  - [使用 tf.function 提升性能](#使用-tffunction-提升性能)
    - [@tf.funciton: 图执行模式](#tffunciton-图执行模式)
    - [@tf.function 基础使用方法](#tffunction-基础使用方法)
    - [@tf.function 内在机制](#tffunction-内在机制)
    - [AutoGraph: 将 Python 控制流转化为 TensorFlow 计算图](#autograph-将-python-控制流转化为-tensorflow-计算图)
    - [使用传统的 tf.Session](#使用传统的-tfsession)
  - [分析 TenforFlow 的性能](#分析-tenforflow-的性能)
  - [图优化](#图优化)
  - [混合精度](#混合精度)
</p></details><p></p>


计算图由节点(nodes)和线(edges)组成:

* 节点表示操作符(Operation)，或者称之为算子
* 线表示计算间的依赖

实现表示有数据的传递依赖，传递的数据即张量，虚线通常可以表示控制依赖，即执行先后顺序

TensorFlow 有三种计算图的构建方式

* 静态计算图
* 动态计算图
* AutoGraph

# 静态计算图

TensorFlow 1.0 采用的是静态计算图，需要先使用 TensorFlow 的各种算子创建计算图，
然后再开启一个会话 Session，显式执行计算图

在 TensorFlow 1.0 中，使用静态计算图分两步:

1. 定义计算图
2. 在会话中执行计算图

## TensorFlow 1.0 静态计算图

```python
import tensorflow as tf

# 定义计算图
graph = tf.Graph()
with graph.as_default():
    # placeholder 为占位符，执行会话的时候指定填充对象
    x = tf.placeholder(name = "x", shape = [], dtype = tf.string)
    y = tf.placeholder(name = "y", shape = [], dtype = tf.string)
    z = tf.string_join([x, y], name = "join", separator = " ")

# 执行计算图
with tf.Session(graph = graph) as sess:
    print(sess.run(
        fetches = z, 
        feed_dict = {x: "hello", y: "world"}
    ))
```

## TensorFlow 2.0 计算图

TensorFlow 2.0 为了确保对老版本 TensorFlow 项目的兼容性，
在 `tf.compat.v1` 子模块中保留了对 TensorFlow 1.0 静态计算图构建风格的支持。
已经不推荐使用了

```python
import tensorflow as tf

# 定义计算图
graph = tf.compat.v1.Graph()
with graph.as_default():
    # placeholder 为占位符，执行会话的时候指定填充对象
    x = tf.compat.v1.placeholder(
        name = "x", 
        shape = [], 
        dtype = tf.string
    )
    y = tf.compat.v1.placeholder(
        name = "y", 
        shape = [], 
        dtype = tf.string
    )
    z = tf.strings.join([x, y], name = "join", separator = " ")

# 执行计算图
with tf.compat.v1.Session(graph = graph) as sess:
    # fetches 的结果非常像一个函数的返回值
    # feed_dict 中的占位符相当于函数的参数序列
    result = sess.run(
        fetches = z,
        feed_dict = {
            x: "hello",
            y: "world",
        }
    )
    print(result)
```

# 动态计算图

TensorFlow 2.0 采用的是动态计算图，即每使用一个算子后，
该算子会被动态加入到隐含的默认计算图中立即执行得到结果，
而无需开启 Session

在 TensorFlow 2.0 中，使用的是动态计算图和 AutoGraph。
动态计算图已经不区分计算图的定义和执行了，
而是定义后立即执行，因此称之为 Eager Excution，立即执行

* 使用动态计算图(Eager Excution)的好处是方便调试程序
    - 动态计算图会让 TensorFlow 代码的表现和 Python 原生代码的表现一样，
      写起来就像 Numpy 一样，各种日志打印，控制流全部都是可以使用的
* 使用动态图的缺点是运行效率相对会低一点
    - 因为使用动态图会有许多次 Python 进程和 TensorFlow 的 C++ 进程之间的通信
    - 而静态计算图构建完成之后几乎全部在 TensorFlow 内核上使用 C++ 代码执行，效率更高。
      此外，静态图会对计算步骤进行一定的优化，去除和结果无关的计算步骤 

## 普通动态计算图

* 动态计算图在每个算子处都进行构建，构建后立即执行

```python
x = tf.constant("hello")
y = tf.constant("world")
z = tf.strings.join([x, y], separator = " ")
tf.print(z)
```

```
hello world
```

## 动态计算图封装

* 可以将动态计算图代码的输入和输出关系封装成函数

```python
def strjoin(x, y):
    z = tf.strings.join([x, y], separator = " ")
    tf.print(z)
    return z

result = strjoin(
    x = tf.constant("hello"),
    y = tf.constant("world"),
)
pritn(result)
```

```
hello world
tf.Tensor(b'hello world', shape=(), dtype=string)
```

# AutoGraph

动态计算图运行效率相对较低，如果需要在 TensorFlow 中使用静态图，
可以使用 `@tf.function` 装饰器将普通 Python 函数转换成对应的 TensorFlow 计算图构建代码。
运行该函数就相当于在 TensorFlow 1.0 中用 Session 执行代码。
使用 `@tf.function` 构建静态图的方式叫做 AutoGraph

在 TensorFlow 2.0 中，使用 AutoGraph 的方式使用计算图分两步:

1. 定义计算图变成了定义函数
2. 执行计算图变成了调用函数

在 AutoGraph 中不需要使用会话，一切都像原始的 Python 语法一样自然。
实践中，一般会先用动态计算图调试，
然后在需要提高性能的地方利用 `@tf.function` 切换成 AutoGraph 获得更高的效率。
当然，`@tf.function` 的使用需要遵循一定的规范

## AutoGraph 示例

* 使用 AutoGraph 构建静态图

```python
import tensorflow as tf

@tf.function
def strjoin(x, y):
    z = tf.strings.join([x, y], separator = " ")
    tf.print(z)
    return z

result = strjoin(
    x = tf.constant("hello"),
    y = tf.constant("world"),
)
print(result)
```

```
hello world
tf.Tensor(b'hello world', shape=(), dtype=string)
```

* 创建日志

```python
import os
import datetime
from pathlib import Path

stamp = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:S')
logdir = os.path.join("data", "autograph", stamp)
# or
stamp = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:S')
logdir = str(Path("./data/autograph/" + stamp))
```

* 查看计算图

```python
import tensorflow as tf

# 日志写入器
writer = tf.summary.create_file_writer(logdir)

# 开启 AutoGraph 跟踪
tf.summary.trace_on(graph = True, profiler = True)

# 执行 AutoGraph
result = strjoin(
    x = "hello",
    y = "world",
)

# 将计算图信息写入日志
with writer.as_default():
    tf.summary.trace_export(
        name = "autograph",
        step = 0,
        profile_outdir = logdir,
    )
```

* 启动 tensorboard 在 jupyter 中的魔法命令

```python
%load_ext tensorboard
%tensorboard --logdir ./data/autograph/
```

## AutoGraph 使用规范

* 静态计算图执行效率很高，但较难调试
* 动态计算图易于调试，编码效率低，执行效率偏低
* AutoGraph 机制可以将动态图转换成静态计算图，兼收执行效率和编码效率之利

AutoGraph 机制能够转换的代码并不是没有任何约束的，
有一些编码规范需要遵循，否则有可能会转换失败或者不符合预期。
所以这里总结了 AutoGraph 编码规范

### 使用 TensorFlow 函数

被 `@tf.function` 修饰的函数应尽可能使用 TensorFlow 中的函数而不是 Python 中的其他函数。例如:

- 使用 `tf.print()` 而不是 `print()`
- 使用 `tf.range()` 而不是 `range()`
- 使用 `tf.constant(True)` 而不是 `True`

解释：Python 中的函数仅仅会在跟踪执行函数以创建静态图的阶段使用，
普通 Python 函数是无法嵌入到静态计算图中的，所以在计算图构建好之后再次调用的时候，
这些 Python 函数并没有被计算，而 TensorFlow 中的函数则可以嵌入到计算图中。
使用普通的 Python 函数会导致 被 `@tf.function` 修饰前【eager 执行】和
被 `@tf.function` 修饰后【静态图执行】的输出不一致

```python
import numpy as np
import tensorflow as tf

@tf.function
def np_random():
    a = np.random.randn(3, 3)
    tf.print(a)

# np_random() 每次执行都是一样的结果
np_random()
np_random()
```

```python
import tensorflow as tf

@tf.function
def tf_random():
    a = tf.random.normal((3, 3)
    tf.print(a)

# tf_random() 每次执行都会重新生成随机数
tf_random()
tf_random()
```

### 避免装饰的函数内部定义 tf.Variable

避免在 `@tf.function` 修饰的函数内部定义 `tf.Variable`

解释：如果函数内部定义了 `tf.Variable`, 那么在【eager执行】时，
这种创建 `tf.Variable` 的行为在每次函数调用时候都会发生。但是在【静态图执行】时，
这种创建 `tf.Variable` 的行为只会发生在第一步跟踪 Python 代码逻辑创建计算图时，
这会导致被 `@tf.function` 修饰前【eager执行】和被 `@tf.function` 修饰后【静态图执行】的输出不一致。
实际上，TensorFlow 在这种情况下一般会报错

```python
import tensorflow as tf

x = tf.Variable(1.0, dtype = tf.float32)

@tf.function
def outer_var():
    x.assign_add(1.0)
    tf.print(x)
    return (x)

outer_var()
```

```python
import tensorflow as tf

@tf.function
def inner_var():
    x = tf.Variable(1.0, dtype = tf.float32)
    x.assign_add(1.0)
    tf.print(x)
    return (x)

# 执行将报错
inner_var()
```

### 装饰的函数内部不可修改外部 Python 数据结构

被 `@tf.function` 修饰的函数不可修改该函数外部的 Python 列表或字典等数据结构变量

解释：静态计算图是被编译成 C++ 代码在 TensorFlow 内核中执行的。
Python 中的列表和字典等数据结构变量是无法嵌入到计算图中，
它们仅仅能够在创建计算图时被读取，
在执行计算图时是无法修改 Python 中的列表或字典这样的数据结构变量的

```python
tensor_list = []

def append_tensor(x):
    tensor_list.append(x)
    return tensor_list

append_tensor(tf.constant(5.0))
append_tensor(tf.constant(6.0))
print(tensor_list)
```

```
[<tf.Tensor: shape=(), dtype=float32, numpy=5.0>, 
 <tf.Tensor: shape=(), dtype=float32, numpy=6.0>]
```

```python
tensor_list = []

@tf.function  # 加上这一行切换成 AutoGraph 结果将不符合预期
def append_tensor(x):
    tensor_list.append(x)
    return tensor_list

append_tensor(tf.constant(5.0))
append_tensor(tf.constant(6.0))
print(tensor_list)
```

```
[<tf.Tensor 'x:0' shape=() dtype=float32>]
```

## AutoGraph 机制原理

### 创建装饰函数原理

当使用 `@tf.function` 装饰一个函数的时候，背后发生了什么？

- 背后什么都没发生，仅仅是在 Python 堆栈中记录了这样一个函数的签名

```python
import numpy as np
import tensorflow as tf

@tf.function(autograph = True)
def my_add(a, b):
    for i in tf.range(3):
        tf.print(i)
    c = a + b
    print("tracing")
    
    return c
```

### 调用装饰函数原理

当第一次调用被 `@tf.function` 装饰的函数时，背后发生了什么？

```python
my_add(
    a = tf.constant("hello"),
    b = tf.constant("world"),
)
```

```
traceing
0
1
2
```

1. 创建计算图

即创建一个静态计算图，跟踪执行一遍函数体中的 Python 代码，
确定各个变量的 Tensor 类型，并根据执行顺序将算子添加到计算图中

在这个过程中，如果开启了 `autograph=True` (默认开启)，
会将 Python 控制流转换成 TensorFlow 图内控制流。 
主要是将 `if` 语句转换成 `tf.cond` 算子表达，
将 `while` 和 `for` 循环语句转换成 `tf.while_loop` 算子表达，
并在必要的时候添加 `tf.control_dependencies` 指定执行顺序依赖关系

相当于在 TensorFlow 1.0 执行了类似下面的语句:

```python
graph = tf.Graph()
with graph.as_default():
    a = tf.placeholder(shape = [], dtype = tf.string)
    b = tf.placeholder(shape = [], dtype = tf.string)
    cond = lambda i: i < tf.constant(3)
    def body(i):
        tf.print(i)
        return (i + 1)
    loop = tf.while_loop(cond, body, loop_vars = [0])
    loop
    with tf.control_dependencies(loop):
        c = tf.strings.join([a, b])
    print("tracing")
```

2. 执行计算图

相当于在 TensorFlow 1.0 中执行了下面的语句

```python
with tf.Session(graph = graph) as sess:
    sess.run(c, feed_dict = {
        a: tf.constant("hello"),
        b: tf.constant("world"),
    })
```

### 再次调用装饰函数原理

当再次用相同的输入参数类型调用被 `@tf.function` 装饰的函数时，背后发生了什么？

```python
my_add(
    a = tf.constant("good"),
    b = tf.constant("morning"),
)
```

```
0
1
2
```

### 再次调用装饰函数原理

当再次用不同的的输入参数类型调用被 `@tf.function` 装饰的函数时，背后到底发生了什么？

* 由于输入参数的类型已经发生变化，已经创建的计算图不能够再次使用。
  需要重新做2件事情：创建新的计算图、执行计算图

```python
my_add(
    a = tf.constant(1),
    b = tf.constant(2),
)
```

```
tracing
0
1
2
```

* 需要注意的是，如果调用被 `@tf.function` 装饰的函数时输入的参数不是 `Tensor` 类型，
  则每次都会重新创建计算图

```python
my_add("hello", "world")
my_add("good", "moning")
```

```
tracing
0
1
2
tracing
0
1
2
```

## AutoGraph 和 tf.Module

### AutoGraph 和 tf.Module 概述

AutoGraph 的编码规范中提到，在构建 AutoGraph 时应该避免在 `@tf.function` 修饰的函数内部定义 `tf.Variable`。
但是如果在被修饰的函数外部定义 `tf.Variable`，又会显得这个函数由外部变量依赖，封装不够完美。
一种简单的思路是定义一个类，并将相关的 `tf.Variable` 创建放在类的初始化方法中。而将函数的逻辑放在其他方法中。

TensorFlow 提供了一个基类 `tf.Module`，通过继承它构建子类，不仅可以解决上述问题，而且可以非常方便地管理变量，
还可以非常方便地管理它引用的其他 Module，最重要的是，能够利用 `tf.saved_model` 保存模型并实现跨平台部署使用

实际上，`tf.keras.models.Model`、`tf.keras.layers.Layer` 都是继承自 `tf.Module` 的，
提供了方便的变量管理和所引用的子模块管理的功能

### 应用 tf.Module 封装 AutoGraph

* 定义一个简单的 function

```python
import tensorflow as tf

x = tf.Variable(1.0, dtype = tf.float32)

@tf.function(
    input_signature = [tf.TensorSpec(shape = [], dtype = tf.float32)]
)
def add_print(a):
    x.assign_add(a)
    tf.print(x)
    return (x)

add_print(tf.constant(3.0))
add_print(tf.constant(3))  # 输入不符合张量签名的参数将报错
```

```
4
```

* 利用 `tf.Module` 封装函数

```python
class DemoModule(tf.Module):
    def __init__(self, init_value = tf.constant(0.0), name = None):
        super(DemoModule, self).__init__(name = name)
        with self.name_scope:
            self.x = tf.Variable(
                init_value, 
                dtype = tf.float32, 
                trainable = True
            )
    
    @tf.function(input_signature = [tf.TensorSpec(shape = [], dtype = tf.float32)])
    def add_print(self, a):
        with self.name_scope:
            self.x.assign_add(a)
            tf.print(self.x)
            return (self.x)
```

* 调用类

```python
demo = DemoModule(init_value = tf.constant(1.0))
result = demo.add_print(tf.constant(5.0))
```

```
6
```

* 查看模块中的全部变量和全部可训练变量

```python
print(demo.variables)
print(demo.trainable_variables)
```

```
(<tf.Variable 'demo_module/Variable:0' shape=() dtype=float32, numpy=6.0>,)
(<tf.Variable 'demo_module/Variable:0' shape=() dtype=float32, numpy=6.0>,)
```

* 查看模块中的全部子模块

```python
demo.submodules
```

* 使用 `tf.saved_model` 保存模型，并指定需要跨平台部署的方法

```python
tf.saved_model.save(
    demo, 
    "./data/demo/1", 
    signatures = {
        "serving_default": demo.add_print
    }
)
```

* 加载模型

```python
demo2 = tf.saved_model.load("./data/demo/1")
demo2.add_print(tf.constant(5.0))
```

```
11
```

* 查看模型文件相关信息

```bash
$ !sabed_model_cli show --dir ./data/demo/1 --all
```

![img](images/)

* 在 TensorBoard 中查看计算图，模块会被添加模块名 `demo_module`，方便层次化呈现计算图结构

```python
import datetime

# 创建日志
stamp = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:S')
logdir = f"./data/demomodule/{stamp}"
writer = tf.summary.create_file_writer(logdir)

# 开启 autograph 跟踪
tf.summary.trace_on(graph = True, profiler = True)

# 执行 autograph
demo = DemoModule(init_value = tf.constant(0.0))
result = demo.add_print(tf.constant(5.0))

# 将计算图信息写入日志
with writer.as_default():
    tf.summary.trace_export(
        name = "demomodule",
        step = 0,
        profiler_outdir = logdir,
    )

# 启动 tensorboard 在 jupyter 中的魔法命令
%reload_ext tensorboard
from tensorboard import notebook
notebook.start("--logdir ./data/demomodule/")
```

* 通过给 `tf.Module` 添加属性的方法进行封装

```python
my_module = tf.Module()
my_module.x = tf.Variable(0.0)

@tf.function(input_signature = [tf.TensorSpec(shape = [], dtype = tf.float32)])
def add_print(a):
    my_module.x.assign_add(a)
    tf.print(my_module.x)
    return (my_module.x)

my_module.add_print = add_print
my_module.add_print(tf.constant(1.0)).numpy()
print(my_module.variables)

# 使用 tf.saved_model 保存模型
tf.saved_model(
    my_module,
    "./data/my_module",
    signatures = {
        "serving_default": my_module.add_print
    }
)

# 加载模型
my_module2 = tf.saved_model.load("./data/my_module")
my_module2.add_print(tf.constant(5.0))
```

```
1.0
(<tf.Variable 'Variable:0' shape=() dtype=float32, numpy=0.0>,)

INFO:tensorflow:Assets written to: ./data/mymodule/assets
5
```

### tf.Module 和 tf.keras.Model, tf.keras.layers.Layer

`tf.keras` 中的模型和层都是继承 `tf.Module` 实现的，也具有变量管理和子模块管理功能

```python
import tensorflow as tf
from tensorflow.keras import models, layers, losses, metrics

print(issubclass(tf.keras.Model, tf.Module))
print(issubclass(tf.keras.layers.Layer, tf.Module))
print(issubclass(tf.keras.Model, tf.keras.layers.Layer))
```

```
True
True
True
```

```python
tf.keras.backend.clear_session()

model = models.Sequential()
model.add(layers.Dense(4, input_shape = (10,)))
model.add(layers.Dense(2))
model.add(layers.Dense(1))

model.summary()
model.variables
model.layers[0].trainable = False  # 冻结第 0 层的边变量，使其不可训练
model.trainable_variable
model.submodules
model.layers
model.name
model.name_scope()
```

## 使用 tf.function 提升性能

### @tf.funciton: 图执行模式

虽然目前 TensorFlow 默认的即时执行模式具有灵活及易调试的特性, 但在特定的场合, 
例如追求高性能或部署模型时, 依然希望使用图执行模式, 将模型转换为高效的 TensorFlow 图模型。

TensorFlow 2 提供了 ``bashtf.function` 模块, 结合 AutoGraph 机制, 使得我们仅需加入一个简单的
`@tf.function` 修饰符, 就能轻松将模型以图执行模式运行。

### @tf.function 基础使用方法


`@tf.function` 的基础使用非常简单, 只需要将我们希望以图执行模式运行的代码封装在一个函数内, 
并在函数前面加上 `@tf.function` 即可.


### @tf.function 内在机制


### AutoGraph: 将 Python 控制流转化为 TensorFlow 计算图


### 使用传统的 tf.Session


## 分析 TenforFlow 的性能


## 图优化


## 混合精度