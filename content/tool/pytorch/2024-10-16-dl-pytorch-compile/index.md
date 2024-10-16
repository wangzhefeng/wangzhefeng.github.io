---
title: PyTorch torch.Compile
author: 王哲峰
date: '2024-10-16'
slug: dl-pytorch-compile
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
img {
    pointer-events: none;
}
</style>

<details><summary>目录</summary><p>

- [torch.compile 特性](#torchcompile-特性)
    - [torch.compile](#torchcompile)
    - [TorchDynamo](#torchdynamo)
    - [AOTAutograd](#aotautograd)
    - [TorchInductor](#torchinductor)
    - [PrimTorch](#primtorch)
- [torch.compile 效果](#torchcompile-效果)
- [torch.compile 接口](#torchcompile-接口)
- [torch.compile 实验](#torchcompile-实验)
    - [试验设计](#试验设计)
    - [实验结论](#实验结论)
    - [测试代码](#测试代码)
        - [sin 函数](#sin-函数)
        - [resnet 18](#resnet-18)
        - [BERT](#bert)
        - [Numpy 计算](#numpy-计算)
- [参考](#参考)
</p></details><p></p>

# torch.compile 特性

> PyTorch 2.0 虽然发布了很多特性，但是对于入门/普通开发者而言，关系不大。
> PyTorch 从 2017 年发展到 2022 年已经过去五年，从最初的用户易用逐步向工程化、
> 高性能方向迭代，因此后续的版本更多是针对规模化落地、大型 AI 基建、平台兼容性等方面。
> 
> 其中最大的特点是 `torch.compile` 的发布，它是 PyTorch 走向编译范式的重要模块，
> 也是 PyTorch 2.0 发布会重点介绍的部分。

下面围绕 `torch.compile` 展开梳理几个与编译相关的核心特性。

## torch.compile

`torch.complie` 目的是提高计算速度。通常使用只需要一行代码即可完成，
例如：`model = torch.compile(model)`。之所以一行能让整个 PyTorch 运算提速，
是因为 `torch.complie` 是一个高级接口，
它背后 **使用了 `TorchDynamo`、`AOTAutograd` 和 `TorchInductor` 等工具链来对模型的计算图进行分析、优化和编译**。

`torch.complie` 是与开发者关系最大的、最有别于 PyTorch 1.x 的特性，
它背后对于计算图的分析、优化和编译是本次更新的核心构成，但对于普通用户而言，
了解好 `torch.compile` 的接口，了解其可提高模型计算速度就可以。

## TorchDynamo

`TorchDynamo` 是支撑 `torch.compile` 的工具，它可进行快速地捕获计算图（Graph），
计算图在深度学习中至关重要，它描述了数据在网络中的流动形式。
在早期，PyTorch 团队已经对计算图的捕获进行了一些列工具开发，例如 `TorchScript`。
但 `TorchDynamo` 相较于之前的工具，在速度上有了更大提升，并且在 99% 的情况下都能正确、
安全地获取计算图。

## AOTAutograd

`AOTAutograd` 的目的是希望在计算运行之前，捕获计算的反向传播过程，
即 “ahead of time Autograd”。`AOTAutograd` 通过重用和扩展 PyTorch 的现有自动微分系统，
实现提高训练速度。

## TorchInductor

`TorchInductor` 是一个新的编译器后端，可以为多个硬件平台进行生成优化的代码，
例如针对 NVIDIA 和 AMD 的 GPU，
使用 OpenAI 的 Triton 语言（一门 GPU 编程语言，不是 NVIDIA 的推理框架）作为目标语言，
针对 CPU，可生成 C++ 代码。由此可见，`TorchInductor` 能够为多种加速器和后端生成快速的代码。

## PrimTorch

`PrimTorch` 是将 PyTorch 底层操作符（operators）进行归约、精简，
使下游编译器开发更容易和高效。PyTorch 包含 1200+ 操作符，算上重载，
有 2000+，操作符过多，对于后端和编译器开发式不友好的。
为了简化后端开发，提高效率，`PrimTorch` 项目整理了两大类基础操作符，包括：

1. `Prim` 操作符：相对底层的约 250 个操作符
2. `ATen` 操作符：约 750 个操作符，适合直接导出

小结：`TorchDynamo`、`AOTAutograd`、
`TorchInductor` 和 `PrimTorch` 都在为 PyTorch 的计算效率服务，
让 PyTorch 计算速度更快、更 pythonic。

对于普通用户，重点关注 `torch.compile` 的接口使用，
接下来将对 `torch.compile` 的概念和使用展开说明。

# torch.compile 效果

得益于多个模块的优点组合，`torch.compile` 模式对大部分模型均有加速效果。

# torch.compile 接口

根据官方文档定义，"Optimizes given model/function using TorchDynamo and specified backend."。
`torch.compile` 是采用 `TorchDynamo` 和指定的后端对模型/计算进行优化，
期望使模型/函数在未来应用时，计算速度更快。

使用上，`torch.compile` 接收一个可调用对象(Callable)， 返回一个可调用对象(Callable)，
对于用户，只需要一行代码，调用 `torch.compile` 进行优化。

`torch.compile` 参数如下：

* `model`(Callable) : Module 或者是 Function，这个 Function 可以是 PyTorch 的函数，
  也可以是 Numpy 语句，当前 `torch.compile` 还支持 Numpy 的加速优化；
* mode：优化模式的选择，目前（2024 年 7 月 16 日）提供了四种模式，
  区别在于不同的存储消耗、时间消耗、性能之间的权衡；
    - `default`: 默认模式， 在性能和开销之间有不错的平衡；
    - `reduce-overhead`：这个模式旨在减少使用 CUDA 图时的 Python 开销。
      该模式会增加内存占用，提高速度，并且不保证总是有效。
      目前，这种方法只适用于那些不改变输入的 CUDA 图；
    - `max-autotune`：基于 Triton 的矩阵乘法和卷积来提高性能；
    - `max-autotune-no-cudagraphs`：与 `max-autotune` 一样，但是不会使用 CUDA 计算图。
* `fullgraph`(bool) : 是否将整个对象构建为单个图(a single graph)，否认是 `False`，
  即根据 `torch.compile` 的机制拆分为多个子图；
* `dynamic`(bool or None) : 是否采用动态形状追踪，默认为 `None`，对于输入形状是变化的，
  `torch.compile` 会尝试生成对应的 kernel 来适应动态形状，从而减少重复编译，
  但并不是所有动态形状都能这样操作，随缘吧，这个过程可以设置 `TORCH_LOGS=dynamic` 来观察日志信息；
* `backend`(str or Callable): 选择所用的后端，默认是 `"inductor"`，
  可以较好平衡性能和开销，可用的后端可以通过 `torch._dynamo.list_backends()` 查看，
  注册自定义后端库，可参考 https://pytorch.org/docs/main/compile/custom-backends.html；
* `options`(dict): 用于向后端传入额外数据信息，key-value 可以自定义，
  只要后端可读取即可，这个参数预留了较好的接口；
* `disable`(bool): Turn `torch.compile()` into a no-op for testing；


# torch.compile 实验

## 试验设计

为了充分观察 `torch.compile` 带来的速度变化，以及不同 `mode` 之间的影响，
下面针对四种情况分别进行速度的观察。四种情况包括：

* 简单的 PyTorch 运算
* ResNet18
* BERT
* Numpy 计算

并在三种型号 GPU 进行了测试，分别是 RTX 4060 Laptop GPU 、L20、H20。
注意，`torch.compile` 目前仅支持与 Linux 系统，并且不支持 python≥3.12。

## 实验结论

* 常见模型均有 10-30% 的耗时降低
* Numpy 也可以用 `torch.compile` 加速，并且耗时降低高达 90%
* 简单的 PyTorch 计算并无法带来速度提升（应该是没有复杂的计算图，无优化空间了）

## 测试代码

```python
import torch

mode_list = "default reduce-overhead max-autotune".split()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
```

### sin 函数

```python
import time

import torch


def sin_func(x):
    return torch.sin(x) + torch.cos(x)

run_times = 100000
i_data = torch.tensor(1).to(device)
for mode in mode_list:
    torch.cuda.synchronize()
    time_0 = time.time()
    module_compiled = torch.compile(sin_func, mode=mode)
    torch.cuda.synchronize()
    time_1 = time.time()
    
    # warmup
    sin_func(i_data)
    module_compiled(i_data)
    
    torch.cuda.synchronize()
    time_2 = time.time()
    for i in range(run_times):
        sin_func(i_data)
        
    torch.cuda.synchronize()
    time_3 = time.time()
    for i in range(run_times):
        module_compiled(i_data)
    torch.cuda.synchronize()
    time_4 = time.time()
    
    compile_time = time_1 - time_0
    pre_time = time_3 - time_2
    post_time = time_4 - time_3
    speedup_ratio = (pre_time - post_time)/pre_time
    
    print(f"mode: {mode}, 编译耗时:{compile_time:.2f}，编译前运行耗时:{pre_time:.2f}, 编译后运行耗时:{post_time:.2f}，速度提升比例:{speedup_ratio:.2%}")
    
```

### resnet 18

```python
import torch
from torchvision import models


resnet18 = models.resnet18().to(device)
resnet18.eval()
fake_img = torch.randn(16, 3, 224, 224).to(device)

run_times = 100
with torch.no_grad():
    for mode in mode_list:
        torch.cuda.synchronize()
        time_0 = time.time()
        module_compiled = torch.compile(resnet18, mode=mode)
        torch.cuda.synchronize()
        time_1 = time.time()
        
        # warmup 非常关键！
        resnet18(fake_img)
        module_compiled(fake_img)
        
        #
        torch.cuda.synchronize()
        time_2 = time.time()
        for i in range(run_times):
            resnet18(fake_img)
        
        torch.cuda.synchronize()
        time_3 = time.time()
        for i in range(run_times):
            module_compiled(fake_img)
        
        torch.cuda.synchronize()
        time_4 = time.time()

        compile_time = time_1 - time_0
        pre_time = time_3 - time_2
        post_time = time_4 - time_3
        speedup_ratio = (pre_time - post_time)/pre_time

        print(f"mode: {mode}, 编译耗时:{compile_time:.2f}，编译前运行耗时:{pre_time:.2f}, 编译后运行耗时:{post_time:.2f}，速度提升比例:{speedup_ratio:.2%}")
```

### BERT

```python
import time

import torch
from transformers import BertModel, BertTokenizer


bert = BertModel.from_pretrained('bert-base-uncased')
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')


# 准备一批输入数据
input_text = "Here is some text to encode"
inputs = tokenizer(input_text, return_tensors='pt', padding=True, truncation=True)
inputs = {k: v.to(device) for k, v in inputs.items()}
bert.to(device)
bert.eval()

run_times = 100
with torch.no_grad():
    for mode in mode_list:
        
        # 编译
        torch.cuda.synchronize()
        time_0 = time.time()
        bert_compiled = torch.compile(bert, mode=mode)
        torch.cuda.synchronize()
        time_1 = time.time()
        
        # warmup 非常关键！
        bert(**inputs)
        bert_compiled(**inputs)

        torch.cuda.synchronize()
        time_2= time.time()
        for _ in range(run_times): 
            _ = bert(**inputs)

        torch.cuda.synchronize()
        time_3= time.time()
        for _ in range(run_times):
            _ = bert_compiled(**inputs)
        
        torch.cuda.synchronize()
        time_4= time.time()
        
        compile_time = time_1 - time_0
        pre_time = time_3 - time_2
        post_time = time_4 - time_3
        speedup_ratio = (pre_time - post_time)/pre_time
        
        
        print(f"mode: {mode}, 编译耗时:{compile_time:.2f}，编译前运行耗时:{pre_time:.2f}, 编译后运行耗时:{post_time:.2f}，速度提升比例:{speedup_ratio:.2%}")
```

### Numpy 计算

```python
import numpy as np


run_times = 100

def numpy_fn2(X: np.ndarray, Y: np.ndarray) -> np.ndarray:
    return np.sum(X[:, :, None] * Y[:, None, :], axis=(-2, -1))


def numpy_fn(X: np.ndarray, Y: np.ndarray) -> np.ndarray:
    # Step 1: Normalize the input arrays to have zero mean and unit variance
    X_mean, X_std = X.mean(axis=0), X.std(axis=0)
    Y_mean, Y_std = Y.mean(axis=0), Y.std(axis=0)
    
    # Avoid division by zero in case of zero standard deviation
    X_std[X_std == 0] = 1
    Y_std[Y_std == 0] = 1
    
    X_normalized = (X - X_mean) / X_std
    Y_normalized = (Y - Y_mean) / Y_std
    
    # Step 2: Perform the tensor product followed by sum over last two dimensions
    intermediate_result = np.sum(X_normalized[:, :, None] * Y_normalized[:, None, :], axis=(-2, -1))
    
    # Step 3: Apply thresholding to clip values outside of [-1, 1]
    intermediate_result = np.clip(intermediate_result, -1, 1)
    
    # Step 4: Apply exponential function for non-linearity
    result = np.exp(intermediate_result)
    
    # Step 5: Add a small regularization term to avoid overfitting
    regularization_term = 0.001 * np.sum(X_normalized ** 2 + Y_normalized ** 2, axis=1)
    result += regularization_term
    
    return result


x = np.random.randn(1024, 640)
y = np.random.randn(1024, 640)

for mode in mode_list:
    torch.cuda.synchronize()
    time_0 = time.time()
    numpy_fn_compiled = torch.compile(numpy_fn, mode=mode)
    torch.cuda.synchronize()
    time_1 = time.time()

    # warmup 非常关键！
    numpy_fn(x, y)
    numpy_fn_compiled(x, y)

    #
    torch.cuda.synchronize()
    time_2 = time.time()
    for i in range(run_times):
        numpy_fn(x, y)

    torch.cuda.synchronize()
    time_3 = time.time()
    for i in range(run_times):
        numpy_fn_compiled(x, y)

    torch.cuda.synchronize()
    time_4 = time.time()

    compile_time = time_1 - time_0
    pre_time = time_3 - time_2
    post_time = time_4 - time_3
    speedup_ratio = (pre_time - post_time)/pre_time

    print(f"mode: {mode}, \
           编译耗时:{compile_time:.2f}，\
           编译前运行耗时:{pre_time:.2f}, \
           编译后运行耗时:{post_time:.2f}，\
           速度提升比例:{speedup_ratio:.2%}")
```

# 参考

* [PyTorch 2.0 与torch.compile](https://tingsongyu.github.io/PyTorch-Tutorial-2nd/chapter-1/1.7-PyTorch2.0.html)
* [torch-compile.ipynb](https://github.com/TingsongYu/PyTorch-Tutorial-2nd/blob/main/code/chapter-1/03-torch-compile.ipynb)
