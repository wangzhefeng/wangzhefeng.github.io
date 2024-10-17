---
title: PyTorch 自动微分
author: wangzf
date: '2022-07-17'
slug: dl-pytorch-api-autograd
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

- [autograd 包](#autograd-包)
    - [requires\_grad 参数和属性](#requires_grad-参数和属性)
    - [requires\_grad\_ 方法](#requires_grad_-方法)
    - [grad\_fn 属性](#grad_fn-属性)
    - [no\_grad 方法](#no_grad-方法)
    - [detech 方法](#detech-方法)
    - [zero\_grad 方法](#zero_grad-方法)
- [使用 backward 方法求导数](#使用-backward-方法求导数)
    - [标量的反向传播](#标量的反向传播)
    - [非标量的反向传播](#非标量的反向传播)
    - [非标量的反向传播可以用标量的反向传播实现](#非标量的反向传播可以用标量的反向传播实现)
- [使用 grad 方法求导数](#使用-grad-方法求导数)
    - [标量的反向传播](#标量的反向传播-1)
    - [多个自变量求导](#多个自变量求导)
- [使用自动微分和优化器求最小值](#使用自动微分和优化器求最小值)
- [参考](#参考)
</p></details><p></p>

神经网络通常依赖反向传播算法求梯度来更新网络参数，
而求梯度过程通常是一件非常复杂繁琐而且容易出错的事情。
深度学习框架可以帮助自动地完成这种求梯度的运算。

PyTorch 一般通过反向传播 `torch.autograd.backward()` 方法实现求梯度计算，
该方法求得的梯度将保存在对应自变量张量的 `grad` 属性下。除此之外，
也能够调用 `torch.autograd.grad()` 函数实现求梯度计算。这就是 PyTorch 的自动微分机制。

# autograd 包

`torch.autograd` 包提供了对所有 Tensor 的自动微分操作：

* `torch.Tensor(requires_grad = True)`
    - `requires_grad` 参数：跟踪 `torch.Tensor` 上所有的操作
    - `.requires_grad` 属性：`torch.Tensor` 是否被跟踪
* `.requires_grad_()` 方法：能够改变一个已经存在的 Tensor 的 `.requires_grad` 属性
* `.grad_fn` 属性：内存中的梯度函数
* `torch.autograd.backward()`：自动计算所有的梯度
    - `.grad` 属性：`torch.Tensor` 上的梯度
* `torch.autograd.grad()`：自动计算所有的梯度
* `.detach()` 方法：停止跟踪 `torch.Tensor` 上的跟踪历史、未来的跟踪
* `with torch.no_grad(): pass`
    - 用在测试数据集模型推断过程中
* `.zero_grad()` 方法
    - `optimizer.zero_grad()`

## requires_grad 参数和属性

> `requires_grad` 参数：跟踪 `torch.Tensor` 上所有的操作

```python
import torch

x = torch.ones(2, 2, requires_grad = True)
print("x:", x)
```

```
tensor([[1., 1.],
        [1., 1.]], requires_grad=True)
```

> `.requires_grad` 属性：`torch.Tensor` 是否被跟踪

```python
print(f"x.requires_grad: {x.requires_grad}")
```

```python
True
```

## requires_grad_ 方法

> `.requires_grad_()` 方法：能够改变一个已经存在的 Tensor 的 `.requires_grad` 属性

```python
import torch

x = torch.ones(2, 2, requires_grad = True)
print("x:", x)
print(f"x.requires_grad: {x.requires_grad}")
```

```
tensor([[1., 1.],
        [1., 1.]], requires_grad=True)
True
```

```python
x = x.requires_grad_(False)
print(f"x.requires_grad: {x.requires_grad}")

x = x.requires_grad_()
print(f"x.requires_grad: {x.requires_grad}")
```

```
False
True
```

## grad_fn 属性

> `.grad_fn` 属性：内存中的梯度函数

前向传播：

```python
import torch

x = torch.ones(2, 2, requires_grad = True)
y = x + 2
print("y:", y)
print("y.grad_fn:", y.grad_fn)
```

```
tensor([[3., 3.],
        [3., 3.]], grad_fn=<AddBackward0>)

<AddBackward0 object at 0x10cd1b950>
```

前向传播：

```python
z = y * y * 3
print("z:", z)
print("z.grad_fn", z.grad_fn)
```

```
tensor([[27., 27.],
        [27., 27.]], grad_fn=<MulBackward0>)

<MulBackward0 object at 0x10cd1b950>
```

前向传播：

```python
out = z.mean()
print("out:", out)
print("out.grad_fn:", out.grad_fn)
```

```
tensor(27., grad_fn=<MeanBackward0>)

<MeanBackward0 object at 0x10cd1b990>
```

后向传播求梯度：

```python
out.backward()
print("x.grad:", x.grad)
```

```
tensor([[4.5000, 4.5000],
        [4.5000, 4.5000]])
```

## no_grad 方法

> 用在测试数据集模型推断过程中，不需要求梯度
> 
> ```python
> with torch.no_grad(): 
>     pass
> ```

```python
import torch

x = torch.randn(3, requires_grad = True)
y = x * 2
while y.data.norm() < 1000:
    y = y * 2
v = torch.tensor([0.1, 1.0, 0.0001], dtype = torch.float)
y.backward(v)

# .requires_grad
print(x)
print(x.requires_grad)
print((x ** 2).requires_grad)

# .grad
print(x.grad)

# torch.no_grad()
with torch.no_grad():
    print((x ** 2).requires_grad)
```

```
tensor([-1.3832,  1.2448, -0.7524], requires_grad=True)
True
True
tensor([5.1200e+01, 5.1200e+02, 5.1200e-02])
False
```

## detech 方法

> `.detach()` 方法：停止跟踪 `torch.Tensor` 上的跟踪历史、未来的跟踪

```python
print(x.requires_grad)
y = x.detach()

print(y.requires_grad)
print(x.eq(y).all())
```

```
True
False
tensor(True)
```

## zero_grad 方法

> `.zero_grad()` 方法
>
> * `optimizer.zero_grad()`

* TODO

# 使用 backward 方法求导数

PyTorch 自动微分机制使用的是 `torch.autograd.backward()` 方法，
功能就是自动求取梯度。

* `torch.autograd.backward()` 方法通常在一个标量张量上调用，
  该方法求得的梯度将保存在对应自变量张量的 `grad` 属性下
* 如果调用的张量非标量，则要传入一个和它形状相同的 `gradient` 参数张量，
  相当于用该 `gradient` 参数张量与调用张量作**向量点乘**，得到的标量结果再反向传播

下面是 `torch.autograd.backward()` API:

```python
torch.autograd.backward(
   tensors, 
   gard_tensors = None, 
   retain_graph = None, 
   create_graph = False
)
```

## 标量的反向传播

```python
import numpy as np
import torch

"""
求 f(x) = a*x**2 + b*x + c 关于 x 的在 a=1.0，b=-2.0，c=1.0 处的导数
"""

# 自变量(标量)
x = torch.tensor(0.0, requires_grad = True)  # x 需要被求导
# 参数
a = torch.tensor(1.0)
b = torch.tensor(-2.0)
c = torch.tensor(1.0)
# 因变量
y = a * torch.pow(x, 2) + b * x + c

# 因变量 y 反向传播计算对 自变量 x 求导
y.backward()
dy_dx = x.grad
print(dy_dx)
```

```
tensor(-2.)
```

## 非标量的反向传播

```python
import numpy as np
import torch

"""
求 f(x) = a*x**2 + b*x + c 关于 x 的在 a=1.0，b=-2.0，c=1.0 处的导数
"""

# 自变量(张量)
x = torch.tensor(
    [[0.0, 0.0],
     [1.0, 2.0]], 
    requires_grad = True
)  # x 需要被求导
# 参数
a = torch.tensor(1.0)
b = torch.tensor(-2.0)
c = torch.tensor(1.0)
# 因变量
y = a * torch.pow(x, 2) + b * x + c
# 梯度张量
gradient = torch.tensor([[1.0, 1.0], [1.0, 1.0]])

print(f"x:\n{x}")
print(f"y:\n{y}")

y.backward(gradient = gradient)
x_grad = x.grad
print(f"x_grad:\n{x_grad}")
```

```
x:
 tensor([[0., 0.],
        [1., 2.]], requires_grad=True)
y:
 tensor([[1., 1.],
        [0., 1.]], grad_fn=<AddBackward0>)
x_grad:
 tensor([[-2., -2.],
        [ 0.,  2.]])
```

## 非标量的反向传播可以用标量的反向传播实现

```python
import numpy as np
import torch

"""
求 f(x) = a*x**2 + b*x + c 关于 x 的在 a=1.0，b=-2.0，c=1.0 处的导数
"""

# 自变量(张量)
x = torch.tensor(
    [[0.0, 0.0],
     [1.0, 2.0]], 
    requires_grad = True
)  # x 需要被求导
# 参数
a = torch.tensor(1.0)
b = torch.tensor(-2.0)
c = torch.tensor(1.0)
# 因变量
y = a * torch.pow(x, 2) + b * x + c
# 梯度
gradient = torch.tensor([[1.0, 1.0], [1.0, 1.0]])
z = torch.sum(y * gradient)

print(f"x:\n{x}")
print(f"y:\n{y}")

z.backward()
x_grad = x.grad
print(f"x_grad:\n{x_grad}")
```

```
x: 
tensor([[0., 0.],
        [1., 2.]], requires_grad=True)
y: 
tensor([[1., 1.],
        [0., 1.]], grad_fn=<AddBackward0>)
x_grad:
tensor([[-2., -2.],
        [ 0.,  2.]])
```

# 使用 grad 方法求导数

## 标量的反向传播

```python
import numpy as np
import torch

"""
求 f(x) = a*x**2 + b*x + c 关于 x 的在 a=1.0，b=-2.0，c=1.0 处的导数
"""

# 自变量(标量)
x = torch.tensor(0.0, requires_grad = True)  # x 需要被求导
# 参数
a = torch.tensor(1.0)
b = torch.tensor(-2.0)
c = torch.tensor(1.0)
# 因变量
y = a * torch.pow(x, 2) + b * x + c

# create_graph 设置为 True 将允许创建更高阶的导数
dy_dx = torch.autograd.grad(y, x, create_graph = True)[0]
print(dy_dx.data)

# 求二阶导数
dy2_dx2 = torch.autograd.grad(dy_dx, x)[0]
print(dy2_dx2.data)
```

```
tensor(-2.)
tensor(2.)
```

## 多个自变量求导

```python
import numpy as np
import torch

# 自变量
x1 = torch.tensor(1.0, requires_grad = True)  # x1 需要被求导
x2 = torch.tensor(2.0, requires_grad = True)  # x2 需要被求导
# 因变量
y1 = x1 * x2
y2 = x1 + x2

# 允许同时对多个自变量求导数
(dy1_dx1, dy1_dx2) = torch.autograd.grad(
    outputs = y1,
    inputs = [x1, x2], 
    retain_graph = True
)
print(dy1_dx1, dy1_dx2)

# 如果有多个因变量，相当于把多个因变量的梯度结果求和
(dy12_dx1, dy12_dx2) = torch.autograd.grad(
    outputs = [y1, y2], 
    inputs = [x1, x2],
)
print(dy12_dx1, dy12_dx2)
```

```
tensor(2.) tensor(1.)
tensor(3.) tensor(2.)
```

# 使用自动微分和优化器求最小值

```python
import numpy as np
import torch

"""
求 f(x) = a*x**2 + b*x + c 的最小值
"""

# 自变量(标量)
x = torch.tensor(0.0, requires_grad = True)  # x 需要被求导
# 参数
a = torch.tensor(1.0)
b = torch.tensor(-2.0)
c = torch.tensor(1.0)
# 因变量函数
def f(t):
    result = a * torch.pow(t, 2) + b * x + c
    return result

# 优化器
optimizer = torch.optim.SGD(params = [x], lr = 0.01)

for i in range(500):
    # forward
    y = f(x)
    # backward
    optimizer.zero_grad()
    y.backward()
    optimizer.step()

print(f"y = {f(x).data}; x = {x.data}")
```

```
y= tensor(0.) ; x= tensor(1.0000)
```

# 参考

* [A GENTLE INTRODUCTION TO TORCH.AUTOGRAD](https://pytorch.org/tutorials/beginner/blitz/autograd_tutorial.html)
