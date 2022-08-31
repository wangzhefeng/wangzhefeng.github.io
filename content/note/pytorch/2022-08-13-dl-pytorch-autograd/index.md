---
title: PyTorch 自动微分机制与动态计算图机制
author: 王哲峰
date: '2022-08-13'
slug: dl-pytorch-autograd
categories:
  - deeplearning
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

- [内容](#内容)
- [PyTorch 自动微分机制](#pytorch-自动微分机制)
  - [使用 backward() 方法求导数](#使用-backward-方法求导数)
    - [标量的反向传播](#标量的反向传播)
    - [非标量的反向传播](#非标量的反向传播)
    - [非标量的反向传播可以用标量的反向传播实现](#非标量的反向传播可以用标量的反向传播实现)
  - [使用 grad() 方法求导数](#使用-grad-方法求导数)
    - [标量的反向传播](#标量的反向传播-1)
    - [多个自变量求导](#多个自变量求导)
  - [使用自动微分和优化器求最小值](#使用自动微分和优化器求最小值)
  - [autograd 包功能](#autograd-包功能)
- [PyTorch 动态计算图机制](#pytorch-动态计算图机制)
  - [计算图](#计算图)
  - [PyTorch 动态计算图机制](#pytorch-动态计算图机制-1)
    - [计算图的正向传播时立即执行的](#计算图的正向传播时立即执行的)
    - [计算图在反向传播后立即销毁](#计算图在反向传播后立即销毁)
  - [PyTorch 计算图中的 Function](#pytorch-计算图中的-function)
  - [PyTorch 计算图与反向传播](#pytorch-计算图与反向传播)
  - [PyTorch 计算图在 TensorBoard 中的可视化](#pytorch-计算图在-tensorboard-中的可视化)
</p></details><p></p>


# 内容

- 计算图
   - 描述运算的有向无环图
      - Tensor 的 `is_leaf` 属性
      - Tensor 的 `grad_fn` 属性
- PyTorch 动态图机制
   - 动态图与静态图
- PyTorch 自动微分机制
   - `torch.autograd.backward()` 方法自动求取梯度
   - `torch.autograd.grad(  )` 方法可以高阶求导

**Note**

- 梯度不自动清零
- 依赖叶节点的节点, `requires_grad` 默认为 True
- 叶节点不能执行原位操作


# PyTorch 自动微分机制

神经网络通常依赖反向传播算法求梯度来更新网络参数，
而求梯度过程通常是一件非常复杂繁琐而且容易出错的事情。
深度学习框架可以帮助自动地完成这种求梯度的运算

PyTorch 一般通过反向传播 `torch.autograd.backward()` 方法实现求梯度计算，
该方法求得的梯度将保存在对应自变量张量的 `grad` 属性下。
除此之外，也能够调用 `torch.autograd.grad()` 函数实现求梯度计算。
这就是 PyTorch 的自动微分机制

## 使用 backward() 方法求导数

PyTorch 自动微分机制使用的是 `torch.autograd.backward()` 方法，
功能就是自动求取梯度

* `torch.autograd.backward()` 方法通常在一个标量张量上调用，
  该方法求得的梯度将保存在对应自变量张量的 `grad` 属性下
* 如果调用的张量非标量，则要传入一个和它形状相同的 `gradient` 参数张量，
  相当于用该 `gradient` 参数张量与调用张量作向量点乘，得到的标量结果再反向传播

下面是 `torch.autograd.backward()` API:

```python
torch.autograd.backward(
   tensors, 
   gard_tensors = None, 
   retain_graph = None, 
   create_graph = False
)
```

### 标量的反向传播

```python
import numpy as np
import torch

"""
# 目标:
# 求 f(x) = a*x**2 + b*x + c 关于 x 的在 a=1.0,b=-2.0,c=1.0 处的导数
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

### 非标量的反向传播

```python
import numpy as np
import torch

"""
# 目标:
# 求 f(x) = a*x**2 + b*x + c 关于 x 的在 a=1.0,b=-2.0,c=1.0 处的导数
"""

# 自变量(张量)
x = torch.tensor([
    [0.0, 0.0],
    [1.0, 2.0],
], requires_grad = True)  # x 需要被求导
# 参数
a = torch.tensor(1.0)
b = torch.tensor(-2.0)
c = torch.tensor(1.0)
# 因变量
y = a * torch.pow(x, 2) + b * x + c

# gradient = torch.tensor([[1.0, 1.0], [1.0, 1.0]])
gradient = torch.ones_like(x)

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

### 非标量的反向传播可以用标量的反向传播实现

```python
import numpy as np
import torch

"""
# 目标:
# 求 f(x) = a*x**2 + b*x + c 关于 x 的在 a=1.0,b=-2.0,c=1.0 处的导数
"""

# 自变量(张量)
x = torch.tensor([
    [0.0, 0.0],
    [1.0, 2.0],
], requires_grad = True)  # x 需要被求导
# 参数
a = torch.tensor(1.0)
b = torch.tensor(-2.0)
c = torch.tensor(1.0)
# 因变量
y = a * torch.pow(x, 2) + b * x + c

gradient = torch.tensor([[1.0, 1.0], [1.0, 1.0]])
z = torch.sum(y * gradient)

print(f"x:\n{x}")
print(f"y:\n{y}")
z.backward(gradient = gradient)
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

## 使用 grad() 方法求导数

### 标量的反向传播

```python
import numpy as np
import torch

"""
# 目标:
# 求 f(x) = a*x**2 + b*x + c 关于 x 的在 a=1.0,b=-2.0,c=1.0 处的导数
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

### 多个自变量求导

```python
import numpy as np
import torch

x1 = torch.tensor(1.0, requires_grad = True)  # x1 需要被求导
x2 = torch.tensor(2.0, requires_grad = True)  # x2 需要被求导
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

## 使用自动微分和优化器求最小值

```python
import numpy as np
import torch

"""
# 目标:
# 求 f(x) = a*x**2 + b*x + c 的最小值
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
    optimizer.zero_grad()
    y = f(x)
    y.backward()
    optimizer.step()

print(f"y = {f(x).data}; x = {x.data}")
```

```
y= tensor(0.) ; x= tensor(1.0000)
```

## autograd 包功能

`torch.autograd` 包提供了对所有 Tensor 的自动微分操作:

- `torch.Tensor(requires_grad = True)`
    - 跟踪 `torch.Tensor` 上所有的操作
- `requires_grad` 属性
    - `torch.Tensor` 是否被跟踪
- `torch.autograd.backward()` 方法
    - 自动计算所有的梯度
- `grad` 属性
    - `torch.Tensor` 上的梯度
- `torch.autograd.grad()` 方法
    - 自动计算所有的梯度
- `.detach()`
    - 停止跟踪 `torch.Tensor` 上的跟踪历史、未来的跟踪
- `with torch.no_grad(): pass`
- `.grad_fn`
- `.zero_grad()`

```python
import torch

# --------------------
# 创建 Tensor 时设置 requires_grad 跟踪前向计算
# --------------------
x = torch.ones(2, 2, requires_grad = True)
print("x:", x)

y = x + 2
print("y:", y)
print("y.grad_fn:", y.grad_fn)

z = y * y * 3
print("z:", z)
print("z.grad_fn", z.grad_fn)

out = z.mean()
print("out:", out)
print("out.grad_fn:", out.grad_fn)
out.backward()
print("x.grad:", x.grad)

# --------------------
# .requires_grad_() 能够改变一个已经存在的 Tensor 的 `requires_grad`
# --------------------
a = torch.randn(2, 2)
a = ((a * 3) / (a - 1))
print("a.requires_grad", a.requires_grad)
a.requires_grad_(True)
print("a.requires_grad:", a.requires_grad)
b = (a * a).sum()
print("b.grad_fn", b.grad_fn)



# 梯度
x = torch.randn(3, requires_grad = True)
y = x * 2
while y.data.norm() < 1000:
      y = y * 2
v = torch.tensor([0.1, 1.0, 0.0001], dtype = torch.float)
y.backward(v)
print(x.grad)

# .requires_grad
print(x.requires_grad)
print((x ** 2).requires_grad)

with torch.no_grad():
      print((x ** 2).requires_grad)

# .detach()
print(x.requires_grad)
y = x.detach()
print(y.requires_grad)
print(x.eq(y).all())
```

# PyTorch 动态计算图机制

## 计算图

计算图是用来描述运算的有向无环图。主要有两个因素:节点、边。
其中节点表示数据，如向量、矩阵、张量；而边表示运算，如加减乘除、卷积等。

使用计算图的好处不仅是让计算看起来更加简洁，
还有个更大的优势是让梯度求导也变得更加方便

示例:

```python
x = torch.tensor([2.], requires_grad = True)
w = torch.tensor([1.], requires_grad = True)

a = torch.add(w, x)
b = torch.add(w, 1)

y = torch.mul(a, b)

y.backward()
print(w.grad)
```

## PyTorch 动态计算图机制

![img](./images/torch动态图.gif)

PyTorch 的计算图由节点和边组成，节点表示张量或者 Function，边表示张量和 Function 之间的依赖关系。
PyTorch 中的计算图是动态度，这里的动态主要有两重含义:

* 计算图的正向传播时立即执行的
    - 无需等待完成的计算图创建完毕，每条语句都会在计算图中动态添加节点和边，并立即执行正向传播得到计算结果
* 计算图在反向传播后立即销毁，下次调用需要重新构建计算图
    - 如果在程序中使用了 `backward` 方法执行了反向传播，或者利用了 `torch.autograd.grad()` 方法计算了梯度，
      那么创建的计算图会被立即销毁，释放了存储空间，下次调用需要重新创建

### 计算图的正向传播时立即执行的

```python
import torch

w = torch.tensor([[3.0, 1.0]], requires_grad = True)
b = torch.tensor([[3.0]], requires_grad = True)
X = torch.randn(10, 2)
Y = torch.randn(10, 1)
# Y_hat 定义后其正向传播被立即执行，与其后面的 loss 创建语句无关
Y_hat = X @ w.t() + b
loss = torch.mean(torch.pow(Y_hat - Y, 2))

print(loss.data)
print(Y_hat.data)
```

```
tensor(17.8969)
tensor([[3.2613],
        [4.7322],
        [4.5037],
        [7.5899],
        [7.0973],
        [1.3287],
        [6.1473],
        [1.3492],
        [1.3911],
        [1.2150]])
```

### 计算图在反向传播后立即销毁

```python
import torch

w = torch.tensor([[3.0, 1.0]], requires_grad = True)
b = torch.tensor([[3.0]], requires_grad = True)
X = torch.randn(10, 2)
Y = torch.randn(10, 1)
# Y_hat 定义后其正向传播被立即执行，与其后面的 loss 创建语句无关
Y_hat = X @ w.t() + b
loss = torch.mean(torch.pow(Y_hat - Y, 2))

# 计算图在反向传播后立即销毁
loss.backward()  # 如果再次执行反向传播(loss.backward())将报错

# 如果需要保留计算图，需要设置 retain_graph = True
# loss.backward(retain_graph = True)
```

## PyTorch 计算图中的 Function


## PyTorch 计算图与反向传播


## PyTorch 计算图在 TensorBoard 中的可视化

