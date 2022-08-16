---
title: PyTorch 自动微分
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

- [计算图](#计算图)
- [Pytorch 动态图机制](#pytorch-动态图机制)
- [PyTorch 自动求导机制](#pytorch-自动求导机制)
</p></details><p></p>

- 计算图
   - 描述运算的有向无环图
      - Tensor 的 `is_leaf` 属性
      - Tensor 的 `grad_fn` 属性
- PyTorch 动态图机制
   - 动态图与静态图
- PyTorch 自动求导机制
   - `torch.autograd.backward()` 方法自动求取梯度
   - `torch.autograd.grad(  )` 方法可以高阶求导

**Note**

- 梯度不自动清零
- 依赖叶节点的节点, `requires_grad` 默认为 True
- 叶节点不能执行原位操作

## 计算图

计算图是用来描述运算的有向无环图。主要有两个因素:节点、边。
其中节点表示数据，如向量、矩阵、张量；而边表示运算，如加减乘除、卷积等。

使用计算图的好处不仅是让计算看起来更加简洁，还有个更大的优势是让梯度求导也变得更加方便。

- 示例:

```python
x = torch.tensor([2.], requires_grad = True)
w = torch.tensor([1.], requires_grad = True)

a = torch.add(w, x)
b = torch.add(w, 1)

y = torch.mul(a, b)

y.backward()
print(w.grad)
```

## Pytorch 动态图机制


## PyTorch 自动求导机制

- package `autograd`
- torch.Tensor
- .requires_grad = True
- .backward()
- .grad
- .detach()
- with torch.no_grad(): pass
- .grad_fn

PyTorch 自动求导机制使用的是 ``torch.autograd.backward`` 方法，功能就是自动求取梯度。

- API:

```python
torch.autograd.backward(
   tensors, 
   gard_tensors = None, 
   retain_graph = None, 
   create_graph = False
)
```

`autograd` 包提供了对所有 Tensor 的自动微分操作

```python
#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
`autograd` package:
-------------------
跟踪 torch.Tensor 上所有的操作:torch.Tensor(requires_grad = True)
自动计算所有的梯度:.backward()
torch.Tensor 上的梯度:.grad
torch.Tensor 是否被跟踪:.requires_grad
停止跟踪 torch.Tensor 上的跟踪历史、未来的跟踪:.detach()

with torch.no_grad():
      pass

Function
.grad_fn
"""

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

