---
title: 整数规划
author: 王哲峰
date: '2023-08-20'
slug: integer-programming
categories:
  - optimizer algorithm
tags:
  - model
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

- [整数规划](#整数规划)
- [整数规划的求解方式](#整数规划的求解方式)
- [分支定界法](#分支定界法)
- [割平面法](#割平面法)
- [Gurobi 求解整数规划](#gurobi-求解整数规划)
- [总结](#总结)
</p></details><p></p>


# 整数规划

通常，决策变量的取值是大于或等于 0 的自然数，然而在许多实际问题中，
都要求决策变量的取值为正整数，如机器台数、商品数量、工人数量、装载货物的汽车数量等，
这类要求变量为整数的问题称为 <span style='border-bottom:1.5px dashed red;'>整数规划(Integer Progamming, IP)</span> 问题。

* 如果只要求一部分决策变量取整数，则称为 <span style='border-bottom:1.5px dashed red;'>混合整数规划(Mix Integer Programming, MIP)</span>；
* 如果决策变量的取值只能是 0 或 1，则成为 <span style='border-bottom:1.5px dashed red;'>0-1 整数规划(Binary Integer Programming, BIP)</span>；
* 如果模型是线性模型，则成为 <span style='border-bottom:1.5px dashed red;'>整数线性规划(Integer Linear Programming, ILP)</span>。

一个整数规划模型：

`$$max Z = 3x_{1} + 2x_{2}$$`

`$$s.t.\begin{cases}
2x_{1} + 3x_{2} \leq 14 \\
4x_{1} + 2x_{2} \leq 18 \\
x_{1},x_{2} \geq \text{且为整数}
\end{cases}$$`


# 整数规划的求解方式

求解整数规划的常用方法有 <span style='border-bottom:1.5px dashed red;'>分支定界法</span> 和 <span style='border-bottom:1.5px dashed red;'>割平面法</span>，
这两种方法的共同特点是：在线性规划的基础上，通过增加附加约束条件，
使整数最优解成为线性规划的一个极点（可行域的一个顶点），
于是就可以用单纯形法等方法找到整个最优解，
它们的区别在于约束条件的选取规划和方式不同。



# 分支定界法



# 割平面法



# Gurobi 求解整数规划

使用 Gurobi 求解以下整数规划模型：

`$$max Z = 3x_{1} + 2x_{2}$$`

`$$s.t.\begin{cases}
2x_{1} + 3x_{2} \leq 14 \\
4x_{1} + 2x_{2} \leq 18 \\
x_{1},x_{2} \geq \text{且为整数}
\end{cases}$$`

```python
import gurobipy as grb

model = grb.Model()

# 定义整数变量
x1 = model.addVar(vtype = grb.GRB.INTEGER, name = "x1")
x2 = model.addVar(vtype = grb.GRB.INTEGER, name = "x2")

# 添加约束
model.addConstr(2 * x1 + 3 * x2 <= 14)
model.addConstr(4 * x1 + 2 * x2 <= 18)
model.addConstr(x1 >= 0)
model.addConstr(x2 >= 0)

# 定义目标函数
model.setObjective(3 * x1 + 2 * x2, sense = grb.GRB.MAXIMIZE)

# 模型求解
model.optimize()
print(f"目标函数值：{model.objVal}")
for v in model.getVars():
    print(f"参数 {v.varName} = {v.x}")
```

```
# 目标函数值：14.0
# 参数 x1 = 4.0
# 参数 x2 = 1.0
```

上面的代码很简单，基本与线性规划的代码一样，不同之处在于，
定义变量时线性规划不限定变量的类型，而整数规划中限定变量类型为整数。


# 总结
