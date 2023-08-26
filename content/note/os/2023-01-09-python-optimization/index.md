---
title: 数值优化求解器
author: 王哲峰
date: '2023-01-09'
slug: python-optimizaion
categories:
  - Python
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

- [最优化算法求解器](#最优化算法求解器)
- [Python Gurobi](#python-gurobi)
    - [Gurobi 简介](#gurobi-简介)
    - [Gurobi 数据结构](#gurobi-数据结构)
        - [Multidict](#multidict)
        - [Tuplelist](#tuplelist)
        - [Tupledict](#tupledict)
        - [示例](#示例)
    - [Gurobi 参数和属性](#gurobi-参数和属性)
        - [参数类型](#参数类型)
        - [属性类型](#属性类型)
    - [Gurobi 线性化技巧](#gurobi-线性化技巧)
        - [最大值](#最大值)
        - [最小值](#最小值)
        - [绝对值](#绝对值)
        - [逻辑与](#逻辑与)
        - [逻辑或](#逻辑或)
        - [指示函数](#指示函数)
        - [带固定成本约束](#带固定成本约束)
        - [分段线性函数](#分段线性函数)
    - [Gurobi 多目标优化](#gurobi-多目标优化)
    - [callback 函数](#callback-函数)
- [Python Ortools](#python-ortools)
- [Python Scipy](#python-scipy)
    - [数值优化](#数值优化)
    - [APIs 说明](#apis-说明)
    - [多元标量函数的无约束最小化(minimize)](#多元标量函数的无约束最小化minimize)
        - [示例](#示例-1)
    - [多元标量函数的约束最小化(minimize)](#多元标量函数的约束最小化minimize)
    - [全局最优](#全局最优)
    - [最小二乘最小化](#最小二乘最小化)
    - [单变量函数最小化器](#单变量函数最小化器)
    - [自定义最小化器](#自定义最小化器)
    - [寻根](#寻根)
    - [线性规划](#线性规划)
- [Python CPLEX](#python-cplex)
- [Python Pyomo](#python-pyomo)
    - [安装 pyomo 和 GLPK](#安装-pyomo-和-glpk)
    - [参考](#参考)
- [Python PuLP](#python-pulp)
- [Python Geatpy](#python-geatpy)
    - [Population 类是一个表示种群的类](#population-类是一个表示种群的类)
    - [Algorithm 类是进化算法的核心类](#algorithm-类是进化算法的核心类)
</p></details><p></p>

# 最优化算法求解器

无论是在生产制造领域，还是在金融、保险、交通等其他领域，当实际问题越来越复杂、
问题规模越来越庞大，就需要借助计算机的快速计算能力，求解器的作用就是能够简化编程问题，
求解器的作用就是能简化编程问题，使得工程师能专注于问题的分析和建模，而不是编程。

算法优化的求解器有很多，其中商用的求解器包括 Gurobi、Cplex、Xpress 等；
开源的求解器有 SCIP、GLPK、Ortools 等，这些求解器都有 Python 接口，
因此，能够用比较简单的方式对运筹优化问题进行建模。

* Gurobi 是由美国 Gurobi 公司开发的针对算法最优化领域的求解器，可以高效求解算法优化中的建模问题。
* Ortools 是 Google 开源维护的算法优化求解器，针对 Google 的商业场景进行优化，如 VRP 问题，
  对于中小规模的商业场景的使用是个不错的选择

# Python Gurobi

> gurobipy，Gurobi 的 Python API

## Gurobi 简介

运筹优化软件 Gurobi 虽然核心是使用 C/C++ 编写的，但也开发了 Python 接口，
使 Python 使用者能够以其熟悉的方式用 Gurobi 求解算法最优化问题。

安装 Gurobi：

Gurobi 的安装根据参考文档进行安装即可，在安装了 Gurobi 软件之后，
Gurobi 的 Python 扩展就可以直接到 Gurobi 的安装目录用 `python setup.py install` 命令进行安装

## Gurobi 数据结构

虽然用基础的 Python 数据结构也能实现 Gurobi 的建模，但在建模过程中，经常要对带不同下标的数据进行组合，
如果使用 Python 内置的数据结构，则效率会比较低，为了提高建模效率，Gurobi 封装了更高级的 Python 数据结构，
即 `Multidict`、`Tuplelist`、`Tupledict`。在对复杂或大规模问题建模时，它们可以大大提高模型求解的效率。

### Multidict

Multidict，即复合字典，就是多重字典的意思，`multidict` 函数允许在一个语句中初始化一个或多个字典

```python
import gurobipy as grb

student, chinese, math, english = grb.multidict({
    "student1": [1, 2, 3],
    "student2": [2, 3, 4],
    "student3": [3, 4, 5],
    "student4": [4, 5, 6],
})

# 字典的键
print(student)
# 语文成绩的字典
print(chinese)
# 数学成绩的字典
print(math)
# 英语成绩的字典
print(english)
```

```
['student1', 'student2', 'student3', 'student4']
{'student1': 1, 'student2': 2, 'student3': 3, 'student4': 4}
{'student1': 2, 'student2': 3, 'student3': 4, 'student4': 5}
{'student1': 3, 'student2': 4, 'student3': 5, 'student4': 6}
```

### Tuplelist

Tuplelist，即元组列表，就是 `tuple` 和 `list` 的组合，也就是 `list` 元素的 `tuple` 类型，
其设计的目的是为了高效地在元组列表中构建子列表

```python
import gurobipy as grb

t1 = grb.tuplelist([
    (1, 2),
    (1, 3),
    (2, 3),
    (2, 5),
])

# 输出第一个值是 1 的元素
print(tl.select(1, "*"))
# 输出第二个值是 3 的元素
print(tl.select("*", 3))
```

Tuplelist 继承自 list，所以向 `tuplelist` 中添加新元素和普通 `list` 添加元素一样，有 `append`、`pop` 等方法，
同样用迭代地方式遍历元素

```python
# 添加一个元素
t1.append((3, 5))
print(t1.select(3, "*"))

# 使用迭代地方式实现 select 功能
print(t1.select(1, "*"))
```

### Tupledict

Tupledict 是 Python 的 dict 的一个子类，通过 `tupledict` 可以更加高效地操作 Gurobi 中的变量子集，也就是说当定义了很多变量，
需要对一部分变量进行操作时，可以使用 tupledict 的内置方法来高效轻松地构建线性表达式，如 `sum` 和 `prod`。

`tupledict` 的键在内部存储格式是 `tuplelist`，因此可以使用 `tuplelist` 的 `select` 方法选择集合的子集。在实际使用中，
通过将元组与每个 Gurobi 变量关联起来，可以有效地创建包含匹配变量子集的表达式。

下面创建一个 `$3 \times 3$` 的矩阵，里面的每个元素表示线性表达式的变量，取其中一部分变量的操作就显得很方便了。

`$$\begin{bmatrix}
x_{11} & x_{12} & x_{13} \\
x_{21} & x_{22} & x_{23} \\
x_{31} & x_{32} & x_{33}
\end{bmatrix}$$`

```python
import gurobipy as grb

model = grb.Model()

# 定义变量的下标
t1 = [
    (1, 1), (1, 2), (1, 3),
    (2, 1), (2, 2), (2, 3),
    (3, 1), (3, 2), (3, 3),
]
vars = model.addVars(t1, name = "d")
```

对变量求和：

```python
# 基于元素下标的操作，对第一行求和
print(sum(vars.select(1, "*")))
print(vars.sum(1, "*"))
```

如果变量系数不是 1，就不能用 `sum` 方法，而需要用 `prod` 方法来构建线性表达式，
`prod` 方法用于变量和系数相乘后的累加。首先创建一个系数矩阵，用 `tupledict` 存储，键与 `vars` 是一样的，
这样就可以快速匹配系数和对应的变量，然后采用 `prod` 方法来选定的变量和系数来构建线性表达式。

```python
import gurobipy as grb

# 创建一个系数矩阵，用 tupledict 格式存储
c1 = [
    (1, 1),
    (1, 2),
    (1, 3),
]
coeff = grb.tupledict(c1)

coeff[(1, 1)] = 1
coeff[(1, 2)] = 0.3
coeff[(1, 3)] = 0.4

print(vars.prod(coeff, 1, "*"))
```

如果不是选择部分变量而是选择全部变量，`prod` 函数实现的功能就是具有相同下标的变量相乘后加和。

```python
obj = grb.quicksum(cost[i, j] * x[i, j] for i , j in arcs)
obj = x.prod(cost)
```

由于 `tupledict` 是 `dict`a 的子类，因此可以使用标准的 `dict` 方法来修改 `tupledict`。
Gurobi 变量一般都是 `tupledict` 类型，用 `tupledict` 定义变量的好处是可以快速选择部分变量，
创建各种各样的约束，因为 `tupledict` 有 `sum` 函数和 `select` 函数。

```python
import gurobipy as grb

# tupledict 类型的变量快速创建约束条件
m = grb.Model()
x = m.addVars(3, 4, vtype = grb.GRB.BINARY, name = "x")
m.addConstrs((x.sum(i, "*") <= 1 for i in range(3)), name = "con")
m.update()
m.write("tupledict_vars.lp")
```

### 示例

## Gurobi 参数和属性

Gurobi 的默认参数并不能高效求解模型，因此有必要了解 Gurobi 的重要参数和属性，以便在建模过程中能写出更高效的代码，
能更快地进行试验和求解模型。

这里将讲解 Gurobi 的参数（Parameterss）和属性（Attributes）。通过参数来控制优化器的行为，
如求解时间限制、在命令行窗口中输出日志、MIP 可行解数量等，因此需要在优化求解启动前设置。
通过属性来控制模型（变量、约束、目标等对象）的特征，如模型优化方向、变量的上界和下界等。

### 参数类型

参数控制 Gurobi 优化求解器的行为，需要在启动之前设置，一般来说 Gurobi 已经选择了最优的参数作为默认值，
因此除非有必要，否则不用修改这些默认参数。

前 8 种类型参数使用比较多，后 3 种类型参数在部署 Gurobi 服务器或部署服务计算时使用，
这些参数的详细说明可以参考 Gurobi 接口文档的相关内容。

1. Termination 停止参数，用于控制求解的停止条件。
    - TimeLimit 设定整个求解过程耗时限制；
    - SolutionLimit 设定 MIP 可行解数量；
    - BarIterLimit 设定障碍法（Barrier）迭代次数限制；
    - IterationLimit 设定单纯形法迭代次数限制；
2. Tolerances 容差参数，用于控制结果的精度，在大多数情况下，这个限制是通过数值公差来管理的；
   如果冲突小于相应的公差，求解器将结果视为满足约束。
3. Simplex 单纯形参数，用于控制单纯形法的应用。如 InfUnbdInfo 控制是否生成不可行或无界模型附加信息。
4. Barrier 障碍法参数，用于控制障碍法的操作，障碍法也称罚函数法。如 QCPDual 控制是否获取二次模型的对偶值。
5. MIP 混合整数规划参数，用于控制混合整数规划算法。如 BranchDir 用于设定分支割平面搜索方向，默认值是自动选择的
6. MIP Cuts 割平面参数，用于控制割平面的形式。如 Cuts 用于控制全局割平面法的强度。
7. Tuning 调参参数，用于控制求解器的调参行为。如 `TuneCriterion` 可设定调参的准则，`TuneTimeLimit` 可设定调参的事件。
8. Multiple Solutions 多解参数，用于修改 MIP 的搜索行为，用于尝试为 MIP 模型寻找多个解。如 `PoolSolutions` 决定存储可行解的数量。
9. Distributed Algorithms 分布式计算参数，用于控制分布式并行计算（分布式 MIP、分布式并发和分布式调优）参数。
10. Compute Server 计算服务器参数，用于配置和启动 Gurobi 计算服务器作业的参数。
11. Cloud 云计算参数，用于启动 Gurobi 即时云实例的参数。
12. Token Server 令牌服务参数，用于通信加密等功能。
13. 其他参数，是上述 12 种参数之外的参数，部分参数和上述参数有关联但是又不完全符合该分类，故独立出来。
    比如 `LogFile` 参数用于指定将模型求解信息保存到日志文件 `LogFile`

### 属性类型

通过属性（Attributes）能够控制模型（变量、约束、目标等对象）的特征，Gurobi 中的属性共分为 8 种类型，分别是：

> * 模型属性
> * 变量属性
> * 线性约束属性
> * SOS 约束属性
> * 二次约束属性
> * 广义约束属性
> * 解的质量属性
> * 多目标属性

1. 模型属性（Model Attributes）包括 `ModelSense` 模型优化方向（最大化或最小化）、`ObjVal` 当前的目标值。
2. 变量属性（Variable Attributes），如 `X` 获取当前变量的取值，`Start` 属性用于设置 MIP 模型的初始解。
3. 线性约束属性（Linear Constraint Attributes），这些属性提供与特定线性约束相关的信息，
   如 `Pi` 约束对应的对偶值，`Slack` 约束对应的松弛量，`RHS` 约束对应的右端项。
4. SOS 约束属性（Special-Ordered Set Constraints Attributes），这些属性提供与特定的顺序集（SOS）约束相关的信息，
   如 `IISSOS` 对不可行的模型，指示约束是否属于 IIS （Irreducible Inconsistent Subsystem）。
5. 二次约束属性（Quadratic Constraint Attributes），这些属性提供与特定二次约束相关的信息，如 QCRHS 约束右端项。
6. 广义约束属性（General Constraint Attributes），这些属性提供与特定常规约束关联的信息，如 GenConstrName 约束名称。
7. 解质量属性（Solution Quality Attributes），用于评价解质量的相关属性，
   如 BoundVio 最大的界违反，IntVio 整数变量离最近整数的最大距离。
8. 多目标属性（Multi-objective Attributes），用于多目标优化问题的相关属性，
   如 `ObjN` 对应多目标表达式中的变量系数，ObjNVal 对应目标函数值。

查看和修改 Gurobi 参数属性的方法很简单，用于查看属性的函数是 `getAttr(attrname, objs)`，
用于修改属性的函数是 `setAttr(attrname, newvalue)`。注意：并不是所有属性都能进行修改，对于只读属性就只能查看而不能修改。

1. 查看属性
    - 方法：`getAttr(attrname, objs)`，其中 `attrname` 是属性名称，`objs`（可选）是列表或字典对象用来存储查询的值。  
    - 例如：`model.getAttr(GRB.Attr.ObjVal)` 或简写为 `model.ObjVal`。
2. 修改属性
    - 方法：`setAttr(attrname, newvalue)`，其中 `attrname` 是属性名称，`newvalue` 是属性的值。
    - 例如：`var.setAtrr(GRB.Attr.VType, "C")` 或简写为 `var.Vtype = "C"`   

## Gurobi 线性化技巧

### 最大值


### 最小值


### 绝对值


### 逻辑与

### 逻辑或

### 指示函数

### 带固定成本约束


### 分段线性函数





## Gurobi 多目标优化


## callback 函数



# Python Ortools


安装 Ortools:

```bash
$ pip install ortools
```

# Python Scipy

## 数值优化

scipy.optimize 提供了多种常用的优化算法. 

1. 无约束和约束多元标量函数
   - scipy.optimize.minimize
   - 算法
      - BFGS
      - Nelder-Mead单纯形法
      - 牛顿共轭梯度
      - COBYLA
      - SLSQP
2. 全局优化
   - scipy.optimize.basinhopping
   - scipy.optimize.differential_evolution
   - scipy.optimize.shgo
   - scipy.optimize.dual_annealing
3. 最小二乘最小化和曲线拟合
   - scipy.optimize.least_sequares
   - scipy.optimize.curve_fit
4. 单变量函数最小化和根查找器
   - scipy.optimize.minimize_scalar
   - scipy.optimize.root_scalar
5. 多元方程组求解
   - scipy.optimize.root
   - 算法
      - 混合鲍威尔
      - 莱文贝格-马夸特

## APIs 说明

```python
scipy.optimize.minimize(
    fun, 
    x0, 
    args = (), 
    method = 'SLSQP', 
    jac = None, 
    bounds = None, 
    constraints = (), 
    tol = None, 
    callback = None, 
    options = {
        'func': None, 
        'maxiter': 100, 
        'ftol': 1e-06, 
        'iprint': 1, 
        'disp': False, 
        'eps': 1.4901161193847656e-08
    }
)
```

```python
scipy.optimize.basinhopping(
    func, 
    x0, 
    niter = 100, 
    T = 1.0, 
    stepsize = 0.5, 
    minimizer_kwargs = None, 
    take_step = None, 
    accept_test = None, 
    callback = None, 
    interval = 50, 
    disp = False, 
    niter_success = None, 
    seed = None
)
```

- scipy.optimize.minimize
   - Minimization of scalar function of one or more variables.
   - `method`
      - Unconstrained minimization
         - 'Nelder-Mead'
         - 'Powell'
         - 'CG'
         - 'BFGS'
         - 'Newton-CG'
         - 'dogleg'
         - 'trust-ncg'
         - 'trust-krylov'
         - 'trust-exact'
      - Constrained Minimization
         - 'COBYLA'
         - 'SLSQP'
         - 'trust-constr'
      - Bound-Constrained minimization
         - 'L-BFGS-B'
         - 'TNC'
      - Finite-Difference Options
         - 'trust-constr'
      - Custom minimizers
         - custom - a callable object

## 多元标量函数的无约束最小化(minimize)

- Nelder-Mead 单纯形算法
   - `method = 'Nelder-Mead'`
- Broyden-Fletcher-Goldfarb-Shanno 算法
   - `method = 'BFGS'`
- 牛顿共轭梯度算法
   - `method = 'Newton-CG'`
- 信赖域牛顿共轭梯度算法
   - `method = 'trust-ncg'`
- 信任区域截断的广义Lanczos /共轭梯度算法
   - `method = 'trust-krylov'`
- 信任区域几乎精确的算法
   - `method = 'trust-exact'`


### 示例

官方示例: 

```python
from scipy.optimize import minimize, rosen, rosen_der

# -----------------------
# 
# -----------------------
x0 = [1.3, 0.7, 0.8, 1.9, 1.2]
res = minimize(rosen, x0, method = "Nelder-Mead", tol = 1e-6)
print(res.x)


# -----------------------
# 
# -----------------------
x0 = [1.3, 0.7, 0.8, 1.9, 1.2]
res = minimize(rosen, x0, method = "BFGS", jac = rosen_der, options = {"gtol": 1e-6, "disp": True})
print(res.x)
print(res.message)
print(res.hess_inv)


# -----------------------
# 
# -----------------------
fun = lambda x: (x[0] - 1) ** 2 + (x[1] - 2.5) ** 2
cons = ({"type": "ineq", "fun": lambda x: x[0] - 2 * x[1] + 2},
        {"type": "ineq", "fun": lambda x: -x[0] - 2 * x[1] + 6},
        {"type": "ineq", "fun": lambda x: -x[0] + 2 * x[1] + 2})
bnds = ((0, None), (0, None))
res = minimize(fun, (2, 0), method = "SLSQP", bounds = bnds, constraints = cons)
```

示例2: 

```python
# -*- coding: utf-8 -*-

import logging
from scipy.optimize import minimize
import numpy as np


def fun(args):
    """
    待优化函数: [1 / x + x]
    """
    a = args
    v = lambda x: a / x[0] + x[0]

    return v


def con(args):
    """
    约束条件: 
        None
    """
    pass


def optimizer():
    args_fun = (1)
    args_con = None
    x0 = np.asarray((2))
    res = minimize(fun = fun(args_fun), x0 = x0, method = "SLSQP")

    return res


def main():
    result = optimizer()
    print("优化得到的目标函数最小值: ", result.fun)
    print("优化状态: ", result.success)
    print("优化路径: ", result.x)

if __name__ == "__main__":
    main()
```

示例3: 

```python
# -*- coding: utf-8 -*-

from scipy.optimize import minimize
import numpy as np

"""
目标函数:  min[(2+x1)/(1+x2) -3 * x1 + 4 * x3]
约束条件:  x1, x2, x3 的范围都在 [0.1, 0.9] 范围内
"""

def fun(args):
    """
    待优化函数: [(2+x1)/(1+x2) -3 * x1 + 4 * x3]
    """
    a, b, c, d = args
    v = lambda x: (a + x[0]) / (b + x[1]) - c * x[0] + d * x[2]

    return v


def con(args):
    """
    约束条件: 
        x1 - x1_min >= 0
        x1_max - x1 >= 0
        x2 - x2_min >= 0
        x2_max - x2 >= 0
        x3 - x3_min >= 0
        x3_max - x3 >= 0
    """
    x1_min, x1_max, x2_min, x2_max, x3_min, x3_max = args
    cons = (
        {
            "type": "ineq", 
            "fun": lambda x: x[0] - x1_min 
        },
        {
            "type": "ineq",
            "fun": lambda x: -x[0] + x1_max
        },
        {
            "type": "ineq",
            "fun": lambda x: x[1] - x2_min
        },
        {
            "type": "ineq",
            "fun": lambda x: -x[1] + x2_max
        },
        {
            "type": "ineq",
            "fun": lambda x: x[2] - x3_min
        },
        {
            "type": "ineq",
            "fun": lambda x: -x[2] + x3_max
        }
    )
    return cons


def optimizer():
    """
    目标函数优化器
    """
    # 目标函数系数
    args_fun = (2, 1, 3, 4)
    # 约束条件参数范围
    args_con = (
        0.1, 0.9,
        0.1, 0.9,
        0.1, 0.9
    )
    # 构造约束条件
    cons = con(args_con)
    # 设置优化变量初始猜测值
    x0 = np.asarray((0.5, 0.5, 0.5))
    # 目标函数优化
    res = minimize(fun(args_fun), 
                    x0, 
                    method = "SLSQP", 
                    constraints = cons)
    return res


def main():
    result = optimizer()
    print("优化得到的目标函数最小值: ", result.fun)
    print("优化状态: ", result.success)
    print("优化路径: ", result.x)

if __name__ == "__main__":
    main()
```

## 多元标量函数的约束最小化(minimize)

- 信任区域约束算法
   - `method = "trust-constr"`
   - 定义边界约束
   - 定义线性约束
   - 定义非线性约束
   - 解决优化问题
- 顺序最小二乘法(SLSQP)算法
   - `method = "SLSQP"`

## 全局最优

## 最小二乘最小化

## 单变量函数最小化器

- 无约束最小化
   - `method = "brent"`
- 有界最小化
   - `method = "bounded"`

## 自定义最小化器

## 寻根

## 线性规划

```python
# -*- coding: utf-8 -*-

from scipy import optimize as op
import numpy as np

'''
线性规划demo

求解 max z = 2x1 + 3x2 - 5x3
s.t. x1 + x2 + x3 = 7
      2x1 - 5x2 + x3 >= 10
      x1 + 3x2 + x3 <= 12
      x1, x2, x3 >= 0

scipy.optimize.linprog(c, A_ub=None, b_ub=None, A_eq=None, b_eq=None, bounds=None, method='simplex', callback=None, options=None)
- c 函数系数数组, 最大化参数为c, 最小化为-c, 函数默认计算最小化. 
- A_ub 不等式未知量的系数, 默认转成 <= , 如果原式是 >= 系数乘负号. 
- B_ub 对应A_ub不等式的右边结果
- A_eq 等式的未知量的系数
- B_eq 等式的右边结果
- bounds 每个未知量的范围
'''

c = np.array([2, 3, -5])
A_ub = np.array([[-2, 5, -1], [1, 3, 1]])
B_ub = np.array([-10, 12])
A_eq = np.array([[1, 1, 1]])
B_eq = np.array([7])
x1 = (0, 7)
x2 = (0, 7)
x3 = (0, 7)
res = op.linprog(-c, A_ub, B_ub, A_eq, B_eq, bounds = (x1, x2, x3))

print(res)
```

# Python CPLEX 

> docplex，用于 Python 的 IBM Decision Optimization CPLEX 建模包

* [docplex Doc](http://ibmdecisionoptimization.github.io/docplex-doc/)
* [docplex Examples](https://github.com/IBMDecisionOptimization/docplex-examples)

# Python Pyomo

Pyomo 是一个基于 Python 的开源软件包，它支持多种优化功能，用于制定和分析优化模型。
Pyomo 可用于定义符号问题、创建具体的问题实例，并使用标准解决程序解决这些实例。Pyomo 支持多种问题类型，包括:

* 线性规划
* 二次规划
* 非线性规划
* 整数线性规划
* 混合整数二次规划
* 混合整数非线性规划
* 整数随机规划
* 广义分隔编程
* 微分代数方程
* 具有平衡约束的数学规划

Pyomo 支持全功能编程语言中的分析和脚本编制。此外，Pyomo 还证明了开发高级优化和分析工具的有效框架。
例如，PySP 包提供了随机规划的通用求解程序。PySP 利用了 Pyomo 的建模对象嵌入在功能全面的高级编程语言中的事实，
这种语言允许使用 Python 并行通信库透明地并行化子问题

## 安装 pyomo 和 GLPK

pyomo：

```bash
$ pip install -q pyomo
```

GLPK 是一个开源的 GNU 线性编程工具包，可在 GNU 通用公共许可证 3 下使用。GLPK 是一个单线程单形解算器，
通常适用于中小型线性整数规划问题。它是用 C 语言编写的，依赖性很小，因此在计算机和操作系统之间具有很高的可移植性。
对于许多示例来说，GLPK 通常“足够好”。对于较大的问题，用户应该考虑高性能的解决方案，如 COIN-OR CBC，它们可以利用多线程处理器

```bash
$ apt-get install -y -qq glpk-utils
```

## 参考

* [Pyomo Tutorial](https://www.osti.gov/servlets/purl/1376827)

# Python PuLP

> pulp，用 Python 编写的 LP/MILP 建模

# Python Geatpy

Geatpy2 整体上看由工具箱内核函数（内核层）和面向对象进化算法框架（框架层）两部分组成。
其中面向对象进化算法框架主要有四个大类：

* Problem 问题类
* Algorithm 算法模板类
* Population 种群类
* PsyPopulation 多染色体种群类

![img](images/geatpy.png)

## Population 类是一个表示种群的类

一个种群包含很多个个体，而每个个体都有一条染色体(若要用多染色体，则使用多个种群、并把每个种群对应个体关联起来即可)。
除了染色体外，每个个体都有一个译码矩阵 Field(或俗称区域描述器)来标识染色体应该如何解码得到表现型，
同时也有其对应的目标函数值以及适应度。种群类就是一个把所有个体的这些数据统一存储起来的一个类。比如：

* Chrom 是一个存储种群所有个体染色体的矩阵，它的每一行对应一个个体的染色体
* ObjV 是一个目标函数值矩阵，每一行对应一个个体的所有目标函数值，每一列对应一个目标

PsyPopulation 类是继承了 Population 的支持多染色体混合编码的种群类。一个种群包含很多个个体，而每个个体都有多条染色体

* Chroms 列表存储所有的染色体矩阵(Chrom)
* Encodings 列表存储各染色体对应的编码方式(Encoding)
* Fields 列表存储各染色体对应的译码矩阵(Field)

## Algorithm 类是进化算法的核心类

Algorithm 类既存储着跟进化算法相关的一些参数，同时也在其继承类中实现具体的进化算法。
比如 Geatpy 中的 `moea_NSGA3_templet.py` 是实现了多目标优化 NSGA-III 算法的进化算法模板类，
它是继承了 Algorithm 类的具体算法的模板类

关于 Algorithm 类中各属性的含义可以查看 `Algorithm.py` 源码。这些算法模板通过调用 Geatpy 工具箱提供的进化算法库函数实现对种群的进化操作，
同时记录进化过程中的相关信息，其基本层次结构如下图：

![img](images/algo.png)
