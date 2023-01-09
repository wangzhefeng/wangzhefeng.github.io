---
title: Python 数值优化
author: 王哲峰
date: '2023-01-09'
slug: optimize
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
</style>

<details><summary>目录</summary><p>

- [数值优化](#数值优化)
- [APIs 说明](#apis-说明)
- [多元标量函数的无约束最小化(minimize)](#多元标量函数的无约束最小化minimize)
  - [示例](#示例)
- [多元标量函数的约束最小化(minimize)](#多元标量函数的约束最小化minimize)
- [全局最优](#全局最优)
- [最小二乘最小化](#最小二乘最小化)
- [单变量函数最小化器](#单变量函数最小化器)
- [自定义最小化器](#自定义最小化器)
- [寻根](#寻根)
- [线性规划](#线性规划)
</p></details><p></p>

# 数值优化

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

# APIs 说明

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

```python
scipy.optimize.minimize(fun, 
                        x0, 
                        args = (), 
                        method = None, 
                        jac = None, 
                        hess = None, 
                        hessp = None, 
                        bounds = None, 
                        constraints = (), 
                        tol = None, 
                        callback = None, 
                        options = None)
```

```python
scipy.optimize.basinhopping(func, 
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
                            seed = None)
```

# 多元标量函数的无约束最小化(minimize)

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


## 示例

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

# 多元标量函数的约束最小化(minimize)

- 信任区域约束算法
   - `method = "trust-constr"`
   - 定义边界约束
   - 定义线性约束
   - 定义非线性约束
   - 解决优化问题
- 顺序最小二乘法(SLSQP)算法
   - `method = "SLSQP"`

# 全局最优

# 最小二乘最小化

# 单变量函数最小化器

- 无约束最小化
   - `method = "brent"`
- 有界最小化
   - `method = "bounded"`

# 自定义最小化器

# 寻根

# 线性规划

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