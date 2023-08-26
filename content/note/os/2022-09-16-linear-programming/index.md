---
title: 线性规划
author: 王哲峰
date: '2022-09-16'
slug: linear-programming
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

- [线性规划的标准型](#线性规划的标准型)
- [单纯形法](#单纯形法)
  - [单纯形法的数学规范型](#单纯形法的数学规范型)
- [内点法](#内点法)
- [列生成法](#列生成法)
- [求偶问题](#求偶问题)
- [拉格朗日乘子法](#拉格朗日乘子法)
- [参考](#参考)
</p></details><p></p>

# 线性规划的标准型

对于线性规划，先来看一个简单的数学规划模型，即：

`$$max Z = 70x_{1} + 30x_{2}$$`

`$$s.t.\begin{cases}
3x_{1} + 9x_{2} \leq 540 \\
5x_{1} + 5x_{2} \leq 450 \\
9x_{1} + 3x_{2} \leq 720 \\
x_{1}, x_{2} \geq 0
\end{cases}$$`

显然这不是线性规划数学模型的标准形式，在线性规划求解方法中，模型的标准形式如下：

1. 目标函数求最大值；
2. 约束条件为等式约束；
3. 约束条件右边的常数项大于等于 0；
4. 所有变量大于或等于 0。

对于非标准形式的模型，约束方程可以通过引入 **松弛变量** 使不等式约束转化为等式约束：

* 如果目标函数是求最小值，则两边乘以 `$-1$` 将求 `$min$` 转成求 `$max$`；
* 如果遇到约束方程右边常数项为负数，则将约束方程乘以 `$-1$` 使常数项非负；
* 如果变量 `$x_{k}$` 没有约束，则既可以使正数也可以是负数，另 `$x_{k} = x_{k}^{'} - x_{k}^{''}$`，其中 `$x_{k}^{'}x_{k}^{''} \geq 0$`。

通过变换，上面模型的标准型如下：

`$$max Z = 70x_{1} + 30x_{2}$$`

`$$s.t.\begin{cases}
3x_{1} + 9x_{2} + x_{3} = 540 \\
5x_{1} + 5x_{2} + x_{4} = 450 \\
9x_{1} + 3x_{2} + x_{5} = 720 \\
x_{1}, x_{2}, x_{3}, x_{4}, x_{5} \geq 0
\end{cases}$$`

# 单纯形法

## 单纯形法的数学规范型

# 内点法

# 列生成法

# 求偶问题

# 拉格朗日乘子法


# 参考

* [十分钟快速掌握单纯形法](https://mp.weixin.qq.com/s?__biz=MzU0NzgyMjgwNg==&mid=2247484683&idx=1&sn=32fbd323572549ebe1d7ceca7e5c79dd&chksm=fb49c8b2cc3e41a4005d70d926c48e4c538ebd573d5ffbdeeba6b10dadc4d03012cc311249c8&scene=21#wechat_redirect)
