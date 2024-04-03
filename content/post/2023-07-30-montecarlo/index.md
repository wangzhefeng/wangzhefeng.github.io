---
title: 蒙特卡洛模拟
author: 王哲峰
date: '2023-07-30'
slug: montecarlo
categories:
  - algorithms
tags:
  - algorithm
---

<style>
h1 {
    background-color: #2B90B6;
    background-image: linear-gradient(45deg, #4EC5D4 10%, #146b8c 20%);
    background-size: 100%;
    -webkit-background-clip: text;
    -moz-background-clip: text;
    -webkit-text-fill-color: transparent;
    -moz-text-fill-color: transparent;
}
h2 {
    background-color: #2B90B6;
    background-image: linear-gradient(45deg, #4EC5D4 10%, #146b8c 20%);
    background-size: 100%;
    -webkit-background-clip: text;
    -moz-background-clip: text;
    -webkit-text-fill-color: transparent;
    -moz-text-fill-color: transparent;
}
h3 {
    background-color: #2B90B6;
    background-image: linear-gradient(45deg, #4EC5D4 10%, #146b8c 20%);
    background-size: 100%;
    -webkit-background-clip: text;
    -moz-background-clip: text;
    -webkit-text-fill-color: transparent;
    -moz-text-fill-color: transparent;
}
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

- [蒙特卡洛方法](#蒙特卡洛方法)
  - [蒙特卡洛方法的起源和发展](#蒙特卡洛方法的起源和发展)
- [EM 算法及其 MCMC 方法](#em-算法及其-mcmc-方法)
- [总结](#总结)
- [参考](#参考)
</p></details><p></p>


# 蒙特卡洛方法

## 蒙特卡洛方法的起源和发展

蒙特卡洛方法（Monte Carlo method），也称统计模拟方法，<span style='border-bottom:1.5px dashed green;'>利用随机数进行数值模拟的方法</span>。
是 1940 年代中期由于科学技术的发展和电脑的发明，而提出的一种以概率统计理论为指导的数值计算方法。
是指使用随机数（或更常见的伪随机数）来解决很多计算问题的方法。

20 世纪 40 年代，在科学家 **冯·诺伊曼**、**斯塔尼斯拉夫·乌拉姆** 和 **尼古拉斯·梅特罗波利斯** 于洛斯阿拉莫斯国家实验室为核武器计划工作时，
发明了蒙特卡罗方法。因为乌拉姆的叔叔经常在摩纳哥的蒙特卡洛赌场输钱得名，而蒙特卡罗方法正是以概率为基础的方法。
与蒙特卡洛方法对应的是确定型算法。

事实上，Monte Carlo 方法的基本思想很早以前就被人们所发现和利用。
早在 17 世纪，人们就知道用事件发生的“频率”来决定事件的“概率”。
18 世纪下半叶的法国学者 Buffon 提出用投针试验的方法来确定圆周率 `$\pi$` 值。这个著名的 Buffon 试验是 Monte Carlo 方法的最早
的尝试。

历史上曾有几位学者相继做过这样的试验。不过呢，他们的试验是费时费力的，同时精度不够高，
实施起来也很困难。然而，随着计算机技术的飞速发展，人们不需要具体实施这些试验，
而只要在计算机上进行大量的、快速的模拟试验就可以了

在大众的心目中，科学的代言人是心不在焉的牛顿或者爆炸式发型的爱因斯坦，但这只是传统形象，
比他们更了解现代计算技术的冯·诺伊曼是个衣着考究，风度翩翩的人物，
他说：<span style='border-bottom:1.5px dashed green;'>纯粹数学和应用数学的许多分支非常需要计算工具，
用以打破目前由于纯粹分析的研究方法不能解决非线性问题而形成的停滞状态。</span>。
Monte Carlo 方法是现代计算技术的最为杰出的成果之一，它在工程领域的作用是不可比拟的。

Monte Carlo 方法的发展历史：

1. Buffon 投针实验：1786 年，法国数学家 Buffon 利用投针实验估计 `$\pi$` 的值。
2. 1930 年，Enrico Fermi 利用 Monte Carlo 方法研究中子的扩散，
   并设计了一个 Monte Carlo 机械装置 Fermiac，用于计算核反应堆的临界状态。
3. Von Meumann 是 Monte Carlo 方法的正式奠基者，他与 Stanislaw Ulam 合作建立了概率密度函数、
   反累积分布函数的数学基础，以及伪随机数产生器。在这些工作中，Stanislaw Ulam 意识到了数字计算机的重要性。

# EM 算法及其 MCMC 方法

EM 算法是一种迭代方法，最初由 Dempster 等提出，并主要应用于较为复杂的后验分布，来计算后验均值或者后验众数，
即极大似然估计的一种实现算法。最大优点是简单有效。

MCMC 算法：当后验分布较为复杂时，对于后验分布的积分计算，
像后验均值、后验方差、后验分布的分位数等就不得不求助于 MCMC 算法。
它在统计物理学中得到广泛的应用。近年来迅速发展到贝叶斯统计、显著性检验、极大似然估计等方面。

# 总结

Monte Carlo 方法与数值方法的不同：

1. Monte Carlo 方法利用随机抽样的方法来求解物理问题；
2. 数值解法从一个物理系统的数学模型出发，通过求解一系列的微分方程的导出系统的未知状态；

Monte Carlo 方法并非只能用来解决包含随机的过程的问题，许多利用 Monte Carlo 方法进行求解的问题中并不包含随机过程。
例如：用 Monte Carlo 方法计算定积分。对这样的问题可将其转换成相关的随机过程，然后用 Monte Carlo 方法进行求解。

Monte Carlo 算法的主要组成部分：

1. 概率密度函数（PDF）：必须给出描述一个物理系统的一组概率密度函数；
2. 随机数产生器：能够产生在区间 `$[0, 1]$` 上的均匀分布的随机数；
3. 抽样规则：如何从在区间 `$[0, 1]$` 上的均匀分布的随机数出发，随机抽取服从给定的 PDF 的随机变量；
4. 模拟结果记录：记录一些感兴趣的量的模拟结果
5. 误差估计：必须确定统计误差（或方差）随模拟次数以及其他一些量的变化；
6. 减少方差的技术：利用该技术可减少模拟过程中计算的次数；
7. 并行和矢量化：可以在先进的并行计算机上运行的有效算法；

# 参考

* [wiki-蒙特卡罗方方法](https://zh.wikipedia.org/zh-sg/%E8%92%99%E5%9C%B0%E5%8D%A1%E7%BE%85%E6%96%B9%E6%B3%95)
* [Monte-Carlo方法](https://dsp.whu.edu.cn/course/signalde/image/ch16.pdf)