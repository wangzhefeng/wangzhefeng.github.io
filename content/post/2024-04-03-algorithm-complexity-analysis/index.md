---
title: 算法复杂度分析
author: 王哲峰
date: '2024-04-03'
slug: algorithm-complexity-analysis
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

- [算法效率评估](#算法效率评估)
- [算法效率的实际测试](#算法效率的实际测试)
- [算法效率的理论估算](#算法效率的理论估算)
    - [什么是复杂度分析](#什么是复杂度分析)
    - [为什么要进行复杂度分析](#为什么要进行复杂度分析)
    - [常用的复杂度分析方法](#常用的复杂度分析方法)
- [迭代与递归](#迭代与递归)
    - [迭代](#迭代)
        - [for 循环](#for-循环)
        - [while 循环](#while-循环)
        - [嵌套循环](#嵌套循环)
    - [递归](#递归)
- [时间复杂度](#时间复杂度)
- [空间复杂度](#空间复杂度)
- [其他复杂度](#其他复杂度)
- [其他资源](#其他资源)
</p></details><p></p>


> Algorithm complexity analysis

# 算法效率评估

算法设计中追求的两个目标：

1. 找到问题的解法
2. 寻求最优解法

在能够解决问题的前提下，衡量算法优劣的评价指标是算法效率，它包括两个维度：

* 时间效率：算法运行速度的快慢
* 空间效率：算法占用内存空间的大小

我们的目标是设计“既快又省”的数据结构与算法。而算法效率评估方法有两种：

1. 实际测试
2. 理论估算

# 算法效率的实际测试

假设现在有算法 A 和算法 B ，它们都能解决同一问题，现在需要对比这两个算法的效率。
最直接的方法是找一台计算机，运行这两个算法，并监控记录它们的运行时间和内存占用情况。
这种评估方式能够反映真实情况，但也存在较大的局限性。

* 一方面，难以排除测试环境的干扰因素。硬件配置会影响算法的性能。
  这意味着我们需要在各种机器上进行测试，统计平均效率，而这是不现实的。
* 另一方面，展开完整测试非常耗费资源。随着输入数据量的变化，算法会表现出不同的效率。
  因此，为了得到有说服力的结论，我们需要测试各种规模的输入数据，而这需要耗费大量的计算资源。

# 算法效率的理论估算

由于实际测试具有较大的局限性，因此我们可以考虑仅通过一些计算来评估算法的效率。
这种估算方法被称为<span style='border-bottom:1.5px dashed red;'>渐近复杂度分析（asymptotic complexity analysis）</span>，简称<span style='border-bottom:1.5px dashed red;'>复杂度分析</span>。
复杂度分析为我们提供了一把评估算法效率的“标尺”，
使我们可以衡量执行某个算法所需的时间和空间资源，对比不同算法之间的效率。

复杂度分析能够体现算法运行所需的<span style='border-bottom:1.5px dashed red;'>时间和空间资源</span>与<span style='border-bottom:1.5px dashed red;'>输入数据大小</span>之间的关系。
它描述了随着输入数据大小的增加，算法执行所需时间和空间的增长趋势。
可以将其分为三个重点来理解：

* “时间和空间资源”：分别对应<span style='border-bottom:1.5px dashed red;'>时间复杂度（time complexity）</span>和<span style='border-bottom:1.5px dashed red;'>空间复杂度（space complexity）</span>。
* “随着输入数据大小的增加”：意味着复杂度反映了<span style='border-bottom:1.5px dashed red;'>算法运行效率</span>与<span style='border-bottom:1.5px dashed red;'>输入数据体量</span>之间的关系。
* “时间和空间的增长趋势”：表示复杂度分析关注的不是运行时间或占用空间的具体值，
  而是<span style='border-bottom:1.5px dashed red;'>时间或空间增长的“快慢”</span>。

复杂度分析克服了实际测试方法的弊端，体现在以下两个方面：

* 它独立于测试环境，分析结果适用于所有运行平台。
* 它可以体现不同数据量下的算法效率，尤其是在大数据量下的算法性能。

## 什么是复杂度分析

1. 数据结构与算法的作用是什么？
   - 数据结构与算法的诞生是让计算机执行得更快、更省空间。
2. 用什么来评判数据结构与算法的好坏？
   - 可以从执行时间和占用空间两个方面来评判数据结构与算法的好坏。
3. 什么是复杂度？
   - 用时间复杂度和空间复杂度来描述性能问题, 两者统称为复杂度。
4. 复杂度描述了什么？
   - 复杂度描述的是算法执行时间(或占用空间)与数据规模的增长关系。

## 为什么要进行复杂度分析

1. 复杂度分析和性能分析相比有什么有点？
   - 复杂度分析有不依赖于环境、成本低、效率高、易操作、指导性强的特点。
2. 为什么要进行复杂度分析？
   - 复杂度描述的是算法执行时间(或占用空间)与数据规模的增长关系。

## 常用的复杂度分析方法

一般而言, 应选择效率最高的算法, 以最大限度地节省运行时间或占用空间。

1. 什么方法可以进行复杂度分析？
    - 大 O 表示法
2. 什么是大 O 表示法？
    - 算法的执行时间与每行代码的执行次数成正比 `$T(n) = O(f(n))$`,
     其中 `$T(n)$` 表示算法执行总时间,  `$f(n)$` 表示每行代码执行总次数, 
     而 `$n$` 往往表示数据的规模
    - 大 O 表示法是一种特殊的表示法, 指出了随着输入的增加, 算法的运行和时间将以什么样的速度增加
    - 大 O 表示法指的并非算法以秒为单(时间)位的速度 
    - 大 O 表示法能够比较 **操作数**, 它指出了算法操作数的增速
    - 大 O 表示法指出了最糟糕情况下的运行时间
3. 大 O 表示法的特点？
    - 由于时间复杂度描述的是算法执行时间与数据规模的增长变化趋势, 
      常量阶、低阶以及系数实际上对这种增长趋势不产决定性影响, 
      所以在做时间复杂度分析时忽略这些项
4. 复杂度分析法则
    - 单段代码看频率: 看代码片段中循环代码的时间复杂度
    - 多段代码看最大: 如果多个 for 循环, 看嵌套循环最多的那段代码的时间复杂度
    - 嵌套代码求乘积: 循环、递归代码, 将内外嵌套代码求乘积取时间复杂度
    - 多个规模求加法: 假如有两个参数控制两个循环的次数, 那么这时就取二者复杂度相加
5. 常见的大 O 运行时间(由块到慢)
    - `$O(\log n)$` 
        - 对数时间(log time)
        - 例如: 二分查找
    - `$O(n)$` 
        - 线性时间(linear time)
        - 例如: 简单查找
    - `$O(n \cdot \log n)$`
        - 快速排序
    - `$O(n^{2})$` 
        - 选择排序
    - `$O(n!)$` 
        - 旅行商问题

# 迭代与递归

重复执行某个任务与复杂度分析息息相关，程序中的重复执行任务为来年各种基本的程序控制结构：迭代、递归。

## 迭代

### for 循环


### while 循环


### 嵌套循环



## 递归




# 时间复杂度

1. 什么是时间复杂度？
    - 所有代码的执行时间 T(n) 与每行代码的执行次数 n 成正比: `$T(n) = O(f(n))$`
2. 分析的三个方法
    - 最多法则：忽略掉公式中的常量、低阶、系数，取最大循环次数就可以了，也就是循环次数最多的那行代码
    - 加法法则：总复杂度等于循环次数最多的那段复杂度
    - 乘法法则：当遇到嵌套的 `for` 循环的时候，时间复杂度就是内外循环的乘积

# 空间复杂度

1. 什么是空间复杂度？
    - 表示算法存储空间与数据规模之间的增长关系
2. 最常见的空间复杂度
    - `$O(1)$`：常量级的空间复杂度表示方法，无论是一行代码，还是多行，只要是常量级的就用 O(1) 表示
    - `$O(n)$`
    - `$O(n^{2})$`
    - `$O(log n) | O(n log n)$`：对数阶空间复杂度，最难分析的一种空间复杂度
    - `$O(m+n)$`：加法法则
    - `$O(m \times n)$`：乘法法则

# 其他复杂度

1. 最好、最坏时间复杂度
    - 所谓的最好、最坏时间复杂度分别对应代码最好的情况和最坏的情况下的执行
2. 平均时间复杂度
    - 平均时间复杂度需要借助概率论的知识去分析，也就是我们概率论中所说的加权平均值，也叫做期望值
3. 均摊时间复杂度
    - 什么是均摊时间复杂度
        - 比如我们每 n 次插入数据的时间复杂度为 O(1)，就会有一次插入数据的时间复杂度为 O(n)，
          将这一次的时间复杂度平均到 n 次插入数据上，时间复杂度还是 O(1)
    - 适用场景
        - 一般应用于某一数据结构，连续操作时间复杂度比较低，但是个别情况时间复杂度特别高，
          将特别高的这一次进行均摊到较低的操作上
    - 几种复杂度性能对比
        - `$O(n^{2}) > O(n logn) > O(n) > O(logn)$`

# 其他资源

* [Helo 算法--第 2 章 复杂度分析](https://www.hello-algo.com/chapter_computational_complexity/)
