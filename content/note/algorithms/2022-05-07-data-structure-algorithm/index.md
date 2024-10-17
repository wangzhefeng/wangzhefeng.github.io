---
title: 数据结构与算法概览
author: wangzf
date: '2022-05-07'
slug: data-structure-algorithm
categories:
  - algorithms
tags:
  - algorithm
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

- [日常生活中的算法](#日常生活中的算法)
- [算法](#算法)
- [数据结构](#数据结构)
- [数据结构与算法的关系](#数据结构与算法的关系)
</p></details><p></p>

# 日常生活中的算法

1. 查字典：二分查找
2. 整理扑克：插入排序
3. 货币找零：贪心算法
4. ...

# 算法

算法（algorithm）是在有限时间内解决特定问题的一组指令或操作步骤，它具有以下特性：

1. 问题是明确的，包含清晰的输入和输出定义。
2. 具有可行性，能够在有限步骤、时间和内存空间下完成。
3. 各步骤都有确定的含义，在相同的输入和运行条件下，输出始终相同。

# 数据结构

数据结构（data structure）是计算机中组织和存储数据的方式，具有以下设计目标。

1. 空间占用尽量少，以节省计算机内存。
2. 数据操作尽可能快速，涵盖数据访问、添加、删除、更新等。
3. 提供简洁的数据表示和逻辑信息，以便算法高效运行。

数据结构设计是一个充满权衡的过程。如果想在某方面取得提升，往往需要在另一方面作出妥协。

* 链表相较于数组，在数据添加和删除操作上更加便捷，但牺牲了数据访问速度。
* 图相较于链表，提供了更丰富的逻辑信息，但需要占用更大的内存空间。

# 数据结构与算法的关系

![img](images/relationship_between_data_structure_and_algorithm.png)

数据结构与算法高度相关、紧密结合，具体表现在以下三个方面：

1. 数据结构是算法的基石。数据结构为算法提供了结构化存储的数据，以及操作数据的方法。
2. 算法是数据结构发挥作用的舞台。数据结构本身仅存储数据信息，结合算法才能解决特定问题。
3. 算法通常可以基于不同的数据结构实现，但执行效率可能相差很大，选择合适的数据结构是关键。

