---
title: 搜索算法
author: 王哲峰
date: '2023-02-02'
slug: search
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

- [二分查找](#二分查找)
     - [区间表示方法](#区间表示方法)
     - [优点与局限性](#优点与局限性)
- [二分查找插入点](#二分查找插入点)
- [二分查找边界](#二分查找边界)
- [哈希优化策略](#哈希优化策略)
- [重识搜索算法](#重识搜索算法)
- [TODO-搜索算法](#todo-搜索算法)
     - [回溯](#回溯)
     - [递归](#递归)
     - [剪枝](#剪枝)
- [参考](#参考)
</p></details><p></p>

# 二分查找

二分查找（binary search）是一种基于分治策略的高效搜索算法。
它利用数据的有序性，每轮缩小一般搜索范围，直至找到目标元素或搜索区间为空为止。
二分查找算法的输入是一个有序的元素列表，如果要查找的元素包含在列表中, 
二分查找返回其位置；否则返回 `null`。

一般而言, 对于包含 `$n$` 个元素的列表, 用二分查找最多需要 `$log_2 n$` 步, 
而简单查找最多需要 `$n` 步。

二分查找实现: 

```python
# -*- coding: utf-8 -*-

def binary_search(List, item):
    # low 和 high 用于跟踪要在其中查找的列表部分
    low = 0
    high = len(List) - 1

    while low <= high:
        mid = (low + high) // 2
        guess = List[mid]
        if guess == item:
            return mid
        elif guess > item:
            high = mid - 1
        else:
            low = mid + 1
    
    return None


if __name__ == "__main__":
    my_list = [1, 3, 5, 7, 9]
    result1 = binary_search(my_list, 3)
    print(result1)

    result2 = binary_search(my_list, -1)
    print(result2)
```

## 区间表示方法

## 优点与局限性


# 二分查找插入点

# 二分查找边界

# 哈希优化策略

# 重识搜索算法


# TODO-搜索算法

## 回溯

## 递归

## 剪枝


# 参考

* [动画讲编程](https://www.zhihu.com/zvideo/1363902580368814081)

