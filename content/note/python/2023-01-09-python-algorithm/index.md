---
title: Python 算法图解
author: 王哲峰
date: '2023-01-09'
slug: python-algorithm
categories:
  - Python
tags:
  - tool
---

算法图解--像小说一样有趣的算法入门书

- 二分查找
- 递归

# 算法简介

## 算法运行时间

一般而言, 应选择效率最高的算法, 以最大限度地介绍运行时间或占用空间. 

- 大 O 表示法
    - 大 O 表示法是一种特殊的表示法, 指出了随着输入的增加, 算法的运行和时间将以什么样的速度增加
    - 大 O 表示法指的并非算法以秒为单(时间)位的速度 
    - 大 O 表示法能够比较 **操作数**, 它指出了算法操作数的增速
    - 大 O 表示法指出了最糟糕情况下的运行时间
- 常见的大 O 运行时间[由块到慢]
    - `$O(\log n)$` 
        - 对数时间(log time)
        - 例如: 二分查找
    - `$O(n)$` 
        - 线性时间(linear time)
        - 例如: 简单查找
    - `$O(n * \log n)$`
        - 快速排序
    - `$O(n^{2})$` 
        - 选择排序
    - `$O(n!)$` 
        - 旅行商问题

## 二分查找

二分查找简介: 

* 二分查找是一种算法, 其输入是一个有序的元素列表. 如果要查找的元素包含在列表中, 二分查找返回其位置; 否则返回 `null`.
    - 一般而言, 对于包含 `$n$` 个元素的列表, 用二分查找最多需要 `$log_2 n$` 步, 而简单查找最多需要 `$n` 步

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

## 内存工作原理

计算机就像是很多抽屉的集合, 每个抽屉都有地址. 

# 选择排序

```python
# -*- coding: utf-8 -*-

def findSmallest(arr):
    """
    寻找数组中最小元素
    """
    # 存储最小的值
    smallest = arr[0]
    # 存储最小元素的索引
    smallest_index = 0
    for i in range(1, len(arr)):
        if arr[i] < smallest:
            smallest = arr[i]
            smallest_index = i
    
    return smallest_index

def selectionSort(arr):
    """
    选择排序
    """
    newArr = []
    for i in range(len(arr)):
        smallest_index = findSmallest(arr)
        newArr.append(arr.pop(smallest_index))
    
    return newArr

if __name__ == "__main__":
    my_list2 = [5, 3, 6, 2, 10]
    result3 = selectionSort(my_list2)
    print(result3)
```

# 递归

# 快速排序

# 散列表

# 广度、深度优先搜索

# 狄克斯特拉算法

# 贪婪算法

# 动态规划

# K 最近邻算法

# 反向索引

# 傅里叶变换

# 并行算法

# MapReduce

# 布隆过滤器和 HyperLogLog

# SHA 算法

# 局部敏感的散列算法

# Diffie-Hellman 密钥交换


# 线性规划



