---
title: 排序算法
author: 王哲峰
date: '2023-02-02'
slug: sort
categories:
  - algorithms
tags:
  - algorithm
---

![img](images/sort.png)

# 快速排序


# 归并排序


# 计数排序



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

