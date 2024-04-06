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
    - [双闭区间二分查找](#双闭区间二分查找)
        - [二分查找示例](#二分查找示例)
        - [二分查找实现](#二分查找实现)
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
二分查找算法的输入是一个有序的元素列表，如果要查找的元素包含在列表中，
二分查找返回其位置；否则返回 `null`。

一般而言, 对于包含 `$n$` 个元素的列表, 用二分查找最多需要 `$log_2 n$` 步, 
而简单查找最多需要 `$n$` 步。

## 双闭区间二分查找

### 二分查找示例

> 示例问题：给定一个长度为 `$n$` 的数组 `nums` ，元素按从小到大的顺序排列且不重复。
> 请查找并返回元素 `target` 在该数组中的索引。若数组不包含该元素，则返回 `$-1$`。
> ![img](images/binary_search_example.png)

先初始化指针 `$i=0$` 和 `$j = n-1$`，分别指向数组首元素和尾元素，代表搜索区间 `$[0, n-1]$`。
请注意，中括号表示闭区间，其包含边界值本身。接下来，循环执行以下两步：

1. 计算中点索引 `$m = \lfloor(i + j) / 2\rfloor$`，
   其中 `$\lfloor$` 与 `$\rfloor$` 表示向下取整操作。
2. 判断 `nums[m]` 和 `target` 的大小关系，分为以下三种情况。
    - 当 `nums[m] < target` 时，说明 `target` 在区间 `$[m+1, j]$` 中，因此执行 `$i = m+1$`。
    - 当 `nums[m] > target` 时，说明 `target` 在区间 `$[i, m-1]$`中，因此执行 `$j = m-1$`。
    - 当 `nums[m] = target` 时，说明找到 `target` ，因此返回索引 `$m$`。

<img src="images/binary_search_step1.png" width="100%" />
<img src="images/binary_search_step2.png" width="100%" />
<img src="images/binary_search_step3.png" width="100%" />
<img src="images/binary_search_step4.png" width="100%" />
<img src="images/binary_search_step5.png" width="100%" />
<img src="images/binary_search_step6.png" width="100%" />
<img src="images/binary_search_step7.png" width="100%" />

若数组不包含目标元素，搜索区间最终会缩小为空，此时返回 `$-1$`。
值得注意的是，由于 `$i$` 和 `$j$` 都是 `int` 类型，因此 `$i+j$` 可能会超出 `int` 类型的取值范围。
为了避免大数越界，我们通常采用公式 `$m = \lfloor i + (j-i) / 2\rfloor$` 来计算中点。

### 二分查找实现

```python
def binary_search(nums: List[int], target: int) -> int:
    """
    二分查找（双闭区间）
    """
    # 初始化双闭区间 [0, n-1]，即 low, high 分别指向数组首尾元素
    low, high = 0, len(nums) - 1
    # 循环，当搜索区间为空时跳出(当 low>high 时为空)
    while low <= high:
        # 理论上 Python 的数字可以无限大(取决于内存大小)，无须考虑大数越界问题
        mid = (low + high) // 2  # 计算中点索引 m
        if nums[m] < target:
            low = m + 1  # 此情况说明 target 在区间[m+1,high]中
        elif nums[m] > target:
            high = mid - 1  # 此情况说明 target 在区间[low,m-1]中
        else:
            return mid  # 找到目标元素，返回其索引
    
    return -1  # 未找到目标元素，返回 -1


if __name__ == "__main__":
    my_list = [1, 3, 5, 7, 9]
    result1 = binary_search(my_list, 3)
    print(result1)

    result2 = binary_search(my_list, -1)
    print(result2)
```

* 时间复杂度为 `$O(log n)$`：在二分循环中，区间每轮缩小一半，因此循环次数为 `$log_{2}n$`。
* 空间复杂度为 `$O(1)$`：指针 `$i$` 和 `$j$` 使用常数大小空间。

## 区间表示方法

除了上述双闭区间外，常见的区间表示还有 “左闭右开” 区间，定义为 `$[0, n)$`，
即左边界包含自身，右边界不包含自身。在该表示下，区间 `$[i, j)$` 在 `$i = j$` 时为空。

```python
def binary_search_lcro(nums: list[int], target: int) -> int:
    """
    二分查找（左闭右开区间）
    """
    # 初始化双闭区间 [0, n)，即 low, high 分别指向数组首尾元素
    low, high = 0, len(nums)
    # 循环，当搜索区间为空时跳出(当 low>high 时为空)
    while low < high:
        mid = (low + high) // 2  # 计算中点索引 m
        if nums[m] < target:
            low = m + 1  # 此情况说明 target 在区间[m+1,high)中
        elif nums[m] > target:
            high = mid  # 此情况说明 target 在区间[low,m)中
        else:
            return mid  # 找到目标元素，返回其索引
    
    return -1  # 未找到目标元素，返回 -1


if __name__ == "__main__":
    my_list = [1, 3, 5, 7, 9]
    result1 = binary_search_lcro(my_list, 3)
    print(result1)

    result2 = binary_search_lcro(my_list, -1)
    print(result2)
```

在两种区间表示下，二分查找算法的初始化、循环条件和缩小区间操作皆有所不同。
由于“双闭区间”表示中的左右边界都被定义为闭区间，
因此通过指针 `$i$` 和指针 `$j$` 缩小区间的操作也是对称的。
这样更不容易出错，因此一般建议采用“双闭区间”的写法。

![img](images/binary_search_ranges.png)

## 优点与局限性

二分查找在时间和空间方面都有较好的性能：

* 二分查找的时间效率高。在大数据量下，对数阶的时间复杂度具有显著优势。
  例如，当数据大小 `$n = 2^{20}$` 时，线性查找需要 `$2^{20}=1048576$` 轮循环，
  而二分查找仅需 `$log_{2}2^{20}=20$` 轮循环。
* 二分查找无须额外空间。相较于需要借助额外空间的搜索算法（例如哈希查找），
  二分查找更加节省空间。

然而，二分查找并非适用于所有情况，主要有以下原因。

* <span style='border-bottom:1.5px dashed red;'>二分查找仅适用于有序数据</span>。
  若输入数据无序，为了使用二分查找而专门进行排序，得不偿失。
  因为排序算法的时间复杂度通常为 `$O(n log n)$`，比线性查找和二分查找都更高。
  对于频繁插入元素的场景，为保持数组有序性，需要将元素插入到特定位置，时间复杂度为 `$O(n)$`，
  也是非常昂贵的。
* <span style='border-bottom:1.5px dashed red;'>二分查找仅适用于数组</span>。
  二分查找需要跳跃式（非连续地）访问元素，而在链表中执行跳跃式访问的效率较低，
  因此不适合应用在链表或基于链表实现的数据结构。
* 小数据量下，线性查找性能更佳。在线性查找中，每轮只需 1 次判断操作；而在二分查找中，
  需要 1 次加法、1 次除法、1 ~ 3 次判断操作、1 次加法（减法），共 4 ~ 6 个单元操作；
  因此，当数据量 `$n$` 较小时，线性查找反而比二分查找更快。


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

