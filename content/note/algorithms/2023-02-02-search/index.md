---
title: 搜索算法
author: wangzf
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
    - [问题](#问题)
    - [双闭区间二分查找](#双闭区间二分查找)
        - [算法](#算法)
        - [实现](#实现)
    - [左闭右开区间二分法](#左闭右开区间二分法)
        - [算法](#算法-1)
        - [实现](#实现-1)
    - [优点与局限性](#优点与局限性)
- [二分查找插入点](#二分查找插入点)
    - [无重复元素的情况](#无重复元素的情况)
        - [问题](#问题-1)
        - [算法](#算法-2)
        - [实现-双闭区间](#实现-双闭区间)
        - [实现-左闭右开区间](#实现-左闭右开区间)
    - [存在重复元素的情况](#存在重复元素的情况)
        - [问题](#问题-2)
        - [算法](#算法-3)
        - [实现-双闭区间](#实现-双闭区间-1)
        - [实现-左闭右开区间](#实现-左闭右开区间-1)
- [二分查找边界](#二分查找边界)
    - [查找左边界](#查找左边界)
    - [查找又边界](#查找又边界)
- [哈希优化策略](#哈希优化策略)
    - [线性查找--以时间换空间](#线性查找--以时间换空间)
    - [哈希查找--以空间换时间](#哈希查找--以空间换时间)
- [重识搜索算法](#重识搜索算法)
    - [暴力搜索](#暴力搜索)
    - [自适应搜索](#自适应搜索)
    - [搜索方法选取](#搜索方法选取)
- [参考](#参考)
</p></details><p></p>

# 二分查找

二分查找（binary search）是一种基于分治策略的高效搜索算法。
它利用数据的有序性，每轮缩小一般搜索范围，直至找到目标元素或搜索区间为空为止。
二分查找算法的输入是一个有序的元素列表，如果要查找的元素包含在列表中，
二分查找返回其位置；否则返回 `null`。

一般而言, 对于包含 `n` 个元素的列表, 用二分查找最多需要 `$log_2 n$` 步, 
而简单查找最多需要 `n` 步。

## 问题

> 给定一个长度为 `n` 的数组 `nums` ，元素按从小到大的顺序排列且不重复。
> 请查找并返回元素 `target` 在该数组中的索引。若数组不包含该元素，则返回 `-1`。
> ![img](images/binary_search_example.png)

## 双闭区间二分查找

### 算法

先初始化指针 `i=0` 和 `j = n-1`，分别指向数组首元素和尾元素，代表搜索区间 `[0, n-1]`。
请注意，中括号表示闭区间，其包含边界值本身。接下来，循环执行以下两步：

1. 计算中点索引 `$m = \lfloor(i + j) / 2\rfloor$`，
   其中 `$\lfloor$` 与 `$\rfloor$` 表示向下取整操作。
2. 判断 `nums[m]` 和 `target` 的大小关系，分为以下三种情况。
    - 当 `nums[m] < target` 时，说明 `target` 在区间 `[m+1, j]` 中，因此执行 `i = m+1`。
    - 当 `nums[m] > target` 时，说明 `target` 在区间 `[i, m-1]` 中，因此执行 `j = m-1`。
    - 当 `nums[m] = target` 时，说明找到 `target` ，因此返回索引 `m`。

<img src="images/binary_search_step1.png" width="48%" />
<img src="images/binary_search_step2.png" width="48%" />

<img src="images/binary_search_step3.png" width="48%" />
<img src="images/binary_search_step4.png" width="48%" />

<img src="images/binary_search_step5.png" width="48%" />
<img src="images/binary_search_step6.png" width="48%" />

<img src="images/binary_search_step7.png" width="48%" />

若数组不包含目标元素，搜索区间最终会缩小为空，此时返回 `-1`。
值得注意的是，由于 `i` 和 `j` 都是 `int` 类型，因此 `i+j` 可能会超出 `int` 类型的取值范围。
为了避免大数越界，我们通常采用公式 `$m = \lfloor i + (j-i) / 2\rfloor$` 来计算中点。

### 实现

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
* 空间复杂度为 `$O(1)$`：指针 `i` 和 `j` 使用常数大小空间。

## 左闭右开区间二分法

### 算法

除了上述双闭区间外，常见的区间表示还有 “左闭右开” 区间，定义为 `[0, n)`，
即左边界包含自身，右边界不包含自身。在该表示下，区间 `[i, j)` 在 `i = j` 时为空。

在两种区间表示下，二分查找算法的初始化、循环条件和缩小区间操作皆有所不同。
由于“双闭区间”表示中的左右边界都被定义为闭区间，
因此通过指针 `$i$` 和指针 `$j$` 缩小区间的操作也是对称的。
这样更不容易出错，因此一般建议采用“双闭区间”的写法。

![img](images/binary_search_ranges.png)

### 实现

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

二分查找不仅可以搜索目标元素，还可以解决许多变种问题，比如搜索目标元素的插入位置。

## 无重复元素的情况

### 问题

> 给定一个长度为 `n` 的有序数组 `nums` 和一个元素 `targe`，数组不存在重复元素。
> 现将 `target` 插入数组 `nums` 中，并保持其有序性。若数组中已存在元素 `target`，
> 则插入到其左方。请返回插入后 `target` 在数组中的索引。
> ![img](images/binary_search_example2.png)

### 算法

如果想复用上一节的二分查找代码，则需要回答以下两个问题：

1. 问题一：当数组中包含 `target` 时，插入点的索引是否是该元素的索引？
    - 题目要求将 `target` 插入到相等元素的左边，这意味着新插入的 `target` 替换了原来 `target` 的位置。
      也就是说，当数组包含 `target` 时，插入点的索引就是该 `target` 的索引。
2. 问题二：当数组中不存在 `target` 时，插入点是哪个元素的索引？
    - 进一步思考二分查找过程：当 `nums[m] < target` 时 `i` 移动，
      这意味着指针 `i` 在向大于等于 `target` 的元素靠近。同理，
      指针 `j` 始终在向小于等于 `target` 的元素靠近。
      因此二分结束时一定有：`i` 指向首个大于 target 的元素，
      `j` 指向首个小于 `target` 的元素。易得当数组不包含 `target` 时，
      插入索引为 `i`。

### 实现-双闭区间

```python
def binary_search_insertion_simple(nums: list[int], target: int) -> int:
    """
    二分查找插入点（无重复点）
    """
    i, j = 0, len(nums) - 1  # 初始化双闭区间
    while i <= j:
        m = (i + j) // 2  # 计算中点索引 m
        if nums[m] < target:
            i = m + 1  # target 在区间 [m+1,j] 中
        elif nums[m] > target:
            j = m - 1  # target 在区间 [i,m-1] 中
        else:
            return m  # 找到 targe, 返回插入点 m
    # 未找到 target，返回插入点 i
    return i
```

### 实现-左闭右开区间

```python

```

## 存在重复元素的情况

### 问题

> 给定一个长度为 `n` 的有序数组 `nums` 和一个元素 `targe`，数组存在重复元素。
> 现将 `target` 插入数组 `nums` 中，并保持其有序性。若数组中已存在元素 `target`，
> 则插入到其左方。请返回插入后 `target` 在数组中的索引。
> ![img](images/binary_search_example2.png)

### 算法

假设数组中存在多个 `target`，则普通二分查找只能返回其中一个 `target` 的索引，
而无法确定该元素的左边和右边还有多少 `target`。

题目要求将目标元素插入到最左边，所以我们需要查找数组中最左一个 `target` 的索引。
初步考虑通过下图所示的步骤实现：

![img](images/binary_search_insertion_step.png)

1. 执行二分查找，得到任意一个 `target` 的索引，记为 `k`；
2. 从索引 `k` 开始，向左进行线性遍历，当找到最左边的 `target` 时返回。

此方法虽然可用，但其包含线性查找，因此时间复杂度为 `$O(n)$`。
当数组中存在很多重复的 `target` 时，该方法效率很低。

现考虑拓展二分查找代码，如下图所示，整体流程保持不变，每轮先计算中点索引 `m`，
再判断 `target` 和 `nums[m]` 的大小关系，分为以下几种情况：

* 当 `nums[m] < target` 或 `nums[m] > target` 时，说明还没有找到 `target`，
  因此采用普通二分查找的缩小区间操作，从而使指针 `i` 和 `j` 向 `target` 靠近；
* 当 `nums[m] == target` 时，说明小于 `target` 的元素在区间 `[i, m-1]` 中，
  因此采用 `j = m - 1` 来缩小区间，从而使指针 `j` 向小于 `target` 的元素靠近；

循环完成后，`i` 指向最左边的 `target`，`j` 指向首个小于 `target` 的元素，
因此索引 `i` 就是插入点。

<img src="images/binary_search_insertion_step1.png" width="48%" />
<img src="images/binary_search_insertion_step2.png" width="48%" />

<img src="images/binary_search_insertion_step3.png" width="48%" />
<img src="images/binary_search_insertion_step4.png" width="48%" />

<img src="images/binary_search_insertion_step5.png" width="48%" />
<img src="images/binary_search_insertion_step6.png" width="48%" />

<img src="images/binary_search_insertion_step7.png" width="48%" />
<img src="images/binary_search_insertion_step8.png" width="48%" />

观察以下代码，判断分支 `nums[m] > target` 和 `nums[m] == target` 的操作相同，
因此两者可以合并。

即便如此，我们仍然可以将判断条件保持展开，因为其逻辑更加清晰、可读性更好。

### 实现-双闭区间

```python
def binary_search_insertion(nums: list[int], target: int) -> int:
    """
    二分查找插入点（存在重复元素）
    """
    i, j = 0, len(nums) - 1  # 初始化双闭区间 [0, n-1]
    while i <= j:
        m = (i + j) // 2
        if nums[m] < target:
            i = m + 1  # target 在区间 [m+i, j] 中
        elif nums[m] > target:
            j = m - 1  # target 在区间 [i, m-1] 中
        else:
            j = m - 1  # 首个小于 target 的元素在区间 [i, m-1] 中
    # 返回插入点 i
    return i
```

### 实现-左闭右开区间

```python
# TODO
def binary_search_insertion(nums: list[int], target: int) -> int:
    """
    二分查找插入点（存在重复元素）
    """
    i, j = 0, len(nums) - 1  # 初始化双闭区间 [0,n-1]
    while i <= j:
        m = (i + j) // 2  # 计算中点索引 m
        if nums[m] < target:
            i = m + 1  # target 在区间 [m+1,j] 中
        elif nums[m] > target:
            j = m - 1  # target 在区间 [i,m-1] 中
        else:
            j = m - 1  # 首个小于 target 的元素在区间 [i,m-1] 中
    # 返回插入点 i
    return i
```

总的来看，二分查找无非就是给指针 `$i$` 和 `$j$` 分别设定搜索目标，
目标可能是一个具体的元素（例如 `target`），也可能是一个元素范围（例如小于 `target` 的元素）。

在不断的循环二分中，指针 `$i$` 和 `$j$` 都逐渐逼近预先设定的目标。
最终，它们或是成功找到答案，或是越过边界后停止。

# 二分查找边界

## 查找左边界


## 查找又边界


# 哈希优化策略

## 线性查找--以时间换空间



## 哈希查找--以空间换时间

# 重识搜索算法

搜索算法（searching algorithm）用于在数据结构（例如数组、链表、树或图）中搜索一个或一组满足特定条件的元素。
搜索算法可根据实现思路分为以下两类：

* 通过遍历数据结构来定位目标元素，例如数组、链表、树和图的遍历等。
* 利用数据组织结构或数据包含的先验信息，实现高效元素查找，例如二分查找、哈希查找和二叉搜索树查找等。

## 暴力搜索

暴力搜索通过遍历数据结构的每个元素来定位目标元素：

* <span style='border-bottom:1.5px dashed red;'>线性搜索</span> 适用于数组和链表等线性数据结构。
  它从数据结构的一端开始，逐个访问元素，直到找到目标元素或到达另一端仍没有找到目标元素为止；
* <span style='border-bottom:1.5px dashed red;'>广度优先搜索</span> 和 <span style='border-bottom:1.5px dashed red;'>深度优先搜索</span> 是图和树的两种遍历策略。
    - 广度优先搜索从初始节点开始逐层搜索，由近及远访问各个节点。
    - 深度优先搜索从初始节点开始，沿着一条路径走到头，再回溯并尝试其他路径，直到遍历完整个数据结构。

暴力搜索地优点是简单且通用性好，无须对数据做预处理和借助额外的数据结构。
然而，此类算法的时间复杂度为 `$O(n)$`，其中 `$n$` 为元素数量，因此在数量较大的情况下性能较差。

## 自适应搜索

自适应搜索利用数据的特有属性（例如有序性）来优化搜索过程，从而更高效地定位目标元素。

* <span style='border-bottom:1.5px dashed red;'>二分查找</span> 利用数据地有序性实现高效查找，
  仅使用于数组。
* <span style='border-bottom:1.5px dashed red;'>哈希查找</span> 利用哈希表将搜索数据和目标数据建立为键值对映射，
  从而实现查询操作。
* <span style='border-bottom:1.5px dashed red;'>树查找</span> 在特定地树结构（例如二叉搜索树）中，
  基于比较节点值来快速排除节点，从而定位目标元素。

此类算法地优点是效率高，时间复杂度可达到 `$O(log n)$` 甚至 `$O(1)$`。
然而，使用这些算法往往需要对数据进行预处理。例如，二分查找需要预先对数组进行排序。
哈希查找和树查找都需要借助额外的数据结构，维护这些数据结构也需要额外的时间和空间开销。



## 搜索方法选取




# 参考

* [动画讲编程](https://www.zhihu.com/zvideo/1363902580368814081)

