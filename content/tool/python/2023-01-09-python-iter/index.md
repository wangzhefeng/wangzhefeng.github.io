---
title: Python 迭代
author: wangzf
date: '2023-01-09'
slug: python-iter
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
img {
    pointer-events: none;
}
</style>

<details><summary>目录</summary><p>

- [Python 可迭代对象](#python-可迭代对象)
- [循环计数器: while 和 range](#循环计数器-while-和-range)
- [非完备遍历: range 和 分片](#非完备遍历-range-和-分片)
- [修改列表: range](#修改列表-range)
- [并行遍历: zip \& map](#并行遍历-zip--map)
  - [zip](#zip)
  - [map](#map)
- [产生偏移和元素: `enumerate`](#产生偏移和元素-enumerate)
- [break, continue, pass, iter-else](#break-continue-pass-iter-else)
- [文件扫描](#文件扫描)
- [Python 列表、字典解析、生成器](#python-列表字典解析生成器)
  - [列表解析](#列表解析)
    - [列表解析基础知识](#列表解析基础知识)
    - [在文件上使用列表解析](#在文件上使用列表解析)
    - [扩展的列表解析语法](#扩展的列表解析语法)
    - [列表解析与 `map`](#列表解析与-map)
    - [增加测试和嵌套循环](#增加测试和嵌套循环)
    - [列表解析和矩阵](#列表解析和矩阵)
    - [理解列表解析](#理解列表解析)
  - [字典解析](#字典解析)
  - [生成器](#生成器)
</p></details><p></p>

`for` 循环包括多数计数器式的循环. 一般而言, `for` 比 `while` 容易写, 
执行时也比较快. 所以每当你需要遍历序列时, 都应该把它作为首选的工具. 

但是, 有些情况下, 需要以更为特定的方式来进行迭代. 例如, 
如果你需要在列表中每隔一个元素或每隔两个元素访问, 或者在过程中修改列表呢? 
如果在同一个 `for` 循环内, 并行遍历一个以上的序列呢? 

Python 提供了两个内置函数, 在 for 循环内定制迭代:

* 内置 `range` 函数返回一系列连续增加的整数, 可作为 for 的索引. 
* 内置 `zip` 函数返回并行元素的元组的列表, 可用于在 for 中内遍历数个序列. 

# Python 可迭代对象


# 循环计数器: while 和 range


# 非完备遍历: range 和 分片


# 修改列表: range


# 并行遍历: zip & map

## zip

`zip` 在 Python 3.0 中也是一个可迭代对象, 因此, 必须将其包含在一个 `list` 调用中以便一次性显示所有结果. 

* 内置函数 `range` 允许我们在 `for` 循环中以非完备的方式遍历序列. 本着同样的精神, 内置的 `zip` 
  函数也让我们使用 `for` 循环来 **并行** 使用多个序列. 
* 在基本运算中, `zip` 会取得一个或多个序列为参数, 然后返回元组的列表, 将这些序列中的并排的元素配成对. 

示例 1:

```python
>>> L1 = [1, 2, 3, 4]
>>> L2 = [5, 6, 7, 8]
>>> zip(L1, L2)
>>> list(zip(L1, L2))
```

示例 2:

```python
>>> for (x, y) in zip(L1, L2):
>>>     print(x, y, '--', x + y)
```

## map

# 产生偏移和元素: `enumerate`

enumerate 函数返回一个生成器对象: 这种对象支持迭代协议, 这个对象有一个 `__next__` 方法, 
由下一个内置函数调用它, 并且循环中每次迭代的时候它会返回一个 `(index, value)` 的元组. 

通过 `range` 来产生字符串中元素的偏移值, 而不是那些偏移值处的元素. 不过, 在有些程序中, 两者都需要: 

- 要用的元素
- 元素的偏移值

从传统意义上来讲, 这是简单的 `for` 循环, 它同时持有一个记录当前偏移值的计数器. 

- 示例 1:

```python
# 普通的 for 循环
>>> S = "spam"
>>> offset = 0
>>> for item in S:
>>>     print(item, "appears at offset", offset)
>>>     offset += 1

# enumerate
>>> S = "spam"
>>> for (offset, item) in enumerate(S):
>>>     print(item, "appears at offset", offset)
```

- 示例 2:

```python
>>> S = "spam"
>>> E = enumerate(S)
>>> E
>>> next(E)
>>> next(E)
>>> next(E)

>>> [c * i for (i, c) in enumerate(S)]
```

# break, continue, pass, iter-else

# 文件扫描

# Python 列表、字典解析、生成器

前提条件: 了解 Python 迭代协议

## 列表解析

* 与 `for` 循环一起使用, 列表解析是最常用的迭代协议的环境之一
* 当我们开始考虑在一个序列中的每项上执行一个操作时, 都可以考虑使用列表解析
* 列表解析的优点: 
    - 列表解析编写起来更加精简
    - 由于构建结果列表的这种代码样式在 Python 代码中十分常见, 因此可以将他们用于多种环境
    - 列表解析比手动的 for 循环语句运行地更快(往往速度快一倍), 因为他们的迭代在解析器内部是以 C 
      语言的速度执行的, 而不是以手动 Python 代码执行的, 特别是对于较大的数据集合, 
      这是使用列表解析的一个主要的性能优点

### 列表解析基础知识

在 Python 中, 列表解析看上去就像一个反向的 `for` 循环, 下面分析一个例子

```python
>>> L = [x + 10 for x in L]
```

其中: 

- `x + 10`: 任意表达式
- `for x in L`: 可迭代对象(for 循环头部)
- `[...]`: 列表

### 在文件上使用列表解析

文件对象有个 `readlines` 方法, 它能一次性地把文件载入到行字符串的一个列表中:

```python
with open("script1.py") as f:
    lines = f.readlines()
    lines = [line.rstrip() for line in lines]
```

```python
lines = [line.rstrip() for line in open("script1.py")]
```

### 扩展的列表解析语法

- 表达式中嵌套的 for 循环可以有一个相关的 if 子句, 来过滤那些测试不为真的结果项
- 列表解析可以变得复杂, 比如, 它们可能包含嵌套的循环, 也可能被编写为一系列的 for 子句. 
  实际上, 它们的完整语法允许任意数目的 for 子句, 每个子句有一个可选的相关的 if 子句

```python
lines = [line.rstrip() for line in open("script1.py") if line[0] == "p"]
```

```python
[x + y for x in 'abc' for y in 'lmn']
```

### 列表解析与 `map`

### 增加测试和嵌套循环

### 列表解析和矩阵

### 理解列表解析

## 字典解析

## 生成器

