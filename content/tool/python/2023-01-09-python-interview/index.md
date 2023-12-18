---
title: Python 面试问题
author: 王哲峰
date: '2023-01-09'
slug: python-interview
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

- [Python 数据结构](#python-数据结构)
- [Python Tuple 赋值问题](#python-tuple-赋值问题)
- [python buildin functions](#python-buildin-functions)
- [python collections 库](#python-collections-库)
- [Python 单例模式](#python-单例模式)
- [Python 深拷贝和浅拷贝](#python-深拷贝和浅拷贝)
  - [浅拷贝](#浅拷贝)
  - [深拷贝](#深拷贝)
- [Python 序列反转](#python-序列反转)
- [Python 字典反转](#python-字典反转)
  - [压缩器](#压缩器)
  - [字典解析](#字典解析)
- [Python求众数](#python求众数)
  - [numpy.argmax(np.bincount())](#numpyargmaxnpbincount)
  - [scipy.stats](#scipystats)
- [Panda 和 Numpy 连接](#panda-和-numpy-连接)
- [字符串拼接](#字符串拼接)
  - [+=](#)
  - [% / .format()](#--format)
  - ["".join()](#join)
- [随机数生成 (函数参数)](#随机数生成-函数参数)
- [Python-Numpy](#python-numpy)
- [解析字符串](#解析字符串)
</p></details><p></p>

# Python 数据结构

对象分类: 

| 对象类型  | 对象名            | 分类 | 是否可变 |
|----------|------------------|-----|----------|
| 数字      | int,float,complex | 数值 | 否       |
| 字符串    | string            | 序列 | 否       |
| 列表      | list              | 序列 | 是       |
| 字典      | dict              | 对应 | 是       |
| 元组      | tuple             | 序列 | 否       |
| 文件      | file              | 扩展 | ...      |
| Sets      | set               | 集合 | 是       |
| frozenset | ...               | 集合 | 否       |
| bytearay  | ...               | 序列 | 是       |

要点: 

- 对象根据分类来共享操作来; 例如, 字符串、列表、元组都共享: 合并(`+`), 长度(`length`), 索引(`[]`)等序列操作. 
- 只有可变对象(列表, 字典, 集合)可以原处修改; 不能原处修改数字, 字符串, 或元组; 
- 数字类型: 整数, 浮点数, 复数, 小数, 分数; 
- 字符串类型: str, bytes, bytearray; 
- 集合类似于一个无值的字典的键, 但是, 他们不能映射为值, 并且没有顺序; 因此, 集合不是一个映射类型或者一个序列类型, frozenset是集合的一个不可变版本; 
- 除了类型分类操作, 所有类型都有可调用的方法, 这些方法通常特定于它们的类型; 
- 列表、字典、元组可以包含任何种类的对象; 
- 列表、字典、元组可以任意嵌套; 
- 列表、字典可以动态地扩大克缩小; 

# Python Tuple 赋值问题

下面的代码输出什么结果?

```python
try:
   a, b = (123,)
except Exception as e:
   print("Error:", e)
finally:
   print("End")
```

输出结果: 

```
> Error: Exception
> End
```


# python buildin functions

- `int()`
- `float()`
- `long()`
- `complex()`
- `str()`
- `bytearray()`
- `bool()`
- `list()`
- `tuple()`
- `dict()`
- `forzenset()`
- `set()`
- `hash()`
- `bin()`
- `dir()`
- `print()`
- `type()`
- `delattr()`
   - 删除属性
   - delattr(object, name)

```python
   class Coordinate():
       x = 10
       y = -5
       z = 0

   point = Coordinate()
   delattr(Coordinate, "z")
```

- `getattr()`
- `hasattr()`
- `setattr()`
- `chr()`
- `unichr()`
- `open()`
- `raw_input()`
- `input()`
- `file()`
- `sum()`
- `abs()`
- `pow()`
- `max()`
- `min()`
- `divmod()`
- `all()`
- `any()`
- `format()`
- `callable()`
- `classmethod()`
- `cmp()`
- `compile()`
- `eval()`
- `execfile()`
- `globals()`
- `help()`
- `id()`
- `issubclass()`
- `iter()`
- `len()`
- `locals()`
- `memoryview()`
- `next()`
- `object()`
- `otc()`
- `ord()`
- `property()`
- `range()`
- `xrange()`
- `reload()`
- `repr()`
- `reverse()`
- `round()`
- `slice()`
- `sorted()`
   - 对所有可迭代的对象进行排序, 返回新的list
   - sort()是list上的方法, 返回的是对已存在的list进行操作, 会改变原来的list

```python
sotred(iterable, key = None, reverse = False)

L = [5, 2, 3, 1, 4]
L.sort()
L_sorted = sorted(L)

D = {1: "D", 2: "B", 3: "B", 4: "E", 5: "A"}
D_sorted = sorted(D)

L2 = [5, 0, 6, 1, 2, 7, 3, 4]
L2_sorted = sorted(L2, key = lambda x: x * -1, reverse = True)
```

- `staticmethod()`
- `super()`
- `unicode()`
- `vars()`
- `__import__()`
   - 动态加载类和函数, 一个模块经常变化
- `zip()`, `zip(*)`
   - 用于将可迭代对象作为参数, 将对象中对应的元素打包成一个个tuple,
      然后返回由这些tuple组成的对象, 节约内存
   - 如果各个迭代器的元素个数不一致, 则返回列表长度与最短的对象相同, 利用 `*` 可以将tuple解压为list

```python
a = [1, 2, 3]
b = [4, 5, 6]
c = [4, 5, 6, 7, 8]

zipped = zip(a, b)
list(zipped)

list(zip(a, c))

a1, a2 = zip(*zip(a, b))
list(a1)
list(a2)
```

- `map()`
   - 根据提供的函数对指定序列做映射
   - 返回iterator

```python
def square(x):
    return x ** 2

map(square, [1, 2, 3, 4, 5])
map(lambda x: x ** 2, [1, 2, 3, 4, 5])
map(lambda x, y: x + y, [1, 3, 5, 7, 9], [2, 4, 6, 8, 10])
```

- `filter()`
   - 过滤序列, 返回一个迭代器对象

```python
def is_odd(n):
   return n % 2 == 1

tmplist = filter(is_odd, [1, 2, 3, 4, 5, 6, 7, 8, 9])
print(list(tmplist))
```

- `enumerate()`
   - 将一个可遍历的数据对象(list, tuple, str)组合为一个索引序列, 同时列出数据和数据索引

```python
seq = ["one", "two", "three"]
for i, element in enumerate(seq):
    print(i, seq[i])
````

- `isinstance()`
    - 判断一个对象是否是已知的类型
    - `type()`: 不会认为子类是一种父类类型, 不考虑继承关系; 
    - `isinstance()`: 会认为子类是一种父类类型, 考虑继承关系; 

```python
a = 2
b = "two"
isinstance(a, int)
isinstance(a, [str, int, list])
```

# python collections 库

```python
from collections import OrderedDict, Counter

# Remembers the order the keys are added
x = OrderedDict(a = 1, b = 2, c = 3)

# Counts the frequency of each character
y = Counter("Hello World!")
```

# Python 单例模式

> Singleton Pattern

- 单例模式介绍
    - 单例模式(Singleton Pattern): 主要目的是确保某一个类(class)中只有一个实例(instance)存在, 从而避免浪费内存资源; 
- Python实现单例模式的几种方式
    - 使用模块
        - Python 的模块就是天然的单例模式, 因为模块第一次导入时, 会生成 `.pyc` 文件, 
          当第二次导入时, 就会直接加载 `.pyc` 文件, 而不会再次执行模块代码, 
          因此, 只需要把相关的函数和数据定义在一个模块中, 就可以获得一个单例对象. 
    - 使用装饰器
    - 实用类
    - 基于 `__new__` 方法
    - 基于 `metaclass` 方式实现
        - 相关知识
        - 实现单例模式

# Python 深拷贝和浅拷贝 

* https://docs.python.org/3.6/library/copy.html

* 直接赋值
   - 原始对象的引用, 别名. 赋值后的对象id相同, 对象类型相同, 对象值相同; 
   - 修改原对象, 赋值对象也会改变; 修改赋值对象, 原对象也会改变; 
      - 修改原对象中的可变元素, 赋值对象也会改变; 
      - 修改赋值对象中的可变元素, 原对象也会改变; 
* 浅拷贝(shadow copy)
   - 拷贝父对象, 不会拷贝对象内部的子对象. 浅拷贝后的对象id不同, 对象类型相同, 对象值相同; 
   - 修改原对象(可变), 浅拷贝对象不会改变; 修改浅拷贝对象(可变), 原对象也不会变; 
      - 修改原对象中的可变元素, 浅拷贝对象会改变; 
      - 修改浅拷贝对象中的可变元素, 原对象也会变; 
   - 仅仅复制了容器中元素的地址; 
* 深拷贝(deepcopy)
   - 完全拷贝了父对象及其子对象(副本). 深拷贝后的对象id不同, 对象类型相同, 对象值相同; 
   - 修改原对象(可变), 深拷贝对象不改变; 修改深拷贝对象(可变), 原对象也不会变; 
      - 修改原对象中的可变元素, 深拷贝对象不会改变; 
      - 修改深拷贝对象中的可变元素, 原对象也不会会变; 
   - 完全拷贝了一个副本, 容器内部元素和地址都不一样; 

```python
import copy

a = [1, 2, 3, 4, ['a', 'b'], 'strings'] # 原始对象
b = a 						 # 赋值, 传递对象的引用
c = copy.copy(a) 			 # 对象拷贝, 浅拷贝
d = copy.deepcopy(a) 		 # 对象拷贝, 深拷贝

print(id(a))
print(id(b))
print(id(c))
print(id(d))

a.append(5)					 # 修改对象a
a[4].append('c')			 # 修改对象a中的可变对象list
a[5].upper()				 # 修改对象a中的不可变对象str

b.append(5)					 # 修改对象b
b[4].append('c')			 # 修改对象b中的可变对象list
b[5].upper()				 # 修改对象b中的不可变对象str

c.append(5)					 # 修改对象c
c[4].append('c')			 # 修改对象c中的可变对象list
c[5].upper()				 # 修改对象c中的不可变对象str

d.append(5)					 # 修改对象d
d[4].append('c')			 # 修改对象d中的可变对象list
d[5].upper()				 # 修改对象d中的不可变对象str

print('a = ', a)
print('b = ', b)
print('c = ', c)
print('d = ', d)
```

## 浅拷贝

在Python中标识一个对象的唯一身份是: 对象的 `id, id(object)` (内存地址)、对象类型、对象值. 
浅拷贝就是创建一个具有相同类型、相同值但不同`id` 的新对象

对可变对象, 对象的值一样可能包含有对其他对象的引用, 浅拷贝产生的新对象, 虽然具有完全不同的 `id`, 
但是其值若包含可变对象, 这些对象和原始对象中的值包含同样的引用

可见浅拷贝产生的新对象中可变对象的值发生改变时, 会对原对象的值产生副作用, 因为这些值是同一个引用

浅拷贝仅仅对对象自身创建了一份拷贝, 而没有再进一步处理对象中包含的值, 
因此使用浅拷贝的典型使用场景是:  **对象自身发生改变的同时需要保持对象中的值完全相同** , 
比如: list 排序 

## 深拷贝

深拷贝不仅拷贝了原始对象的地址, 也对其包含的值进行拷贝. 他会递归的查找对象中包含的其他对象的引用, 
来完成更深层次拷贝. 因此, 深拷贝产生的副本可以随意修改而不需要担心会引起原始值的改变

值的注意的是, 深拷贝并非完完全全递归查找所有对象, 因为一旦对象引用了自身, 
完全递归可能会导致无限循环. 一个对象被拷贝了, python会对该对象做个标记, 
如果还有其他需要拷贝的对象引用着该对象, 他们的拷贝其实指的是同一份拷贝

使用 `__copy__` 和 `__deepcopy` 可以完成对一个对象拷贝的定制

# Python 序列反转

- 将序列(有序)中的元素位置反转
    - list
    - tuple
    - string

```python
L = [1, 2, 3, 4]
T = (1, 2, 3, 4)
S = "Python"
```

```python
# List
L_reversed_v1 = L[::-1]
L_reversed_v2 = reversed(L)
print(L_reversed_v1, list(L_reversed_v2))
```

```python
# Tuple
T_reversed_v1 = T[::-1]
T_reversed_v2 = reversed(T)
print(T_reversed_v1, tuple(T_reversed_v2))
```

```python
# String
S_reversed_v1 = S[::-1]
S_reversed_v2 = "".join(reversed(S))
print(S_reversed_v1, S_reversed_v2)
```

# Python 字典反转

字典的 key 和 value 对换

## 压缩器

```python
D = {
   'a': 1,
   'b': 2,
   'c': 3,
   'd': 4
}

print(D.items())
print(D.keys())
print(D.values())
D_ziped = zip(D.values(), D.keys())
print(list(D_ziped))

D_reversed_v1 = dict(zip(D.values(), D.keys()))
print(D_reversed_v1)
```

## 字典解析

```python
D_reversed_v2 = {v: k for k, v in D.items()}
print(D_reversed_v2)
```


# Python求众数

- np.argmax(numpy.bincount()) : 只在在非负数集上有效
- scipy.stats.mode()

```python
import numpy as np

np.random.seed(123)
nums = np.random.randint(low = 1, high = 10, size = 10)
```

## numpy.argmax(np.bincount())

```python
nums_mode_v1 = np.argmax(np.bincount(nums))
print(nums_mode_v1)
```

## scipy.stats

```python
from scipy import stats
nums_mode_v2 = stats.mode(nums)[0][0]
print(nums_mode_v2)
```

# Panda 和 Numpy 连接

> Pandas DataFrame 合并,连接 & Numpy ndarray 合并, 连接

* DataFrame
   - merge
   - join
   - concat
* ndarray
   - concatenate
   - vstack
   - row_stack
   - hstack
   - column_stack
   - dstack
   - split
   - hsplit
   - vsplit
   - dsplit
   - r\_
   - c\_

# 字符串拼接

- +=
- %
- .format()
- "".join()

## +=

```python
pieces = ['Today', 'is', 'really', 'a', 'nice', 'day', '!']
```

```python
BigString = ''
for s in pieces:
    BigString += s + ' '
print(BigString)
```

## % / .format()

```python
S1 = '%s, Your current money is %.1f' % ('Nupta', 500.52)
S2 = '{}, Your current money is {:.1f}'.format('Nupta', 500.52)
print(S1)
print(S2)
```

## "".join()

```python
S = " ".join(pieces)
print(S)
```

# 随机数生成 (函数参数)

```python
np.random.seed(123)                    # 设置随机数
np.random.permutation(np.arange(16))   # 返回一个序列的随机排列或返回一个随机排列的范围
np.random.shuffle(np.arange(5))        # 对一个序列就地随机排列
np.random.rand((5)                     # 生成均匀分布随机数[0, 1)
np.random.uniform(10)                  # 均匀分布(0, 1)
np.random.randint()                    # 从给定范围内随机取整数

正态分布
np.random.normal(loc = 0, scale = 1, size = (6))
np.random.normal(size = (5))
# from random import normalvariate
# normalvariate(, )
np.random.randn()                      # 标准正态分布
np.random.normal(loc = 0, scale = 1, size = (6))

其他分布
np.random.binomial(5)
np.random.beta(5)
np.random.chisquare(5)
np.random.gamma(5)
```

# Python-Numpy

* `.ravel()`, `.flatten()`, `.flat()`
   - 功能: 将多维array降为一维array
   - 拷贝(copy): 对拷贝所做的修改不会影响原始数组
   - 视图(view): 对视图所做的修改会影响原始数组
- `numpy.ravel(a, order = {'C', 'F', 'A', 'K'})`
   - 返回copy
   - `order = 'C'`: 按行索引排列
   - `order = 'F'`: 按列索引排列
   - `order = 'A'`:
   - `order = 'K'`:
- `numpy.ndarray.flatten()`
   - 返回 view
   - `numpy.flat`
   - `order = 'C'`: 按行索引排列
   - `order = 'F'`: 按列索引排列
   - `order = 'A'`:
   - `order = 'K'`:
- `numpy.flatiter`
   - `.flat`, 返回iterator
- `numpy.reshape()` or `numpy.ndarray.reshape()`
   - 返回 view
   - `order = 'C'`: 按行索引排列
   - `order = 'F'`: 按行索引排列
   - `order = 'A'`

```python
# np.ravel()
import numpy as np

x = np.array([[1, 2, 3], 
             [4, 5, 6]])
np.ravel(x)
np.ravel(x, order = 'C')
np.ravel(x, order = 'K')
np.ravel(x.T, order = 'F')
x.reshape(-1)

np.ravel(x, order = 'F')
np.ravel(x, order = 'A')
np.ravel(x.T, order = 'C')
```

```python
# .flatten()
import numpy as np

x = np.arange(1, 7).reshape(2, 3)
x.flatten()
x.flatten(order = 'F')
```

```python
# np.flat
x = np.arange(6).reshape(2, 3)
f1 = x.flat
type(f1)

for item in f1:
   print(item)

f1[2:4]
f1 = 3
f1[[1, 4]] = 1

f1.base
f1.coords
f1.index
f1.next()
f1.copy()
```

# 解析字符串

```python
import json

string = '[{"requestid": "b9a9b0f264a44ec28f7d20ed4826c691", "content": "前台Ennma 退房很快"阮冬琴开发票也很快"两个人配合不错"}]'

# 正则表达式
# res = "".join([i for i in string if u'\u4e00' <= i <= u'\u9fff'])
# print(res)


# Json解析
while True:
    try:
        res = json.loads(string)[0]["content"]
        print(res)
        break
    except json.JSONDecodeError as e:
        str_index = int(str(e).split(" ")[-1][:-1])
        string = string[:str_index - 1] + string[str_index:]
```

