---
title: Python Class & OOP
author: 王哲峰
date: '2023-01-09'
slug: python-class
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
</style>

<details><summary>目录</summary><p>

- [Python OOP](#python-oop)
  - [为何使用类?](#为何使用类)
  - [类属性继承搜索](#类属性继承搜索)
  - [类和实例](#类和实例)
  - [类方法调用](#类方法调用)
  - [编写类树](#编写类树)
  - [OOP 是为了代码重用](#oop-是为了代码重用)
- [类产生多个实例对象](#类产生多个实例对象)
  - [类对象提供默认行为](#类对象提供默认行为)
  - [实例对象是具体元素](#实例对象是具体元素)
- [类通过继承进行定制](#类通过继承进行定制)
- [运算符重载](#运算符重载)
  - [构造函数和表达式](#构造函数和表达式)
  - [常见的运算符重载方法](#常见的运算符重载方法)
  - [索引和分片](#索引和分片)
  - [索引迭代](#索引迭代)
  - [迭代器对象](#迭代器对象)
    - [用户定义的迭代器](#用户定义的迭代器)
    - [有多个迭代器的对象](#有多个迭代器的对象)
  - [成员关系](#成员关系)
  - [属性引用](#属性引用)
  - [返回字符串表达形式](#返回字符串表达形式)
  - [右侧加法和原处加法](#右侧加法和原处加法)
  - [Call 表达式](#call-表达式)
  - [比较](#比较)
  - [布尔测试](#布尔测试)
  - [对象析构函数](#对象析构函数)
- [类与字典的关系](#类与字典的关系)
- [实例](#实例)
  - [步骤 1: 创建实例](#步骤-1-创建实例)
    - [编写构造函数](#编写构造函数)
    - [以两种方法使用代码](#以两种方法使用代码)
  - [步骤 2: 添加行为方法](#步骤-2-添加行为方法)
  - [步骤 3: 运算符重载](#步骤-3-运算符重载)
    - [提供打印显示](#提供打印显示)
  - [步骤 4: 通过子类定制行为](#步骤-4-通过子类定制行为)
    - [编写子类](#编写子类)
    - [扩展方法](#扩展方法)
    - [多态的作用](#多态的作用)
    - [继承、定制和扩展](#继承定制和扩展)
    - [OOP: 大思路](#oop-大思路)
  - [步骤 5: 定制构造函数](#步骤-5-定制构造函数)
    - [OOP 比我们认为的要简单](#oop-比我们认为的要简单)
    - [组合类的其他方式](#组合类的其他方式)
  - [步骤 6: 使用内省工具](#步骤-6-使用内省工具)
  - [步骤 7: 把对象存储在数据库中](#步骤-7-把对象存储在数据库中)
    - [Pickle 和 Shelve](#pickle-和-shelve)
    - [在 shelve 数据库中存储对象](#在-shelve-数据库中存储对象)
    - [交互地探索 shelve](#交互地探索-shelve)
    - [更新 shelve 中的对象](#更新-shelve-中的对象)
- [类的设计](#类的设计)
  - [class 语句](#class-语句)
    - [class 语句的一般形式](#class-语句的一般形式)
    - [示例](#示例)
  - [方法](#方法)
    - [方法](#方法-1)
    - [self 参数](#self-参数)
    - [示例](#示例-1)
    - [调用超类构造函数](#调用超类构造函数)
    - [其他方法调用的可能性](#其他方法调用的可能性)
  - [继承](#继承)
    - [属性树的构造](#属性树的构造)
    - [继承方法的专有性](#继承方法的专有性)
    - [类接口技术](#类接口技术)
    - [抽象超类](#抽象超类)
  - [命名空间: 完整的内容总结](#命名空间-完整的内容总结)
    - [简单变量名: 如果赋值就不是全局变量](#简单变量名-如果赋值就不是全局变量)
    - [属性名称: 对象命名空间](#属性名称-对象命名空间)
    - [赋值将变量名分类](#赋值将变量名分类)
    - [命名空间字典](#命名空间字典)
    - [命名空间链接](#命名空间链接)
  - [类的文档字符串](#类的文档字符串)
  - [类的设计](#类的设计-1)
    - [Python 和 OOP](#python-和-oop)
    - [OOP 和 继承: “是一个”关系](#oop-和-继承-是一个关系)
    - [OOP 和组合: “有一个”关系](#oop-和组合-有一个关系)
    - [OOP 和委托: “包装”对象](#oop-和委托-包装对象)
- [类的高级主题](#类的高级主题)
  - [与设计相关的其他话题](#与设计相关的其他话题)
- [其他实例](#其他实例)
</p></details><p></p>


# Python OOP

- Python OOP
    - OOP 提供了一种不同寻常而往往更有效的检查程序的方式, 利用这种设计方法, 
      我们分解代码, 把代码的冗余度将至最低, 并且通过定制现有的代码来编写新的程序, 
      而不是在原处进行修改
    - OOP 不仅仅是一门技术, 更是一种经验
- Python class
    - 类是 Python 面向对象程序设计 (OOP) 的主要工具
    - 类是在 Python 实现支持 **继承** 的新种类的内部对象的部件, 这是一种代码定制和复用的机制
    - 类是属性和方法的 **组合**, 类就是一些函数的包, 这些函数大量使用并处理内置对象类型
    - 类的设计是为了创建和管理新的对象
- 在 Python 中, OOP 完全是可选的, 并且在初学阶段不需要使用类. 
    - 实际上, 可以用比较简单的结构, 例如函数, 甚至简单顶层脚本代码, 这样就可以做很多事. 
      因为妥善使用类需要一些预先的规则. 因此和那些采用战术模式工作的人相比(时间有限), 
      采用战略模式工作的人(做长期产品开发)对类会更感兴趣一些
- 类是 Python 所能提供的最有用的工具之一
    - 合理使用时, 类实际上可以大量减少开发的时间, 类也在流行的 Python 工具中使用

## 为何使用类? 

程序就是“用一些东西来做事”, 简而言之, 类就是一种定义新种类的东西的方式, 它反映了在程序领域中的真实对象

类是 Python 的程序组成单元, 就像函数和模块一样: 类是封装逻辑和数据的另一种方式. 
实际上类也定义了新的命名空间, 在很大程序上就像模块. 但是, 类有三个重要的独到之处, 
使其在建立对象时更为有用: 

- 多重实例
    - 类基本上就是产生对象的工厂, 每次调用一个类, 就会产生一个有独立命名空间的新对象. 
      每个由类产生的对象都能够读取类的属性, 并获得自己的命名空间来存储数据, 这些数据对于每个对象都不同. 
- 通过继承进行定制(命名空间继承)
    - 类可以建立命名空间的层次结构, 而这种层次结构可以定义该结构中类创建的对象所使用的变量名
- 运算符重载
    - 通过提供特定的协议方法, 类可以定义对象来响应在内置类型上的几种运算
    - Python 提供了一些可以有类使用的钩子, 从而能够中断并实现任何的内置类型运算
    - 例如: 通过类创建的对象可以进行切片、级联、索引等


## 类属性继承搜索

Python 中大多数 OOP 的故事, 都可以简化成这个表达式: 

- `object.attribute`
    - 读取模块的 **属性**
    - 调用对象的 **方法**
- 即: **找出 attribute 首次出现的地方, 先搜索 object, 然后是该对象之上的所有类, 由下至上, 由左至右**

在 Python 对象模型中, 类和通过类产生的实例是两种不同的对象类型: 

- 类: 超类、子类
    - 类是实例的工厂
    - 类的属性提供了行为(数据以及函数), 所有从类产生的示例都继承该类的属性
- 实例
    - 实例代表程序中具体的元素
    - 实例属性记录数据, 而每个特定对象的数据都不同

实例从它的类继承属性, 而类是从搜索树中所有比它更上层的类中继承属性: 

- 树中位置较高的类称为超类(superclass)
- 树中位置较低的类称为子类
- 超类提供了所有子类共享的行为, 但是因为搜索时由下至上, 子类可能会在树中较低位置重新定义超类的变量名, 从而覆盖超类定义的行为

> 读取属性只是简单地搜索“树”而已, 称这种搜索程序为继承, 因为树中位置较低的对象继承了树中位置较高的对象拥有的属性. 
从下至上进行搜索时, 连接至树中的对象就是树中所有上层对象所定义的所有属性的集合体, 直到树的最顶端. 

## 类和实例

- 类和实例的主要差异在于, 类是产生实例的工厂
- 从操作的角度来说, 类通常都有函数, 而实例有其他基本的数据项, 类的函数中使用了这些数据
- 在 OOP 中, 实例就像是带有数据的记录, 而类是处理这些记录的程序, 不过, 在 OOP 中, 还有继承层次的概念, 和之前的模型相比, 更好地支持了软件定制. 


## 类方法调用

每当我们调用附属于类的函数时, 总会隐含着这个类的实例. 这个隐含的主体或环境就是称之为面向对象模型的一部分原因: 当运算执行时, 总是有个主体对象. 

- Python 把隐含的实例传进方法中的第一个特殊参数, 习惯上将其称为 `self`
- 方法能通过实例或类进行调用, 这两种形式在脚本中都有各自的用途
    - 实例调用: `instance.method()`
    - 类调用: `Class.method(instance)`

## 编写类树

**类树: **

- 以 class 语句和类调用来构造一些树和对象: 
    - 每个 class 语句会生成一个新的类对象
    - 每次类调用时, 就会生成一个新的实例对象
    - 实例自动连接至创建了这些实例的类
    - 类连接至超类的方式是, 将超类列在类头部的括号内. 从左至右的顺序会决定树中的次序
        - 多重继承: 在类树中, 类有一个以上的超类

```python
class C2: ...
class C3: ...
class C1(C2, C3): ...

I1 = C1()
I2 = C1()
```

**属性: **

- 属性通常是在 class 语句中通过赋值语句添加在类中的, 而不是嵌入在函数的 def 语句内
- 属性通常是在类内, 对传给函数的特殊参数(也就是 self), 做赋值运算而添加在实例中的

**方法: **

- 当 def 出现在类的内部时, 通常称为方法, 而且会自动接收第一个特殊参数(通常称为 self), 
  这个参数提供了被处理的实例的参照值
- Python 中的 self 一定是明确写出的, 这样使属性的读取更为明显

**构造函数: **

- 类和实例属性并没有事先声明, 而是在首次赋值时它的值才会存在, 当方法对 self 属性进行赋值时, 
  会创建或修改类树底端实例内的属性, 因为 self 自动引用正在处理的实例
- 写好并继承后, 每次从类产生实例时, Pyton 会自动调用名为 `__init__` 的方法. 
  新实例会如往常那样传入 `__init__` 的 self 参数而列在类调用小括号内的任何值会成为第二以及其后的参数, 
  其效果就是在创建实例时初始化了这个实例, 而不需要额外的方法调用


## OOP 是为了代码重用

- OOP 就是在树中搜索属性:
    - 类其实就是由函数和其他变量名所构成的包, 很像模块, 然而, 我们从类得到的自动属性继承搜索, 支持了软件的高层次的定制, 而这是我们通过模块和函数做不到的
    - 类提供了自然的结构, 让代码可以把逻辑和变量名区域化, 这样有助于程序的调试
    - 可以对类树中任何类创建实例, 而不是只针对底端的类, 创建的示例所用的类会决定其属性搜索从哪个层次开始

# 类产生多个实例对象

类对象、实例对象: 

- 类对象: 提供默认行为
- 实例对象: 是程序处理的实际对象, 各自都有独立的命名空间, 但是继承(可自动存取)创建该实例的类中的变量名
- 类来自于语句, 而实例来自于调用

## 类对象提供默认行为


## 实例对象是具体元素

# 类通过继承进行定制


# 运算符重载

运算符重载只是意味着在类方法中拦截内置的操作, 当类的实例出现内置操作中, Python 自动调用方法, 
并且方法的返回值变成了相应操作的结果. 

运算符重载就是让用类写成的对象, 可以截获并响应用在内置类型上的运算: 加法、切片、打印和点号运算.

- 运算符重载让类拦截常规的 Python 运算
- 类可重载所有 Python 表达式运算符
- 类也可重新打印、函数调用、属性点号运算等内置运算
- 重载使类实例的行为像内置类型
- 重载是通过提供特殊名称的类方法来实现
- 以双下划线命名的方法(`__X__`)是特殊的钩子
    - Python 运算符重载的实现是提供了特殊命名的方法来拦截运算
    - Python 语言替每种运算和特殊命名的方法之间定义了固定不变的映射关系
- 当实例出现在内置运算时, 这类方法会自动调用
- 运算符覆盖方法没有默认值, 而且也不需要
    - 如果类没有定义或继承运算符重载方法, 就是说相应的运算在类实例中并不支持, 
      例如, 如果没有 `__add__`, `+` 表达式就会引发异常
- 运算符可让类与 Python 的对象模型相集成
    - 重载类型运算时, 以类实现的用户定义对象的行为就会像内置对象一样

## 构造函数和表达式

- `__init__` 方法, 也称为构造函数方法, 它是用于初始化对象的状态的.
- `__init__` 和 `self` 参数是了解 Python 的 OOP 程序的关键之一.


**示例 1: **

    ```python
    
        # number.py
        class Number:

            def __init__(self, start):
                self.data = start
            
            def __sub__(self, other):
                return Number(self.data - other)

        >>> from number import Number
        >>> X = Number(5)
        >>> Y = X - 2
        >>> Y.data
```

**示例 2: 构造函数参数使用方法**

    ```python

        class Person_v1(object):

            def __init__(self, name, gender, **kw):
                self.name = name
                self.gender = gender
                for key, value in kw.items():
                    setattr(self, key, value)


        class Person_v2(object):

            def __init__(self, name, gender, **kw):
                self.name = name
                self.gender = gender
                self.__dict__.update(kw)

        p1 = Person_v1("wangzf", "male", age = 18, course = "Python")
        p2 = Person_v2("wangzf", "male", age = 18, course = "Python")

        print(p1.age)
        print(p1.course)

        print(p2.age)
        print(p2.course)
```

## 常见的运算符重载方法

在类中, 对内置对象所能做的事, 几乎都有相应的特殊名称的重载方法:

- `__init__`
    - 重载: 构造函数
    - 调用: 对象建立
- `__del__`
    - 重载: 析构函数
    - 调用: X 对象回收
- `__add__`
    - 重载: 运算符 + 
    - 调用
- `__or__`
    - 重载
    - 调用
- `__repr__`, `__str__`
    - 重载
    - 调用
- `__call__`
    - 重载
    - 调用
- `__getattr__`
    - 重载
    - 调用
- `__setattr__`
    - 重载
    - 调用
- `__delattr__`
    - 重载
    - 调用
- `__getattribute__`
    - 重载
    - 调用
- `__getitem__`
    - 重载
    - 调用
- `__setitem__`
    - 重载
    - 调用
- `__delitem__`
    - 重载
    - 调用
- `__len__`
    - 重载
    - 调用
- `__bool__`
    - 重载
    - 调用
- `__lt__`, `__gt__`
    - 重载
    - 调用
- `__le__`, `__ge__`
    - 重载
    - 调用
- `__eq__`, `__ne__`
    - 重载
    - 调用
- `__radd__`
    - 重载
    - 调用
- `__iadd__`
    - 重载
    - 调用
- `__iter__`, `__next__`
    - 重载
    - 调用
- `__contains__`
    - 重载
    - 调用
- `__inddx__`
    - 重载
    - 调用
- `__enter__`, `__exit__`
    - 重载
    - 调用
- `__get__`, `__set__`
    - 重载
    - 调用
- `__delete__`
    - 重载
    - 调用
- `__new__`
    - 重载
    - 调用
- `__format__`
    - 重载
    - 调用
- `__dict__`
    - 重载
    - 调用
- `__slots__`
    - 重载
    - 调用
- `__class__`
    - 重载
    - 调用
- `__bases__`
    - 重载
    - 调用
- `__name__`
    - 重载
    - 调用
- `__main__`
    - 重载
    - 调用

所有重载方法的名称前后都有两个下划线字符, 以便把同类中定义的变量名区别开来. 
特殊方法名称和表达式或运算的映射关系, 是由 Python 语言预先定义好的(在标准语言手册中有说明). 

运算符重载方法也都是可选的, 如果没有编写或继承一个方法, 类直接不支持这些运算, 并且试图使用它们会引发一个异常. 


## 索引和分片

- 索引
    - 如果在类中定义或继承了的话, 则对于实例的索引运算, 会自动调用 `__getitem__`. 
      当实例 X 出现在 X[i] 这样的索引运算中时, Python 会调用这个实例继承的 __getitem__ 方法, 
      把 X 作为第一个参数传递, 并且括号内的索引值传递给第二个参数. 

```python

class Indexer:
    
    def __getitem__(self, index):
        return index ** 2

>>> X = Indexer()
>>> X[2]
>>> for i in range(5):
>>>     print(X[i], end = " ")
```

- 切片
    - 除了索引, 对于分片表达式也调用 __getitem__, 内置类型以同样的方式处理分片
    - 切片中的分片边界绑定到了一个分片对象中, 并且传递给索引的列表实现
    - 总可以手动地传递一个分片对象
        - 分片语法主要是用一个分片对象进行索引的语法糖

    ```python
    >>> L = [5, 6, 7, 8, 9]

    # 内置分片运算
    >>> L[2:4]
    >>> L[1:]
    >>> L[:-1]
    >>> L[::2]

    # 分片对象
    >>> L[slice(2, 4)]
    >>> L[slice(1, None)]
    >>> L[slice(None, -1)]
    >>> L[slice(None, None, 2)]
    ```

    - 对于带有一个 __getitem__ 的类, 该方法将即针对基本索引(带有一个索引)调用, 又针对分片(带有一个分片对象)调用
        - 当针对分片调用的时候, 方法接收一个分片对象, 它在一个新的索引表达式中直接传递给嵌套的列表索引


    ```python
    class Indexer:

        data = [5, 6, 7, 8, 9]
        
        def __getitem__(self, index):
            print("getitem:", index)
            return self.data[index]

    >>> X = Indexer()
    >>> X[0]
    >>> X[1]
    >>> X[-1]
    >>> X[2:4]
    >>> X[1:]
    >>> X[:-1]
    >>> X[::2]
    ```

    - 如果使用的话, __setitem__ 索引赋值方法类似地拦截索引和分片赋值, 它为后者接收了一个分片对象, 他可能以同样的方式传递到另一个索引赋值中

    ```python
    def __setitem__(self, index, value):
        ...
        self.data[index] = value
    ```

## 索引迭代

- for 语句的作用是从 0 到更大的索引值, 重复对序列进行索引运算, 直到检测到超出边界的异常. 
- __getitem__ 也可以是 Python 中一种重载迭代的方式, 如果定义了这个方法, 
  for 循环每次循环时都会调用类的 __getitem__, 并持续搭配有更高的偏移值. 
  这是买一送一的情况: 任何会响应索引运算的内置或用户定义的对象, 同样会响应迭代. 

## 迭代器对象

尽管 __getitem__ 技术有效, 但它真的只是迭代的一种退而求其次的方法. 
如今, Python 中的所有的迭代环境都会先尝试 __iter__ 方法, 再尝试 __getitem__.
也就是说, 它们宁愿使用迭代协议, 然后才是重复对对象进行索引运算. 只有在对象不支持迭代协议
的时候, 才会尝试索引运算. 一般来讲, 你也应该优先使用 __iter__, 
它能够比 __getitem__ 更好地支持一般的迭代环境

从技术角度来讲, 迭代环境是通过调用内置的 iter 去尝试寻找 __iter__ 方法来实现的, 而这种方法
应该返回一个迭代器对象. 如果已经提供了, Python 就会重复调用这个迭代器对象的 next 方法, 直到发生 
StopIteration 异常. 如果没有找到这类 __iter__ 方法, Python 会改用 __getitem__ 机制, 就像之前说的那样
通过偏移量重复索引, 直到引发 IndexError 异常(对于手动迭代来说, 一个 next 内置函数也可以很方便地使用: 
next(I) 与 I.__next__() 是相同的). 


### 用户定义的迭代器

在 __iter__ 机制中, 类就是通过实现迭代器协议来实现用户定义的迭代器的.

```python
# iters.py file

class Squares:

    def __init__(self, start, stop):
        self.value = start - 1
        self.stop = stop

    def __iter__(self):
        return self 
    
    def __next__(self):
        if self.value == self.stop:
            raise StopIteration
        self.value += 1
        return self.value ** 2

>>> from iters import Squares
>>> for i in Squares(1, 5):
>>>     print(i, end = " ")

>>> X = Squares(1, 5) # iterate manually: what loops do
>>> I = iter(X)       # iter calls __iter__
>>> next(I)           # next calls __next__
>>> next(I)
>>> next(I)
```

### 有多个迭代器的对象


## 成员关系

## 属性引用


## 返回字符串表达形式

- `__repr__()` 和 `__str()`
   - 重新定义实例的 `__repr__()` 和 `__str__()` 方法可以改变对象实例的打印或显示输出, 
     让它们更具可读性 `__repr__()` 方法返回一个实例的代码表示形式, 通常用来重新构造这个实例, 
     内置的 `repr()` 函数返回这个字符串, 跟使用交互式解释器显示的值是一样的 `__str__()` 
     方法将实例转换为一个字符串, 使用 `str()` 或 `print()` 函数会输出这个字符串

示例: 

```python
class Pair:
    def __init__(self, x, y):
        self.x = x
        self.y = y
    def __repr__(self):
        return "Pair({0.x!r}, {0.y!r})".format(self) 
        # "Pair({%r}, {%r})".format(self.x, self.y)
    def __str__(self):
        return "({0.x!s}, {0.y!s})".format(self)
        # "Pair({%s}, {%s})".format(self.x, self.y)

>>> p = Pair(3, 4)
>>> p
>>> # Pair(3, 4)
>>> print(p)
>>> # (3, 4)
```


## 右侧加法和原处加法


## Call 表达式

## 比较

## 布尔测试

类可能也定义了赋予其实例布尔特性的方法. 在布尔环境中, Python 首先尝试 `__bool__` 来获取一个直接的布尔值, 
然后, 如果没有该方法, 就尝试 `__len__` 类根据对象的长度确定一个真值. 通常首先使用对象状态或其他信息来生成
一个布尔结果. 

示例 1: 

```python

# class 1
class Truth:

    def __bool__(self):
        return True

>>> X = Truth()
>>> if X: 
>>>     print("yes!")

# class 2
class Truth:

    def __bool__(self):
        return False

>>> X = Truth()
>>> bool(X)
```

示例 2: 

```python

class Truth:
    def __len__(self):
        return 0

>>> X = Truth()
>>> if not X:
>>>    print("no!")
```

示例 3: 如果两个方法都有, Python 喜欢 __bool__ 胜过 __len__, 因为它更具体

```python

class Truth:

    def __bool__(self):
        return True
    
    def __len__(self):
        return 0

>>> X = Truth()
>>> if X:
>>>     print("yes!")
```

示例 4: 如果没有定义真的方法, 对象毫无疑义地看作真

```python

class Truth:
    pass

>>> X = Truth()
>>> bool(X)
```

## 对象析构函数

每当实例产生时, 就会调用 `__init__` 构造函数. 每当实例空间被回收时(在垃圾收集时), 
它的对立面 `__del__`, 也就是 **析构函数(destructor method)**, 就会自动执行. 

- 在 Python 中, 析构函数不像其他 OOP 语言那么常用: 
    - 原因之一就是, 因为 Python 在实例回收时, 会自动回收该实例所拥有的所有空间, 对于空间管理器来说, 是不需要析构函数的. 
    - 原因之二是: 无法轻易地预测实例何时回收, 通常最好是在有意调用的方法中编写代码终止活动, 比如: `try/finally`. 在某种情况下, 系统表中可能还在引用该对象, 使析构函数无法执行. 

示例: 

```python
class Life:

    def __init__(self, name = "unknown"):
        print("Hello", name)
        self.name = name
    
    def __del__(self):
        print("Goodbye", self.name)
    
brian = Life("Brian")
brian = "loretta"
```

# 类与字典的关系

类产生的基本继承模型其实非常简单: 所涉及的就是在连续的对象树中搜索属性, 实际上, 建立的类中可以什么东西都没有(空的命名空间对象). 

```python
class rec:
    pass
```

- 命名空间对象的属性通常都是以字典的形式实现的, 而类继承只是连接其他字典的字典而已
- 每个实例都有一个不同的属性字典, 实际上是不同的命名空间
    - `__dict__` 属性是针对大多数基于类的对象的命名空间字典, 一些类可能在 `__slots__` 中定义了属性
        - `class_name.__dict__.keys()`
        - `instance_name.__dict__.keys()`

基于字典的记录的示例:

```python
rec = {}
rec["name"] = "mel"
rec["age"] = 45
rec["job"] = "trainer/writer"
print(rec["name"])
```

基于类的记录的示例:

```python
class rec:
    pass

rec.name = "mel"
rec.age = 45
rec.job = "trainer/writer"
print(rec["name"])
```


实例都有一个不同的属性字典:

```python
class rec:
    pass

pers1 = rec()
pers1.name = "rel"
pers1.job = "trainer"
pers1.age = 40

pers2 = rec()
pers2.name = "vls"
pers2.job = "developer"

print(pers1.name)
print(pers2.name)
```

完整的类实现记录及其处理:

```python
class Person:
    def __init__(self, name, job):
        self.name = name
        self.job = job
    
    def info(self):
        return (self.name, self.job)

rec1 = Person("mel", "trainer")
rec2 = Person("vls", "developer")

print(rec1.job)
print(rec2.info())
```

# 实例

在这里, 我们将编写两个类: 

- Person —— 创建并处理关于人员的信息的一个类
- Manager —— 一个定制的 Person, 修改了继承的行为

在这个过程中, 将创建两个类的实例并测试它们的功能. 完成实例之后, 将给出实用类的一个漂亮的例子, 
把实例存储到一个 shelve 的面上对象数据库中, 使它们持久化. 通过这种方式, 可以把这些代码用作模板, 
从而发展为完全用 Python 编写的一个完备的个人数据库. 

最后, 这里创建的类在代码量上相对较小, 但是他们将演示 Python 的 OOP 模型的所有主要思想. 
不管其语法细节如何, Python 的类系统实际上很大程度上就是在一堆对象中查找属性, 并为函数给定一个特殊的第一个参数. 

## 步骤 1: 创建实例

- 在 Python 中, 模块名使用小写字母开头, 而类名使用一个大写字母开头, 这是通用的惯例
- 在 Python 中的单个模块文件中, 我们可以编写任意多个函数和类, 但是当模块拥有一个单一、一致的用途的时候, 它们会工作地更好

```python
# person.py
class Person:
    pass
```

### 编写构造函数

- 在 Python 的术语中, 字段叫做实例对象的属性, 并且它们通常通过给类方法函数中的 `self` 属性赋值来创建, 并且保存持久化
- 赋给实例属性第一个值的通常方法是在 `__init__` 构造函数方法中将它们赋给 `self`, 
  构造函数方法包含了每次创建一个示例的时候 Python 会自动运行的代码
- 在 OOP 的术语中, `self` 就是新创建的实例对象, 而 `name`、`job`、`pay` 变成了状态信息, 
  即保存在对象中供随后使用的描述性数据


```python
# Add record field initialization
class Person:

    def __init__(self, name, job, pay):
        self.name = name
        self.job = job
        self.pay = pay
```

参数名 `name`、`job`、`pay` 出现了两次:

- `name` 参数在 `__init__` 函数的作用域里是一个本地变量
- `self.name` 是实例的一个属性, 它暗示了方法调用的内容
- 上面这是两个不同的变量, 但恰好具有相同的名字, 可以对实例属性取其他的名字
- 可以给实例的属性 `self.name` 赋值为默认值 `None`, 即所创建的实例没有名字 `name`

```python
class Person:

    def __init__(self, name, job = None, pay = 0):
        self.name = name
        self.job = job
        self.pay = pay
```

### 以两种方法使用代码

- 模块文件底部运行测试语句时, 增加 `__name__` 检查模块可以实现在进行中测试: 
    - 文件作为顶层脚本运行的时候, 测试它, 因为其 `__name__` 是 `__main__`
    - 文件作为类库导入的时候, 则检查模块不运行

```python
# person.py

# class
class Person:

    def __init__(self, name, job = None, pay = 0):
        self.name = name
        self.job = job
        self.pay = pay

# 进行中测试代码, __name__ 检查模块
if __name__ == "__main__":
    # self-test code
    bob = Person("Bob Smith")
    sue = Person("Sue Jones", job = "dev", pay = 100000)
    print(bob.name, bob.pay)
    print(sue.name, sue.pay)
```

## 步骤 2: 添加行为方法

- 尽管类添加了结构的一个额外的层级, 它们最终还是通过嵌入和处理列表及字符串这样的基本 **核心数据类型** 来完成其大部分工作
- 类的实例是一个可修改的对象
- 代码的一般方法在实际中并非好办法, 在类之外的硬编码操作可能会导致未来的维护问题

    ```python

        # person.py
            
        # class
        class Person:

            def __init__(self, name, job = None, pay = 0):
                self.name = name
                self.job = job
                self.pay = pay
        
        
        if __name__ == "__main__":
            bob = Person("Bob Smith")
            sue = Person(name = "Sue Jones", job = "dev", pay = 100000)
            print(bob.name, bob.pay)
            print(sue.name, sue.pay)
            print(bob.name.split()[-1])
            sue.pay *= 1.10
            print(sue.pay)

编写方法
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

- 封装的思想就是把操作逻辑包装到界面之后, 这样每种操作在程序里只编码一次, 
  通过这种方式, 如果将来需要修改, 只需要修改一个版本. 
- 方法只是附加给类并旨在处理那些类的实例的常规函数. 
  示例是方法调用的主体, 并且会自动传递给方法的 self 参数

```python

# person.py
    
# class
class Person:

    def __init__(self, name, job = None, pay = 0):
        self.name = name
        self.job = job
        self.pay = pay
    
    def lastName(self):
        return self.name.split()[-1]
    
    def giveRaise(self, percent):
        self.pay = int(self.pay * (1 + percent))


if __name__ == "__main__":
    bob = Person("Bob Smith")
    sue = Person(name = "Sue Jones", job = "dev", pay = 100000)
    print(bob.name, bob.pay)
    print(sue.name, sue.pay)
    print(bob.lastName(), sue.lastName())
    sue.giveRaise(.10)
    print(sue.pay)
```


## 步骤 3: 运算符重载

运算符重载, 在一个类中编写这样的方法, 当方法在类的实例上运行的时候, 方法截获并处理内置的操作. 

常见运算符重载方法: 

- `__init__`: 构造函数方法, 在构建的时候自动运行, 以初始化一个新创建的实例
- `__str__`: 允许输入专门的操作, 提供专门的打印操作行为
- `__repr__`: 提供对象的一种代码低层级显示, 便于开发者看到额外的细节, 打印运行 `__str__`, 交互提示模式使用 `__repr__`

### 提供打印显示

`__str__` 方法的原理是, 每次一个实例转换为其打印字符的串的时候, `__str__` 都会自动运行. 
由于这就是打印一个对象所会做的事情, 所以直接的效果就是, 打印一个对象会显示对象的 `__str__` 
方法所返回的内容, 要么自己定义一个方法, 要么从一个超类继承一个该方法. 

```python
# person.py
    
# class
class Person:

    def __init__(self, name, job = None, pay = 0):
        self.name = name
        self.job = job
        self.pay = pay
    
    def lastName(self):
        return self.name.split()[-1]
    
    def giveRaise(self, percent):
        self.pay = int(self.pay * (1 + percent))
    
    def __str__(self):
        return "[Person: %s, %s]" % (self.name, self.pay)


if __name__ == "__main__":
    bob = Person("Bob Smith")
    sue = Person(name = "Sue Jones", job = "dev", pay = 100000)
    print(bob)
    print(sue)
    print(bob.lastName(), sue.lastName())
    sue.giveRaise(.10)
    print(sue)
```

## 步骤 4: 通过子类定制行为

要展示 OOP 的真正的能力, 我们需要定义一个超类/子类关系, 以允许我们扩展软件并替代一些继承的行为. 毕竟, 这是 OOP 背后的
主要思想; 基于已经完成的工作的定制来促进一种编码模式, 可以显著地缩减开发时间. 

### 编写子类


### 扩展方法

### 多态的作用


### 继承、定制和扩展


### OOP: 大思路


## 步骤 5: 定制构造函数

调用重定义的超类构造函数, 在 Python 中是一种很常见的编码模式. 

在构造的时候, Python 自身使用继承来查找并调用唯一的一个 `__init__` 方法, 
也就是类树中最低的一个. 如果需要在构造的时候运行更高的 `__init__` 方法, 
必须通过超类的名称调用它们. 

这种方法的积极之处在于, 你可以明确指出哪个参数传递给超类的构造函数, 
并且可以选择根本就不调用它: 不调用超类的构造函数允许你整个替代其逻辑, 而不是扩展它. 

```python
# person.py

# class
class Person:

    def __init__(self, name, job = None, pay = 0):
        self.name = name
        self.job = job
        self.pay = pay
    
    def lastName(self):
        return self.name.split()[-1]
    
    def giveRaise(self, percent):
        self.pay = int(self.pay * (1 + percent))
    
    def __str__(self):
        return "[Person: %s, %s]" % (self.name, self.pay)

class Manager(Person):
    def __init__(self, name, pay):
        Person().__init__(self, name, "mgr", pay)
    
    def giveRaise(self, name, "mgr", pay):
        Person.giveRaise(self, percent + bonus)



if __name__ == "__main__":
    bob = Person("Bob Smith")
    sue = Person(name = "Sue Jones", job = "dev", pay = 100000)
    print(bob)
    print(sue)
    print(bob.lastName(), sue.lastName())
    sue.giveRaise(.10)
    print(sue)
    tom = Manager("Tom Jones", 50000)
    tom.giveRaise(.10)
    print(tom.lastName())
    print(tom)
```

### OOP 比我们认为的要简单

在完整的形式中, 不管类的大小如何, 它捕获了 Python 的 OOP 机制中几乎所有重要的概念:

- 实例创建——填充实例属性
- 行为方法——在类方法中封装逻辑
- 运算符重载——为打印这样的内置操作提供行为
- 定制行为——重新定义子类中的方法以使其特殊化
- 定制构造函数——为超类步骤添加初始化逻辑

这些概念中的大多数都只是基于3个简单的思路: 

- 在对象树中继承查找属性
- 方法中特殊的 self 参数
- 运算符重载对方法的自动派发

通过这种方法, 我们可以使自己的代码在未来易于修改, 通过驾驭类的倾向以构造代码减少冗余. 

大体上, 这就是 Python 中的 OOP 的全部. 

### 组合类的其他方式

## 步骤 6: 使用内省工具


## 步骤 7: 把对象存储在数据库中

Python 对象持久化: 让对象在创建它们的程序退出后依然存在

### Pickle 和 Shelve

对象持久化通过 3 个标准库模块来实现, 这三个模块在 Python 都可用: 

- `pickle` 
    - 任意 Python 对象和字符串之间的序列化
- `dbm`
    - 实现一个可通过键访问的文件系统, 以存储字符串
- `shelve`
    - 使用另两个模块按照键把 Python 对象存储到一个文件中

### 在 shelve 数据库中存储对象

### 交互地探索 shelve

### 更新 shelve 中的对象

# 类的设计

## class 语句

- 类几乎就是命名空间, 也就是定义变量名的工具, 把数据和逻辑导出给客户端
- 在类或实例对象中找不到的所引用的属性, 就会从其他类中获取
- 怎样从 class 语句得到命名空间的呢? 
    - 就像函数一样, class 语句是本地作用域, 由内嵌的赋值语句建立的变量名, 就存在于这个本地作用域中
    - 就像模块内的变量名, 在 class 语句内赋值的变量名(非函数对象、函数对象)会变成类对象的属性: 
      当 Python 执行 class 语句时(不是调用类), 会从头至尾执行其主体内的所有语句, 在这个过程中, 
      进行的赋值运算会在这个类作用域中创建变量名, 从而成为对应的类对象内的属性
        - 把函数对象赋值给类属性, 就会产生 **实例方法**
        - 把简单的非函数的对象赋值给两类属性, 就会产生 **数据属性**, 由所有实例共享
            - 可以通过实例或类引用它
            - 可以通过类名称修改它

### class 语句的一般形式

```python
class Class_name(superclass, ...):
    data = value
    def method(self, ...):
        self.member = value
```


在 class 语句内, 任何赋值语句都会产生类属性, 而且还有特殊名称方法重载运算符

### 示例

示例 1

```python

class ShareData:
    spam = 42

x = ShareData()
y = ShareData()
print(x.spam)
print(y.spam)
print(ShareData.spam)

# 通过类修改修改了实例和类的数据
ShareData.spam = 99
print(x.spam, y.spam, ShareData.spam)

# 通过实例修改只能修改实例本身的数据
x.spam = 88
print(x.spam, y.spam, ShareData.spam)
```


- 通常情况下, 继承搜索只会在属性引用时发生, 而不是在赋值运算时发生
- 对对象属性进行赋值运算时总是会修改该对象, 除此之外没有其他的影响


示例 2

```python
class MixedNames:
    data = "spam"

    def __init__(self, value):
        self.data = value
    
    def display(self):
        print(self.data, MixedNames.data)

x = MixedNames(1)
y = MixedNames(2)
x.display()
y.display()
```

- 利用上面示例中的技术把属性存在不同对象内, 可以决定其可见范围: 
    - 附加在类上时, 变量名是共享的
    - 附加在实例上时, 变量名是属于每个实例的数据, 而不是共享的行为或数据

## 方法

### 方法
    
- 方法位于 class 语句的主体内, 是由 def 语句建立的函数对象. 从抽象的视角看, 方法替实例对象提供了要继承的行为. 
- 从程序设计的角度看, 方法的工作方式与简单函数完全一致, 只是有个重要差异: 类方法的第一个参数总是接收方法调用的隐性主体, 
也就是实例对象. 换句话说, Python 会自动把实例方法的调用对应到类方法函数:

```python
instance.method(args, ...)

class.method(instance, args, ...)
```

### self 参数

- 除了方法属性名称是正常的继承外, 第一个参数就是方法调用背后唯一的神奇之处. 
  在类方法中, 按惯例第一个参数通常都称为 self(严格地说, 只有其位置重要, 而不是它的名称). 
  这个参数给方法提供了一个钩子, 从而返回调用的主体, 也就是实例对象: 因为类可以产生许多实例对象, 
  所以需要这个参数来管理每个实例彼此各不相同的数据. 
- 让 self 明确化的本质是有意设计的: 这个变量名存在, 会让你明确脚本中使用的是实例属性名称, 而不是本地作用域中的变量名

### 示例

```python  
class NextClass:
    def printer(self, text):
        self.message = text
        print(self.message)

x = NextClass()
x.printer("instance call")
x.message
```

### 调用超类构造函数

由于所有属性 __init__ 方法是由继承进行查找的, 在构造时, Python 会找出并且只调用一个 __init__. 
如果要保证子类的构造函数也会执行超类构造时的逻辑, 一般都必须通过类明确地调用超类的 __init__ 方法. 

这种通过类调用方法的模式, 是扩展继承方法行为(而不是完全取代)的一般基础. 

```python
class Super:
    def __init__(self, x):
        ...default code...
    
class Sub(Super):
    def __init__(self, x, y):
        Super.__init__(self, x):
            ..custom code...

I = Sub(1, 2)
```

这是代码有可能直接调用运算符重载方法的环境之一. 如果真的想运行超类的构造方法, 
自然只能用这种方式进行调用: 没有这样的调用, 子类会完全取代超类的构造函数. 

### 其他方法调用的可能性

- 静态方法: 可以让编写不预期第一参数为实例对象的方法, 这类方法可像简单的无实例的函数那样运行, 
  其变量名属于其所在类的作用域, 并且可以用来管理类数据. 
- 类方法: 当调用的时候接受一个类而不是一个实例, 并且他可以用来管理基于每个类的数据, 这是高级的选用扩展功能. 
  通常来说, 一定要为方法传入实例, 无论通过实例还是类调用都行. 

## 继承

### 属性树的构造

- 实例属性
    - 由对方法内 self 属性进行赋值运算而生成的
- 类属性
    - 通过 class 语句内的赋值语句而生成的
        - 数据属性
        - 方法属性
- 超类的连接
    - 通过 class 语句首行的括号内列出的类而生成的

### 继承方法的专有性

继承树搜索模式变成了将系统专有化的最好方式, 因为继承会先在子类寻找变量名, 然后才查找超类, 
子类就可以对超类的属性重新定义来取代默认的行为. 实际上, 可以把整个系统做成类的层次, 再新增
外部的子类来对其进行扩展, 而不是在原处修改已经存在的逻辑. 

重新定义继承变量名的概念引出了各种专有化技术: 

- 子类可以完全取代继承的属性, 提供超类可以找到的属性, 并且通过已覆盖的方法回调超类来扩展超类的方法, 这种扩展编码模式常常用于构造函数. 
- ...

**示例: **

```python
class Super:

    def method(self):
        print("in Super.method")

class Sub(Super):
    
    def method(self):                   # override method
        print("starting Sub.method")    # add actions
        Super.method(self)              # run default action
        print("ending Sub.method")

>>> x = Super()
>>> x.method()

# in Super.method

>>> x = Sub()
>>> x.method()
# starting Sub.method
# in Super.method
# ending Sub.method
```


直接调用超类方法是重点. 

- Sub 类以其专有化的版本取代了 Super 的方法函数, 但是, 
    取代时 Sub 又回调了 Super 所导出的版本, 从而实现了默认的行为. 
    换句话说, Sub.method 只是扩展了 Super.method 的行为, 而不是完全取代了它. 


### 类接口技术


> 扩展只是一种与超类接口的方式

- **示例: **

```python
clas Super(object):

    def method(self):
        print("in Super.method")
    
    def delegate(self):
        self.action()
    
class Inheritor(Super):
    pass

class Replacer(Super):

    def method(self):
        print("in Replacer.method")

class Extender(Super):

    def method(self):
        print("starting Extender.method")
        Super.method(self)
        print("ending Extender.method")

class Provider(Super):
    def action(self):
        print("in Provider.action")
    
if __name__ == "__main__":
    for klass in (Inheritor, Replacer, Extender):
        print("\n" + klass.__name__ + "...")
        klass().method()
        print("\nProvider...")
        x = Provider()
        x.delegate()
```

- **示例分析: **
    - `Super`
        - 定义了一个 `method` 函数以及在子类中期待一个动作的 `delegate`
    - `Inheritor`
        - 没有提供任何新的变量名, 因此会获得 `Super` 中定义的一切内容
    - `Replacer`
        - 用自己的版本覆盖 `Super` 的 `method`
    - `Extender`
        - 实现 `Super` 的 `delegate` 方法预期的 `action` 方法

### 抽象超类

**抽象超类:** 类的部分行为默认是由其子类所提供的, 如果预期的方法没有在子类中定义, 
当继承搜索失败时, Python 会引发未定义变量名的异常. 

**Python3 抽象超类: ** 在一个 `class` 头部使用一个关键字参数, 以及特殊的 `@` 装饰器语法. 

类的编写者偶尔会使用 assert 语句, 使这种子类需求更加明显, 或者引发内置的异常 `NotImplementedError`: 

- version 1:

```python
class Super:
    
    def delegate(self):
        self.action()
    
    def action(self):
        assert False, "action must be defined!"

X = Super()
X.delegate()
```

- version 2:

```python
class Super:
    
    def delegate(self):
        self.action()
    
    def action(self):
        raise NotImplementedError("action must be defined!")
```

对于子类的实例, 将得到异常, 除非子类提供了期待的方法来替代超类中的默认方法: 

- version 1:

```python
    class Super:
            
        def delegate(self):
            self.action()
        
        def action(self):
            raise NotImplementedError("action must be defined!")

    class Sub(Super):
        pass
    
    X = Sub()
    X.delegate()
```

- version 2:

```python
class Super:
        
    def delegate(self):
        self.action()
    
    def action(self):
        raise NotImplementedError("action must be defined!")

class Sub(Super):
    def action(self):
        print("spam")

X = Sub()
X.delegate()
```

Python3 抽象超类

- 带有一个抽象方法的类是不能继承的(即, 我们不能通过调用它来创建一个实例), 除非其所有的抽象方法都已经在子类中定义了. 
  尽管这需要更多的代码, 但这种方法的优点是, 当我们试图产生一个实例的时候, 由于没有方法会产生错误, 这不会比我们试图调用一个没有
  的方法更晚. 这一功能可以用来定义一个期待的接口, 在客户类中自动验证. 
    - 示例 1: 

```python
from abc import ABCMeta, abstractmethod

class Super(metaclass = ABCMeta):

    @abstractmethod
    def method(self, ...):
        pass
```

- 示例 2: 

```python
from abc import ABCMeta, abstractmethod

# -------------------------
# 不能产生一个实例, 除非在类树的较低层级定义了该方法
# -------------------------
class Super(metaclass = ABCMeta):

    def delegate(self):
        self.action()
    
    @abstractmethod
    def action(self):
        pass
    
X = Super()
# -------------------------
# class 2
# -------------------------
class Super(metaclass = ABCMeta):

    def delegate(self):
        self.action()
    
    @abstractmethod
    def action(self):
        pass

class Sub(Super):
    pass

X = Sub()
# -------------------------
# class 3
# -------------------------
class Super(metaclass = ABCMeta):

    def delegate(self):
        self.action()
    
    @abstractmethod
    def action(self):
        pass

class Sub(Super):
    def action(self):
        print("spam")

X = Sub()
X.delegate()
```

## 命名空间: 完整的内容总结

这里将用于解析变量名的所有规则进行总结, 首先要记住的是, 
点号和无点号的变量名会用不同的方式处理, 
而有些作用域是用于对对象命名空间做初始设定的: 

- 无点号运算的变量名(例如: X)与作用域想对应
- 点号的属性名(例如: object.X)  使用的是对象的命名空间
- 有些作用域会对对象的命名空间进行初始化(模块、类)

### 简单变量名: 如果赋值就不是全局变量

无点号的简单变量名遵循函数的 LEGB 作用域法则, 具体如下: 

- 赋值语句(X = value)
    - 使变量名成为本地变量: 在当前作用域内, 创建或改变变量名 X,除非声明它是全局变量
- 引用(X)
    - 在当前作用域内搜索变量名 X, 之后是在任何以及所有的嵌套的函数中, 
      然后是在当前的全局作用域中搜索, 最后在内置的作用域中搜索

### 属性名称: 对象命名空间

### 赋值将变量名分类

### 命名空间字典

### 命名空间链接

## 类的文档字符串

- 文档字符串是出现在各种结构的顶部的字符串常量, 由 Python 在相应对象的 `__doc__` 属性自动保存, 
  它适用于模块文件、函数定义、类、方法. 
- 文档字符串的主要优点是, 它们在运行时能够保持, 并且, 它们从语法上比 `#` 注释(可以出现在程序中的任何地方)要缺乏灵活性
- 针对功能性文档(你的对象做什么), 使用文档字符串; 
- 针对更加微观的文档(令人费解的表达式是如何工作的), 使用 `#` 注释; 

示例: 

```python
# docstr.py file 

"""I am: docstr.__doc__"""

def func(args):
    """I am: docstr.func.__doc__"""
    pass

class spam:
    """I am: spam.__doc__ or docstr.spam.__doc__"""
    def method(self, arg):
        """I am: spam.method.__doc__ or self.method.__doc__"""
        pass
```


```python
import docstr

docstr.__doc__
docstr.func.__doc__
docstr.spam.__doc_
docstr.spam.method.__doc__

help(docstr)
```

## 类的设计

OOP 的设计问题, 就是如何使用类来对有用的对象进行建模！

- Python 中常用的 OOP 设计模式: 
    - 继承
    - 组合
    - 委托
    - 工厂
- 类设计概念
    - 伪私有属性
    - 多继承
    - 边界方法

### Python 和 OOP

Python 的 OOP 实现可以概括为三个概念: 

- 继承
    - 继承是基于 Python 中的属性查找的(在 X.name 表达式中)
- 多态
    - 在 X.method 方法中, method 的意义取决于 X 的类型
    - 因为 Python 没有类型声明而出现的, 属性总是在运行期间解析, 
      实现相同接口的对象是可互相交换的, 所以客户端不需要知道实现它们调用的方法的对象种类. 
- 封装
    - 方法和运算符实现行为, 数据隐藏默认是一种惯例

### OOP 和 继承: “是一个”关系

### OOP 和组合: “有一个”关系

### OOP 和委托: “包装”对象

# 类的高级主题

## 与设计相关的其他话题

- 继承
- 复合
- 委托
- 多继承
- 绑定方法
- 工厂
- 抽象超类
- 装饰器
- 类型子类
- 静态方法和类方法
- 管理属性
- 元类

# 其他实例

```python
class FirstClass:
    
    def setdata(self, value):
        self.data = value
    
    def display(self):
        print(self.data)

x = FirstClass()
y = FirstClass()

x.setdata("King Arthur")
y.setdata(3.14159)

x.display()
y.display()

x.data = "New value"
x.display()

x.anothername = "spam"

# ------------------------------

class SecondClass(FirstClass):

    def display(self):
        print("Current value = %s" % self.data)

z = SecondClass()
z.setdata(42)
z.display()

x.display()

# ------------------------------

class ThirdClass(SecondClass):

    def __init__(self, value):
        self.data = value
    
    def __add__(self, other):
        return ThirdClass(self.data + other)

    def __str__(self):
        return '[ThirdClass: %s]' % self.datas
    
    def mul(self, other):
        self.data = other
    
a = ThirdClass('abc')
a.display()
print(a)

b = a + "xyz"
b.display()
print(b)

a.mul(3)
print(a)
```
