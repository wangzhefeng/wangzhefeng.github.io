---
title: Python 描述符
author: 王哲峰
date: '2022-07-25'
slug: python-class-descriptor
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

- [为什么要使用描述符?](#为什么要使用描述符)
    - [问题](#问题)
    - [常规思路:](#常规思路)
    - [加入判断逻辑](#加入判断逻辑)
    - [Property 特性](#property-特性)
    - [描述符](#描述符)
- [参考](#参考)
</p></details><p></p>

# 为什么要使用描述符? 

## 问题

假想你正在给学校写一个成绩管理系统, 并没有太多编码经验的你, 可能按照下面的常规思路来写. 
看起来一切都很合理. 

但是程序并不像人那么智能, 不会自动根据使用场景判断数据的合法性, 如果老师在录入成绩的时候, 
不小心录入了将成绩录成了负数, 或者超过100, 程序是无法感知的. 

聪明的你, 马上在代码中加入了判断逻辑. 这下程序稍微有点人工智能了, 能够自己明辨是非了. 
程序是智能了, 但在__init__里有太多的判断逻辑, 很影响代码的可读性. 巧的是, 
你刚好学过 Property 特性, 可以很好的应用在这里. 于是你将代码修改成如下, 代码的可读性瞬间提升了不少. 
程序还是一样的人工智能, 非常好. 

你以为你写的代码, 已经非常优秀, 无懈可击了. 

没想到, 人外有天, 你的主管看了你的代码后, 深深地叹了口气: 类里的三个属性, math、chinese、english, 
都使用了 Property 对属性的合法性进行了有效控制. 
功能上, 没有问题, 但就是太啰嗦了, 三个变量的合法性逻辑都是一样的, 
只要大于0, 小于100 就可以, 代码重复率太高了, 这里三个成绩还好, 
但假设还有地理、生物、历史、化学等十几门的成绩呢, 这代码简直没法忍. 
去了解一下 Python 的描述符吧. 

经过主管的指点, 你知道了「描述符」这个东西. 怀着一颗敬畏之心, 你去搜索了下关于 描述符的用法. 

其实也很简单, 一个实现了 描述符协议 的类就是一个描述符. 

什么描述符协议: 在类里实现了 `__get__()`、`__set__()`、`__delete__()` 其中至少一个方法. 

- `__get__`: 用于访问属性. 它返回属性的值, 若属性不存在、不合法等都可以抛出对应的异常. 
- `__set__`: 将在属性分配操作中调用. 不会返回任何内容. 
- `__delete__`: 控制删除操作. 不会返回内容. 

对描述符有了大概的了解后, 你开始重写上面的方法. 

如前所述, Score 类是一个描述符, 当从 Student 的实例访问 math、chinese、english这三个属性的时候, 都会经过 Score 类里的三个特殊的方法. 这里的 Score 避免了 使用Property 出现大量的代码无法复用的尴尬. 

以上, 我举了下具体的实例, 从最原始的编码风格到 Property , 最后引出描述符. 由浅入深, 一步一步带你感受到描述符的优雅之处. 

到这里, 你需要记住的只有一点, 就是描述符给我们带来的编码上的便利, 它在实现 保护属性不受修改、属性类型检查 的基本功能, 同时有大大提高代码的复用率. 

> Python 描述符(descriptor): 一个实现了描述符协议的类就是一个描述符
> 
> 描述符给我们带来的编码上的便利, 它在实现 保护属性不受修改、属性类型检查 的基本功能, 同时有大大提高代码的复用率。
> 什么是描述符协议: 在类里实现了 __get__()、__set__()、__delete__() 其中至少一个方法: 
> 
> *  __get__: 用于访问属性. 它返回属性的值, 若属性不存在、不合法等都可以抛出对应的异常
> *  __set__: 将在属性分配操作中调用. 不会返回任何内容
> *  __delete__: 控制删除操作. 不会返回内容
> 
> 描述符分两种: 
> * 数据描述符: 实现了__get__ 和 __set__ 两种方法的描述符
> * 非数据描述符: 只实现了__get__ 一种方法的描述符
> 
> 数据描述器和非数据描述器的区别在于: 它们相对于实例的字典的优先级不同, 
> 如果实例字典中有与描述符同名的属性, 如果描述符是数据描述符, 优先使用数据描述符, 
> 如果是非数据描述符, 优先使用字典中的属性。

## 常规思路: 

```python
class Student:

    def __init__(self, name, math, chinese, english):
        self.name = name
        self.math = math
        self.chinese = chinese
        self.english = english

    def __repr__(self):
        return f"<Student: {self.name}, math:{self.math}, chinese: {self.chinese}, english:{self.english}>"


# 测试代码 main 函数
def main():
    std1 = Student("xiaoming", 7, 8, 9)
    print(std1)

if __name__ == "__main__":
    main()
```

## 加入判断逻辑

```python
class Student:

    def __init__(self, name, math, chinese, english):
        self.name = name
        if 0 <= math <= 100:
            self.math = math
        else:
            raise ValueError("Valid value must be in [0, 100]")
        
        if 0 <= chinese <= 100:
            self.chinese = chinese
        else:
            raise ValueError("Valid value must be in [0, 100]")
    
        if 0 <= chinese <= 100:
            self.english = english
        else:
            raise ValueError("Valid value must be in [0, 100]")
        
    def __repr__(self):
        return "<Student: {}, math:{}, chinese: {}, english:{}>".format(
                self.name, self.math, self.chinese, self.english
            )
```

## Property 特性

```python
class Student:
    def __init__(self, name, math, chinese, english):
        self.name = name
        self.math = math
        self.chinese = chinese
        self.english = english

    @property
    def math(self):
        return self._math

    @math.setter
    def math(self, value):
        if 0 <= value <= 100:
            self._math = value
        else:
            raise ValueError("Valid value must be in [0, 100]")

    @property
    def chinese(self):
        return self._chinese

    @chinese.setter
    def chinese(self, value):
        if 0 <= value <= 100:
            self._chinese = value
        else:
            raise ValueError("Valid value must be in [0, 100]")

    @property
    def english(self):
        return self._english

    @english.setter
    def english(self, value):
        if 0 <= value <= 100:
            self._english = value
        else:
            raise ValueError("Valid value must be in [0, 100]")

    def __repr__(self):
        return "<Student: {}, math:{}, chinese: {}, english:{}>".format(
                self.name, self.math, self.chinese, self.english
            )
```

## 描述符

```python
class Score:

    def __init__(self, default=0):
        self._score = default

    def __set__(self, instance, value):
        if not isinstance(value, int):
            raise TypeError('Score must be integer')
        if not 0 <= value <= 100:
            raise ValueError('Valid value must be in [0, 100]')

        self._score = value

    def __get__(self, instance, owner):
        return self._score

    def __delete__(self):
        del self._score
        
class Student:

    math = Score(0)
    chinese = Score(0)
    english = Score(0)

    def __init__(self, name, math, chinese, english):
        self.name = name
        self.math = math
        self.chinese = chinese
        self.english = english


    def __repr__(self):
        return "<Student: {}, math:{}, chinese: {}, english:{}>".format(
                self.name, self.math, self.chinese, self.english
            )
```

# 参考

* [描述符](https://mp.weixin.qq.com/s?__biz=Mzg3MjU3NzU1OA==&mid=2247496467&idx=1&sn=927f0093e62a78a1a04d1b0305c45c7a&source=41#wechat_redirect)
