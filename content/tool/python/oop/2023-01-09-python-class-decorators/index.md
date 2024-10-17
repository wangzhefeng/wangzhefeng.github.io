---
title: Python 装饰器
author: wangzf
date: '2023-01-09'
slug: python-class-decorators
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

- [什么是装饰器](#什么是装饰器)
    - [为什么使用装饰器](#为什么使用装饰器)
    - [管理调用和实例](#管理调用和实例)
    - [管理函数和类](#管理函数和类)
    - [使用和定义装饰器](#使用和定义装饰器)
- [装饰器基础知识](#装饰器基础知识)
    - [函数装饰器](#函数装饰器)
        - [用法](#用法)
        - [实现](#实现)
        - [支持方法装饰](#支持方法装饰)
    - [类装饰器](#类装饰器)
        - [用法](#用法-1)
        - [实现](#实现-1)
        - [支持多个实例](#支持多个实例)
    - [装饰器嵌套](#装饰器嵌套)
    - [装饰器参数](#装饰器参数)
    - [装饰器管理函数和类](#装饰器管理函数和类)
- [编写函数装饰器](#编写函数装饰器)
    - [跟踪调用](#跟踪调用)
    - [状态信息保持选项](#状态信息保持选项)
        - [类实例属性](#类实例属性)
        - [封闭作用域和全局作用域](#封闭作用域和全局作用域)
        - [封闭作用域和 nonlocal](#封闭作用域和-nonlocal)
        - [函数属性](#函数属性)
    - [类错误之一: 装饰类方法](#类错误之一-装饰类方法)
    - [计时调用](#计时调用)
    - [添加装饰器参数](#添加装饰器参数)
- [编写类装饰器](#编写类装饰器)
    - [单体类](#单体类)
    - [跟踪对象接口](#跟踪对象接口)
    - [类错误之二: 保持多个实例](#类错误之二-保持多个实例)
    - [装饰器与管理函数的关系](#装饰器与管理函数的关系)
    - [为什么使用装饰器](#为什么使用装饰器-1)
- [直接管理函数和类](#直接管理函数和类)
- [示例——"私有"和"公有"属性](#示例私有和公有属性)
- [示例——验证函数参数](#示例验证函数参数)
    - [目标](#目标)
    - [针对位置参数的一个基本范围测试装饰器](#针对位置参数的一个基本范围测试装饰器)
    - [针对关键字和默认泛化](#针对关键字和默认泛化)
    - [实现细节](#实现细节)
</p></details><p></p>


```python
# -*- coding: utf-8 -*-

from functools import wraps

def logit(func):
    @wraps(func)
    def with_logging(*args, **kwargs):
        print(func.__name__ + "was called!")
        return func(*args, **kwargs)
    return with_logging

@logit
def addition_func(x):
    """Do some math."""
    return x + x


result = addition_func(4)
print(result)
```

# 什么是装饰器

**装饰** 是为函数和类指定管理代码的一种方式. 装饰器本身的形式是处理其它的可调用对象的可调用对象(如函数). 

Python 装饰器以两种相关的形式呈现: 

* **函数装饰器**
    - 在函数定义的时候进行名称重绑定, 提供一个逻辑层来管理函数和方法或随后对它们的调用
* **类装饰器**
    - 在类定义的时候进行名称重绑定, 提供一个逻辑层来管理函数和方法或随后对它们的调用

简而言之, 装饰器提供了一种方法, 在函数和类定义的语句的末尾插入自动运行代码——对于函数装饰器, 
在 def 的末尾; 对于类装饰器, 在 class 的末尾. 这样的代码可以扮演不同的角色. 

## 为什么使用装饰器

* 装饰器的优点
    - 装饰器为下面这样的任务提供了一种显示的语法, 它使得意图明确, 可以最小化扩展代码的冗余, 并且有助于确保正确的 API 使用: 
        - 装饰器有一种非常明确的语法, 这使得它们比那些可能任意地远离主体函数或类的辅助函数调用更容易被人们发现 
        - 当主体函数或类定义的时候, 装饰器应用一次; 在对类或函数的每次调用的时候, 不必添加额外的代码(在未来可能必须改变)
        - 由于前面的两点, 装饰器使得一个 API 的用户不太可能忘记根据 API 需求扩展一个函数或类
    - 换句话说, 除了其技术模型之外, 装饰器提供了一些和代码维护性和审美相关的优点. 
      此外, 作为结构化工具, 装饰器自然地促进了代码的封装, 这减少了冗余性并使得未来变得更容易. 
* 装饰器的缺点
    - 当装饰器插入包装类的逻辑, 它们可以修改装饰的对象的类型, 并且它们可能引发额外的调用
    - 另外, 同样的考虑也适用于任何为对象添加包装逻辑的技术

> 从纯技术的视角来看, 并不是严格需要装饰器:它们的功 能往往可以使用简单的辅助函数调用或其它的技术来实现

## 管理调用和实例

装饰器通过自动把函数和类名重绑定到其他的可调用对象来实现这些效果, 在 def 和 class 语句的末尾做到这点. 
当随后调用的时候, 这些可调用对象可以执行诸如对函数调用跟踪和计时、管理对类实例属性的访问等任务. 

- 函数装饰器安装包装器对象, 以在需要的时候拦截随后的 **函数调用** 并处理它们
- 类装饰器安装包装器对象, 以在需要的时候拦截随后的 **实例创建调用** 并处理它们

## 管理函数和类

尽管大多数实例都使用包装器来拦截随后对函数和类的调用, 但这并非使用装饰器的唯一方法:

* 函数装饰器: 也可以用来管理函数对象, 而不是随后对它们的调用
    - 例如, 把一个函数注册到一个 API
* 类装饰器: 也可以用来直接管理类对象, 而不是实例创建调用
    - 例如, 用新的方法扩展类

换句话说, 函数装饰器可以用来管理函数调用和函数对象, 类装饰器可以用来管理类实 例和类自身. 
通过返回装饰的对象自身而不是一个包装器, 装饰器变成了针对函数和类 的一种简单的后创建步骤. 

## 使用和定义装饰器

Python 本身带有具有特定角色的内置装饰器——静态方法装饰器、属性装饰器以及更多. 
此外, 很多流行的 Python 工具包括了执行管理数据库或用户接口逻辑等任务的装饰器. 
在这样的情况中, 我们不需要知道装饰器如何编码就可以完成任务. 

对于更为通用的任务, 程序员可以编写自己的任意装饰器. 例如, 函数装饰器可能实现下面的功能代码来扩展函数:

- 通过添加跟踪调用
- 在调试时执行参数验证测试
- 自动获取和释放线程锁
- 统计调用函数的次数以进行优化

你可以想象添加到函数调用中的任何行为, 都可以作为定制函数装饰器的备选. 

另外一方面, 函数装饰器设计用来只增强一个特定函数或方法调用, 而不是一个完整的对象接口. 
类装饰器更好地充当后一种角色——因为它们可以拦截实例创建调用, 它们可以用来实现任意的对象接口扩展或管理任务. 
例如, 定制的类装饰器可以跟踪或验证对一个对象的每个属性引用. 它们也可以用来实现代理对象、
单体类以及其他常用的编程模式. 实际上, 我们将会发现很多类装饰器与在第30章中见到的委托编程模式有很大的相似之处. 

# 装饰器基础知识

> 装饰器的很多神奇之处可归结为自动绑定操作

## 函数装饰器

函数装饰器主要只是一种语法糖: 通过在一个函数的 def 语句的末尾来运行另一个函数, 
把最初的函数名重新绑定到结果.

### 用法

函数装饰器是一种关于函数的运行时声明, 函数的定义需要遵守此声明. 
装饰器在紧挨着定义一个函数或方法的 def 语句之前的一行编写, 
并且它由 `@` 符号以及紧随其后的对于元函数的一个引用组成——
这是管理另一个函数的函数(或其他的可调用对象). 

在编码方面, 函数装饰器自动将如下的语法: 

```python
@decorator  # Decorate function
def F(arg):
    ...

F(99)       # Call function
```

映射为这一对等的形式, 其中装饰器是一个单参数的可调用对象, 
它返回与 F 具有相同数目的参数的一个可调用对象: 

```python
def F(arg):
    ...

F = decorator(F)
F(99)
```

这一自动名称重绑定在 def 语句上有效, 不管它针对一个简单的函数或是类中的一个方法. 
当随后调用 F 函数的时候, 它自动调用装饰器所返回的对象, 该对象可能是实现了所需的包装逻辑的另一个对象, 
或者是最初的函数本身. 


- 示例 1: 
    - 装饰实际把如下的第一行映射为第二行: 

```python
func(6, 7)
decorator(func)(6, 7)
```

- 示例 2: 
    - 在 def 语句的末尾, 方法名重新绑定到一个内置函数装饰器的结果, 随后再调用最初的名称, 将会调用装饰器所返回的对象: 

```python
class C:
@staticmethod
def meth(*args): # meth = staticmethod(meth)
    pass

class C:
@property
def name(self): # name = property(name)
    pass
```

### 实现

装饰器自身是一个返回可调用对象的可调用对象. 也就是说, 它返回了一个对象, 
当随后装饰的函数通过其最初的名称调用的时候, 将会调用这个对象——不管是拦截了随后调用的一个包装器对象, 
还是最初的函数以某种方式的扩展. 实际上, 装饰器可以是任意类型的可调用对象, 
并且返回任意类型的可调用对象: 函数和类的任何组合都可以使用, 尽管一些组合更适合于特定的背景. 

- 示例 1: 
    - 在一个函数创建之后接入协议以管理函数, 这么做将直接向函数的定义添加创建之后的步骤, 
    这样的一个结构可能会用来把一个函数注册到一个 API、赋值函数属性等

```python
def decorator(F):
    # Process function F
    return F

@decorator
def func():
    pass # func = decorator(func)
```

- 示例 2: 
    - 更典型的用法是: 插入逻辑以拦截对函数的随后调用, 
      可以编写一个装饰器来返回和最初函数不同的一个对象

```python
def decorator(F):
    # Save or use function F
    # Return a different callable: nested def, class with __call__, etc.
    pass

@decorator
def func():
    pass # func = decorator(func)
```

- 示例 3: 
    - 有一种常用的编码模式--装饰器返回了一个包装器, 包装器把最初的函数保持到一个封闭的作用域中

    ```python
    def decorator(F):      # ON @decorator
        def wrapper(*args): # On wrapped function call
            # Use F and args
            # F(*args) call original function
            pass
        return wrapper
    
    @decorator             # func = decorator(func)
    def func(x, y):        # func is passed to decorator's F
        pass 

    func(6, 7)             # 6, 7 are passed to warpper's *args
    ```

    - 当随后调用名称 func 的时候, 它确实调用装饰器所返回的包装器函数;
    随后包装器函数可能会运行最初的 func, 因为它在一个封闭的作用域中仍然可以使用. 
    当以这种方式编码的时候, 每个装饰的函数都会产生一个新的作用域来保持状态. 

- 示例 4: 
    - 为了对类做类似 wrapper 的事情, 可以重载调用操作, 并且使用实例属性而不是封闭的作用域

    ```python
    class decorator:
        def __init__(self, func):   # On @decorator
            self.func = func
        
        def __call__(self, *args):  # On wrapped function call
            # Use self.func and args
            # self.func(*args) calls original function
            pass

    @decorator
    def func(x, y):                # func = decorator(func)
        pass                        # func is passed to __init__
    
    func(6, 7)                     # 6, 7 are passed to __call__'s *args
    ```

    - 随后再调用 func 的时候, 他确实会调用装饰器所创建的实例的 __call__ 运算符重载方法; 
    然后, __call__ 方法可能运行最初的 func, 因为它在一个 **实例属性** 中仍然可用. 
    当按照这种方式编写代码的时候, 每个装饰的函数都会产生一个新的实例来保持状态. 


### 支持方法装饰

- 尽管前面关于类的装饰器代码对于拦截简单函数调用有效, 但当它应用于类方法函数的时候, 并不是很有效 

```python
# 类装饰器
class decorator:
    def __init__(self, func):     # func is method without instance
        self.func = func
  
    def __call__(self, *args):    # self is decorator instance
        # self.func(*args) fails!  # C instance not in args!
        pass

# 装饰类方法
class C:
    @decorator
    def method(self, x, y):        # method = decorator(method)
        pass                        # Rebound to decorator instance
```

- 当按照这种方式编码的时候, 装饰的方法(method)重绑定到装饰器类(decorator)的一个实例, 
而不是一个简单的函数. 这一点带来的问题是, 当装饰器类的 __call__ 方法随后运行的时候, 
其中的 self 接收装饰器类(decorator)实例, 并且类 C 的实例不会包含到一个 `*args` 中. 
这使得有可能把调用分派给最初的方法--即保持了最初的方法函数的装饰器对象, 但是, 没有实例传递给它. 

- 为了支持函数和方法, 嵌套函数的替代方法工作得更好

```python
# 函数装饰器
def decorator(F):          # F is func or method without instance
    def wrapper(*args):     # class instance in args[0] for method
        # F(*args) runs func or method
        pass
    return wrapper

# 装饰函数
@decorator
def func(x, y):            # func = decorator(func)
    pass

func(6, 7)                 # Really calls wrapper(6, 7)

# 装饰类
class C:
    @decorator
    def method(self, x, y): # method = decorator(method)
        pass

c = C()
c.method(6, 7)             # Really calls wrapper(c, 6, 7)
```

- 当按照这种方法编写的包装类在其第一个参数里接收了 C 类实例的时候, 它可以分派到最初的方法和访问状态信息

## 类装饰器

类装饰器与函数装饰器密切相关, 实际上, 它们使用相同的语法和非常相似的编码模式. 
然而, 不是包装单个的函数或方法, 类装饰器是管理类的一种方式, 
或者用管理或扩展类所创建的实例的额外逻辑, 来包装实例构建调用. 

### 用法

从语法上讲, 类装饰器就像前面的 class 语句一样(就像前面函数定义中出现的函数装饰器). 
在语法上, 假设装饰器是返回一个可调用对象的一个单参数的函数, 类装饰器语法: 

```python
# Decorate class
@decorator
class C:
    ...

# Make an instance 
x = C(99)
```

类自动地传递给装饰器函数, 并且装饰器的结果返回来分配给类名, 
直接的结果就是, 随后调用类名会创建一个实例, 
该实例会触发装饰器所返回的可调用对象, 而不是调用最初的类本身: 

```python
class C:
    ...

c = decorator(C)  # Rebind class name to decorator result
x = C(99)         # Essentially calls decorator(C)(99)
```

### 实现

新的类装饰器使用函数装饰器所使用的众多相同的技术来编码. 
由于类装饰器也是返回一个可调用对象的一个可调用对象, 
因此大多数函数和类的组合已经最够了. 尽管先编码, 
但装饰器的结果是当随后创建一个实例的时候才运行的. 

- 示例 1: 
    - 要在一个类创建之后直接管理它, 返回最初的类自身

```python
def decorator(C):
    # Process class C
    return C

@decorator
class C:       # C = decorator(C)
    pass
```

- 示例 2: 
    - 不是插入一个包装器层来拦截随后的实例创建调用, 而是返回一个不同的可调用对象

```python
def decorator(C):
    # Save or Use class C
    # Return a different callable: nested def, class with __call__, etc.
    pass

@decorator
class C:  # C = decorator(C)
    pass
```

```python
def decorator(cls):
    class Wrapper:
        def __init__(self, *args):
            self.wrapped = cls(*args)
        def __getattr__(self, name):
            return getattr(self.wrapped, name)

    return Wrapper

@decorator
class C:
    def __init__(self, x, y):
        self.attr = "spam"

x = C(6, 7)
print(x.attr)
```

### 支持多个实例

和函数装饰器一样, 使用类装饰器的时候, 一些可调用对象组合比另一些工作得更好

- 示例 1: 
    - 这段代码处理多个被装饰的类(每个都产生一个新的 Decorator 实例), 
    并且会拦截实例创建调用(每个运行 __call__ 方法). 然而, 和前面的版本不同, 
    这个版本没有能够处理给定的类的多个实例——每个实例创建调用都覆盖了前面保存的实例. 
    最初的版本确实支持多个实例, 因为每个实例创建调用产生了一个新的独立的包装器对象.

```python
class Decorator:
    def __init__(self, C):                     # On @decorator
        self.C = C
    def __call__(self, *args):                 # On instance creation
        self.wrapped = self.C(*args)
        return self
    def __getattr__(self, attrname):           # On atrribute fetch
        return getattr(self.wrapper, attrname)

@Decorate
class C:                                      # C = Decorator(C)
    ...

x = C()
y = C()                                       # Overwrites x!
```

- 示例 2: 
    - 每一个都支持多个包装的实例

```python
def decorator(C):                # On @decorator
    class Wrapper:
        def __init__(self, *args): # On instance creation
            self.wrapped = C(*args)
return Wrapper

class Wrapper:
    pass

def decorator(C):                # On @decorator
    def onCall(*args):            # On instance creation
        return Wrapper(C(*args))   # Embed instance in instance
    return onCall
```

## 装饰器嵌套

有的时候, 一个装饰器不够, 为了支持多步骤的扩展, 
装饰器语法允许我们向一个装饰的函数或方法添加包装器逻辑的多个层. 
当使用这一功能的时候, 每个装饰器必须出现在自己的一行中. 语法如下: 

- 函数嵌套装饰器

```python
@A
@B
@C
def f(args):
    pass
```

如下这样运行: 

```python
def f(args):
    pass

f = A(B(C(f)))
```

- 类嵌套装饰器

```python
@spam
@eggs
class C:
    pass

X = C()
```

等同于如下的代码: 

```python
class C:
    pass

C = spam(eggs(C))
X = C()
```

## 装饰器参数

函数装饰器和类装饰器似乎都能够接受参数, 尽管实际上这些参数传递给了真正返回装饰器的一个可调用对象, 
而装饰器反过来又返回一个可调用对象. 

```python
@decorator
def F(arg):
    pass

F(99)
```

自动地映射到其对等形式, 其中装饰器是一个可调用对象, 它返回实际的装饰器. 返回的装饰器反过来返回可调用的对象, 
这个对象随后运行以调用最初的函数名. 装饰器参数在装饰发生之前就解析了, 并且它们通常用来保持状态信息供随后的调用使用: 

```python
def F(arg):
    pass

F = decorator(A, B)(F) # Rebind F to result of decorator's return value
F(99)                  # Essentially calls decorator(A, B)(F)(99)
```

- 示例: 

```python
def decorator(A, B):
    # Save or use A, B
    def actualDecorator(F):
        # Save or use function F
        # Return a callable: nested def, class with __call__, etc.
        return callable
    reurn actualDecorator
```

- 这个结构中的外围函数通常会把装饰器参数与状态信息分开保存, 
以便在实际的装饰器中使用, 或者在它所返回的可调用对象中使用, 
或者在二者中都使用. 这段代码在封闭的函数作用域引用中保存了状态信息参数, 
但是通常也可以使用类属性. 

换句话说, 装饰器参数往往意味着可调用对象的3个层级:

- 接受装饰器参数的一个可调用对象(actualDecorator(F)), 它返回一个可调用对象(callable)以作为装饰器, 
  该装饰器返回一个可调用对象(actualDecorator)来处理对最初的函数或类的调用这 3 个层级的每一个都可能是一个函数或类, 
  并且可能以作用 域或类属性的形式保存了状态. 

## 装饰器管理函数和类

装饰器机制是在函数和类创建之后通过一个可调用对象传递它们的一种协议. 
因此, 它可以用来调用任意的创建后处理. 只要以这种方式返回最初装饰的对象, 
而不是返回一个包装器, 我们就可以管理函数和类自身, 而不只是管理随后对它们的调用. 

```python
def decorator(0):
    # Save or augment function or class O
    return 0

@decorator
def F():    # F = decorator(F)
    pass

@decorator
class C:    # C = decorator(C)
    pass
```

# 编写函数装饰器

## 跟踪调用

* 示例 1: 
    - 定义并应用一个函数装饰器, 来统计对装饰的函数的调用次数, 并且针对每一次调用打印跟踪信息

```python
class tracer:
    def __init__(self, func):  # On @decoration: save origin func
    self.calls = 0
    self.func = func
    
    def __call__(self, *args):  # On later calls: run original func 
    self.calls += 1
    print(f"call {self.calls} to {self.func.__name__}")
    self.func(*args)

@trace
def spam(a, b, c):  # spam = tracer(spam)
    print(a + b + c) # Wraps spam in a decorator object

from decorator1 import spam
>>> spam(1, 2, 3)
call 1 to spam
6
>>> spam("a", "b", "c")
call 2 to spam
abc
>>> spam.calls
2
>>> spam
<decorator1.tracer object at 0x02D9A730>
```

* 运行的时候, tracer 类和装饰的函数分开保存, 并且拦截对装饰的函数随后的调用, 
以便添加一个逻辑层来统计和打印每次调用. 注意, 调用的总数如何作为装饰的函数的一个属性显示——装饰的时候, 
spam 实际上是 tracer 类的一个实例(对于进行类型检查的程序, 可能还会衍生一次查找, 但是通常是有益的). 

```python
# 下面的非装饰器代码与上面的代码对等
calls = 0
def tracer(func, *args):
    global calls
    calls += 1
    print(f"call {calls} to {func.__name__}")
    func(*args)

def spam(a, b, c):
    print(a, b, c)

>>> spam(1, 2, 3)
1, 2, 3
>>> tracer(spam, 1, 2, 3)
call 1 to spam
1, 2, 3
```

## 状态信息保持选项

函数装饰器有各种选项来保持装饰的时候所提供的状态信息, 以便在实际函数调用过程中使用. 
它们通常需要支持多个装饰的对象以及多个调用, 但是, 有多种方法来实现这些目标:实例属性、
全局变量、非局部变量和函数属性, 都可以用于保持状态. 

### 类实例属性

* 示例 1: 
    - 这里是前面示例的一个扩展版本, 其中添加了对关键字参数的支持, 并且返回包装函数的结果, 以支持更多的用例
    - 这里的代码使用类实例属性来显式地保存状态, 包装的函数和调用计数器都是针对每个实例的信息--每个装饰都有自己的拷贝

```python
class tracer:
def __init__(self, func):
    self.calls = 0
    self.func = func
def __call__(self, *args, **kwargs):
    self.calls += 1
    print(f"call {self.calls} to {self.func.__name__}")
    return self.func(*args, **kwargs)

@tracer
def spam(a, b, c):   # Same as: spam = tracer(spam)
print(a + b + c)  # Triggers tracer.__init__

@tracer
def eggs(x, y):      # Same as: eggs = tracer(eggs)
print(x ** y)     # Wraps eggs in a tracer object

spam(1, 2, 3)        # Really calls tracer instanc: runs trace.__call__
spam(a = 4, b = 5, c = 6)

eggs(2, 16)    # Really calls tracer instance, self.func is eggs
eggs(4, y = 4) # self.calls is pre-function here
```

### 封闭作用域和全局作用域

```python
calls = 0
def tracer(func):
    def wrapper(*args, **kwargs):
    global calls
    calls += 1
    print(f"call {calls} to {func.__name__}")
    return func(*args, **kwargs)
    return wrapper

@tracer
def spam(a, b, c):   # Same as: spam = tracer(spam)
    print(a + b + c)

@tracer
def eggs(x, y):      # Same as: eggs = tracer(eggs)
    print(x ** y)

spam(1, 2, 3)             # Really calls wrapper, bound to func
spam(a = 4, b = 5, c = 6) # wrapper calls spam

eggs(2, 16)    # Really calls wrapper, bound to eggs
eggs(4, y = 4) # Global calls is not pre-function here!
```

### 封闭作用域和 nonlocal

```python
def tracer(func):
    calls = 0
    def wrapper(*args, **kwargs):
    nonlocal calls
    calls += 1
    print(f"call {calls} to {func.__name__}")
    return func(*args, **kwargs)
    return wrapper

@tracer
def spam(a, b, c):   # Same as: spam = tracer(spam)
    print(a + b + c)

@tracer
def eggs(x, y):      # Same as: eggs = tracer(eggs)
    print(x ** y)

spam(1, 2, 3)             # Really calls wrapper, bound to func
spam(a = 4, b = 5, c = 6) # wrapper calls spam

eggs(2, 16)    # Really calls wrapper, bound to eggs
eggs(4, y = 4) # Nonlocal calls is not pre-function here!
```

### 函数属性

```python
def tracer(func):                   # State via enclosing scope and func attr
    def wrapper(*args, **kwargs):    # calls is per-function, not global
        wrapper.calls += 1
        print(f"call {wrapper.calls} to {func.__name__}")
        return func(*args, **kwargs)
    wrapper.calls = 0
    return wrapper
```

## 类错误之一: 装饰类方法

* 基于类的跟踪装饰器

```python
class tracer:
    def __init__(self, func):
    self.calls = 0
    self.func = func
    def __call__(self, *args, **kwargs):
    self.calls += 1
    print(f"call {self.calls} to {self.func.__name__}")
    return self.func(*args, **kwargs)
```

* 对于简单函数的装饰是生效的

```python
@tracer
def spam(a, b, c):
print(a, b, c)

spam(1, 2, 3)
spam(a = 4, b = 5, c = 6)
```

* 对于类方法的装饰失效了

```python
class Person:
def __init__(self, name, pay):
    self.name = name
    self.pay = pay

@tracer
def giveRaise(self, percent):
    self.pay *= (1.0 + percent)

@tracer
def lastName(self):
    return self.name.split()[-1]

bob = Person("Bob Smith", 50000) # tracer remembers method funcs
bob.giveRaise(0.25)
```

## 计时调用

## 添加装饰器参数

# 编写类装饰器

尽管类似于函数装饰器的概念, 但类装饰器应用于类——它们可以用于管理类自身, 
或者用来拦截实例创建调用以管理实例. 和函数装饰器一样, 类装饰器其实只是可选的语法糖, 
尽管很多人相信, 它们使程序员的意图更为明显并且能使不正确的调用最小化. 

## 单体类

## 跟踪对象接口

## 类错误之二: 保持多个实例

## 装饰器与管理函数的关系

## 为什么使用装饰器

# 直接管理函数和类

上面大多数示例都是设计来拦截函数和实例创建调用. 尽管这对于装饰器来说很典型, 它们并不限于这一角色. 
因为装饰器通过装饰器代码来运行新的函数和类, 从而有效地工作, 它们也可以用来管理函数和类本身, 
而不只是对它们随后的调用. 

# 示例——"私有"和"公有"属性


# 示例——验证函数参数

开发一个函数装饰器, 它自动测试传递给一个函数或方法的参数是否在有效的数值范围内. 
它设计用来在任何开发或产品阶段使用, 并且它可以用作类似任务的一个模板

## 目标

- 示例 1(不好用): 

```python
class Person:
    """
    根据一个传入的百分比用来给表示人的对象涨工资
    """
    def giveRaise(self, percent):
    self.pay = int(self.pay * (1 + percent))
```

- 示例 2(不好用): 

```python
class Person:
    """
    根据一个传入的百分比用来给表示人的对象涨工资
    """
    def giveRaise(self, percent):
    if percent < 0.0 or percent > 1.0:
        raise TypeError('percent invalid')
    self.pay = int(self.pay * (1 + percent))
```

- 示例 3(不好用): 

```python
class Person:
    """
    根据一个传入的百分比用来给表示人的对象涨工资
    """
    def giveRaise(self, percent):
    assert percent >= 0.0 and percent <= 1.0, "percent invalid"
    self.pay = int(self.pay * (1 + percent))
```

- 示例 4: 
    - 开发一个通用的工具来自动为我们执行范围测试,  针对我们现在或将来要编写的任何函数或方法的参数. 装饰器方法使得这明确而方便. 
      在装饰器中隔离验证逻辑, 这简化了客户类和未来的维护. 注意, 我们这里的目标和前面编写的属性验证不同. 这里, 
      我们想要验证传入的函数参数的值, 而不是设置的属性的值. 

```python
class Person:
    """
    根据一个传入的百分比用来给表示人的对象涨工资
    """
    @rangetest(percent = (0.0, 1.0)) # Use decorator to validate
    def giveRaise(self, percent):
    self.pay = int(self.pay * (1 + percent))
```

## 针对位置参数的一个基本范围测试装饰器

## 针对关键字和默认泛化

## 实现细节

