---
title: Python 类使用详解
author: 王哲峰
date: '2024-04-02'
slug: python-usage
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

- [定义一个类](#定义一个类)
- [类的实例化](#类的实例化)
- [类的封装](#类的封装)
- [类的继承](#类的继承)
- [类的私有属性、方法](#类的私有属性方法)
    - [父类](#父类)
    - [子类](#子类)
- [类的实例方法、类方法、静态方法](#类的实例方法类方法静态方法)
    - [示例方法](#示例方法)
    - [类方法](#类方法)
    - [静态方法](#静态方法)
    - [示例](#示例)
- [子类实例直接调用父类属性、方法](#子类实例直接调用父类属性方法)
- [重写父类属性、方法](#重写父类属性方法)
- [强制调用父类私有属性、方法](#强制调用父类私有属性方法)
- [调用父类 `__init__` 方法](#调用父类-__init__-方法)
- [继承父类初始化过程中的参数](#继承父类初始化过程中的参数)
- [多继承与 super() 用法](#多继承与-super-用法)
- [本地测试](#本地测试)
</p></details><p></p>

# 定义一个类

```python
class Example1:

    def __init__(self, data1: int, data2: int) -> None:
        self.__data1 = data1  # 类的私有属性
        self.data2 = data2
    
    def __func1(self):
        print("Example 类的私有方法可以被类和实例调用.")

    def show_data(self):
        """
        调用类的私有方法
        调用类的私有属性
        调用类的普通属性
        """
        self.__func1()
        print(self.__data1)
        print(self.data2)


def test1():
    exp = Example1(50, 100)
    exp.show_data()
```

# 类的实例化

```python
class Example2:
    def __init__(self) -> None:
        print("Example 类的私有方法可以被类和实例调用.")


def test2():
    # 类 Example2 的实例    
    exp = Example2()
        

class Example22:
    def __init__(self, data1: int, data2: int) -> None:
        self.data1 = data1
        self.data2 = data2

def test22():
    # 类 Example22 的实例
    exp22 = Example22(10, 20)
    print(exp22.data1)
```

# 类的封装

```python
class Example3:
    """
    类对数据的封装
    """
    def __init__(self, data1: int, data2: int) -> None:
        self.data1 = data1
        self.data2 = data2

def test3():
    exp = Example3(10, 20)
    print(exp.data1)


class Example33:
    """
    类对逻辑的封装
    """
    def __init__(self, data1: int, data2: int) -> None:
        self.data1 = data1
        self.data2 = data2

    def add(self):
        print(self.data1 + self.data2)


def test33():
    exp = Example33(10, 20)
    exp.add()
```


# 类的继承

```python
class Father:
    def __init__(self, age: int) -> None:
        self.age = age
        print(f"age: {self.age}")

    def getAge(self):
        return self.age


class Son(Father):
    def __init__(self, age: int) -> None:
        """
        重写父类 Father 的属性

        Args:
            age (int): [description]
        """
        self.age = age


def test4():
    # 子类的实例继承了父类的方法
    son = Son(18)
    print(son.getAge())
```

# 类的私有属性、方法

## 父类

1. 没有下划线的方法或属性, 在类的定义中可以调用和访问, 类的实例也可以直接访问, 子类也可以访问, 能通过 `from module import *` 的方式导入
2. 单下划线的方法或属性, 在类的定义中可以调用和访问, 类的实例也可以直接访问, 子类也可以访问, 不能通过 `from module import *` 的方式导入
3. 双下划线的方法或属性, 在类的定义中可以调用和访问, 类的实例不可以直接访问, 子类不可以访问, 不能通过 `from module import *` 的方式导入

```python
class Father0:
    """
    父类
    """
    # 类属性
    __class_private_param = None  # 类的私有属性
    # TODO _class_protected_param = None  # 类的受保护的属性
    # TODO class_public_param = None  # 类的公有属性

    def __init__(self, private_param, protected_param, public_param) -> None:
        self.__private_param = private_param  # 实例的私有属性
        # TODO self._protected_param = protected_param  # 类、实例的受保护属性
        self.public_param = public_param  # 实例的公有属性

    def __private(self):
        print("父类的私有方法 private.")
    
    def _protected(self):
        print("父类的受保护的方法 protected.")

    def public(self):
        print("父类的公有方法 public.")
```

## 子类

1. 子类双划线方法 `__private` 方法并没有覆盖父类双下划线 `__private` 方法的权限
2. 子类的单下划线方法 `_protected` 方法是可以覆盖父类的单下划线 `_protected` 方法

```python
class Son0(Father0):
    """
    继承了 Father0 的子类
    """
    def __private(self):
        """
        子类的私有方法, 没有重写父类的 __private
        """
        print("子类的重载私有方法 private.")
    
    def _protected(self):
        """
        子类的受保护放方法, 重写父类的 _protected
        """
        print("子类的重载受保护方法 protected.")

    def public(self):
        """
        子类的重载公有方法, 重写父类的 public
        """
        print("子类的重载公有方法 public.")
```


# 类的实例方法、类方法、静态方法

## 示例方法

1. 除了静态方法与类方法外, 类的其他方法都属于实例方法
2. 实例方法隐含的参数为类实例 self
3. 实例方法需要将类实例化后才可以调用, 如果使用类直接调用实例方法, 需要显示地将实例对象作为参数传入
4. 类方法被类和实例调用, 实例方法被实例调用, 静态方法类和实例都能调用, 主要区别在于参数传递上的区别: 
    - 4.1 实例方法隐藏传递的是类实例 self 引用作为参数
    - 4.1 类方法隐藏传递的是 cls 引用作为参数
    - 4.3 静态方法无隐含参数

## 类方法

1. 类方法可以通过类直接调用, 或通过实例直接调用. 但无论哪种调用方式, 方法隐含的参数为类本身 cls
2. 类方法可以实例化对象和类访问, 不随实例属性的变化而变化

总结: 

1. 都是类属性的值, 不随实例属性的变化而变化
2. 类方法可以被实例对象和类访问
3. 类方法则更接近类似 Java 面向对象概念中的静态方法

## 静态方法

1. 静态方法是类中的函数, 不需要实例
2. 静态方法主要是用来存放逻辑性的代码, 主要是一些逻辑属于类, 
   但是和类本身没有交互, 即在静态方法中, 不会涉及到类中的方法和属性的操作. 
3. 可以理解为将静态方法存在此类的名称空间中. 
4. 事实上, 在python引入静态方法之前, 通常是在全局名称空间中创建函数
5. 关于静态方法只需要记住两方面: 可以被实例对象和类访问; 静态方法直接输出传入方法的值

Python 的静态方法和类方法都可以被类或实例访问, 两者概念不容易理清, 但还是有区别的: 

1. 静态方法无需传入 self 参数, 类方法需传入代表本类的 cls 参数
2. 从第 1 条, 静态方法是无法访问实例变量的, 而类方法也同样无法访问实例变量, 但可以访问类变量
3. 静态方法有点像函数工具库的作用, 而类方法则更接近类似 Java 面向对象概念中的静态方法


## 示例

```python
class A:
    """
    类
    """
    sentence = "this is a learning testing"

    def __init__(self, instance_param) -> None:
        """
        类的构造函数(初始化方法)
        """
        self.instance_param = instance_param

    def normalMethod(self):
        """
        普通方法、实例方法
        总结: 
            输出结果随着实例属性改变而改变
        """
        print(self.sentence)

    @classmethod
    def classMethod(cls, sentence):
        """
        类方法
        """
        print(cls.sentence)
    
    @staticmethod
    def staticMethod(sentence):
        """
        静态方法
        """
        print(sentence)


def test61():
    """
    使用实例调用实例方法
        1.不改变原始类的属性
        2.改变原始类属性
    """
    a = A("instance_param")
    # 1.不改变原始类的属性    
    a.normalMethod()  # this is a learning testing
    # 2.改变原始类属性
    sentence = "this is a normalMethod sentence" # 只是修改变量 sentence, 并没有修改类属性
    a.normalMethod()  # this is a learning testing

    a.sentence = "this is a normalMethod sentence" # 修改类属性
    a.normalMethod()  # this is a normalMethod sentence


def test62():
    """
    通过实例、类访问类的静态方法
    """
    # 实例对象访问静态方法
    a = A("instance_param")
    a.staticMethod("static sentence")  # static sentence
    a.staticMethod("changing sentence")  # changing sentence
    # 用类直接访问静态方法
    A.staticMethod("staticMethod sentence")  # staticMethod sentence
    A.staticMethod("changing staticMethod sentence")  # changing staticMethod sentence


def test63():
    """
    通过类、实例访问类方法
    """
    # 实例访问类方法
    a = A("instance_param")
    a.classMethod("classMethod sentence")  # this is a learning testing
    a.classMethod("changing classMethod sentence")  # this is a learning testing
    # 类访问类方法
    A.classMethod("classMethod sentence")  # this is a learning testing
    A.classMethod("changing classMethod sentence")  # this is a learning testing
```

# 子类实例直接调用父类属性、方法

```python
class Father1:
    """
    父类
    """
    def __init__(self):
        self.a = "aaa"
    
    def action(self):
        print("调用父类的方法.")


class Son1(Father1):
    """
    子类继承父类的属性和方法
    """
    pass


def test7():
    son1 = Son1()
    son1.action()
    print(son1.a)
```


# 重写父类属性、方法

```python
class Father2():
    def __init__(self) -> None:
        self.a = "aaa"
    
    def action(self):
        print("调用父类的方法.")


class Son2(Father2):
    def __init__(self) -> None:
        """
        子类重写父类的属性
        """
        self.a = "bbb"

    def action(self) -> None:
        """
        子类重写父类的方法
        """
        print("子类重写父类的方法.")


class Son22(Father2):
    def __init__(self) -> None:
        """
        子类重写父类的属性
        """
        self.a = "bbb"

    def action(self):
        """
        子类重写了父类的方法, 子类调用父类的方法
        """
        super().action()

def test8():
    son2 = Son2()
    son2.action()
    print(son2.a)
    son22 = Son22()
    son22.action()
    print(son22.a)
```

# 强制调用父类私有属性、方法

```python
class Father3():
    def __action(self):
        print("调用父类的方法.")


class Son3(Father3):
    """
    强制调用父类私有属性、方法
    """
    def action(self):
        super()._Father3__action()

def test9():
    son3 = Son3()
    son3.action()
```


# 调用父类 `__init__` 方法

```python
class Father4:
    def __init__(self, age: int) -> None:
        self.age = age
        print(f"age: {self.age}")
    
    def getAge(self):
        print("父类的返回结果:")
        return self.age


class Son4(Father4):
    """
    显式地调用父类的构造方法
    """
    def __init__(self, age: int) -> None:
        # method 1: 调用父类的 __init__ 方法
        super().__init__(age)
        # method 2: 调用父类的 __init__ 方法
        # Father4.__init__(self, age)

    def getAge(self):
        print("子类的返回结果:")
        return self.age


class Son44(Father4):
    """
    不重写父类的构造方法, 实例化子类时, 会自动调用父类定义的构造方法
    """
    def getAge(self):
        print("子类的返回结果:")
        return self.age


def test10():
    son4 = Son4(18)  # 18
    print(son4.getAge()) # 子类返回的结果: 18

    son44 = Son44(19)
    print(son44.getAge()) # 子列返回的结果: 19
```


# 继承父类初始化过程中的参数

```python
class Father5:
    def __init__(self) -> None:
        self.a = 1
        self.b = 2
    

class Son5(Father5):
    def __init__(self) -> None:
        super().__init__()

    def add(self):
        """
        继承父类初始化过程中的参数
        """
        return self.a + self.b


def test11():
    son5 = Son5()
    print(son5.add())


class Father6:
    def __init__(self, a: int, b: int) -> None:
        self.a = a
        self.b = b
    
    def dev(self):
        return self.a - self.b


class Son6(Father6):
    def __init__(self, a: int, b: int, c: int = 10) -> None:
        super().__init__(a, b)
        self.c = c

    def add(self):
        return self.a + self.b

    def compare(self):
        if self.c > (self.a + self.b):
            return True
        else:
            return False


def test111():
    son6 = Son6(1, 2)
    print(son6.dev())
    print(son6.add())
    print(son6.compare())
    son7 = Son6(1, 2, 1)
    print(son7.dev())
    print(son7.add())
    print(son7.compare())
```

# 多继承与 super() 用法

```python
class A:
    """
    经典类, 支持多继承
    """
    def __init__(self) -> None:
        print("A")
    

class B(A):
    """
    B 继承 A
    """
    def __init__(self) -> None:
        """
        B 类调用 A 的构造函数方法
        """
        A.__init__(self)
        print("B")


class C(B, A):
    """
    C 继承 B, A
    """
    def __init__(self):
        """
        C 调用 A, B 的 构造函数方法
        """
        A.__init__(self)
        B.__init__(self) # 会调用 A 的 __init__ 两次
        print("C")


class NewA:
    def __init__(self) -> None:
        print("NewA")


class NewB(NewA):
    def __init__(self) -> None:
        super(NewB, self).__init__()
        print("NewB")


class NewC(NewB, NewA):
    def __init__(self) -> None:
        super(NewC, self).__init__()
        print("NewC")
```

# 本地测试

```python
# 测试代码 main 方法
def main():
    test1()
    test2()
    test22()
    test3()
    test33()
    test4()
    test61()
    test62()
    test63()
    test7()
    test8()
    test9()
    test10()
    test11()
    test111()


if __name__ == "__main__":
    main()

```

