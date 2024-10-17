---
title: C++ 类
author: wangzf
date: '2022-10-01'
slug: cpp-class
categories:
  - c++
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

- [定义抽象数据类型](#定义抽象数据类型)
- [访问控制与封装](#访问控制与封装)
- [类的其他特性](#类的其他特性)
- [类的作用域](#类的作用域)
- [构造函数](#构造函数)
- [类的静态成员](#类的静态成员)
- [类设计模式](#类设计模式)
  - [通用工厂类](#通用工厂类)
    - [基础的工厂类设计](#基础的工厂类设计)
    - [参考](#参考)
</p></details><p></p>

在 C++ 中，使用类定义自己的数据类型。通过定义新的类型来反映待解决问题中的各种概念，
可以使我们更容易编写、调试和修改程序

类的基本思想是数据抽象(data abstraction)和封装(encapsulation)

* 数据抽象是一种依赖于接口(interface)和实现(implementation)分离的编程(以及设计)技术。
    - 类的接口包括用户所能执行的操作
    - 类的实现则包括类的数据成员、负责接口实现的函数体以及定义类所需的各种私有函数
* 封装实现了类的接口和实现的分离
    - 封装后的类隐藏了它的实现细节，也就是说，类的用户只能使用接口而无法访问实现部分

类想要实现数据抽象和封装，需要首先定义一个抽象数据类型(abstract data type)。
在抽象数据类型中，由类的设计者负责考虑类的实现过程；使用该类的程序员则只需要抽象地思考类型做了什么，
而无须了解类型的工作细节

# 定义抽象数据类型



# 访问控制与封装

# 类的其他特性


# 类的作用域


# 构造函数

# 类的静态成员



# 类设计模式

## 通用工厂类

抽象工厂模式经常被使用，那么同时也会多次写相同或类似的代码。
设计一个通用的工厂类，以避免重复设计工厂类

### 基础的工厂类设计


```cpp
class Button {}
```

### 参考

- https://blog.csdn.net/tenghui0425/article/details/23838535

