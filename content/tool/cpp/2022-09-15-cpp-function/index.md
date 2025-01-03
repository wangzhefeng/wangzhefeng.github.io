---
title: C++ 函数
author: wangzf
date: '2022-09-15'
slug: cpp-function
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

- [函数基础](#函数基础)
  - [函数简介](#函数简介)
    - [编写函数](#编写函数)
    - [调用函数](#调用函数)
    - [形参和实参](#形参和实参)
    - [函数的形参列表](#函数的形参列表)
  - [局部对象](#局部对象)
  - [分离式编译](#分离式编译)
- [参数传递](#参数传递)
- [返回类型和 return 语句](#返回类型和-return-语句)
- [函数重载](#函数重载)
- [特殊用途语言特性](#特殊用途语言特性)
- [函数匹配](#函数匹配)
- [函数指针](#函数指针)
</p></details><p></p>

函数是一个命名了的代码块，可以通过调用函数执行相应的代码。
函数可以有 0 个或多个参数，而且(通常)会产生一个结果。
可以重载函数，也就是说，同一个名字可以对应几个不同的函数

# 函数基础

## 函数简介

一个典型的函数(function):

* 返回类型(return type)
* 函数名字
* 形参(parameter)
* 函数体(function body)

通过调用运算符(call operator)来执行函数，调用运算符的形式是一对圆括号，
它作用于一个表达式，该表达式是函数或者指向函数的指针:

* 圆括号之内是一个用逗号隔开的实参(argument)列表，用实参初始化函数的形参
* 调用表达式的类型就是函数返回的类型

### 编写函数

```cpp
// val 的阶乘是 val*(val-1)*(val-2)...*((val-(val-1))*1)
int fact(int val) {
    int ret = 1;
    while (val > 1) {
        ret *= val--;
    }
    return ret;
}
```

### 调用函数

```cpp
int main() {
    int j = fact(5);
    cout << "5! is " << j << endl;
    return 0;
}
```

### 形参和实参

实参是形参的初始值，尽管实参与形参存在对应关系，
但是并没有规定实参的求值顺序，编译器能以任意可行的顺序对实参求值

实参的类型必须与对应的形参类型匹配，函数有几个形参，就必须提供相同数量的实参，
因为函数的调用规定实参数量与形参数量一致，所以形参一定会被初始化

```python
int j = fact("hello");  // 错误: 实参类型不正确
int j = fact();  // 错误: 实参数量不足
int j = fact(42, 10, 0);  // 错误: 实参数量过多
int j = fact(3.14);  // 正确: 该实参能隐式地转换成 int 类型(截去小数部分)
```

### 函数的形参列表

函数的形参列表可以为空，但是不能省略。要想定义一个不带形参的函数，最常用的办法是书写一个空的形参列表。
不过为了与 C 语言兼容，也可以使用关键字 `void` 表示函数没有形参

形参列表中的形参通常用逗号隔开，其中每个形参都是含有一个声明符的声明。
即使两个形参的类型一样，也必须把两个类型都写出来

任意两个形参都不能同名，而且函数最外层作用域中的局部变量也不能使用与函数形参一样的名字

```cpp
void f1() {
    /* ... */
}


void f2(void) {
    /* ... */
}
```






## 局部对象


## 分离式编译

# 参数传递


# 返回类型和 return 语句



# 函数重载



# 特殊用途语言特性



# 函数匹配


# 函数指针

