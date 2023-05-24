---
title: C++ 快速入门
author: 王哲峰
date: '2022-04-08'
slug: cpp-helloworld
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

- [编写一个简单的 C++ 程序](#编写一个简单的-c-程序)
  - [一个简单的 C++ 程序](#一个简单的-c-程序)
  - [一个函数的定义包含四部分](#一个函数的定义包含四部分)
  - [main 函数](#main-函数)
- [输入输出](#输入输出)
  - [C++ IO 标准库 iostream](#c-io-标准库-iostream)
  - [一个使用 IO 库的程序](#一个使用-io-库的程序)
- [注释](#注释)
  - [C++ 中的注释](#c-中的注释)
  - [注释界定符不能嵌套](#注释界定符不能嵌套)
    - [不能嵌套](#不能嵌套)
    - [最佳实践](#最佳实践)
- [控制流](#控制流)
  - [while 语句](#while-语句)
  - [for 语句](#for-语句)
  - [读取数量不定的输入数据](#读取数量不定的输入数据)
  - [if 语句](#if-语句)
- [类简介](#类简介)
  - [书店问题](#书店问题)
  - [Sales_item 头文件](#sales_item-头文件)
  - [Sales_item 类](#sales_item-类)
    - [定义 Sales_item 类类型的变量](#定义-sales_item-类类型的变量)
    - [读写 Sales_item](#读写-sales_item)
    - [Sales_item 对象的加法](#sales_item-对象的加法)
  - [Sales_item 类成员函数](#sales_item-类成员函数)
  - [书店程序](#书店程序)
</p></details><p></p>


# 编写一个简单的 C++ 程序

## 一个简单的 C++ 程序

每个 C++ 程序都包含一个或多个函数(function), 其中一个必须命名为 `main`. 
操作系统通过调用 `main` 来运行 C++ 程序

```cpp
int main() {
    return 0;
}
```

## 一个函数的定义包含四部分

* 返回类型(return type)
* 函数名(function name)
* 一个括号 `()` 包围的形参列表(parameter list, 允许为空)
* 函数体(function body)


## main 函数 

- 返回类型必须为 `int`, 即整数类型
- 函数定义的最后一部分是函数体, 它是一个以做左花括号 (curly brace)开始,
  以右花括号结束的语句块 (block of statements)
- 函数体中的语句是 `return`, 当 `return` 语句包括一个值时, 
  此返回值的类型必须与函数的返回类型相同
- 在大多数系统中, `main` 的返回值被用来指示状态. 返回值 0 表明成功, 
  非 0 的返回值的含义由系统定义,通常用来指出错误类型

***

**Note:**

- 请注意, `return` 语句末尾的分号. 在 C++ 中, 大多数 C++ 语句以分号结束. 
  它们很容易被忽略, 如果忘记了分号, 就会导致莫名其妙的编译错误

***

# 输入输出

## C++ IO 标准库 iostream

C++ 语言并未定义任何输入输出 (IO) 语句, 取而代之, 
包含了一个全面的 **标准库** (standard library) `iostream` 
来提供 IO 机制(以及很多其他设施)

标准库 `iostream` 包含两个基础类型: 

- `istream`: 表示输入流
- `ostream`: 表示输出流

> 一个流就是一个字符序列, 是从 IO 设备读出或写入 IO 设备的. 
  术语 “流” (stream) 想表达的是, 随着时间的推移, 
  字符是顺序生成或消耗的

标准库 `iostream` 定义了 4 个标准输入输出 IO 对象: 

- 为了处理输入, 使用一个名为 **cin** (发音为 see-in)的 `istream` 类型的对象. 
  这个对象也被称为 **标准输入** (standard input)
- 对于输出, 使用一个名为 **cout** (发音为 see-out)的 `ostream` 类型的对象. 
  此对象也被称为 **标准输出** (standard output)
    - 另外一个 `ostream` 对象是 **cerr** (发音为 see-err), 通常用来输出警告和错误消息，
      因此它也被称为 **标准错误** (standard error)
    - 最后一个 `ostream` 对象是 **clog** (发音为 see-log), 用来输出程序运行时的一般性信息

系统通常将程序所运行的窗口与这些对象关联起来. 因此, 当我们读取 cin, 
数据将从程序正在运行的窗口读入, 当我们向 cout、cerr 和 clog 写入输入时, 
将会写到同一个窗口

## 一个使用 IO 库的程序

```cpp
#include <iostream>

/*
* 简单主函数:
* 读取两个数,求他们的和
* */
int main() {
    // 提示用户输入两个数
    std::cout << "Enter two numbers:" << std::endl;
    int v1 = 0, v2 = 0;  // 保存我们读入的输入数据的变量
    std::cin >> v1 >> v2;  // 读取输入数据
    std::cout << "The sum of " 
              << v1 << " and " << v2 
              << " is " << v1 + v2 << std::endl;
    return 0;
}
```

程序解释:

- `#include <iostream>` 告诉编译器我们想要使用 `iostream` 库, 
  尖括号中的名字指出了一个 **头文件** (header)
- 每个使用标准库设施的程序都必须包含相关的头文件
- `#include` 指令和头文件必须写在同一行中, 并且, 通常情况下, 
  `#include` 指令必须出现在所有函数之外. 
  我们一般将一个程序的所有 `#include` 指令都放在源文件的开始位置

# 注释

注释可以帮助人类读者理解程序. 注释通常用于概述算法, 确定变量的用途, 或者解释晦涩南通的代码段. 
编译器会忽略注释, 因此注释对程序的行为或性能不会有任何影响.

## C++ 中的注释

C++ 中有两种注释: 单行注释和界定符对注释:

- 单行注释以双划线 (//) 开始，以换行符结束. 当前双斜线右侧的所有内容都会被编译器忽略, 
  这种注释可以包含任何文本, 包括额外的双斜线
- 另一种注释使用继承自 C 语言的两个界定符 (`/*` 和 `*/`). 
  这种注释以 `/*` 开始, 以 `*/` 结束, 可以包含除 `*/` 以外的任意内容, 包含换行符. 
  编译器将落在 `/*` 和 `*/` 之间的所有内容都当做注释

```cpp
/*
 * 注释
 * 注释
 */
```

## 注释界定符不能嵌套

界定符对形式的注释是以 `/*` 开始，以 `*/` 结束的。因此，一个注释不能嵌套在另一个注释之内

### 不能嵌套

```cpp
/*
 * 注释对 /* */ 不能嵌套
 * "不能嵌套" 几个字符会被认为是源码，像剩余程序一样处理
*/

int main() {
    return 0;
}
```

### 最佳实践

通常需要在调试期间注释掉一些代码，由于这些代码可能包含界定符对形式的注释，
因此，可能导致注释嵌套错误，因此最好的方式是用单行注释方式注释掉代码段的每一行

```cpp
// /*
//  * 单行注释中的任何内容都会被忽略
//  * 包括嵌套的注释对也一样会被忽略
// */
```

# 控制流

## while 语句

```cpp
// while statement

#include <iostream>

int main() {
    int sum = 0;
    int val = 1;
    // 只要 val 的值小于10,while 循环就会持续执行
    while (val <= 10) {
        sum += val;  // 将 sum + val 赋予 sum
        ++val;  // 将 val 加 1
    }
    std::cout << "Sum of 1 to 10 inclusive is "
            << sum
            << std::endl;
    return 0;
}
```

## for 语句

```cpp
#include <iostream>

int main() {
    // currVal 是我们正在统计的数; 我们将读入的新值存入 val
    int currVal = 0;
    int val = 0;
    if (std::cin >> currVal) {
        int cnt = 1;
        while (std::cin >> val) {
            if (val == currVal) {
                ++cnt;
            }
            else {
                std::cout << currVal << " occurs " << cnt << " times" << std::endl;
            }
        }
        std::cout << currVal << " occurs " << cnt << " times" << std::endl;
    }
    return 0;
}
```

## 读取数量不定的输入数据

```cpp
#include <iostream>

int main() {
    int sum = 0;
    int value = 0;
    // 读取数据直到遇到文件尾,计算所有读入的值的和
    while (std::cin >> value) {
        sum += value;
    }
    std::cout << "Sum is: " << sum << std::endl;
    return 0;
}
```

## if 语句


```cpp
// 统计在输入中每个值连续出现了多少次
#include <iostream>

int main() {
    // currVal 是正在统计的数，将读入的新值存入 val
    int currVal = 0, val = 0;
    // 读取第一个数，并确保确实有数据可以处理
    if (std::cin >> currVal) {
        int cnt = 1;
        while (std::cin >> val) {
            if (val == currVal) {
                ++cnt;
            } else {
                std::cout << currVal << " occurs " 
                          << cnt << " times" << std::endl;
                currVal = val;
                cnt = 1;
            }
        }
        // 记住打印文件中最后一个值的个数
        std::cout << currVal << " occurs "
                  << cnt << " times" << std::endl;
    }
    return 0;
}
```

# 类简介

在 C++ 中，通过定义一个类(class)来定义自己的数据结构。
一个类定义了一个类型，以及与其关联的一组操作

类机制是 C++ 最重要的特性之一，
实际上，C++ 最初的一个设计焦点就是能定义使用上像内置类型一样自然的类类型(class type)

为了使用类，需要了解三件事情:

* 类名是什么？
* 类是在哪里定义的？
* 类支持什么操作？

为了使用标准库设施，必须包含相关的头文件。类似的，也需要使用头文件来访问为自己的应用程序所定义的类。
习惯上，头文件根据其中定义的类的名字来命名

通常使用 `.h` 作为头文件的后缀，但也有一些程序员习惯 `.H`、`.hpp` 或 `.hxx`。
标准库头文件通常不带后缀。编译器一般不关心头文件名的形式，但有的 IDE 对此有特定要求

## 书店问题

书店保存所有销售记录的档案，每条记录保存了某本书的一次销售信息(一册或多册)。
每条记录包含三个数据项:

```
0-201-70353-x   4   24.99
```

* 第一项是书的 ISBN 号(国际标准书号，一本书的唯一标识)
* 第二项是售出的册数
* 最后一项是书的单价

有时，书店老板需要查询此档案，计算每本书的销售量、销售额及平均售价

## Sales_item 头文件

头文件 `Sales_item.h` 中已经定义了 `Sales_item` 类

```cpp
// Sales_item.h

/* This file defines the Sales_item class used in chapter 1.
 * The code used in this file will be explained in
 * Chapter 7 (Classes) and Chapter 14 (Overloaded Operators)
 * Readers shouldn't try to understand the code in this file
 * until they have read those chapters.
*/

#ifndef SALESITEM_H
// we're here only if SALESITEM_H has not yet been defined 
#define SALESITEM_H

// Definition of Sales_item class and related functions goes here
#include <iostream>
#include <string>

class Sales_item {
    // these declarations are explained section 7.2.1, p. 270 
    // and in chapter 14, pages 557, 558, 561
    friend std::istream& operator>>(std::istream&, Sales_item&);
    friend std::ostream& operator<<(std::ostream&, const Sales_item&);
    friend bool operator<(const Sales_item&, const Sales_item&);
    friend bool 
    operator==(const Sales_item&, const Sales_item&);
    public:
        // constructors are explained in section 7.1.4, pages 262 - 265
        // default constructor needed to initialize members of built-in type
        Sales_item(): units_sold(0), revenue(0.0) { }
        Sales_item(const std::string &book): 
                    bookNo(book), units_sold(0), revenue(0.0) { }
        Sales_item(std::istream &is) { is >> *this; }
    public:
        // operations on Sales_item objects
        // member binary operator: left-hand operand bound to implicit this pointer
        Sales_item& operator+=(const Sales_item&);
        
        // operations on Sales_item objects
        std::string isbn() const { return bookNo; }
        double avg_price() const;
    // private members as before
    private:
        std::string bookNo;      // implicitly initialized to the empty string
        unsigned units_sold;
        double revenue;
};

// used in chapter 10
inline
bool compareIsbn(const Sales_item &lhs, const Sales_item &rhs) { 
    return lhs.isbn() == rhs.isbn(); 
}

// nonmember binary operator: must declare a parameter for each operand
Sales_item operator+(const Sales_item&, const Sales_item&);

inline bool 
operator==(const Sales_item &lhs, const Sales_item &rhs) {
    // must be made a friend of Sales_item
    return lhs.units_sold == rhs.units_sold &&
           lhs.revenue == rhs.revenue &&
           lhs.isbn() == rhs.isbn();
}

inline bool 
operator!=(const Sales_item &lhs, const Sales_item &rhs) {
    return !(lhs == rhs); // != defined in terms of operator==
}

// assumes that both objects refer to the same ISBN
Sales_item& Sales_item::operator+=(const Sales_item& rhs) {
    units_sold += rhs.units_sold; 
    revenue += rhs.revenue; 
    return *this;
}

// assumes that both objects refer to the same ISBN
Sales_item 
operator+(const Sales_item& lhs, const Sales_item& rhs) {
    Sales_item ret(lhs);  // copy (|lhs|) into a local object that we'll return
    ret += rhs;           // add in the contents of (|rhs|) 
    return ret;           // return (|ret|) by value
}

std::istream& 
operator>>(std::istream& in, Sales_item& s)
{
    double price;
    in >> s.bookNo >> s.units_sold >> price;
    // check that the inputs succeeded
    if (in)
        s.revenue = s.units_sold * price;
    else 
        s = Sales_item();  // input failed: reset object to default state
    return in;
}

std::ostream& 
operator<<(std::ostream& out, const Sales_item& s)
{
    out << s.isbn() << " " << s.units_sold << " "
        << s.revenue << " " << s.avg_price();
    return out;
}

double Sales_item::avg_price() const
{
    if (units_sold) 
        return revenue/units_sold; 
    else 
        return 0;
}
#endif
```

## Sales_item 类

在解决书店程序之前，定义一个数据结构来表示销售数据。

假定类名为 `Sales_item`，`Sales_item` 类的作用是表示一本书的总销售额、售出册数和平均售价。
`Sales_item` 类对象可以执行的操作:

* 定义 `Sales_item` 类类型的变量
* 调用一个名为 `isbn` 的函数从一个 `Sales_item` 对象中提取 ISBN 书号
* 用输入运算符(`>>`)和输出运算符(`<<`)读、写 `Sales_item` 类型的对象
* 用赋值运算符(`=`)将一个 `Sales_item` 对象的值赋予另一个 `Sales_item` 对象
* 用加法操作符(`+`)将两个 `Sales_item` 对象相加
    - 两个对象必须表示同一本书(相同的 ISBN)s
    - 加法结果是一个新的 `Sales_item` 对象，其 ISBN 与两个运算对选哪个相同，
      而其总销售额和售出册数则是两个运算对象的对应值之和
* 使用复合赋值运算符(`+=`)将一个 `Sales_item` 对象加到另一个对象上

### 定义 Sales_item 类类型的变量

```cpp
Sales_item item;
```

### 读写 Sales_item

从标准输入读入数据，存入一个 `Sales_item` 对象中，然后将 `Sales_item` 的内容写回到标准输出 

```cpp
#include <iostream>
#include "Sales_item.h"

int main() {
    // 新建 book 对象
    Sales_item book;
    // 读入 ISBN 号、售出的册数以及销售价格
    std::cin >> book;
    // 写入 ISBN 号、售出的册数、总销售额和平均价格
    std::cout << book << std::endl;

    return 0;
}
```

输入:

```
0-201-70353    4    24.99
```

输出:

```
0-201-70353    4    99.96   24.99
```

### Sales_item 对象的加法

将两个 `Sales_item` 对象相加:

```cpp
#include <iostream>
#include "Sales_item.h"

int main() {
    Sales_item item1, item2;
    std::cin >> item1 >> item2;
    std::cout << item1 + item2 << std::endl;

    return 0;
}
```

输入:

```
0-201-78345-X   3   20.00
0-201-78345-X   2   25.00
```

输出:

```
0-201-78345     5   101    22
```

## Sales_item 类成员函数

将两个 `Sales_item` 对象相加的程序首先应该检查两个对象是否具有相同的 ISBN

```cpp
#include <iostream>
#include "Sales_item.h"

int main() {
    Sales_item item1, item2;
    std::cin >> item1 >> item2;
    if (item1.isbn() == item2.isbn()) {
        std::cout << item1 + item2 << std::endl;
        return 0;  // 表示成功
    } else {
        std::cerr << "Data must refer to same ISBN"
                  << std::endl;
        return -1;  // 表示失败
    }
}
```

成员函数(member function) `isbn()` 是定义为 `Sales_item` 类的一部分的函数，
有时也被称为方法(method)。通常以一个类对象的名义来调用成员函数，
使用点运算符(`.`)来表达需要“名为 `item1` 的对象的 `isbn` 成员”

点运算符只能用于类类型的对象，其左侧运算对象必须是一个类类型的对象，
右侧运算对象必须是该类型的一个成员名，运算结果为右侧运算对象指定的成员

## 书店程序

```cpp
#include <iostream>
#include "Sales_item.h"

int main() {
    Sales_item total;
    if (std::cin >> total) {
        Sales_item trans;
        while (std::cin >> trans) {
            if (total.isbn() == trans.isbn()) {
                total += trans;
            } else {
                std::cout << total << std::endl;
                total = trans;
            }
        }
        std::cout << total << std::endl;
    } else {
        std::cerr << "No data?" << std::endl;
        return -1;
    }
    return 0;
}
```

