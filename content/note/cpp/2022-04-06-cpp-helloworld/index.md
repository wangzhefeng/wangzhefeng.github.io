---
title: C++ 快速入门
author: 王哲峰
date: '2022-04-06'
slug: cpp-helloworld
categories:
  - C++
tags:
  - tool
---


# 1.C++ 开始

## 1.1 编写一个简单的 C++ 程序

每个 C++ 程序都包含一个或多个函数(function), 其中一个必须命名为 `main`. 
操作系统通过调用 `main` 来运行 C++ 程序. 

```cpp
int main() {
    return 0;
}
```

一个函数的定义包含四部分: 

* 返回类型(return type)
* 函数名(function name)
* 一个括号 `()` 包围的形参列表(parameter list, 允许为空)
* 函数体(function body)


`main` 函数: 

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

## 1.2 初识输入输出

### 1.2.1 C++ IO 标准库 `iostream`

C++ 语言并未定义任何输入输出 (IO) 语句, 取而代之, 
包含了一个全面的 **标准库** (standard library) `iostream` 
来提供 IO 机制(以及很多其他设施). `iostream` 库包含两个基础类型: 

- `istream`: 表示输入流
- `ostream`: 表示输出流

一个流就是一个字符序列, 是从 IO 设备读出或写入 IO 设备的. 术语 “流” (stream) 想表达的是, 随着时间的推移, 
字符是顺序生成或消耗的. 

标准库 `iostream` 定义了 4 个标准输入输出 IO 对象: 

- 为了处理输入, 我们使用一个名为 **cin** (发音为 see-in) 的 istream 类型的对象. 
  这个对象也被称为 **标准输入** (standard input)
- 对于输出, 我们使用一个名为 **cout** (发音为 see-out) 的 ostream 类型的对象. 
  此对象也被称为 **标准输出** (standard output)
- 另外一个 ostream 对象是 **cerr** (发音为 see-err), 通常用来输出警告和错误消息，
  因此它也被称为 **标准错误** (standard error)
- 最后一个 ostream 对象是 **clog** (发音为 see-log), 用来输出程序运行时的一般性信息

系统通常将程序所运行的窗口与这些对象关联起来. 因此, 当我们读取 cin, 
数据将从程序正在运行的窗口读入, 当我们向 cout、cerr 和 clog 写入输入时, 
将会写到同一个窗口.

### 1.2.2 一个使用 IO 库的程序

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

- `#include <iostream>` 告诉编译器我们想要使用 iostream 库, 
  尖括号中的名字指出了一个 **头文件** (header)
- 每个使用标准库设施的程序都必须包含相关的头文件
- `#include` 指令和头文件必须写在同一行中, 并且, 通常情况下, 
  `#include` 指令必须出现在所有函数之外. 
  我们一般将一个程序的所有 `#include` 指令都放在源文件的开始位置





## 1.3 注释

注释可以帮助人类读者理解程序. 注释通常用于概述算法, 确定变量的用途, 或者解释晦涩南通的代码段. 
编译器会忽略注释, 因此注释对程序的行为或性能不会有任何影响.

### 1.3.1 C++ 中注释的种类

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

### 1.3.2 注释界定符不能嵌套

## 1.4 控制流

### 1.4.1 `while` 语句

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

### 1.4.2 `for` 语句

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

### 1.4.3 读取数量不定的输入数据

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

### 1.4.4 `if` 语句


```cpp
    
```    


## 1.5 类简介



