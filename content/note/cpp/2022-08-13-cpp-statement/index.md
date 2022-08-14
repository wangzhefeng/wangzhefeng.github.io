---
title: C++ 语句
author: 王哲峰
date: '2022-08-13'
slug: cpp-statement
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
</style>

<details><summary>目录</summary><p>

- [内容](#内容)
- [简单语句](#简单语句)
  - [空语句](#空语句)
  - [别漏写分号也别多写分号](#别漏写分号也别多写分号)
  - [复合语句(块)](#复合语句块)
- [语句作用域](#语句作用域)
- [条件语句](#条件语句)
  - [if 语句](#if-语句)
    - [if 语句语法](#if-语句语法)
    - [垂悬 else](#垂悬-else)
  - [switch 语句](#switch-语句)
    - [switch 语句语法](#switch-语句语法)
    - [switch 内部的控制流](#switch-内部的控制流)
    - [case 标签](#case-标签)
    - [default 标签](#default-标签)
- [迭代语句](#迭代语句)
  - [while 语句](#while-语句)
    - [while 语句语法](#while-语句语法)
  - [传统的 for 语句](#传统的-for-语句)
  - [范围 for 语句](#范围-for-语句)
  - [do while 语句](#do-while-语句)
- [跳转语句](#跳转语句)
  - [break 语句](#break-语句)
  - [continue 语句](#continue-语句)
  - [goto 语句](#goto-语句)
- [try 语句和异常处理](#try-语句和异常处理)
  - [throw 表达式](#throw-表达式)
  - [try 语句块](#try-语句块)
  - [标准异常](#标准异常)
</p></details><p></p>

# 内容

* 条件执行语句
* 循环语句
* 控制流跳转语句

# 简单语句

C++ 语言中的大多数语句都以分号结束，
一个表达式末尾加上分号就变成了表达式语句(expression statement)。
表达式语句的作用是执行表达式并丢弃掉求值结果

```cpp
ival + 5;  // 一条没什么实际用处的表达式
cont << ival;  // 一条有用的表达式语句
```

## 空语句

最简单的语句是空语句(null statement)，空语句中只有一个单独的分号：

```cpp
;  // 空语句
```

如果在程序的某个地方，语法上需要一条语句但是逻辑上不需要，
此时应该使用空语句，一种常见的情况是，当循环的全部工作在条件部分就可以完成时，
我们通常会用到空语句

```cpp
// 重复读入数据直至到达文件末尾或某次输入的值等于 sought
while (cin >> s && s != sought)
    ;  // 空语句
```

***
**Note:**

* 使用空语句时应该加上注释，从而令读这段代码的人知道该语句是有意沈略的
***

## 别漏写分号也别多写分号

因为空语句是一条语句，所以可用在任何允许使用语句的地方。
由于这个原因，某些看起来非法的分号往往只不过是一条空语句而已。
从语法上说得过去

```cpp
ival = v1 + v2;;  // 正确: 第二个分号表示一条多余的空语句
```

多余的空语句一般来说是无害的，
但是如果在 `if` 或者 `while` 的条件后面跟了一个额外的分号就可能完全改变程序员的初衷

```cpp
// 出现了糟糕的情况: 额外的分号，循环体是那条空语句, 将无休止地循环下去
// 虽然从形式上来看执行递增运算的语句前面有缩进，但是它并不是循环的一部分
while (iter != svec.end()) ;  // while 循环体是那条空语句
    ++iter;  // 递增运算不属于循环的一部分
```

***
**Note:**

* 多余的空语句并非总是无害的
***

## 复合语句(块)

复合语句(compound statement)是指用花括号括起来的(可能为空的)语句和声明的序列，
复合语句也被称为块(block)。一个块就是一个作用域，
在块中引入的名字只能在块内部以及嵌套在块中的子块里访问。
通常，名字有限的区域内可见，该区域从名字定义处开始，到名字所在的(最内层)块的结尾为止

如果在程序的某个地方，语法上需要一条语句，但是逻辑上需要多条语句，则应该使用复合语句

***
**Note:**

* 块不一分号作为结束
***

# 语句作用域

可以在 `if`、`switch`、`while`、和 `for` 语句的控制结构内定义变量。
定义在控制结构当中的变量只在相应语句的内部可见，一旦语句结束，变量也就超出其作用范围了

```c++
while (int i = get_num())  // 每次迭代时创建并初始化 i
    cout << i << endl;

i = 0;  // 错误: 在循环外部无法访问 1
```

如果其他代码也需要访问控制变量，则变量必须定义在语句的外部:

```cpp
// 寻找第一个负值元素
auto beg = v.begin();
while (beg != v.end() && *beg >= 0)
    ++beg;

if (beg == v.end())
    // 此时我们知道 v 中的所有元素都大于等于 0
```

# 条件语句

## if 语句

### if 语句语法

* `if`

```cpp
if (condition)
    statement
```

* `if else`

```cpp
if (condition)
    statement1
else
    statement2
```

* `if ... else if ...`

```cpp
if (condition)
    statement1
else if (condition)
    statement2
else
    statement3
```

有些编码风格要求在 `if` 或 `else` 之后必须写上花括号，对于 `while` 和 `for` 语句的循环体两端也有同样的要求。
这么做的好处是可以避免代码混乱不堪，以后修改代码时如果想添加别的语句，也可以很容易地找到正确的位置

### 垂悬 else

当一个 `if` 语句嵌套在另一个 `if` 语句内部时，很可能 `if` 分支会多于 `else` 分支。
这时问题出现了: 我们怎么知道某个给定的 `else` 是和哪个 `if` 匹配呢？

这个问题通常称为垂悬 else(dangling else)，
在那些既有 `if` 语句又有 `if else` 汉语句的编程语言中是个普遍存在的问题。
不同语言解决该问题的思路也不同，就 C++ 而言，
它规定 `else` 与离它最近的尚未匹配的 `if` 匹配，从而消除了程序的二叉性。
可以使用花括号控制执行路径

## switch 语句

switch 语句(switch statement)提供了一条便利的途径使得我们能够在若干固定选项中做出选择

### switch 语句语法

```cpp
switch (expression) {
    case value1:
        statement1;
        break;
    case value2:
        statement2;
        break;
    case value3:
        statement3;
        break;
}
```

### switch 内部的控制流

switch 语句首先对括号里的表达式求值，
该表达式紧跟在关键字 switch 的后面，可以是一个初始化的变量声明。
表达式的值转换成整数类型，然后与每个 case 标签的值比较

* 如果表达式和某个 case 标签的值匹配成功，程序从该标签之后的第一条语句开始执行，
  直到到达了 switch 的结尾或者是遇到了一条 break 语句为止
* 如果 switch 语句的表达式和所有 case 都没有匹配上，
  将直接跳转到 switch 结构之后的第一条语句

如果某个 case 标签匹配成功，将从该标签开始往后顺序执行所有 case 分支，
除非程序显式地中断了这一过程，否则直到 swtich 的结尾处才会停下来。
要想避免执行后续 case 分支的代码，必须显式地告诉编译器终止执行过程。
大多数情况下，在下一个 case 标签之前应该有一条 break 语句。
有一种常见的错觉是程序只执行匹配成功的那个 case 分支的语句，
实际上如果漏写 break 语句，程序会直接执行后面 case 标签后的代码，
直到遇到 break 语句或者 switch 语句结束

然而，也有一些时候默认的 switch 行为才是程序真正需要的。
每个 case 标签只能对应一个值，但是有时候我们希望两个或更多个值共享同一组操作。
此时，我们就故意省略掉 break 语句，使得程序能够连续执行若干个 case 标签

```cpp
unsigned vowelCnt = 0;
// ...
switch (ch) {
    // 出现了 a、e、i、o、u 中的任意一个都会将 vowelCnt 的值加 1
    case 'a':
    case 'e':
    case 'i':
    case 'o':
    case 'u':
        ++vowelCnt;
        break;
}
```

C++ 程序的形式比较自由，所以 case 标签之后不一定非得换行。
把几个标签写在一行里，强调这些 case 代表的是某个范围内的值

```cpp
switch (ch) {
    // 另一种合法的书写形式
    case 'a': case 'e': case 'i': case 'o': case 'u':
        ++vowelCnt;
        break;
}
```

### case 标签

case 关键字和它对应的值一起被称为 case 标签(case label)。case 标签必须是整型常量表达式。
任何两个 case 标签的值不能相同，否则就会引发错误。另外，default 也是一种特殊的 case 标签

### default 标签






# 迭代语句

## while 语句

### while 语句语法

```cpp
while (condition)
    statement
```

## 传统的 for 语句


## 范围 for 语句


## do while 语句



# 跳转语句

## break 语句


## continue 语句

## goto 语句

# try 语句和异常处理

## throw 表达式


## try 语句块


## 标准异常