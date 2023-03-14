---
title: Java 面向对象编程
author: 王哲峰
date: '2020-10-01'
slug: java-oop
categories:
  - java
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

- [面向对象程序设计概述](#面向对象程序设计概述)
  - [类](#类)
  - [对象](#对象)
  - [类之间的关系](#类之间的关系)
- [class 和 instance](#class-和-instance)
  - [类(class)](#类class)
  - [实例(instance)](#实例instance)
  - [封装(encapsulation)](#封装encapsulation)
- [instance field](#instance-field)
  - [实例域](#实例域)
  - [final 实例域](#final-实例域)
  - [static 实例域](#static-实例域)
- [method](#method)
  - [构造器](#构造器)
  - [public method](#public-method)
  - [private method](#private-method)
  - [static method](#static-method)
  - [更改器方法和访问器方法](#更改器方法和访问器方法)
  - [方法参数](#方法参数)
    - [可变参数](#可变参数)
    - [隐式参数与显式参数](#隐式参数与显式参数)
- [package](#package)
  - [类的导入](#类的导入)
  - [静态导入](#静态导入)
  - [将类放入包中](#将类放入包中)
  - [包作用域](#包作用域)
- [类路径](#类路径)
  - [类文件的设计](#类文件的设计)
  - [设置类路径](#设置类路径)
- [文档注释](#文档注释)
  - [注释的插入](#注释的插入)
  - [类注释](#类注释)
  - [方法注释](#方法注释)
  - [域注释](#域注释)
  - [通用注释](#通用注释)
  - [包与概述注释](#包与概述注释)
  - [注释的抽取](#注释的抽取)
- [类设计技巧](#类设计技巧)
- [继承](#继承)
  - [类、超类、子类](#类超类子类)
    - [定义子类](#定义子类)
  - [Object: 所有类的超类](#object-所有类的超类)
  - [反省数组列表](#反省数组列表)
  - [对象包装器、自动装箱](#对象包装器自动装箱)
  - [参数数量可变的方法](#参数数量可变的方法)
  - [枚举类](#枚举类)
  - [反射](#反射)
  - [继承的设计技巧](#继承的设计技巧)
</p></details><p></p>

面向对象编程，是一种通过对象的方式，把现实世界映射到计算机模型的一种编程方法。

- Java 面向对象的基本概念:
    - 类
        - 公有类 `public ClassName {}`
        - 普通类 `ClassName {}`
    - 实例
        - 构造器创建实例对象
    - 实例域
        - 普通私有实例域 `private` 
        - final 实例域 `private final`
        - 静态实例域 `private static`
    - 方法
        - 构造方法 `private ClassName() {}`
        - main 方法 `public static void main(String[] args) {}`
        - 公有方法 `public methodName() {}`
        - 私有方法 `private methodName() {}`
- Java 面向对象的实现方式:
    - 继承
    - 多态
- Java 语言本身提供的机制:
    - 包 package
    - 类路径 classpath
    - jar
- Java 标准库提供的核心类
    - 字符串
    - 包装类型
    - JavaBean
    - 枚举
    - 常用工具类

# 面向对象程序设计概述

面向对象的程序是由对象组成的，每个对象包含对用户公开的特定功能部分和隐藏的实现部分。
程序中的很多对象来自标准库，还有一些是自定义的。

在 OOP 中，不必关心对象的具体实现，只要能够满足用户的需求即可。

## 类

- **类(class)**: 是构造对象的模板或蓝图，用 Java 编写的所有代码都位于某个类的内部。类基本分为两类:
    - 标准的 Java 类库提供了几千个类，可以用于用户界面设计、日期、日历、网络程序设计
    - 用户自定义类
        - 公有类 `public class`
        - 普通类 `class`
- **实例(instance)**: 由类构造(construct)对象的过程称为创建类的实例(instance)
- **封装(encapsulation)**: 有时称为数据隐藏，是与对象有关的一个重要概念。从形式上看，封装不过是将数据和行为组合在一个包中，
  并对对象的使用者隐藏了数据的实现方式
    - 对象中的数据称为 **实例域(instance field)**
    - 操纵数据的过程称为 **方法(method)**
    - 对于每个特定的类实例(对象)都有一组特定的实例域值，这些值的集合就是这个对象的当前 **状态(state)**
    - 实现封装的关键在于绝对不能让类中的方法直接地访问其他类的实例域，程序仅通过对象的方法与对象数据进行交互
- 在 Java 中，所有的类都源自于一个“神通广大的超类”，它就是 `Object`

## 对象

要想使用 OOP，一定要清楚对象的三个主要特性：

- 对象的行为(behavior): 可以对对象施加哪些操作，或可以对对象施加哪些方法
- 对象的状态(state): 当施加那些方法时，对象如何响应
- 对象识别(identity): 如何辨识具有相同行为与状态的不同对象

## 类之间的关系

在类之间，最常见的关系有：

- 依赖(dependence): `uses-a`
- 聚合(aggregation): `has-a`
- 继承(inheritance): `is-a`

# class 和 instance

## 类(class)

如何设计复杂应用程序所需要的各种主力类(workhorse class)，通常，这些类没有 `main` 方法，
却有自己的实例域和实例方法。要想创建一个完整的程序，应该将若干类组合在一起，
其中只有一个类有 `main` 方法。

语法:

- 公有类

```java
// ----------------------
// ClassNameTest.java
// ----------------------
public class ClassNameTest {
    public static void main(String[] args) {
        ...
    }
}
```

- 公有类结构分析
    - 源文件名是 `ClassNameTest.java`，这是因为文件名必须与 **public 类** 的名字相匹配。
      在一个源文件中，只能有一个公有类(public)，但可以有任意数量的非共有类
    - 非公有类

```java
// ----------------------
// ClassName.java file 
// ----------------------
class ClassName {
    // instance field
    field1
    field2
    ...

    // constructor
    constructor1
    constructor2
    ...

    // method 
    method1
    method2
    ...
}
```

- 非公有类结构分析
    - `ClassName` 类的实例中有三个实例域用来存放要操作的数据，关键字 `private` 确保只有 `ClassName` 
      类自身的方法能够访问这些实例域，而其他类的方法不能够读写这些域
    - `ClassName` 类包含一个构造器(constructor)和多个方法(method)，构造器也是一种方法
    - `ClassName` 类的所有方法都被标记为 `public`，关键字 `public` 意味着任何类的任何方法都可以调用这些方法

## 实例(instance)

- 在 Java 程序设计语言中，使用 **构造器(constructor)** 构造新 **实例(instance)**
- 构造器是一种特殊的方法，用来构造并初始化对象，构造器与其他方法有一个不同，要想构造一个对象，需要在构造器前面加上 `new` 操作符
- 构造器的名字应该与类名相同
- 在 Java 中，任何对象变量的值都是对存储在另一个地方的对象的引用，`new` 操作符的返回值也是一个引用
- 在构造类的对象时，构造器会运行，以便将实例域初始化为所希望的状态

## 封装(encapsulation)

在有些时候，需要获得或设置实例域的值。因此，一个基本的类应该提供下面三项内容：

- 一个私有的数据域
    - `private ClassName field;`
- 一个公有的域访问器方法
    - `public ClassName getXXX() {}`
- 一个公有的域更改器方法
    - `public ClassName setXXX() {}`

这样做比提供一个简单的公有数据域复杂些，但是却有着下列明显的好处：

- 首先，可以改变内部实现，除了该类的方法之外，不会影响其他代码
- 其次，更改器方法可以执行错误检查，然而直接对域进行赋值将不会进行这些处理

如果需要返回一个可变对象的引用，应该首先对它进行克隆(clone)。对象 clone 是指存放在另一个位置上的对象副本:

```java
class Employee {
    ...
    public Date getHireDay() {
        return (Date) hireDay.clone();
    }
}
```


# instance field

## 实例域

- 类定义时可以定义实例域来存放将要操作的数据，实例域必须标记为 `private` 以确保只有类自身的方法能够访问这些实例域，而其他类的方法不能够读写这些域

## final 实例域

可以将实例域定义为 `final`，构建对象时必须初始化这样的域。也就是说，必须确保在每一个构造器执行之后，这个域的值被设置，并且在后面的操作中，不能够在对它进行修改。

- final 修饰符大都应用于基本 (primitive) 类型域，或不可变(immutable)类的域
    - 如果类中的每个方法都不会改变其对象，这种类就是不可变类
- 对于可变的类，使用 final 修饰符可能会对读者造成混乱

## static 实例域

如果将域定义为 `static`，每个类中只有一个这样的域。而每个对象对于所有的实例域却都有自己的一份拷贝。

- 静态域属于类，而不属于任何独立的对象(静态域被称为类域)

# method

一个 `class` 可以包含多个 `field`. 但是，直接把 `field` 用 
`public` 暴露给外部可能会破坏封装性。显然，直接操作 `field`，容易造成逻辑混乱。

为了避免外部代码直接去访问 `field`，我们可以用 `private` 修饰 `field`，
拒绝外部访问. 把 `field` 从 `public` 改成 `private`，
外部代码不能访问这些 `field`，那我们定义这些 `field` 有什么用？
怎么才能给它赋值？怎么才能读取它的值？所以我们需要使用方法(`method`)
来让外部代码可以间接修改 `field`.

所以，一个类通过定义方法，就可以给外部代码暴露一些操作的接口，同时，内部自己保证逻辑一致性。

## 构造器

构造器是一种特殊的方法，用来构造并初始化对象，构造器与其他方法有一个不同，要想构造一个对象，需要在构造器前面加上 `new` 操作符

**构造器** 与 **public 类** 同名, 在构造类的对象时，构造器会运行，以便将实例域初始化为所希望的状态。

- 构造器被标记为 `public`
- 构造器与类同名
- 每个类可以有一个以上的构造器
- 构造器可以有0个、1个或多个参数
- 构造器没有返回值
- 构造器与其他的方法有一个重要的不同：构造器总是伴随着 `new` 操作符的执行被调用，
  而不能对一个已经存在的对象调用高早期来达到重新设置实例域的目的


> 不要在构造器中定义与实例域重名的局部变量

## public method

## private method

- 在实现一个类时，由于公有数据非常危险，所以应该将所有的 **数据域** 都设置为私有的
- 尽管绝大多数方法都被设计为公有的，但在某些特殊情况下，也可能将它们设计为私有的，
  为了实现一个私有的方法，只需将关键字 `public` 设计为 `private` 即可

## static method

## 更改器方法和访问器方法

- 更改器方法(mutator method)
    - 调用方法后，原对象的状态会改变
- 访问器方法(accessor method)
    - 只访问对象而不修改对象的方法

## 方法参数

方法可以包含 0 个或任意个参数，方法参数用于接收传递给方法的的变量值

- 调用方法时，必须严格按照参数的定义一一传递
- 调用方把参数传递给实例方法时，调用时传递的值会按参数位置一一绑定

### 可变参数

可变参数用 `类型...` 定义，可变参数相当于数组类型

### 隐式参数与显式参数

在方法内部，可以使用一个隐含的变量 `this`，它始终指向当前实例。
因此，通过 `this.field` 就可以访问当前实例的字段.

如果有局部变量和字段重名，那么局部变量优先级更高，就必须加上 `this`.

方法用于操作对象以及存取它们的实例域，在每一个方法中，关键字 `this` 表示隐式参数。

示例：

```java
public void raiseSalary(double byPercent) {
    double raise = salary * byPercent / 100;
    salary += raise;
}

number007.raiseSalary(5);
```

```java
// 下面的语句与 number007.raiseSalary(5); 相同效果
double raise = number007.salary * 5 /100;
number007.salary += raise;
```

`raiseSalary` 方法有两个参数：

- 第一个参数称为隐式参数(implicit)，是出现在方法名前的类对象
- 第二个参数位于方法名后面括号中的数值，这是一个显示参数(expplicit)


示例:

```java
public void raiseSalary(double byPercent) {
    double raise = this.salary * byPercent / 100;
    this.salary += raise;
}
```

# package

Java 允许使用 **包(package)** 将类组织起来。借助于包可以方便地组织自己的代码，
并将自己的代码与别人提供的代码库分开管理。使用包的主要原因是确保类名的唯一性。

标准的 Java 类库分布在多个包中，包括 `java.lang`、`java.util`、`java.net` 等。
标准的 Java 包具有一个层次结构，如同硬盘的目录嵌套一样，也可以使用嵌套层次组织包。
所有标准的 Java 包都处于 `java` 和 `javax` 包层次中。

- `java` 
    - `java.lang`
    - `java.util`
    - `java.net`
    - `java.sql`
- `javax`
    - test1
    - test2
    - test3

## 类的导入

一个类可以使用所属包中的所有类，以及其他包中的 **公有类(public class)**。

- 可以采用两种方式访问另一个包中的公有类：
    - 第一种方式是在每个类名之前添加完整的包名

    ```java
    java.time.LocalDate today = java.time.LocalDate.now();
    ```

    - 第二种方式是使用 `import` 语句。import 语句应该位于源文件的顶部，位于 package 语句的后面
        - import 语句是一种引用包含在包中的类的简明描述，一旦使用了 import 语句，在使用类时，就不必写出包的全名来了
        - 可以使用 import 语句导入一个特定的类或者整个包

    ```java
    // 包名
    package PackageName;

    // 导入 java.time 包中的 LocalDate 类
    import java.time.LocalDate;

    // 导入 java.util 包中的所有的类
    import java.util.*;

    LocalDate today = LocalDate.now();
    ```

> 需要注意的是：只能使用 `*` 导入一个包，
> 而不能使用 `import java.*` 或 `import java.*.*` 导入以 `java` 为前缀的所有包
> 
> - 在大多数情况下，只导入所需的包，并不必过多地理睬它们。但在发生命名冲突时候，就不能不注意包的名字了
> - 例如：java.util 和 java.sql 包中都有 Date 类。如果在程序中导入了这两个包：
> 
> ```java
> // 错误的做法
> import java.util.*;
> import java.sql.*;
> 
> // 正确的做法(需要使用 java.sql.Date 类)
> import java.util.*;
> import java.sql.*;
> import java.sql.Date;
> 
> // 如果需要使用两个包中的 Date 类
> java.util.Date deadline = new java.util.Date();
> java.sql.Date today = new java.sql.Date();
> ```
> 
> - 在包中定位类是编译器(compile)的工作，类文件中的字节码肯定使用完整的包名来引用包名

## 静态导入

import 语句不仅可以导入类，还增加了导入 **静态方法** 和 **静态域** 的功能。


```java
// 导入 System 类的静态方法和静态域
import static java.lang.System.*;

out.println(Goodbye, World!); // i.e., System.out
exit(0);                      // i.e., System.exit()

// 导入特定的方法域
import static java.lang.System.out;
import static java.Math.sqrt;
import static java.Math.pow;

sqrt(pow(x, 2) + pow(y, 2));
```

## 将类放入包中

要想将一个类放入包中，就必须将包的名字放在源文件的开头，包中定义类的代码之前

- 如果没有在源文件中放置 `package` 语句，这个源文件中的类就被放置在一个 **默认包(default package)** 中。默认包是一个没有名字的包。
- 将包中的文件放到与完整的包名匹配的子目录中

示例：

- 包 `com.horstmann.corejava` 中的所有源文件应该被放置在子目录 `com/horstmann/corejava` 中，
编译器将类文件也放在相同的目录结构中。

```java
// PackageTest 类放置在默认包中
// PackageTest/PackageTest.java
import com.horstmann.corejava.*;
public class PackageTest {
    ...
}
```

```java
// Employee 类放置在 com.horstmann.corejava 包中，因此，
// Employee.java 文件必须包含在子目录 com/horstmann/corejava 中
// PackageTest/com/horstmann/corejava/Employee.java
package com.horstmann.corejava;

public class Employee {
    ...
}
```

- 编译程序
        
```bash
javac PackageTest.java
```

> - 编译器对文件(带有文件分隔符合扩展名 `.java` 的文件)进行操作
> - Java 解释器加载类(带有 `.` 分隔符)

## 包作用域

- 标记为 public 的部分可以被任意的类使用
- 标记为 private 的部分只能被定义它们的类使用
- 如果没有指定 public 或 private，这个部分(类、方法、变量)可以被同一个包中的所有方法访问

# 类路径

## 类文件的设计

- 类存储在文件系统的子目录中，类的路径必须与包名匹配
- 类文件也可以存储在 JAR(Java 归档)文件中
    - 在一个 JAR 文件中，可以包含多个压缩形式的类文件和子目录，这样既可以节省又可以改善性能
    - 在程序中用到第三方(third-party) 的库文件时，通常会给出一个或多个需要包含的 JAR 文件
    - JDK 也提供了许多的 JAR 文件，例如：在 `jre/lib/rt.jar` 中包含数千个类库文件
    - JAR 文件使用 ZIP 格式组织文件和子目录。可以使用所有 ZIP 实用程序查看内部的 `rt.jar` 以及其他的 JAR 文件
- 为了使类能够被多个程序共享，需要做到下面几点：
    - (1)把类放到一个目录中:
        - `/home/user/classdir`，需要注意：这个目录是包树状结构的基目录
    - (2)将 JAR 文件放在一个目录中:
        - `/home/user/archives`
    - (3)设置 **类路径(class path)**。类路径是所有包含类文件的路径的集合:
        - UNIX 环境：`/home/user/classidr:.:/home/user/archives/archive.jar`
            - `.` 表示当前目录
        - Windows 环境： `c:\classdir;.;c:\archives\archive.jar`
            - `.` 表示当前目录
    - (4)Java 类路径包括:
        - 基目录 `/home/user/classdir` 或 `c:\classes`
        - 当前目录 `.`
        - JAR 文件 `/home/user/archives/archive.jar` 或 `c:\archives\archive.jar`

## 设置类路径

- 最好采用 `-classpath` 或 `-cp` 选项指定类路径:

```bash
# UNIX
java -classpath /home/user/classdir:.:/home/user/archives/archive.jar MyProg

# Windows shell
java -classpath c:\classdir;.;c:\archives\archive.jar MyProg
```

- 通过设置 `CLASSPATH` 环境变量也可以指定类路径:

```bash
# UNIX
export CLASSPATH=/home/user/classdir:.:/home/user/archives/archive.jar

# Windows shell
set CLASSPATH=c:\classdir;.;c:\archives\archive.jar
```

# 文档注释

JDK 包含一个很有用的工具，叫做 `javadoc`，它可以由源文件生成一个 HTML 文档。

如果在源代码中添加以专用的定界符 `/**` 开始的注释，那么可以很容易地生成一个看上去具有专业水准的文档。

## 注释的插入

javadoc 实用程序(utility) 从下面几个特性中抽取信息：

- 包
- 公有类与接口
- 公有的和受保护的构造器及方法
- 公有的和受保护的域

应该为上面几部分编写注释。注释应该放置在所描述特性的前面：

- 注释以 `/**` 开始，并以 `*/` 结束
- 每个 `/** ... */` 文档注释在 **标记** 之后紧跟着自 **由格式文本(free-form text)**，标记由 `@` 开始
    - 自由格式文本的第一句应该是一个概要性的句子，`javadoc` 使用程序自动地讲这些句子抽取出来形成概要页
    - 在自由格式文本中，可以使用 HTML 修饰符，例如：
        - `<em>...</em>` 用于强调
        - `<strong></strong>` 用户着重强调
        - `<img ...>` 包含图像
        - 一定不要使用 `<h1>` 或 `<hr>`，因为它们会与文档的格式产生冲突
        - `{@code ...}` 用于键入等宽代码

## 类注释

类注释必须放在 `import` 语句之后，类定义之前。

- 示例:

```java
import java.util.*;
import java.sql.*;
import java.lang.*;
/**
    * A {@code Card} object represents a playing card, such as "Queen of Hearts". 
    * A card has a suit (Diamond, Heart, Spade or Club)
    * and a value (1 = Ace, 2 ... 10, 11 = Jack, 12 = Queen, 13 = King)
    */
public class Card {
    ...
}
```

## 方法注释

每一个方法注释必须放在所描述的方法之前。除了通用标记之外，还可以使用下面的标记：

- `@param` 变量描述
    - 这个标记将对当前方法的 “param(参数)” 部分添加一个条目，一个方法的所有 `@param` 标记必须放在一起
      这个描述可以占据多行，并可以使用 HTML 标记
- `@return` 描述
    - 这个标记将对当前方法添加 “return(返回)” 部分。
      这个描述可以跨越多行，并可以使用 HTML 标记
- `@throws` 类描述
    - 这个标记将添加一个注释，用于表示这个方法有可能抛出异常

示例:

```java
import java.util.*;
/**
    * Raise the salary of an employee.
    * @param byPercent the percentage by which to raise the salary (e.g. 10 means 10%)
    * @return the amount of the raise
    */
public double raiseSalary(double byPercent) {
    double raise = salary * byPercent / 100;
    salary += raise;
    return raise;
}
```

## 域注释

只需要对公有域(通常指的是静态常量)建立文档。

示例:

```java
/**
    * The "Hearts" card suit
    */
public static final int HEARTS = 1;
```

## 通用注释

下面的标记可以用在类文档的注释中：

- @author
    - 作者，可以用多个
- @version
    - 版本
- @since
    - 始于
- @deprecated
    - 这个标记将对类、方法、变量添加一个不再使用的注释
        - @deprecated Use <code> setVisible(true) </code> instead
- @see
    - 这个标记将在 "see also" 部分增加一个超级链接。它可以用于类中，也可以用于方法中。
    - 这里的引用可以选择下列情形之一：

```
pacakge.class@feature label
<a href="...">label</a>
"text"
```

## 包与概述注释

可以直接将类、方法、变量的注释放置在 Java 源文件中，只要用 `/** ... */` 文档注释界定就可以了。
但是，要想产生包注释，就需要在每一个包目录中添加一个单独的文件。可以有如下两个选择：

- (1)提供一个以 `package.html` 命名的 HTML 文件。
    - 在标记 `<body></body>` 之间的所有文本都会被抽取出来
- (2)提供一个以 `pacakge-info.java` 命名的 Java 文件。
    - 这个文件必须包含一个初始的以 `/**` 和 `*/` 界定的 `Javadoc` 注释，跟随在一个包语句之后，它不应该包含更多的代码或注释
- (3)还可以为所有的源文件提供一个概述性的注释。这个注释将被放置在一个名为 `overview.html` 的文件中，这个文件位于包含所有源文件的父目录中。
  标记 `<body></body>` 之间的所有文本将被抽取出来。当用户从导航栏中选择 “Overview” 时，就会显示出这些注释内容。


## 注释的抽取

假设 HTML 文件将被存放在目录 `docDirectory` 下，执行以下的步骤：

- (1) 切换到包含想要生成文档的源文件目录
- (2) 运行 `javadoc` 命令生成文档
    - 如果是一个包，应该运行命令

    ```bash
    javadoc -d docDirectory nameOfPackage
    ```

    - 对于多个包生成文档

    ```bash
    javadoc -d docDirectory nameOfPackage1 nameOfPackage2 ...
    ```

    - 如果文件在默认包中

    ```bash
    javadoc -d docDirectory *.java
    ```

    - 如果省略了 `-d docDirectory` 选项，那 HTML 文件就会被提取到当前目录下(不推荐)

- (3) 可以使用多种形式的命令行选项对 `javadoc` 程序进行调整
    - `javadoc -d docDirectory -author -version`
        - 在文档中包含 `@author` 和 `@version` 标记
    - `javadoc -d docDirectory -link http://docs.oracle.com/javase/8/docs/api *.java`
        - 用来为标准类添加超链接，所有的标准类库都会自动地链接到 Oracle 网站的文档
    - `javadoc -d docDirectory -linksource`
        - 每个源文件被转换为 HTML，并且每个类和方法名将转变为指向源代码的超链接
    - 其他选项
        - `javadoc 实用程序的联机文档 <https://docs.oracle.com/javase/8/docs/technotes/guides/javadoc/index.html>`_ 

# 类设计技巧

1. 一定要保证数据私有;
2. 一定要对数据初始化;
3. 不要在类中使用过多的基本类型;
4. 不是所有的域都需要独立的域访问器和域更改器;
5. 将职责过多的类进行分解;
6. 类名和方法名要能够体现它们的职责;
7. 优先使用不可变的类;

# 继承

利用继承(inheritance)，可以基于已存在额类构造一个新类。继承已存在的类就是复用(继承)这些类的方法和域。
在此基础上，还可以添加一些新的方法和域，以满足新的需求。这是 Java 程序设计中的一项核心技术。

反射(reflection)是指在程序运行期间发现更多的类及其属性的能力。这是一个功能强大的特性，使用起来也比较复杂。

##  类、超类、子类

### 定义子类

关键字 `extends` 表示继承。关键字 `extends` 表明正在构造的 **新类** 派生于一个 **已存在的类**

- 已存在的类称为 **超类(superclass)**、`基类(base class)`、`父类(parent class)`
- 新类称为 **子类(subclass)**、**派生类(derived class)**、**孩子类(child class)**

示例：

```java
// 父类: 雇员
public class Employee {}

// 子类: 经理
public class Manager extends Employee {
    private double bonus;
    ...
    public void setBonus(double bonus) {
        this.bonus = bouns;
    }
}
```

在通过扩展超类定义子类的时候，仅需要指出子类与超类的不同之处。因此在设计类的时候，应该将通用的方法放在超类中，
而将具有特殊用途的方法放在子类中，这种将通用的功能放到超类的做法，在面向对象程序设计中十分普遍。

## Object: 所有类的超类

## 反省数组列表

## 对象包装器、自动装箱

## 参数数量可变的方法

## 枚举类

## 反射

## 继承的设计技巧

