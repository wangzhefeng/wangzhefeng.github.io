---
title: Java EE
author: 王哲峰
date: '2020-10-01'
slug: javaee
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
img {
    pointer-events: none;
}
</style>

<details><summary>目录</summary><p>

- [Java 概念解释](#java-概念解释)
- [Java 基础语法](#java-基础语法)
  - [Java 程序基本结构](#java-程序基本结构)
    - [Java 程序的基本结构](#java-程序的基本结构)
    - [Java 程序解释](#java-程序解释)
    - [运行 Java 程序](#运行-java-程序)
  - [变量、数据类型、常量](#变量数据类型常量)
    - [变量](#变量)
    - [基本类型的变量——数据类型](#基本类型的变量数据类型)
      - [整数](#整数)
      - [浮点数](#浮点数)
      - [字符类型](#字符类型)
      - [布尔类型](#布尔类型)
    - [引用类型的变量——字符串](#引用类型的变量字符串)
    - [常量](#常量)
  - [运算符](#运算符)
    - [整数运算](#整数运算)
    - [浮点数运算](#浮点数运算)
    - [布尔运算](#布尔运算)
    - [强制类型转换](#强制类型转换)
    - [枚举类型](#枚举类型)
  - [字符和字符串](#字符和字符串)
    - [子串](#子串)
    - [拼接](#拼接)
    - [不可变字符串](#不可变字符串)
    - [检测字符串是否相等](#检测字符串是否相等)
    - [空串与 Null 串](#空串与-null-串)
    - [码点与代码单元](#码点与代码单元)
    - [String API](#string-api)
    - [阅读联机 API 文档](#阅读联机-api-文档)
    - [构建字符串](#构建字符串)
  - [输入输出](#输入输出)
    - [读取输入](#读取输入)
    - [格式化输出](#格式化输出)
    - [文件输入与输出](#文件输入与输出)
  - [流程控制](#流程控制)
    - [块作用域](#块作用域)
    - [条件语句](#条件语句)
    - [循环语句](#循环语句)
    - [switch 语句](#switch-语句)
    - [中断控制流程的 break 和 continue 关键字](#中断控制流程的-break-和-continue-关键字)
      - [不带标签的 break 语句](#不带标签的-break-语句)
      - [带标签的 break 语句](#带标签的-break-语句)
      - [不带标签的 continue 语句](#不带标签的-continue-语句)
      - [带标签的 continue 语句](#带标签的-continue-语句)
  - [大数值](#大数值)
</p></details><p></p>

# Java 概念解释

- JDK: Java Development Kit
    - 如果只有 Java 源码，要编译成 Java 字节码，就需要 JDK，因为 JDK 除了包含 JRE，还提供了编译器、调试器等开发工具
- JRE: Java Runtime Environment
    - JRE 就是运行 Java 字节码的虚拟机
- JKD 与 JRE 关系

```
┌─     ┌──────────────────────────────────┐
│      │     Compiler, debugger, etc.     │
│      └──────────────────────────────────┘
JDK ┌─ ┌──────────────────────────────────┐
│   │  │                                  │
│  JRE │      JVM + Runtime Library       │
│   │  │                                  │
└─  └─ └──────────────────────────────────┘
       ┌───────┐┌───────┐┌───────┐┌───────┐
       │Windows││ Linux ││ macOS ││others │
       └───────┘└───────┘└───────┘└───────┘
```

- JSR 规范: Java Specification Request
    - 为了保证Java语言的规范性，SUN公司搞了一个JSR规范，凡是想给Java平台加一个功能，比如说访问数据库的功能，
      大家要先创建一个JSR规范，定义好接口，这样，各个数据库厂商都按照规范写出Java驱动程序，开发者就不用担心自
      己写的数据库代码在MySQL上能跑，却不能跑在PostgreSQL上
- RI: Reference Implementation
- TCK: Technology Compatibility Kit
- JCP 组织: Java COmmunity Process

# Java 基础语法

## Java 程序基本结构

### Java 程序的基本结构

```java
// Hello.java 文件

// ====================
// 一个 Java 类 Hello
// ====================

/**
    * 可以用来自动创建文档的注释
    */

public class Hello {

    // -------
    // 一个 main 方法, 也是 Java 程序的固定入口方法
    // Java 总是从 main 方法开始执行
    // -------
    public static void main(String[] args) {
        
        // ~~~~~~~~~~~~~~
        // 向屏幕输出文本
        // ~~~~~~~~~~~~~~
        System.out.println("Hello, world!");
        
        /* 多行注释开始
        注释内容
        注释结束*/

    }

} // class 定义结束
```

### Java 程序解释

- Java 是面向对象的语言，一个程序的基本单位就是 `class`
    - 类名要求：字母(开头,大写)、数字、下划线
- Java 源码只能定义一个 `public` 类型的 `class`，并且 `class` 名称和文件名要完全一致
    - `public` 是访问修饰符(access modifier)，表示 `class` 是公开的
        - 访问修饰符用于控制程序的其他部分对这段代码的访问级别
        - 不写 `public`，也能正确编译，但是这个类将无法从命令行执行
- 在 `public class` 内部，可以定义若干方法(method)
    - 每个 Java 应用程序都必须有一个 `main` 方法
        - 运行已编译的程序时，Java 虚拟机将从指定类中的 `main` 方法开始执行，`main` 方法是 Java 程序的固定入口方法
        - `public static void main(String[] args) {}` 解释
            - `public`: 根据 Java 的语言规范，main 方法必须声明为 `public`
            - `static`: 静态方法，Java 入口程序规定的方法必须是静态方法
            - `void`: 返回值类型为 `void`，即没有任何返回值
            - `main`: Java 入口程序规定的名称必须是 `main`
            - `String[]`
            - `args`
    - 方法命名要求：字母(开头,小写)、数字、下划线
- Java 有 3 种注释
    - 单行注释 `//`
    - 多行注释 `/**/`
    - Java 文档字符串 `/** */`


### 运行 Java 程序

- Java 源码本质上是一个文本文件，需要先用 `javac` 命令把 `.java` 编译成字节码文件
  `.class`，然后用 `java` 命令执行 `.class` 字节码文件

```
┌──────────────────┐
│    Hello.java    │<─── source code
└──────────────────┘
        │ compile
        ▼
┌──────────────────┐
│   Hello.class    │<─── byte code
└──────────────────┘
        │ execute
        ▼
┌──────────────────┐
│    Run on JVM    │
└──────────────────┘
```

- 可执行文件 `javac` 是编译器，可执行文件 `java` 及时虚拟机
- 命令行运行 Java 程序

```bash
$ javac Hello.java
$ java Hello
```

> - 给虚拟机传递的参数 `Hello` 是定义的类名，虚拟机自动查找对应的 `.class` 文件并执行
> - Java 11新增的一个功能，它可以直接运行一个单文件源码
> 
> ```bash
> $ java Hello.java
> ```       
> 
> - 在实际项目中，单个不依赖第三方库的 Java 源码是非常罕见的，所以，绝大多数情况下，我们无法直接运行一个 Java 源码文件，原因是它需要依赖其他的库

## 变量、数据类型、常量

### 变量

在 Java 中，变量分为两种：**基本类型的变量** 和 **引用类型的变量**

- **变量类型声明**：
    - 在 Java 中，每个变量都有一个类型 (type)，在声明变量时，变量的类型位于变量名之
    - 在 Java 中，变量的声明尽可能地靠近变量第一次使用的地方，这是一种良好的程序编写风格
- **变量定义及初始化**：
    - 在 Java 中，变量必须先定义后使用
    -在定义变量的时候，可以给它一个初始值，不写初始值，就相当于给它指定了默认值，默认值总是 `0`
- **变量赋值**：变量既可以重新赋值, 也可以赋值给其他变量
- **var 关键字**：有些时候，类型的名字太长，写起来比较麻烦，如果想省略变量类型，可以使用 `var` 关键字，
  编译器会根据赋值语句自动推断出变量的类型
- **变量的作用范围**：定义变量时，要遵循作用域最小化原则，尽量将变量定义在尽可能小的作用域，并且，不要重复使用变量名

示例:

```java
// 定义并打印变量
// Main.java
public class Main {
    public static void main(String[] args) {
        int x = 100;
        System.out.println(x);
    }
}

// 重新赋值变量
// Main.java
public class Main {
    public static void main(String[] args) {
        int x = 100;
        System.out.println(x);

        // 变量 x 已经存在了，不能再重复定义，因此不能指定变量类型 int,必须使用语句:
        x = 200;
        System.out.println(x);
    }
}

// 变量之间的赋值
// Main.java
public class Main {
    public static void main(String[] args) {
        int n = 100;
        System.out.println("n = " + n);

        n = 200;
        System.out.println("n = " + n);

        int x = n;
        System.out.println("x = " + x);

        x = x + 100;
        System.out.println("x = " + x);
        System.out.println("n = " + n);
    }
}
```

### 基本类型的变量——数据类型

Java 是一种强类型的语言。这就意味着必须为每一个变量声明一种类型。基本数据类型是 CPU 可以直接进行运算的类型。

在 Java 中，一共有 8 种基本类型(primitive type) 其中有 4 种整型、2种浮点型、
1 种用于表示 Unicode 编码的字符单元的字符类型 char，和一种用于表示真值的 boolean 类型：

- 整型：byte，short，int，long
- 浮点型：float，double
- 字符类型：char
- 布尔类型：boolean

> Java 有一个能够表示任意精度的算术包，通常称为“大数值(big number)”, 虽然被称为大数值，
但它并不是一种新的 Java 类型，而是一个 Java 对象。


计算机内存的基本结构：

- 计算机内存的最小存储单元是 **字节(byte)**，一个字节(byte)就是一个 8 位二进制数，即 8 个 bit。
  它的二进制表示范围从 `00000000~11111111`，换算成十进制是 `0~255`，换算成十六进制是 `00~ff`
- 内存单元从 0 开始编号，称为内存地址。每个内存单元可以看作一间房间，内存地址就是门牌号

```
0   1   2   3   4   5   6  ...
┌───┬───┬───┬───┬───┬───┬───┐
│   │   │   │   │   │   │   │...
└───┴───┴───┴───┴───┴───┴───┘
```

- 8 bit 是 1 byte，一个字节是 1 byte，1024 字节是 1K，1024K 是 1M，1024M 是 1G，1024G 是 1T
- 不同的数据类型占用的字节数不一样, Java 基本数据类型占用的字节数(一个方框代表一个字节(byte))

```
     ┌───┐
byte │   │
     └───┘
     ┌───┬───┐
short │   │   │
     └───┴───┘
     ┌───┬───┬───┬───┐
 int │   │   │   │   │
     └───┴───┴───┴───┘
     ┌───┬───┬───┬───┬───┬───┬───┬───┐
long │   │   │   │   │   │   │   │   │
     └───┴───┴───┴───┴───┴───┴───┴───┘
     ┌───┬───┬───┬───┐
float │   │   │   │   │
     └───┴───┴───┴───┘
     ┌───┬───┬───┬───┬───┬───┬───┬───┐
double │   │   │   │   │   │   │   │   │
     └───┴───┴───┴───┴───┴───┴───┴───┘
     ┌───┬───┐
char │   │   │
     └───┴───┘
```

#### 整数

整型用于表示没有小数部分的数值，它允许是负数。对于整型类型，Java 只定义了带符号的整型，
因此，最高位的 bit 表示符号位(0 表示正数，1 表示负数)。在通常情况下：

- `int` 类型最常用
- `long` 类型用于表示 数值范围超过 `int` 类型所能表示的长度时
    - `long` 类型有一个后缀 `l` 或 `L`
- `byte` 和 `short` 类型主要用于特定的场合，例如：底层的文件处理或者需要控制占用的存储空间的大数组


> - 十六进制数值有一个前缀 `0x` 或者 `0X`
> - 八进制有一个前缀 `0`，建议最好不要使用八进制表示常数
> - 从 Java 7 开始，加上前缀 `0b` 或者 `0B` 就可以表示二进制数
> - 从 Java 7 开始，可以为数字字面量加下划线 `_`，这些下划线只是为了让人更易读

- 各种整型能表示的最大范围如下:
    - byte：-128 ~ 127
        - 1 字节
    - short: -32768 ~ 32767
        - 2 字节
    - int: -2147483648 ~ 2147483647
        - 4 字节
    - long: -9223372036854775808 ~ 9223372036854775807
        - 8 字节
- 示例

```java
public class Main {
    public static void main(String[] args) {
        int i1 = 2147483647;
        int i2 = -2147483648;
        int i3 = 2_000_000_000;        // 加下划线更容易识别
        int i4 = 0xff0000;             // 十六进制表示的 16711680
        int i5 = 0b1000000000;         // 二进制表示的 512
        long l = 9000000000000000000L; // long型的结尾需要加 L
    }
}
```

#### 浮点数

浮点类型的数就是小数，因为小数用科学计数法表示的时候，小数点是可以“浮动”的，
如 1234.5 可以表示成 12.345x102，也可以表示成 1.2345x103，所以称为浮点数。
Java 中有两种浮点类型：

- `float`
    - `float` 类型的数值需要加上 `f` 或者 `F` 后缀，没有后缀的浮点数值默认为 `double` 类型
- `double`
    - `double` 表示这种类型的数值精度是 `float` 类型的两倍
    - 绝大部分应用程序都采用 `double`，因为在很多情况下，`float` 类型的精度很难满足需求，
      只有很少的情况适合使用 `float` 类型，例如，需要单精度数据的库，或者需要存储大量数据
    - 可以在 `double` 类型的数值后面添加后缀 `d` 或者 `D`
- 所有的浮点数值计算都遵循 IEEE 754 规范。具体来说，下面是用于表示溢出和出错情况下的三个特殊的浮点数值：
    - 正无穷大 `Double.POSITIVE_INFINITY`
        - 一个正整数除以 0 的结果为 正无穷大
    - 负无穷大 `Double.NEGATIVE_INFINITY`
    - NaN (不是一个数字) `Double.NaN`
        - 计算 0/0 或者负数的平方根结果为 NaN


> - `Double.POSITIVE_INFINITY`、`Double.NEGATIVE_INFINITY`、`Double.NaN`， 
>   以及 `Float.POSITIVE_INFINITY`、`Float.NEGATIVE_INFINITY`、`Float.NaN`
>   分别表示上面三个特殊的值，但在实际应用中很少遇到。
>     - `if (x == Double.NaN)` 这样检测一个特定值是否等于 `Double.NaN` 是不可行的
>     - `if (Double.isNaN(x))` 才是可行的
> - 浮点数值不适用于无法接受舍入误差的金融计算中，如果在数值计算中不允许有任何舍入误差，就应该使用 `BigDecimal` 
>   类

- 浮点数可表示的范围非常大
    - float 类型可最大表示 :math:`3.4 \times 10^{38}`
        - 4 字节
    - double 类型可最大表示 :math:`1.79 \times 10^{308}`
        - 8 字节
- 示例

```java
public class Main {
    public static void main(String[] args) {
        float f1 = 3.14f;
        float f2 = 3.14e38f; // 科学计数法表示的3.14x10^38
        double d1 = 1.79e308;
        double d2 = -1.79e308;
        double d3 = 4.9e-324; // 科学计数法表示的4.9x10^-324
    }
}
```

#### 字符类型

字符类型 char 表示一个字符。

- char 类型的字面量值要用单引号括起来
- Java 的 char 类型除了可表示标准的 ASCII 外，还可以表示一个 Unicode 字符
- char 类型的值可以表示为十六进制值，其范围从 `\u0000` 到 `\Uffff`
    - `\u2122` 表示注册符号
    - `\u03C0` 表示希腊字母 :math:`\pi` 
- char 类型数值还包含一系列转义字符
    - `\u`：十六进制
    - `\b`：退格
    - `\t`：制表
    - `\n`：换行
    - `\r`：回车
    - `\"`：双引号
    - `\'`：单引号
    - `\\`：反斜杠

- 示例

```java
public class Main {
    public static void main(String[] args) {
        char a = 'A';
        char zh = '中';
        System.out.println(a);
        System.out.println(zh);
    }
}
```


> - char类型使用单引号 `'`，且仅有一个字符，要和双引号 `"` 的字符串类型区分开

#### 布尔类型

布尔类型 boolean 只有 `true` 和 `false` 两个值，用来判断逻辑条件。整型值和布尔值之间不能进行互相转换。

> - Java 语言对布尔类型的存储并没有做规定，因为理论上存储布尔类型只需要 1 bit，
  但是通常 JVM 内部会把 boolean 表示为 4 字节整数

- 示例

```java
public class Main {
    public static void main(String[] args) {
        boolean b1 = true;
        boolean b2 = false;
        boolean isGreater = 5 > 3;
        int age = 12;
        boolean isAdult =  age > 18;
    }
}
```

### 引用类型的变量——字符串

- 引用类型的变量类似于 C 语言的指针，它内部存储一个“地址”，指向某个对象在内存的位置
- 引用变量在栈内存，保存了指向对象的地址，指向的对象在堆内存

**String 字符串:**

```java
public class Main {
    public static void main(String[] args) {
        String s = "hello";
    }
}
```

### 常量

在 Java 中，利用关键字 `final` 指示常量。定义变量的时候，如果加上 `final` 修饰符，这个变量就变成了常量。
    
- 关键字 `final` 表示这个变量只能被赋值一次。一旦被赋值之后，就不能再更改了。
- 根据习惯，常量名通常全部大写
- 在 Java 中，经常希望某个常量可以在一个类中的多个方法中使用，通常将这些常量称为 **类常量**。
    - 用关键字 `static final` 设置一个类常量
    - 类常量的定义位于 `main` 方法的外部，因此在同一类的其他方法中亦可以使用这个常量
    - 如果一个常量被声明为 `public`，那么其他类的方法也可以使用这个常量


- 示例：

```java
public class Main {
    public static void main(String[], args) {

        // 类常量
        public static final double CM_PER_INCH = 2.54;

        // 常量
        final double PI = 3.14;    // PI是一个常量
        double r = 5.0;            // r 是一个变量
        double area = PI * r * r;  // area 是一个变量
        PI = 300; // compile error!
    }
}
```

## 运算符

### 整数运算

- Java 的整数运算遵循四则运算规则，可以使用任意嵌套的小括号
    - 整数的数值表示不但是精确的，而且整数运算永远是精确的，即使是除法也是精确的，因为两个整数相除只能得到结果的整数部分.
    - 整数的除法对于除数为 0 时运行时将报错，但编译不会报错
    - 整数由于存在范围限制，如果计算结果超出了范围，就会产生溢出，而溢出不会出错，却会得到一个奇怪的结果
    - `+=`、`-=`
    - Java 还提供了 `++` 运算和 `--` 运算，它们可以对一个整数进行加 1 和减 1 的操作
        - `++` 写在前面和后面计算结果是不同的，`++n` 表示先加 1 再引用 n，`n++` 表示先引用 n 再加 1。不建议把 `++` 运算混入到常规运算中，容易自己把自己搞懵了
    - 移位运算、位运算
    - 类型自动提升与强制转型：在运算过程中，如果参与运算的两个数类型不一致，那么计算结果为较大类型的整型
        - 也可以将结果强制转型，即将大范围的整数转型为小范围的整数。强制转型使用(类型)，例如，将int强制转型为short

```java
// 四则运算
public class Main {
    public static void main(String[], args) {
        // 加、减、乘
        int i = (100 + 200) * (99 - 88);
        int j = 7 * (5 + (i - 9));
        System.out.println(i);
        System.out.println(j);

        // 除法、求余
        int x =  12345 / 67;
        int y = 12345 % 67;

        // 溢出
        int a = 2147483640;
        int b = 15;
        int sum1 = a + b;
        System.out.println(sum1); // -2147483641

        // 解决溢出
        long m = 214748640;
        long n = 15;
        long sum2 = m + n;
        System.out.println(sum2);

        // 自增、自减
        int p = 3300;
        n ++;
        n --;
    }
}
```

### 浮点数运算

- 浮点数运算和整数运算相比，只能进行加减乘除这些数值计算，不能做位运算和移位运算
- 在计算机中，浮点数虽然表示的范围大，但是，浮点数有个非常重要的特点，就是浮点数常常无法精确表示
- 由于浮点数存在运算误差，所以比较两个浮点数是否相等常常会出现错误的结果。正确的比较方法是判断两个浮点数之差的绝对值是否小于一个很小的数
- 浮点数在内存的表示方法和整数比更加复杂。Java的浮点数完全遵循IEEE-754标准，这也是绝大多数计算机平台都支持的浮点数标准表示方法
- **类型提升**：如果参与运算的两个数其中一个是整型，那么整型可以自动提升到浮点型
- **溢出**： 整数运算在除数为0时会报错，而浮点数运算在除数为0时，不会报错，但会返回几个特殊值：
    - `NaN` 表示 Not a Number
    - `Infinity` 表示无穷大
    - `-Infinity` 表示负无穷大
- **强制转型**：可以将浮点数强制转型为整数。在转型时，浮点数的小数部分会被丢掉。如果转型后超过了整型能表示的最大范围，将返回整型的最大值

### 布尔运算

- 对于布尔类型 boolean，永远只有 `true` 和 `false` 两个值
- 布尔运算是一种关系运算，包括以下几类：
    - 比较运算符：`>`，`>=`，`<`，`<=`，`==`，`!=`
    - 与运算 `&&`
    - 或运算 `||`
    - 非运算 `!`
- 关系运算符的优先级从高到低依次是：
    - `!`
    - `>`, `>=`，`<`，`<=`
    - `==`，`!=`
    - `&&`
    - `||`
- 布尔运算的一个重要特点是短路运算。如果一个布尔运算的表达式能提前确定结果，则后续的计算不再执行，直接返回结果。
- 三元运算符 `b ? x : y`
    - 先计算 `b`
    - `x` 和 `y` 的类型必须相同，因为返回值不是 `boolean`, 而是 `x` 和 `y` 之一


```java
public class Main {
    public static void main(String[] args) {
        boolean  isGreater = 5 > 3;
        int age = 12;
        boolean isZero = age = 0;
        boolean isNonZero = !isZero;
        boolean isTeenager = age > 6 && age < 18;

        // 三元运算
        int n = -100;
        int x = n >= 0 ? n : -n;
        System.out.println(x);
    }
}
```


### 强制类型转换

### 枚举类型

- 示例

```java
// 定义枚举类型 Size
enum Size {SMALL, MEDIUM, LARGE, EXTRA_LARGE};

// 声明 Size 类型的变量
Size s = Size.MEDIUM;
```


## 字符和字符串

在Java中，字符和字符串是两个不同的类型，从概念上讲，Java 字符串就是 Unicode 字符序列。

Java 没有内置的字符串类型，而在标准 Java 类库中提供了一个预定义类，很自然地叫做 String。
每个用双引号括起来的字符串都是 String 类的一个实例。

**字符类型**：

- 字符类型 `char` 是基本数据类型，它是 `character` 的缩写。
  一个 `char` 保存一个 Unicode 字符
- 因为 Java 在内存中总是使用 Unicode 表示字符，所以，一个英文字符和一个中文字符都用一个 `char` 类型表示，
  它们都占用两个字节。要显示一个字符的 Unicode 编码，只需将 `char` 类型直接赋值给 `int` 类型即可
- 可以直接用转义字符 `\u` + Unicode 编码来表示一个字符

```java
public class Main {
    public static void main(String[] args) {
        // char
        char c1 = 'A';
        char c2 = '中';

        // Unicode
        int n1 = 'A';  // 字母“A”的Unicodde编码是65
        int n2 = '中'; // 汉字“中”的Unicode编码是20013

        // \u
        char c3 = '\u0041';
        char c4 = '\u4e2d';
    }
}
```

**字符串类型**:

- 和 `char` 类型不同，字符串类型 `String` 是引用类型，用双引号 `"..."` 表示字符串。一个字符串可以存储 0 个到任意个字符
- 因为字符串使用双引号 `"..."` 表示开始和结束，那如果字符串本身恰好包含一个 `"` 字符怎么表示？例如，`"abc"xyz"`，编译器就无法判断中间的引号究竟是字符串的一部分还是表示字符串结束。这个时候，我们需要借助转义字符 `\`
    - 常见的转义字符包括：
        - \" 表示字符"
        - \' 表示字符'
        - \\ 表示字符\
        - \n 表示换行符
        - \r 表示回车符
        - \t 表示Tab
        - \u#### 表示一个Unicode编码的字符
- **字符串连接**: Java的编译器对字符串做了特殊照顾，可以使用 `+` 连接任意字符串和其他数据类型，这样极大地方便了字符串的处理
    - 如果用 `+` 连接字符串和其他数据类型，会将其他数据类型先自动转型为字符串，再连接
- **多行字符串**:
    - 从 Java 13 开始，字符串可以用 `"""..."""` 表示多行字符串(Text Blocks)了
- **不可变性**:
    - Java 的字符串除了是一个引用类型外，还有个重要特点，就是字符串不可变

```java
public class Main {
    public static void main(String[] args) {
        String s = "";         // 空字符串，包含0个字符
        String s1 = "A";       // 包含一个字符
        String s2 = "ABC";     // 包含3个字符
        String s3 = "中文 ABC"; // 包含6个字符，其中有一个空格
        String s4 = "abc\"xyz";
        String s5 = "abc\\xyz";

        Strings6 = "first line \n"
                    + "second line \n"
                    + "end";

        // 7行，末尾有 \n
        String s7 = """
                    SELECT 
                        * 
                    FROM
                        users
                    WHERE id > 100
                    ORDER BY name DESC
                    """;
        System.out.println(s7);

        // 6行
        String s8 = """
                    SELECT 
                        * 
                    FROM
                        users
                    WHERE id > 100
                    ORDER BY name DESC""";
        System.out.println(s8);

        // 字符串不可变
        String s = "hello";
        System.out.println(s); // 显示 hello
        s = "world";
        System.out.println(s); // 显示 world

        // 空值 null
        String s1 = null; // s1是null
        String s2;        // 没有赋初值值，s2也是null
        String s3 = s1;   // s3也是null
        String s4 = "";   // s4指向空字符串，不是null
    }
}
```

### 子串

String 类的 `substring` 方法可以从一个较大的字符串取出一个子串。

- 示例

```java
String greeting = "Hello";
String s = greeting.substring(0, 3);
```

### 拼接

Java 语言允许使用 `+` 号连接(拼接)两个字符串.

- 当将一个字符串与一个非字符串的值进行拼接时，后者被转换成字符串。这种特性通常用在输出语句中
- 如果需要把多个字符串放在一起，用一个定界符分隔，可以使用静态 `join` 方法

示例 1：

```java
String expletive = "Expletive";
String PG13 = "deleted";
String message = expletive + PG13;
```

示例 2：

```java
ing age = 13;
String rating = "PG" + age;
```

示例 3：

```java
System.out.println("The answer is " + answer);
```

示例 4：

```java
String all = String.join(" /", "S", "M", "L", "XL");
```

### 不可变字符串

String 类没有提供用于修改字符串的方法。由于不能修改 Java 字符串中的字符，
所以在 Java 文档中奖 String 类对象称为不可变字符串。

- 不可变字符串有一个有点：编译器可以让字符串共享

### 检测字符串是否相等

可以使用 `equals` 方法检测两个字符串是否相等. 对于表达式：

- `s.equals(t)`
    - 如果字符串 `s` 与字符串 `t` 相等，则返回 `true`
    - 如果字符串 `s` 与字符串 `t` 不相等，则返回 `false`
    - `s` 与 `t` 可以是字符串变量，也可以是字符串字面量

示例 1：

```java
String s = "Hello";
String t = "World";

s.equals(t)
```

示例 2：

```java
String greeting = "Hello";
"Hello".equals(greeting);
```

示例 3：

```java
"Hello".equalsIgnoreCase("hello");
```

> 一定不要使用 `==` 运算符检测两个字符串是否相等！这个运算符只能确定两个字符串是放置在同一个位置上。
  当然，如果字符串放置在同一个位置上，它们必然相等。但是，完全有可能将内容相同的多个字符串的拷贝放置在不同的位置上。

### 空串与 Null 串

- 空串 `""` 是长度为 0 的字符串. 空串是一个 Java 对象，有自己的串长度 (0) 和内容 (空)。
  要检查一个字符串是否为空，要使用以下条件：：
    - `if (str.length() == 0)`
    - `if (str.equals(""))`
- String 变量还可以存放一个特殊的值，名为 `null`，表示目前没有任何对象与改变量关联。
  要检查一个字符串是否为 `null`，要使用以下条件：
    - `if (str == null)`
- 要检查一个字符串既不是 `null`，也不为空串 `""`，要使用以下条件：
    - `if (str != null && str.length() != 0)`

### 码点与代码单元

Java 字符串由 char 值序列组成。char 数据类型是一个采用 UTF-16 编码表示 Unicode 码点的代码单元。
大多数的常用 Unicode 字符使用一个代码单元就可以表示，而辅助字符需要一对代码单元表示。

- `length()` 方法将返回采用 UTF-16 编码表示的给定字符串所表示所需要的代码单元数量
- `codePointCount()` 方法返回的是字符串的实际的长度，即码点数量
- `s.charAt(n)` 将返回位置 n 的代码单元，n 介于 0~s.length()-1 之间
- 得到第 i 个码点:

```java
int index = greeting.offsetByCodePoint(0, i)
codePointAt(index)
```

示例：

```java
String greeting = "Hello";
int n = greeting.length(); // is 5

int cpCount = greeting.codePoints(0, greeting.length());
char first = greeting.charAt(0); // first is "H"
char last = greeting.charAt(4);  // last is "o"

int index = greeting.offsetByCodePoint(0, i);
int cp = greeting.codePointsAt(index);
```

> 为了避免出现问题，不要使用 `char` 类型，这太底层了。

### String API

- java.lang.string 1.0
    - char charAt(int index)
    - int codePointAt(int index) 5.0
    - int offsetByCodePoints(int startIndex, int cpCount) 5.0
    - int compareTo(String other)
    - IntStream codePoints() 8
    - new String(int[] codePoints, int offset, int count) 5.0
    - boolean equals(Object other)
    - boolean equalsIgnoreCase(String other)
    - boolean startsWith(String prefix)
    - boolean endsWith(String suffix)
    - int indexOf(String str)
    - int indexOf(String str, int fromIndex)
    - int indexOf(int cp)
    - int indexOf(int cp, int fromIndex)
    - int lastIndexOf(String str)
    - int lastIndexOf(String str, int fromIndex)
    - int lastIndexOf(int cp)
    - int lastIndexOf(int cp, int fromIndex)
    - int length()
    - int codePointCount(int startIndex, int endIndex) 5.0
    - String replace(CharSequence oldString, CharSequence newString)
    - 返回一个新字符串。这个字符串包含原字符串中从 beginIndex 到串尾货 endIndex-1 的所有代码单元
        - String substring(int beginIndex)
        - String substring(int beginIndex, int endIndex)
    - 返回一个新的字符串，字符串大小写切换
        - String toLowerCase()
        - String toUpperCase()
    - String trim()
        - 返回一个新字符串。这个字符串将删除了原始字符串头部和尾部的空格
    - String join(CharSequence delimiter, CharSequence... elements) 8

### 阅读联机 API 文档

- `JDK API Doc <https://docs.oracle.com/javase/8/docs/api/>`_ 

### 构建字符串

如果需要用许多小段的字符串构建一个字符串，可以按照下列步骤进行：

```java
// 1.构建一个空的字符串构建器
StringBuilder builder = new StringBuilder();

// 2.添加字符、字符串内容
builder.append(ch);
builder.append(str);

// 3.在需要构建字符串时就调用 toString 方法，将得到一个 String 对象，其中包括了构建器中的字符序列
String completedString = builder.toString();
```

- `java.lang.StringBuilder 5.0` API 重要方法
    - `StringBuilder()`
        - 构造一个空的字符串构建器
    - `int length()`
        - 返回构建器或缓冲器中的代码单元数量
    - `StringBuilder append(String str)`
        - 追加一个代码单元并返回 this
    - `StringBuilder append(char c)`
        - 追加一个代码单元并返回 this
    - `StringBuilder appendCodePoint(int cp)`
        - 追加一个代码点，并将其转换为一个或两个代码单元并返回 `this`
    - `void setCharAt(int i, char c)`
    - `StringBuilder insert(int offset, String str)`
    - `StringBuilder insert(int offset, Char c)`
    - `StringBuilder deleted(int startIndex, int endIndex)`
    - `String toString()`
        - 返回一个与构建起或缓冲器内容相同的字符串

## 输入输出

- Scanner 类、Console 类常用方法：
    - java.util.Scanner 5.0
        - `String nextLine()`
            - 读取输入的下一行内容，输入行中可能包含空格
        - `String next()`
            - 读取一个单词，以空格作为分隔符
        - `int nextInt()`
            - 读取并转换下一个整数
        - `double nextDouble()`
            - 读取并转换下一个浮点数的字符序列
        - 检测输入中是否还有其他单词、整数、浮点数的下一个字符序列
            - `boolean hasNext()`
            - `boolean hasNextInt()`
            - `boolean hasNextDouble()`
    - `Console`
        - java.lang.System 1.0
            - `static Console console`
        - java.io.Console 6
            - static char[] readPassword(String prompt, Object ... args)
            - static String readLine(String prompt, Object ... args)

### 读取输入

打印输出到 “标准输出流”(即控制台窗口)是一件非常容易的事情，只要调用 `System.out.println()` 即可。
然而，读取 “标准输入流” `System.in` 就没有那么简单了。

要想通过控制台进行输入，需要三个步骤：

- 首先，构造一个 `Scanner` 对象
- 其次，与 “标准输入流” `System.in` 关联
- 现在，可以使用 `Scanners` 类的各种方法实现输入操作了

示例

```java
import java.util.Scanner;

// 构造一个 Scanner 对象，并与标准输入流 System.in 关联
Scanner in = new Scanner(System.in);

// 输入一行
System.out.println("What is your name?");
String name = in.nextLine();
```


> 因为输入是可见的，所以 Scanner 类不适用于从控制台读取密码。
> 
> Java SE6 特别引入了 `Console` 类实现这个目的。要想读取一个密码，可以采用下列代码：
> 
> ```java
> import java.util.Console;
> 
> Console cons = System.console();
> String username = cons.readLine("User name: ");
> char[] password = cons.readPassword("Password: ");
> ```
> 
> 为了安全起见，返回的密码存放在一堆字符数组中，而不是字符串中。在对密码进行处理之后，应该马上用一个填充值覆盖数组元素。
> 
> 采用 `Console` 对象处理输入不如采用 `Scanner` 方便。每次只能读取一行输入，而没有能够读取一个单词或一个数值的方法。


### 格式化输出

可以使用 `System.out.print(x)` 将数值 `x` 输出到控制台上。
这条命名将以 `x` 对应的数据类型所允许的最大非0数字位数打印输出 `x`。
如果希望显示美元、美分、等符号，则可能出现问题。

Java SE 5.0 沿用了 C 语言库函数中的 `System.out.printf()` 方法。

### 文件输入与输出

要想对文件进行读取，需要两个步骤：

- 首先，用 `File` 对象构造一个 `Scanner` 对象
- 现在，可以使用 `Scanners` 类的各种方法对文件进行读取

示例

```java
Scanner in = new Scanner(Paths.get("C:\\mydirectory\\myfile.txt"), "UTF-8");
```

要想写入文件，需要两个步骤：

- 首先，需要构造一个 `PrintWriter` 对象，在构造器中，只需要提供文件名。
    - 如果文件不存在，创建该文件
- 现在，可以像输出到 `System.out` 一样使用 `print`、`println` 以及 `printf` 命令

示例

```java
PrintWriter out = new PrintWriter("C:\\mydirectory\\myfile.txt", "UTF-8");
```

> 如果文件名中包含反斜杠符号，就要记住在每个反斜杠之前再加一个额外的反斜杠.

## 流程控制

在Java程序中，JVM 默认总是顺序执行以分号 `;` 结束的语句。但是，在实际的代码中，程序经常需要做条件判断、循环，因此，需要有多种流程控制语句，来实现程序的跳转和循环等功能

- 条件判断语句
- 循环语句
- `switch` 语句
- `break` 关键字
- `continue` 关键字

### 块作用域

- 块(block)，即复合语句，是指由一对大括号括起来的若干条简单的 Java 语句
    - 块确定了变量的作用域, 一个块可以嵌套在另一个块中
    - 不能在嵌套的两个块中声明同名的变量

### 条件语句

- `if (condition) statement;`
- `if (condition) statement1 else statement2;`
- `if (condition) statement1 else if (condition) statement else if ...;`

示例1：

```java
if (yourSales >= target) {
    performance = "Satisfactory";
    bonus = 100;
}
```

示例2：

```java
if (yourSales >= target) {
    performance = "Satisfactory";
    bonus = 100 + 0.01 * (yourSales - target);
}
else {
    performance = "UnStatifactory";
    bonus = 0;
}
```

示例3：

```java
if (yourSales >= 2 * target) {
    performance = "Excellent";
    bonus = 1000;
}
else if (yourSales >= 1.5 * target) {
    performance = "Fine";
    bonus = 500;
}
else if (yourSales >= "target") {
    performance = "Satisfactory";
    bonus = 100;
}
else {
    System.out.println("You'r fired");
}
```

示例4：

```java
if (x <= 0) {
    if (x == 0) {
        sign = 0;
    } else {
        sign = -1;
    }
}
```

### 循环语句

- `while (condition) statement;`
- `for (int i; i = 0; i++);`;
- `do statement while (condition);`

示例1：

```java
// Retirement.java
while (balance < goal) {
    balance += payment;
    double interest = balance * interestRate / 100;
    balance += interest;
    years++;
}
System.out.println(years + "years.");
```

示例2：

```java
// Retirement2.java
do {
    // add this year's payment and interest
    balance += payment;
    double interest = balance * interestRate / 100;
    year++;

    // print current balance
    System.out.printf("After year %d, your balance is %,.2f%n", year, balance);

    // ask if ready to retire and get input
    System.out.print("Ready to retire? (Y/N)");
    input = in.next();
}
while (input.equals("N"));
```

示例3：

```java
for (int i; i <= 10; i++) {
    System.out.println(i);
}
```

```java
int i;
for (int i; i <= 10; i++) {
    System.out.println(i);
}
```

```java
// LotteryOdds.java
int lotteryOdds = 1;
for (int i = 1; i <= k; i++) {
    lotteryOdds = lotteryOdds * (n - i + 1) / i;
}
```

### switch 语句

在处理多个选项时，使用 `if/esle` 结构显得有些笨拙，Java 有一个与 C/C++ 完全一样的 switch 语句。

`switch` 语句将从与选项值相匹配的 `case` 标签处开始执行直到遇到 `break` 语句，
或者执行到 `switch` 语句的结束处为止。如果没有相匹配的 `case` 标签，
而有 `default` 子句，就执行这个子句。

`case` 标签可以是：

- 类型为 `char`, `byte`, `short`, `int` 的常量表达式
- 枚举常量
- 从 Java SE7 开始，`case` 标签还可以是字符串字面量

当在 `switch` 语句中使用枚举常量时，不必在每个标签中指明枚举名，可以由 `switch` 的表达式值确定.

示例 1：

```java
import java.util.*;

public class switchClass {
    public static void main(String[] args) {

        Scanner in = new Scanner(System.in);
        System.out.print("Select an option (1, 2, 3, 4) ");
        int choice = in.nextInt();

        switch (choice) {
            case 1:
                // statement1;
                break;
            case 2:
                // statement2;
                break;
            case 3:
                // statement3;
                break;
            case 4:
                // statement4;
                break;
            default:
                // bad input;
                break;
        }
    }
}
```

示例 2：

```java
String input = ...;
switch (input.toLowerCase()) {
    case "yes":
        ...
        break;
    ...
}
```

示例 3：

```java
Size sz = ...;
switch (sz) {
    case SMALL:
        ...
        break;
    ...
}
```

### 中断控制流程的 break 和 continue 关键字

- `break` 语句：
    - 不带标签的 `break` 语句；
    - 带标签的 `break` 语句；
        - 用于跳出多重潜逃的循环语句；
- `continue` 语句


#### 不带标签的 break 语句

```java
while (years <= 100) {
    balance += payment;
    double interest = balance * interestRate / 100;
    balance += interest;
    if (balance >= goal) 
        break;
    years++;
}
```

#### 带标签的 break 语句

```java
Scanner in new Scanner(System.in);
int n;
read_data:
while (...) {
    ...
    for () {
        System.out.print("Enter a number >= 0: ");
        n = in.nextInt();
        if (n < 0) 
            break read_data;
        ...
    }
}

if (n < 0) {
    // ...
}
else {
    // ...
}
```

#### 不带标签的 continue 语句

```java
Scanner in = new Scanner(System.in);
while (sum < goal) {
    System.out.print("Enter a number: ");
    n = in.nextInt();
    if (n < 0) 
        continue;
    sum += n;
}
```

```java
for (count = 1; count <= 100; count++) {
    System.out.print("Enter a number, -1 to quit: ");
    n = in.nextInt();
    if (n < 0) 
        continue;
    sum += n;
}
```

#### 带标签的 continue 语句

- 略


## 大数值

如果基本的整数和浮点数 **精度** 不能够满足需求，那么可以使用 `java.math` 包中的两个很有用的类. 
这两个类可以处理包含任意长度数字序列的数值：

- `BigInteger`
    - 实现了任意精度的整数运算
- `BigDecimal`
    - 实现了任意精度的浮点数运算
- 使用静态的 `valueOf` 方法可以将普通的数值转换为大数值
- 不能使用诸如 `+` 和 `*` 等算术运算符处理大数值，而需要使用大数值类中的 `add()` 和 `multiply()`

示例：

```java
BigInteger a = BigInteger.valueOf(100);
BigInteger b = BigInteger.valueOf(100);

// c = a + b
BigInteger c = a.add(b);
// d = c * (b + 2)
BigInteger d = c.multiply(b.add(BigInteger.valueOf(2)));
```

- java.math.BigInteger 1.1
    - 返回这个大整数和另一个大整数 other 的和、差、积、商、余数：
        - `BigInteger add(BigInteger other)`
        - `BigInteger subtract(BigInteger other)`
        - `BigInteger multiply(BigInteger other)`
        - `BigInteger divide(BigInteger other)`
        - `BigInteger mod(BigInteger other)`
    - `int compareTo(BigInteger other)`
        - 如果这个大整数与另一个大整数 other 相等，返回 0
        - 如果这个大整数小于另一个大整数 other，返回负数
        - 如果这个大整数大于另一个大整数 other，返回整数
    - `static BigInteger valueOf(long x)`
        - 返回值等于 :math:`x` 的大整数
- java.math.BigDecimal 1.1
    - 返回这个大实数和另一个大实数 other 的和、差、积、商：
        - `BigDecimal add(BigDecimal other)`
        - `BigDecimal subtract(BigDecimal other)`
        - `BigDecimal multiply(BigDecimal other)`
        - `BigDecimal divide(BigDecimal other)`
    - `int compareTo(BigDecimal other)`
        - 如果这个大实数与另一个大实数 other 相等，返回 0
        - 如果这个大实数小于另一个大实数 other，返回负数
        - 如果这个大实数大于另一个大实数 other，返回整数
    - `static BigDecimal valueOf(long x)`
        - 返回值等于 :math:`x` 的大实数
    - `static BigDecimal valueOf(long x, int scale)`
        - 返回值为 :math:`x / 10^{scale}` 的一个大实数

