---
title: Scala OOP
author: 王哲峰
date: '2022-09-05'
slug: scala-oop
categories:
  - scala
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

- [Scala OOP 概述](#scala-oop-概述)
- [Scala面向对象编程](#scala面向对象编程)
  - [类、字段、方法](#类字段方法)
    - [创建类](#创建类)
    - [创建对象](#创建对象)
    - [创建类、定义私有字段、方法](#创建类定义私有字段方法)
  - [单例对象](#单例对象)
    - [单例对象举例](#单例对象举例)
    - [单例对象创建Scala应用程序入口](#单例对象创建scala应用程序入口)
  - [基础类型、操作](#基础类型操作)
    - [基础类型](#基础类型)
    - [字面量](#字面量)
  - [函数式对象](#函数式对象)
    - [背景、类设计](#背景类设计)
    - [构建类](#构建类)
      - [重新实现toString方法](#重新实现tostring方法)
      - [检查前置条件](#检查前置条件)
      - [添加字段](#添加字段)
      - [自引用](#自引用)
      - [辅助构造方法](#辅助构造方法)
      - [私有字段和方法](#私有字段和方法)
      - [定义操作符](#定义操作符)
  - [内建控制结构](#内建控制结构)
    - [if 表达式](#if-表达式)
    - [while 循环](#while-循环)
    - [for 表达式](#for-表达式)
      - [遍历集合](#遍历集合)
      - [过滤](#过滤)
      - [嵌套迭代](#嵌套迭代)
      - [中途(mid-stream)变量绑定](#中途mid-stream变量绑定)
      - [输出一个新的集合](#输出一个新的集合)
    - [try表达式异常处理](#try表达式异常处理)
    - [Scala 中没有 break 和 continue](#scala-中没有-break-和-continue)
  - [函数和闭包](#函数和闭包)
    - [方法](#方法)
    - [局部函数](#局部函数)
    - [一等函数](#一等函数)
      - [函数字面量](#函数字面量)
      - [函数字面量简写](#函数字面量简写)
    - [闭包](#闭包)
    - [特殊的函数调用形式(传参)](#特殊的函数调用形式传参)
    - [尾递归](#尾递归)
  - [控制抽象](#控制抽象)
  - [组合继承](#组合继承)
    - [2.8.1](#281)
    - [Scala的继承关系](#scala的继承关系)
  - [特质](#特质)
    - [特质的定义](#特质的定义)
    - [把特质混入类中](#把特质混入类中)
    - [特质可以包含具体实现](#特质可以包含具体实现)
    - [把多个特质混入类中](#把多个特质混入类中)
- [包(package)和包引入(import)](#包package和包引入import)
  - [将代码放进包里(模块化)](#将代码放进包里模块化)
  - [对相关代码的精简访问](#对相关代码的精简访问)
  - [包引入](#包引入)
  - [隐式引入](#隐式引入)
    - [受保护成员(protected)](#受保护成员protected)
    - [公共成员](#公共成员)
    - [保护的范围](#保护的范围)
    - [可见性和伴生对象](#可见性和伴生对象)
  - [包对象(package object)](#包对象package-object)
- [断言和测试](#断言和测试)
  - [断言](#断言)
  - [测试](#测试)
- [样例类和匹配模式](#样例类和匹配模式)
  - [样例类](#样例类)
  - [模式匹配](#模式匹配)
    - [模式匹配形式](#模式匹配形式)
    - [模式种类](#模式种类)
      - [通配模式](#通配模式)
      - [常量模式](#常量模式)
      - [变量模式](#变量模式)
      - [构造方法模式](#构造方法模式)
      - [序列模式](#序列模式)
      - [带类型的模式](#带类型的模式)
      - [变量绑定](#变量绑定)
</p></details><p></p>

# Scala OOP 概述







# Scala面向对象编程


## 类、字段、方法

- 类、对象
- 类是对象的蓝本(blueprint)；一旦定义好了一个类，就可以用 `new` 关键字从这个类蓝本创建对象；
- 字段、方法
- 在类定义中，可以填入 `字段(field)` 和 `方法(method)` ，这些被统称为 `成员(member)` ；
   - 通过 `val` 或 `var` 定义的 `字段` 是指向对象的变量；字段保留了对象的状态，或者说是数据；
      - 字段又叫做 `实例变量(instance variable)` ，因为每个实例都有自己的变量，这些实例变量合在一起，构成了对象在内存中的映像；
      - 追求健壮性的一个重要手段是确保对象的状态(实例变量的值)在其整个声明周期都是有效的；
      - 首先，通过将字段标记为 `私有(private)` 来防止外部直接访问字段，因为私有字段只能被定义在同一个类中的方法访问，所有对状态的更新的操作的代码，都在类的内部；
      - 在Scala中，除非显式声明 `private` ，否则变量都是公共访问的(public)；
      - 通过 `def` 定义的 `方法` 则包含了可执行的代码；方法用字段定义的数据来对对象执行计算；
         - 传递给方法的任何参数都能在方法内部使用。Scala方法参数的一个重要特征是他们都是val。因此，如果试图在Scala的方法中对参数重新赋值，编译会报错；
            - 在Scala方法定义中，在没有任何显式的return语句时，方法返回的是该方法计算出的最后一个值；仅仅因为其副作用而被执行的方法被称作 `过程(procedure)` ；
- 实例
- 当 `实例化` 一个类，运行时会指派一些内存来保存对象的状态图（即它的变量的内容）；

### 创建类

```scala

   class ChecksumAccumulator {
      // 类定义
   }
```


### 创建对象

```scala

   new ChecksumAccumulator
````


### 创建类、定义字段

```scala

   class ChecksumAccumulator {
      // 类定义
      var sum = 0
   }
```

```scala

   val acc = ChecksumAccumulator
   val csa = ChecksumAccumulator
```

- acc 和 csa 是同一个类的两个不同的 ChecksumAccumulator 对象，它们都有一个实例变量 `sum` ，并且指向相同的内存对象 `0` ；
- 由于 `sum` 是定义在类 ChecksumAccumulator 中的可变 `var` 字段，
  可以对其重新进行赋值 `acc.sum = 3` ，此时，acc 和 csa 的实例变量指向了不同的内存对象, 
  acc.sum 指向了 `3` ，而 csa.sum 指向了 `0` ；
   - acc和csa本身是val对象，不能将他们重新赋值指向别的 `对象,object` ，但是可以将他们的实例变量指向不同的对象；



### 创建类、定义私有字段、方法

```scala

   // ChecksumAccumulator.scala

   class ChecksumAccumulator {
      private var sum = 0
      
      def add(b: Byte): Unit = {
         sum += b
      }

      def checksum(): Int = {
         ~(sum & 0xFF) + 1
      }
   }
```

因为类 `ChecksumAccumulator` 的字段 `sum` 现在是 `private` ，
所以对 `ChecksumAccumulator` 的对象 `acc` 在类的外部重新赋值是不能编译的: 

```scala

   val acc = new ChecksumAccumulator

   // 下面的定义不能编译 
   acc.sum = 5
```


## 单例对象

- Scala比Java更面向对象的一点，是Scala的class不允许有 `static` 成员；对于这种使用场景，Scala提供了 `单例对象(singleton object)` ；
- `单例对象(singleton object)` 的定义跟类定义很像，只不过 `class` 关键字换成了 `object` 关键字；
- 当单例对象跟某个类共用同一个名字时，它被称为这个类的 `伴生对象(companion object)` ；同名的类又叫作这个单例对象的 `伴生类(companion class)` ；必须在同一个源码文件中定义类和类的伴生对象；类和它的伴生对象可以互相访问对方的私有成员；
- 没有同名的伴生类的单例对象称为 `孤立对象(standalone object)` ；孤立对象有很多用途，包括将工具方法归集在一起，或定义Scala应用程序的入口等；
- 定义单例对象并不会定义类型；不过单例对象可以扩展自某个超类，还可以混入特质，可以通过这些类型来调用他的方法，用这些类型的变量来引用它，还可以将它传入那些预期这些类型的入参的方法中；
- 类和单例对象的一个区别是单例对象不接收参数，而类可以；
- 每个单例对象都是通过一个静态变量引用合成类(synthetic class)的实例来实现的，因此，单例对象从初始化的语义上跟Java的静态成员是一致的，尤其体现在单例对象有代码首次访问时才被初始化；
    - 合成类的名称为对象加上美元符号: `objectName$`

### 单例对象举例

```scala

   // ChecksumAccumulator.scala

   import scala.collection.mutable

   class ChecksumAccumulator {
      private var sum = 0
      
      def add(b: Byte): Unit = {
         sum += b
      }

      def checksum(): Int = {
         ~(sum & 0xFF) + 1
      }
   }

   object ChecksumAccumulator {
      
      // field cache
      private val cache = mutable.Map.empty[String, Int]

      // method calculate
      def calculate(s: String): Int = {
         if (cache.contains(s)) {
            cache(s)
         }
         else {
            val acc = new ChecksumAccumulator
            for (c <- s) {
               acc.add(c.toByte)
            }
            val cs = acc.checksum()
            cache += (s -> cs)
            cs
         }
      }
   }
```

调用单例对象: 

```scala

   ChecksumAccumulator.calculate("Every value is an object.")
```


### 单例对象创建Scala应用程序入口

- 要运行一个Scala程序，必须提供一个独立对象的名称，这个独立对象需要包含一个 `main` 方法，该方法接收一个 `Array[String]` 作为参数，结果类型为 `Unit` ；
- 任何带有满足正确签名的 `main` 方法的独立对象都能被用作Scala应用程序的入口；
- Scala在每一个Scala源码文件都隐式地引入了 `java.lang` 和 `scala` 包的成员，以及名为 `Predef` 的单例对象的所有成员；
- `java.lang` 中包含的常用方法: 
    - 
- `scala` 中包含的常用方法: 
    - 
- `Predef` 中包含了很多有用的方法: 
    - `Predef.println()`
    - `Predef.assert`
- Scala和Java的区别之一是，Scala总可以任意命名 `.scala` 文件，不论放什么类或代码到这个文件中；
- 通常对于非脚本的场景，把类放入以类名命名的文件是推荐的做法；
- 非脚本: 以定义结尾；
- 脚本: 必须以一个可以计算出结果的表达式结尾；

**Scala应用程序: **

```scala

   // Summer.scala
   import ChecksumAccumulator.calculate

   object Summer {
      def main(args: Array[String]) {
         for (arg <- args) {
            println(arg + ":" + calculate(arg))
         }
      }
   }
```

**调用Scala应用程序: **

- 需要用Scala编译器实际编译程序文件，然后运行编译出来的类；
- `scalac` or `fsc` 编译程序文件；
- `scala` 运行编译出的类；
- Scala基础编译器 `scalac`:
   - 编译源文件，会有延迟，因为每一次编译器启动，都会花时间扫描jar文件的内容以及执行其他一些初始化的工作，然后才开始关注提交给它的新的源码文件；

```bash
$ scalac ChecksumAccumulator.scala Summer.scala
```

- Scala编译器的守护进程 `fsc`:
   - Scala 的分发包包含了一个名为 `fsc` 的 Scala 编译器的守护进程(daemon)，
     第一次运行 fsc 时，它会创建一个本地的服务器守护进程，绑定到计算机的某个端口上；
     然后它会通过这个端口将需要编译的文件发送给这个守护进程。
     下次运行fsc的时候，这个守护进程已经在运行了，
     所以fsc会简单地将文件清单发给这个守护进程，
     然后守护进程就会立即编译这些文件，使用fsc，
     只有在首次运行时才需要等待java运行时启动。

```bash
$ fsc ChecksumAccumulator.scala Summer.scala
```

如果想停止fsc这个守护进程，可以执行

```bash
$ fsc -shutdown
```

- scala运行Java类文件: 
   - 不论是运行 `scalac` 还是 `fsc` 命令，都会产生出Java类文件，这些类文件可以用 `scala` 命令来运行；

```bash
$ scala Summer of love
```

**App特质调用Scala应用程序: **

- Scala提供了一个特质 `scala.App` ，帮助节省敲键盘的动作；
- 要使用这个特质，首先要在单例对象名后加上 `extends App` ，然后，并不是直接编写 `main` 方法，而是通过将打算放在main方法中的代码直接写在单例对象的花括号中；可以通过名为 `args` 的字符串数组来访问命令行参数；

```scala

   // FallWinterSpringSummer.scala
   import ChecksumAccumulator.calculate

   object FallWinterSpringSummer extends App {
      for (season <- List("fall", "winter", "spring")) {
         print(season + ": " + calculate(season))
      }
   }
```


## 基础类型、操作

**内容: **

- Scala基础类型
   - 字符类型: 

      - String

   - 数值类型: 

      - 整数类型: 

         - Byte

         - Short

         - Int

         - Long

         - Char

      - 浮点数类型: 

         - Float

         - Double

   - 布尔类型: 

      - Boolean

- Scala基础类型支持的操作

   - 操作

   - Scala表达式的操作符优先级

- 隐式转换“增强”(enrich)基础类型



### 基础类型

- 数值类型
   - 整数类型
      - `scala.Byte`
         - 8位带符号二进制补码整数
      - `scala.Short`
         - 16位带符号二进制补码整数
      - `scala.Int`
         - 32位带符号二进制补码整数
      - `scala.Long`
         - 64位带符号二进制补码整数
      - `scala.Char`
         - 16位无符号Unicode字符
   - 浮点数类型
      - `scala.Float`
         - 32位IEEE754单精度浮点数
      - `scala.Double`
         - 64位IEEE754双精度浮点数
- `java.lang.String`
   - Char的序列
- `scala.Boolean`
   - true
   - false



### 字面量



## 函数式对象

**主要内容: **

- 定义函数式对象的类；
- 类参数和构造方法；
- 方法和操作符；
- 私有成员
- 重写；
- 前置条件检查；
- 重载；
- 自引用；



### 背景、类设计

- 设计类对有理数的各项行为进行建模，包括允许它们被加、减、乘、除: 
   - 有理数(rational number):
      - `$\frac{n}{d}$`
         - `$n$`: 分子(numerator)
         - `$d$`: 分母(denominator)
   - 有理数相加、相减: 
      - 首先得到一个公分母，然后将分子相加；
   - 有理数相乘: 
      - 将另个有理数的分子和分母相乘；
   - 有理数相除: 
      - 将右操作元的分子分母对调，然后相乘；
- 数学中有理数没有可变的状态，可以将一个有理数跟另一个有理数相加，但结果是一个新的有理数，原始的有理数并不会“改变”；
   - 每一个有理数都会有一个 `Rational` 对象表示；

### 构建类

- 类名后面圆括号中的标识符称作类参数，Scala编译器会采集到类参数，并且创建一个主构造方法，接收同样的参数；
- 类参数(class parameter)
- 主构造方法(primary constructor)
- 在Scala总，类可以直接接收参数，Scala的表示法精简，类定义体内可以直接使用类参数，不需要定义字段并编写将构造方法参数赋值给字段的代码；
- Scala编译器会将在类定义体中给出的非字段或方法定义的代码编译进类的主构造方法中；



#### 重新实现toString方法

- Scala中的类中调用 `println()` 时，默认继承了java.lang.Object类的 `toString` 实现；

   - java.lang.Object.toString的主要作用是帮助程序员在调试输出语句，日志消息，测试失败报告，以及解释器和调试输出给出相应的信息；

   - 可以在类定义中重写(override)默认的toString实现；

```scala

   // Rational.scala

   class Rational(n: Int, d: Int) {
      override def toString = {
         n + "/" + d
      }
   }
```


#### 检查前置条件

- 面向对象编程的好处是可以将数据封装在对象里，以确保整个生命周期中的数据都是合法的；

- 对Rational类(class)，要确保对象(object)在构造时数据合法；

   - 分母`$`b`不能为0；

- 解决方式是对构造方法定义一个前置条件(precondition)，`$`d`必须为非0；

   - 前置条件是对传入方法或构造方法的值的约束，这是方法调用者必须要满足的，实现这个目的的一种方式是用 `require` ；

   - `require` 方法接收一个boolean的参数。如果传入的参数为true，require将会正常返回；否则，require会抛出 `IllegalArgumentException` 来阻止对象的构建；

```scala

   // Rational.scala

   class Rational(n: Int, d: Int) {
      
      // 前置条件检查
      require(d != 0)
      
      // 重写toString方法
      override def toString = {
         n + "/" + d
      }
   }
```


#### 添加字段

- 支持有理数加法: 

   - 定义一个 `add` 方法: 接收另一个Rational对象作为参数，为了保持Rational对象不变，这个add方法不能将传入的有理数加到自己身上，它必须创建并返回一个新的持有这两个有理数的和的Rational对象；

   - 当在 `add` 方法实现中用到类参数 `n` 和 `d` 时，编译器会提供这些类参数对应的值，但它不允许使用 `that.n` 和 `that.d` ，因为 `that` 并非指向执行 `add` 调用的那个参数对象；要访问 `that` 的分子和分母，需要将他们做成字段；

```scala

   // Rational.scala

   class Rational(n: Int, d: Int) {
      
      // 前置条件检查
      require(d != 0)
      
      // 初始化n和d的字段
      val numer: Int = n
      val denom: Int = d

      // 重写toString方法
      override def toString = {
         numer + "/" + denom
      }

      // 有理数加法
      def add(that: Rational): Rational = {
         new Rational(
            numer * that.denom + that.numer * denom, 
            denom * that.denom
         )
      }
   }
```


#### 自引用

- 关键字 `this` 指向当前执行方法的调用对象，当被用在构造方法里的时候，指向被构造的对象实例；

```scala

   def lessThan(that: Rational) = {
      this.numer * that.denom < that.numer * this denom
   }

   // or

   def lessThan(that: Rational) = {
      numer * that.denom < that.numer * denom
   }
```

```scala

   def max(that: Rational) = {
      if (this.lessThan(that)) {
         that
      }
      else {
         this
      }
   }
```



#### 辅助构造方法

- 有时需要给某个类定义多个构造方法，在Scala中，主构造方法之外的构造方法称为**辅助构造方法(auxiliary
   constructor)** ；

   - Scala中的每个辅助构造方法都必须首先调用同一个类的另一个构造方法；

      - Scala的辅助构造方法以 `def this(...)` 开始；

   - Scala被调用的这个构造方法要么是主构造方法(类的实例)，要么是另一个出现在发起调用的构造方法之前的另一个辅助构造方法；

- 定义一个当分母为1时的Rational类方法，只接受一个参数，即分子，而分母被定义为1；

```scala

   // Rational.scala

   class Rational(n: Int, d: Int) {
      
      // 前置条件检查
      require(d != 0)
      
      // 初始化n和d的字段
      val numer: Int = n
      val denom: Int = d

      // 辅助构造方法
      def this(n: Int) = {
         this(n, 1)
      }

      // 重写toString方法
      override def toString = {
         numer + "/" + denom
      }

      // 有理数加法
      def add(that: Rational): Rational = {
         new Rational(
            numer * that.denom + that.numer * denom, 
            denom * that.denom
         )
      }
   }
```


#### 私有字段和方法

- 实现正规化: 分子分母分别除以它们的最大公约数；

```scala

   // Rational.scala

   class Rational(n: Int, d: Int) {
      
      // 前置条件检查
      require(d != 0)
      
      // 分子分母的最大公约数
      private val g = gcd(n.abs, d.abs)

      // 初始化n和d的字段
      val numer = n / g
      val denom = d / g

      // 辅助构造方法
      def this(n: Int) = {
         this(n, 1)
      }

      // 重写toString方法
      override def toString = {
         numer + "/" + denom
      }

      // 有理数加法
      def add(that: Rational): Rational = {
         new Rational(
            numer * that.denom + that.numer * denom, 
            denom * that.denom
         )
      }

      // 求最大公约数方法
      private def gcd(a: Int, b: Int): Int = {
         if (b == 0) {
            a
         }
         else {
            gcd(b, a % b)
         }
      }
   }
```


#### 定义操作符

```scala

   // Rational.scala

   class Rational(n: Int, d: Int) {
      
      // 前置条件检查
      require(d != 0)
      
      // 分子分母的最大公约数
      private val g = gcd(n.abs, d.abs)

      // 初始化n和d的字段
      val numer = n / g
      val denom = d / g

      // 辅助构造方法
      def this(n: Int) = {
         this(n, 1)
      }

      // 重写toString方法
      override def toString = {
         numer + "/" + denom
      }

      // 有理数加法
      def + (that: Rational): Rational = {
         new Rational(
            numer * that.denom + that.numer * denom, 
            denom * that.denom
         )
      }

      def * (that: Rational): Rational = {
         new Rational(numer * that.numer, denom * that.denom)
      }

      // 求最大公约数方法
      private def gcd(a: Int, b: Int): Int = {
         if (b == 0) {
            a
         }
         else {
            gcd(b, a % b)
         }
      }
   }
```


## 内建控制结构

Scala只有为数不多的几个内建控制结构

- if
- while
- for
- try
- match
- 函数调用

Scala所有的控制结构都返回某种值作为结果，这是函数式编程语言采取的策略，程序被认为是用来计算出某个值，因此程序的各个组成部分也应该计算出某个值；



### if 表达式

**指令式风格: **

```scala

   var filename = "default.txt"
   if (!args.isEmpty) 
      filename = args(0)
```

**函数式风格: **

- val变量filename一旦初始化就不会改变，省去了扫描该变量整个作用域的代码来搞清楚他会不会变的必要；

```scala

   val filename = 
      if (!args.isEmpty) args(0)
      else "default.txt"
```

- 使用val的另一个好处是对等推理(equational
   reasoning)的支持；引入的变量等于计算出它的表达式(假设这个变量没有副作用),因此，可以在任何打算写变量的地方都可以直接用表达式来替换；

```scala

   println(if (!args.isEmpty) args(0) else "default.txt")
```


### while 循环

两种循环: 

- while

- do-while

while 语句示例: 

```scala

   def gcdLoop(x: Long, y: Long): Long = {
      var a = x
      var b = y
      while (a != 0) {
         val temp = a
         a = b % a
         b = temp
      }
      b
   }
```

do while 语句示例: 

```scala

   var line = ""

   do {
      line = readLine()
      println("Read: " + line)
   } while (line != "")
```

- while 和 do-while
  不是表达式，因为它们并不会返回一个有意义的值，返回值的类型是Unit；

- 实际上存在一个也是唯一一个类型为 `Unit`
  的值，这个值叫做单元值(unit value)，写作 `()` ；

- 用 `!=` 对类型为 `Unit` 的值和 `String`
  做比较将永远返回true；

- while 循环和 var 通常都是一起出现的；由于 while 循环
  没有返回值，想要对程序产生任何效果，while 循环通常要么更新一个 var
  要么执行I/O；因此，建议对代码中的 while
  循环保持警惕，如果对于某个特定的 while 或者 do-while
  循环，找不到合理的理由来使用它，那么应该尝试采用其他方案来完成同样的工作；

示例: 

```scala

   def greet() = {
      println("hi")
   }

   println(() == greet())
```

```scala

   val line = ""
   while (line = readLine() != "") {
      println("Read: " + line)
   }
```

```scala

   def gcd(x: Long, y: Long): Long = {
      if (y == 0) x else gcd(y, x % y)
   }
```


### for 表达式

- `<-`:生成器(generator)；



#### 遍历集合

遍历数组: 

```scala

   val fileHere = (new java.io.File(".")).listFiles
   for (file <- fileHere) {
      println(file)
   }
```

遍历区间(Range): 

```scala

   for (i <- 1 to 4) {
      println("Iteration " + i)
   }
```

遍历区间(不包含区间上届): 

```scala

   for (i <- 1 until 4) {
      println("Iteration " + i)
   }
```


#### 过滤

   如果不想完整地遍历集合，只想把集合过滤成一个子集，可以给for表达式添加**过滤器(filter)** ，过滤器是for表达式的圆括号中的一个if子句；

```scala

   val fileHere = (new java.io.File(".")).listFiles
   for (file <- fileHere if file.getName.endsWith(".scala")) {
      println(file)
   }
```

添加更多的过滤器: 

```scala

   for (
      file <- fileHere
      if file.isFile
      if file.getName.endsWith(".scala")
   ) {
      println(file)
   }
```


#### 嵌套迭代

   - 如果想添加多个<-子句，将得到嵌套的“循环”；

```scala

   def fileLines(file: java.io.File) = {
      scala.io.Source.fromFile(file).getLines().toList()
   }

   def grep(pattern: String) = {
      for (
         file <- fileHere
         if file.getName.endsWith(".scala")
         line <- fileLines(file)
         if line.trim.matches(pattern)
      ) {
         println(file + ":" + line.trim)
      }
   }

   grep(".*gcd.*")
```

可以省去分号: 

```scala

   val fileHere = (new java.io.File(".")).listFiles

   def fileLines(file: java.io.File) = {
      scala.io.Source.fromFile(file).getLines().toList()
   }

   def grep(pattern: String) = {
      for {
         file <- fileHere
         if file.getName.endsWith(".scala")
         line <- fileLines(file)
         if line.trim.matches(pattern)
      } {
         println(file + ":" + line.trim)
      }
   }

   grep(".*gcd.*")
```


#### 中途(mid-stream)变量绑定

   - 可以用 `=` 将表达式的结果绑定到新的变量上，被绑定的这个变量引入和使用起来跟val一样；

```scala

   val fileHere = (new java.io.File(".")).listFiles

   def fileLines(file: java.io.File) = {
      scala.io.Source.fromFile(file).getLines().toList()
   }

   def grep(pattern: String) = {
      for (
         file <- fileHere
         if file.getName.endsWith(".scala")
         line <- fileLines(file)
         trimed = line.trim
         if trimed.matches(pattern)
      ) {
         println(file + ": " + trimed)
      }
   }

   grep(".*gcd.*")
```


#### 输出一个新的集合

   - 可以在每次迭代中生成一个可以被记住的值；做法是在for表达式的代码体之前加上关键字yield；

```scala

   def scalaFiles = {
      for {
         file <- fileHere
         if file.getName.endsWith(".scala")
      } yield file
   }
```


### try表达式异常处理

- 方法除了正常地返回某个值外，也可以通过抛出异常终止执行；
- 方法的调用方要么捕获并处理这个异常，要么自我终止，让异常传播到更上层调用方；
- 异常通过这种方式传播，逐个展开调用栈，直到某个方法处理该异常或者没有更多的方法了为止；

### Scala 中没有 break 和 continue

- Scala 中没有 break 和 continue，因为它们会跟函数字面量不搭；

## 函数和闭包

   - 随着程序变大，需要某种方式将它们切成更小的、便于管理的块；

   - Scala将代码切分成函数；


### 方法

   - 定义函数最常用的方式是作为某个对象的成员，这样的函数被称为方法(method)；

示例: 

```scala

   // LongLines.scala

   import scala.io.Source

   object LongLines {
      def processFile(filename: String, width: Int) = {
         val source = Source.fromFile(filename)
         for (line <- source.getLines()) {
            processLine(filename, width, line)
         }
      }

      // processFile方法的助手方法
      private def processLine(filename: String, width: Int, line: String) = {
         if (line.length > width) {
            println(filename + ": " + line.trim)
         }
      }
   }
```


```scala

   // FindLongLines.scala

   import LongLines

   object FindLongLines {
      def main(args: Array[String]) = {
         val width = args(0).toInt
         for (arg <- args.drop(1)) {
            LongLines.processFile(arg, width)
         }
      }
   }
```

```bash
# 运行程序
$ fsc LongLines.scala FindLongLines.scala
$ scala FindLongLines 45 LongLines.scala
```


### 局部函数

   - 函数式编程风格的一个重要设计原则: 程序应该被分解成许多小函数，每个函数都只做明确的任务；
   - 上面的设计带来的问题: 助手函数的名称会污染整个程序的命名空间；
   - 局部函数: 可以在函数内部定义函数，就像局部变量一样，这样的函数只在包含它的代码块中可见；
   - 局部函数可以访问包含他们的函数的参数；

```scala

   //

   import scala.io.Source

   object LongLines {
      def processFile(filename: String, width: Int) = {
         // processLine只在函数processFile内部有效
         def processLine(line: String) = {
            if (line.length > width) {
               println(filename + ": " + line.trim)
            }
         }

         val source = Source.fromFile(filename)
         for (line <- source.getLines()) {
            processLine(line)
         }
      }
   }
```

### 一等函数

- Scala支持**一等函数(first-class function)** ；
- 不仅可以定义函数并调用它们，还可以用**匿名的字面量** 来编写函数并将它们作为**值(value)** 进行传递；
- **函数字面量** 被编译成类，并在运行时实例化成**函数值(function
value)** ，因此，函数字面量和函数值的区别在于，函数字面量存在于源码，而函数值以对象的形式存在于运行时，这跟类和对象的区别很相似；
- 函数值是对象，因此可以将他们存放在变量中，它们同时也是函数，所以可以用常规的圆括号来调用它们；



#### 函数字面量

```scala

   (x: Int) => x + 1
```

- 这里 `=>` 表示该函数将左侧的内容(任何整数 `$x$`)转换成右侧的内容 `$(x + 1)$`；
   - 这是一个将任何整数 `$x$` 映射成 `$x + 1$` 的函数

**函数字面量示例 1: **

```scala

   // 将函数值存放在变量中
   var increase = (x: Int) => x + 1

   // 调用函数值
   increase(0)
```

**函数字面量示例 2: **

```scala

   increase = (x: Int) => {
      println("We")
      println("are")
      println("here!")
      x + 1
   }

   // 函数调用
   increase(10)
```

**函数字面量示例 3: **

```scala

   // 所有的集合类都提供了foreach, filter方法
   val someNumbers = List(-11, -10, -5, 0, 5, 10)

   someNumbers.foreach((x: Int) => println(x))
   someNumbers.filter((x: Int) => x > 0) 
```


#### 函数字面量简写

**省去类型声明: **

- 函数字面量简写形式: 略去参数类型声明
- Scala编译器知道变量是什么类型，因为它看到这个函数用来处理的集合是一个什么类型元素组成的集合，这被称作**目标类型(target typing)** ，因为一个表达式的目标使用场景可以影响该表达式的类型；
- 当编译器报错时再加上类型声明，随着经验的积累，什么时候编译器能推断类型，什么时候不可以就慢慢了解了；

```scala

   val someNumbers = List(-11, -10, -5, 0, 5, 10)
   someNumbers.filter((x) => x > 0)
```

**省去圆括号: **

- 函数字面量简写形式: 省去某个靠类型判断的参数两侧的圆括号

```scala

   val someNumbers = List(-11, -10, -5, 0, 5, 10)
   someNumbers.filter(x => x > 0)
```

**占位符语法: **

- 为了让函数字面量更加精简，还可以使用下划线作为占位符，用来表示一个或多个参数，只要满足每个参数只在函数字面量中出险一次即可；
- 可以将下划线当成是表达式中需要被“填”的“空”，函数每次被调用，这个“空”都会被一个入参“填”上；
    - 多个下划线意味着多个参数，而不是对单个参数的重复使用；
- 可以用冒号给出入参的类型，当编译器没有足够多的信息来推断缺失的参数类型时；

```scala

   val someNumbers = List(-11, -10, -5, 0, 5, 10)
   someNumbers.filter(_ > 0)
```

```scala

   val f = (_: Int) + (_: Int)
   f(5, 10)
```

**部分应用函数(partially applied function): **

- 用下划线替换整个参数列表；
- 部分应用函数是一个表达式，在这个表达式中，并不给出函数需要的所有参数，而是给出部分，或者完全不给；

```scala

   val someNumbers = List(-11, -10, -5, 0, 5, 10)

   // 一般形式
   someNumbers.foreach(x => println(x))

   // 部分应用函数
   someNumbers.foreach(println _)
```

```scala

   def sum(a: Int, b: Int, c: Int) = {
      a + b + c
   }

   val a = sum _
```

这里，名为a的变量指向一个函数值对象，这个函数值是一个从Scala编译器自动从 `sum _` 这个部分应用函数表达式生成的类的实例，由编译器生成的这个类有一个接收三个参数的apply方法；



### 闭包

闭包示例: 

```scala

   var more = 1
   val addMore = (x: Int) => x + more
```

- 运行时从函数字面量 `(x: Int) => x + more` 创建出来的函数值(对象) `val addMore` 被称作**闭包(closure)** ；
- 自由变量(free varialbe): `more`
- 绑定变量(bound variable): `x`



### 特殊的函数调用形式(传参)



### 尾递归



## 控制抽象



## 组合继承



### 2.8.1 



### Scala的继承关系



## 特质

- 特质是 Scala 代码复用的基础单元。特质将方法和字段定义封装起来，
 然后通过将他们混入(mix in) 类的方式实现复用。它不同于类继承，
 类继承要求每个类都继承自一个(明确的)超类，而类可以同时混入任意数量的特质。
- Java 中提供了接口，允许一个类实现任意数量的接口。在 Scala 中没有接口的概念，
 而是提供了 **特质(trait)** ，它不仅实现了接口的功能，还具备了很多其他特性。
 Scala 的特质是代码重用的基本单元，可以同时拥有抽象方法和具体方法。
 在 Scala 中，一个类只能继承自一个超类，却可以实现多个特质，从而重用特质中的方法和字段，实现了多重继承。
- 特质的两种最常见的适用场景: 
- 将“瘦”接口拓宽为“富”接口
- 定义可叠加的修改



### 特质的定义

特质的定义和类的定义非常相似，区别是特质定义使用关键字 `trait`.

抽象方法不需要使用 `abstract`
关键字，特质中没有方法体的方法，默认就是抽象方法。

```scala

   trait CarId {
      // 定义一个抽象字段
      var id: Int 
      // 定义一个抽象方法
      def currentId(): Int 
   }
```


### 把特质混入类中

特质定义好后，就可以使用 `extend` 或 `with` 关键字把特质混入类中.

- 特质 `CarId`
- 混入特质 `CarId` 的类 `BYDCarId`
- 混入特质 `CarId` 的类 `BMWCarId`

```scala

   trait CarId {
      // 定义一个抽象字段
      var id: Int 
      // 定义一个抽象方法
      def currentId(): Int 
   }
```

```scala

   class BYDCardId extend CarId {
      // BYD 汽车编号从 10000 开始
      override var id = 10000
      // 返回汽车编号
      def currentId(): Int = {
         id += 1
         id
      }
   }
```

```scala

   class BMWCarId extend CarId {
      // BMW 汽车编号从 10000 开始
      override var id = 20000
      // 返回汽车编号
      def currentId(): Int = {
         id += 1
         id
      }
   }
```

整合代码: 

```scala

   // test.scala

   trait CarId {
      // 定义一个抽象字段
      var id: Int 
      // 定义一个抽象方法
      def currentId(): Int 
   }

   class BYDCarId extends CarId {
      // BYD 汽车编号从 10000 开始
      override var id = 10000
      // 返回汽车编号
      def currentId(): Int = {
         id += 1
         id
      }
   }

   class BMWCarId extends CarId {
      // BMW 汽车编号从 10000 开始
      override var id = 20000
      // 返回汽车编号
      def currentId(): Int = {
         id += 1
         id
      }
   }

   object MyCar {
      def main(args: Array[String]) {
         val myCarId1 = new BYDCarId()
         val myCarId2 = new BMWCarId()
         printf("My first CarId is %d.\n", myCarId1.currentId)
         printf("My second CarId is %d.\n", myCarId2.currentId)
      }
   }
```

编译、执行: 

```bash
$ scalac test.scala
$ scala -classpath . MyCar
```


### 特质可以包含具体实现

如果特质只包含了抽象字段和抽象方法，相当于实现了类似Java接口的功能。
实际上，特质也可以包含具体实现，也就是说，特质中的字段和方法不一定要是抽象的。

```scala

   trait CarGreeting {
      def greeting(msg: String) {
         println(msg)
      }
   }
```


### 把多个特质混入类中

```scala

   // test.scala

   trait CarId {
      // 定义一个抽象字段
      var id: Int 
      // 定义一个抽象方法
      def currentId(): Int 
   }

   trait CarGreeting {
      def greeting(msg: String) {
         println(msg)
      }
   }

   class BYDCarId extends CarId with CarGreeting {
      // BYD 汽车编号从 10000 开始
      override var id = 10000
      // 返回汽车编号
      def currentId(): Int = {
         id += 1
         id
      }
   }

   class BMWCarId extends CarId with CarGreeting {
      // BMW 汽车编号从 10000 开始
      override var id = 20000
      // 返回汽车编号
      def currentId(): Int = {
         id += 1
         id
      }
   }

   object MyCar {
      def main(args: Array[String]) {
         val myCarId1 = new BYDCarId()
         val myCarId2 = new BMWCarId()
         myCarId1.greeting("Welcome my first car.")
         printf("My first CarId is %d.\n", myCarId1.currentId)
         myCarId2.greeting("Welcomde my second car.")
         printf("My second CarId is %d.\n", myCarId2.currentId)
      }
   }
```

编译、执行: 

```bash
$ scalac test.scala
$ scala -classpath . MyCar
```


# 包(package)和包引入(import)

在处理程序，尤其是大型程序时，减少耦合(coupling)是很重要的。
所谓的耦合就是指程序不同部分依赖其他部分的程度。
低耦合能减少程序某个局部的某个看似无害的改动对其他部分造成严重后果的风险。
减少耦合的一种方式是以模块化的风格编写代码。
可以将程序切分成若干较小的模块，每个模块都有所谓的内部和外部之分。



## 将代码放进包里(模块化)

**在Scala中，可以通过两种方式将代码放进带名字的包里: **

- 在文件顶部放置一个 `package` 子句，让整个文件的内容放进指定的包: 
   - 也可以包含多个包的内容，可读性不好；

```scala

   package bobsrockets.naviagation
   class Navigator {}
```

- 在package子句之后加上一段用花括号包起来的代码块:
   - 更通用，可以在一个文件里包含多个包的内容；

```scala

   package bobsrockets {
      package naviagation {
         class Navigator {}
         package test {
            class NavigatorSuite {}
         }	
      }
   }
```



## 对相关代码的精简访问

1. 一个类不需要前缀就可以在自己的包内被别人访问；
2. 包自身也可以从包含他的包里不带前缀地访问到；
3. 使用花括号打包语法时，所有在包外的作用域内可被访问的名称，在包内也可以访问到；
4. Scala提供了一个名为 `__root__` 的包，这个包不会跟任何用户编写的包冲突，每个用户能编写的顶层包都被当做是 `__root__` 的成员；

```scala

   package bobsrockets {
      package navigation {
         class Navigation {
            // 一个类不需要前缀就可以在自己的包内被别人访问
            val map = new StarMap
         }
         class StarMap {}
      }

      class Ship {
         // 包自身也可以从包含他的包里不带前缀地访问到
         val nav = new navigation.Naviagtor
      }

      package fleets {
         class Fleet {
            def addShip() = {
               // 使用花括号打包语法时，所有在包外的作用域内可被访问的名称，在包内也可以访问到
               new Ship
            }
         }
      }
   }
```

```scala

   // =========================================
   // launch.scala
   // =========================================
   // launch_3
   package launch {
      class Booster3 {}
   }

   // =========================================
   // bobsrockets.scala
   // =========================================
   package bobsrockets {
      package navigation {

         // launch_1
         package launch {
            class Booster1 {}
         }

         class MissionControl {
            val booster1 = new launch.Booster1
            val booster2 = new bobsrockets.launch.Booster2
            val booster3 = new __root__launch.Booster3
         }
      }

      // launch_2
      package launch {
         class Booster2 {}
      }
   }
```



## 包引入

在Scala中，可以用 `import` 子句引入包和它们的成员；

**Scala包引入方式: **

- 对应Java的单类型引入；
- 对应Java的按需(on-demand)引入；
- 对应Java的对静态字段的引入；

编写包: 

```scala

   package bobsdelights {
      abstract class Fruit(val name: String, val color: String)

      object Fruits {
         object Apple extends Fruit("apple", "red")
         object Orange extends Fruit("orange", "orange")
         object Pear extends Fruit("pear", "yellowwish")

         val menu = List(Apple, Orange, Pear)
      }
   }
```

包引入: 

```scala

   // 到bobsdelights包中Fruit类的便捷访问, 对应Java的单类型引入
   import bobsdelights.Fruit

   // 到bobsdelights包中所以成员的便捷访问, 对应Java的按需(on-demand)引入
   import bobsdelights._

   // 到Fruits对象所有成员的便捷访问, 对应Java的对静态字段的引入
   import bobsdelights.Furits._

   // 引入函数showFruit的参数fruit(类型为Fruit)的所有成员
   def showFruit(fruit: Fruit) = {
      import fruit._
      println(name + "s are "+ color)
   }
```

**Scala包引入的灵活性: **

1. 引入可以出现在任意位置；
2. 引入可以引用对象(不论是单例还是常规对象)，而不只是包；
3. 引入可以重命名并隐藏某些被引入的成员；\\
   - 做法是将需要选择性引入的对象包在花括号内的引入选择器子句(import selector clause)中，引入选择器子句跟在要引入成员的对象后面；
      - 引入选择器可以包含: 
         - 一个简单的名称 `x` 。这将把x包含在引入的名称集里；
         - 一个重命名子句 `x => y` 。这会让名为x的成员以y的名称可见；
         - 一个隐藏子句 `x => _` 。这会从引入的名称集里排除掉x；
         - 一个捕获所有(catch-all)的 `_` 。这会引入除了之前子句中提到的成员之外的所有成员。如果要给出捕获所有子句，它必须出现在引入选择器的末尾；

```scala

   // 引入对象(object)
   import bobsdelights.Fruits.{Apple, Orange}

   // 引入对象的所有成员
   import Fruits.{_}

   // 对引入对象(Apple)重命名
   import bobsdelights.Fruits.{Apple => McIntosh, Orange}

   import java.sql.{Date => SDate}
   import java.{sql => s}

   // 引入Fruits对象的所有成员，并把Apple重命名为McIntosh
   import Fruits.{Apple => McIntosh, _}

   // 引入Pear之外的所有成员
   import Fruits.{Pear => _, _}
```


## 隐式引入

Scala对每个程序都隐式地添加了一些引入；即每个扩展名为 `.scala` 的源码文件的顶部都添加了如下三行引入子句: 

- `java.lang包` 包含了标准的Java类
   - 总是被隐式地引入到Scala文件中，由于java.lang是隐式引入的，举例来说，可以直接写Thread，而不是java.lang.Thread；
- `scala包` 包含了Scala的标准库
   - 包含了许多公用的类和对象，由于scala是隐式引入的，举例来说，可以直接写List，而不是scala.List
- `Predef` 对象包含了许多类型、方法、隐式转换的定义，由于Predef是隐式引入的，举例来说，可以直接写assert，而不是Predef.assert；

```scala

   // java.lang包的全部内容
   import java.lang._ 

   // scala包的全部内容
   import scala._

   // Predef对象的全部内容
   import Predef._



## 访问修饰符

包、类、对象的成员可以标上 `private` 和 `protected` 等访问修饰符，这些修饰符将对象的访问限定在特定的代码区域。

### 私有成员(private)

标为private的成员只在包含该定义的类(class)或对象(object)内部可见；

```scala

   class Outer {
      class Inner {
         private def f() = {println("f")}
         class InnerMost {
            // 可以访问f
            f()
         }
      }
      // 错误: 无法访问f, Java可以
      (new Inner).f()
   }
```


### 受保护成员(protected)

标为protected的成员只能从定义该成员的子类访问；

```scala

   package p {
      class Super {
         protected def f() = {println("f")}
      }

      class Sub extends Super {
         // 可以访问f，Sub是Super的子类
         f()
      }

      class Other {
         // 错误: 无法访问f, Java可以
         (new Super).f()
      }
   }
```


### 公共成员

Scala没有专门的修饰符用来标记公共成员: 任何没有标为private或protected的成员
都是公共的；公共成员可以从任意位置访问到；



### 保护的范围

- 可以用限定词对Scala中的访问修饰符机制进行增强
- 形如 `private[X]` ， `protected[X]` 的修饰符的含义是对此成员的访问限制“上至”X都是私有或受保护的，其中X表示某个包含该定义的包、类、对象；

```scala

   package bobsrockets {

      package navigation {

         // Navigator类对bobsrockets包内的所有类和对象都可见，比如: launch.Vehicle对象中对Navigator的访问是允许的
         private[bobsrockets] class Navigator {

            // 
            protected[navigation] def useStarChart() = {}

            class LegOfJourney {
               //
               private[Navigator] val distance = 100
            }

            // 仅在当前对象内访问
            private[this] var speed = 200
         }
      }

      package launch {
         import navigation._

         object Vehicle {
            private[launch] val guide = new Navigator
         }
      }
   }
```


### 可见性和伴生对象


## 包对象(package object)

- 任何能放在类级别的定义，都能放在包级别；
- 每个包都允许有一个包对象，任何放在包对象里的定义都会被当做这个包本身的成员；
- 包对象经常用于包级别的类型别名和隐式转换；
- 包对象会被编译为名为package.class的类文件，改文件位于它增强的包的对应目录下；

举例: 

```scala

   package bobsdelights {
      abstract class Fruit(val name: String, val color: String)

      object Fruits {
         object Apple extends Fruit("apple", "red")
         object Orange extends Fruit("orange", "orange")
         object Pear extends Fruit("pear", "yellowwish")

         val menu = List(Apple, Orange, Pear)
      }
   }
```

```scala

   // bobsdelights/package.scala文件
   // 包对象
   package object bobsdelights {
      def showFruit(fruit: Fruit) = {
         import fruit._
         println(name + "s are " + color)
      }
   }
```

```scala

   // PrintMenu.scala文件

   package printmenu
   import bobsdelights.Fruits
   import bobsdelights.showFruit

   object PrintMenu {
      def main(args: Array[String]) = {
         for (fruit <- Fruits.menu) {
            showFruit(fruit)
         }
      }
   }
```


# 断言和测试

断言和测试是用来检查程序行为符合预期的两种重要手段；


## 断言

在Scala中，断言的写法是对预定义方法 `assert` 的调用；

- `assert` 方法定义在 `Predef` 单例对象中，每个Scala源文件都会自动引入该单例对象的成员；
- `assert(condition)`
   - 若condition不满足，抛出AssertionError
- `assert(condition, explanation)`
   - 首先检查condition是否满足，如果不满足，抛出包含给定explanation的AssertionError;
   - explanation 的类型是 Any,因此可以传入任何对象，assert方法将调用explanation的toString方法来获取一个字符串的解释放入AssertionError；

用assert进行断言: 

```scala

   def above(that: Element): Element = {
      val this1 = this widen that.width
      val that1 = that widen this.width
      assert(this1.width == that1.width)
      elem(this1.contents ++ that1.contents)
   }
```
用Predef.ensuring进行断言: 

```scala

   private def widen(w: Int): Element = {
      if (w <= width) {
         this
      }
      else {
         val left = elem(" ", (w - width) / 2, height)
         var right = elem(" ", w - widht - left.width, height)
         left beside this beside right
      } ensuring (w <= _.width)
   }
```


## 测试



# 样例类和匹配模式

## 样例类

- 样例类是Scala用来对对象进行模式匹配二进行的不需要大量的样板代码的方式。笼统的说，要做的就是对那些希望能做模式匹配的类加上一个 `case` 关键字；
- 样例类会让Scala编译器对类添加一些语法上的便利；
   - 1.首先，它会添加一个跟类同名的工厂方法；
   - 2.其次，参数列表中的参数都隐式地获得了一个val前缀，因此它们会被当做字段处理；
   - 3.再次，编译器会帮我们以自然地方式实现toString,hashCode和equals方法；
   - 4.最后，编译器还会添加一个copy方法用于制作修改过的拷贝，这个方法可以用于制作除了一两个属性不同之外其余完全相同的该类的新实例；
   - 5.样例类最大的好处是他们支持模式匹配；

示例: 

```scala

   abstract class Expr

   // 变量
   case class Var(name: String) extends Expr
   // 数
   case class Number(num: Double) extends Expr
   // 一元操作符
   case class UnOp(operator: String, arg: Expr) extends Expr
   // 二元操作符
   case class BinOp(operator: String, left: Expr, right: Expr) extends Expr
```

- 样例类会添加一个跟类同名的工厂方法，嵌套定义，不需要 `new`

```scala

   val v = Var("x")
   val op = BinOp("+", Number(1), v)
```

- 参数列表中的参数都隐式地获得了一个val前缀，因此它们会被当做字段处理

```scala

   v.name
   op.operator
   op.left
   op.right
```

- 编译器会帮我们以自然地方式实现toString,hashCode和equals方法

```scala

   println(op)
   op.right == Var("x")
```

- 编译器还会添加一个copy方法用于制作修改过的拷贝

```scala

   op.copy(operator = "-")
   print(op)
```


## 模式匹配



### 模式匹配形式

   - 模式匹配包含一系列以case关键字开头的可选分支(alternative)
   - 每一个可选分支都包括一个模式(pattern)以及一个或多个表达式，如果模式匹配成功了，这些表达式就会被求值，箭头 `=>` 用于将模式和表达式分开;
   - 一个mathc表达式的求值过程是按照模式给出的顺序逐一进行尝试的；
   - 模式匹配mathc特点
   - Scala的match是一个表达式，也就是说它总是能得到一个值；
   - Scala的可选分支不会贯穿到下一个case；
   - 如果没有一个模式匹配上，会抛出MatchError的异常，所以需要确保所有的case被覆盖到，哪怕意味着需要添加一个什么都不做的缺省case；

基本形式: 

**选择器 match {可选分支}**

示例函数: 

```scala

   def simplifyTop(expr: Expr): Expr = expr match {
      case UnOp("-", UnOp("-", e)) => e
      case BinOP("+", e, Number(0)) => e
      case BinOp("+", e, Number(1)) => e
      case _ => expr
   }
```


### 模式种类



#### 通配模式



#### 常量模式



#### 变量模式



#### 构造方法模式



#### 序列模式



#### 带类型的模式



#### 变量绑定
