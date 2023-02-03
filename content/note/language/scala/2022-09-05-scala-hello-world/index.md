---
title: Scala 入门
author: 王哲峰
date: '2022-09-05'
slug: scala-hello-world
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

- [Scala Env](#scala-env)
  - [安装 Java](#安装-java)
  - [安装 IDEA](#安装-idea)
  - [安装 sbt](#安装-sbt)
  - [安装 Scala](#安装-scala)
- [Scala 入门](#scala-入门)
  - [使用 Scala 解释器](#使用-scala-解释器)
  - [定义 Scala 变量](#定义-scala-变量)
  - [定义 Scala 函数](#定义-scala-函数)
  - [编写 Scala 脚本](#编写-scala-脚本)
  - [用 while 做循环，用 if 做判断](#用-while-做循环用-if-做判断)
  - [用 foreach 和 for 遍历](#用-foreach-和-for-遍历)
  - [[Array] 用类型参数化数组](#array-用类型参数化数组)
  - [[List] 使用列表](#list-使用列表)
  - [[Tuple] 使用元组](#tuple-使用元组)
  - [[Set, Map]使用集和映射](#set-map使用集和映射)
    - [Set](#set)
      - [Scala Set 的类继承关系](#scala-set-的类继承关系)
      - [创建、初始化一个不可变 Set](#创建初始化一个不可变-set)
      - [创建、初始化一个可变 Set](#创建初始化一个可变-set)
      - [创建、初始化一个不可变 HashSet](#创建初始化一个不可变-hashset)
    - [Map](#map)
      - [Scala Map 的类继承关系](#scala-map-的类继承关系)
      - [创建、初始化一个不可变 Map](#创建初始化一个不可变-map)
      - [创建、初始化一个可变 HashMap](#创建初始化一个可变-hashmap)
  - [识别函数式编程风格](#识别函数式编程风格)
    - [指令式示例](#指令式示例)
    - [函数式示例](#函数式示例)
  - [从文件读取文本行](#从文件读取文本行)
</p></details><p></p>

# Scala Env

The Scala Programming Language

Scala combines object-oriented and functional programming in one concise, high-level language. 
Scala's static types help avoid bugs in complex applications, and its JVM and JavaScript runtimes 
let you build high-performance systems with easy access to huge ecosystems of libraries.

Scala runs on ...

- JVM
- JavaScript in your browser
- Natively with LLVM

Scala in a Nutshell

- 与 Java 互操作(Seamless Java Interop)
    - Scala runs on the JVM, so Java and Scala stacks can be freely mixed for totally semless integration.
- 类型推断(Type Inference)
    - So the type system doesn’t feel so static. Don’t work for the type system. Let the type system work for you!
- 并发与分配(Concurrency & Distribution)
- 特质(Traits)
    - Combine the flexibility of Java-style interfaces with the power of classes. Think principled multiple-inheritance.
- 模式匹配(Pattern Matching)
    - Think “switch” on steroids. Match against class hierarchies, sequences, and more.
- 高阶函数(Higher-order Functions)
    - Functions are first-class objects. Compose them with guaranteed type safety. Use them anywhere, pass them to anything.

## 安装 Java

- [JaveSE](https://www.oracle.com/cn/java/technologies/javase-downloads.html) 

## 安装 IDEA

- [IDEA](https://www.jetbrains.com/idea/download/#section=mac) 

## 安装 sbt

- [sbt](https://www.scala-sbt.org/index.html) 

## 安装 Scala

- [Scala Download](https://www.scala-lang.org/download/) 
- [Scala Org](https://www.scala-lang.org/) 



# Scala 入门


## 使用 Scala 解释器

启动 Scala Shell:

```bash
$ scala
```

退出解释器:

```bash
$ :quit
```

```bash
$ :q
```

## 定义 Scala 变量


Scala 变量分为两种:

- `val`:一旦初始化就不能被重新赋值;
- `var`:在整个生命周期内可以被重新赋值;

示例:

```scala
val msg = "Hello world!"
val msg2: java.lang.String = "Hello again, world!"
val msg3: String = "Hello yet again, world!"
```


## 定义 Scala 函数


- 函数定义由 `def` 开始，然后是函数名和圆括号中的以逗号隔开的参数列表;
  - 每个参数的后面都必须加上冒号(`:`)开始的类型标注。因为 Scala 编译器并不会推断函数参数的类型;
- 可以在参数列表的后面加上以冒号开始的函数的结果类型(reslut type)。有时，
  Scala 编译器需要给出函数的结果类型，比如:如果函数是**递归的(recursive)**，就必须显式地给出函数的结果类型;
- 在函数的结果类型之后，是一个等号和用花括号括起来的函数体。
  - 函数体之前的等号有特别的含义，表示在函数式的世界观里，函数定义的是一个可以获取到结果值的表达式;
  - 如果函数只有一条语句，也可以不使用花括号;
  - 函数返回类型为 `()Unit` 表示该函数并不返回任何有实际意义的结果
     - Scala 的 `Unit` 类型和 Java 的 `void` 类型类似，
       每一个 Java 中返回 `void` 的方法都能被映射称 Scala 中返回 `Unit` 的方法。
       因此，结果类型为 `Unit` 的方法之所以被执行，完全是为了它们的副作用;

一般形式:

```scala
def max(x: Int, y: Int): Int = {
   if (x > y) 
      x
   else 
      y
}
```

最复杂形式:

```scala
def max(x: Int, y: Int): Int = {
   if (x > y) {
      x
   }
   else {
      y
   }
}
```

最简形式:

```scala
def max(x: Int, y: Int) = if (x > y) x else y
```

## 编写 Scala 脚本

- 脚本不过是一组依次执行的语句;
- 命令行参数可以通过名为 `args` 的 Scala 数组 (Array) 获取;

**编写脚本1,执行脚本1:**

```scala
// hello.scala

println("Hello world, from a script!")
```

```bash
$ scala hello.scala
```

**编写脚本2,执行脚本2:**

```scala
// helloarg.scala

/* 对第一个命令行参数说hello */
println("Hello, " + args(0) + "!")
```

```bash
$ scala helloarg.scala plant
```



## 用 while 做循环，用 if 做判断

**打印命令行参数(带换行符):**

```scala
// printargs.scala

var i = 0
while (i < args.length) {
    println(args(i))
    i += 1
}
```

```bash
$ scala printargs.scala Scala is fun
```

**打印命令行参数(不带换行符):**

```scala

      // echoargs.scala

      var i = 0
      while (i < args.length) {
         if (i != 0)
            print(" ")
         print(args(i))
         i += 1
      }
      println()
```

```bash
$ scala echoargs.scala Scala is more even fun
```

## 用 foreach 和 for 遍历

- 指令式编程风格(imperative)     
    - 依次给出指令，通过循环来遍历，经常变更别不同函数共享的状态;
- 函数式编程风格(functional)
    - 函数式编程语言的主要特征之一就是函数是一等的语法单元;

foreach:

```scala
// foreachargs.scala

args.foreach(arg => println(arg))
args.foreach((arg: String) => println(arg))
args.foreach(println)
````

```bash
$ scala foreachargs.scala Concise is nice
```


for: Scala 只支持指令式 for 语句的 `for表达式`

```scala
// forargs.scala

for (arg <- args) {
    println(arg)
}
```

```bash
$ scala forargs.scala for arg in args
```

***
**Note:**

- 在 for 表达式中，符号 `<-` 左边的 arg 是一个 `val`
      变量的名字，尽管 arg 看上去像是
      `var`，因为每一次迭代都会拿到新的值，但他确实是个
      `val`，虽然 arg 不能在 for 表达式的循环体内被重新赋值，但对于
      `<-`右边的 args 数组中的每一个元素，一个新的名为 arg 的
      `val` 会被创建出来，初始化成元素的值，这时 for
      表达式的循环体才被执行;
***

## [Array] 用类型参数化数组

- 用 `new` 来实例化 `对象(object)`、`类(class)` 的 `实例(instance)`;
- 用值来参数化一个实例，做法是在构造方法的括号中传入对象参数;
- 用类型来参数化一个实例，做法是在方括号里给出一个或多个类型;
    - 当同时用类型和值来参数化一个实例时，先是方括号包起来的类型(参数)，然后才是用圆括号包起来的值(参数);

用值来参数化一个实例:

```scala
val big = new java.math.BigInteger("12345")
```

用类型来参数化一个实例:

```scala
val greetStrings = new Array[String](3)
greetStrings(0) = "Hello"
greetStrings(1) = ", "
greetStrings(2) = "world!\n"
for (i <- 0 to 2) {
    print(greetStrings(i))
}
```

```scala
val greetStrings: Array[String] = new Array[String](3)
```

***
**Note:**

* 当用 `val` 定义一个变量时，变量本身不能被重新赋值，但它指向的那个对象是有可能发生变化的。
      可以改变 `Array[String]` 的元素，因此 `Array` 本身是可变的;
*** 

## [List] 使用列表


- Scala 数组 `Array` 是一个拥有相同类型的对象的
    `可变序列`。虽然无法在数组实例化后改变其长度，却可以改变它的元素值，因此数组是可变的对象;

- 对于需要拥有相同类型的对象的不可变序列的场景，可以使用 Scala 的
    `List` 类。Scala 的 `List`（即 `scala.List`）跟 Java 的
    `java.util.List` 的不同在于 Scala 的 `List` 不可变的，而 Java
    的 `List` 是可变的。更笼统的说，Scala 的 `List` 被设计为
    `允许函数是风格的编程`。

创建并初始化列表:

```scala
val oneTwoThree = List(1, 2, 3)
```

列表拼接方法 `:::`:

```scala
val oneTwo = List(1, 2)
val threeFour = List(3, 4)
val oneTwoThreeFour = oneTwo ::: threeFour
println(oneTwo + " and " + threeFour + " were not mutated.")
println("Thus, " + oneTwoThreeFour + " is a new list.")
```

列表增加方法 `::`:

```scala
val twoThree = List(2, 3)
val oneTwoThree = 1 :: twoThree
println(oneTwoThree)
```

空列表快捷方式 `Nil`:

初始化一个新的列表的另一种方式是用`::`将元素串接起来，并将`Nil`作为最后一个元素;

```scala
val oneTwoThree = 1 :: 2 :: 3 :: Nil
println(oneTwoThree)
```

列表的一些方法和用途:


| List方法 | 方法用途 |
|----|----|
| `List()`, `Nil`                                                | 空 List  |
| `List("Cool", "tools", "rules")`                                 | 创建一个新的List[String] |
| `val thril = "Will" :: "fill" :: "until" :: Nil`                 | 创建一个新的List[String]  |
| `List("a", "b") ::: List("c", "d")`                              | 将两个List拼接起来 |
| `thril.mkString(", ")`                                           | 返回一个用List的所有元素组合成的字符串  |
| `thril(2)`                                                       | 返回List中下标为2的元素 |
| `thril.head`                                                     | 返回List首个元素  |
| `thril.init`                                                     | 返回List除最后一个元素之外的其他元素组成的List |
| `thril.last`                                                     | 返回List的最后一个元素 |
| `thril.tail`                                                     | 返回List除第一个元素之外的其他元素组成的List |
| `thril.length`                                                   | 返回List的元素个数 |
| `thril.isEmpty`                                                  | 判断List是否是空List |
| `thril.drop(2)`                                                  | 返回去掉了List的头两个元素的List |
| `thril.dropRight(2)`                                             | 返回去掉了List的后两个元素的List |
| `thril.reverse`                                                  | 返回包含List的所有元素但顺序反转的List |
| `thril.count(s => s.length == 4)`                                | 对List中满足条件表达式的元素计数 |
| `thril.filter(s => s.length == 4)`                               | 按顺序返回List中所有长度为4的元素List |
| `thril.filterNot(s => s.length == 4)`                            | 按顺序返回List中所有长度不为4的元素List |
| `thril.exists(s => s == "until")`                                | 判断List中是否有字符串元素为"until" |
| `thril.forall(s => s.endsWith("l"))`                             | 判断List中是否所有元素都以字母"l"结尾 |
| `thril.foreach(s => println(s))`                                 | 对List中的每个字符串进行print |
| `thril.foreach(println)`                                         | 对List中的每个字符串进行print,精简版 |
| `thril.map(s => s + "y")`                                        | 返回一个对List所有字符串元素末尾添加"y"的新字符串的List |
| `thril.sort((s, t) => s.charAt(0).toLower < t.charAt(0).toLower)`| 返回包含List的所有元素，按照首字母小写的字母顺序排列的List |


## [Tuple] 使用元组

- 元组是不可变的;
- 元组可以容纳不同类型的元素;
- 要实例化一个新的元组，只需要将对象放在圆括号当中，用逗号隔开即可;
- 实例化好一个元组后，就可以用`._n`, `n = 1, 2,...`来访问每一个元素;
- 元组中的每个元素有可能是不同的类型;
- 目前 Scala 标准类库仅定义到 Tuple22，即包含 22 个元素的数组;
- 元组类型:`Tuplel[Type1, Type2, ...]`， 比如: `Tuple2[Int, String]`

```scala
val pair = (99, "Luftballons")
println(pair._1)
println(pair._2)
```


## [Set, Map]使用集和映射

- Array 永远是可变的
- 必须容纳相同类型的元素;
- 长度不可变;
- 元素可变;
- List 永远是不可变的
- 必须容纳相同类型的元素;
- 长度不可变;
- 元素不可变;
- Tuple 永远是不可变的
- 可以容纳不同类型的元素;
- 长度不可变;
- 元素不可变;
- Scala 通过不同的类继承关系来区分 Set、Map 的可变和不可变;
- Scala 的 API 包含了一个基础的`特质(trait)`来表示 Set、Map
- Scala 提供了两个`子特质(subtrait)`
    - 一个用于表示可变 Set、可变 Map
    - 另一个用于表示不可变 Set、不可变 Map;



### Set

#### Scala Set 的类继承关系

- scala.collection **Set** 《trait》
   - scala.collection.immutable **Set** 《trait》
      - scala.collection.immutable
         - **HashSet**
   - scala.collection.mutable **Set** 《trait》
      - scala.collection.mutable
         - **HashSet**

#### 创建、初始化一个不可变 Set

```scala
var jetSet = Set("Boeing", "Airbus")
jetSet += "Lear"
println(jetSet.contains("Cessna"))
```

```scala
import scala.collection.immutable

var jetSet = immutable.Set("Boeing", "Airbus")
jetSet += "Lear"
println(jetSet.contains("Cessna"))
```


#### 创建、初始化一个可变 Set

```scala
import scala.collection.mutable

val movieSet = mutable.Set("Hitch", "Poltergeist")
movieSet += "Shrek"
println(movieSet)
```

#### 创建、初始化一个不可变 HashSet

```scala
import scala.collection.immutable.HashSet

val hashSet = HashSet("Tomatoes", "Chilies")
println(hashSet + "Coriander")
```

### Map

#### Scala Map 的类继承关系

- scala.collection **Map** 《trait》
   - scala.collection.immutable **Map** 《trait》
      - scala.collection.immutable
         - **HashMap**
   - scala.collection.mutable **Map** 《trait》
      - scala.collection.mutable
         - **HashMap**

#### 创建、初始化一个不可变 Map

```scala
val romanNumeral = Map(
    1 -> "I",
    2 -> "II",
    3 -> "III",
    4 -> "IV",
    5 -> "V"
)
println(romanNumeral)
```

```scala
import scala.colection.immutable

val romanNumeral = immutable.Map(
    1 -> "I",
    2 -> "II",
    3 -> "III",
    4 -> "IV",
    5 -> "V"
)
println(romanNumeral)
```


#### 创建、初始化一个可变 HashMap

```scala
import scala.collection.mutable

val treasureMap = mutable.Map[Int, String]()
treasureMap += (1 -> "Go to island.")
treasureMap += (2 -> "Find big X on ground.")
treasureMap += (3 -> "Dig.")
println(treasureMap(2))
```


## 识别函数式编程风格

- 代码层面:
    - 一个显著的标志是: 如果代码包含任何 `var` 变量，通常是指令式风格的，而如果代码完全没有 `var` (只包含 `val`)，那么很可能是函数式风格的。因此，一个向函数式风格转变的方向是尽可能不用 `var`;
- 每个有用的程序都会有某种形式的副作用。否则，它对外部世界就没有任何价值。倾向于使用无副作用的函数鼓励你设计出将带有副作用的代码最小化的额程序。这样做的好处之一就是让你的程序更容易测试;


### 指令式示例

```scala
def printArgs(args: Array[String]): Unit = {
    var i = 0
    while (i < args.length) {
    println(args(i))
    i += 1
    }
}
```


### 函数式示例

- “不纯”的函数式:
   - 函数有副作用(向标准输出流输出打印)，带有副作用的函数的标志特征是结果类型是 `Unit`.

```scala
def printArgs(args: Array[String]): Unit = {
    for (arg <- args) {
    println(s)
    }
}
```

```scala
def printArgs(args: Array[String]): Unit = {
    args.foreach(println)
}
```

- “纯”函数式:
   - 函数没有副作用，没有 `var`

```scala
def formatArgs(args: Array[String]) = {
    args.mkString("\n")
}
```


## 从文件读取文本行

日常任务的脚本处理文件中的文本行

```scala
import scala.io.Source

if (args.length > 0) {
    for (line <- Source.fromFile(args(0)).getLines()) {
    println(line.length + " " + line)
    }
}
else {
    Console.err.println("Please enter filename")
}
```

```bash
$ scala countchars1.scala countchars1.scala
```

```scala
import scala.io.Source

def widthOfLength(s: String) = {
    s.length.toString.length
}

if (args.length > 0) {
    val lines = Source.fromFile(args(0)).getLines().toList
    val longestLine = lines.reduceLeft((a, b) => if (a.length > b.length) a else b)
    val maxWidth = widthOfLength(longestLine)
    for (line <- lines) {
    val numSpace = maxWidth - widthOfLength(line)
    val padding = " " * numSpace
    println(padding + line.length + " | " + line)
    }
}
else {
    Console.err.println("Please enter filename.")
}
```

```bash
$ scala countchars2.scala countchars2.scala
```