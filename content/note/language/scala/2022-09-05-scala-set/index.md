---
title: Scala Set
author: 王哲峰
date: '2022-09-05'
slug: scala-set
categories:
  - scala
tags:
  - tool
---


Scala集合对象

- 可变集合
- List()
- 不可变集合
- Array()
- 同构
- List()
- Array()
- 异构

# 数组 Array


# 列表 List

1. List 是不可变的，即 List 的元素不能通过赋值改变；
2. List 的机构是递归的，即链表(linked list)；


## List 字面量

示例: 

```scala
val fruit = List("apples", "oranges", "pears")

val nums = List(1, 2, 3)

val diag3 = List(
    List(1, 0, 0),
    List(0, 1, 0),
    List(0, 0, 1)
)

val empty = List()
```


## List 类型

1. List是同构的(homogeneous)的，即同一个List的所有元素都必须是相同的类型；
    - 元素类型为T的List的类型写作 `List[T]` ；
2. List类型是协变的(covariant)，即对每一组类型S和T，如果S是T的子类型，那么 `List[S]` 就是 `List[T]` 的子类型；
   - 例如: List[String]是List[Object]的子类型；
   - 空列表的类型为 `List[Nothing]` ；在Scala的累继承关系中，Nothing是底类型，所以对于任何T类型而言，List[String]都是List[T]的子类型；

示例: 

```scala
val xs: List[String] = List()
```


## List 构建

- 所有的List都构建自两个基础的构建单元:  `Nil` ， `::` (读作“cons”)；
- `Nil` : 空List；
- `::` : 在List前追加元素；
- 用 `List()` 定义的List字面量不过是最终展开成由 `Nil` 和 `::` 组合成的这样形式的包装方法而已；
- 因为 `::` 是右结合的，因此，可以在构建List时去掉圆括号；

示例: 

```scala
val fruit = "apples" :: ("oranges" :: ("pears" :: Nil))
val nums = 1 :: (2 :: (3 :: (4 :: Nil)))
val diag3 = (1 :: (0 :: (0 :: Nil))) ::
        (0 :: (1 :: (0 :: Nil))) ::
        (0 :: (0 :: (1 :: Nil))) :: Nil
val empty = Nil

// 去掉圆括号
val fruit = "apples" :: "oranges" :: "pears" :: Nil
val nums = 1 :: 2 :: 3 :: 4 :: Nil
val diag3 = (1 :: 0 :: 0 :: Nil) ::
        (0 :: 1 :: 0 :: Nil) ::
        (0 :: 0 :: 1 :: Nil) :: Nil
val empty = Nil
```


## List 基本操作

   - 对List的所有操作都可以用三项来表述: 
   - `head`: 返回List的第一个元素；
      - 只对非空List有定义，否则抛出异常
   - `tail`: 返回List中除第一个元素之外的所有元素；
      - 只对非空List有定义，否则抛出异常
   - `isEmpty`: 返回List是否为空List；

示例: 对一个数字List `x :: xs` 的元素进行升序排列，采用插入排序(insertion sort)

```scala
def isort(xs: List[Int]): List[Int] = {
    if (xs.isEmpty) {
        Nil
    } else {
        insert(xs.head, isort(xs.tail))
    }
}

def insert(x: Int, xs: List[Int]): List[Int] = {
    if (xs.isEmpty || x <= xs.head) {
        x :: xs
    } else {
        xs.head :: insert(x, xs.tail)
    }
}

val L = 1 :: (3 :: 4 :: 2 :: Nil)
```


## List 模式

- List可以用模式匹配解开， List模式可以逐一对应到List表达式；
- 用List(...)这样的模式来匹配List的所有元素；
- 用::操作符和Nil常量一点点地将List解开；

示例 1: 

```scala
val fruit = List("apples", "oranges", "pears")
val fruit = "apples" :: "oranges" :: "pears" :: Nil
```

```scala
// List(...)模式匹配
val List(a, b, c) = fruit
```

```
// output
a: String = apples
b: String = oranges
c: String = pears
```

```scala

   // ::和Nil将List解开
   val a :: b :: rest = fruit
```

```
// output
a: String = apples
b: String = oranges
rest: List[String] = List(pears)
```

示例 2: 实现插入排序

```scala
def isort(xs: List[Int]): List[Int] =  xs match {
    case List() => List()
    case x :: xs1 => insert(x, isort(xs1))
}

def insert(x: Int, xs: List[Int]): List[Int] = xs match {
case List() => List(x)
case y :: ys => if (x <= y) {
                    x :: xs
                } else {
                    y :: insert(x, ys)
                }
}
```


## List 方法


### List 类的初阶方法

- 如果一个方法不接收任何函数作为入参，就被称为初阶(first-order)方法；

#### List拼接方法

- `xs ::: ys` : 接收两个列表参数作为操作元；
- 跟 `::` 类似， `:::` 也是右结合的，因此 `xs ::: ys ::: zs` 等价于 `xs ::: (ys ::: zs)` ；

示例: 

```scala
List(1, 2) ::: List(3, 4, 5)
List() ::: List(1, 2, 3, 4, 5)
List(1, 2, 3, 4) ::: List(4)
```


#### 分治(Divide and Conquer)原则



### List类的高阶方法






# 序列 Seq


序列类型可以用来处理依次排列分组的数据；

- List()
- Array()
- StringOps()

## List

List支持在头部快速添加和移除条目，不过并不提供快速地按下标访问的功能，因为事先这个功能需要线性地遍历List；

相关内容: 

- 第三章第8步；
- 第十六章；
- 第22章；

## Array

Array保存一个序列的元素，并使用从0开始的下标高效地访问(获取或更新)指定位置的元素值；

相关内容: 

- 第三章第步；
- 7.3节使用for表达式遍历数组；
- 第10章的二维布局类库；

```scala
// 创建Array
val fiveInts = new Array[Int](5)
val fiveToOne = Array(5, 4, 3, 2, 1)

// 访问数组中的元素,改变数组的元素
fiveInts(0)
fiveToOne(4)
fiveInts(0) = fiveToOne(4)
println(fiveInts)
```


## List buffer (List 缓冲)

- List类提供对List头部的快速访问，对尾部访问则没有那么高效，因此，当需要往List尾部追加元素来构建List时，通常要考虑反过来往头部追加元素，追加完成后，再调用 `reverse` 来获得想要的顺序；
- 另一种可选方案是 `ListBuffer` ，ListBuffer是一个可变对象，在需要追加元素来构建List时可以更高效。
- ListBuffer提供了常量时间的往后追加和往前追加的操作，可以用 `+=` 操作符来往后追加元素，用 `+=:` 来往前追加元素，完成构建后再调用ListBuffer的 `toList` 来获取最终的List；
- ListBuffer可以防止出现栈溢出；

```scala
import scala.collection.mutable.ListBuffer

// 创建一个类型为Int的ListBuffer
val buf = new ListBuffer[Int]

// 往List后追加元素
buf += 1
buf += 2
println(buf)

// 往List前追加元素
3 +=: buf

buf.toList
```


## Arrray buffer (Arrray 缓冲)

- ArrayBuffer 跟数组很像，除了可以额外从序列头部或尾部添加或移除元素；
- 所有的 Array 操作在
    ArrayBuffer都可用，不过由于实现的包装，会稍慢一些；
- `+=`: 往 Array 后面追加元素；

```scala
import scala.collection.mutable.ArrayBuffer

// 创建一个类型为Int的ArrayBuffer
val buf = new ArrayBuffer[Int]()

// 往Array后追加元素
buf += 12
buf += 15
println(buf)

// Array操作
buf.length
buf(0)
```


## StringOps


由于 `Predef` 有一个从 String 到 StringOps 的隐式转换，可以将任何字符串当做序列来处理；

```scala
// 在下面的函数里，对字符串调用了其本身没有的方法`exists`
// Scala编译器会隐式地将 s 转换成 StringOps，StringOps 有 exists 方法，exists 方法将字符串当做字符的序列
def hasUpperCase(s: String) = {
s.exists(_.isUpper)
}

hasUpperCase("Wang Zhefeng")
hasUpperCase("e e cumming")
```


# 集合 Set




# 映射 Map




# 元组 Tuple




# 集合操作




## 遍历




## map 操作 和 flatMap 操作




## filter 操作




## reduce 操作




## fold 操作



