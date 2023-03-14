---
title: Java 集合
author: 王哲峰
date: '2020-10-01'
slug: java-collection
categories:
  - java
tags:
  - tool
---

在 Java 中，如果一个 Java 对象可以在内部持有若干其他 Java 对象，并对外提供访问接口，
我们把这种 Java 对象称为集合

既然Java提供了数组这种数据类型，可以充当集合，那么，为什么还需要其他集合类？这是因为数组有如下限制：

- 数组初始化后大小不可变；
- 数组只能按索引顺序存取

需要各种不同类型的集合类来处理不同的数据，例如：

- 可变大小的顺序链表；
- 保证无重复元素的集合；
- 等等

# Collection

Java 标准库自带的 `java.util` 包提供了集合类：`Collection`，
它是除 `Map` 外所有其他集合类的根接口。

```java
import java.util.Collection;
```

- Java 的 `java.util` 包主要提供了三种类型集合:
    - `List`：一种有序列表的集合
    - `Set`：一种保证没有重复元素的集合
    - `Map`：一种通过(key-value)查找的映射表集合
- Java 集合的设计有几个特点：
    - 一是实现了接口和实现类相分离，例如，有序表的接口是 `List`，具体的实现类有 `ArrayList`，`LinkedList` 等
    - 二是支持泛型，我们可以限制在一个集合中只能放入同一种数据类型的元素
    - 最后，Java 访问集合总是通过统一的方式——迭代器(Iterator)来实现，它最明显的好处在于无需知道集合内部元素是按什么方式存储的
- 示例:

```java
import java.util.List;
import java.util.Set;
import java.util.Map;

List<String> list = new ArrayList<>(); // 只能放入 String 类型
```

# 数组类型

## 数组介绍

Java 数组是一种数据结构，用来存储同一类型值的集合。通过一个整型下标可以访问数组中的每一个值。

- **数组索引**
    - Java 可以通过整数索引访问数组元素，索引超出范围将报错
- **创建数组**
    - **数组声明**
        - 在声明数组变量时，需要指出数组的类型，数据元素类型紧跟 `[]` 和数组变量的名字
            - `int[] a;` 或 `int a[];`
            - `float[] a;`
            - `String[] a;`
            - `boolean[] a;`
    - **数组初始化**
        - 数组初始化时需要使用 `new` 运算符创建数组，数组长度不要求是常量
            - `int[] a = new int[100];`
            - `String[] names = new String[10];`
        - 创建一个数字数组时，所有元素都初始化为 `0`
        - 创建一个 boolean 数组时，所有元素都初始化为 `false`
        - 创建一个对象数组时，所有元素都初始化为一个特殊的 `null`，表示这些元素还未存放任何对象
    - **数组元素赋值**
        - 一旦创建了数组，就可以给数组元素赋值

```java
int[] a = new int[100];
for (int i = 0; i < 100; i++) {
    // fills the array with numbers 0 to 99
    a[i] = i;
}
```

- **数组长度**
    - 可以通过 `.length` 获取数组中元素的个数

```java
int[] a = new int[100];
for (int i = 0; i < a.length; i++) {
    System.out.println(a[i]);
}
```

- **数组大小不可变**
    - 数组一旦创建，就不能再改变它的大小，但可以改变每一个数组的元素
    - 如果经常需要在运行过程中扩展数组的大小，就应该使用另一种数据结构 —— `数组列表(array list)`

## for each 循环

Java 有一种功能很强的循环结构，可以用来依次处理数组中的每个元素(其他类型的元素集合亦可)而不必为指定下标值而分心。

- for each 语法

```java
for (variable : collection) {
    statement        
}
```

- for each 结构解析
    - 定义一个变量 `variable` 用于暂存集合中的每一个元素，并执行相应的语句或语句块
    - `collection` 这一集合表达式必须是一个数组或者是一个实现了 `Iterable` 接口的类对象


> 有个更加简单的方式打印数组中的所有值，即利用 `Arrays` 类的 `toString` 方法：
> 
> - 调用 `Arrays.toString(a)`，返回一个包含数组元素的字符串，这些元素被放置在方括号内 `[]`，并用逗号分隔

## 数组初始化以及匿名数组

在 Java 中，提供了一种创建数组对象并同时赋予初始值的简化书写形式。
这种表示法将创建一个新数组并利用括号中提供的值进行初始化，数组的大小就是初始值的个数。
使用这种语法形式可以在不创建新变量的情况下重新初始化一个数组。

- 示例

```java
// 创建数组的标准形式
int[] smallPrimes = new int[6];
smallPrimes[0] = 2;
smallPrimes[1] = 3;
smallPrimes[2] = 5;
smallPrimes[3] = 7;
smallPrimes[4] = 11;
smallPrimes[5] = 13;

// 创建数组的简化书写形式
int[] smallPrimes = { 2, 3, 5, 7, 11, 13 };

// 创建匿名数组
new int[] { 17, 19, 23, 29, 31, 37 };

// 重新初始化数组 smallPrimes
smallPrimes = new int[] { 17, 19, 23, 29, 31, 37 };
// 简写形式
int[] anonymous = { 17, 19, 23, 29, 31, 37 };
smallPrimes = anonymous;
```


> 在 Java 中，允许数组长度为 0. 在编写一个结果为数组的方法时，如果碰巧结果为空，
> 则这种语法形式就显得非常有用。此时可以创建一个长度为 0 的数组:
> 
> ```java
> elementType[] empty = new elementType[0];
> ```
> 数组长度为 0 与 null 不同.

## 数组拷贝

在 Java 中，允许使用一个数组变量拷贝给另一个数组变量，这时，两个变量将引用同一个数组:

```java
int[] smallPrimes = { 2, 3, 5, 7, 11, 13 };
int[] luckNumbers = smallPrimes;
luckNumbers[5] = 12;              // now smallPrimes[5] is also 12
```

- 如果希望将一个数组的所有值拷贝到一个新的数组中去，就要使用 `Arrays` 类的 `copyOf` 方法:

```java
int[] copiedLuckyNumbers = Arrays.copyOf(luckNumbers, luckNumbers.length);
```

- `Arrays.copyOf()` 第 2 个参数是新数组的长度，这个方法通常用来增加数组的大小
    - 如果数组元素是数值型，那么多余的元素将被赋值为 `0`
    - 如果数组元素是布尔型，则将赋值为 `false`
    - 相反，如果长度小于原始数组的长度，则只拷贝前面的数据元素

```java
luckNumbers = Arrays.copyOf(luckNumbers, 2 * luckNumbers.length);
```

## 命令行参数

每一个 Java 应用程序都有一个带 `String arg[]` 参数的 `main` 方法。
这个参数表明 `main` 方法将接收一个字符串数组，也就是命令行参数。

示例：

- Java 程序

```java
public class Message {
    public static void main(String[] args) {
        if (args.length == 0 || args[0].equals("-h")) {
            System.out.print("Hello, ");
        } else if (args[0].equals("-g")) {
            System.out.print("Goodbye, ");
        }

        // print the other command-line arguments
        for (int i = 1; i < args.length; i++) {
            System.out.print(" ".args[i]);
        }

        System.out.println("!");
    }
}
```

- 运行程序

```bash
$ java Message -g cruel world
```

## 数组排序

要想对数值型数组进行排序，可以使用 `Arrays` 类中的 `sort` 方法.

- 这个方法使用了优化的快速排序算法，快速排序算法对于大多数数据集合来说都是效率比较高的

示例：

```java
int[] a = new int[10000];
Arrays.sort(a);
```

- java.util.Arrays 1.2
    - `static String toString(type[] a)` 5.0
    - `static type copyOf(type[] a, int length)` 6
    - `static type copyOfRange(type[] a, int start, int end)` 6
    - `static void sort(type[] a)`
    - `static int binarySearch(type[] a, type v)`
    - `static int binarySearch(type[] a, int start, int end, type v)` 6
    - `static void fill(type[] a, type v)`
    - `static boolean equals(type[] a, type[] b)`

## 多维数组

多维数组将使用多个下标访问数组元素，它适用于表示表格或更加复杂的排列形式。

- 略

## 不规则数组

- 略

# List

## List 介绍

在集合类中，`List` 是最基础的一种集合：它是一种有序列表

- `List` 的行为和数组几乎完全相同：`List` 内部按照放入元素的先后顺序存放，每个元素都可以通过索引确定自己的位置，`List` 的索引和数组一样，从 0 开始
    - 在实际应用中，需要增删元素的有序列表，使用最多的是 `ArrayList`, 实际上 `ArrayList` 在内部使用了数组来存储所有元素
        - `ArrayList` 把添加和删除的操作封装起来，让操作 `List` 类似于操作数组，却不用关心内部元素如何移动
    - 实现 `List` 接口也可以通过 `LinkedList`, 即通过“链表”实现 `List` 接口，在 `LinkedList` 中，它的内部每个元素
      都指向下一个元素
    - `ArrayList`(优先使用) vs `LinkedList`

| Item          |         `ArrayList`|            `LinkedList`  |
|---------------|--------------------|--------------------------|
| 获取指定元素        |     速度很快           |    需要从头开始查找元素 |
| 添加元素到末尾       |    速度很快            |   速度很快 |
| 在指定位置添加和删除    |  需要移动元素            |不需要移动元素 |
| 内存占用          |      少             |       较大 |

## List<E> 接口

主要接口方法：

- 在末尾添加一个元素
    - `boolean add(E e)`
- 在指定索引添加一个元素
    - `boolean add(int index, E e)`
- 删除指定索引的元素
    - `int remove(int index)`
- 删除某个元素
    - `int remove(Object e)`
- 获取指定索引的元素
    - `E get(int index)`
- 获取链表的大小
    - `int size()`

## `List` 的特点

- `List` 接口的规范
- `List` 允许添加 `null`

示例：
        
```java
import java.util.ArrayList;
import java.util.List;

public class Main {
    public static void main(String[] args) {
        
        // 添加重复值
        List<String> list = new ArrayList<>();
        list.add("apple");
        list.add("pear");
        list.add("apple");
        System.out.println(list.size());

        // List 添加 null
        list.add(null);
        String second = list.get(1);
        System.out.println(second);
    }
}
```

## 创建 List

- `List<E> list = new ArrayList<>();`
- `List<E> list = new LinkedList<>();`
- 通过 `List` 接口提供的 `of()` 方法，根据给定元素快速创建 `List`
    - `List.of()` 方法不接受 `null` 值，如果传入 `null`，会抛出 `NullPointerException` 异常

```java
import java.util.List;

List<Integer> list = List.of(1, 2, 5);
```

## 遍历 List

- 和数组类型类似，要遍历一个 `List`，完全可以用 `for` 循环根据索引配合 `get(int)` 方法遍历

```java
import java.util.List;

public class Main {
    public static void main() {
        List<String> list = List.of("apple", "pear", "banana");
        for (int i=0; i<list.size(); i++) {
            String s = list.get(i);
            System.out.println(s);
        }
    }
}
```

# Set

# Map

## Map 介绍

- `Map` 也是一个接口，最常用的实现类是 `HashMap`


```java
import java.util.HashMap;
import java.util.Map;

public class Main {
    public static void main(String[] args) {
        
        Student s = new Student("Xiao Ming", 99);
        Map<String, Student> map = new HashMap<>();
        map.put("Xiao Ming", s)

        Student target = map.get("Xiao Ming")
        System.out.println(target == s);
        System.out.println(target.score);
        Student another = map.get("Bob");
        System.out.println(another);

    }
}

class Student {
    public String name;
    public int score;
    public Student(String name, int score) {
        this.name = name;
        this.score = score;
    }
}
```

# Map<,> 接口

- `V put(K key, V value)`
- `V get(K key)`
- `boolean containsKey(K key)`

## 遍历 Map

```java
import java.util.HashMap;
import java.util.Map;

public class Main {
    public static void main(String[] args) {
        Map<String, Integer> map = new HashMap<>();
        map.put("apple", 123);
        map.put("pear", 456);
        map.put("banana", 789);
        for (String key : map.keySet()) {
            Integer value = map.get(key);
            System.out.println(key + " =" + value);
        }
    }
}
```

