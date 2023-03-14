---
title: Java 接口、lambda 表达式、内部类
author: 王哲峰
date: '2020-10-01'
slug: java-interface
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

- [Java 接口与类](#java-接口与类)
- [Java 接口](#java-接口)
  - [接口的声明](#接口的声明)
  - [接口的实现](#接口的实现)
  - [接口的继承](#接口的继承)
  - [接口的多继承](#接口的多继承)
  - [标记接口](#标记接口)
  - [接口示例](#接口示例)
    - [接口与回调](#接口与回调)
    - [Comparator 接口](#comparator-接口)
    - [对象克隆](#对象克隆)
- [lambda 表达式](#lambda-表达式)
- [内部类](#内部类)
</p></details><p></p>


- 接口(interface)
    - 主要用来描述类具有什么功能，而并不给出每个功能的具体实现
    - 一个类(class)可以实现(implement)一个或多个接口(interface), 并在需要接口的地方，随时使用实现了相应接口的对象
- lambda 表达式
    - lambda 表达式是一种可以在将来某个时间点执行的代码块的简洁方法
    - 使用 lambda 表达式，可以用一种精巧而简洁的方式表示使用回调或变量行为的代码
- 内部类(inner class)
    - 理论上讲，内部类有些浮在，内部类定义在另外一个类的内部，其中的方法可以访问包含它们的外部类的域
    - 内部类主要用于设计具有相互协作关系的类集合
- 代理(proxy)
    - 代理(proxy)是一种实现任意接口的对象
    - 代理是一种非常专业的构造工具，它可以用来构建系统级的工具

# Java 接口与类


# Java 接口

- 接口(interface): 在 Java 程序设计语言中，接口不是类，而是对类的一组需求描述，这些类要遵从接口描述的统一格式进行定义.
    - **接口方法**: 接口中的所有方法自动地属于 `public`，因此，在接口中声明方法时，不必提供关键字 `public`
    - **实例域**: 接口决不能含有 **实例域**
    - **接口方法实现**: 在 Java SE 8 之前不能在接口中实现 **方法**，现在可以在接口中提供简单方法了
    - **接口实现类**: 提供实例域和方法实现的任务应该由实现接口的那个类来完成，因此可以将接口看成是 **没有实例域的抽象类**
- 为了让类实现一个接口，通常需要下面两个步骤:
    - (1)将类声明为实现给定的接口
        - 要将类声明为实现某个接口，需要使用关键字 `implements`
    - (2)对接口中的所有方法进行定义
- 接口有以下特性：
    - **接口是隐式抽象的**
        - 当声明一个接口的时候，不必使用 `abstract` 关键字
    - **接口的每一个方法也是隐式抽象的**
        - 声明时同样不需要 `abstract` 关键字
    - **接口中的方法都是共有的(public)**

- 示例
    - `Arrays` 类中的 `sort` 方法承诺可以对对象数组进行排序，
      但要求满足前提：对象所属的类必须实现了 `Comparable` 接口，
      任何实现 `Comparable` 接口的类都需要包含 `compareTo` 
      方法，并且这个方法的参数必须是一个 Object 对象，返回一个整型数值.
    - `Comparable` 接口代码
        
    ```java
    public interface Comparable<T> {
        int compareTo(T other);
    }
    ```

    - 假设希望使用 Arrays 类的 sort 方法对 Employee 对象数组进行排序，
      Employee 类就必须实现 Compareable 接口

    ```java
    class Employee implement Comparable<Employee> {
        public int compareTo(Employee other) {
            Employee other = (Employee) otherObject;
            return Double.compare(salary, other.salary);
        }
    }
    ```


> Java 程序设计语言是一种强类型(strongly typed)语言。在调用方法的时候，编译器将会检查这个方法是否存在


- java.lang.Comparable<T> 1.0
    - int compareTo(T other)
- java.util.Arrays 1.2
    - static void sort(Object[] a)

## 接口的声明

接口语法

- `interface` 关键字用来声明一个接口. 接口的声明语法格式如下：

```java
[可见度] interface 接口名称 [extends 其他的接口名] {
    // 声明变量

    // 抽象方法

}
```

接口示例

- 接口的声明示例 1：

```java
/* file name: NameOfInterface.java */

// 引入包
import java.lang.*;

public interface NameOfInterface {
    // 任何类型 final, static 字段
    // 抽象方法
}
```

- 接口的声明示例 2：

```java
/* file name: Animal.java */

interface Animal {

    public void eta();

    public void travel();
}
```

## 接口的实现

当类实现接口的时候，类要实现接口中所有的方法。否则，类必须声明为抽象的类。

类使用 `implements` 关键字实现接口。在类声明中，`implements` 关键字放在 `class` 声明后面。
实现一个接口的语法，可以使用这个公式：

```java
class NameOfClass implements NameOfInterface, [NameOfInterface2, ...] {
    // 
    // 
}
```

重写接口中声明的方法时，需要注意一下规则：

- 类在实现接口的方法时，不能抛出强制性异常，只能在接口中，或者继承接口的抽象类中抛出该强制性异常。
- 类在重写方法时要保持一致的方法名，并且应该保持相同或者相兼容的返回值类型。
- 如果实现接口的类是抽象类，那么就没必要实现该接口的方法。

在实现接口的时候，也要注意一些规则：

- 一个类可以同时实现多个接口。
- 一个类只能继承一个类，但是能实现多个接口。
- 一个接口能继承另一个接口，这和类之间的继承比较相似。

接口实现示例：

```java
/* filename: MammalInt.java */

public class MammalInt implements Animal {
    
    public void eat() {
        System.out.println("Mammal eats");
    }
    public void travel() {
        System.out.println("Mammal travels");
    }

    public int noOfLegs() {
        return 0;
    }

    public static void main(String args[]) {
        MammalInt m = new MammalInt();
        m.eat();
        m.travel();
    }
}
```

## 接口的继承

一个接口能继承另一个接口，和类之间的继承方式比较相似。接口的继承使用extends关键字，
子接口继承父接口的方法。

接口继承示例:

```java
/* Sports.java */

public interface Sports {
    public void setHomeTeam(String name);
    public void setVisitingTeam(String name);
}
```

```java
/* Football.java */

public interface Football extends Sports {
    public void homeTeamScored(int points);
    public void visitingTeamScored(int points);
    public void endOfQuarter(int quarter);
}
```

```java
/* Hockey.java */

public interface Hockey extends Sports {
    public void homeGoalScored();
    public void visitingTeamScored();
    public void endOfPeriod(int period);
    public void overtimePeriod(int ot);
}
```

## 接口的多继承

在 Java 中，类的多继承是不合法，但接口允许多继承。

在接口的多继承中 `extends` 关键字只需要使用一次，在其后跟着继承接口。如下所示：

```java
public interface Hockey extends Sports, Event
```

## 标记接口

最常用的继承接口是没有包含任何方法的接口. 标记接口是没有任何方法和属性的接口. 
它仅仅表明它的类属于一个特定的类型, 供其他代码来测试允许做一些事情。

标记接口作用：简单形象的说就是给某个对象打个标（盖个戳），使对象拥有某个或某些特权。

没有任何方法的接口被称为标记接口。标记接口主要用于以下两种目的：

- 建立一个公共的父接口：
    - 正如 `EventListener` 接口，这是由几十个其他接口扩展的 Java API，
      你可以使用一个标记接口来建立一组接口的父接口。例如：当一个接口继承了 
      `EventListener` 接口，Java 虚拟机(JVM)就知道该接口将要被用于一个事件的代理方案。
- 向一个类添加数据类型：
    - 这种情况是标记接口最初的目的，实现标记接口的类不需要定义任何接口方法
      (因为标记接口根本就没有方法)，但是该类通过多态性变成一个接口类型。

标记接口示例：

- `java.awt.event` 包中的 `MouseListener` 接口继承的 `java.util.EventListener` 接口定义如下：

```java
package java.util;

public interface EventListener {}
```

## 接口示例

### 接口与回调


### Comparator 接口


### 对象克隆

# lambda 表达式



# 内部类

内部类(inner class)是定义在另一个类中的类。

- 内部类存在的原因：
    - 内部类方法可以访问该类定义所在的作用域中的数据，包括私有的数据
    - 内部类可以对同一个包中的其他类隐藏起来
    - 当想要定义一个回调函数且不想编写大量代码时，使用匿名(anonymous)内部类比较便捷

