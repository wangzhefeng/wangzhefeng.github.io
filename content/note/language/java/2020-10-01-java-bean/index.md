---
title: Java Bean
author: 王哲峰
date: '2020-10-01'
slug: java-bean
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

- [JavaBean 介绍](#javabean-介绍)
- [JavaBean 的作用](#javabean-的作用)
- [枚举 JavaBean 属性](#枚举-javabean-属性)
</p></details><p></p>

JavaBean 是一种负荷命名规范的 `class`，它通过 `getter` 和 `setter` 来定义属性

- 属性一种通用的叫法，并非 Java 语法规定
- 可以利用 IDE 快速生成 `getter` 和 `setter`
- 使用 `Introspector.getBeanInfo()` 可以获取属性列表

# JavaBean 介绍

在 Java 中，有很多 class 的定义都符合这样的规范:

- 若干 `private` 实例字段
- 通过 `public` 方法来读写实例字段

如果类方法的读写方法符合以下命名规范, 那么这种 `class` 就称为 `JavaBean`:

```java
class Test {
    // 读方法
    public Type getXyz()
    // 写方法
    public void setXyz(Type value)

    // boolean 字段读方法
    public boolean isXyz()
    // boolean 字段写方法
    public void setXyz(boolean value)
}
```

通常把一组对应的方法 `getter` 和写方法 `setter` 称为属性 `property`

- 只有 `getter` 的属性称为只读属性(read-only)
- 只有 `setter` 的属性称为只写属性(write-only)
- 只读属性很常见，只写属性不常见
- 属性只需要定义 `getter` 和 `setter` 方法，不一定需要对应的字段
- `getter` 和 `setter` 也是一种数据封装的方法


示例：

```java
public class Person {
    private String name;
    private int age;

    public String getName() {
        return this.name;
    }

    public void setName(String name) {
        this.name = name;
    }

    public int getAge() {
        return this.age;
    }

    public void setAge(int age) {
        this.age = age;
    }

    // 没有字段
    public boolean isChild() {
        return age <= 6;
    }
}
```

# JavaBean 的作用

JavaBean 主要用来传递数据，即把一组数据组合成一个 JavaBean 便于传输。此外，JavaBean 可以方便地被 IDE 工具分析，生成读写属性的代码，
主要用在图形界面的可视化设计中。通过IDE，可以快速生成 `getter` 和 `setter`。

# 枚举 JavaBean 属性

要枚举一个 JavaBean 的所有属性，可以直接使用 Java 核心库提供的 `Introspector`

```java
import java.beans.*;

public class Main {
    public statics void main(String[] args) throws Exception {
        BeanInfo info = Introspector.getBeanInfo(Person.class);
        for (PropertyDescriptor pd: info.getPropertyDescriptions()) {
            System.out.println(pd.getName());
            System.out.println("  " + pd.getReadMethod());
            System.out.println("  " + od.getWriteMethod());
        }
    }
}

class Person {
    private String name;
    private int age;

    public String getName() {
        return name;
    }

    public void setName(String name) {
        this.name = name;
    }

    public int getAge() {
        return age;
    }

    public void setAge(int age) {
        this.age = age;
    } 
  }
```

