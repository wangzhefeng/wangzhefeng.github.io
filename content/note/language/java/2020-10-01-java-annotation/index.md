---
title: Java 注解
author: 王哲峰
date: '2020-10-01'
slug: java-annotation
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

- [使用注解](#使用注解)
  - [Java 的注解的三种类型](#java-的注解的三种类型)
  - [Java 注解参数配置](#java-注解参数配置)
- [定义注解](#定义注解)
  - [定义 Java 注解](#定义-java-注解)
  - [元注解](#元注解)
    - [@Target](#target)
    - [Retention](#retention)
    - [Repeatable](#repeatable)
    - [Inherited](#inherited)
- [处理注解](#处理注解)
</p></details><p></p>

Java 注解是放在 Java 源码的类、方法、字段、参数前的一种特殊 **注释**

- 注释会被编译器直接忽略
- 注解则可以被编译器打包进入 `.class` 文件
- 注解是一种用作标注的元数据, 是 Java 语言用于工具处理的标注
- 注解可以配置参数，没有指定配置的参数使用默认值，如果参数名称是 `value`，且只有一个参数，那么可以省略参数名称

```java
// this is a component:
@Rescource("hello")
public class Hello {
    @Inject
    int n;

    @PostConstruct
    public void hello(@Param String name) {
        System.out.println(name);
    }

    @Override
    public String toString() {
        return "Hello";
    }
}
```

# 使用注解

从 JVM 的角度看，注解本身对代码逻辑没有任何影响，如何使用注解完全由工具决定。

## Java 的注解的三种类型

- (1)由编译器使用的注解, 这类注解不会被编译进入 `.class` 文件，它们在编译后就被编译器扔掉了。例如:
    - `@Override`：让编译器检查该方法是否正确地实现了覆写
    - `@SuppressWarnings`: 告诉编译器忽略此处代码产生的警告
- (2)由工具处理 `.class` 文件使用的注解
    - 有些工具会在加载 class 的时候，对 class 做动态修改，实现一些特殊的功能。这类注解会被编译进 `.class` 文件，
      但加载结束后并不会存在于内存中。这类注解只被一些底层库使用，一般我们不必自己处理
- (3)在程序运行期能够读取的注解，它们在加载后一直存在于 JVM 中，这也是最常用的注解
    - 一个配置了 `@PostConstruct` 的方法会在调用构造方法后自动被调用(这是 Java 代码读取该注解实现的功能，
      JVM 并不会识别该注解)

## Java 注解参数配置

- 定义一个注解时，可以定义配置参数。配置参数可以包括：
    - 所有基本类型
    - String
    - 枚举类型
    - 基本类型、String、Class 以及枚举的数组
- 配置参数必须是常量，所以，上述限制保证了注解在定义时就已经确定了每个参数的值
- 注解的配置参数可以有默认值，缺少某个配置参数时将使用默认值。
    - 大部分注解会有一个名为`value` 的配置参数，对此参数赋值，可以只写常量，相当于省略了 `value` 参数
    - 如果只写注解，相当于全部使用默认值
- 示例：

```java
public class Hello {
    @Check(min = 0, max = 100, value = 55)
    public int n;

    @Check(value = 99)
    public int p;
}
```

# 定义注解

## 定义 Java 注解

Java 使用 `@interface` 语法来定义注解(Annotation)，格式如下：

```java
public @interface Report {
    int type() default 0;
    String level() default "info";
    String value() default "";
}
```


> 注解的参数类似无参数方法，可以用 `default` 设定一个默认值(强烈推荐)。最常用的参数应当命名为 `value`

定义 Java 注解的步骤：

- (1) 用 `@interface` 定义注解

```java
public @interface Report {
    ...
}
```
- (2) 添加参数、默认值

```java
public @interface Report {
    int type() default 0;
    String level() default "info";
    String value() default "";
}
```

- (3) 用元注解配置注解

```java
@Target(ElementType.TYPE)
@Retention(RetentionPolicy.RUNTIME)
public @interface Report {
    int type() default 0;
    String level() default "info";
    String value() default "";
}
```

> - 必须设置 `@Target` 和 `@Retention`
>     - `@Retention` 一般设置为 `RUNTIME`，因为我们自定义的注解通常要求在运行期读取
>     - 一般情况下，不必写 `@Inherited` 和 `@Repeatable`

## 元注解

- 修饰其他注解的注解就称为 **元注解(meta annotation)**
- Java 标准库已经定义了一些元注解，需要要使用元注解，通常不需要字节取编写元注解

### @Target

使用 `@Target` 可以定义 `Annotation` 能够被应用于源码的哪些位置：

- 类或接口：`ElementType.TYPE`
- 字段：`ElementType.FIELD`
- 方法：`ElementType.METHOD`
- 构造方法：`ElementType.CONSTRUCTOR`
- 方法参数：`ElementType.PARAMETER`

### Retention

`@Retention` 定义了 `Annotation` 的生命周期:

- 仅编译期：RetentionPolicy.SOURCE
- 仅 class 文件：RetentionPolicy.CLASS
- 运行期：RetentionPolicy.RUNTIME


### Repeatable

### Inherited

# 处理注解

