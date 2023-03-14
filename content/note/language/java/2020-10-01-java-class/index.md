---
title: Java 类库
author: 王哲峰
date: '2020-10-01'
slug: java-class
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

- [Java 类库中的 LocalDate 类](#java-类库中的-localdate-类)
</p></details><p></p>

# Java 类库中的 LocalDate 类

时间是用距离一个固定的时间点的毫秒数(可正可负)表示的，这个点就是所谓的纪元(epoch)，
它是 UTC 时间 1970 年 1 月 1 日 00:00:00。

UTC 是 Coordinated Universal Time 的缩写，与大家熟悉的 GMT(即 Greenwith Mean Time，
格林威治时间一样)，是一种具有实践意义的科学标准时间。

类库设计者决定将保存时间与给时间点命名分开，所以标准 Java 类库分别包含了两个类：

- 一个是用来表示时间点的 `Date` 类；
- 另一个是用来表示大家熟悉的日历表示法的 `LocalDate` 类；
- Java SE8 引入了另外一些类来处理日期和时间的不同方面

不要使用构造器来构造 LocalDate 类的对象。实际上，应当使用静态工厂方法(factory method)代表你调用构造器。

- 示例：

```java
LocalDate newYearsEve = LocalDate.of(1999, 12, 31);
int year = newYearsEve.getYear();
int month = newYearsEve.getMonthValue();
int day = newYearsEve.getDayOfMonth();

LocalDate aThousandDaysLater = newYearsEve.plusDays(1000);

year = aThousandDaysLater.getYear();
month = aThousandDaysLater.getMonthValue();
day = aThousandDaysLater.getDayOfMonth();
```

- java.time.LocalDate 8
    - static LocalTime now()
    - static LocalTime of(int year, int month, int day)
    - int getYear()
    - int getMonthValue()
    - int getDayOfMonth()
    - DayOfWeek getDayOfWeek
    - LocalDate plusDays(int n)
    - LocalDate minusDays(int n)

