---
title: 代理与反射
author: 王哲峰
date: '2020-07-01'
slug: js-proxy
categories:
  - 前端
tags:
  - tool
---

<style>
h1 {
  background-color: #2B90B6;
  background-image: linear-gradient(45deg, #4EC5D4 10%, #146b8c 20%);
  background-size: 100%;
  -webkit-background-clip: text;
  -moz-background-clip: text;
  -webkit-text-fill-color: transparent;
  -moz-text-fill-color: transparent;
}
h2 {
  background-color: #2B90B6;
  background-image: linear-gradient(45deg, #4EC5D4 10%, #146b8c 20%);
  background-size: 100%;
  -webkit-background-clip: text;
  -moz-background-clip: text;
  -webkit-text-fill-color: transparent;
  -moz-text-fill-color: transparent;
}

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

- [1.代理基础](#1代理基础)
  - [1.1 创建空代理](#11-创建空代理)
  - [1.2 定义捕获器](#12-定义捕获器)
  - [1.3 捕获器参数和反射 API](#13-捕获器参数和反射-api)
  - [1.4 捕获器不变式](#14-捕获器不变式)
  - [1.5 可撤销代理](#15-可撤销代理)
  - [1.6 实用反射 API](#16-实用反射-api)
  - [1.7 代理另一个代理](#17-代理另一个代理)
  - [1.8 代理的问题与不足](#18-代理的问题与不足)
- [2.代理捕获器与反射方法](#2代理捕获器与反射方法)
- [3.代理模式](#3代理模式)
</p></details><p></p>


- ECMAScript 6 新增的代理和反射为开发者提供了 **拦截并向基本操作嵌入额外行为的能力**
- 具体地说，可以给**目标对象**定义一个关联的**代理对象**，而这个代理对象可以作为抽象的目标对象来使用
- 在对目标对象的各种操作影响目标对象之前，可以在代理对象中对这些操作加以控制

## 1.代理基础

- 代理是目标对象的抽象。从很多方面看，代理类似 C++指针，因为它可以 用作目标对象的替身，但又完全独立于目标对象。目标对象既可以直接被操作，也可以通过代理来操作。 但直接操作会绕过代理施予的行为

### 1.1 创建空代理

- 最简单的代理是空代理，即除了作为一个抽象的目标对象，什么也不做

- 默认情况下，代理对象上执行的所有操作都会无障碍地传播到目标对象，因此，在任何可以使用目标对象的地方，都可以通过同样的方式来使用与之关联的代理对象

- 代理是使用 `Proxy` 构造函数创建的，这个构造函数接收两个参数：

	- 目标对象
	- 处理程序对象

	> - 缺少其中任何一个参数都会抛出 `TypeError`
	>
	> - 要创建空代理，可以传入一个简单的对象字面量作为处理程序对象，从而让所有操作畅通无阻地抵达目标对象

- 示例

```js
// 目标对象
const target = {
    id: "target"
};

// 处理程序对象
const handler = {};

// 创建空代理
const proxy = new Proxy(target, handler);

// id 属性会访问同一个值
console.log(target.id); // target
console.log(proxy.id); // target

// 给目标属性赋值会反映在两个对象上，因为两个对象访问的是同一个值
target.id = "foo";
console.log(target.id); // foo
console.log(proxy.id); // foo

// 给代理属性赋值会反映在两个对象上，因为这个赋值会转移到目标对象上
proxy.id = "bar";
console.log(target.id); // bar
console.log(proxy.id); // bar

// hasOwnProperty() 方法在两个地方都会应用到目标对象
console.log(target.hasOwnProperty("id")); // true
console.log(proxy.hasOwnProperty("id")); // true

// Proxy.prototype 是 undefined，因此不能使用 instanceof 操作符
console.log(target instanceof Proxy); // TypeError: Function has non-object prototype 9 'undefined' in instanceof check
console.log(proxy instanceof Proxy); // TypeError: Function has non-object prototype 9 'undefined' in instanceof check

// 严格相等可以用来区分代理和目标
console.log(target === proxy); // false
```

### 1.2 定义捕获器

### 1.3 捕获器参数和反射 API

### 1.4 捕获器不变式

### 1.5 可撤销代理

### 1.6 实用反射 API

### 1.7 代理另一个代理

### 1.8 代理的问题与不足



## 2.代理捕获器与反射方法

## 3.代理模式

