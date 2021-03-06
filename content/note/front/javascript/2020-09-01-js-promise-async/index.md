---
title: 期约与异步函数
author: 王哲峰
date: '2020-09-01'
slug: js-promise-async
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

- [TODO](#todo)
- [概述](#概述)
- [异步编程](#异步编程)
  - [同步与异步](#同步与异步)
    - [同步行为](#同步行为)
    - [异步行为](#异步行为)
  - [以往的异步编程模式](#以往的异步编程模式)
    - [异步返回值](#异步返回值)
    - [失败处理](#失败处理)
    - [嵌套异步回调](#嵌套异步回调)
- [期约](#期约)
  - [Promise/A+ 规范](#promisea-规范)
  - [期约基础](#期约基础)
  - [期约的实例方法](#期约的实例方法)
  - [期约扩展](#期约扩展)
- [异步函数](#异步函数)
  - [异步函数](#异步函数-1)
  - [停止和回复执行](#停止和回复执行)
</p></details><p></p>

# TODO

- [ ] item 1
- [ ] item 2

# 概述

ECMAScript 6 及之后的几个版本逐步加大了对异步编程机制的支持，
提供了令人眼前一亮的新特性。ECMAScript 6 新增了正式的 Promise(期约)引用类型，
支持优雅地定义和组织异步逻辑。接下来几个版本增加了使用 `async` 和 `await` 关键字定义异步函数的机制

长期以来，掌握单线程 JavaScript 运行时的异步行为一直都是个艰巨的任务。
随着 ES6 新增了期约和 ES8 新增了异步函数，ECMAScript 的异步编程特性有了长足的进步。
通过期约和 `async/await`，不仅可以实现之前难以实现或不可能实现的任务，
而且也能写出更清晰、简洁，并且容易理解、调试的代码

期约的主要功能是为异步代码提供了清晰的抽象。可以用期约表示异步执行的代码块，
也可以用期约表示异步计算的值。在需要串行异步代码时，期约的价值最为突出。
作为可塑性极强的一种结构，期约可以被序列化、连锁使用、复合、扩展和重组

异步函数是将期约应用于 JavaScript 函数的结果。异步函数可以暂停执行，而不阻塞主线程。
无论是编写基于期约的代码，还是组织串行或平行执行的异步代码，使用异步函数都非常得心应手。
异步函数可以说是现代 JavaScript 工具箱中最重要的工具之一

# 异步编程

同步行为和异步行为的对立统一是计算机科学的一个基本概念。
特别是在 JavaScript 这种单线程事件循环模型中，同步操作与异步操作更是代码所要依赖的核心机制
 
异步行为是为了优化因计算量大而时间长的操作。如果在等待其他操作完成的同时，即使运行其他指令，
系统也能保持稳定，那么这样做 就是务实的

重要的是，异步操作并不一定计算量大或要等很长时间。只要你不想为等待某个异步操作而阻塞线程执行，
那么任何时候都可以使用

## 同步与异步

### 同步行为

同步行为对应内存中顺序执行的处理器指令。每条指令都会严格按照它们出现的顺序来执行，
而每条指令执行后也能立即获得存储在系统本地(如寄存器或系统内存)的信息。
这样的执行流程容易分析程序在执行到代码任意位置时的状态(比如变量的值)

* 示例: 执行一次简单的数学计算

```js
let x = 3;
x = x + 4;
```

在程序执行的每一步，都可以推断出程序的状态。这是因为后面的指令总是在前面的指令完成后才会执行。
等到最后一条指定执行完毕，存储在 `$x$` 的值就立即可以使用。

这两行 JavaScript 代码对应的低级指令(从 JavaScript 到 x86)并不难想象:

* 首先，操作系统会在栈内存上分配一个存储浮点数值的空间
* 然后针对这个值做一次数学计算
* 再把计算结果写回之前分配的内存中

所有这些指令都是在单个线程中按顺序执行的。在低级指令的层面，有充足的工具可以确定系统状态


### 异步行为

异步行为类似于系统中断，即当前进程外部的实体可以触发代码执行。
异步操作经常是必要的，因为强制进程等待一个长时间的操作通常是不可行的(同步操作则必须要等)。
如果代码要访问一些高延迟的资源，比如向服务器发送请求并等待相应，那么就会出现长时间的等待

* 示例: 在定时回调中执行一次简答的数学计算

```js
let x = 3;
setTimeout(() => x = x + 4, 1000);
```

这段程序最终与同步代码执行的任务一样，都是把两个数加在一起，
但这一次执行线程不知道 `$x$` 值 5 何时会改变，因为这取决于回调何时从消息队列出列并执行

异步代码不容易推断。虽然这个例子对应的低级代码最终跟前面的例子没什么区别，
但第二个指令块(加操作及赋值操作)是由系统计时器触发的，这会生成一个入队执行的中断。
到底什么时候会触发这个中断，这对 JavaScript 运行时来说是一个黑盒，
因此实际上无法预知(尽管可以保证这发生在当前线程的同步代码执行之后，否则回调都没有机会出列被执行)。
无论如何，在排定回调以后基本没办法知道系统状态何时变化

为了让后续代码能够使用 `$x$`，异步执行的函数需要在更新 `$x$` 的值以后通知其他代码。
如果程序不需要这个值，那么就只管继续执行，不必等待这个结果了

设计一个能够知道 `$x$` 什么时候可以读取的系统是非常难的。
JavaScript 在实现这样一个系统的过程 中也经历了几次迭代

## 以往的异步编程模式

下面是一个异步函数，这里的代码没什么神秘的，但关键是理解为什么说它是一个异步函数。
setTimeout 可以定义一个在指定时间之后会被调度执行的回调函数。

在这个例子中，1000毫秒(1秒)之后，JavaScript

```js
function double(value) {
  setTimeout(() => setTimeout(console.log, 0, value * 2), 1000);
}
```

### 异步返回值


### 失败处理


### 嵌套异步回调



# 期约

> 期约(promise)是对尚不存在结果的一个替身
> 
> 期约(promise)、终局(eventual)、期许(future)、延迟(delay)、
> 迟付(deferred) 等术语指代的是同样的概念。所有这些概念描述的都是一种异步程序执行的机制

## Promise/A+ 规范

## 期约基础

## 期约的实例方法

## 期约扩展











# 异步函数


## 异步函数

## 停止和回复执行

