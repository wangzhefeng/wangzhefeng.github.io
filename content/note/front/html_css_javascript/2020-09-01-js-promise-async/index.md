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

- [异步编程](#异步编程)
  - [同步与异步](#同步与异步)
    - [同步行为](#同步行为)
    - [异步行为](#异步行为)
  - [异步编程模式](#异步编程模式)
    - [异步返回值](#异步返回值)
    - [失败处理](#失败处理)
    - [嵌套异步回调](#嵌套异步回调)
- [期约](#期约)
- [异步函数](#异步函数)
</p></details><p></p>


# 异步编程

> 同步行为和异步行为的对立统一是计算机科学的一个基本概念。
> 特别是在 JavaScript 这种单线程事件循环模型中，同步操作与异步操作更是代码所要依赖的核心机制
> 
> 异步行为是为了优化因计算量大而时间长的操作。如果在等待其他操作完成的同时，即使运行其他指令，
> 系统也能保持稳定，那么这样做 就是务实的
> 
> 重要的是，异步操作并不一定计算量大或要等很长时间。只要你不想为等待某个异步操作而阻塞线程执行，
> 那么任何时候都可以使用

## 同步与异步

### 同步行为

同步行为对应内存中顺序执行的处理器指令。每条指令都会严格按照它们出现的顺序来执行，
而每条指令执行后也能立即获得存储在系统本地(如寄存器或系统内存)的信息。
这样的执行流程容易分析程序在执行到代码任意位置时的状态(比如变量的值)

### 异步行为

异步行为类似于系统中断，即当前进程外部的实体可以触发代码执行。
异步操作经常是必要的，因为强制进程等待一个长时间的操作通常是不可行的(同步操作则必须要等)。
如果代码要访问一些高延迟的资源，比如向服务器发送请求并等待相应，那么就会出现长时间的等待

## 异步编程模式

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

# 异步函数


