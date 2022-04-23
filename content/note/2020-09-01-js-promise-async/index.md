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



# 期约

> 期约(promise)是对尚不存在结果的一个替身
> 
> 期约(promise)、终局(eventual)、期许(future)、延迟(delay)、
> 迟付(deferred) 等术语指代的是同样的概念。所有这些概念描述的都是一种异步程序执行的机制

# 异步函数


