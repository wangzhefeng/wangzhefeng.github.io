---
title: 错误处理与调试
author: 王哲峰
date: '2020-12-02'
slug: js-error
categories:
  - javascript
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
img {
    pointer-events: none;
}
</style>

<details><summary>目录</summary><p>

- [TODO](#todo)
- [概览](#概览)
- [浏览器错误报告](#浏览器错误报告)
- [错误处理](#错误处理)
  - [try/catch 语句](#trycatch-语句)
    - [try/catch 语法](#trycatch-语法)
    - [finally 子句](#finally-子句)
    - [错误类型](#错误类型)
    - [try/catch 用法](#trycatch-用法)
  - [抛出错误](#抛出错误)
  - [error 事件](#error-事件)
  - [错误处理策略](#错误处理策略)
  - [区分重大与非重大错误](#区分重大与非重大错误)
  - [把错误记录到服务器中](#把错误记录到服务器中)
- [调试技术](#调试技术)
- [旧版 IE 的常见错误](#旧版-ie-的常见错误)
</p></details><p></p>

# TODO

- [ ] item 1
- [ ] item 2

# 概览



# 浏览器错误报告


# 错误处理

## try/catch 语句


### try/catch 语法

* EMCA-262 第 3 版新增了 `try/catch` 语句，作为在 JavaScript 中处理异常的一种方式
* 任何可能出错的代码都应该放到 `try` 块中，而处理错误的代码则放在 `catch` 块中
* 即使在 `catch` 块中不使用错误对象，也必须为它定义名称
* 错误对象中暴露的实际信息因浏览器而异，但至少包含保存错误消息的 `message` 属性。
  `message` 属性是唯一一个在 IE、Firefox、Safari、Chrome 和 Opera 中都有的属性，
  尽管每个浏览器添加了其他属性。为了保证跨浏览器兼容，最好只依赖 `message` 属性
    - IE 添加了 `description` 属性(其值始终等于 `message`) 和 `number` 属性(它包含内部错误号)
    - Firefox 添加了 `filename`、`lineNumber` 和 `stack`(包含栈跟踪信息)属性
    - Safari 添加了 `line`(行号)、`sourceId`(内部错误号)、`sourceURL` 属性
* EMCA-262 也制定了定义错误类型的 `name` 属性，目前所有浏览器中都有这个属性

* 语法

```js
try {
    // 可能出错的代码
} catch (error) {
    // 出错时要做什么
}
```

* 示例

```js
try {
    window.someNoneexistentFunction();
} catch (error) {
    console.log(error.message);
}
```

### finally 子句

`try/catch` 语句中可选的 `finally` 子句始终运行。如果 `try` 块中的代码运行完，
则接着执行 `finally` 块中的代码。如果出错并执行 `catch` 块中的代码，
则 `finally` 块中的代码仍执行。`try` 或 `catch` 块无法阻止 `finally` 块执行，
包括 `return` 语句

* 示例

```js
function testFinally() {
    try {
        return 2;
    } catch (error) {
        return 1;
    } finally {
        return 0;
    }
}
```

### 错误类型

代码执行过程中会发生各种类型的错误，每种类型都会对应一个错误发生时抛出的错误对象。
ECMA-262 定义了以下 8 种错误类型:

* `Error`
    - `Error` 是基类型，其他错误类型继承该类型
    - 所有错误类型都共享相同的属性(所有错误)
* `InternalError`
    - todo
* `EvalError`
    - todo
* `RangeError`
    - todo
* `ReferenceError`
    - todo
* `SyntaxError`
    - todo
* `TypeError`
    - todo
* `URLError`
    - todo


### try/catch 用法

当 `try/catch` 中发生错误时，浏览器会认为错误被处理了，因此就不会再使用本章前面提到的机制报告错误。
如果应用程序的用户不懂技术，那么他们即使看到错误也看不懂，这是一个理想的结果。
使用 `try/catch` 可以针对特定错误类型实现自定义的错误处理

* `try/catch` 语句最好用在自己无法控制的错误上
    - 例如，假设你的代码中使用了一个大型 JavaScript 库的某个函数，
      而该函数可能会有意或由于出错而抛出错误。因为不能修改这个库的代码，所以为防止这个函数报告错误，
      就有必要通过 `try/catch` 语句把该函数调用包装起来，对可能的错误进行处理
* 如果你明确知道自己的代码会发生某种错误，那么就不适合使用 `try/catch` 语句
    - 例如，如果给函数传入字符串而不是数值时就会失败，就应该检查该函数的参数类型并采取相应的操作。
      这种情况下， 没有必要使用 `try/catch` 语句

## 抛出错误


## error 事件


## 错误处理策略

## 区分重大与非重大错误


## 把错误记录到服务器中



# 调试技术


# 旧版 IE 的常见错误