---
title: XML、JSON
author: 王哲峰
date: '2020-12-03'
slug: js-xml-json
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

- [XML](#xml)
  - [浏览器对 XML DOM 的支持](#浏览器对-xml-dom-的支持)
  - [在 JavaScript 中使用 XPath](#在-javascript-中使用-xpath)
  - [使用 XSLT 处理器](#使用-xslt-处理器)
- [JSON](#json)
  - [JSON 语法](#json-语法)
    - [简单值](#简单值)
    - [对象](#对象)
    - [数组](#数组)
  - [JSON 解析](#json-解析)
    - [JSON 对象](#json-对象)
    - [序列化选项](#序列化选项)
    - [解析选项](#解析选项)
  - [JSON 序列化](#json-序列化)
</p></details>

<p></p>

# XML

XML 曾一度是在互联网上存储和传输结构化数据的标准

## 浏览器对 XML DOM 的支持

## 在 JavaScript 中使用 XPath


## 使用 XSLT 处理器




# JSON

2006 年，Douglas Crockford 在国际互联网工程任务组(IETF, The Internet Engineering Task Force)
指定了 JavaScript 对象简谱(JSON, JavaScript Ojbect Notation)标准，即 RFC 4627。
但实际上，JSON 早在 2001 年 就开始使用了。

JSON 是 JavaScrip 的严格子集，利用 JavaScript 中的集中模式来表示结构化数据。
Crockford 将 JSON 作为替代 XML 的一个方案提出，
因为 JSON 可以直接传给 `eval()` 而不需要创建 DOM。

理解 JSON 最关键的一点是要把它当成是一种数据格式，
而不是编程语言。JSON 不属于 JavaScript， 
它们只是拥有相同的语法而已。JSON 也不是只能在 JavaScirpt 中使用，
它是一种通用数据格式。 很多语言都有解析和序列化 JSON 的内置能力。

## JSON 语法

JSON 语法支持表示 3 种类型的值:

* 简单值
    - 数值
    - 字符串: JSON 字符串必须使用双引号，单引号会导致语法错误
    - 布尔值
    - `null`
* 对象
    - 复杂数据类型
    - 对象表示有序键/值对，每个值可以是简单值，也可以是复杂类型
* 数组
    - 复杂数据类型
    - 数组表示可以通过数值索引访问的值的有序列表
    - 数组的值可以是任意类型，包括简单值、对象、甚至其他数组

### 简单值

| 类型   | 示例           | 说明           |
|--------|----------------|----------------|
| 数值   | 1              |                |
| 字符串 | "Hello world!" | 必须使用双引号 |
| 布尔值 | true, false    |                |
| null   | null           |                |

### 对象

JSON 对象与 JavaScript 对象字面量略为不同，与 JavaScript 对象字面量相比，JSON 主要有两处不同:

* JSON 没有变量，没有变量声明
* JSON 最后没有分号
* JSON 必须使用双引号把属性名包围起来

* JavaScript 字面量

```js
let person = {
    name: "Nicholas",
    age: 29
};

// or

let object = {
    "name": "Nicholas",
    "age": 29
};
```

* JSON 对象

```json
{
    "name": "Nicholas",
    "age": 29,
    "school": {
        "name": "Merrimack College",
        "location": "North Andover, MA"
    }
}
```

### 数组

数组在 JSON 中使用 JavaScript 的数组字面量形式表示

* JavaScript 数组字面量

```js
let values = ["25, "hi", true];
```

* 数组 JSON

```json
[25, "hi", "true"]
```

## JSON 解析

JSON 的迅速流行并不仅仅因为其语法与 JavaScript 类似，
很大程度上还因为 JSON 可以直接被解析成可用的 JavaScript 对象。
与解析为 DOM 的 XML 相比，这个优势非常明显。

### JSON 对象

* 早期的 JSON 解析器基本上相当于 JavaScript 的 `eval()` 函数，因为 JSON 是 JavaScript 语法的子集，
  所以 `eval()` 可以解析、解释，并将其作为 JavaScript 对象和数组返回
* ECMAScript5 增加了 JSON 全局对象，正式引入了解析 JSON 的能力，
  这个对象在所有主流浏览器中都得到了支持。旧版本的浏览器可以使用垫片脚本
  (参见 GitHub 上 douglascrockford/JSON-js 中的 JSON in JavaScript)。
  考虑到直接执行代码的风险，最好不要在旧版本浏览器中只使用 `eval()` 求值 JSON。
  这个 JSON 垫片脚本最好只在浏览器原生不支持 JSON 解析时使用

JSON 对象有两个方法:

* `stringify()`
* `parse()`

### 序列化选项

* `JSON.stringify()`
    - 将 JavaScript 序列化为 JSON 字符串
    - 在序列化 JavaScript 对象时，所有函数和原型成员都会有意地在结果中省略
       此外，值为 undefined 的任何属性也会被跳过。
       最终得到的就是所有实例属性均为有效 JSON 数据类型的表示


```js
// js 对象
let book = {
    title: "Professional JavaScript",
    authors: [
        "Nicholas C. Zakas",
        "Matt Frisbie"
    ],
    edition: 4,
    year: 2017
};
// JSON 字符串
let jsonText = JSON.stringify(book);
console.log(jsonText);
```

```json
'{
    "title":"Professional JavaScript",
    "authors":[
        "Nicholas C. Zakas",
        "Matt Frisbie"
    ],
    "edition":4,
    "year":2017
}'
```

### 解析选项

* `JSON.parse()`
    - 将 JavaScript 解析为原生 JavaScript 值

```js
// js 对象
let bookCopy = JSON.parse(jsonText);
console.log(bookCopy);
```

<img src="images/js.jpg" width="100%">

* `JSON.parse()` 方法也可以接收一个额外的参数，这个函数会针对每个键值对都调用一次，
  这个函数被称为**还原函数(reviver)**


## JSON 序列化











