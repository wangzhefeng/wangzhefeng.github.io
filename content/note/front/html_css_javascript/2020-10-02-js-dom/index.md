---
title: DOM
author: 王哲峰
date: '2020-10-02'
slug: js-dom
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
</style>

<details><summary>目录</summary><p>

- [概览](#概览)
- [DOM](#dom)
- [节点层级](#节点层级)
  - [HTML 节点层级](#html-节点层级)
  - [XML 节点层级](#xml-节点层级)
  - [Node 类型](#node-类型)
    - [Node 接口和 Node 类型](#node-接口和-node-类型)
    - [nodeType](#nodetype)
    - [nodeName 与 nodeValue](#nodename-与-nodevalue)
    - [节点关系](#节点关系)
    - [操纵节点](#操纵节点)
    - [其他方法](#其他方法)
  - [Document 类型](#document-类型)
  - [Element 类型](#element-类型)
    - [Element 类型节点特征](#element-类型节点特征)
    - [HTML 元素](#html-元素)
    - [取得属性](#取得属性)
    - [设置属性](#设置属性)
    - [attributes 属性](#attributes-属性)
    - [创建元素](#创建元素)
    - [元素后代](#元素后代)
  - [Text 类型](#text-类型)
  - [Comment 类型](#comment-类型)
  - [CDATASection 类型](#cdatasection-类型)
  - [DocumentType 类型](#documenttype-类型)
  - [Attr 类型](#attr-类型)
- [DOM 编程](#dom-编程)
  - [动态脚本](#动态脚本)
  - [动态样式](#动态样式)
  - [操作表格](#操作表格)
  - [使用 NodeList](#使用-nodelist)
- [MutationObserver 接口](#mutationobserver-接口)
- [DOM 扩展](#dom-扩展)
- [DOM2 和 DOM3](#dom2-和-dom3)
</p></details><p></p>


# 概览

文档对象模型(DOM, Document Object Model)是语言中立的 HTML 和 XML 文档的 API。
DOM Level 1 将 HTML 和 XML 文档定义为一个节点的多层级结构，
并暴露出 JavaScript 接口以操作文档的底层结构和外观

# DOM

文档对象模型(DOM, Document Object Model)是 HTML 和 XML 文档的编程接口。
DOM 表示由多层节点构成的文档，通过它开发者可以添加、删除、修改页面的各个部分。
脱胎于网景和微软早期的动态 HTML(DHTML，Dynamic HTML)，DOM 现在是真正跨平台、
语言无关的表示和操作网页的方式

DOM Level 1 在 1998 年成为 W3C 推荐标准，提供了基本文档结构和查询的接口。
本章之所以介绍 DOM，主要因为它与浏览器中的 HTML 网页相关，并且在 JavaScript 中提供了 DOM API

# 节点层级

任何 HTML 或 XML 文档都可以用 DOM 表示一个由节点构成的层级结构

节点分很多类型，每种类型对应着文档中不同的信息和(或)标记，也有自己不同的特性、数据和方法，
而且与其他类型有某种关系。这些关系构成了层级，让标记可以表示为一个以特定节点为根的树形结构

## HTML 节点层级

* HTML DOM 树形结构

```html
<html>
    <head>
        <title>Sample Page</title>
    </head>
    <body>
        <p>Hello World!</p>
    </body>
</html>
```

* HTML DOM 层级结构

<img src="images/html_dom.jpg" alt="image" style="zoom:33%;" />

* HTML DOM 文档元素
    - document 节点表示每个文档的根节点，根节点的唯一子节点是 `<html>` 元素，称为**文档元素(documentElement)**，
      文档元素是文档最外层的元素，所有元素都存在于这个元素之内。在 HTML 页面中，文档元素始终是 `<html>` 元素

* HTML DOM 节点类型
    - HTML 中每段标记都可以表示为这个树形结构中的一个节点，DOM 中总共有 12 中节点类型，这些节点都继承一种基本类型:
        - 元素节点表示 HTML 元素
        - 属性节点表示属性
        - 文档类型节点表示文档类型
        - 注释节点表示注释
        - ...

## XML 节点层级

* XML DOM 树形结构
* XML DOM 文档元素
    - 在 XML 文档中，任何元素都可能成为文档元素

## Node 类型

### Node 接口和 Node 类型

DOM Level 1 描述了名为 `Node` 的接口，这个接口是所有 DOM 节点类型都必须实现的。
`Node` 接口在 JavaScript 中被实现为 `Node` 类型，在除 IE 之外的所有浏览器中都可以直接访问这个类型。
在 JavaScript 中，所有节点类型都继承自 `Node` 类型，因此所有类型都共享相同的基本属性和方法

### nodeType

每个节点都有 `nodeType` 属性，表示该节点的类型，节点类型由定义在 `Node` 类型上的 12 个数值常量表示:

* Node.ELEMENT_NODE (1)
* Node.ATTRIBUTE_NODE (2)
* Node.TEXT_NODE (3)
* Node.CDATA_SECTION_NODE (4)
* Node.ENTITY_REFERENCE_NODE (5)
* Node.ENTITY_NODE (6)
* Node.PROCESSING_INSTRUCTION_NODE (7)
* Node.COMMENT_NODE (8)
* Node.DOCUMENT_NODE (9)
* Node.DOCUMENT_TYPE_NODE (10)
* Node.DOCUMENT_FRAGMENT_NODE (11)
* Node.NOTATION_NODE (12)

节点类型可通过与这些常量比较来确定:

```js
if (someNode.nodeType == Node.ELEMENT_NODE) {
    alert("Node is an element.");
}
```

浏览器并不支持所有节点类型，开发者最常用的是元素节点和文本节点

### nodeName 与 nodeValue

`nodeName` 与 `nodeValue` 保存着有关节点的信息。
这两个属性的值完全取决于节点类型。在使用这两个属性前，最好先检测节点类型

```js
if (someNode.nodeType == 1) {
    name = someNode.nodeName;    // 会显示元素的标签名
    value = someNode.nodeValue;  // null
}
```

### 节点关系




### 操纵节点


### 其他方法


## Document 类型

## Element 类型

除了 Document 类型，Element 类型就是 Web 开发中最常用的类型了。
Element 表示 XML 或 HTML 元素，对外暴露出访问元素标签名、子节点、属性的能力

### Element 类型节点特征

Element 类型的节点具有以下特征:

* `nodeType` 等于 1
* `nodeName` 和 `tagName` 值为元素的标签名
    - 在 HTML 中，元素标签名始终以全大写表示；
    - 在 XML 中，元素标签名始终与源代码中的大小写一致
* `nodeValue` 值为 `null`
* `parentNode` 的值为 Document 或 Element 对象
* 子节点可以是 Element、Text、Comment、ProcessingInstruction、CDATASection、EntityReference 类型

示例:

```html
<div id="myDiv"></div>
```

```js
let div = document.getElementById("myDiv");

alert(div.nodeName);                 // "DIV"
alert(div.tagName);                  // "DIV"
alert(div.tagName == div.nodeName);  // true
```

```js
// 不要这样做，可能出错
if (elemnet.tagName == "div") {
    // do something here
}

// 推荐，使用于所有文档: HTML 或 XML
if (element.tagName.toLowerCase() == "div") {
    // do something here
}
```

### HTML 元素

### 取得属性

### 设置属性


### attributes 属性

### 创建元素

### 元素后代






## Text 类型

## Comment 类型

## CDATASection 类型

## DocumentType 类型

## Attr 类型


# DOM 编程

## 动态脚本


## 动态样式


## 操作表格


## 使用 NodeList



# MutationObserver 接口



# DOM 扩展



# DOM2 和 DOM3



