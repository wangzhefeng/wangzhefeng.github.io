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

- [TODO](#todo)
- [概览](#概览)
  - [DOM](#dom)
  - [DOM 扩展](#dom-扩展)
  - [DOM2 和 DOM3](#dom2-和-dom3)
- [DOM 节点层级](#dom-节点层级)
  - [HTML 节点层级](#html-节点层级)
  - [XML 节点层级](#xml-节点层级)
  - [Node 类型](#node-类型)
    - [Node 接口和 Node 类型](#node-接口和-node-类型)
    - [nodeType 属性](#nodetype-属性)
    - [nodeName 与 nodeValue](#nodename-与-nodevalue)
    - [节点关系](#节点关系)
    - [操纵节点](#操纵节点)
    - [其他方法](#其他方法)
  - [Document 类型](#document-类型)
    - [文档子节点](#文档子节点)
    - [文档信息](#文档信息)
    - [定位元素](#定位元素)
    - [特殊集合](#特殊集合)
    - [DOM 兼容性检测](#dom-兼容性检测)
    - [文档写入](#文档写入)
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
    - [Comment 类型](#comment-类型-1)
    - [Comment 类型属性](#comment-类型属性)
    - [浏览器不承认注释](#浏览器不承认注释)
  - [CDATASection 类型](#cdatasection-类型)
    - [CDATASection 类型的节点特征](#cdatasection-类型的节点特征)
    - [XML 中的 CDATA 区块](#xml-中的-cdata-区块)
  - [DocumentType 类型](#documenttype-类型)
  - [Attr 类型](#attr-类型)
    - [Attr 对象属性](#attr-对象属性)
- [DOM 编程](#dom-编程)
  - [动态脚本](#动态脚本)
    - [引入外部文件](#引入外部文件)
    - [直接插入源代码](#直接插入源代码)
  - [动态样式](#动态样式)
    - [引入外部文件](#引入外部文件-1)
    - [直接嵌入 CSS 规则](#直接嵌入-css-规则)
  - [操作表格](#操作表格)
    - [DOM 编程方式创建表格](#dom-编程方式创建表格)
    - [HTML DOM 给表格元素添加的属性和方法](#html-dom-给表格元素添加的属性和方法)
  - [使用 NodeList](#使用-nodelist)
- [DOM MutationObserver 接口](#dom-mutationobserver-接口)
  - [基本用法](#基本用法)
  - [MutationObserverInit 与观察范围](#mutationobserverinit-与观察范围)
  - [异步回调与记录队列](#异步回调与记录队列)
  - [性能、内存与垃圾回收](#性能内存与垃圾回收)
- [DOM 扩展](#dom-扩展-1)
  - [Selectors API](#selectors-api)
  - [元素遍历](#元素遍历)
  - [HTML5](#html5)
  - [专有扩展](#专有扩展)
- [DOM2 和 DOM3](#dom2-和-dom3-1)
  - [DOM 的演进](#dom-的演进)
  - [样式](#样式)
  - [遍历](#遍历)
  - [范围](#范围)
</p></details><p></p>


# TODO

- [ ] item 1
- [ ] item 2

# 概览

文档对象模型(DOM, Document Object Model)是语言中立的 HTML 和 XML 文档的 API。
DOM Level 1 将 HTML 和 XML 文档定义为一个节点的多层级结构，
并暴露出 JavaScript 接口以操作文档的底层结构和外观

## DOM

文档对象模型(DOM, Document Object Model)是 HTML 和 XML 文档的编程接口。
DOM 表示由多层节点构成的文档，通过它开发者可以添加、删除、修改页面的各个部分。
脱胎于网景和微软早期的动态 HTML(DHTML，Dynamic HTML)，DOM 现在是真正跨平台、
语言无关的表示和操作网页的方式

DOM Level 1 在 1998 年成为 W3C 推荐标准，提供了基本文档结构和查询的接口。
本章之所以介绍 DOM，主要因为它与浏览器中的 HTML 网页相关，
并且在 JavaScript 中提供了 DOM API

## DOM 扩展

尽管 DOM API 已经相当不错，但仍然不断有标准或专有的扩展出现，以支持更多功能。
2008 年以前，大部分浏览器对 DOM 的扩展是专有的。
此后，W3C 开始着手将这些已成为事实标准的专有扩展编制成正式规范

基于以上背景，诞生了描述 DOM 扩展的两个标准: Selectors API 与 HTML5。
这两个标准体现了社区需求和标准化某些手段及 API 的愿景。
另外还有较小的 Element Traversal 规范，增加了一些 DOM 属性

专有扩展虽然还有，但这两个规范(特别是 HTML5)已经涵盖其中大部分

## DOM2 和 DOM3


DOM1(DOM Level 1)主要定义了 HTML 和 XML 文档的底层结构。
DOM2(DOM Level 2)和 DOM3(DOM Level 3)在这些结构之上加入更多交互能力，
提供了更高级的 XML 特性。实际上，DOM2 和 DOM3 是按照模块化的思路来制定标准的，
每个模块之间有一定关联，但分别针对某个 DOM 子集。 这些模式如下所示:

* DOM Core: 在 DOM1 核心部分的基础上，为节点增加方法和属性
* DOM Views: 定义基于样式信息的不同视图
* DOM Events: 定义通过事件实现 DOM 文档交互
* DOM Style: 定义以编程方式访问和修改 CSS 样式的接口
* DOM Traversal and Range: 新增遍历 DOM 文档及选择文档内容的接口
* DOM HTML: 在 DOM1 HTML 部分的基础上，增加属性、方法和新接口
* DOM Mutation Observers: 定义基于 DOM 变化触发回调的接口。
  这个模块是 DOM4 级模块，用于取代 Mutation Events

本章介绍除 DOM Events 和 DOM Mutation Observers 之外的其他所有模块，
第 17 章会专门介绍事件，而 DOM Mutation Observers 第 14 章已经介绍过了。
DOM3 还有 XPath 模块和 Load and Save 模块，将在 第 22 章介绍


# DOM 节点层级

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
    - `document` 节点表示每个文档的根节点，根节点的唯一子节点是 `<html>` 元素，称为**文档元素(documentElement)**，
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

### nodeType 属性

每个节点都有 `nodeType` 属性，表示该节点的类型，节点类型由定义在 `Node` 类型上的 12 个数值常量表。
浏览器并不支持所有节点类型，开发者最常用的是元素节点和文本节点:

* `Node.ELEMENT_NODE` (1)
    - 元素节点
* `Node.ATTRIBUTE_NODE` (2)
* `Node.TEXT_NODE` (3)
    - 文本节点
* `Node.CDATA_SECTION_NODE` (4)
* `Node.ENTITY_REFERENCE_NODE` (5)
* `Node.ENTITY_NODE` (6)
* `Node.PROCESSING_INSTRUCTION_NODE` (7)
* `Node.COMMENT_NODE` (8)
* 文档节点
    - `Node.DOCUMENT_NODE` (9)
    - `Node.DOCUMENT_TYPE_NODE` (10)
    - `Node.DOCUMENT_FRAGMENT_NODE` (11)
* `Node.NOTATION_NODE` (12)

节点类型可通过与这些常量比较来确定:

```js
if (someNode.nodeType == Node.ELEMENT_NODE) {
    alert("Node is an element.");
}
```

### nodeName 与 nodeValue

`nodeName` 与 `nodeValue` 保存着有关节点的信息。
这两个属性的值完全取决于节点类型。在使用这两个属性前，最好先检测节点类型

```js
if (someNode.nodeType == 1) {
    name = someNode.nodeName;    // 会显示元素的标签名，对元素节点而言，nodeName 始终等于元素的标签名
    value = someNode.nodeValue;  // 对元素节点而言，nodeValue 始终为 null
}
```

### 节点关系




### 操纵节点


### 其他方法


## Document 类型

`Document` 类型是 JavaScript 中表示文档节点的类型。
在浏览器中，文档对象 `document` 是 `HTMLDocument` 的实例，
`HTMLDocument` 继承 `Document`，表示整个 HTML 页面。
`document` 是 `window` 对象的属性，因此是一个全局对象

* `Document` 类型的节点有以下特征
    - nodeType 等于 9
    - nodeName 值为 `"#document"`
    - nodeValue 值为 `null`
    - parentNode 值为 `null`
    - ownerDocument 值为 `null`
    - 子节点可以是 `DocumentType` (最多一个)、`Element` (最多一个)、`ProcessingInstruction` 或 `Comment` 类型

`Document` 类型可以表示 HTML 页面或其他 XML 文档，
但最常用的还是通过 `HTMLDocument` 的实例取得 `document` 对象。
`document` 对象可以获取关于页面的信息以及操纵其外观和底层结构

```
- Document 类型
    - HTMLDocument 类型 => document 实例对象: 整个 HTML 页面
    - nodeType
    - nodeName
    - nodeValue
    - parentNode
    - ownerDocument
    - 子节点
        - DocumentType
        - Element
        - ProcessingInstruction
        - Comment
```

### 文档子节点

DOM 规范规定 `Document` 节点的子节点可以是 `DocumentType`、`Element`、`ProcessingInstruction` 或 `Comment`，

1. 两个访问子节点的快捷方式，`document.documentElement`、`document.body` 属性。
   所有主流浏览器都支持这两个属性

* `documentElement` 属性
    - 始终指向 HTML 页面中的 `<html>` 元素，虽然 `document.childNodes` 中始终有 `<html>` 元素，
      但使用 `documentElement` 属性可以更快更直接地访问该元素

    ```html
    <html>
        <body>
        </body>
    </html>
    ```

    ```js
    let html = document.documentElement;  // 取得对 <html> 的引用

    alert(html === document.childNodes[0]);  // true
    alert(html === document.firstChild);  // true
    ```

* `body` 属性
    - 作为 `HTMLDocument` 的实例，`document` 对象还有一个 `body` 属性，
      直接指向 `<body>` 元素。因为这个元素是开发者使用最多的元素，
      所以 JavaScript 代码中经常可以看到 `document.body`

    ```js
    let body = document.body;  // 取得对 <body> 的引用
    ```

2. `<!doctype>` 标签是文档中独立的部分，
其信息可以通过 `doctype` 属性(在浏览器中是 `document.doctype`)来访问

```js
let doctype = document.doctype;  // 取得对 <!doctype> 的引用
```

3. 严格来讲出现在 `<html>` 元素外面的注释也是文档的子节点，它们的类型是 `Comment`。
   不过，由于浏览器实现不同，这些注释不一定能被识别，或者表现可能不一致。
   这个页面看起来有 3 个子节点: 注释、`<html>` 元素、注释。
   逻辑上讲，`document.childNodes` 应该包含 3 项，对应代码中的每个节点。
   但实际上，浏览器有可能以不同方式对待 `<html>` 元素外部的注释，比如忽略一个或两个注释

```js
<!-- 第一条注释 -->
<html>
    <body>

    </body>
</html>
<!-- 第二条注释 -->
```

4. 一般来说，`appendChild()`、`removeChild()` 和 `replaceChild()` 方法不会用在 `document` 对象上。
   这是因为文档类型(如果存在)是只读的，而且只能有一个 `Element` 类型的子节点(即 `<html>`，已经存在了)


### 文档信息

`document` 作为 `HTMLDocument` 的实例，还有一些标准 `Document` 对象上所没有的属性。
这些属性提供浏览器所记载网页的信息

* 第一个属性是 `title`，包含 `<title>` 元素中的文本，通常显示在浏览器窗口或标签页的标题栏。
  通过这个属性可以读写页面的标题，修改后的标题也会反映在浏览器标题栏上。
  不过，修改 `title` 属性并不会改变 `<title>` 元素

```js
// 读取文档标题
let originalTitle = document.title;

// 修改文档标题
document.title = "New page title";
```

* `URL` 属性
    - URL 包含当前页面的完整 URL(地址栏中的 URL)
    - 可以在请求的 HTTP 头部信息中获取，只是在 JavaScript 中通过这几个属性暴露出来而已
    - URL 跟域名是相关的，比如，如果 `document.URL` 是 `http://www.wrox.com/WileyCDA/`，
      则 `document.domain`  就是 `www.wrox.com`
* `domain` 属性
    - `domain` 包含页面的域名
    - 可以在请求的 HTTP 头部信息中获取，只是在 JavaScript 中通过这几个属性暴露出来而已
    - `domain` 是可以设置的，出于安全考虑，给 `domain` 属性设置的值是有限制的。
      如果 URL 包含子域名如 `p2p.wrox.com`，则可以将 `domain` 设置为 `wrox.com`。
      不能给 `domain` 属性设置 URL 中不包含的值
    - 当页面中包含来自某个不同子域的窗格(`<frame>`)或内嵌窗格(`<iframe>`)时，
      设置 `document.domain` 是有用的。因为跨源通信存在安全隐患，所以不同子域的页面间无法通过 JavaScript 通信，
      在每个页面上把 `document.domain` 设置为相同的值，这些页面就可以访问对方的 JavaScript 对象了
    - 浏览器对 `domain` 属性还有一个限制，即这个属性一旦放松就不能再收紧了，比如，把 `document.domain` 设置为 `wrox.com` 之后，
      就不能再将其设置回 `p2p.wrox.com`，后者会导致错误
* `reference` 属性
    - `reference` 包含链接到当前页面的那个页面的 URL。
      如果当前页面没有来源，则 `reference` 属性包含空字符串
    - 可以在请求的 HTTP 头部信息中获取，只是在 JavaScript 中通过这几个属性暴露出来而已

```js
// 取得完整的 URL
let url = document.URL;

// 取得域名
let domain = document.domain;

// 取得来源
let reference = document.reference;
```

### 定位元素

使用 DOM 最常见的情形可能就是获取某个或某组元素的引用，然后对它们执行某些操作。
`document` 对象上暴露了一些方法，可以实现这些操作

* `getElementById()`
    - 接收一个参数，即要获取元素的 ID，如果找到了则返回这个元素，
      如果没找到则返回 `null`。参数 ID 必须跟元素在页面中的 `id` 属性值完全匹配，包括大小写
    - 如果页面中存在多个具有相同 ID 的元素，则返回在文档中第一次出现的第一个元素
* `getElementByTagName()`
    - 



### 特殊集合

`document` 对象上还暴露了几个特殊集合，这些集合也都是 `HTMLCollection` 的实例。
这些集合是访问文档中公共部分的快捷方式，这些特殊集合始终存在于 `HTMLDocument` 对象上，
而且与所有 `HTMLCollection` 对象一样，其内荣也会实时更新以符合前文档的内容

* `document.anchors` 包含文档中所有带 `name` 属性的 `<a>` 元素
* `document.applets` 包含文档中所有 `<applet>` 元素，已废弃
* `document.forms` 包含文档中所有 `<form>` 元素
    - 与 `document.getElementsByTagName("form")` 返回的结果相同
* `document.images` 包含文档中所有 `<img>` 元素
    - 与 `document.getElementsByTagName("form")` 返回的结果相同
* `document.links` 包含文档中所有带 `href` 属性的 `<a>` 元素

### DOM 兼容性检测





### 文档写入





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

### Comment 类型

DOM 中的注释通过 `Comment` 类型表示。`Commont` 类型的节点具有以下特征：

* `nodeType` 等于 8
* `nodeName` 等于 `"#comment"`
* `nodeValue` 值为注释的内容
* `parentNode` 值为 `Document` 或 `Element` 对象
* 不支持子节点

### Comment 类型属性

`Comment` 类型与 `Text` 类型继承同一个基类(`CharacterData`)，
因此拥有除 `splitText()` 之外 `Text` 节点所有的字符串操作方法。
与 `Text` 类型相似，注释的实际内容可以通过 `nodeValue` 或 `data` 属性获得

* 注释节点可以作为父节点的子节点来访问

```html
<div id="myDiv">
    <!-- A comment -->
</div>
```

```js
let div = document.getElementById("myDiv");
let comment = div.firstChild;

alert(comment.data);  // "A comment"
```

* 可以用 `document.createComment()` 方法创建注释节点，参数为注释文本

```js
let comment = document.createComment("A comment");
```

### 浏览器不承认注释

显然，注释节点很少通过 JavaScript 创建和访问，因为注释几乎不涉及算法逻辑。
此外，浏览器不承认结束的 `</html>` 标签之后的注释。如果要访问注释节点，
则必须确定它们是 `<html>` 元素的后代

## CDATASection 类型

CDATASection 类型表示 XML 中特有的 CDATA 区块。
CDATASection 类型继承 Text 类型，
因此拥有包括 `splitText()` 在内的所有字符串操作方法。

### CDATASection 类型的节点特征

CDATASection 类型的节点具有以下特征：

* `nodeType` 等于 4
* `nodeName` 值为 `"#cdata-section"`
* `nodeValue` 值为 CDATA 区块的内容
* `parentNode` 值为 `Document` 或 `Element` 对象
* 不支持子节点

### XML 中的 CDATA 区块

CDATA 区块只在 XML 文档中有效，
因此某些浏览器比较陈旧的版本会错误地将 CDATA 区块解析为 `Comment` 或 `Element`





## DocumentType 类型

## Attr 类型

元素数据在DOM 中通过 `Attr` 类型表示。`Attr` 类型构造函数和原型在所有浏览器中都可以直接访问。
技术上讲，属性是存在于元素 `attributes` 属性中的节点。`Attr` 节点具有以下特征：

* `nodeType` 等于 2
* `nodeName` 值为属性名
* `nodeValue` 值为属性值
* `parentNode` 值为 `null`
* 在 HTML 中不支持子节点
* 在 XML 中子节点可以是 `Text` 或 `EntityReference`

### Attr 对象属性

* name
* value
* specified





# DOM 编程

很多时候，操作 DOM 是很直观的。通过 HTML 代码能实现的，也一样能通过 JavaScript 实现。
但有时候，DOM 也没有看起来那么简单。浏览器能力的参差不齐和各种问题，
也会导致 DOM 的某些方面会复杂一些

## 动态脚本

`<script>` 元素用于向网页中插入 JavaScript 代码，可以是 `src` 属性包含的外部文件，
也可以是作为该元素内容的源代码。

动态脚本就是在页面初始加载时不存在，之后又通过 DOM 包含的脚本。
与对应的 HTML 元素一样，有两种方式通过 `<script>` 动态为网页添加脚本

### 引入外部文件

`script` 引入外部文件:

```html
<script src="foo.js"></script>
```

DOM 编程:

```js
let script = document.createElement("script");
script.src = "foo.js";

document.body.appendChild(script);  // 把 <script> 元素添加到页面之前，是不会开始下载外部文件的
```

上面的 DOM 编程过程可以抽象为一个函数:

```js
function loadScript(url) {
    let script = document.createElement("script");
    script.src = url;
    
    document.body.appendChild(script);
}

loadScript("client.js");
```

### 直接插入源代码

`<script>` 元素插入源代码:

```html
<script>
    function sayHi() {
        alert("hi");
    }
</script>
```

在 Firefox、Safari、Chrome 和 Opera 中使用 DOM 编程可以实现以下逻辑:

```js
let script = document.createElement("script");
script.appendChild(
    document.createTextNode("function sayHi() {alert('hi');}")
);

document.body.appendChild(script);
```

在旧版本的 IE 中可能会导致问题，需要使用 `<script>` 元素的 `text` 属性，
修改后的代码能够在 IE、Firefox、Opera 和 Safari 3 及更高版本中运行：

```js
var script = document.createElement("script");
script.text = "function sayHi() {alert('hi');}";

document.body.appendChild(script);
```

对于早期的 Safari 版本，需要使用以下代码：

```js
var script = document.createElement("script");
var code = "function sayHi() {alert('hi');}";
try {
    script.appendChild(document.createTextNode("code"));
} catch (ex) {
    script.text = "code";
}

document.body.appendChild(script);
```

抽象出一个跨浏览器的函数:

```js
function loadScriptString(code) {
    var script = document.createElement("script");
    script.type = "text/javascript";
    try {
        script.appendChild(document.createTextNode(code));
    } catch (ex) {
        script.text = code;
    }

    document.body.appendChild(script);
}

loadScriptString("function sayHi() {alert('hi');}");
```

通过 `innerHTML` 属性创建的 `<script>` 元素永远不会执行。
浏览器会尽责地创建 `<script>` 元素，以及其中的脚本文本，
但解析器会给这个 `<script>` 元素打上永不执行的标签。
只要是使用 `innerHTML` 创建的 `<script>` 元素，以后也没有办法强制其执行

## 动态样式

CSS 样式在 HTML 页面中可以通过两个元素加载

* `<link>` 元素用于包含 CSS 外部文件
* `<style>` 元素用于添加嵌入样式

动态样式在页面初始加载时病不存在，而是在之后才添加到页面中的

### 引入外部文件

通过外部文件加载样式是一个异步过程。因此，样式的加载和正执行的 JavaScript 代码并没有先后顺序。
一般来说，也没有必要知道样式什么时候加载完成

`<link>` 元素引入外部文件：

```html
<link rel="stylesheet" type="text/css" href="style.css">
```

DOM 编程：

```js
let link = document.createElement("link");
link.rel = "stylesheet";
link.type = "text/css";
link.href = "style.css";

let head = document.getElementsByTagName("head")[0];
head.appendChild(link);
```

上面的 DOM 编程过程可以抽象为一个函数:

```js
function loadStyles(url) {
    let link = document.createElement("link");
    link.rel = "stylesheet";
    link.type = "text/css";
    link.href = url;

    let head = document.getElementsByTagName("head")[0];
    head.appendChild(link);
}

loadStyles("style.css");
```

### 直接嵌入 CSS 规则

使用 `<style>` 元素直接嵌入 CSS 规则:

```html
<style type="text/css">
    body {
        background-color: red;
    }
</style>
```

DOM 编程:

```js
let style = document.createElement("style");
style.type = "text/css";
style.appendChild(
    document.createTextNode("body{background-color:red}");
);

let head = document.getElementsByTagName("head")[0];
head.appendChild(style);
```

以上代码在 Firefox、Safari、Chrome 和 Opera 中都可以运行，但 IE 除外。
IE 对 `<style>` 节点会施加限制，不允许访问其子节点，
这一点与它对 `<script>` 元素施加的限制一样。
事实上，IE 在执行到给 `<style>` 添加子节点的代码时，
会抛出与给 `<script>` 添加子节点时同样的错误。

对于 IE，解决方案是访问元素的 `styleSheet` 属性，这个属性又有一个 `cssText` 属性，
然后给这个属性添加 CSS 代码

```js
let style = document.createElement("style");
style.type = "text/css";
try {
    style.appendChild(
        document.createTextNode("body{background-color:red}")
    );
} catch (ex) {
    style.styleSheet.cssText = "body{background-color:red}";
}

let head = document.getElementsByTagName("head")[0];
head.appendChild(style);
```

抽象为一个通用函数:

```js
function loadStyleString(css) {
    let style = document.createElement("style");
    style.type = "text/css";
    try {
        style.appendChild(
            document.createTextNode(css)
        );
    } catch (ex) {
        style.styleSheet.cssText = css;
    }

    let head = document.getElementsByTagName("head")[0];
    head.appendChild(style);
}

loadStyleString("body{background-color:red}");
```

## 操作表格

表格是 HTML 中最复杂的结构之一。通过 DOM 编程 创建 `<table>` 元素，通常要涉及大量标签，
包括表行、表元、表题，等等。因此，通过编程创建和修改表格时要写很多代码

### DOM 编程方式创建表格


`<table>` 元素创建表格:

```html
<table border="1" width="100%">
    <tbody>
        <tr>
            <td>Cell 1,1</td>
            <td>Cell 2,1</td>
        </tr>
        <tr>
            <td>Cell 1,2</td>
            <td>Cell 2,2</td>
        </tr>
    </tbody>
</table>
```

DOM 编程:

```js

```

### HTML DOM 给表格元素添加的属性和方法

为了方便创建表格，HTML DOM 给 `<table>`、`<tbody>` 和 `<tr>` 元素添加了一些属性和方法






## 使用 NodeList










# DOM MutationObserver 接口


## 基本用法

## MutationObserverInit 与观察范围


## 异步回调与记录队列


## 性能、内存与垃圾回收








# DOM 扩展

## Selectors API


## 元素遍历


## HTML5


## 专有扩展









# DOM2 和 DOM3

## DOM 的演进

## 样式


## 遍历

## 范围