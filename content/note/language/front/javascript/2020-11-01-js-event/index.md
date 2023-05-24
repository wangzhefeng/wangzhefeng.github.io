---
title: 事件
author: 王哲峰
date: '2020-11-01'
slug: js-event
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
- [事件概览](#事件概览)
- [事件流](#事件流)
  - [事件冒泡](#事件冒泡)
    - [事件冒泡介绍](#事件冒泡介绍)
    - [事件冒泡示例](#事件冒泡示例)
  - [事件捕获](#事件捕获)
    - [事件捕获介绍](#事件捕获介绍)
    - [事件捕获示例](#事件捕获示例)
  - [DOM 事件流](#dom-事件流)
    - [DOM 事件流示例](#dom-事件流示例)
- [事件处理程序](#事件处理程序)
  - [HTML 事件处理程序](#html-事件处理程序)
    - [HTML 事件处理程序示例](#html-事件处理程序示例)
    - [HTML 事件处理程序问题](#html-事件处理程序问题)
  - [DOM0 事件处理程序](#dom0-事件处理程序)
    - [赋值事件处理程序属性一个函数](#赋值事件处理程序属性一个函数)
    - [事件处理程序在元素的作用域中运行](#事件处理程序在元素的作用域中运行)
    - [移除事件处理程序](#移除事件处理程序)
  - [DOM2 事件处理程序](#dom2-事件处理程序)
    - [DOM2 事件处理程序方法](#dom2-事件处理程序方法)
    - [addEventListener()](#addeventlistener)
    - [removeEventListener()](#removeeventlistener)
  - [IE 事件处理程序](#ie-事件处理程序)
    - [attachEvent()](#attachevent)
    - [detachEvent()](#detachevent)
  - [跨浏览器事件处理程序](#跨浏览器事件处理程序)
- [事件对象](#事件对象)
  - [DOM 事件对象](#dom-事件对象)
    - [DOM event 事件对象](#dom-event-事件对象)
    - [HTML 属性事件处理程序使用 event 引用事件对象](#html-属性事件处理程序使用-event-引用事件对象)
    - [事件对象的公共属性和方法](#事件对象的公共属性和方法)
  - [IE 事件对象](#ie-事件对象)
    - [IE event 事件对象](#ie-event-事件对象)
    - [HTML 属性事件处理程序使用 event 引用事件对象](#html-属性事件处理程序使用-event-引用事件对象-1)
    - [事件对象的公共属性和方法](#事件对象的公共属性和方法-1)
  - [跨浏览器事件对象](#跨浏览器事件对象)
- [事件类型](#事件类型)
  - [用户界面事件](#用户界面事件)
    - [load 事件](#load-事件)
    - [unload 事件](#unload-事件)
    - [resize 事件](#resize-事件)
    - [scroll 事件](#scroll-事件)
  - [焦点事件](#焦点事件)
  - [鼠标和滚轮事件](#鼠标和滚轮事件)
  - [键盘与输入事件](#键盘与输入事件)
  - [合成事件](#合成事件)
  - [变化事件](#变化事件)
  - [HTML5 事件](#html5-事件)
    - [contextmenu 事件](#contextmenu-事件)
    - [beforeunload 事件](#beforeunload-事件)
    - [DOMContentLoaded 事件](#domcontentloaded-事件)
    - [readystatechange 事件](#readystatechange-事件)
    - [pageshow 与 pagehide 事件](#pageshow-与-pagehide-事件)
    - [hashchange 事件](#hashchange-事件)
  - [设备事件](#设备事件)
    - [orientationchange 事件](#orientationchange-事件)
    - [deviceorientation 事件](#deviceorientation-事件)
    - [devicemotion 事件](#devicemotion-事件)
  - [触摸及手势事件](#触摸及手势事件)
    - [触摸事件](#触摸事件)
    - [手势事件](#手势事件)
    - [触摸事件与手势事件的联系](#触摸事件与手势事件的联系)
  - [事件参考](#事件参考)
- [内存与性能](#内存与性能)
  - [事件委托](#事件委托)
  - [删除事件处理程序](#删除事件处理程序)
- [模拟事件](#模拟事件)
  - [DOM 事件模拟](#dom-事件模拟)
    - [模拟鼠标事件](#模拟鼠标事件)
    - [模拟键盘事件](#模拟键盘事件)
    - [模拟其他事件](#模拟其他事件)
    - [自定义 DOM 事件](#自定义-dom-事件)
  - [IE 事件模拟](#ie-事件模拟)
</p></details><p></p>


# TODO

* [ ] 事件对象的公共属性和方法

# 事件概览

JavaScript 与 HTML 的交互是通过事件实现的，
事件代表文档或浏览器窗口中某个有意义的时刻。
可以使用仅在事件发生时执行的监听器(也叫处理程序)订阅事件

在传统软件工程领域，这个模型叫做“观察者模式”，
其能够做到页面行为(在 JavaScript 中定义)与页面展示(在 HTML 和 CSS 中定义)的分离

事件是 JavaScript 与网页结合的主要方式。最常见的事件是在 DOM3 Events 规范或 HTML5 中定义的。
虽然基本的事件都有规范定义，但很多浏览器在规范之外实现了自己专有的事件，
以方便开发者更好地满足用户交互需求，其中一些专有事件直接与特殊的设备相关。

围绕着使用事件，需要考虑内存与性能问题。例如:

* 最好限制一个页面中事件处理程序的数量，因为它们会占用过多内存，导致页面响应缓慢
* 利用事件冒泡，事件委托可以解决限制事件处理程序数量的问题
* 最好在页面卸载之前删除所有事件处理程序

使用 JavaScript 也可以在浏览器中模拟事件。DOM2 Events 和 DOM3 Events 规范提供了模拟方法，
可以模拟所有原生 DOM 事件。键盘事件一定程度上也是可以模拟的，有时候需要组合其他技术。
IE8 及更早版本也支持事件模拟，只是接口与 DOM 方式不同。
事件是 JavaScript 中最重要的主题之一，理解事件的原理及其对性能的影响非常重要

# 事件流

> 在第四代 Web 浏览器(IE4 和 Netscape Communicator 4)开始开发时，开发团队碰到一个有意思的问题：
> **页面哪个部分拥有特定的事件呢？**
> 要理解这个问题，可以在一张纸上画几个同心圆。把手指放到圆心上，
> 则手指不仅是在一个圆圈里，而且是在所有的圆圈里。
> 两家浏览器的开发团队都是以同样的方式看待浏览器事件的。
> 当你点击一个按钮时，实际上不光点击了这个按钮，还点击了它的容器以及整个页面

事件流描述了页面接收事件的顺序。结果非常有意思，
IE 和 Netscape 开发团队提出了几乎完全相反的事件流方案:

* IE 支持事件冒泡流
* Netscape Communicator 支持事件捕获流

## 事件冒泡

### 事件冒泡介绍

IE 事件流被称为事件冒泡，这是因为事件被定义为从最具体的元素(文档树中最深的节点)开始触发，
然后向上传播至没有那么具体的元素(文档)

所有的现代浏览器都支持事件冒泡，只是在实现方式上会有一些变化:

* IE5.5 及早期版本会跳过 `<html>`元素(从 `<body>`直接到 `document`)
* 现代浏览器中的事件会一直冒泡到 `window` 对象

### 事件冒泡示例

* HTML 页面

```html
<!DOCTYPE html>
<html>
    <head>
        <title>Event Building Example</title>
    </head>
    <body>
        <div id="myDiv">
            Click Me
        </div>
    </body>
</html>
```

* 在点击页面中的 `<div>` 元素后，`click` 事件会以如下顺序发生：
  1. `<div>`
  2. `<body>`
  3. `<html>`
  4. `document`

* 过程图

<img src="images/event1.png" alt="event" style="zoom:50%;" />

## 事件捕获

### 事件捕获介绍

Netscape Communicator 团队提出了另一种名为事件捕获的事件流。
事件捕获的意思是最不具体的节点应该最先收到事件，
而最具体的节点应该最后收到事件。
事件捕获实际上是为了在事件到达最终目标前拦截事件

虽然这是 Netscape Communicator 唯一的事件流模型，
但事件捕获得到了所有现代浏览器的支持

* 所有浏览器都是从 `window` 对象开始捕获事件
* DOM2 Events 规范规定的是从 `document` 开始

由于旧版浏览器不支持，因此实际当中几乎不会使用事件捕获。
通常建议使用事件冒泡，特殊情况下可使用事件捕获

### 事件捕获示例

* HTML 页面

```html
<!DOCTYPE html>
<html>
    <head>
        <title>Event Building Example</title>
    </head>
    <body>
        <div id="myDiv">
            Click Me
        </div>
    </body>
</html>
```

* 在点击页面中的 `<div>` 元素后，`click` 事件会以如下顺序发生：
    1. `document`
    2. `<html>`
    3. `<body>`
    4. `<div>`

* 过程图

<img src="images/event2.png" alt="event" style="zoom:50%;" />

## DOM 事件流

DOM2 Events 规范规定事件流分为 3 个阶段：

* 事件捕获：事件捕获最先发生，为提前拦截事件提供了可能
* 到达目标：然后，实际的目标元素接收到事件后
* 事件冒泡：最后一个阶段是冒泡，最迟要在这个阶段响应事件

大多数支持 DOM 事件流的浏览器实现了一个小小拓展。
虽然 DOM2 Events 规范明确捕获阶段不命中事件目标，
但现代浏览器都会在捕获阶段在事件目标上触发事件。
最终的结果是在事件目标上有两个机会来处理事件

所有现代浏览器都支持 DOM 事件流，只有 IE8 及更早版本不支持

### DOM 事件流示例

* HTML 页面

```html
<!DOCTYPE html>
<html>
    <head>
        <title>Event Building Example</title>
    </head>
    <body>
        <div id="myDiv">
            Click Me
        </div>
    </body>
</html>
```

* 在点击页面的 `<div>` 元素后，`click` 事件会以如下顺序发生
    1. 在 DOM 事件流中，实际的目标(`<div>` 元素)在捕获阶段不会接收到事件。
       这是因为捕获阶段从 `document` 到 `<html>` 再到 `<body>` 就结束了
    2. 下一阶段，即会在 `<div>` 元素上触发事件的“到达目标”阶段。
       通常在事件处理时被认为是冒泡阶段的一部分
    3. 然后，冒泡阶段开始，事件反向传播至文档

* 过程图

<img src="images/event3.png" alt="event" style="zoom:50%;" />

# 事件处理程序

事件意味着用户或浏览器执行的某种动作:

* 单击(`click`)
* 加载(`load`)
* 鼠标悬停(`mouseover`)

为响应事件而调用的函数被称为**事件处理程序**(或**事件监听器**)。
事件处理程序的名字以 `on` 开头:

* `click` 事件的处理程序叫作 `onclick`
* `load` 事件的处理程序叫作 `onload`

有很多方式可以指定事件处理程序:

* HTML
* DOM0
* DOM2
* IE
* 跨浏览器事件处理程序

## HTML 事件处理程序

特定元素支持的每个事件都可以使用事件处理程序的名字以 HTML 属性的形式来指定，
此时属性的值必须是能够执行的 JavaScript 代码

### HTML 事件处理程序示例

(1) 要在按钮被点击时执行某些 JavaScript 代码，可以使用以下 HTML 属性，
点击这个按钮后，控制台会输出一条消息，这种交互能力是通过为 `onclick` 属性指定 JavaScript 代码来实现的

* 因为属性的值是 JavaScript 代码，所以不能在未经转义的情况下使用 HTML 语法字符，比如：`&`、`"`、`<`、`>`
* 为了避免使用 HTML 实体，可以使用单引号代替双引号，如果确实需要使用双引号，将双引号用 `&quot;` 代替

```html
<input type="button" value="Click Me" onclick="console.log('Clicked')" />
<input type="button" value="Click Me" onclick="console.log(&quot;Clicked&quot;)" />
```

(2) 在 HTML 中定义的事件处理程序可以包含精确的动作指令，也可以调用在页面其他地方定义的脚本。
作为事件处理程序执行的代码可以访问全局作用域中的一切

```html
// 单击按钮会调用 showMessage() 函数，
// showMessage() 函数是在单独的 script 元素中定义的，
//  而且也可以在外部文件中定义
<script>
    function showMessage() {
        console.log("Hello World!");
    }
</script>
<input type="button" value="Click Me" onclick="showMessage()">
```

(3) 上面两种方式指定的事件处理程序有一些特殊的地方

* 首先，会创建一个函数来封装属性的值，这个函数有一个特殊的局部变量 `event`，
  其中保存的就是 `event` 对象，有了这个对象，就不用开发者另外定义其他变量，
  也不用从包装函数的参数列表中去取了

```html
<!-- 输出 "click" -->
<input type="button" value="Click Me" onclick="console.log(event.type)">
```

* 其次，在这个函数中，`this` 值相当于事件的目标元素

```html
<!-- 输出 "Click Me" -->
<input type="button" value="Click Me" onclick="console.log(this.value)">
```

* 最后，这个动态创建的包装函数其作用域链被扩展了。
  在这个函数中，`document` 和元素自身的成员都可以被当成局部变量来访问，
  这是通过使用 `with` 实现的

```js
function () {
    with(document) {
        with(this) {
            // 属性值
        }
    }
}
```

* 这意味着事件处理程序可以更方便地访问自己的属性

```html
<!-- 1. 输出 "Click Me" -->
<input type="button" value="Click Me" onclick="console.log(value)">
```

* 如果这个元素是一个表单输入框，则作用域中还会包含表单元素，事件处理程序对应的函数等价于

```js
function () {
    with(document) {
        with(this.form) {
            with(this) {
                // 属性值
            }
        }
    }
}
```

* 本质上，经过这样的扩展，事件处理程序的代码就可以不必引用表单元素，
  而直接访问同一表单中的其他成员了 

```html
<form method="post">
    <input type="text" name="username" value="">
    <input type="button" value="Echo Username" onclick="console.log(username.value)">
</form>
```

* (4)try/catch

```html
<input type="button" value="Click Me" onclick="try{showMessage();} catch(ex) {}">
```

### HTML 事件处理程序问题

在 HTML 中指定事件处理程序有一些问题：

* (1) 时机问题：有可能 HTML 元素已经显示在页面上，用户都与其交互了，而事件处理程序的代码还无法执行。
  为此，大多数 HTML 事件处理程序会封装在 `try/catch` 块中，以便在这种情况下静默失败
* (2) 对事件处理程序作用域链的扩展在不同浏览器中可能导致不同的结果。
  不同 JavaScript 引擎中标识符解析的规则存在差异，因此访问无限定的对象成员可能导致错误
* (3) HTML 与 JavaScript 强耦合。如果需要修改事件处理程序，则必须在两个地方，
  即 HTML 和 JavaScript 中，修改代码。这也是很多开发者不使用 HTML 事件处理程序，
  而使用 JavaScript 指定事件处理程序的主要原因

## DOM0 事件处理程序

在 JavaScript 中指定事件处理程序的传统方式是把一个函数赋值给(DOM元素的)一个事件处理程序属性。
这也是在第四代 Web 浏览器中开始支持的事件处理程序赋值方法，直到现在所有现代浏览器仍然都支持此方法，
主要原因是简单，要使用 JavaScript 指定事件处理程序，必须先取得要操作对象的引用

### 赋值事件处理程序属性一个函数

每个元素(包括 `window` 和 `document`)都有通常小写的事件处理程序属性，
比如：`onclick`，只要把这个属性赋值为一个函数即可

```js
let btn = document.getElementById("myBtn");

btn.onclick = function() {
    console.log("Clicked");
}
```

### 事件处理程序在元素的作用域中运行

像这样使用 DOM0 方式为事件处理程序赋值时，所赋函数被视为元素的方法，
因此，事件处理程序会在元素的作用域中运行，即 `this` 等于元素

在事件处理程序里通过 `this` 可以访问元素的任何属性和方法。
以这种方式添加事件处理程序是注册在事件流的冒泡阶段的

```js
let btn = document.getElementById("myBtn");
btn.onclick = function () {
    console.log(this.id); // "myBtn"
}
```

### 移除事件处理程序

通过将事件处理程序属性的值设置为 `null`，可以移除通过 DOM0 方式添加的事件处理程序，
把事件处理程序设置为 `null`，再点击按钮就不会执行任何操作了

```js
btn.onclick = null; // 移除事件处理程序
```

如果事件处理程序是在 HTML 中指定的，则 `onclick` 属性的值是一个包装相应 HTML 事件处理程序属性值的函数。
这些事件处理程序也可以通过在 JavaScript 中将相应 属性设置为 `null` 来移除

## DOM2 事件处理程序

### DOM2 事件处理程序方法

DOM2 Events 为事件处理程序的赋值和移除定义了两个方法:

* `addEventListener()`
* `removeEventListener()`

这两个方法暴露在所有 DOM 节点上，他们接收 3 个参数:

* 事件名
* 事件处理函数
* 一个布尔值
    - `true` 表示在捕获阶段调用事件处理程序
    - `false` 默认值，表示在冒泡阶段调用事件处理程序

大多数情况下，事件处理程序会被添加到事件流的冒泡阶段，主要原因是跨浏览器兼容性好。
把事件处理程序注册到捕获阶段通常用于在事件到达其指定目标之前拦截事件。
如果不需要拦截，则不要使用事件捕获

### addEventListener()

* 为按钮添加了会在时间冒泡阶段触发的 onclick 事件处理程序

```js
let btn = document.getElementById("myBtn");

btn.addEventListener("click", () => {
    console.log(this.id);
}, false);
```

* 与 DOM0 方式类似，这个事件处理程序同样被附加到元素的作用域中运行。
  使用 DOM2 方式的主要优势是可以为同一个事件添加多个事件处理程序。
  多个事件处理程序以添加的顺序来触发

```js
let btn = document.getElementById("myBtn");

btn.addEventListener("click", () => {
    console.log(this.id);
}, false);

btn.addEventListener("click", () => {
    console.log("Hello world!");
}, false);
```

### removeEventListener()

通过 `addEventListener()` 添加的事件处理程序，
只能使用 `removeEventListener()` 并传入与添加时同样的参数来移除。
这意味着使用 `addEventListener()` 添加的匿名函数无法移除

```js
let btn = document.getElementById("myBtn");
btn.addEventListener("click", () => {
    console.log(this.id);
}, false);

// 其他代码

btn.removeEventListener("click", function() {  // 没有效果
    console.log(this.id);
}, false);
```

传给 `removeEventListener()` 的事件处理函数必须与传给 `addEventListener()` 的是同一个

```js
let btn = document.getElementById("myBtn");

let handler = function() {
    console.log(this.id);
};

btn.addEventListener("click", handler, false);

// 其他代码

btn.removeEventListener("click", handler, false);  // 有效果
```

## IE 事件处理程序

IE 实现了与 DOM 类似的方法:

* `attachEvent()`
* `detachEvent()`

这两个方法接收两个同样的参数:

* 事件处理程序的名字
* 事件处理函数

### attachEvent()

因为 IE8 及更早的版本只支持事件冒泡，
所以使用 `attachEvent()` 添加的事件处理程序会添加到冒泡阶段

在 IE 中使用 `attachEvent()` 与使用 DOM0 方式的主要区别是事件处理程序的作用域。
使用 DOM0 方式时，事件处理程序中的 `this` 值等于目标元素，而使用 `attachEvent()` 时，
事件处理程序是在全局作用域中运行的，因此 `this` 等于 `window`。
理解这些差异对于编写跨浏览器代码时非常重要的

与使用 `addEventListener()` 一样，使用 `attachEvent()` 方法也可以给一个元素添加多个事件处理程序。
不过，与 DOM 方法不同，这里的事件处理程序会以添加它们的顺序方反向触发

* 示例：给按钮添加 click 事件处理程序

```js
var btn = document.getElementById("myBtn");

btn.attachEvent("onclick", function() {
    console.log("Clicked");
});
```

* 示例：`attachEvent()` 事件处理程序中的 `this` 等于 `window`

```js
var btn = document.getElementById("myBtn");

btn.attachEvent("onclick", function() {
    console.log(this === window);  // true
});
```

* 示例：添加多个事件处理程序

```js
var btn = document.getElementById("myBtn");

btn.attachEvent("onclick", function() {
    console.log("Clicked");
});
btn.attachEvent("onclick", function() {
    console.log("Hello world!");
});
```

### detachEvent()

与使用 DOM 方法类似，作为事件处理程序添加的匿名函数也无法移除。
但只要传给 `detachEvent()` 方法相同的函数引用，就可以移除

```js
var btn = document.getElementById("myBtn");

var handler = function() {
    console.log("Clicked");
};

btn.attachEvent("onclick", handler);

// 其他代码

btn.detachEvent("onclick", handler);
```

## 跨浏览器事件处理程序

为了以跨浏览器兼容的方式处理事件，很多开发者会选择使用一个 JavaScript 库，
其中抽象了不同浏览器的差异。有些开发者也可能会自己编写代码，以便使用最合适的事件处理手段。
自己编写跨浏览器事件处理代码也很简单，主要依赖能力检测。
要确保事件处理代码具有最大兼容性，只需要让代码在冒泡阶段运行即可

1. 首先创建一个 `addHanler()` 方法
    - 这个方法的任务是根据需要分别使用 DOM0 方式、DOM2 方式或 IE 方式来添加事件处理程序。
      这个方法会在 `EventUtil` 对象上添加一个方法，以实现跨浏览器事件处理
    - 添加的这个 `addHandler()` 方法接收 3 个参数
        - 目标元素
        - 事件名
        - 事件处理函数
2. 还要写一个也接收同样的 3 个参数的 `removeHandler()`
    - 这个方法的任务是移除之前添加的事件处理程序，不管是通过何种方式添加的，默认为 DOM0 方式

```js
var EventUtil = {
    addHandler: function(element, type, handler) {
        if (element.addEventListener) {
            element.addEventListener(type, handler, false);
        } else if (element.attachEvent) {
            element.attachEvent("on" + type, handler);
        } else {
            element["on" + type] = handler;
        }
    },

    removeHandler: function(element, type, handler) {
        if (element.removeEventListener) {
            element.removeEventListener(type, handler, false);
        } else if (element.detachEvent) {
            element.detachEvent("on" + type, handler);
        } else {
            element["on" + type] = null;
        }
    }
};
```

这里的 `addHandler()` 和 `removeHandler()` 方法并没有解决所有跨浏览器一致性问题，
比如 IE 的作用域问题、多个事件处理程序执行顺序问题等。
不过，这两个方法已经实现了跨浏览器添加和移除事件处理程序。
另外也要注意，DOM0 只支持给一个事件添加一个处理程序。
好在 DOM0 浏览器已经很少有人使用了，所以影响应该不大

```js
let btn = document.getElementById("myBtn");

let handler = function() {
    console.log("Clicked");
};

EventUtil.addHandler(btn, "click", handler);

// 其他代码

EventUtil.removeHandler(btn, "click", handler);
```

# 事件对象

在 DOM 中发生事件时，所有相关信息都会被收集并存储在一个名为 `event` 的对象中。
这个对象包含了一些基本信息，比如导致事件的元素、发生的事件类型，以及可能与特定事件相关的任何其他数据。
例如，鼠标操作导致的事件会生成鼠标位置信息，而键盘操作导致的事件会生成与被按下的键有关的信息。
所有浏览器都支持这个 `event` 对象，尽管支持方式不同

## DOM 事件对象

### DOM event 事件对象

在 DOM 合规的浏览器中，`event` 对象是传给事件处理程序的唯一参数。
不管以哪种方式，(DOM0 或 DOM2)指定事件处理程序，都会传入这个 `event` 对象

* 示例

```js
let btn = document.getElementById("myBtn");

btn.onclick = function(event) {
    console.log(event.type);  // "click"
};

btn.addEventListener("click", (event) => {
    console.log(event.type);  // "click"
}, false);
```

### HTML 属性事件处理程序使用 event 引用事件对象

在通过 HTML 属性指定的事件处理程序中，同样可以使用变量 `event` 引用事件对象。
已这种方式提供 `event` 对象，可以让 HTML 属性中的代码实现与 JavaScript 函数同样的功能

* 示例

```html
<input type="button" value="Click Me" onclick="console.log(event.type)">
```

### 事件对象的公共属性和方法

事件对象包含与特定事件相关的属性和方法。不同的事件生成的事件对象也会包含不同的属性和方法。
不过，所有事件对象都会包含一些公共属性和方法

|属性/方法                   |类型          |读/写 |说明                                                                              |
|---------------------------|-------------|-----|----------------------------------------------------------------------------------|
|bubbles                    |布尔值        |只读  |表示事件是否冒泡                                                                    |
|cancelable                 |布尔值        |只读  |表示是否可以取消事件的默认行为                                                         |
|currentTarget              |元素          |只读  |当前事件处理程序所在的元素                                                            |
|defaultPrevented           |布尔值        |只读  |ture 表示已经调用 preventDefault() 方法(DOM3 Events 中新增)                           |
|detail                     |整数          |只读  |事件相关的其他信息                                                                   |
|eventPhase                 |整数          |只读  |表示调用事件处理程序的阶段：1 代表捕获阶段，2 代表到达目标，3 代表冒泡阶段                    |
|preventDefault()           |函数          |只读  |用于取消事件的默认行为。只有 cancelable 为 true 才可以调用这个方法                         |
|stopImmediatePropagation() |函数          |只读  |于取消所有后续事件捕获或事件冒泡，并阻止调用任何后续事件处理程序(DOM3 Events 中新增)           |
|stopPropagation()          |函数          |只读  |用于取消所有后续事件捕获或事件冒泡。只有 bubbles为 true 才可以调用这个方法                   |
|target                     |元素          |只读  |事件目标                                                                            |
|trusted                    |布尔值        |只读  |true 表示事件是由浏览器生成的。false 表示事件是开发者通过 JavaScript 创建的(DOM3 Event 新增) |
|type                       |字符串        |只读  |被触发的事件类型                                                                      |
|View                       |AbstractView |只读  |与事件相关的抽象视图。等于事件所发生的 window 对象                                        |


* 在事件处理程序内部，`this` 对象始终等于 `currentTarget` 的值，而 `target` 只包含事件的实际目标。
  如果时间处理程序直接添加在了意图的目标，则 `this`、`currentTarget` 和 `target` 的值是一样的

```js
let btn = document.getElementById("myBtn");

btn.onclick = function(event) {
    console.log(event.currentTarget === this);  // true
    console.log(event.target === this);  // true
};
```

```js
document.body.onclick = function(event) {
    console.log(event.currentTarget === document.body);  // true
    console.log(this === document.body);  // true
    console.log(event.target === document.getElementById("myBtn"));  // true
};
```

* `type` 属性在一个处理程序处理多个事件时很有用

```js
let btn = document.getElementById("myBtn");

let handler = function(event) {
    
}
```





## IE 事件对象

### IE event 事件对象

与 DOM 事件对象不同，IE 事件对象可以基于事件处理程序被指定的方式以不同方式来访问

* 如果事件处理程序是使用 DOM0 方式指定的，则 `event` 对象只是 `window` 对象的一个属性

```js
var btn = document.getElementById("myBtn");

btn.onclick = function() {
    let event = window.event;
    console.log(event.type);  // "click" 事件类型
};
```

* 如果事件处理程序是使用 `attachEvent()` 指定的，则 `event` 对象会作为唯一的参数传给处理函数。
  `event` 对象仍然是 `window` 对象的属性，只是出于方便也将其作为参数传入

```js
var btn = document.getElementById("myBtn");

btn.attachEvent("onclick", function(event) {
    console.log(event.type);  // "click"
});
```

### HTML 属性事件处理程序使用 event 引用事件对象

如果是使用 HTML 属性方式指定的事件处理程序，则 `event` 对象同样可以通过变量 `event` 访问

```html
<input type="button" value="Click me" onclick="console.log(event.type)">
```

### 事件对象的公共属性和方法

IE 事件对象也包含与导致其创建的特定事件相关的属性和方法，
其中很多都与相关的 DOM 属性和方法对应

与 DOM 事件对象一样，基于触发的事件类型不同，`event` 对象中包含的属性和方法也不一样。
不过，所有 IE 事件对象都会包含下表所列的公共属性和方法

|属性/方法     |类型    |读/写 |说明                                                                         |
|-------------|-------|-----|-----------------------------------------------------------------------------|
|cancelBubble |布尔值  |读/写 |默认为 false，设置为 true 可以取消冒泡(与 DOM 的 stopPropagation() 方法相同)       |
|returnValue  |布尔值  |读/写 |默认为 true，设置为 false 可以取消事件默认行为(与 DOM 的 preventDefault() 方法相同) |
|srcElement   |元素    |只读  |事件目标(与 DOM 的 target 属性相同)                                             |
|type         |字符串  |只读  |触发的事件类型                                                                 |

* 由于事件处理程序的作用域取决于指定它的方式，因此 `this` 值并不总是等于事件目标。
  为此，更好的方式是使用事件对象的 `scrElement` 属性代替 `this`

```js
var btn = document.getElementById("myBtn");

btn.onclick = function() {
    console.log(window.event.srcElement === this);  // true
};

btn.attachEvent("onclick", function(event)) {
    console.log(event.srcElement === this);  // false
}
```

* `returnValue` 属性等价于 DOM 的 `preventDefault()` 方法，
  都是用于取消给定事件默认的行为

```js
var link = document.getElemenById("myLink");

link.onclick = function() {
    window.event.returnValue = false;
};
```

* `cancelBubble` 属性与 DOM `stopPropagation()` 方法用途一样，都可以阻止事件冒泡。
  因为 IE8 及更早版本不支持捕获阶段，所以只会取消冒泡

```js
var btn = document.getElementById("myBtn");

btn.onclick = function() {
    console.log("Clicked");
    window.event.cancelBubble = true;
};

document.body.onclick = function() {
    console.log("Body clicked");
};
```




## 跨浏览器事件对象

虽然 DOM 和 IE 的事件对象并不相同，但他们有足够的相似性可以实现跨浏览器方案。
DOM 事件对象中包含 IE 事件对象的所有信息和能力，只是形式不同。
这些共性可让两种事件模型之间的映射称为可能


```js
var EventUtil = {
    addHandler: function(element, type, handler) {
        if (element.addEventListener) {
            element.addEventListener(type, handler, false);
        } else if (element.attachEvent) {
            element.attachEvent("on" + type, handler);
        } else {
            element["on" + type] = handler;
        }
    },

    getEvent: function(event) {
        return event ? event : window.event;
    },

    getTarget: function(event) {
        return event.target || event.srcElement;
    },

    preventDefault: function(event) {
        if (event.preventDefault) {
            event.preventDefault();
        } else {
            event.returnValue = false;
        }
    },

    removeHandler: function(element, type, handler) {
        if (element.removeEventListener) {
            element.removeEventListener(type, handler, false);
        } else if (element.detachEvent) {
            element.detachEvent("on" + type, handler);
        } else {
            element["on" + type] = null;
        }
    },

    stopPropagation: function(event) {
        if (event.stopPropagation) {
            event.stopPropagation();
        } else {
            event.cancelBubble = true;
        }
    }
};
```

这里一共给 `EventUtil` 增加了 4 个新方法：

* 方法 `getEvent()` 返回对 `event` 对象的引用。
  IE 中事件对象的位置不同，而使用这个方法可以不用管事件处理程序是如何指定的，
  都可以获取到 `event` 对象。使用这个方法的前提是，事件处理程序必须接收 `event` 对象，
  并把它传给这个方法

```js
btn.onclick = function(event) {
    event = EventUtil.getEvent(event);
};
```

* 方法 `getTarget()` 返回事件目标。在这个方法中

```js
btn.onclick = function(event) {
    event = EventUtil.getEvent(event) {
        let target = EventUtil.getTarget(event);
    }
};
```

* 方法 `preventDefault()` 用于阻止事件的默认行为

```js
let link = document.getElementById("myLink");

link.onclick = function(event) {
    event = EventUtil.getEvent(event);
    EventUtil.preventDefault(event);
};
```

* 方法 `stopPropagation()` 用于检测用于停止事件流的 DOM 方法，
  如果没有再使用 `cancelBubble` 属性

```js
let btn = document.getElementById("myBtn");

btn.onclick = function(event) {
    console.log("Clicked");
    event = EventUtil.getEvent(event);
    EventUtil.stopPropagation(event);
};

document.body.onclick = function(event) {
    console.log("Body clicked");
};
```


# 事件类型

Web 浏览器中可以发生很多事件。发生事件的类型决定了事件对象中会保存什么信息

DOM3 Events 定义了如下事件类型:

* 用户界面事件(`UIEvent`): 涉及与 BOM 交互的通用浏览器事件
* 焦点事件(`FocusEvent`): 在元素获得和失去焦点时触发
* 鼠标事件(`MouseEvent`): 使用鼠标在页面上执行某些操作时触发
* 滚轮事件(`WheelEvent`): 使用鼠标滚轮(或类似设备)时触发
* 输入事件(`InputEvent`): 向文档中输入文本时触发
* 键盘事件(`KeyboardEvent`): 使用键盘在页面上执行某些操作时触发
* 合成事件(`CompositionEvent`): 在使用某种(Input Method Editor，输入法编辑器)输入字符时触发

HTML5 定义了专有事件:

* 除了这些事件类型之外，HTML5 还定义了另一组事件，而浏览器通常在 DOM 和 BOM 上实现专有事件。
  这些专有事件基本上都是根据开发者需求而不是按照规范增加的，因此不同浏览器的实现可能不同

DOM3 Events 在 DOM2 Events 基础上重新定义了事件，并增加了新的事件类型。
所有主流浏览器都支持 DOM2 Events 和 DOM3 Events

## 用户界面事件

用户界面事件或 UI 事件不一定跟用户操作有关。这类事件在 DOM 规范出现之前就已经以某种形式存在了，
保留了它们是为了向后兼容。UI 事件主要有以下几种:

* `DOMActivate`: 元素被用户通过鼠标或键盘操作激活时触发(比 `click 或 keydown 更通用`)。
  这个事件在 DOM3 Events 中已经废弃。因此浏览器实现之间存在差异，所以不要使用它
* `load`: 
    - 在 window 上当页面加载完成后触发
    - 在窗套(`<frameset>`)上当所有窗格(`<frame>`)都加载完成后触发
    - 在 `<img>` 元素上当图片加载完成后触发
* `unload`:
    - 在 window 上当页面完全卸载后触发
    - 在窗套(`<frameset>`)上当所有窗格(`<frame>`)都卸载完成后触发
    - 在 `<object>` 元素上当相应对象卸载完成后触发
* `abort`: 在 `<object>` 元素上当相应对象加载完成前被用户提前终止下载时触发
* `error`: 
    - 在 window 上当 JavaScript 报错时触发
    - 在 `<img>` 元素上当无法加载指定图片时触发，
    - 在 `<object>` 元素上当无法加载相应对象时触发
    - 在窗套上当一个或多个窗格无法完成加载时触发
* `select`: 在文本框(`<input>` 或 `textarea`)上当用户选择了一个或多个字符时触发
* `resize`: 在 window 或窗格上当窗口或窗格被缩放时触发
* `scroll`: 
    - 当用户滚动包含滚动条的元素时在元素上触发
    - `<body>` 元素包含已加载页面的滚动条

大多数 HTML 事件与 `window` 对象和表单控件有关。除了 `DOMActivate`，
这些事件在 DOM2 Events 中都被归为 HTML Events(`DOMActivate` 在 DOM2 中仍旧是 UI 事件)

### load 事件

`load` 事件可能是 JavaScript 中最常用的事件。在 `window` 对象上，
`load` 事件会在整个页面(包括外部资源如图片、JavaScript 文件和 CSS 文件)加载完成后触发。
可以通过两种方式指定 `load` 事件处理程序

* JavaScript 方式

```js
window.addEventListener("load", (event) => {
    console.log("Loaded!");
});
```

* 指定 `load` 事件处理程序的方式是向 `<body>` 元素添加 `onload` 属性

```html
<!DOCTYPE html>
<html>
    <head>
        <title>Load Event Example</title>
    </head>
    <body onload="console.log('Loaded!')">
    
    </body>
</html>
```

### unload 事件


### resize 事件


### scroll 事件

## 焦点事件

焦点事件在页面元素获得或失去焦点时触发。
这些事件可以与 `document.hasFocus()` 和 `document.activeElement` 一起为开发者提供用户在页面中导航的信息。
焦点事件有以下 6 种:

* blur
* DOMFocusIn
* DOMFocusOut
* focus
* focusin
* focusout
  
当焦点从页面中的一个元素移到另一个元素上时，会一次发生如下事件:

* focusout 在失去焦点的元素上触发
* focusin 在获得焦点的元素上触发
* blur 在失去焦点的元素上触发
* DOMFocusOut 在失去焦点的元素上触发
* focus 在获得焦点的元素上触发
* DOMFocusIn 在获得焦点的元素上触发


## 鼠标和滚轮事件

鼠标事件是 Web 开发中最常用的一组事件，这是因为鼠标是用户的主要定位设备。
DOM3 Events 定义了 9 种鼠标事件:

* click
* dblclick
* mousedwon
* mouseenter
* mouseleave
* mousemove
* mouseout
* mouseover
* mouseup


## 键盘与输入事件

键盘事件是用户操作键盘时触发的。DOM2 Events 最初定义了键盘事件，但该规范在最终发布前删除了相应内容。
因此，键盘事件很大程度上是基于原始的 DOM0 实现的

DOM3 Events 为键盘事件提供了一个首先再 IE9 中完全实现的规范。
其他浏览器也开始实现改规范，但仍然存在很多遗留的实现

键盘事件包含 3 个事件:

* keydown
* keypress
* textInput
* keyup

## 合成事件

合成事件是 DOM3 Events 中新增的，用于处理通常使用 IME 输入时的复杂输入序列。
IME 可以让用户输入物理键盘上没有的字符。例如，使用拉丁字母键盘的用户还可以使用 IME 输入日文。

IME 通常需要同时按下多个键才能输入一个字符。合成事件用于检测和控制这种输入。

合成事件在很多方面与输入事件很类似。
在合成事件触发时，事件目标是接收文本的输入字段。
唯一增加的事件属性是 `data`，其中包含的值视情况而异

与文本事件类似，合成事件可以用来在必要时过滤输入内容

合成事件有以下 3 种:

* compositionstart
    - 在 `compositionstart` 事件中，包含正在编辑的文本(例如，已经选择了文本但还没替换)
* compositionupdate
    - 在 `compositionupdate` 事件中，包含要插入的新字符
* compositionend
    - 在 `compositionend` 事件中，包含本次合成过程中输入的全部内容

```js
let textbox = document.getElementById("myText");

textbox.addEventListener("compositionstart", (event) => {
    console.log(event.data);
});

textbox.addEventListener("compositionupdate", (event) => {
    console.log(event.data);
});

textbox.addEventListener("compositionend", (event) => {
    console.log(event.data);
})
```


## 变化事件

DOM2 的变化事件(Mutation Events)是为了在 DOM 发生变化时提供通知

这些事件已经被废弃，浏览器已经在有计划地停止对它们的支持。
变化事件已经被 Mutation Observers 所取代

## HTML5 事件

DOM 规范并未涵盖浏览器都支持的所有事件。很多浏览器根据特定的用户需求或使用场景实现了自定义事件。

HTML5 详尽地列出了浏览器支持的所有事件，这里列出 HTML5 中得到浏览器较好支持的一些事件。
注意这些并不是浏览器支持的所有事件

### contextmenu 事件



### beforeunload 事件



### DOMContentLoaded 事件


### readystatechange 事件


### pageshow 与 pagehide 事件


### hashchange 事件


## 设备事件


随着只能手机和平板计算机的出现，用户与浏览器交互的新方式应运而生。
为此，一批新事件别发明出来

设备事件可以用于确定用户使用设备的方式。W3C 在 2011 年就开始起草一份新规范，
用于定义新设备及设备相关的事件

### orientationchange 事件

### deviceorientation 事件

### devicemotion 事件


## 触摸及手势事件

Safari 为 iOS 定制了一些专有事件，以方便开发者。因为 iOS 设备没有鼠标和键盘，
所以常规的鼠标和键盘事件不足以创建具有完整交互能力的网页

同时，WebKit 也为 Android 定制了很多专有事件，成为了事实标准，
并被纳入 W3C 的 Touch Events 规范

### 触摸事件

iPhone 3G 发布时，iOS 2.0 内置了新版本的 Safari。
这个新的移动 Safari 支持一些与触摸交互有关的新事件。
后来的 Android 浏览器也实现了同样的事件。
当手指放在屏幕上、在屏幕上滑动或从屏幕移开时，触摸事件即会触发

触摸事件有如下几种:

* `touchstart`
    - 手指放到屏幕上时触发(即使有一个手指已经放在了屏幕上)
* `touchmove`
    - 手指在屏幕上滑动时连续触发。在这个事件中调用 `preventDefault()` 可以阻止滚动
* `touchend`
    - 手指从屏幕上移开时触发
* `touchcancel`
    - 系统停止跟踪触摸时触发。文档中并未明确什么情况下停止跟踪

这些事件都会冒泡，也都可以被取消。尽管触摸事件不属于 DOM 规范，
但浏览器仍然以兼容 DOM 的方式实现了它们。
因此，每个触摸事件的 event 对象都提供了鼠标事件的公共属性: 

* `bubbles`
* `cancelable`
* `view`
* `clientX`
* `clientY`、
* `screenX`
* `screenY`
* `detail`
* `altKey`
* `shiftKey`
* `ctrlKey`
* `metaKey`

除了这些公共的 DOM 属性，触摸事件还提供了以下 3 个属性用于跟踪触点:

* touches
* targetTouches
* changedTouches

每个 `Touch` 对象都包含下列属性:

* clientX
    - 触点在视口中的 x 坐标
* clientY
    - 触点在视口中的 y 坐标
* identifier
    - 触点 ID
* pageX
    - 触点在页面上的 x 坐标
* pageY
    - 触点在页面上的 y 坐标
* screenX
    - 触点在屏幕上的 x 坐标
* screenY
    - 触点在屏幕上的 y 坐标
* target
    - 触摸事件的事件目标


### 手势事件

iOS 2.0 中的 Safari 还增加了一种手势事件。
手势事件会在两个手指触碰屏幕且相对距离或旋转角度变化时触发

只有在两个手指同时接触事件接收者时，这些事件才会触发。在一个元素上设置事件处理程序，
意味着两个手指必须都在元素边界以内才能触发手势事件(这个元素就是事件目标)。
因为这些事件会冒泡，所以也可以把事件处理程序放到文档级别，从而可以处理所有手势事件。
使用这种方式时，事件的目标就是两个手指均位于其边界内的元素

手势事件有以下 3 种:

* `gesturestart`
    - 一个手指已经放在屏幕上，再把另一个手指放到屏幕上时触发
* `gesturechange`
    - 任何一个手指在屏幕上的位置发生变化时触发
* `gestureend`
    - 其中一个手指离开屏幕时触发

### 触摸事件与手势事件的联系

触摸


## 事件参考

- [ ] TODO

# 内存与性能

因为事件处理程序在现代 Web 应用中可以实现交互，所以很多开发者会错误地在页面中大量使用它们。
在创建 GUI 的语言如 C# 中，通常会给 GUI 上的每个按钮设置一个 `onclick` 事件处理程序。
这样做不会有什么性能损耗。

在 JavaScript 中，页面中事件处理程序的数量与页面整体性能直接相关。
原因有很多

* 首先，每个函数都是对象，都占用内存空间

## 事件委托




## 删除事件处理程序







# 模拟事件

## DOM 事件模拟

### 模拟鼠标事件


### 模拟键盘事件

### 模拟其他事件


### 自定义 DOM 事件


## IE 事件模拟





