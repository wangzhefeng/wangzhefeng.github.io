---
title: BOM
author: 王哲峰
date: '2020-10-01'
slug: js-BOM
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
  - [BOM](#bom)
  - [客户端检测](#客户端检测)
- [JavaScript 浏览器简介](#javascript-浏览器简介)
- [BOM](#bom-1)
  - [window 对象](#window-对象)
    - [Global 作用域](#global-作用域)
    - [窗口关系](#窗口关系)
    - [窗口位置](#窗口位置)
    - [像素比](#像素比)
    - [窗口大小](#窗口大小)
    - [视口位置](#视口位置)
    - [导航与打开新窗口](#导航与打开新窗口)
    - [定时器](#定时器)
    - [系统对话框](#系统对话框)
    - [导航与打开新窗口](#导航与打开新窗口-1)
    - [定时器](#定时器-1)
    - [系统对话框](#系统对话框-1)
  - [location](#location)
  - [navigator](#navigator)
  - [screen](#screen)
  - [history](#history)
- [客户端检测](#客户端检测-1)
  - [能力检测](#能力检测)
  - [用户代理检测](#用户代理检测)
  - [软件与硬件检测](#软件与硬件检测)
</p></details><p></p>

# TODO

- [ ] 客户端检测

# 概览

## BOM

浏览器对象模型(BOM, Browser Object Model)是以 `window` 对象为基础的，
这个对象代表了浏览器窗口和页面可见的区域。

`window` 对象也被复用为 ECMAScript 的 `Global` 的对象，因此所有全局变量和函数都是它的属性，
而且所有原生类型的构造函数和普通函数也都从一开始就存在于这个对象之上。

* 要引用其他 `window` 对象，可以使用几个不同的窗口指针
* 通过 `location` 对象可以以编程的方式操纵浏览器的导航系统。
  通过设置这个对象的属性，可以改变浏览器 `URL` 中的某一部分或全部
* 使用 `replace()` 方法可以替换浏览器历史记录中当前显示的页面，并导航到新 URL
* `navigator` 对象提供关于浏览器的信息。提供的信息类型取决于浏览器，不过有些属性如 `userAgent` 是所有浏览器都支持的

## 客户端检测

客户端检测是 JavaScript 中争议最多的话题之一。因为不同浏览器之间存在差异，
所以经常需要根 据浏览器的能力来编写不同的代码。客户端检测有不少方式，但下面两种用得最多。

* 能力检测，在使用之前先测试浏览器的特定能力。例如，脚本可以在调用某个函数之前先检查它是否存在。
  这种客户端检测方式可以让开发者不必考虑特定的浏览器或版本，而只需关注某些能力是否存在。
  能力检测不能精确地反映特定的浏览器或版本
* 用户代理检测，通过用户代理字符串确定浏览器。用户代理字符串包含关于浏览器的很多信息，
  通常包括浏览器、平台、操作系统和浏览器版本。用户代理字符串有一个相当长的发展史，
  很多浏览器都试图欺骗网站相信自己是别的浏览器。用户代理检测也比较麻烦，
  特别是涉及 Opera 会在代理字符串中隐藏自己信息的时候。即使如此，
  用户代理字符串也可以用来确定浏览器使 用的渲染引擎以及平台，包括移动设备和游戏机。

在选择客户端检测方法时，首选是使用能力检测。特殊能力检测要放在次要位置，
作为决定代码逻辑的参考。用户代理检测是最后一个选择，因为它过于依赖用户代理字符串

浏览器也提供了一些软件和硬件相关的信息。这些信息通过 `screen` 和 `navigator` 对象暴露出来。
利用这些 API，可以获取关于操作系统、浏览器、硬件、设备位置、电池状态等方面的准确信息。


# JavaScript 浏览器简介

目前主流的浏览器分这么几种: 

- IE 6~11: 国内用得最多的 IE 浏览器, 历来对 W3C 标准支持差。从 IE10 开始支持 ES6 标准
- Chrome: Google 出品的基于 Webkit 内核浏览器, 内置了非常强悍的 JavaScript 引擎——V8。
  由于 Chrome 一经安装就时刻保持自升级, 所以不用管它的版本, 最新版早就支持 ES6 了
- Safari: Apple 的 Mac 系统自带的基于 Webkit 内核的浏览器, 从 OS X 10.7 Lion 自带的 6.1 版本开始支持 ES6, 
  目前最新的 OS X 10.11 El Capitan 自带的 Safari 版本是 9.x, 早已支持 ES6
- Firefox: Mozilla 自己研制的 Gecko 内核和 JavaScript 引擎 OdinMonkey。早期的 Firefox 按版本发布, 
  后来终于聪明地学习 Chrome 的做法进行自升级, 时刻保持最新
- 移动设备上目前 iOS 和 Android 两大阵营分别主要使用 Apple 的 Safari 和 Google 的 Chrome, 由于两者都是 Webkit 核心, 
  结果 HTML5 首先在手机上全面普及(桌面绝对是 Microsoft 拖了后腿), 对 JavaScript 的标准支持也很好, 最新版本均支持 ES6
- 其他浏览器如 Opera 等由于市场份额太小就被自动忽略了
- 另外还要注意识别各种国产浏览器, 如某某安全浏览器, 某某旋风浏览器, 它们只是做了一个壳, 其核心调用的是 IE, 
  也有号称同时支持 IE 和 Webkit 的“双核”浏览器

<div class="warning" style='background-color:#E9D8FD; color: #69337A; border-left: solid #805AD5 4px; border-radius: 4px; padding:0.7em;'>
    <span>
        <p style='margin-top:1em; text-align:left'>
            <b>Note</b>
        </p>
        <p style='margin-left:1em;'>
            <ul>
               <li>
                  不同的浏览器对 JavaScript 支持的差异主要是, 有些 API 的接口不一样, 比如 AJAX, File 接口。
                  对于 ES6 标准, 不同的浏览器对各个特性支持也不一样
               </li>
               <li>
                  在编写 JavaScript 的时候, 就要充分考虑到浏览器的差异, 尽量让同一份 JavaScript 代码能运行在不同的浏览器中
               </li>
            </ul>
        </p>
        <p style='margin-bottom:1em; margin-right:1em; text-align:right; font-family:Georgia'> 
            <b></b> 
            <i></i>
        </p>
    </span>
</div>

JavaScript 可以获取浏览器提供的很多对象, 并进行操作:

* window
* navigator
* screen
* location
* document
* history

# BOM

虽然 EMCAScript 把 BOM 描述为 JavaScript 的核心，
但实际上 BOM 是使用 JavaScript 开发 Web 应用程序的核心。
BOM 提供了与网页无关的浏览器功能对象

## window 对象

BOM 的核心是 `window` 对象，表示浏览器的实例。`window` 对象在浏览器中有两重身份:

* ECMAScript 中的 `Global` 对象
    - 网页中定义的所有对象、变量和函数都以 `window` 作为其 `Global` 对象
* 浏览器窗口的 JavaScript 接口
    - 都可以访问其上定义的全局方法

### Global 作用域

因为 `window` 对象被复用为 ECMAScript 的 `Global` 对象，
所以通过 `var` 声明的所有全局变量和函数都会变成 `window` 对象的属性和方法。
使用 `let` 或 `const` 则不会把变量添加给全局对象

```js
var age = 29;
var sayAge = () => { 
    // 因为 sayAge 存在于全局作用域，this.age 映射到 window.age
    alert(this.age);
};
alert(window.age);  // 29
window.sayAge();    // 29


let age2 = 29;
const sayAge2 = () => {
    alert(this.age);
}
alert(window.age2);  // undefined
sayAge2();           // undefined
window.sayAge2();    // TypeError: window.sayAge is not a function
```

* 访问未声明的变量会抛出错误，但是可以在 `window` 对象上查询是否存在可能未声明的变量

```js
var newValue = oldValue;  // 报错

var newValue = window.oldValue;
console.log(newValue);  // undefined
```

* JavaScript 中很多对象都暴露在全局作用域中，比如 `location` 和 `navigator`，
  因而它们也是 `window` 对象的属性

### 窗口关系

* `window`
    - `window.self`
        - `self` 对象是终极 `window` 属性，始终会指向 `window`，实际上，`self` 和 `window` 就是同一个对象。
          之所以还要暴露 `self`，就是为了和 `top`、`parent` 保持一致 
    - `window.top`
        - `top` 对象始终指向最上层(最外层)窗口，即浏览器本身
    - `window.parent`
        - `parent` 对象始终指向当前窗口的父窗口。如果当前窗口是最上层窗口，则 `parent` 等于 `top`(都等于 `window`)
        - 可以把多个窗口的 `window` 对象串联起来，比如 `window.parent.parent`

### 窗口位置

`window` 对象的位置可以通过不同的属性和方法来确定

* 现代浏览器提供了 `screenLeft` 和 `screenTop` 属性，
  用于表示窗口相对于屏幕左侧和顶部的位置，返回值的单位是 CSS 像素
* 使用 `moveTo()` 和 `moveBy()` 方法可以移动窗口。这两个方法都接收两个参数
    - `moveTo()` 接收要移动的新位置的绝对坐标 `$x$` 和 `$y$`
    - `moveBy()` 接收相对当前位置在两个方向上移动的像素数

```js
// 把窗口移动到左上角
window.moveTo(0, 0);

// 把窗口向下移动 100 像素
window.moveBy(0, 100);

// 把窗口移动到坐标位置 (200, 300)
window.moveTo(200, 300);

// 把窗口向左移动 50 像素
window.moveBy(-50, 0);
```

### 像素比

CSS 像素是 Web 开发中使用的统一像素单位。这个单位的背后其实是一个角度: 0.0213°。
如果屏 幕距离人眼是一臂长，则以这个角度计算的 CSS 像素大小约为 1/96 英寸。
这样定义像素大小是为了在 不同设备上统一标准。比如，低分辨率平板设备上 
12 像素(CSS 像素)的文字应该与高清 4K 屏幕下 12 像素(CSS 像素)的文字具有相同大小。
这就带来了一个问题，不同像素密度的屏幕下就会有不同的 缩放系数，
以便把物理像素(屏幕实际的分辨率)转换为 CSS 像素(浏览器报告的虚拟分辨率)

举个例子，手机屏幕的物理分辨率可能是 1920×1080，但因为其像素可能非常小，
所以浏览器就需 要将其分辨率降为较低的逻辑分辨率，比如 640×320。
这个物理像素与 CSS 像素之间的转换比率由 5 `window.devicePixelRatio` 属性提供。
对于分辨率从 1920×1080 转换为 640×320 的设备，`window.devicePixelRatio` 的值就是 3。
这样一来，12 像素(CSS 像素)的文字实际上就会用 36 像素的物理像素来显示

物理像素与 CSS 像素之间的转换比率由 `window.devicePixelRatio` 属性提供。
`window.devicePixelRatio` 实际上与每英寸像素数(DPI，dots per inch)是对应的。
DPI 表示单位像素密度，而 `window.devicePixelRatio` 表示物理像素与逻辑像素之间的缩放系数

### 窗口大小

为了在不同的浏览器中确定浏览器窗口的大小，所有现代浏览器都支持 4 个属性:

* 在桌面或移动设备，返回浏览器窗口中页面视口的大小，视口即：也就是屏幕上页面可视区域的大小，不包含浏览器边框和工具栏
    - `window.innerWidth`
    - `window.innerHeight`
* 返回浏览器窗口自身的大小，不管是在最外层 `window` 上使用，还是在窗格 `<frame>` 中使用
    - `window.outerWidth`
    - `window.outerHeight`

因为桌面浏览器的差异，所以需要先确定用户是不是在使用移动设备，然后再决定使用哪个属性：

* 布局视口是相对于可见视口的概念，布局视口表示渲染页面的实际大小，可见视口只能显示整个页面的一小部分
  - Mobile Internet Explorer 把可见视口的信息保存在 `document.documentElement.clientWidth` 和 `document.documentElement.clientHeight` 中
    在放大或缩小页面时，这些值也会相应变化
  - Mobile Internet Explorer 把布局视口的信息保存在 `document.body.clientWidth` 和 `document.body.clientHeight` 中
    在放大或缩小页面时，这些值也会相应变化

```js
let pageWidth = window.innerWidth;
let pageHeight = window.innerHeight;

if (typeof pageWidth != "number") {
    if (document.compatMode == "CSS1Compat") {
        pageWidth = document.documentElement.clientWidth;
        pageHeight = document.documentElement.clientHeight;
    } else {
        pageWidth = document.body.clientWidth;
        pageHeight = document.body.clientHeight;
    }
}
```

与移动窗口的方法一样，缩放窗口的方法可能会被浏览器禁用，而且在某些浏览器中默认是禁用的。 
同样，缩放窗口的方法只能应用到最上层的 window 对象。
要调整窗口的大小，可以使用如下两个方法，这个两个方法都接收两个参数：

* `resizeTo()`
    - 新的宽度
    - 新的高度
* `resizeBy()`
    - 宽度要缩放多少
    - 高度要缩放多少

```js
// 缩放到 100x100
window.resizeTo(100, 100);

// 缩放到 200x150
window.resizeBy(100, 50)


// 缩放到 300x300
window.resizeTo(300, 300);
```

### 视口位置


### 导航与打开新窗口


### 定时器


### 系统对话框


### 导航与打开新窗口

- window 对象的 open 方法来创建新的浏览器窗口

```js
window.open(url, name, features)
```

```js
// 调用 popUp 函数的一个办法是使用 伪协议(pseudo-protocol)
function popUp(winURL) {
    window.open(winURL, "popup", "width=320,height=480");
}
```

### 定时器



### 系统对话框











## location


## navigator


## screen


## history









# 客户端检测

## 能力检测


## 用户代理检测

## 软件与硬件检测

