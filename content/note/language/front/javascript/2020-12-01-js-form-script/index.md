---
title: 表单脚本
author: 王哲峰
date: '2020-12-01'
slug: js-form-script
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

- [基本内容](#基本内容)
- [表单基础](#表单基础)
  - [表示表单](#表示表单)
    - [HTML `<form>` 元素](#html-form-元素)
    - [JavaScript `HTMLFormElement` 类型](#javascript-htmlformelement-类型)
  - [引用表单](#引用表单)
    - [getElementById](#getelementbyid)
    - [document.forms](#documentforms)
  - [提交表单](#提交表单)
    - [提交按钮的创建方式](#提交按钮的创建方式)
    - [表单提交](#表单提交)
    - [阻止表单提交](#阻止表单提交)
  - [重置表单](#重置表单)
    - [重置按钮的创建方式](#重置按钮的创建方式)
    - [重置表单](#重置表单-1)
- [文本框编程](#文本框编程)
- [选择框编程](#选择框编程)
- [表单序列化](#表单序列化)
- [富文本编辑](#富文本编辑)
</p></details><p></p>

# 基本内容

* 表单基础
* 文本框验证与交互
* 使用其他表单控件

由于不能直接使用表单解决问题，因此开发者不得不使用 JavaScript 既做表单验证，又用于增强标准表单控件的默认行为

# 表单基础

## 表示表单

### HTML `<form>` 元素

HTML 表单元素 `<form>` 表示文档中的一个区域，此区域包含交互控件，用于向 Web 服务器提交信息

* `<form>` 元素的属性:
    - 全局属性 
        - `accept-charset`: 一个空格分隔或逗号分隔的列表，此列表包括了服务器支持的字符编码
        - `autocapitalize`: 这是一个被 iOS Safari 使用的非标准属性
        - `autocomplete`: 用于指示 input 元素是否能够拥有一个默认值，此默认值是由浏览器自动补全的。
          此设定可以被属于此表单的子元素的 autocomplete 属性覆盖
            - `on`
            - `off` 
        - `name`: 表单的名称。HTML4 中不推荐(应使用 id)。在 HTML5 中，
          该值必须是所有表单中独一无二的，而且不能是空字符串
        - `rel`: 根据 value 创建一个超链接或
    - 提交表单的属性
        - `action`: 处理表单提交的 URL, 可被覆盖
        - `enctype`: 当 method 属性值为 post 时，enctype 就是将表单的内容提交给服务器的 MIME 类型, 可被覆盖
            - application/x-www-form-urlencoded
            - multipart/form-data
            - text/plain
        - `method`: 浏览器使用这种 HTTP 方式来提交表单，可被覆盖
            - post
            - get
            - dialog
        - `novalidate`: 此布尔值属性表示提交表单时不需要验证表单
        - `target`: 表示在提交表单之后，在哪里显示响应信息，可被覆盖
            - _self
            - _blank
            - _parent
            - _top

```html
<form accept-charset="" 
      autocapitalize="" 
      autocomplete="on" 
      name="form1" rel="" 
      action="" 
      entype="" 
      method="post" 
      novalidate="" 
      target="">
    <label>Label name:
        <input name="" autocomplete="">
    </label>
    <input type="text" name="" id="" required>
    <input type="email" name="" id="" required>
    <input type="radio" name="" id="">
    <input type="submit" value="Submit Form">
    <input type="image" src="images/graphic.gif">
    <button type="submit">Submit Form</button>
    <fieldset>
        <legend>Title</legend>
        <label>
            <input type="radio" name="rdio"> Select me
        </label>
    </fieldset>
</form>
```

```
Note:

- 属性被覆盖的情况指，属性值可被 <button>、<input type="submit">、
  <input type="image"> 元素中的 formmethod 属性覆盖
```

### JavaScript `HTMLFormElement` 类型

`HTMLFormElement` 类继承自 `HTMLELement` 类，因此拥有与其他 HTML 元素一样的默认属性。
不过，`HTMLFormElement` 也有自己的属性和方法

* `acceptCharset`: 服务器可以接收的字符集，等价于 HTML 的 `accept-charset` 属性
* `action`: 请求的 URL，等价于 HTML 的 `action` 属性
* `elements`: 表单中所有控件的 `HTMLCollection`
* `enctype`: 请求的编码类型，等价于 HTML 的 `enctype` 属性
* `length`: 表单中控件的数量
* `method`: HTTP 请求的方法类型，通常是 `get` 或 post，等价于 HTML 的 `method` 属性
* `name`: 表单的名字，等价于 HTML 的 `name` 属性
* `reset()`: 把表单字段重置为各自的默认值
* `submit()`: 提交表单
* `target`: 用于发送请求和接收响应的窗口的名字，等价于 HTML 的 `target` 属性

## 引用表单

### getElementById

* 将表单当做普通元素为它指定一个 `id` 属性，从而可以使用 `getElementById()` 来获取表单

```html
<form id="form1" name="form-example"> <!-- 表单可以同时拥有 id 和 name，而且两者可以不相同 -->
    <!-- 内容 -->
</form>
```

```js
let form = document.getElementById("form1");
```

### document.forms

* 使用 `document.forms` 集合可以获取页面上所有的表单元素。
  然后，可以进一步使用数字索引或表单的名字(`name`)来访问特定的表单

```html
<form id="form1">
    <!-- 内容 -->
</form>

<form id="form2">
    <!-- 内容 -->
</form>
```

```js
// 取得页面中的第一个表单
let firstForm = document.forms[0];

// 取得名字为 "form2" 的表单
let myForm = document.forms["form2"];
```

## 提交表单

表单是通过用户点击提交按钮或图片按钮的方式提交的。
如果表单中有上述任何一个按钮，且焦点在表单中某个控件上，
则按回车键也可以提交表单。(textarea 控件是个例外，当焦点在它上面时，
按回车键会换行)，注意，没有提交按钮的表单在按回车键时不会提交

以这种方式提交表单会在向服务器发送请求之前触发 `submit` 事件。
这样就提供了一个验证表单数据的机会，可以根据验证结果决定是否真的要提交。
阻止这个事件的默认行为可以取消提交表单

### 提交按钮的创建方式

```html
<form id="myForm" name="">
    <!-- 通用提交按钮 -->
    <input type="submit" value="Submit Form">
    
    <!-- 自定义提交按钮 -->
    <button type="submit">Submit Form</button>
    
    <!-- 图片按钮 -->
    <input type="image" src="images/graphic.gif">
</form>
```

### 表单提交

可以通过编程方式在 JavaScript 中调用 `submit()` 方法来提交表单。
可以在任何时候调用 这个方法来提交表单，而且表单中不存在提交按钮也不影响表单提交。
通过 `submit()` 提交表单时 `submit` 事件不会触发。因此在调用这个方法前要先做数据验证

表单提交的一个最大的问题是可能会提交两次表单。如果提交表单之后没有什么反应，
那么没有耐心的用户可能会多次点击提交按钮。结果是很烦人的(因为服务器要处理重复的请求)，
甚至可能造成损失(如果用户正在购物，则可能会多次下单)。解决这个问题主要有两种方式: 
在表单提交后禁用提交按钮，或者通过 `onsubmit` 事件处理程序取消之后的表单提交

```js
let form = document.getElementById("myForm");

// 提交表单
form.submit();
```

### 阻止表单提交

调用 `preventDefault()` 方法可以阻止表单提交。
通常，在表单数据无效以及不应该发送到服务器时可以这样处理

```js
let form = document.getElementById("myForm");

form.addEventListener("submit", (event) => {
    // 阻止表单提交
    event.preventDefault();
});
```

## 重置表单

用户单击重置按钮可以重置表单。表单重置后，所有表单字段都会重置回页面第一次渲染时各自拥有的值。
如果字段原来是空的，就会变成空的;如果字段有默认值，则恢复为默认值

### 重置按钮的创建方式

```html
<form id="myForm" name="">
    <!-- 通用重置按钮 -->
    <input type="reset" value="Reset Form">

    <!-- 自定义重置按钮 -->
    <input type="rest">Reset Form</button>
</form>
```

### 重置表单



* 方法 1： 用户单击重置按钮重置表单会触发 `reset` 事件。这个事件为取消重置提供了机会

```js
let form = document.getElementById("myForm");

form.addEventListener("reset", (event) => {
    // 重置表单
    event.preventDefault();
});
```

* 方法 2: 与表单提交一样，重置表单也可以通过 JavaScript 调用 `reset()` 方法来完成

```js
let form = document.getElementById("myForm");

// 重置表单
form.reset();
```

# 文本框编程




# 选择框编程


# 表单序列化


# 富文本编辑

