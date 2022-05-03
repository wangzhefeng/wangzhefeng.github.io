---
title: 语言基础
author: 王哲峰
date: '2020-01-03'
slug: js-basic
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
- [语法](#语法)
  - [区分大小写](#区分大小写)
  - [标识符](#标识符)
  - [注释](#注释)
  - [严格模式](#严格模式)
  - [语句](#语句)
- [关键字与保留字](#关键字与保留字)
  - [ECMA-262 第 5 版](#ecma-262-第-5-版)
  - [ECMA-262 第 6 版](#ecma-262-第-6-版)
  - [严格模式下保留](#严格模式下保留)
  - [模块代码中保留](#模块代码中保留)
- [变量](#变量)
  - [var 声明](#var-声明)
    - [var 定义变量](#var-定义变量)
    - [var 声明作用域](#var-声明作用域)
    - [var 声明提升](#var-声明提升)
    - [var 反复声明](#var-反复声明)
    - [var 全局声明](#var-全局声明)
  - [let 声明](#let-声明)
    - [let 声明变量](#let-声明变量)
    - [let 不允许反复声明](#let-不允许反复声明)
    - [暂时性死区(let 声明不会提升)](#暂时性死区let-声明不会提升)
    - [全局声明](#全局声明)
    - [条件声明](#条件声明)
    - [for 循环中的 let 声明](#for-循环中的-let-声明)
  - [const 声明](#const-声明)
    - [const 声明变量必须初始化且不可改变](#const-声明变量必须初始化且不可改变)
    - [const 不允许重复声明](#const-不允许重复声明)
    - [const 声明范围是块作用域](#const-声明范围是块作用域)
    - [const 声明的限制只适用于它指向的变量的引用](#const-声明的限制只适用于它指向的变量的引用)
    - [for 循环中的 const 声明](#for-循环中的-const-声明)
  - [声明风格及最佳实践](#声明风格及最佳实践)
- [数据类型](#数据类型)
  - [typeof 操作符](#typeof-操作符)
  - [Undefined](#undefined)
    - [变量声明未初始化、变量未声明](#变量声明未初始化变量未声明)
    - [undefined 是假值](#undefined-是假值)
  - [Null](#null)
    - [变量初始化 nul](#变量初始化-nul)
    - [undefined 与 null 表面上相等](#undefined-与-null-表面上相等)
    - [null 是假值](#null-是假值)
  - [Boolean](#boolean)
    - [其他类型的布尔值的等价形式及转换](#其他类型的布尔值的等价形式及转换)
  - [Number](#number)
    - [整数](#整数)
    - [浮点值](#浮点值)
      - [浮点数定义](#浮点数定义)
      - [转换为整数](#转换为整数)
      - [科学计数法](#科学计数法)
      - [浮点数精度](#浮点数精度)
    - [值的范围](#值的范围)
      - [最大最小值](#最大最小值)
      - [无穷值](#无穷值)
    - [NaN](#nan)
      - [NaN](#nan-1)
      - [NaN 属性](#nan-属性)
    - [数值转换](#数值转换)
      - [Number()](#number-1)
      - [parseInt()](#parseint)
      - [parseFloat()](#parsefloat)
  - [String](#string)
    - [字符字面量](#字符字面量)
    - [字符串的特点](#字符串的特点)
    - [转换为字符串](#转换为字符串)
    - [模板字面量](#模板字面量)
    - [字符串插值](#字符串插值)
    - [模板字面量标签函数](#模板字面量标签函数)
    - [原始字符串](#原始字符串)
  - [Symbol](#symbol)
    - [符号的基本用法](#符号的基本用法)
    - [使用全局符号注册表](#使用全局符号注册表)
    - [使用符号作为属性](#使用符号作为属性)
    - [常用内置符号](#常用内置符号)
    - [Symbol.asyncIterator](#symbolasynciterator)
    - [Symbol.hasInstance](#symbolhasinstance)
    - [Symbol.isConcatSpreadable](#symbolisconcatspreadable)
    - [Symbol.iterator](#symboliterator)
    - [Symbol.match](#symbolmatch)
    - [Symbol.replace](#symbolreplace)
    - [Symbol.search](#symbolsearch)
    - [Symbol.species](#symbolspecies)
    - [Symbol.split](#symbolsplit)
    - [Symbol.toPrimitive](#symboltoprimitive)
    - [Symbol.toStringTag](#symboltostringtag)
    - [Symbol.unscopables](#symbolunscopables)
  - [Object](#object)
- [操作符](#操作符)
  - [一元操作符](#一元操作符)
    - [递增/递减操作符](#递增递减操作符)
      - [前缀递增-递减](#前缀递增-递减)
      - [后缀递增-递减](#后缀递增-递减)
      - [递增和递减原则](#递增和递减原则)
    - [一元加和减](#一元加和减)
  - [位操作符](#位操作符)
  - [布尔操作符](#布尔操作符)
    - [逻辑非](#逻辑非)
    - [逻辑与](#逻辑与)
    - [逻辑或](#逻辑或)
  - [乘性操作符](#乘性操作符)
    - [乘法操作符](#乘法操作符)
    - [除法操作符](#除法操作符)
    - [取模操作符](#取模操作符)
  - [指数操作符](#指数操作符)
  - [加性操作符](#加性操作符)
    - [加法操作符](#加法操作符)
    - [减法操作符](#减法操作符)
  - [关系操作符](#关系操作符)
    - [规则](#规则)
    - [字符串比较](#字符串比较)
  - [相等操作符](#相等操作符)
    - [等于和不等于](#等于和不等于)
    - [全等和不全等](#全等和不全等)
  - [条件操作符](#条件操作符)
  - [赋值操作符](#赋值操作符)
  - [逗号操作符](#逗号操作符)
- [语句](#语句-1)
  - [if](#if)
  - [do-while](#do-while)
  - [while](#while)
  - [for](#for)
    - [基本用法](#基本用法)
    - [关键字声明使用 let](#关键字声明使用-let)
    - [初始化,条件表达式和循环后表达式都不是必需的](#初始化条件表达式和循环后表达式都不是必需的)
    - [for 循环等价于 while 循环](#for-循环等价于-while-循环)
  - [for-in](#for-in)
  - [for-of](#for-of)
  - [label](#label)
  - [break 和 continue](#break-和-continue)
    - [基本用法](#基本用法-1)
    - [与 label 语句一起使用](#与-label-语句一起使用)
  - [with](#with)
    - [语法](#语法-1)
    - [使用场景](#使用场景)
  - [switch](#switch)
    - [语法](#语法-2)
    - [示例](#示例)
    - [特性](#特性)
- [函数](#函数)
</p></details><p></p>

# 概览

JavaScript 的核心语言特性在 ECMA-262 中以伪语言 ECMAScript 的形式来定义。
ECMAScript 包含 所有基本语法、操作符、数据类型和对象，
能完成基本的计算任务，但没有提供获得输入和产生输出的机制。
理解 ECMAScript 及其复杂的细节是完全理解浏览器中 JavaScript 的关键。
下面总结一下 ECMAScript 中的基本元素:

* ECMAScript 中的基本数据类型包括 `Undefined`、`Null`、`Boolean`、`Number`、`String` 和 `Symbol`
    - 与其他语言不同，ECMAScript 不区分整数和浮点值，只有 Number 一种数值数据类型
    - `Object` 是一种复杂数据类型，它是这门语言中所有对象的基类
* 严格模式为这门语言中某些容易出错的部分施加了限制
* ECMAScript 提供了 C 语言和类 C 语言中常见的很多基本操作符，包括数学操作符、布尔操作符、 关系操作符、相等操作符和赋值操作符等
* ECMAScript 中的流控制语句大多是从其他语言中借鉴而来的，比如 `if` 语句、`for` 语句和 `switch` 语句等
* ECMAScript 中的函数与其他语言中的函数不一样
    - 不需要指定函数的返回值，因为任何函数可以在任何时候返回任何值。 
    - 不指定返回值的函数实际上会返回特殊值 `undefined`

# 语法

## 区分大小写

* JavaScript 区分大小写

## 标识符

* 标识符包含: 变量、函数、属性、函数参数
* 字母、下划线、美元符号开头
* 字母、数字、下划线、美元符号组成
* 惯例: 驼峰大小写, 第一个单词的首字母小写, 后面每个单词的首字母大写

## 注释

- 单行:  

```js
// 单行注释
```

- 多行: 

```js
/* 
 * 多行注释 
 * 又是单行注释
 */
```

## 严格模式

* 严格模式是一种不同的 JavaScript 解析和执行模型, 
  ECMAScript 3 的一些不规范写法在这种模式下会被处理, 
  对于不安全的活动将抛出错误。选择这种语法形式的目的是不破坏 ECMAScript 3 语法
* 要对整个脚本启用严格模式, 在脚本的开头加上 `"use strict";`
* 所有现代浏览器都支持严格模式

## 语句

* 语句以分号结尾, 不是必须的, 建议加
* 多条语句可以合并到一个 C 语言风格的代码块中。代码块由 `{}` 包含。
  控制流语句在执行多条语句时要求必须有代码块, 最佳实践是始终在控制语句中使用代码块

# 关键字与保留字

## ECMA-262 第 5 版

|关键字        |是否掌握              |
|-------------|---------------------|
|`break`      |:black_square_button:|
|`do`         |:black_square_button:|
|`in`         |:black_square_button:|
|`typeof`     |:black_square_button:|
|`case`       |:black_square_button:|
|`else`       |:black_square_button:|
|`instanceof` |:black_square_button:|
|`var`        |:black_square_button:|
|`catch`      |:black_square_button:|
|`export`     |:black_square_button:|
|`new`        |:black_square_button:|
|`void`       |:black_square_button:|
|`class`      |:black_square_button:|
|`extends`    |:black_square_button:|
|`return`     |:black_square_button:|
|`while`      |:black_square_button:|
|`const`      |:black_square_button:|
|`finally`     |:black_square_button:|
|`super`      |:black_square_button:|
|`with`       |:black_square_button:|
|`continue`   |:black_square_button:|
|`for`        |:black_square_button:|
|`switch`     |:black_square_button:|
|`yield`      |:black_square_button:|
|`debugger`   |:black_square_button:|
|`function`   |:black_square_button:|
|`this`       |:black_square_button:|
|`default`    |:black_square_button:|
|`if`         |:black_square_button:|
|`throw`      |:black_square_button:|
|`delete`     |:black_square_button:|
|`import`     |:black_square_button:|
|`try`        |:black_square_button:|

## ECMA-262 第 6 版

|关键字        |是否掌握              |
|-------------|---------------------|
|`enum`       |:black_square_button:|

## 严格模式下保留

|关键字        |是否掌握              |
|-------------|---------------------|
|`implements` |:black_square_button:|
|`package`    |:black_square_button:|
|`pubilc`     |:black_square_button:|
|`interface`  |:black_square_button:|
|`protected`  |:black_square_button:|
|`static`     |:black_square_button:|
|`let`        |:black_square_button:|
|`private`    |:black_square_button:|

## 模块代码中保留

|关键字        |是否掌握              |
|-------------|---------------------|
|`await`      |:black_square_button:|

# 变量

## var 声明

### var 定义变量

* 定义不初始化

```js
var message;  // undefined
```

* 定义并初始化
    - 只是一个简单的赋值, 可以改变保存的值, 也可以改变值的类型

```js
var message = "hi"
message = 100;  // 合法, 但不推荐
```

* 定义多个变量

```js
var message = "hi", 
    found = false, 
    age = 29;
```

### var 声明作用域

> var 声明的范围是函数作用域

- 使用 `var` 操作符定义的变量会成为包含它的函数的局部变量

```js
function test() {
    var message = "hi";  // 局部变量
}
test();
console.log(message);  // 出错
```

- 在函数内部定义变量时省略 `var` 操作符可以创建一个全局变量

```js
function test() {
    message = "hi";  // 全局变量
}
test();
console.log(message);  // "hi"
```

<div class="warning" 
     style='background-color:#E9D8FD; color: #69337A; border-left: solid #805AD5 4px; border-radius: 4px; padding:0.7em;'>
    <span>
        <p style='margin-top:1em; text-align:left'>
            <b>Note</b>
        </p>
        <p style='margin-left:1em;'>
            虽然可以通过省略var操作符定义全局变量，但不推荐这么做。在局部作用域中定 义的全局变量很难维护，也会造成困惑。这是因为不能一下子断定省略 var 是不是有意而 为之。在严格模式下，如果像这样给未声明的变量赋值，则会导致抛出 ReferenceError</br></br>
            在严格模式下, 不能定义名为 eval 和 arguments 的变量, 否则会导致语法错误
        </p>
        <p style='margin-bottom:1em; margin-right:1em; text-align:right; font-family:Georgia'> 
            <b></b> 
            <i></i>
        </p>
    </span>
</div>

### var 声明提升

- 使用 `var` 关键字声明的变量会自动提升(hoist)到函数作用域顶部, 
  所谓的“提升”(hoist)，也就是把所有变量声明都拉到函数作用域的顶部 

```js
// 正常代码
function foo() {
    var age = 26;
    console.log("age", age);
}
foo();  // age: 26

// 变量声明提升--不会报错
function foo() {
    console.log("age:", age);
    var age = 26;
}
foo(); // age: undefined

// 变量声明提升--不会报错, 与上面代码等价
function foo() {
    var age;
    console.log("age:", age);
    age = 26;
}
foo(); // age: undefined
```

### var 反复声明

* 反复多次使用 `var` 声明同一变量也没有问题

```js
function foo() {
    var age = 16;
    var age = 26;
    var age = 36;
    console.log(age);
}
foo(); // 36
```

```js
var name = "Matt";
var name = "John";
console.log(name);  // "John"
```

### var 全局声明

* 使用 `var` 在全局作用域中声明的变量会成为 `window` 对象的属性, 

```js
var name = "Matt";
console.log(window.name); // "Matt"
```

## let 声明

### let 声明变量

`let` 跟 `var` 最明显的区别是: `let` 声明的范围是**块作用域**；`var` 声明的范围是**函数作用域**

* `var` 声明的范围是**函数作用域**:

```js
if (true) {
    var name = "Matt";
    console.log(name);  // Matt
}
console.log(name);  // Matt
```

* `let` 声明的范围是**块作用域**

```js
if (true) {
    let age = 26;
    console.log(age);  // 26
}
console.log(age);  // ReferenceError: age 没有定义
```

### let 不允许反复声明

* `let` 不允许同一个块作用域中出现冗余声明, 会导致报错，而 `var` 可以

```js
var name;
var name;
```

```js
let age;
let age; // //SyntaxError;标识符age已经声明过了
```

* JavaScript 引擎会记录用于变量声明的标识符及其所在的块作用域, 
  因此嵌套使用相同的标识符不会报错，这是因为同一个块中没有重复声明

```js
// var
var name = "Nicholas";
console.log(name); // "Nicholas"

if (true) {
    var name = "Matt";
    console.log(name); // "Matt"
}
```

```js
// let
let age = 30;
console.log(age); //30

if (true) {
    let age = 26;
    console.log(age); // 26
}
```

* 对声明冗余报错不会因为混用 `let` 和 `var` 而受影响。
  这两个关键字声明的并不是不同类型的变量，
  它们只是指出变量在相关作用域如何存在

```js
var name;
let name; // SyntaxError
```

```js
let age;
var age; // SyntaxError
```

### 暂时性死区(let 声明不会提升)

* `let` 声明的变量不会在作用域中被提升。
  在 `let` 声明之前的执行瞬间被称为"暂时性死区"(temporal dead zone), 
  在此阶段引用任何后面才声明的变量都会抛出 `ReferenceError`

```js
// name 会被提升
console.log(name); // undefined
var name = "Matt";
```

```js
// age 不会被提升
console.log(age); // ReferenceError: age 没有定义
let age = 26;
```

### 全局声明

* 使用 `let` 在全局作用域中声明的变量不会成为 `window` 对象的属性, 
  `var` 声明的变量则会

```js
var name = "Matt";
console.log(window.name); // "Matt"
```

```js
let age = 26;
console.log(window.age); // undefined
```

### 条件声明

* 因为 `let` 的作用域是块, 所以不可能检查前面是否已经使用 `let` 声明过同名变量, 
  同时也就不可能在没有声明的情况下声明它。而 `var` 声明变量时, 由于声明会被提升, 
  JavaScript 引擎会自动将多余的声明在作用域顶部合并为一个声明
* `let` 声明不能依赖条件声明模式

```html
<script>
    var name = "Nicholas";
    let age = 26;
</script>

<script>
    // 假设脚本不确定页面中是否已经声明了同名变量, 那么可以假设还没有声明过
    // 这里没有问题, 因为可以被作为一个提升声明来处理, 
    // 不需要检查之前是否声明过同名变量
    var name = "Matt";
    
    // 如果 age 之前声明过, 这里会报错
    let age = 26;
</script>
```

* 使用 `try/catch` 语句或 `typeof` 操作符也不能解决

```html
<script>
    let name = "Nicholas";
    let age = 26;
</script>

<script>
    // 假设脚本不确定页面中是否已经声明了同名变量, 那么可以假设还没有声明过
    if (typeof name === "undefined") {
        let name = "Matt"; // name 被限制在 if {} 块作用域内
    }
    name = "Matt";         // 因此这个赋值形同全局赋值
    
    try {
        console.log(age);  // 如果 age 没有声明过, 则会报错
    }
    catch(error) {
        let age;           // age 被限制在 catch {} 作用域内
    }
    age = 26;              // 因此这个赋值形同全局赋值
</script>
```

### for 循环中的 let 声明

在 `let` 出现之前, `for` 循环定义的迭代变量会渗透到循环体外部; 
使用 `let` 之后, `for` 循环定义的迭代变量不会渗透到循环体外部

* `for` 循环中的 `var` 定义的迭代变量会渗透到循环体外部

```js
for (var i = 0; i < 5; ++i) {
    console.log("hello world!");
}
console.log(i); // 5
```

* `for` 循环中的 `let` 定义的迭代变量不会渗透到循环体外部

```js
for (let i = 0; i < 5; ++i) {
    console.log("hello world!");
}
console.log(i); // ReferenceError: i 没有定义
```

* 使用 `var` 的时候, 在退出循环的时, 迭代变量保存的是导致循环退出的值, 
  之后执行超时逻辑时, 所有的迭代变量都是同一个变量; 
  而使用 `let` 声明迭代变量时, JavaScript 引擎在后台为每个循环声明一个新的迭代变量

```js
for (var i = 0; i < 5, ++i) {
    setTimeout(() => console.log(i), 0) // 5 5 5 5 5
}

for (let i = 0; i < 5; ++i) {
    setTimeout(() => console.log(i), 0) // 0 1 2 3 4
}
```

## const 声明

### const 声明变量必须初始化且不可改变

* `const` 声明与 `let` 声明唯一一个重要区别是它声明变量时必须同时初始化变量, 
  且尝试修改 `const` 声明的变量会导致运行时错误

```js
const age = 26;
age = 36; // TypeError: 给常量赋值
```

### const 不允许重复声明

```js
const name = "Matt";
const name = "John";
console.log(name); // SyntaxError: Identifier 'name' has already been declared
```

### const 声明范围是块作用域

```js
const name = "Matt";
if (true) {
    const name = "Nicholas";
}
console.log(name); // "Matt"
```

### const 声明的限制只适用于它指向的变量的引用

* 如果 `const` 变量引用的是一个对象, 那么修改这个对象内部的属性并不违反 `const` 对象不能修改变量的限制

```js
const person = {};
person.name = "Matt"; // ok
```

### for 循环中的 const 声明

* 不能用 `const` 来声明迭代变量(因为迭代变量会自增)
    - JavaScript 引擎会为 `for` 循环中的 `let` 声明分别创建独立的变量实例, 但 `const` 不行; 
    - 如果用 `const` 声明一个不被修改的 `for` 循环变量, 那是可以的, 
      也就是说, 每次迭代只是创建一个新变量, 这对 `for-of` 和 `for-in` 循环特别有意义

```js
for (const i = 0; i < 10; ++i) {
    console.log(i); 
}
// TypeError: 给常量赋值

let i = 0;
for (const j = 7; i < 5; ++i) {
    console.log(j);
}
// 7 7 7 7 7

for (const key in {a: 1, b: 2}) {
    console.log(key);
}
// a, b

for (const value of [1, 2, 3, 4, 5]) {
    console.log(value);
}
// 1, 2, 3, 4, 5
```

## 声明风格及最佳实践

* 不使用 `var`
    - 有了 `let` 和 `const`, 大多数开发者会发现自己不再需要 `var` 了。
      限制自己只使用 `let` 和 `const` 有助于提升代码质量, 
      因为变量有了明确的作用域、声明位置, 以及不变的值
* `const` 优先, `let` 次之
    - 使用 `const` 声明可以让浏览器运行时强制保持变量不变, 
      也可以让静态代码分析工具提前发现不合法的赋值操作。
      因此, 很多开发者认为应该优先使用 `const` 来声明变量
    - 只在提前知道未来会有修改时, 再使用 `let`。
      这样可以让开发者更有信心地推断某些变量的值永远不会变, 
      同时也能迅速发现因意外赋值导致的非预期行为

# 数据类型

> ECMAScript 有 6 种简单数据类型(也称为原始类型): 
> 
> * Undefined
> * Null
> * Boolean
> * Number
> * String
> * Symbol
> 
> ECMAScript 有 1 种复杂数据类型: 
> 
> * Object: 无序名值对的集合
>   - function[TODO]

## typeof 操作符

因为 ECMAScript 的类型系统是松散的, 所以需要一种手段来确定任意变量的数据类型。
typeof 操作符就是为此而生的, 对一个值使用 typeof 操作得到的结果:

* "undefined" 表示值未定义
* "boolean" 表示布尔值
* "string" 表示值为字符串
* "number" 表示值为数值
* "object" 表示值为对象(而不是函数) 或 null
    - null 被认为是一个对空对象的引用
* "function" 表示值为函数
* "symbol" 表示值为符号

```js
let message = "some string";
console.log(typeof message);   // "string"
console.log(typeof(message));  // "string"
console.log(typeof 95);        // number
console.log(typeof null);      // object
```

<div class="warning" 
     style='background-color:#E9D8FD; color: #69337A; border-left: solid #805AD5 4px; border-radius: 4px; padding:0.7em;'>
    <span>
        <p style='margin-top:1em; text-align:left'>
            <b>Note</b>
        </p>
        <p style='margin-left:1em;'>
             <ul>
                <li>typeof 是一个操作符而不是函数，所以不需要参数，但可以使用参数</li>
                <li>
                    严格来讲，函数在 ECMAScript 中被认为是对象，并不代表一种数据类型。
                    可是，函数也有自己的特殊属性。为此，就有必要通过 typeof 操作符来区分函数和其他对象。
                </li>
             </ul>
        </p>
        <p style='margin-bottom:1em; margin-right:1em; text-align:right; font-family:Georgia'> 
            <b></b> 
            <i></i>
        </p>
    </span>
</div>

## Undefined

Undefined 类型只有一个值，就是特殊值 `undefined`

### 变量声明未初始化、变量未声明

当使用 `var` 或 `let` 声明了变量但没有初始化时，
就相当于给变量赋予了 `undefined` 值

一般来说，永远不要显式地给某个变量设置 `undefined` 值，字面值 `undefined` 主要用于比较。
增加这个特殊值的目的就是为了正式明确空对象 `null` 和未初始化变量的区别

```js
let message;
console.log(message == undefined);  // true

let message = undefined;
console.log(message == undefined);  // true
```

* 包含 `undefined` 值的变量跟未定义变量是有区别的，对未声明的变量执行除 `typeof` 操作时，
  都会报错。对为声明的变量调用 `delete` 也不会报错，但这个操作没什么用，实际上在严格模式下会抛出错误
* 无论是声明还是未声明，`typeof` 返回的都是字符串 "undefined"。
  逻辑上讲这是对的，因为虽然严格来讲这两个变量存在根本性差异，但它们都无法执行实际操作
* 即使未初始化的变量会被自动赋予 `undefined` 值，但仍然建议在声明变量的同时进行初始化。
  这样，当 `typeof` 返回 "undefined" 时，你就会知道那是因为给定的变量尚未声明，
  而不是声明了但未初始化

```js
let message;

console.log(message);  // undefined
console.log(age);  // 报错


console.log(typeof message);  // "undefined"
console.log(typeof age);  // "undefined"
```

### undefined 是假值

`undefined` 是一个假值。因此，如果需要，可以用更简洁的方式检测它。
不过要记住，也有很多 其他可能的值同样是假值。
所以一定要明确自己想检测的就是 `undefined` 这个字面值，而不仅仅是假值

```js
let message;

if (message) {
    // 这个块不会执行
}

if (!message) {
    // 这个块不会执行
}

if (age) {
    // 这里会报错
}
```

## Null

Null 类型只有一个值，即特殊值 `null`

逻辑上讲，`null` 值表示一个空对象指针，
这也是给 `typeof` 传一个 `null` 会返回 "object" 的原因


### 变量初始化 nul

在定义将来要保存对象值的变量时，建议使用 `null` 来初始化，不要使用其他值，
这样，只要检查这个变量的值是不是 `null` 就可以知道这个变量是否在后来被重新赋予了一个对象的引用

任何时候，只要变量要保存对象，而当时又没有那个 对象可保存，就要用 null 来填充该变量。
这样就可以保持 null 是空对象指针的语义，并进一步将其 与 undefined 区分开来

```js
let car = null;
console.log(typeof car);  // "object"

if (car != null) {
    // car 是一个对象的引用
}
```

### undefined 与 null 表面上相等

`undefined` 值是由 `null` 值派生而来的，因此 ECMA-262 将它们定义为表面上相等

用 等于操作符(`==`)比较 `null` 和 `undefined` 始终返回 `true`，
但是要注意，这个操作符会为了比较而转换它的操作数

```js
console.log(null == undefined);  // true
```

### null 是假值

`null` 是一个假值。因此，如果需要，可以用更简洁的方式检测它。
不过要记住，也有很多 其他可能的值同样是假值。
所以一定要明确自己想检测的就是 `null` 这个字面值，而不仅仅是假值

```js
let message = null;     // null
let age;                // undefined

if (message) {
    // 这个块不会执行
}

if (!message) {
    // 这个块会执行
}

if (age) {
    // 这个块不会执行
}

if (!age) {
    // 这个块会执行
}
```

## Boolean

Boolean 类型有两个字面值: `true` 和 `false`
这两个布尔值不同于数值，因此 `true` 不等于 `1`，`false` 不等于 `0`

### 其他类型的布尔值的等价形式及转换

虽然布尔值只有两个，但所有其他 ECMAScript 类型的值都有相应的布尔值的等价形式。
要将一个其他类型的值转换为布尔值，可以调用特定的 `Boolean()` 转型函数。
`Boolean()` 转型函数可以在任意类型的数据上调用，而且始终返回一个布尔值。
什么值能转换为 `true` 或 `false` 的规则取决于数据类型和实际的值。

下表总结了不同类型与布尔值之间的转换规则:

|数据类型  |转换为 true 的值    |转换为 false 的值 |
|---------|------------------|----------------|
|Undefined |N/A(不存在)        |undefined        |
|Boolean  |true              |false           |
|String   |非空字符串          |\"\"(空字符串)     |
|Number   |非零数值(包括无穷值) |0, NaN          |
|Object   |任意对象            |null            |


```js
let message = "Hello world!";
let messageAsBoolean = Boolean(message);
```

## Number

Number 类型使用 IEEE 754 格式表示整数和浮点数(双精度值)。
不同的数值类型相应地也有不同的数值字面量格式

### 整数

* 十进制整数
* 八进制整数
    - `0` + `0~7` 
    - 如果字面量中包含的数字超出了应有的范围，就会忽略前缀的零，
      后面的数字序列会被当成十进制数
    - 八进制字面量在严格模式下是无效的，会导致 JavaScript 引擎抛出语法错误
* 十六进制整数
    - `0x` + `0~9` 以及 `A~F` 
    - 字母大小写均可

```js
let intNum = 55;  // 整数
let octalNum1 = 070; // 八进制的 56
let octalNum2 = 079; // 无效的八进制值，当成 79 处理
let hexNum1 = 0xA;  // 十六进制 10
let hexNum2 = 0x1f;  // 十六进制 31
```

<div class="warning" 
     style='background-color:#E9D8FD; color: #69337A; border-left: solid #805AD5 4px; border-radius: 4px; padding:0.7em;'>
    <span>
        <p style='margin-top:1em; text-align:left'>
            <b>Note</b>
        </p>
        <p style='margin-left:1em;'>
             <ul>
                <li>使用八进制和十六进制格式创建的数值在所有的数学操作中都被视为十进制数值</li>
                <li>
                    由于 JavaScript 保存数值的方式，实际中可能存在正零(+0)和负零(-0)。
                    正零和负零在所有情况下都被认为是等同的
                </li>
             </ul>
        </p>
        <p style='margin-bottom:1em; margin-right:1em; text-align:right; font-family:Georgia'> 
            <b></b> 
            <i></i>
        </p>
    </span>
</div>


### 浮点值

#### 浮点数定义

要定义浮点值，数值中必须包含小数点，而且小数点后面必须至少有一个数字。
虽然小数点前面不是必须有整数，但推荐加上

```js
let floatNum1 = 1.1;
let floatNum2 = 0.1;
let floatNum3 = .1;  // 有效，但不推荐
```

#### 转换为整数

因为存储浮点值使用的内存空间是存储整数值的两倍，
所以 ECMAScript 总是想方设法把值转换为整数:

* 在小数点后面没有数字的情况下，数值就会变成整数
* 如果数值本身就是整数，只是小数点后面跟着 0，那它也会被转换为整数

```js
let floatNum1 = 1.;  // 1
let floatNum2 = 10.0; // 10
```

#### 科学计数法

对于非常大或非常小的数值，浮点值可以用科学计数法来表示，
科学计数法用于表示一个应该乘以 10 的给定次幂的数值。

ECMAScript 科学计数法的格式要求是一个数值(整数或浮点数)后跟一个大写或小写的字母 `e`，
再加上一个要乘以10的多少次幂

默认情况下，ECMAScript 会将小数点后至少包含 6 个零的浮点值转换为科学计数法

```js
let floatNum = 3.125e7; // 312500000
```

#### 浮点数精度

浮点数的精度最高可达 17 为小数，但在计算中远不如整数精确。
由于存在舍入错误，导致很难测试特定的浮点值。因此永远不要测试某个特定的浮点值

```js
if (a + b == 0.3) {  // 别这么干！
  console.log("You got 0.3.")
}
```

### 值的范围

由于内存的限制，ECMAScript 并不支持表示这个世界上的所有数值

#### 最大最小值

* ECMAScript 可以表示的最小数值保存在 `Number.MIN_VALUE` 中，
  这个值在浏览器中是 `5e-324`
* ECMAScript 可以表示的最大数值保存在 `Number.MAX_VALUE` 中，
  这个值在多数浏览器中是 `1.797 693 134 862 315 7e+308`

#### 无穷值

如果某个计算得到的数值结果超出了 JavaScript 可以表示的范围，
那么这个数值会被自动转换为一个特殊的 `Infinity`(无穷)值

* 任何无法表示的负数以 `-Infinity`(负无穷大)表示，
  使用 `Number.NEGATIVE_INFINITY` 也可以获取负 Infinity
* 任何无法表示的正数以 `Infinity`(正无穷大)表示，
  使用 `Number.POSITIVE_INFINITY`  也可以获取正 Infinity

如果计算返回正 `Infinity` 或负 `Infinity`，则该值将不能再进一步用于任何计算，
这是因为 `Infinity` 没有可用于计算的数值表示形式。
要确定一个值是不是有限大(介于最小值和最大值之间)，
可以使用 `isFinite()` 函数

```js
let result = Number.MAX_VALUE + Number.MAX_VALUE;
console.log(isFinite(result));  // false
```

### NaN

#### NaN

特殊的数值 `NaN`: 不是数值(Not a Number)，
用于表示本来要返回数值的操作失败了(而不是抛出错误)，
比如，用 0 除任意数值在其他语言中通常都会导致错误，从而中止代码执行，
但在 ECMAScript 中，`0`、`+0`、`-0` 相除会返回 `NaN`

```js
console.log(0 / 0);  // NaN
console.log(-0 / +0);  // NaN
```

如果分子是非 0 值，分母是有符号 0 或 无符号 0，则会返回 `Infinity` 或 `-Infinity`

```js
console.log(5 / 0);  // Infinity
console.log(5 / -0);  // -Infinity
```

#### NaN 属性

`NaN` 有几个独特的属性:

* 首先，任何涉及 `NaN` 的操作始终返回 `NaN`，在连续多步计算时这可能是个问题
* 其次，`NaN` 不等于包括 `NaN` 在内的任何值
    - 为此，ECMAScript 提供了 `isNaN()`，
      该函数接收一个参数，可以是任意数据类型，然后判断这个参数是否“不是数值”。
      把一个值传给 `isNaN()` 后，该函数会尝试把它转换为数值，
      某些非数值的值可以直接转换成数值，
      任何不能转换为数值的值都会导致这个函数返回 `true`
    - `isNaN()` 可以用于测试对象。此时，首先会调用对象的 `valueOf()` 方法，
      然后再确定返回的值是否可以转换为数值，如果不能，再调用 `toString()` 方法，
      并测试其返回值，这通常是 ECMAScript 内置函数和操作符的工作方式

```js
console.log(NaN == NaN);  // false
console.log(isNaN(NaN));  // true
console.log(isNaN(10));  // false, 10 是数值
console.log(isNaN("10"));  // false, 可以转换为数值 10
console.log(isNaN("blue"));  // true, 不可以转换为数值
console.log(isNaN(true));  // false, 可以转换为数值 1
```

### 数值转换

将非数值转换为数值: 

* `Number()` 
    - 转型函数，可用于任何数据类型 
* `parseInt()`
    - 用于将字符串转换为数值
* `parseFloat()`
    - 用于将字符串转换为数值 

#### Number()

`Number()` 函数基于如下规则执行转换:

* 数值
    - 直接返回
* `null` -> `0`
* `undefined` -> `NaN`
* 布尔值
    - `true` -> `1`
    - `false` -> `0`
* 字符串(String)，应用以下规则
    - 如果字符串包含**数值字符**，包括数值字符前面带加、减号的情况，则转换为一个十进制数值，忽略前面的零
    - 如果字符串包含有效的**浮点值**格式，如 `"1.1"`，则会转换为相应的浮点值，忽略前面的零
    - 如果字符串包含有效的**十六进制**格式，如 `"0xf"`，则会转换为与该十六进制值对应的十进制整数值
    - 如果是**空字符串** -> `0`
    - 如果字符串包含除上述情况之外的其他字符 -> `NaN`
* 对象(Object)
    - 调用对象的 `valueOf()` 方法，并按照上述规则转换返回的值。如果转换结果是 `NaN`，
      则调用 `toString()` 方法，再按照转换字符串的规则转换

```js
let num1 = Number("Hello world!");  // NaN
let num2 = Number("");  // 0
let num3 = Number("000011");  // 11
let num4 = Number(true);  // 1
```

#### parseInt()

通常在需要得到整数时可以优先使用 `parseInt()` 函数。
`parseInt()` 函数更专注于字符串是否包含数值模式。

* 字符串最前面的空格会被忽略，从第一个非空格字符开始转换
* 如果第一个字符不是数值字符、加号、减号，立即返回 `NaN`
* 如果第一个字符是数值字符、加号、减号，则继续一次检测每个字符，直到字符串末尾，或碰到非数值字符
* 如果字符串的第一个字符是数值字符，可以识别不同的整数格式，十进制、八进制、十六进制
* 不同的数值格式很容易混淆，因此 `parseInt()` 也接收第二个参数，
  用于指定底数(进制数)，如果提供了进制参数，那么字符串前面的前缀可以省略，
  因为不传底数参数相当于让 `parseInt()` 自己决定如何解析，所以为了避免解析出错，
  建议始终传给它第二个参数

```js
let num1 = parseInt("123blue");  // 1234
let num2 = parseInt("");  // NaN
let num3 = parseInt("0xA");  // 10
let num4 = parseInt("22.5");  // 22
let num5 = parseInt("70");  // 70
let num6 = parseInt("0xf");  // 15
let num7 = parseInt("0xAF");  // 175
let num8 = parseInt("AF");  // 175
let num9 = parseInt("AF");  // NaN
```

#### parseFloat()

`parseFloat()` 函数的工作方式跟 `parseInt()` 函数类似

* 从位置 0 开始检测每个字符
* 也是解析到字符串末尾或者解析到一个无效的浮点数值字符为止。
  这意味着第一次出现的小数点是有效的，但第二次出现的小数点就无效了，
  此时字符串的剩余字符都会被忽略
* 始终忽略字符串开头的零，这个函数能识别前面讨论的所有浮点格式，以及十进制格式，
  因为 `parseFloat()` 只解析十进制数值，因此不能指定底数
    - 十六进制数值始终会返回 0
* 如果字符串表示整数，没有小数点或者小数点后面只有一个零，则返回整数

```js
let num1 = parseFloat("1234blue");  // 1234
let num2 = parseFloat("0xA");  // 0
let num3 = parseFloat("22.5");  // 22.5
let num4 = parseFloat("22.34.5");  // 22.34
let num5 = parseFloat("0908.5");  // 908.5
let num6 = parseFloat("3.125e7");  // 31250000
```

## String

String(字符串)数据类型表示零或多个16位 Unicode 字符序列。
字符串可以使用如下符号进行标示：

* `"`
* `'`
* `\``

```js
let firstName = "John";
let lastName = 'Jacob';
let lastName = `Jingleheimerschmidt`;
```

TODO 字符串属性:

* length：字符串的长度，这个属性返回字符串中 16 位字符的个数，
  如果字符串中包含双字节字符，那么 length 属性返回的值可能不是准确的字符数


### 字符字面量

字符串数据类型包含一些字符字面量，用于表示非打印字符或有其他用途的字符。
并且转义字符表示一个字符:

|字面量   |含义                                                                       |
|--------|--------------------------------------------------------------------------|
|`\n`    |换行                                                                       |
|`\t`    |制表                                                                       |
|`\b`    |退格                                                                       |
|`\r`    |回车                                                                       |
|`\f`    |换页                                                                       |
|`\\`    |反斜杠(\\)                                                                 |
|`\'`    |单引号(`'`)，在字符串以单引号标示时使用，例如 'He said, \\'hey.\\''              |
|`\"`    |双引号(`"`)，在字符串以双引号标示时使用，例如 "He said, \\"hey.\\""              |
| \\`    |反引号(`\``)，在字符串以反引号标示时使用                                        |
|`\xnn`  |以十六进制编码 `nn` 表示的字符(其中 `n` 是十六进制数字 `0~F`)，例如 `\x41` 等于 "A"|
|`\unnnn`|以十六进制编码 `nnnn` 表示的 Unicode 字符(其中 `n` 是十六进制数字 `0~F`)|

### 字符串的特点

ECMAScript 中的字符串是不可变的(immutable)，意思是一旦创建，它们的值就不能变了。
要修改某个变量的字符串值，必须先销毁原始的字符串，然后将包含新值的另一个字符串保存到该变量

```js
let lang = "Java";
lang = lang + "Script";
```

整个过程首先会分配一个足够容纳 10 个字符的空间，然后填充上 "Java" 和 "Script"，
最后销毁原始的字符串 "Java" 和字符串 "Script"，因为这两个字符串都没有用了

### 转换为字符串

有两种方式把一个值转换为字符串:

* `toString()`
    - 几乎所有值都有的方法，`null` 和 `undefined` 值没有这个方法，
      唯一的用途就是返回当前值的字符串等价物
    - 字符串值也有这个方法，该方法只是简单地返回自身的一个副本
    - 多数情况下，不接收任何参数。不过，在对数值调用这个方法时，
      可以接收一个底数参数，即以什么底数来输出数值的字符串表示。
      默认情况下，返回数值的十进制字符串表示，而通过传入参数，可以得到数值的
      二级制、八进制、十六进制，或者其他任何有效基数的字符串表示

```js
let age = 11;
let ageAsString = age.toString();  // 字符串 "11"
let found = true;
let foundAsString = found.toString();  // 字符串 "true"
```

```js
let num = 10;
console.log(num.toString());  // "10"
console.log(num.toString(2));  // "1010"
console.log(num.toString(8));  // "12"
console.log(num.toString(10));  // "10"
console.log(num.toString(16));  // "a"
```

* `String()`
    - 如果不确定一个值是不是 `null` 或 `undefined`，可以使用 `String()` 转型函数，
      它始终会返回表示相应类型值的字符串。`String()` 函数遵循以下规则:
        - 如果值有 `toString()` 方法，则调用该方法(不传参数)并返回结果
        - 如果值是 `null`，返回 `"null"`
        - 如果值是 `undefined`，返回 `"undefined"`

```js
let value1 = 10;
let value2 = true;
let value3 = null;
let value4;

console.log(String(value1)); // "10"
console.log(String(value2)); // "true"
console.log(String(value3)); // "null"
console.log(String(value4)); // "undefined"
```

* 号操作符给一个值加上一个空字符串""也可以将其转换为字符串

```js
// TODO
```

### 模板字面量

ECMAScript 6 新增了使用模板字面量定义字符串的能力

* 与使用单引号或双引号不同，模板字面量保留换行符，可以跨行定义字符串

```js
let myMultiLineString = 'first line\nsecond line';
let myMultiLineTemplateLiteral = `first line
second line`;

console.log(myMultiLineString);
// first line
// second line

console.log(myMultiLineTemplateLiteral);
// first line
// second line

console.log(myMultiLineString === myMultiLineTemplateLiteral); // true
```

* 顾名思义，模板字面量在定义模板时特别有用

```js
let pageHTML = `
<div>
    <a href="#">
        <span>Jake</span>
    </a>
</div>
`;
```

* 由于模板字面量会保持反引号内部的空格，因此在使用时要格外注意。
  格式正确的模板字符串看起来可能缩进不当

```js
// 这个模板字面量在换行符之后有25个空格符
let myTemplateLiteral = `first line
                         second line`;
console.log(myTemplateLiteral.length); // 47
```

```js
// 这个模板字面量以一个换行符开头
let secondTemplateLiteral = `
first line
second line`;
console.log(secondTemplateLiteral[0] === '\n');  // true
```


```js
// 这个模板字面量没有意料之外的字符
let thirdTemplateLiteral = `first line
second line`;
console.log(thirdTemplateLiteral);
// first line
// second line
```

### 字符串插值

模板字面量最常用的一个特性是支持字符串插值，也就是可以在一个连续定义中插入一个或多个值

技术上讲，模板字面量不是字符串，而是一种特殊的 JavaScript 句法表达式，只不过求值后得到的是字符串。
模板字面量在定义时立即求值并转换为字符串实例，任何插入的变量也会从它们最接近的作用域中取值。
所有插入的值都会使用 `toString()` 强制转型为字符串，而且任何 JavaScript 表达式都可以用于插值

* 字符串插值通过在 `{}` 中使用一个 JavaScript 表达式实现:

```js
let value = 5;
let exponent = "second";

// 以前，字符串插值
let interpolateString = 
value + " to the " + exponent + " power is " + (value * value);

// 现在，使用模板字面量
let interpolatedTemplateLiteral = 
`${value} to the ${exponent} power is ${ value * value }`;
```

* 嵌套的模板字符串无须转义


```js
console.log(`Hello, ${ `World` }!`); // Hello, World!
```

* 将表达式转换为字符串时会调用 `toString()`

```js
let foo = { toString: () => 'World' };
console.log(`Hello, ${ foo }!`);  // Hellow, World!
```

* 在插值表达式中可以调用函数和方法

```js
function capitalize(word) {
    return `${ word[0].toUpperCase() }${ word.slice(1) }`;
}

console.log(`${ capitalize('hello') }, ${ capitalize('world') }!`); 
// Hello, World!
```

* 模板也可以插入自己之前的值

```js
let value = '';
function append() {
    value = `${value}abc`;
    console.log(value);
}
append();  // abc
append();  // abcabc
append();  // abcabcabc
```

### 模板字面量标签函数

模板字面量也支持定义**标签函数(tag function)**，而通过标签函数可以自定义插值行为。
标签函数会接收被插值记号分割后的模板和对每个表达式求值的结果

标签函数本身是一个常规函数，通过前缀到模板字面量来应用自定义行为

```js
let a = 6;
let b = 9;

function simpleTag(strings, 
                   aValExpression, 
                   bValExpression, 
                   sumExpression) {
    console.log(strings);
    console.log(aValExpression);
    console.log(bValExpression);
    console.log(sumExpression);

    return 'foobar';
}

let untaggedResult = `${ a } + ${ b } = ${ a + b }`;
let taggedResult = simpleTag`${ a } + ${ b } = ${ a + b }`;
// ["", " + ", " = ", ""]
// 6
// 9
// 15

console.log(untaggedResult);  // "6 + 9 = 15"
console.log(taggedResult);  // "foobar"
```

* 因为表达式参数的数量时可变的，所以通常应该使用剩余操作符(rest operator)将它们收集到一个数组中

```js
let a = 6;
let b = 9;

function simpleTag(string ...expressions) {
    console.log(strings);
    for (const expression of expressions) {
        console.log(expression);
    }

    return 'foobar';
}

let taggedResult = simpleTag`${ a } + ${ b } = ${ a + b }`;
// ["", " + ", " = ", ""]
// 6
// 9
// 15
```

* 对于有 n 个插值的模板字面量，传给标签函数的表达式参数的个数始终是 n，
  而传给标签函数第一个参数所包含的字符串个数则始终是 n + 1。
  因此，如果想把这些字符串和对表达式求值的结果拼接起来作为默认返回的字符串，
  需要进行如下操作
  

```js
let a = 6;
let b = 9;

function zipTag(strings, ...expressions) {
    return strings[0] + expressions.map((e, i) => `${e}${strings[i + 1]}`).join('');
}

let untaggedResult = `${ a } + ${ b } = ${ a + b }`;
let taggedResult = zipTag`${ a } + ${ b } = ${ a + b }`;

console.log(untaggedResult);  // "6 + 9 = 15"
console.log(taggedResult);  // "6 + 9 = 15"
```


### 原始字符串

使用模板字面量可以直接获取原始的模板字面量内容(如换行符或 Unicode 字符)，
而不是被转换后的字符表示。为此可以使用默认的 `String.raw` 标签函数

* Unicode 示例

```js
// \u00A9 是版权符号
console.log(`\u00A9`);  // ©
console.log(String.raw`\u00A9`); // \u00A9
```

* 换行符示例

```js
console.log(`first line \nsecond line`); 
// first line
// second line

console.log(String.raw`first line\nsecond line`);  
// "first line\nsecond line"
```

```js
// 对实际的换行符来说是不行的
console.log(`first line
second line`);
// first line
// second line

console.log(String.raw`first line
second line`);
// first line
// second line
```

* 可以通过标签函数的第一个参数，即字符串数组的 `.raw` 属性取得每个字符串的原始内容

```js
function printRaw(strings) {
    console.log('Actual characters:');
    for (const string of strings) {
        console.log(string);
    }

    console.log('Escaped characters:');
    for (const rawString of strings.raw) {
        console.log(rawString);
    }
}

printRaw`\u00A9${ 'and' }\n`;
// Actual characters:
// ©
// 换行符
// Excaped characters:
// \u00A9
// \n
```

## Symbol

* Symbol(符号) 是 ECMAScript 6 新增的数据类型
* 符号是原始值, 且符号实例是唯一、不可变的
* 符号的用途是确保对象属性使用唯一标识符, 不会发生属性冲突的危险
* 尽管听起来跟私有属性有点类似, 但符号并不是为了提供私有属性的行为才增加的
  (尤其是因为 Object API 提供了方法, 可以更方便地发现符号属性)。
  相反, 符号就是用来创建唯一记号, 进而用作非字符串形式的对象属性

### 符号的基本用法

* 符号需要使用 `Symbol()` 函数初始化。因为符号本身是**原始类型**, 
  所以 `typeof` 操作符对符号返回 `symbol`

```js
let sym = Symbol();
console.log(typeof sym); // symbol
```

* 调用 `Symbol()` 函数时, 可以传入一个字符串参数作为对符号的描述(description), 
  将来可以通过这个字符串来调试代码, 但是这个字符串参数与符号定义或表示完全无关

```js
let genericSymbol = Symbol();
let otherGenericSymbol = Symbol();

let fooSymbol = Symbol("foo");
let otherFooSymbol = Symbol("foo");

console.log(genericSymbol == otherGenericSymbol); // false
console.log(fooSymbol == otherFooSymbol); // false
```

* 符号没有字面量语法, 这也是它们发挥作用的关键。按照规范, 
  你只要创建 `Symbol()` 实例并将其用作对象的新属性, 
  就可以保证它不会覆盖已有的对象属性, 无论是符号属性还是字符串属性。

```js
let genericSymbol = Symbol();
console.log(genericSymbol); // Symbol()

let fooSymbol = Symbol("foo");
console.log(fooSymbol); // Symbol(foo)
```

* `Symbol()` 函数不能与 `new` 关键字一起作为构造函数使用。
  这样做是为了避免创建符号包装对象, 像使用 `Boolean`、`String` 或 `Number` 那样, 
  它们都支持构造函数且可用于初始化包含原始值的包装对象

```js
let myBoolean = new Boolean();
console.log(typeof myBoolean); // "object"

let myString = new String();
console.log(typeof myString); // "object"

let myNumber = new Number();
console.log(typeof myNumber); // "object"

let mySymbol = new Symbol(); // TypeError: Symbol is not a constructor

// 使用符号包装对象
let mySymbol = Symbol();
let myWrappedSymbol = Object(mySymbol);
console.log(typeof myWrappedSymbol); // "object"
```

### 使用全局符号注册表

如果运行时的不同部分需要共享和重用符号实例，那么可以用一个字符串作为键，
在全局符号注册表中创建并重用符号，为此需要使用 `Symbol.for()` 方法

* `Symbol.for()` 对每个字符串键都执行幂等操作。第一次使用某个字符串调用时，
  它会检查全局运行时注册表，发现不存在对应的符号，于是就会生成一个新符号实例并添加到注册表中。
  后续使用相同的字符串的调用同样会检查注册表，发现存在于该字符串对应的符号，然后就会返回该符号实例
* 即使采用相同的符号描述，在全局注册表中定义的符号跟使用 Symbol() 定义的符号也并不等同

### 使用符号作为属性

### 常用内置符号

### Symbol.asyncIterator


### Symbol.hasInstance


### Symbol.isConcatSpreadable

### Symbol.iterator


### Symbol.match


### Symbol.replace


### Symbol.search


### Symbol.species



### Symbol.split


### Symbol.toPrimitive


### Symbol.toStringTag


### Symbol.unscopables

## Object

ECMAScript 中的`对象(object)`其实就是一组数据和功能的组合
对象通过 `new` 操作符后跟对象类型的名称来创建, 
可以通过创建 Object 类型的实例来创建自己的对象, 
然后再给对象添加属性和方法

* 严格来讲, ECMA-262 中对象的行为不一定适合 JavaScript 中的其他对象
    - 比如浏览器环境中的 BOM 对象 和 DOM 对象, 都是由宿主环境定义和提供的宿主对象
    - 而宿主对象不受 ECMA-262 约束, 所以它们可能会也可能不会继承 Object
* Object 的实例本身并不是很有用, 但理解与它相关的概念非常重要。
  类似 Java 中的 `java.lang.Object`, ECMAScript 中的 Object 也是派生其他对象的基类。
  Object 类型的所有属性和方法在派生的对象上同样存在。每个 Object 实例都有如下属性和方法: 
    - `constructor`
        - 用于创建当前对象的函数，比如 `Object()` 函数
    - `hasOwnProperty(propertyName)`
        - 用于判断当前对象实例(不是原型)上是否存在给定的属性。要检查的属性名必须是**字符串**或**符号**
    - `isPrototypeOf(object)`
        - 用于判断当前对象是否为另一个对象的原型
    - `propertyIsEnumerable(propertyName)`
        - 用于判断给定的属性是否可以使用 `for-in` 语句枚举。属性名必须是**字符串**或**符号**
    - `toLocaleString()`
        - 返回对象的字符串表示, 该字符反映对象所在的本地化执行环境
    - `toString()`
        - 返回对象的字符串表示
    - `valueOf()`
        - 返回对象对应的字符串、数值或布尔值表示。通常与 `toString()` 的返回值相同

```js
let o1 = new Object();
// ECMAScript 只要求在给构造函数提供参数时使用括号, 合法, 但不推荐
let o2 = new Object;

o2.name = "wangzf";
console.log(o2.constructor);                   // Object()
console.log(o2.hasOwnProperty("name"));        // true
console.log(o2.isPrototypeOf(o1));             // TODO
console.log(o2.propertyIsEnumerable("name"));  // true
console.log(o2.toLocaleString());              // wangzf
console.log(o2.toString());                    // wangzf
console.log(o2.valueOf("name"));               // wangzf
```

# 操作符

ECMAScript 中的操作符是独特的, 因为它们可用于各种值, 包括字符串、数值、布尔值、对象。
在应用给对象时, 操作符通常会调用 `valueOf()` 和/或 `toString()` 方法来取得可以计算的值

- 数学操作符
- 位操作符
- 关系操作符
- 相等操作符

## 一元操作符

只操作一个值的操作符叫**一元操作符(unary operator)**

### 递增/递减操作符

JavaScript 的递增和递减操作符直接照搬自 C 语言，但有两个版本:

* 前缀版
    - 位于要操作的变量前头 
    - 递增操作符会给数值加 1，把两个加号放在变量前头即可
    - 递减操作符会给数值减 1，把两个减号放在变量前头即可
    - 变量的值会在语句被求值之前改变，在计算机科学中，这通常被称为具有副作用
    - 递增和递减在语句中的优先级是相等的，因此会从左到右依次求值
* 后缀版
    - 位于要操作的变量后头
    - 递增操作符会给数值加 1，把两个加号放在变量后头即可
    - 递减操作符会给数值减 1，把两个减号放在变量后头即可
    - 递增和递减在语句被求值后才发生

#### 前缀递增-递减

* 递增

```js
let age = 29;
++age;
console.log(age);  // 30

// 等价于
let age = 29;
age = age + 1;
console.log(age);  // 30
```

* 递减

```js
let age = 29;
--age;
console.log(age);  // 28

// 等价于
let age = 29;
age = age - 1;
console.log(age);  // 28
```

* 求值: 无论使用前缀递增还是前缀递减操作符，变量的值都会在语句被求值之前改变

```js
let age = 29;
let anotherAge = --age + 2;

console.log(age);         // 28
console.log(anotherAge);  // 30
```

* 前缀递增和递减在语句中的优先级是相等的，因此会从左到右依次求值

```js
let num1 = 2;
let num2 = 20;
let num3 = --num1 + num2;
let num4 = num1 + num2;
console.log(num3);  // 21
console.log(num4);  // 21
```

#### 后缀递增-递减

* 前缀、后缀效果相同的情况

```js
let age = 29;
age++;
console.log(age); // 30

// 等价于
let age = 29;
++age;
console.log(age); // 30
```

* 后缀递增、递减与其他操作混合

```js
let num1 = 2;
let num2 = 20;
let num3 = num1-- + num2;
let num4 = num1 + num2;

console.log(num3);  // 22
console.log(num4);  // 21
```

#### 递增和递减原则

4 个操作符可以作用于任何值，意思是不限于整数、字符串、布尔值、浮点值、对象，递增和递减操作遵循如下规则：

* 字符串: 
    - 如果是有效的数值形式，则转换为数值再应用改变。变量类型从字符串变成数值
    - 如果不是有效的数值形式，则将变量的值设置为 NaN。变量类型从字符串变成数值
* 布尔值
    - 如果是 `false`，则转换为 `0` 再应用改变。变量类型从布尔值变成数值
    - 如果是 `true`，则转换为 `1` 再应用改变。变量类型从布尔值变成数值
* 浮点数
    - 加 `1` 或减 `1`
* 对象
    - 调用其 `valueOf()` 方法取得可以操作的值。对得到的值应用上述规则。
      如果是 `NaN`，则调用 `toString()` 并再次应用其他规则，
      变量类型从对象变成数值

```js
let s1 = "2";
let s2 = "z";
let b = false;
let f = 1.1;
let o = {
    valueOf() {
        return -1;
    }
};

s1++;  // 3
s2++;  // NaN
b++;  // 1
f--;  // 0.10000000000000009
o--;  // -2
```

### 一元加和减

一元加和减操作符主要用于基本的算术，但也可以用于数据类型转换

* 一元加由一个加号(`+`)表示    
    - 放在变量前头, 对数值没有任何影响
    - 放在非数值前面, 则会执行与使用 `Number()` 转型函数一样的类型转换

```js
let num = 25;
num = +num;
console.log(num); // 25
```

* 一元减由一个减号(`-`)表示
    - 放在变量前头，主要用于把数值变成负值，对于数值将其变成相应的负值
    - 放在非数值前面，遵循与一元加同样的规则，先对它们进行转换，然后再取负值

```js
let s1 = "01";
let s2 = "1.1";
let s3 = "z";
let b = false;
let f = 1.1;
let o = {
    valueOf() {
        return -1;
    }
};

s1 = -s1;  // 值变成数值 -1
s2 = -s2;  // 值变成数值 -1.1
s3 = -s3;  // 值变成 NaN
b = -b;    // 值变成数值 0
f = -f;    // 值变成 -1.1
o = -o;    // 值变成数值 1
```

## 位操作符

* TODO

## 布尔操作符

布尔操作符一共有 3 个：

* 逻辑非 `!`
* 逻辑与 `&&`
* 逻辑或 `||`

### 逻辑非

* 操作符始终返回布尔值, 无论应用的是什么数据类型,
  逻辑非操作符首先将操作数转换为布尔值, 然后再对其取反
* 规则

| 表达式                  | 结果  |
| ---------------------- | ----- |
| !对象                   | false |
| !空字符串                | true  |
| !非空字符串              | false |
| !0                     | true  |
| !非0数值(包括Infinity)    | false |
| !null                  | true  |
| !NaN                   | true  |
| !undefined              | true  |

* 逻辑非操作符也可以用于吧任意值转换为布尔值: `!!` 相当于调用了转型函数 `Boolean()`，
  无论操作数是什么类型, 第一个叹号总会返回布尔值, 第二个叹号对该布尔值取反, 
  从而给出变量真正对应的布尔值, 结果与对同一个值使用 `Boolean()` 函数是一样的

```js
console.log(!!"blue"); // true
console.log(!!0); // false
console.log(!!NaN); // false
console.log(!!""); // false
console.log(!!12345); // true
```

### 逻辑与

* 逻辑与操作符遵循的真值表

| 第一个操作数   | 第二个操作数   | 结果   |
| ------------ | ------------ | ----- |
| true         | true         | true  |
| true         | false        | false |
| false        | true         | false |
| false        | false        | false |

* 逻辑与操作符可用于任何类型的操作数，不限于布尔值。
  如果有操作数不是布尔值，则逻辑与并不一定返回布尔值，遵循如下规则 
    - 如果第一个操作数是对象，则返回第二个操作数
    - 如果第二个操作数是对象，则只有第一个操作数求值为 true 才会返回该对象
    - 如果两个操作数都是对象，则返回第二个操作数
    - 如果有一个操作数是 `null`，则返回 `null`
    - 如果有一个操作数是 `NaN`，则返回 `NaN`
    - 如果有一个操作数是 `undefined`，则返回 `undefined`
* 逻辑与操作符是一种短路操作符，意思是如果第一个操作数决定了结果，那么永远不会对第二个操作数求值

```js
let found = true;
let result = (found && someUndeclaredVariable);  // 会出错，someUndeclareVariable 没声明
console.log(result);  // 不会执行
```

```js
let found = false;
let result = (found && someUndeclaredVariable);  // 不会出错，someUndeclareVariable 根本不会执行
console.log(result);  // 会执行
```

### 逻辑或

* 逻辑或操作符遵循的真值表

| 第一个操作数   | 第二个操作数   | 结果   |
| ------------ | ------------ | ----- |
| true         | true         | true  |
| true         | false        | true  |
| false        | true         | true  |
| false        | false        | false |

* 逻辑或操作符可用于任何类型的操作数，不限于布尔值。
  如果有操作数不是布尔值，则逻辑或并不一定返回布尔值，遵循如下规则
    - 如果第一个操作数是对象，则返回第一个操作数
    - 如果第一个操作数求值为 `false`，则返回第二个操作数
    - 如果两个操作数都是对象，则返回第一个操作数
    - 如果两个操作数是 `null`，则返回 `null`
    - 如果两个操作数是 `NaN`，则返回 `NaN`
    - 如果两个操作数是 `undefined`，则返回 `undefined`
* 逻辑或操作符是一种短路操作符，意思是如果第一个操作数为 `true`，第二个操作数就不会再被求值了

```js
let found = true;
let result = (found || someUndeclaredVariable);  // 不会出错
console.log(result);  // 会执行
```

```js
let found = false;
let result = (found || someUndeclaredVariable);  // 会出错
console.log(result);  // 不会执行
```

```js
// 利用逻辑或短路的行为，可以避免给变量赋值 null 或 undefined
// preferredObject 变量包含首选值
// backupObject 包含备用的值
let myObject = preferredObject || backupObject;
```

## 乘性操作符

ECMAScript 定义了 3 个乘性操作符:

* 乘法 `*`
* 除法 `/`
* 取模 `%`

在处理非数值时，它们也会包含一些自动的类型转换: 
如果乘性操作符有不是数值的操作数，
则该操作数会在后台被使用 `Number()` 转型函数转换为数值

### 乘法操作符

乘法操作数在处理特殊值时的特殊行为:

* 如果操作数都是数值，则执行常规的乘法运算，即两个正值相乘是正值，两个负值相乘也是正
  值，正负符号不同的值相乘得到负值。如果 ECMAScript 不能表示乘积，则返回 `Infinity` 或 `-Infinity`
* 返回 `NaN`
    - 如果有任一操作数是 `NaN`，则返回 `NaN`
    - 如果是 `Infinity` 乘以 0，则返回 `NaN`
* 如果是 `Infinity` 乘以非 0 的有限数值，则根据第二个操作数的符号返回 `Infinity` 或 `-Infinity`
* 如果有不是数值的操作数，则先在后台用 `Number()` 将其转换为数值，然后再应用上述规则

### 除法操作符

除法操作数在处理特殊值时的特殊行为:

* 如果操作数都是数值，则执行常规的除法运算，即两个正值相除是正值，两个负值相除也是正值，
  符号不同的值相除得到负值。如果 ECMAScript 不能表示商，则返回 `Infinity` 或 `-Infinity`
* 返回 `NaN`
    - 如果有任一操作数是 `NaN`，则返回 `NaN`
    - 如果是 `Infinity` 除以 `Infinity`，则返回 `NaN`
    - 如果是 0 除以 0，则返回 `NaN`
* 返回 `Infinity` 或 `-Infinity`
    - 如果是非 0 的有限值除以 0，则根据第一个操作数的符号返回 `Infinity` 或 `-Infinity`
    - 如果是 `Infinity` 除以任何数值，则根据第二个操作数的符号返回 `Infinity` 或 `-Infinity`
* 如果有不是数值的操作数，则先在后台用 `Number()` 函数将其转换为数值，然后再应用上述规则

### 取模操作符

* 如果操作数是数值，则执行常规除法运算，返回余数
* 如果被除数是无限值，除数是有限值，则返回 `NaN`
* 如果被除数是有限值，除数是 0，则返回 `NaN`
* 如果被除数是 0，除数不是 0，则返回 0
* 如果是 `Infinity` 除以 `Infinity`，则返回 NaN
* 如果被除数是有限值，除数是无限值，则返回被除数
* 如果有不是数值的操作数，则先在后台用 `Number()` 函数将其转换为数值，然后再应用上述规则

## 指数操作符

ECMAScript 7 新增了指数操作符，`Math.pow()` 现在有了自己的操作符 `**`

```js
console.log(Math.pow(3, 2));  // 9
console.log(3 ** 2);  // 9
```

指数操作符也有自己的指数赋值操作符 `**=`，该操作符执行指数运算和结果的赋值操作

```js
let squared = 3;
squared **= 2;
console.log(squared);  // 9
```

## 加性操作符

### 加法操作符

如果两个操作数都是数值，加法操作符执行加法运算并根据如下规则返回结果: 

* 如果有任一操作数是 `NaN`，则返回 `NaN`
* 如果是 `Infinity` 加 `Infinity`，则返回 `Infinity`
* 如果是 `-Infinity` 加 `-Infinity`，则返回 `-Infinity`
* 如果是 `Infinity` 加 `-Infinity`，则返回 `NaN`
* 如果是 +0 加 +0，则返回 +0
* 如果是 -0 加 +0，则返回 +0
* 如果是 -0 加 -0，则返回 -0

不过，如果有一个操作数是字符串，则要应用如下规则:

* 如果两个操作数都是字符串，则将第二个字符串拼接到第一个字符串后面
* 如果只有一个操作数是字符串，则将另一个操作数转换为字符串，再将两个字符串拼接在一起

如果有任一操作数是对象、数值或布尔值，则调用它们的 `toString()` 方法以获取字符串，
然后再应用前面的关于字符串的规则

对于 `undefined` 和 `null`，则调用 `String()` 函数，分别获取 "undefined" 和 "null"

### 减法操作符

减法操作符也有一组规则用于处理 ECMAScript 中不同类型之间的转换。

* 如果两个操作数都是数值，则执行数学减法运算并返回结果
* 返回 `NaN`
    - 如果有任一操作数是 `NaN`，则返回 `NaN`
    - 如果是 `Infinity` 减 `Infinity`，则返回 `NaN`
    - 如果是 `-Infinity` 减 `-Infinity`，则返回 `NaN`
* 如果是 `Infinity` 减 `-Infinity`，则返回 `Infinity`
* 如果是 `-Infinity` 减 `Infinity`，则返回 `-Infinity`
* 如果是 +0 减 +0，则返回 +0
* 如果是 +0 减 -0，则返回 -0
* 如果是 -0 减 -0，则返回 +0
* 如果有任一操作数是字符串、布尔值、`null` 或 `undefined`，
  则先在后台使用 `Number()` 将其转换为数值，
  然后再根据前面的规则执行数学运算。
  如果转换结果是 `NaN`，则减法计算的结果是 `NaN`
* 如果有任一操作数是对象，则调用其 `valueOf()` 方法取得表示它的数值。
  如果该值是 `NaN`，则减法计算的结果是 `NaN`。
  如果对象没有 `valueOf()` 方法，则调用其 `toString()` 方法，
  然后再将得到的字符串转换为数值


## 关系操作符

关系操作符执行比较两个值的操作:

* `<`
* `>`
* `<=`
* `>=`

### 规则

与 ECMAScript 中的其他操作符一样，
在将它们应用到不同数据类型时也会发生类型转换和其他行为:

* 如果操作数都是数值，则执行数值比较
* 如果操作数都是字符串，则逐个比较字符串中对应字符的编码
* 如果有任一操作数是数值，则将另一个操作数转换为数值，执行数值比较
* 如果有任一操作数是对象，则调用其 `valueOf()` 方法，
  取得结果后再根据前面的规则执行比较。 如果没有 `valueOf()` 操作符，
  则调用 `toString()` 方法，取得结果后再根据前面的规则执行比较
* 如果有任一操作数是布尔值，则将其转换为数值再执行比较

### 字符串比较

在使用关系操作符比较两个字符串时，会发生一个有趣的现象。
很多人认为小于意味着“字母顺序靠前”，而大于意味着“字母顺序靠后”，
实际上不是这么回事。对字符串而言，关系操作符会比较字符串中对应字符的编码，
而这些编码是数值。比较完之后，会返回布尔值。
问题的关键在于，大写字母的编码都小于小写字母的编码

* 要得到确实按字母顺序比较的结果，就必须把两者都转换为相同的大小写形式(全大写或全小写)，
然后再比较

```js
let result = "Brick".toLowerCase() < "alphabet".toLowerCase();  
// false
```

* 在比较两个数值字符串的时候，比较奇怪

```js
let result = "23" < "3"; // true
let result = "23" < 3;  // false
```

* 任何关系操作符在涉及比较 `NaN` 时都返回 `false`

```js
let result1 = NaN < 3;  // false
let result2 = NaN >= 3;  // false
```

## 相等操作符

在比较字符串、数值和布尔值是否相等时, 过程都很直观。
但是在比较两个对象是否相等时, 情形就比较复杂了

ECMAScript 中的相等和不相等操作符, 原本在比较之前会执行类型转换, 
但很快就有人质疑这种转换是否应该发生。最终, ECMAScript 提供了两组操作符, 

* 第一组是**等于和不等于**, 它们在比较之前执行转换(强制类型转换)
* 第二组是**全等和不全等**, 它们在比较之前不执行转换

### 等于和不等于

* 语法
	- `==`
	- `!=`
* 规则
	- (1) 如果任一操作数是布尔值, 则将其转换为数值再比较是否相等。`false` 转换为 0, `true` 转换为 1
	- (2) 如果一个操作数是字符串, 另一个操作数不是, 则尝试将字符串转换为数值, 再比较是否相等
	- (3) 如果一个操作数是对象, 另一个操作数不是, 则调用对象的 `valueOf()` 方法取得其原始值, 
	  再根据前面的规则进行比较
	- (4) `null` 和 `undefined` 相等
	- (5) `null` 和 `undefined` 不能转换为其他类型的值再进行比较
	- (6) 如果有任一操作数是 `NaN`，则相等操作符返回 `false`, 不相等操作符返回 `true`
	  记住: 即使两个操作数都是 `NaN`, 相等操作符也返回 `false`, 因为按照规则, `NaN` 不等于 `NaN`
	- (7) 如果两个操作数都是对象, 则比较它们是不是同一个对象。
	  如果两个操作数都指向同一个对象, 则相等操作符返回 `true`, 否则, 两者不相等
* 示例

| 表达式            | 结果  | 遵循的规则 |
| ----------------- | ----- | ---------- |
| null == undefined  | true  | (4)        |
| "NaN" == NaN      | false | (6)        |
| 5 == NaN          | false | (6)        |
| NaN == NaN        | fasle | (6)        |
| NaN != NaN        | true  | (6)        |
| false == 0        | true  | (1)        |
| true == 1         | true  | (1)        |
| true == 2         | false | (1)        |
| undefined == 0     | false | (4)        |
| null == 0         | false | (4)        |
| "5" == 5          | true  | (2)        |

### 全等和不全等

* 语法
	- `===`
	- `!==`
* 示例

```js
let result1 = ("55" == 55); // true, 转换后相等
let result2 = ("55" === 55); // false, 不相等, 因为数据类型不同
```

```js
let result1 = ("55" != 55); // false, 转换后相等
let result2 = ("55" !=== 55); // true, 不相等, 因为数据类型不同
```

- 示例

| 表达式             | 结果  |
| ------------------ | ----- |
| null == undefined  | true  |
| null === undefined | false |

> 由于相等和不相等操作符存在类型转换问题, 因此推荐使用全等和不全等操作符。
  这样有助于在代码中保持数据类型的完整性





## 条件操作符

* 语法

```js
variable = boolean_expression ? true_value : false_value;
```

* 示例

```js
let max = (num1 > num2) ? num1 : num2;
```

## 赋值操作符

* 简单赋值
	- `=`
* 复合赋值：使用乘性、加性、位操作符后跟等于号表示
  这些操作仅仅是简写语法，使用它们不会提升性能
	- 乘后赋值 `*=`
	- 除后赋值 `/=`
	- 取模后赋值 `%=`
	- 加后赋值 `+=`
	- 减后赋值 `-=`
	- 左移后赋值 `<<=`
	- 右移后赋值 `>>=`
	- 无符号右移后赋值 `>>>=`
    - 指数赋值操作符 `**=`

## 逗号操作符

逗号操作符可以用来在一条语句中执行多个操作

* 在一条语句中同时声明多个变量是逗号操作符最常用的场景
* 也可以使用逗号操作符来辅助赋值。在赋值时使用逗号操作符分隔值, 
  最终会返回表达式中最后一个值

```js
let num1 = 1, num2 = 2, num3 = 3;
console.log(num1); // 1
console.log(num2); // 2
console.log(num3); // 3
```

```js
let num = (5, 1, 4, 8, 0); 
console.log(num);  // 0
```


# 语句

## if

* 语法
    - `condition` 可以是任何表达式，并且求值结果不一定是布尔值，
      ECMAScript 会自动调用 `Boolean()` 函数将这个表达式的值转换为布尔值

```js
if (condition) {
    statement1
} 
else {
    statement2
}
```

```js   
if (condition) {
    statement1
}
else if (condition2) {
    statement2
}
else {
    statement3
}
```

## do-while

`do-while` 语句是一种后测试循环语句, 
即循环体中的代码执行后才会对退出条件进行求值。
换句话说, 循环体内的代码至少执行一次

* 语法

```js
do {
    statement
} while (expression);
```

## while

* 语法

```js
while (expression) {
    statement
}
```

## for

### 基本用法

* 语法

```js
for (initialization; expression; post-loop-expression) {
    statement
}
```

* 示例

```js
let count = 10;
for (let i = 0; i < count; i++) {
    console.log(i);
}
```

### 关键字声明使用 let

在 `for` 循环的初始化代码中, 其实是可以不使用变量声明关键字的。
不过, 初始化定义的迭代器变量在循环执行完后几乎不可能再用到了。
因此, 最清晰的写法是使用 `let` 声明迭代器变量, 
这样就可以将这个变量的作用域限定在循环中

### 初始化,条件表达式和循环后表达式都不是必需的

* 无穷循环

```js
for (;;) {
    doSomething();
}
```

### for 循环等价于 while 循环

* 如果只包含条件表达式, 那么 for 循环实际上就变成了 while 循环

```js
let count = 10;
let i = 0;
for (; i < count; ) {
    console.log(i);
    i++;
}
```

## for-in

`for-in` 语句是一种严格的迭代语句, 用于枚举对象中的非符号键属性

ECMAScript 中的对象的属性是无序的, 因此 `for-in` 语句不能保证返回对象属性的顺序。
换句话说, 所有可枚举的属性都会返回一次, 但返回的顺序可能会因浏览器而异

如果 `for-in` 循环要迭代的变量是 `null` 或 `undefined`, 则不执行循环体

* 语法

```js
for (property in expression) {
    statement
}
```

* 示例

```js
// 控制语句中的 const 不是必需的,
// 但是为了确保这个局部变量不被修改, 推荐使用 const
for (const propName in window) { 
    document.write(propName);
}
```

## for-of

`for-of` 语句是一种严格的迭代语句, 用于遍历可迭代对象的元素

* 语法

```js
for (property of expression) {
    statement
}
```

* 示例

```js
// 控制语句中的 const 不是必需的, 
// 但是为了确保这个局部变量不被修改, 推荐使用 const
for (const el of [2, 4, 6, 8]) {
    document.write(el);
}
```

## label

标签语句用于给语句加标签, 标签可以在语句后面通过 break 或 continue 语句引用。
标签语句的典型应用场景是嵌套循环

* 语法

```js
label: statement
```

* 示例

```js
start: 
for (let i = 0; i < count; i++) {
    console.log(i);
}
```

## break 和 continue

`break` 和 `continue` 语句为执行循环代码提供了更严格的控制手段:

* `break` 语句用于立即退出循环，强制执行循环后的下一条语句
* `continue` 语句也用于立即退出循环，但会再次从循环顶部开始执行

### 基本用法

* `break`

```js
let num = 0;;
for (let i = 1; i< 10; i++) {
    if (i % 5 == 0) {
        break;
    }
    num++;
}
console.log(num);  // 4
```

* `continue`

```js
let num = 0;
for (let i = 1; i < 10; i++) {
    if (i % 5 == 0) {
        continue;
    }
    num++;
}
console.log(num);  // 8
```

### 与 label 语句一起使用

`break` 和 `continue` 都可以与标签语句一起使用，返回代码中特定的位置。
这通常是在嵌套循环中

组合使用标签语句和 `break`、`continue` 能实现复杂的逻辑，但也容易出错。
注意标签要使用描述 性强的文本，而嵌套也不要太深

```js
let num = 0;

outermost:
for (let i = 0; i< 10; i++) {
    for (let j = 0; j < 10; j++) {
        if (i == 5 && j == 5) {
            break outermost;
        }
        num++;
    }
}
console.log(num);  // 55
```

```js
let num = 0;

outermost:
for (let i = 0; i < 10; i++) {
    for (let j = 0; j < 10; j++) {
        if (i == 5 && j == 5) {
            continue outermost;
        }
        num++;
    }
}
console.log(num);  // 95
```



## with

with 语句的用途是将代码作用域设置为特定的对象

* 严格模式不允许使用 `with` 语句，否则会抛出错误
* 由于 `with` 语句影响性能且难于调试其中的代码，通常不推荐在产品代码中使用 `with` 语句

### 语法

```js
with (expression) statement;
```

### 使用场景

使用 `with` 语句的主要场景是针对一个对象反复操作，
这时候将代码作用域设置为该对象能提供便利

```js
let qs = location.search.substring(1);
let hostname = location.hostname;
let url = location.href;

// 用 with 实现
with (location) {
    let qs = search.substring(1);
    let hostname = hostname;
    let url = href;
}
```

`with` 语句用于连接 `location` 对象，这意味着在这个语句内部，
每个变量首先会被认为是一个局部变量。如果没有找到局部变量，则会搜索 `location` 对象，
看它是否有一个同名的属性。如果有，则该变量会被求值为 `location` 对象的属性

## switch

`switch` 语句是与 `if` 语句紧密相关的一种流控制语句

### 语法

`switch` 语句在比较每个条件的值时会使用全等操作符(`===`)，因此不会强制转换数据类型

为了避免不必要的条件判断，最好给每个条件后面都加上 `break` 语句。
如果确实需要连续匹配几个条件，那么推荐写个注释表明是故意忽略了 `break`

```js
switch (expression) {
    case value1: 
        statement
        break
    case value2:
        statement
        break
    case value3:
        statement
        break
    case value4:
        statement
        /*跳过*/
    default:
        statement
}
```

### 示例

```js
if (i == 25) {
    console.log("25");
} else if (i == 35) {
    console.log("35");
} else if (i == 45) {
    console.log("45");
} else {
    console.log("Other");
}
```

等价于:

```js
switch (i) {
    case 25:
        console.log("25");
        break
    case 35:
        console.log("35");
        break
    case 45:
        conosole.log("45");
    default:
        console.log("Other");
}
```

### 特性

虽然 switch 语句是从其他语言借鉴过来的，但 ECMAScript 为它赋予了一些独有的特性:

* `switch` 语句可以用于所有数据类型，因此可以使用字符串甚至对象
* 条件的值不需要是常量，也可以是变量或者表达式

```js
switch ("hello world") {
    case "hello" + "world":
        console.log("Greeting was found.");
        break;
    case "goodbye":
        console.log("Closing was found.");
        break;
    default:
        console.log("Unexpected message was found.")
}
```

```js
let num = 25;
switch (true) {
    case num < 0:
        console.log("Less than 0.");
        break;
    case num >= 0 && num <= 10:
        console.log("Between 0 and 10.");
        break;
    case num > 10 && num <= 20:
        console.log("Between 10 and 20.");
        break;
    default:
        console.log("More than 20.");
}
```



# 函数

ECMAScript 中的函数使用 `function` 关键字声明，后跟一组参数，然后是函数体

* 函数基本语法

```js
function functionName(arg0, arg1, ..., argN) {
    statements
}
```

* 示例

```js
function sayHi(name, message) {
    console.log("Hello " + name + ", " + message);
}
sayHi("Nicholas", "how are you today?");
```

* ECMAScript 中的函数不需要指定是否返回值, 任何函数在任何时间都可以使用 return 语句来返回函数的值, 
  用法是后跟要返回的值, 只要碰到 return 语句, 函数就立即停止执行并退出
* 最佳实践是函数要么返回值, 要么不返回值。只在某个条件下返回值的函数会带来麻烦, 尤其是调试时
* 严格模式对函数也有一些限制: 
    - 函数不能以 `eval` 或 `arguments` 作为名称
    - 函数的参数不能叫做 `eval` 或 `arguments`
    - 两个命名参数不能拥有同一个名称





