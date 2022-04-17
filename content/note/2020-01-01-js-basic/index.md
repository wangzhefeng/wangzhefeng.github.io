---
title: 语言基础
author: 王哲峰
date: '2020-01-01'
slug: js-basic
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
  - [let 声明](#let-声明)
    - [let 声明变量](#let-声明变量)
    - [暂时性死区](#暂时性死区)
    - [全局声明](#全局声明)
    - [条件声明](#条件声明)
    - [for 循环中的 let 声明](#for-循环中的-let-声明)
  - [const 声明](#const-声明)
    - [const 声明变量](#const-声明变量)
    - [声明风格及最佳实践](#声明风格及最佳实践)
- [数据类型](#数据类型)
  - [typeof 操作符](#typeof-操作符)
  - [Undefined](#undefined)
  - [Null](#null)
  - [Boolean](#boolean)
  - [Number](#number)
  - [String](#string)
  - [Symbol](#symbol)
    - [符号的基本用法](#符号的基本用法)
    - [使用全局符号注册表](#使用全局符号注册表)
    - [使用符号作为属性](#使用符号作为属性)
    - [常用内置符号](#常用内置符号)
    - [Symbol.asyncIterator](#symbolasynciterator)
  - [Object](#object)
- [操作符](#操作符)
  - [一元操作符](#一元操作符)
  - [位操作符](#位操作符)
  - [布尔操作符](#布尔操作符)
  - [乘性操作符](#乘性操作符)
  - [指数操作符](#指数操作符)
  - [加性操作符](#加性操作符)
  - [关系操作符](#关系操作符)
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
  - [for-in](#for-in)
  - [for-of](#for-of)
  - [label](#label)
  - [break](#break)
  - [continue](#continue)
  - [with](#with)
  - [switch](#switch)
- [函数](#函数)
  - [函数基本知识](#函数基本知识)
</p></details><p></p>

# 语法

## 区分大小写

* JavaScript 区分大小写

## 标识符

- 标识符包含: 变量、函数、属性、函数参数
- 字母、下划线、美元符号开头
- 字母、数字、下划线、美元符号组成
- 惯例: 驼峰大小写, 第一个单词的首字母小写, 后面每个单词的首字母大写

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

- 严格模式是一种不同的 JavaScript 解析和执行模型, 
  ECMAScript 3 的一些不规范写法在这种模式下会被处理, 
  对于不安全的活动将抛出错误。选择这种语法形式的目的是不破坏 ECMAScript 3 语法
- 要对整个脚本启用严格模式, 在脚本的开头加上 `"use strict";`
- 所有现代浏览器都支持严格模式

## 语句

- 语句以分号结尾, 不是必须的, 建议加
- 多条语句可以合并到一个 C 语言风格的代码块中。代码块由 `{}` 包含。
  控制流语句在执行多条语句时要求必须有代码块, 
  最佳实践是始终在控制语句中使用代码块

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
|`await`       |:black_square_button:|

# 变量

## var 声明

### var 定义变量

- 定义不初始化

```js
var message;  // undefined
```

- 定义并初始化
    - 只是一个简单的赋值, 可以改变保存的值, 也可以改变值的类型

```js
var message = "hi"
message = 100;  // 合法, 但不推荐
```

- 重复声明

```js
var name = "Matt";
var name = "John";
console.log(name);  // "John"
```

### var 声明作用域

- 使用 var 操作符定义的变量会成为包含它的函数的局部变量

```js
function test() {
    var message = "hi";  // 局部变量
}
test();
console.log(message);  // 出错
```

- 在函数内部定义变量时省略 var 操作符可以创建一个全局变量

```js
function test() {
    message = "hi";  // 全局变量
}
test();
console.log(message);  // "hi"
```

- 定义多个变量

```js
var message = "hi", found = false, age = 29;
```

<div class="warning" 
     style='background-color:#E9D8FD; color: #69337A; border-left: solid #805AD5 4px; border-radius: 4px; padding:0.7em;'>
    <span>
        <p style='margin-top:1em; text-align:left'>
            <b>Note</b>
        </p>
        <p style='margin-left:1em;'>
            在严格模式下, 不能定义名为 eval 和 arguments 的变量, 否则会导致语法错误
        </p>
        <p style='margin-bottom:1em; margin-right:1em; text-align:right; font-family:Georgia'> 
            <b></b> 
            <i></i>
        </p>
    </span>
</div>

### var 声明提升

- 使用 var 关键字声明的变量会自动提升(hoist)到函数作用域顶部, 
  所谓的“提升”(hoist)，也就是把所有变量声明都拉到函数作用域的顶部 

```js
function foo() {
    console.log(age);
    var age = 26;
}
foo(); // undefined

// 等价代码
function foo() {
    var age;
    console.log(age);
    age = 26;
}
foo(); // undefined
```

- 反复多次使用 var 声明统一变量也没有问题

```js
function foo() {
    var age = 16;
    var age = 26;
    var age = 36;
    console.log(age);
}
foo(); // 36
```

## let 声明

### let 声明变量

* let 跟 var 最明显的区别是: 
    - let 声明的范围是**块**作用域
    - var 声明的范围是**函数**作用域

```js
if (true) {
    var name = "Matt";
    console.log(name);  // Matt
}
console.log(name);  // Matt
```

```js
if (true) {
    let age = 26;
    console.log(age);  // 26
}
console.log(age);  // ReferenceError: age 没有定义
```

- let 不允许同一个块作用域中出现冗余声明, 会导致报错

```js
var name;
var name;

let age;
let age; // //SyntaxError;标识符age已经声明过了
```

- JavaScript 引擎会记录用于变量声明的标识符及其所在的块作用域, 
  因此嵌套使用相同的标识符不会报错，这是因为同一个块中没有重复声明

```js
// var
var name = "Nicholas";
console.log(name); // "Nicholas"

if (true) {
    var name = "Matt";
    console.log(name); // "Matt"
}

// let
let age = 30;
console.log(age); //30

if (true) {
    let age = 26;
    console.log(age); // 26
}
```

- 对声明冗余报错不会因为混用 `let` 和 `var` 而受影响。
  这两个关键字声明的并不是不同类型的变量，它们只是指出变量在相关作用域如何存在

```js
var name;
let name; // SyntaxError

let age;
var age; // SyntaxError
```

### 暂时性死区

* let 声明的变量不会在作用域中被提升
    - 在 let 声明之前的执行瞬间被称为"暂时性死区"(temporal dead zone), 
      在此阶段引用任何后面才声明的变量都会抛出 ReferenceError。

```js
// name 会被提升
console.log(name); // undefined
var name = "Matt";

// age 不会被提升
console.log(age); // ReferenceError: age 没有定义
let age = 26;
```

### 全局声明

- 使用 let 在全局作用域中声明的变量不会成为 window 对象的属性, var 声明的变量则会

```js
var name = "Matt";
console.log(window.name); // "Matt"

let age = 26;
console.log(window.age); // undefined
```

### 条件声明

- 因为 let 的作用域是块, 所以不可能检查前面是否已经使用 let 声明过同名变量, 
  同时也就不可能在没有声明的情况下声明它。而 var 声明变量时, 由于声明会被提升, 
  JavaScript 引擎会自动将多余的声明在作用域顶部合并为一个声明
- let 声明不能依赖条件声明模式

```html
<script>
    var name = "Nicholas";
    let age = 26;
</script>

<script>
    // 假设脚本不确定页面中是否已经声明了同名变量, 那么可以假设还没有声明过
    // 这里没有问题, 因为可以被作为一个提升声明来处理, 不需要检查之前是否声明过同名变量
    var name = "Matt";
    
    // 如果 age 之前声明过, 这里会报错
    let age = 26;
</script>
```

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
    age = 26;          // 因此这个赋值形同全局赋值
</script>
```

### for 循环中的 let 声明

- 在 let 出现之前, for 循环定义的迭代变量会渗透到循环体外部; 使用 let 之后, for 循环定义的迭代变量不会渗透到循环体外部

```js
for (var i = 0; i < 5; ++i) {
    console.log("hello world!");
}
console.log(i); // 5

for (let i = 0; i < 5; ++i) {
    console.log("hello world!");
}
console.log(i); // ReferenceError: i 没有定义
```

- 使用 var 的时候, 在退出循环的时, 迭代变量保存的是导致循环退出的值, 之后执行超时逻辑时, 所有的迭代变量都是同一个变量; 而使用 let 声明迭代变量时, JavaScript 引擎在后台为每个循环声明一个新的迭代变量

```js
for (var i = 0; i < 5, ++i) {
    setTimeout(() => console.log(i), 0) // 5 5 5 5 5
}

for (let i = 0; i < 5; ++i) {
    setTimeout(() => console.log(i), 0) // 0 1 2 3 4
}
```

## const 声明

### const 声明变量

- const 声明与 let 声明唯一一个重要区别是它声明变量时必须同时初始化变量, 且尝试修改 const 声明的变量会导致运行时错误

```js
const age = 26;
age = 36; // TypeError: 给常量赋值
```

- const 不允许重复声明

```js
const name = "Matt";
const name = "John";
console.log(name); // SyntaxError: Identifier 'name' has already been declared
```

- const 声明的作用域也是块

```js
const name = "Matt";
if (true) {
    const name = "Nicholas";
}
console.log(name); // "Matt"
```

- const 声明的限制只适用于它指向的变量的引用, 如果 const 变量引用的是一个对象, 那么修改这个对象内部的属性并不违反 const 对象不能修改变量的限制

```js
const person = {};
person.name = "Matt"; // ok
```

- 不能用 const 来声明迭代变量(因为迭代变量会自增), JavaScript 引擎会为 for 循环中的 let 声明分别创建独立的变量实例, 但 const 不行; 但是, 如果用 const 声明一个不被修改的 for 循环变量, 那是可以的, 也就是说, 每次迭代只是创建一个新变量, 这对 for-of 和 for-in 循环特别有意义

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

### 声明风格及最佳实践

- 不使用 var

    - 有了 let 和 const, 大多数开发者会发现自己不再需要 var 了。限制自己只使用 let 和 const 有助于提升代码质量, 因为变量有了明确的作用域、声明位置, 以及不变的值。

- const 优先, let 次之

    - 使用 const 声明可以让浏览器运行时强制保持变量不变, 也可以让静态代码分析工具提前发现不合法的赋值操作。因此, 很多开发者认为应该优先使用 const 来声明变量。
    - 只在提前知道未来会有修改时, 再使用 let。这样可以让开发者更有信心地推断某些变量的值永远不会变, 同时也能迅速发现因意外赋值导致的非预期行为。

# 数据类型

ECMAScript 有 6 种简单数据类型(也称为原始类型): 

- Undefined
- Null
- Boolean
- Number
- String
- Symbol

1 种复杂数据类型: 

- Object (对象)

    - 无序名值对的集合

## typeof 操作符

- 因为 ECMAScript 的类型系统是松散的, 所以需要一种手段来确定任意变量的数据类型。typeof 操作符就是为此而生的, 对一个值使用 typeof 操作
	- "undefined" 表示值未定义
	- "boolean" 表示布尔值
	- "string" 表示值为字符串
	- "number" 表示值为数值
	- "object" 表示值为对象(而不是函数) 或 null
	- "function" 表示值为函数
	- "symbol" 表示值为符号
- 示例

```js
let message = "some string";
console.log(typeof message); // "string"
console.log(typeof(message)); // "string"
console.log(typeof 95); // number
console.log(typeof null); // object
```

## Undefined



## Null

## Boolean

## Number

## String

## Symbol

- Symbol(符号) 是 ECMAScript 6 新增的数据类型
- 符号是原始值, 且符号实例是唯一、不可变的
- 符号的用途是确保对象属性使用唯一标识符, 不会发生属性冲突的危险
- 尽管听起来跟私有属性有点类似, 但符号并不是为了提供私有属性的行为才增加的(尤其是因为 Object API 提供了方法, 可以更方便地发现符号属性)。相反, 符号就是用来创建唯一记号, 进而用作非字符串形式的对象属性

### 符号的基本用法

- 符号需要使用 Symbol() 函数初始化。因为符号本身是原始类型, 所以 typeof 操作符对符号返回 symbol

```js
let sym = Symbol();
console.log(typeof sym); // symbol
```

- 调用 Symbol() 函数时, 可以传入一个字符串参数作为对符号的描述(description), 将来可以通过这个字符串来调试代码, 但是这个字符串参数与符号定义或表示完全无关

```js
let genericSymbol = Symbol();
let otherGenericSymbol = Symbol();
let fooSymbol = Symbol("foo");
let otherFooSymbol = Symbol("foo");

console.log(genericSymbol == otherGenericSymbol); // false
console.log(fooSymbol == otherFooSymbol); // false
```

- 符号没有字面量语法, 这也是它们发挥作用的关键。按照规范, 你只要创建 `Symbol()` 实例并将其用作对象的新属性, 就可以保证它不会覆盖已有的对象属性, 无论是符号属性还是字符串属性。

```js
let genericSymbol = Symbol();
console.log(genericSymbol); // Symbol()

let fooSymbol = Symbol("foo");
console.log(fooSymbol); // Symbol(foo)
```

- Symbol() 函数不能与 new 关键字一起作为构造函数使用。这样做是为了避免创建符号包装对象, 像使用 Boolean、String 或 Number 那样, 它们都支持构造函数且可用于初始化包含原始值的包装对象

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

### 使用符号作为属性

### 常用内置符号

### Symbol.asyncIterator

## Object

- ECMAScript 中的`对象(object)`其实就是一组数据和功能的组合

- 严格来讲, ECMA-262 中对象的行为不一定适合 JavaScript 中的其他对象

    - 比如浏览器环境中的 BOM 对象 和 DOM 对象, 都是由宿主环境定义和提供的宿主对象
    - 而宿主对象不受 ECMA-262 约束, 所以它们可能会也可能不会继承 Object

- 对象通过 `new` 操作符后跟对象类型的名称来创建, 可以通过创建 Object 类型的实例来创建自己的对象, 然后再给对象添加属性和方法

- Object 的实例本身并不是很有用, 但理解与它相关的概念非常重要。类似 Java 中的 java.lang.Object, ECMAScript 中的 Object 也是派生其他对象的基类。Object 类型的所有属性和方法在派生的对象上同样存在。每个Object 实例都有如下属性和方法: 

    - constructor

        - 用于创建当前对象的函数

    - hasOwnProperty(propertyName)

        - 用于判断当前对象实例(不是原型)上是否存在给定的属性。要检查的属性名必须是字符串或符号

    - isPrototypeOf(object)

        - 用于判断当前对象是否为另一个对象的原型

    - propertyIsEnumerable(propertyName)

        - 用于判断给定的属性是否可以使用 for-in 语句枚举。属性名必须是字符串或符号

    - toLocaleString()

        - 返回对象的字符串表示, 该字符反映对象所在的本地化执行环境

    - toString()

        - 返回对象的字符串表示

    - valueOf()

        - 返回对象对应的字符串、数值或布尔值表示。通常与 toString() 的返回值相同

```js
let o1 = new Object();
let o2 = new Object; // ECMAScript 只要求在给构造函数提供参数时使用括号, 合法, 但不推荐

o2.name = "wangzf";
console.log(o2.constructor);
console.log(o2.hasOwnProperty("name"));
console.log(o2.isPrototypeOf(o1));
console.log(o2.propertyIsEnumerable("name"));
console.log(o2.toLocaleString());
console.log(o2.toString());
console.log(o2.valueOf("name"));
```

# 操作符

ECMAScript 中的操作符是独特的, 因为它们可用于各中值, 包括字符串、数值、布尔值、对象。在应用给对象时, 操作符通常会调用 valueOf() 和/或 toString() 方法来取得可以计算的值.

- 数学操作符
- 位操作符
- 关系操作符
- 相等操作符

## 一元操作符

- 递增/递减操作符
- 一元加和减
	- 一元加由一个加号(+)表示, 
	- 一元加放在变量前头, 对数值没有任何影响
	- 一元加放在非数值, 则会执行与使用 Number() 转型函数一样的类型转换
		- 布尔值 false 和 true 转换为0 和 1

## 位操作符

## 布尔操作符

- 语法

	- 逻辑非 `!` 

		- 这个操作符始终返回布尔值, 无论应用的是什么数据类型, 逻辑非操作符首先将操作数转换为布尔值, 然后再对其取反

		- 规则

			| 表达式                 | 结果  |
			| ---------------------- | ----- |
			| !对象                  | false |
			| !空字符串              | true  |
			| !非空字符串            | false |
			| !0                     | true  |
			| !非0数值(包括Infinity) | false |
			| !null                  | true  |
			| !NaN                   | true  |
			| !undefined             | true  |

		- `!!`: 相当于调用了转型函数 Boolean() , 无论操作数是什么类型, 第一个叹号总会返回布尔值, 第二个叹号对该布尔值取反, 从而给出变量真正对应的布尔值, 结果与对同一个值使用 Boolean() 函数是一样的

			- 示例

			```js
			console.log(!!"blue"); // true
			console.log(!!0); // false
			console.log(!!NaN); // false
			console.log(!!""); // false
			console.log(!!12345); // true
			```

	- 逻辑与 `&&`

		- 逻辑与操作符遵循的真值表

			| 第一个操作数 | 第二个操作数 | 结果  |
			| ------------ | ------------ | ----- |
			| true         | true         | true  |
			| true         | false        | false |
			| false        | true         | false |
			| false        | false        | false |

		- 

	- 逻辑或 `||`
		- 逻辑或操作符遵循的真值表

			| 第一个操作数 | 第二个操作数 | 结果  |
			| ------------ | ------------ | ----- |
			| true         | true         | true  |
			| true         | false        | true  |
			| false        | true         | true  |
			| false        | false        | false |

		- 

## 乘性操作符

## 指数操作符

## 加性操作符

## 关系操作符



## 相等操作符

- 在比较字符串、数值和布尔值是否相等时, 过程都很直观。但是在比较两个对象是否相等时, 情形就比较复杂了
- ECMAScript 中的相等和不相等操作符, 原本在比较之前会执行类型转换, 但很快就有人质疑这种转换是否应该发生。最终, ECMAScript 提供了两组操作符, 第一组是等于和不等于, 它们在比较之前执行转换(强制类型转换)。第二组是全等和不全等, 它们在比较之前不执行转换

### 等于和不等于

- 语法
	- `==`
	- `!=`
- 规则
	- (1) 如果任一操作数是布尔值, 则将其转换为数值再比较是否相等。false转换为0, true转换为1
	- (2) 如果一个操作数是字符串, 另一个操作数不是, 则尝试将字符串转换为数值, 再比较是否相等
	- (3) 如果一个操作数是对象, 另一个操作数不是, 则调用对象的 valueOf() 方法取得其原始值, 再根据前面的规则进行比较
	- (4) null 和 undefined 相等
	- (5) null 和 undefined 不能转换为其他类型的值再进行比较
	- (6) 如果有任一操作数是 NaN,则相等操作符返回 false, 不相等操作符返回 true。记住: 即使两个操作数都是NaN, 相等操作符也返回 false, 因为按照规则, NaN 不等于 NaN
	- (7) 如果两个操作数都是对象, 则比较它们是不是同一个对象。如果两个操作数都指向同一个对象, 则相等操作符返回 true, 否则, 两者不相等
- 示例

| 表达式            | 结果  | 遵循的规则 |
| ----------------- | ----- | ---------- |
| null == undefined | true  | (4)        |
| "NaN" == NaN      | false | (6)        |
| 5 == NaN          | false | (6)        |
| NaN == NaN        | fasle | (6)        |
| NaN != NaN        | true  | (6)        |
| false == 0        | true  | (1)        |
| true == 1         | true  | (1)        |
| true == 2         | false | (1)        |
| undefined == 0    | false | (4)        |
| null == 0         | false | (4)        |
| "5" == 5          | true  | (2)        |

### 全等和不全等

- 语法
	- `===`
	- `!==`
- 示例

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

> 由于相等和不相等操作符存在类型转换问题, 因此推荐使用全等和不全等操作符。这样有助于在代码中保持数据类型的完整性.

## 条件操作符

- 语法

```js
variable = boolean_expression ? true_value : false_value;
```

- 示例

```js
let max = (num1 > num2) ? num1 : num2;
```



## 赋值操作符

- 简单赋值
	- `=`
- 复合赋值
	- `*=`
	- `/=`
	- `%=`
	- `+=`
	- `-=`
	- 左移后赋值 `<<=`
	- 右移后赋值 `>>=`
	- 无符号右移后赋值 `>>>=`

- 示例: 

```js
let num = 10;
num = num + 10;
console.log(num);
```

等价于:

```js
let num = 10;
num += 10;
```

## 逗号操作符

- 逗号操作符可以用来在一条语句中执行多个操作

    - 在一条语句中同时声明多个变量是逗号操作符最常用的场景
    - 也可以使用逗号操作符来辅助赋值。在赋值时使用逗号操作符分隔值, 最终会返回表达式中最后一个值

```js
let num1 = 1, num2 = 2, num3 = 3;
console.log(num1); // 1
console.log(num2); // 2
console.log(num3); // 3

let num = (5, 1, 4, 8, 0); 
console.log(num);  // 0
```



# 语句

## if

- 语法

```js
if (condition) statement1 
else statement2
```

```js
if (condition) 
    statement1 
else if (condition2) 
    statement2 
else 
    statement3
```

## do-while

- do-while 语句是一种后测试循环语句, 即循环体中的代码执行后才会对退出条件进行求值。
  换句话说, 循环体内的代码至少执行一次
- 语法

```js
do {
    statement
} while (expression);
```

## while

- 语法

```js
while (expression) 
    statement
```

## for

- 语法

```js
for (initialization; expression; post-loop-expression) 
    statement
```

- 在 for 循环的初始化代码中, 其实是可以不使用变量声明关键字的。
  不过, 初始化定义的迭代器变量在循环执行完后几乎不可能再用到了。
  因此, 最清晰的写法是使用 let 声明迭代器变量, 这样就可以将这个变量的作用域限定在循环中
- 初始化、条件表达式和循环后表达式都不是必需的

```js
for (;;) {// 无穷循环
    doSomething();
}
```

- 如果只包含条件表达式, 那么 for 循环实际上就变成了 while 循环

```js
let count = 10;
let i = 0;
for (; i < count; ) {
    console.log(i);
    i++;
}
```

## for-in

- for-in 语句是一种严格的迭代语句, 用于枚举对象中的非符号键属性
- ECMAScript 中的对象的属性是无序的, 因此 for-in 语句不能保证返回对象属性的顺序。
  换句话说, 所有可枚举的属性都会返回一次, 但返回的顺序可能会因浏览器而异
- 如果 for-in 循环要迭代的变量是 null 或 undefined, 则不执行循环体
- 语法

```js
for (property in expression) statement
```

- 示例

```js
for (const propName in window) { // 控制语句中的 const 不是必需的, 但是为了确保这个局部变量不被修改, 推荐使用 const
    document.write(propName);
}
```

## for-of

- for-of 语句是一种严格的迭代语句, 用于遍历可迭代对象的元素
- 语法

```js
for (property of expression) statement
```

- 示例

```js
for (const el of [2, 4, 6, 8]) {// 控制语句中的 const 不是必需的, 但是为了确保这个局部变量不被修改, 推荐使用 const
    document.write(el);
}
```

## label

- 标签语句用于给语句加标签, 标签可以在语句后面通过 break 或 continue 语句引用
- 标签语句的典型应用场景是嵌套循环
- 语法

```js
label: statement
```

- 示例

```js
start: for (let i = 0; i < count; i++) {
    console.log(i);
}
```

## break

## continue

## with

- with 语句的用途是将代码作用域设置为特定的对象
- 语法

```js
with (expression) statement;
```

## switch

# 函数

## 函数基本知识

* ECMAScript 中的函数不需要指定是否返回值, 任何函数在任何时间都可以使用 return 语句来返回函数的值, 
  用法是后跟要返回的值, 只要碰到 return 语句, 函数就立即停止执行并退出
* 最佳实践是函数要么返回值, 要么不返回值。只在某个条件下返回值的函数会带来麻烦, 尤其是调试时
* 严格模式对函数也有一些限制: 
    - 函数不能以 `eval` 或 `arguments` 作为名称
    - 函数的参数不能叫做 `eval` 或 `arguments`
    - 两个命名参数不能拥有同一个名称

- 语法

```js
function functionName(arg0, arg1, ..., argN) {
    statements
}
```

- 示例

```js
function sayHi(name, message) {
    console.log("Hello " + name + ", " + message);
}
sayHi("Nicholas", "how are you today?");
```
