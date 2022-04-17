---
title: 变量、作用域、内存
author: 王哲峰
date: '2020-02-01'
slug: js-variable-scope-memory
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

- [原始值与引用值](#原始值与引用值)
  - [动态属性](#动态属性)
  - [复制值](#复制值)
  - [传递参数](#传递参数)
  - [确定类型](#确定类型)
- [执行上下文与作用域](#执行上下文与作用域)
  - [作用域链增强](#作用域链增强)
  - [变量声明](#变量声明)
    - [使用 var 的函数作用域声明](#使用-var-的函数作用域声明)
  - [标识符查找](#标识符查找)
- [垃圾回收](#垃圾回收)
  - [标记清理](#标记清理)
  - [引用计数](#引用计数)
  - [性能](#性能)
  - [内存管理](#内存管理)
</p></details><p></p>

- JavaScript 变量是松散类型的，而且变量不过就是特定时间点一个特定值的名称而已
- 由于没有规则定义变量必须包含什么数据类型，变量的值和数据类型在脚本声明周期内可以改变。这样的变量很有意思，很强大，当然也有不少问题

# 原始值与引用值

- ECMAScript 变量可以包含两种不同类型的数据
  - **原始值(primitive value)** ，就是最简单的数据
    - Undefined
    - Null
    - Boolean--[原始包装类型]
    - Number--[原始包装类型]
    - String--[原始包装类型]
    - Symbol
    - 保存原始值的变量是**按值(by value)**访问的，因为操作的就是存储在变量中的实际值
  - **引用值(reference value)**，就是由多个值构成的对象
    - 引用值是保存在内存中的对象，JavaScript 不允许直接访问内存位置，因此也就不能直接操作对象所在的内存空间
    - 在操作对象时，实际上操作的是对该对象的**引用(reference)**而非实际的对象本身，为此，保存引用值的变量是**按引用(by reference)**访问的

- 在把一个值赋给变量时，JavaScript 引擎必须确定这个值是原始值还是引用值

## 动态属性

- 原始值和引用值的定义方式很类似，都是创建一个变量，然后给它赋一个值，但是，在变量保存了这个值之后，可以对这个值做什么，则大有不同
	- 对于引用值而言，可以随时添加、修改和删除其属性和方法
	- 对于原始值，不能有属性，尽管尝试给原始值添加属性不会报错

```js
// 对象--引用值
let person = new Object();
person.name = "Nicholas";
console.log(person.name); // "Nicholas"

// 字符串--原始值
let name = "Nicholas";
name.age = 27;
console.log(name.age); // undefined
```

- 原始值
	- 原始类型的初始化可以只使用`原始字面量`形式

```js
let name1 = "Nicholas";
name1.age = 27;
console.log(name1.age); // undefined
console.log(typeof name1); // string
```

- 引用值
	- 如果使用的是 `new` 关键字，则 JavaScript 会创建一个 Object 类型的实例，但其行为类似原始值

```js
let name2 = new String("Matt");
name2.age = 26;
console.log(name2.age); // 26
console.log(typeof name2); // object
```

## 复制值

除了存储方式不同，**原始值**和**引用值**在通过**变量**复制时也有所不同。

- 在通过变量把一个原始值复制到另一个变量时，原始值会被复制到新变量的位置

<img src="/Users/zfwang/Library/Application Support/typora-user-images/image-20210412012800208.png" alt="image-20210412012800208" style="zoom:33%;" />

```js
// num1 和 num2 可以独立使用，互不干扰
let num1 = 5;
let num2 = num2;  // num2 = 5, 这个值跟存储在 num1 中的 5 是完全独立的，因为它是那个值的副本
```

- 在把引用值从一个变量赋给另一个变量时，存储在变量中的值也会被复制到新变量所在的位置。区别在于，这里复制的值实际上是一个指针，它指向存储在堆内存中的对象。操作完成后，两个变量实际上指向同一个对象，因此一个对象上面的变化会在另一个对象上反映出来。

<img src="/Users/zfwang/Library/Application Support/typora-user-images/image-20210412012741789.png" alt="image-20210412012741789" style="zoom: 33%;" />

```js
let obj1 = new Object();
let obj2 = obj1;
obj1.name = "Nicholas";
console.log(obj2.name); // "Nicholas"
```

## 传递参数

- ECMAScript 中所有函数的参数都是按值传递的。这意味着函数外的值会被复制到函数内部的参数中，就像从一个变量复制到另一个变量一样
	- 如果是原始值，那么就跟原始值变量的复制一样
	- 如果是引用值，那么就跟引用值变量的复制一样
- 在按值传递参数时，无论是原始值还是引用值，值会被复制到一个局部变量(即一个命名参数，或者用 ECMAScript 的话说，就是 `arguments` 对象中的一个槽位)
- 在按引用传递参数时，值在内存中的位置会被保存在一个局部变量，这意味着对本地变量的修改会反映到函数外部(这在ECMAScript 中是不可能的)

```js
// 按值传递--原始值
function addTen(num) {
    num += 10;
    return num;
}

let count = 20;
let result = addTen(count);
console.log(count); // 20
console.log(result); // 30
```

```js
// 按值传递--引用值
function setName(obj) {
    obj.name = "Nicholas";
}

let person = new Object();
setName(person);
console.log(person.name); // "Nicholas"
```

```js
// 按值传递--引用值
function setName(obj) {
    obj.name = "Nicholas";
    // obj 在函数内部被重写，它变成了一个指向本地对象的指针，这个本地对象在函数执行结束时就被销毁了
    obj = new Object();
    obj.name = "Greg";
}
let person = new Object();
setName(person);
console.log(person.name); // "Nicholas"
```

## 确定类型

- `typeof` 操作符最适合用来判断一个变量是否为`原始类型` . 更确切地说，它是判断一个变量是否为字符串、数值、布尔值或 undefined 的最好方式
  - 如果值是``对象``或 `null`，那么 `typeof` 返回 "object"
  - `typeof` 虽然对原始值很有用，但他对引用值的用处不大。我们通常不关心一个值是不是对象，而是想知道它是什么类型的对象，为了解决这个问题，ECMAScript 提供了 instanceof 操作符

```js
let s = "Nicholas";   // string
let b = true; 		  // boolean
let i = 22; 	 	  // number
let u; 				  // undefined
let n = null; 	      // object
let o = new Object(); // object

console.log(typeof s); // string
console.log(typeof b); // boolean
console.log(typeof i); // number
console.log(typeof u); // undefined
console.log(typeof n); // object
console.log(typeof o); // object
```

- `instanceof` 操作符语法如下:
	
	- 如果变量是给定引用类型(由其原型链决定)的实例，则 `instanceof` 操作符返回 `true`
	- 按照定义，**所有引用值都是 Object 的实例**，因此通过`instanceof` 操作符检测任何引用类型值和 Object 构造函数都会返回 true。类似地，如果用 instanceof 检测原始值，则会返回 false，因为原始值不是对象
	- 语法
	
	```js
	result = variable instanceof constructor
	```
	
	- 示例
	
	```js
	console.log(person instanceof Object);
	console.log(colors instanceof Array);
	console.log(pattern instanceof RegExp);
	```

- 确定一个变量类型的方法
	- (1) 首先判断是原始类型还是引用类型
	- `typeof` variable
		- not object: `"boolean"` 、 `"number"`、``"string"`、 `"undefined"`、`"symbol"`
		- `"object"`:  null and 对象
	- (2) 如果变量是 "object"
		- `instanceof` variable
			- `ture`
			- `false`

# 执行上下文与作用域

- 每个上下文都有一个关联的**变量对象(variable object)**，而这个上下文中定义的所有变量和函数都存在于这个对象上。虽然无法通过代码访问变量对象，但后台处理数据会用到它
	- 全局上下文
		- 根据 ECMAScript 实现的宿主环境，表示全局上下文的对象可能不一样。
		- 在浏览器中，全局上下文就是我们常说的 window 对象
			- 通过 var 定义的全局变量和函数都会成为 window 对象的属性和方法
			- 使用 let 和 const 的顶级声明不会定义在全局上下文中，但在作用域链解析上效果是一样的
	- 每个函数都有自己的上下文
	- 上下文中的代码在执行的时候，会创建变量对象的一个作用域链(scope chain)，这个作用域链决定了各级上下文中代码在访问变量和函数时的顺序。代码正在执行的上下文的变量对象始终位于作用域链的最前端。
		- 如果上下文时函数，则其活动对象(activation object) 用作变量对象，活动对象最初只有一个定义变量：`arguments`。作用域链中的下一个变量对象来自包含上下文，再下一个对象来自再下一个包含上下文，以此类推直至全局上下文，全局上下文的变量对象始终是作用域链的最后一个变量对象
	- 代码执行时的标识符解析是通过沿作用域链逐级搜索标识符完成的，搜索过程始终从作用域链的最前端开始，然后逐级往后，直到找到标识符

## 作用域链增强

- 虽然执行上下文主要有全局上下文和函数上下文两种(`eval` 调用内部存在第三种上下文)，但有其他方式来增强作用域链
- 某些语句会导致在作用域链前端临时添加一个上下文，这个上下文在代码执行后会被删除
- 通常在两种情况下会出现这个现象，即代码执行到下面任意一种情况时，这两种情况下，都会在作用域链前端添加一个变量对象：
	- `try/catch` 语句的 `catch` 块
		- 会创建一个新的变量对象，这个变量对象会包含要抛出的错误对象的声明
	- `with` 语句
		- 会向作用域链前端添加指定的对象
- 示例

```js
function buildUrl() {
    let qs = "?debug=true";
    with(location) {
        let url = href + qs;
    }
    return url;
}
```

## 变量声明

### 使用 var 的函数作用域声明

- 在使用 var 声明变量时，变量会被自动添加到最接近的上下文。在函数中，最接近的上下文就是函 数的局部上下文。在 with 语句中，最接近的上下文也是函数上下文(`window`)

```js
function add(num1, num2) {
    var sum = num1 + num2;
    return sum;
}
let result = add(10, 20); // 30
console.log(sum); // 报错: sum 在这里不是有效变量
```

- **如果变量未经声明就被初始化了(不使用 var, let, const)， 那么它就会自动被添加到全局上下文**

```js
function add(num1, num2) {
    sum = num1 + num2;
    return sum;
}
let result = add(10, 20); // 30
console.log(sum); // 30
```

- var 声明会被拿到函数或全局作用域的顶部，位于作用域中所有代码之前。这个现象叫作“提升”  (hoisting)。提升让同一作用域中的代码不必考虑变量是否已经声明就可以直接使用。可是在实践中，提升也会导致合法却奇怪的现象，即在变量声明之前使用变量

```js
var name = "Jake";
// 等价于
name = "Jake";
```

```js
function fn1() {
    var name = "Jake";
}
// 等价于
function fn2() {
    var name;
    name = "Jake";
}
```

- 通过在声明之前打印变量，可以验证变量会被提升。声明的提升意味着会输出 undefined 而不是 Reference Error(在严格模式下，未经声明就初始化变量 会报错)

```js
console.log(name); // undefined
var name = "Jake";

function demo () {
    console.log(name); // undefined
    var name = 'Jake';
}
demo();
```

##.2 使用 let 的块级作用域声明

- ES6 新增的 let 关键字跟 var 很相似，但它的作用域是块级的。块级作用域由最近的一对包含花括号 `{}` 界定

```js
if (true) {
    let a;
}
console.log(a); // ReferenceError: a 没有定义
```

```js
while (true) {
    let b;
}
console.log(b); // ReferenceError: b 没有定义
```

```js
function foo() {
    let c;
}
console.log(c); // ReferenceError: c 没有定义, var 声明也会导致报错
```

```js
// 这不是对象字面量，而是一个独立的块，JavaScript 解释器会根据其中的内容识别出来
{
    let d;
}
console.log(d); // ReferenceError: d 没有定义
```

- let 和 var 的另一个不同之处是在统一作用域内不能声明两次。重复的 var 声明会被忽略，而重复的 let 声明会抛出 SyntaxError

```js
var a;
var a;
```

```js
{
    let b;
    let b;
}
// SyntaxError: 标识符 b 已经声明过了
```

- let 的行为非常适合在循环中声明迭代变量。使用 var 声明的迭代变量会泄漏到循环外部，这种情况应该避免

```js
for (var i = 0; i < 10; ++i) {}
console.log(i);    // 10
```

```js
for (let j = 0; j < 10; ++j) {}
console.log(j);    // ReferenceError: j 没有定义
```

- 严格来讲，let  在 JavaScript 运行时也会被提升，但由于 "暂时性死区"(temporal dead zone) 的缘故，实际上不能在声明之前使用 let 变量。因此，从写 JavaScript 代码的角度说，let 的提升跟 var 是不一样的

##.3 使用 const 的常量声明

- 使用 const 声明的同时必须同时初始化为某个值

```js
const a; // SyntaxError: 常量声明时没有初始化
```

- 一经声明，在其声明周期的任何时候都不能再重新赋予新值

```js
const b = 3;
console.log(b); // 3
b = 4; // TypeError: 给常量赋值
```

- const 除了要遵循以上规则，其他方面与 let 声明是一样的

```js
if (true) {
    const a = 0;
}
console.log(a); // ReferenceError: a 没有定义
```

```js
while (true) {
    const b = 1;
}
console.log(b); // ReferenceError: b 没有定义
```

```js
function foo() {
    const c = 2;
}
console.log(c); // ReferenceError: c 没有定义
```

```js
{
    const d = 3;
}
console.log(d); // ReferenceError: d 没有定义
```

- const 声明只应用到顶级原语或者对象。换句话说，赋值为对象的 const 变量不能再被重新赋值为其他引用值，但对象的键则不受限制
	- 如果想让整个对象都不能修改，可以使用 Object.freeze()，这样再给属性赋值时虽然不会报错，但会静默失败
	- 由于 const 声明暗示变量的值是单一类型且不可修改。JavaScript 运行时编译器可以将其所有实例都替换成实际的值，而不会通过查询表进行变量查找。Google 的 V8 引擎就执行这种优化

```js
const o1 = {};
o1 = {}; // TypeError: 给变量赋值

const o2 = {};
o2.name = "Jake";
console.log(o2.name); // "Jake"

const o3 = Object.freeze({});
o3.name = "Jake";
console.log(o3.name); // undefined
```

## 标识符查找

- 当在特定上下文中为读取或写入而引用一个标识符时，必须通过搜索确定这个标识符表示什么。搜 索开始于作用域链前端，以给定的名称搜索对应的标识符。如果在局部上下文中找到该标识符，则搜索 停止，变量确定;如果没有找到变量名，则继续沿作用域链搜索。(注意，作用域链中的对象也有一个 原型链，因此搜索可能涉及每个对象的原型链。)这个过程一直持续到搜索至全局上下文的变量对象。 如果仍然没有找到标识符，则说明其未声明

- 示例 1

```js
var color = "blue";

function getColor() {
    return color;
}
console.log(getColor()); // "blue"
```

- 示例 2

```js
var color = "blue";
function getColor() {
    let color = "red";
    return color;
}
console.log(getColor()); // "red"
```

> 标识符查找并非没有代价。访问局部变量比访问全局变量要快，因为不用切换作用域。不过，JavaScript 引擎在优化标识符查找上做了很多工作，将来这个差异可能就微不足道了

# 垃圾回收

- JavaScript 是使用垃圾回收的语言，也就是说执行环境负责在代码执行时管理内存。在 C 和 C++ 等语言中，跟踪内存使用对开发者来说是个很大的负担，也是很多问题的来源
- JavaScript 为开发者卸下了这个负担，通过自动内存管理内存分配和闲置资源回收。基本思路很简单：
	- (1) 首先，确定哪个变量不会再使用
	- (2) 然后释放它占用的内存。
	- (3) 这个过程是周期性的，即垃圾回收程序每隔一定时间(或者说在代码执行过程中某个预定的收集时间)就会自动运行
-  垃圾回收过程是一个近似且不完美的方案，因为某块内存是否还有用，属于“不可判定的”问题，意味着靠算法是解决不了的
- 垃圾回收程序必须跟踪记录哪个变量还会使用，以及哪个变量不会再使用，以便回收内存。如何标记未使用的变量也许有不同的实现方式。不过，在浏览器的发展史上，用到过两种主要的标记策略：
	- 标记清理
	- 引用计数

## 标记清理

- JavaScript 最常用的垃圾回收策略是**标记清理(mark-and-sweep)**
	- 当变量进入上下文，比如在函数内部声明一个变量时，这个变量会被加上存在上下文中的标记。而在上下文中的变量，逻辑上讲，永远不应该释放它们的内存，因为只要上下文中的代码在运行，就有可能用到它们。当变量离开上下文时，也会被加上离开上下文的标记
	- 给变量加标记的方式有很多种。比如，当变量进入上下文时，反转某一位；或者可以维护“在上下文中”和“不在上下文中”两个变量列表，可以把变量从一个列表转移到另一个列表。标记过程的实现并不重要，关键是策略
	- 垃圾回收程序运行的时候，会标记内存中存储的所有变量(记住，标记方法有很多种)。然后，它会将所有在上下文中的变量，以及被在上下文中的变量引用的变量的标记去掉。在此之后再被加上标记的变量就是待删除的了，原因是任何在上下文中的变量都访问不到它们了。随后垃圾回收程序做一次内存清理，销毁带标记的所有值并收回它们的内存

## 引用计数

## 性能

## 内存管理

