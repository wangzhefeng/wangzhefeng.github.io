---
title: 变量、作用域、内存
author: 王哲峰
date: '2020-02-01'
slug: js-variable-scope-memory
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
- [原始值与引用值](#原始值与引用值)
  - [动态属性](#动态属性)
    - [原始值](#原始值)
    - [引用值](#引用值)
  - [复制值](#复制值)
    - [原始值](#原始值-1)
    - [引用值](#引用值-1)
  - [传递参数](#传递参数)
  - [确定类型](#确定类型)
    - [typeof](#typeof)
    - [instanceof](#instanceof)
    - [确定一个变量类型的方法](#确定一个变量类型的方法)
- [执行上下文与作用域](#执行上下文与作用域)
  - [执行上下文](#执行上下文)
    - [全局上下文](#全局上下文)
    - [函数上下文](#函数上下文)
  - [作用域链](#作用域链)
    - [作用域链简介](#作用域链简介)
    - [示例](#示例)
  - [作用域链增强](#作用域链增强)
    - [作用域链增强](#作用域链增强-1)
    - [示例](#示例-1)
  - [变量声明](#变量声明)
    - [使用 var 的函数作用域声明](#使用-var-的函数作用域声明)
  - [使用 let 的块级作用域声明](#使用-let-的块级作用域声明)
  - [使用 const 的常量声明](#使用-const-的常量声明)
  - [标识符查找](#标识符查找)
- [垃圾回收](#垃圾回收)
  - [标记清理](#标记清理)
  - [引用计数](#引用计数)
  - [性能](#性能)
  - [内存管理](#内存管理)
    - [通过 const 和 let 声明提升性能](#通过-const-和-let-声明提升性能)
    - [隐藏类和删除操作](#隐藏类和删除操作)
    - [内存泄漏](#内存泄漏)
    - [静态分配与对象池](#静态分配与对象池)
</p></details><p></p>


# 概览

> JavaScript 变量是松散类型的，而且变量不过就是特定时间点一个特定值的名称而已
> 
> 由于没有规则定义变量必须包含什么数据类型，变量的值和数据类型在脚本声明周期内可以改变。
  这样的变量很有意思，很强大，当然也有不少问题

JavaScipt 变量可以保存两种类型的值: 原始值和引用值

原始值可能是以下 6 种原始数据类型之一:

* Undefined
* Null
* Boolean
* Number
* String
* Symbol 

原始值和引用值有以下特点:

* 原始值大小固定，因此保存在**栈内存上**
* 从一个变量到另一个变量复制原始值会创建该值的第二个副本
* 引用值是对象，存储在**堆内存上**
* 包含引用值的变量实际上只包含指向相应对象的一个指针，而不是对象本身
* 从一个变量到另一个变量复制引用值只会复制指针，因此结果是两个变量都指向同一个对象
* `typeof` 操作符可以确定值的原始类型，而 `instanceof` 操作符用于确保值的引用类型

任何变量(不管包含的是原始值还是引用值)都存在于某个执行上下文中(也称为作用域)。
这个上下文(作用域)决定了变量的生命周期，以及它们可以访问代码的哪些部分。
执行上下文可以总结如下:

* 执行上下文分全局上下文、函数上下文和块级上下文
* 代码执行流每进入一个新上下文，都会创建一个作用域链，用于搜索变量和函数
* 函数或块的局部上下文不仅可以访问自己作用域内的变量，而且也可以访问任何包含上下文乃
  至全局上下文中的变量
* 全局上下文只能访问全局上下文中的变量和函数，不能直接访问局部上下文中的任何数据
* 变量的执行上下文用于确定什么时候释放内存

JavaScript 是使用垃圾回收的编程语言，开发者不需要操心内存分配和回收。
JavaScript 的垃圾回收程序可以总结如下:

* 离开作用域的值会被自动标记为可回收，然后在垃圾回收期间被删除
* 主流的垃圾回收算法是**标记清理**，即先给当前不使用的值加上标记，再回来回收它们的内存
* 引用计数是另一种垃圾回收策略，需要记录值被引用了多少次。
  JavaScript 引擎不再使用这种算法，但某些旧版本的 IE 仍然会受这种算法的影响，
  原因是 JavaScript 会访问非原生 JavaScript 对象(如 DOM 元素)
* 引用计数在代码中存在循环引用时会出现问题
* 解除变量的引用不仅可以消除循环引用，而且对垃圾回收也有帮助。
  为促进内存回收，全局对象、全局对象的属性和循环引用都应该在不需要时解除引用

# 原始值与引用值

ECMAScript 变量可以包含两种不同类型的数据: 原始值和引用值。
在把一个值赋给变量时，JavaScript 引擎必须确定这个值是原始值还是引用值

* **原始值(primitive value)** ，就是最简单的数据
    - Undefined
    - Null
    - Boolean--[原始包装类型]
    - Number--[原始包装类型]
    - String--[原始包装类型]
    - Symbol
    - 保存原始值的变量是 **按值(by value)** 访问的，因为操作的就是存储在变量中的实际值
* **引用值(reference value)**，就是由多个值构成的对象
    - 引用值是保存在内存中的对象，JavaScript 不允许直接访问内存位置，
      因此也就不能直接操作对象所在的内存空间
    - 在操作对象时，实际上操作的是对该对象的 **引用(reference)** 而非实际的对象本身，
      为此，保存引用值的变量是 **按引用(by reference)** 访问的

## 动态属性

原始值和引用值的定义方式很类似，都是创建一个变量，然后给它赋一个值，
但是，在变量保存了这个值之后，可以对这个值做什么，则大有不同

* 对于引用值而言，可以随时添加、修改和删除其属性和方法
* 对于原始值，不能有属性，尽管尝试给原始值添加属性不会报错

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

### 原始值

* 原始类型的初始化可以只使用 **原始字面量** 形式

```js
let name1 = "Nicholas";
name1.age = 27;
console.log(name1.age); // undefined
console.log(typeof name1); // string
```

### 引用值
	
* 如果使用的是 `new` 关键字，则 JavaScript 会创建一个 Object 类型的实例，
  但其行为类似原始值

```js
let name2 = new String("Matt");
name2.age = 26;
console.log(name2.age); // 26
console.log(typeof name2); // object
```

## 复制值

除了存储方式不同，**原始值**和**引用值**在通过**变量**复制时也有所不同

### 原始值

在通过变量把一个原始值复制到另一个变量时，原始值会被复制到新变量的位置

<img src="images/image-20210412012800208.png" alt="image-20210412012800208" style="zoom:33%;" />

```js
// num1 和 num2 可以独立使用，互不干扰
let num1 = 5;
let num2 = num2;  
// num2 = 5, 这个值跟存储在 num1 中的 5 是完全独立的，因为它是那个值的副本
```

### 引用值

在把引用值从一个变量赋给另一个变量时，存储在变量中的值也会被复制到新变量所在的位置。
区别在于，这里复制的值实际上是一个指针，它指向存储在堆内存中的对象。
操作完成后，两个变量实际上指向同一个对象，因此一个对象上面的变化会在另一个对象上反映出来

<img src="images/image-20210412012741789.png" alt="image-20210412012741789" style="zoom: 33%;" />

```js
let obj1 = new Object();
let obj2 = obj1;
obj1.name = "Nicholas";
console.log(obj2.name); // "Nicholas"
```

## 传递参数

ECMAScript 中所有函数的参数都是按值传递的。
这意味着函数外的值会被复制到函数内部的参数中，
就像从一个变量复制到另一个变量一样

* 如果是原始值，那么就跟原始值变量的复制一样
* 如果是引用值，那么就跟引用值变量的复制一样

变量有按值和按引用访问，而传参只有按值传递:

* 在按值传递参数时，值会被复制到一个局部变量(即一个命名参数，
  或者用 ECMAScript 的话说，就是 `arguments` 对象中的一个槽位)
* 在按引用传递参数时，值在内存中的位置会被保存在一个局部变量，
  这意味着对本地变量的修改会反映到函数外部(这在ECMAScript 中是不可能的)

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
    // obj 在函数内部被重写，它变成了一个指向本地对象的指针，
    // 这个本地对象在函数执行结束时就被销毁了
    obj = new Object();
    obj.name = "Greg";
}
let person = new Object();
setName(person);
console.log(person.name); // "Nicholas"
```

## 确定类型

### typeof

`typeof` 操作符最适合用来判断一个变量是否为**原始类型**。
更确切地说，它是判断一个变量是否为字符串、数值、布尔值或 `undefined` 的最好方式

* 如果值是对象或 `null`，那么 `typeof` 返回 "object"
 
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

### instanceof

`typeof` 虽然对原始值很有用，但他对引用值的用处不大。
我们通常不关心一个值是不是对象，而是想知道它是什么类型的对象，
为了解决这个问题，ECMAScript 提供了 `instanceof` 操作符

* `instanceof` 操作符语法如下:	
    - 如果变量是给定引用类型(由其原型链决定)的实例，则 `instanceof` 操作符返回 `true`
    - 按照定义，**所有引用值都是 Object 的实例**，
      因此通过 `instanceof` 操作符检测任何引用类型值和 Object 构造函数都会返回 true。
      类似地，如果用 `instanceof` 检测原始值，则会返回 `false`，因为原始值不是对象

```js
result = variable instanceof constructor
```

- 示例

```js
console.log(person instanceof Object);
console.log(colors instanceof Array);
console.log(pattern instanceof RegExp);
```

### 确定一个变量类型的方法

* (1) 首先判断是原始类型还是引用类型
    - not object: 
        - `"boolean"`
        - `"number"`
        - `"string"`
        - `"undefined"`
        - `"symbol"`
        - `function`
    - "object":  
        - `null`
        - 对象

```js
typeof variable
```

* (2) 如果变量是 "object"
    - `ture`
    - `false`

```js
variable instanceof ...
```

# 执行上下文与作用域

## 执行上下文

变量或函数的上下文决定了它们可以访问那些数据，以及他们的行为。
每个上下文都有一个关联的**变量对象(variable object)**，
而这个上下文中定义的所有变量和函数都存在于这个对象上。
虽然无法通过代码访问变量对象，但后台处理数据会用到它

### 全局上下文

* 全局上下文是最外层的上下文
* 根据 ECMAScript 实现的宿主环境，表示全局上下文的对象可能不一样
    - 在浏览器中，全局上下文就是我们常说的 `window` 对象
	- 通过 `var` 定义的全局变量和函数都会成为 `window` 对象的属性和方法
	- 使用 `let` 和 `const` 的顶级声明不会定义在全局上下文中，
	  但在作用域链解析上效果是一样的
* 上下文在其所有代码都执行完毕后会被销毁，包括定义在它上面的所有变量和函数，
  全局上下文在应用程序退出前才会被销毁，比如关闭网页或退出浏览器

### 函数上下文

每个函数都有自己的上下文:

* 当代码执行流进入函数时，函数的上下文被推到一个上下文栈上
* 在函数执行完毕之后，上下文栈会弹出该函数上下文，将控制权返还给之前的执行上下文
* ECMAScript 程序的执行流就是通过这个上下文栈进行控制的

## 作用域链

### 作用域链简介

上下文中的代码在执行的时候，会创建变量对象的一个作用域链(scope chain)，
这个作用域链决定了各级上下文中代码在访问变量和函数时的顺序。

* 代码正在执行的上下文的变量对象始终位于作用域链的最前端
* 如果上下文是函数，则其活动对象(activation object)用作变量对象，
  活动对象最初只有一个定义变量：`arguments`，全局上下文中没有这个变量
    - 函数参数被认为是当前上下文中的变量，因此也跟上下文中的其他变量遵循相同的访问规则
* 作用域链中的下一个变量对象来自包含上下文，再下一个对象来自再下一个包含上下文，
  以此类推直至全局上下文，全局上下文的变量对象始终是作用域链的最后一个变量对象
* 代码执行时的标识符解析是通过沿作用域链逐级搜索标识符完成的，
  搜索过程始终从作用域链的最前端开始，然后逐级往后，直到找到标识符。
  如果没有找到标识符，那么通常会报错

### 示例

* 示例 1：函数的作用域链包含两个对象
    - 它自己的变量对象，就是定义 `arguments` 对象的那个
    - 全局上下文的变量对象

```js
var color = "blue";

function changeColor() {
    if (color === "blue") {
        color = "red";
    } else {
        color = "blue";
    }
}

changeColor();
```

* 示例 2: 局部作用域中定义的变量可用于在局部上下文中替换全局变量。
  以下代码涉及 3 个上下文：
    - 全局上下文，包含
        - 一个变量 `color`
        - 一个函数 `changeColor()`
    - `changeColor()` 的局部上下文，包含
        - 一个变量 `anotherColor`
        - 一个函数 `swapColor()`
        - 可以访问的全局上下文中的变量 `color`
    - swapColors() 的局部上下文，包含
        - 一个变量 `tempColor`
        - 可以访问到的 `changeColor()` 局部上下文中的变量 `anotherColor`
        - 可以访问到的全局上下文中的变量 `color`

```js
var color = "blue";

function changeColor() {
    let anotherColor = "red";

    function swapColors() {
        let tempColor = anotherColor;
        anotherColor = color;
        color = tempColor;

        // 这里可以访问 color anotherColor tempColor
    }

    // 这里可以访问 color anotherColor, 访问不到 tempColor
    swapColors();
}

// 这里只能访问 color
changeColor();
```

以上示例作用域链图示：

* 图中的矩形表示不同的上下文。内部上下文可以通过作用域链访问外部上下文中的一切，
  但外部上下文无法访问上下文中的任何东西
* 上下文之间的连接是线性的、有序的。每个上下文都可以到上一级的上下文中取搜索变量和函数，
  但任何上下文都不能到下一级上下文中取搜索 


<img src="images/scope.jpg" alt="image-20210412012800208" style="zoom:33%;" />

## 作用域链增强

虽然执行上下文主要有全局上下文和函数上下文两种(`eval()` 调用内部存在第三种上下文)，
但有其他方式来增强作用域链

### 作用域链增强

某些语句会导致在作用域链前端临时添加一个上下文，这个上下文在代码执行后会被删除。
通常在两种情况下会出现这个现象，即代码执行到下面任意一种情况时，
这两种情况下，都会在作用域链前端添加一个变量对象：

- `try/catch` 语句的 `catch` 块
	- 会创建一个新的变量对象，这个变量对象会包含要抛出的错误对象的声明
- `with` 语句
	- 会向作用域链前端添加指定的对象

### 示例

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

在使用 `var` 声明变量时，变量会被自动添加到最接近的上下文

* 在函数中，最接近的上下文就是函数的局部上下文
* 在 `with` 语句中，最接近的上下文也是函数上下文(`window`)
* 如果变量未经声明就被初始化了(不使用 `var`, `let`, `const`)，
  那么它就会自动被添加到全局上下文

```js
// var 声明变量
function add(num1, num2) {
    var sum = num1 + num2;
    return sum;
}
let result = add(10, 20); // 30
console.log(sum); // 报错: sum 在这里不是有效变量

// 省略 var 声明变量
function add(num1, num2) {
    sum = num1 + num2;  // add 函数被调用后，变量 sum 被添加到了全局上下文
    return sum;
}
let result = add(10, 20);  // 30
console.log(sum);  // 30
```

- var 声明会被拿到函数或全局作用域的顶部，位于作用域中所有代码之前。
  这个现象叫作“提升”(hoisting)。提升让同一作用域中的代码不必考虑变量是否已经声明就可以直接使用。
  可是在实践中，提升也会导致合法却奇怪的现象，即在变量声明之前使用变量

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

- 通过在声明之前打印变量，可以验证变量会被提升。
  声明的提升意味着会输出 undefined 而不是 Reference Error
  (在严格模式下，未经声明就初始化变量 会报错)

```js
console.log(name); // undefined
var name = "Jake";

function demo () {
    console.log(name); // undefined
    var name = 'Jake';
}
demo();
```

## 使用 let 的块级作用域声明

- ES6 新增的 `let` 关键字跟 `var` 很相似，但它的作用域是块级的。
  块级作用域由最近的一对包含花括号 `{}` 界定

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
console.log(c); 
// ReferenceError: c 没有定义, var 声明也会导致报错
```

```js
// 这不是对象字面量，而是一个独立的块，JavaScript 解释器会根据其中的内容识别出来
{
    let d;
}
console.log(d); // ReferenceError: d 没有定义
```

- `let` 和 `var` 的另一个不同之处是在统一作用域内不能声明两次。
  重复的 `var` 声明会被忽略，而重复的 `let` 声明会抛出 `SyntaxError`

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

- `let` 的行为非常适合在循环中声明迭代变量。
  使用 `var` 声明的迭代变量会泄漏到循环外部，这种情况应该避免

```js
for (var i = 0; i < 10; ++i) {}
console.log(i);    // 10
```

```js
for (let j = 0; j < 10; ++j) {}
console.log(j);    // ReferenceError: j 没有定义
```

- 严格来讲，`let` 在 JavaScript 运行时也会被提升，
  但由于 "暂时性死区"(temporal dead zone) 的缘故，
  实际上不能在声明之前使用 `let` 变量。因此，
  从写 JavaScript 代码的角度说，`let` 的提升跟 `var` 是不一样的

## 使用 const 的常量声明

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

- `const` 除了要遵循以上规则，其他方面与 `let` 声明是一样的

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

- `const` 声明只应用到顶级原语或者对象。换句话说，赋值为对象的 `const` 变量不能再被重新赋值为其他引用值，但对象的键则不受限制
	- 如果想让整个对象都不能修改，可以使用 `Object.freeze()`，
	  这样再给属性赋值时虽然不会报错，但会静默失败
	- 由于 `const` 声明暗示变量的值是单一类型且不可修改。
	  JavaScript 运行时编译器可以将其所有实例都替换成实际的值，
      而不会通过查询表进行变量查找。Google 的 V8 引擎就执行这种优化

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

当在特定上下文中为读取或写入而引用一个标识符时，必须通过搜索确定这个标识符表示什么。
搜索开始于作用域链前端，以给定的名称搜索对应的标识符

* 如果在局部上下文中找到该标识符，则搜索停止，变量确定
* 如果没有找到变量名，则继续沿作用域链搜索。
  注意，作用域链中的对象也有一个原型链，因此搜索可能涉及每个对象的原型链

这个过程一直持续到搜索至全局上下文的变量对象。如果仍然没有找到标识符，则说明其未声明

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

> 标识符查找并非没有代价。访问局部变量比访问全局变量要快，
  因为不用切换作用域。不过，JavaScript 引擎在优化标识符查找上做了很多工作，
  将来这个差异可能就微不足道了

# 垃圾回收

JavaScript 是使用垃圾回收的语言，也就是说执行环境负责在代码执行时管理内存。
在 C 和 C++ 等语言中，跟踪内存使用对开发者来说是个很大的负担，也是很多问题的来源

JavaScript 为开发者卸下了这个负担，通过自动内存管理内存分配和闲置资源回收。
基本思路很简单：

* (1) 首先，确定哪个变量不会再使用
* (2) 然后释放它占用的内存
* (3) 这个过程是周期性的，即垃圾回收程序每隔一定时间(或者说在代码执行过程中某个预定的收集时间)就会自动运行

垃圾回收过程是一个近似且不完美的方案，因为某块内存是否还有用，
属于“不可判定的”问题，意味着靠算法是解决不了的

垃圾回收程序必须跟踪记录哪个变量还会使用，以及哪个变量不会再使用，
以便回收内存。如何标记未使用的变量也许有不同的实现方式。
不过，在浏览器的发展史上，用到过两种主要的标记策略：

* 标记清理
* 引用计数

## 标记清理

JavaScript 最常用的垃圾回收策略是**标记清理(mark-and-sweep)**

- 当变量进入上下文，比如在函数内部声明一个变量时，这个变量会被加上存在上下文中的标记。
  而在上下文中的变量，逻辑上讲，永远不应该释放它们的内存，因为只要上下文中的代码在运行，
  就有可能用到它们。当变量离开上下文时，也会被加上离开上下文的标记
- 给变量加标记的方式有很多种。比如，当变量进入上下文时，反转某一位；
  或者可以维护“在上下文中”和“不在上下文中”两个变量列表，
  可以把变量从一个列表转移到另一个列表。标记过程的实现并不重要，关键是策略
- 垃圾回收程序运行的时候，会标记内存中存储的所有变量(记住，标记方法有很多种)。
  然后，它会将所有在上下文中的变量，以及被在上下文中的变量引用的变量的标记去掉。
  在此之后再被加上标记的变量就是待删除的了，原因是任何在上下文中的变量都访问不到它们了。
  随后垃圾回收程序做一次内存清理，销毁带标记的所有值并收回它们的内存

## 引用计数

另一种没那么常用的垃圾回收策略是**引用计数(reference counting)**。
其思路是对每个值都记录它被引用的次数

声明变量并给它赋一个引用值时，这个值的引用数为 1。
如果同一个值又被赋给另一个变量，那么引用数加 1。
类似地，如果保存对该值引用的变量被其他值给覆盖了，那么引用数减 1
当一个值的引用数为 0 时，就说明没办法再访问到这个值了，因此可以安全地收回其内存了。
垃圾回收程序下次运行的时候就会释放引用数为 0 的值的内存



## 性能

垃圾回收程序会周期性运行，如果内存中分配了很多变量，则可能造成性能损失，
因此垃圾回收的**时间调度**很重要。尤其是在内存有限的移动设备上，
垃圾回收有可能会明显拖慢渲染的速度和帧速率。 开发者不知道什么时候运行时会收集垃圾，
因此最好的办法是在写代码时就要做到: 

> 无论什么时候开始 收集垃圾，都能让它尽快结束工作

在某些浏览器中是有可能(但不推荐)主动触发垃圾回收的

## 内存管理

在使用垃圾回收的编程环境中，开发者通常无须关心内存管理

不过，JavaScript 运行在一个内存管理与垃圾回收都很特殊的环境。
分配给浏览器的内存通常比分配给桌面软件的要少很多，分配给移动浏览器的就更少了。
这更多出于安全考虑而不是别的，就是为了避免运行大量 JavaScript 的网页耗尽系统内存而导致操作系统崩溃。
这个内存限制不仅影响变量分配，也影响调用栈以及能够同时在一个线程 中执行的语句数量

将内存占用量保持在一个较小的值可以让页面性能更好。优化内存占用的最佳手段就是保证在执行代码时只保存必要的数据。
如果数据不再必要，那么把它设置为 `null`，从而释放其引用。这也可以叫作**解除引用**。
这个建议最适合全局变量和全局对象的属性。

局部变量在超出作用域后会被自动解除引用。不过要注意，解除对一个值的引用并不会自动导致相关内存被回收。
解除引用的关键在于确保相关的值已经不在上下文里了，因此它在下次垃圾回收时会被回收

* 示例

```js
function createPerson(name) {
    let localPerson = new Object();
    localPerson.name = name;
    return localPerson;
}

let globalPerson = createPerson("Nicholas");
// 其他操作

// 解除 globalPerson 对值的引用
globalPerson = null;
```

### 通过 const 和 let 声明提升性能

ES6 增加这两个关键字不仅有助于改善代码风格，而且同样有助于改进垃圾回收的过程。
因为 `const` 和 `let` 都以块(而非函数)为作用域，
所以相比于使用 `var`，使用这两个新关键字可能会更早地让垃圾回收程序介入，
尽早回收应该回收的内存。在块作用域比函数作用域更早终止的情况下，这就有可能发生

### 隐藏类和删除操作

根据 JavaScript 所在的运行环境，有时候需要根据浏览器使用的 JavaScript 引擎来采取不同的性能优化策略

截至 2017 年，Chrome 是最流行的浏览器，使用 V8 JavaScript 引擎。
V8 在将解释后的 JavaScript 代码编译为实际的机器码时会利用“隐藏类”。
如果你的代码非常注重性能，那么这一点可能对你很重要

运行期间，V8 会将创建的对象与隐藏类关联起来，以跟踪它们的属性特征。
能够共享相同隐藏类 的对象性能会更好，V8 会针对这种情况进行优化，但不一定总能够做到

* 示例 1

```js
function Article() {
    this.title = "Inauguration Ceremony Features Kazoo Band";
}

// V8 会在后台配置，让这两个类实例共享相同的隐藏类，
// 因为这两个实例共享同一个构造函数和原型
let a1 = new Article();
let a2 = new Article();

// 假设之后又添加了下面这行代码
// 此时两个 Article 实例就会对应两个不同的隐藏类。
// 根据这种操作的频率和隐藏类的大小，这有 可能对性能产生明显影响
a2.author = "Jake";
```

* 示例 1 优化

```js
// 当然，解决方案就是避免 JavaScript 的 
// “先创建再补充”(ready-fire-aim)式的动态属性赋值，
// 并在 构造函数中一次性声明所有属性
function Article(opt_author) {
    this.title = "Inauguration Ceremony Features Kazoo Band";
    this.author = opt_author;
}

// 两个实例基本上就一样了(不考虑 hasOwnProperty 的返回值)，
// 因此可以共享一个隐藏类， 从而带来潜在的性能提升
let a1 = new Article();
let a2 = new Article("Jake");
```

* 示例 2
    - 使用 delete 关键字会导致生成相同的隐藏类片段
    - 在代码结束后，即使两个实例使用了同一个构造函数，
      它们也不再共享一个隐藏类。动态删除属性与动态添加属性导致的后果一样
    - 最佳实践是把不想要的属性设置为 `null`。这样可以保持隐藏类不变和继续共享，
      同时也能达到删除引用值供垃圾回收程序回收的效果

```js
function Article() {
    this.title = "Inauguration Ceremony Features Kazoo Band";
    this.author = "Jake";
}

let a1 = new Article();
let a2 = new Article();

delete a1.author;
```

* 示例 2 优化

```js
function Article() {
    this.title = "Inauguration Ceremony Features Kazoo Band";
    this.author = "Jake";
}

let a1 = new Article();
let a2 = new Article();

a1.author null;
```

### 内存泄漏

写得不好的 JavaScript 可能出现难以察觉且有害的内存泄漏问题。
在内存有限的设备上，或者在函数会被调用很多次的情况下，
内存泄漏可能是个大问题。JavaScript 中的内存泄漏大部分是由不合理的引用导致的

* 意外声明全局变量是最常见但也最容易修复的内存泄漏问题

```js
// 解释器会把变量 name 当作 window 的属性来创建(相当于 window.name = 'Jake')。 
// 可想而知，在 window 对象上创建的属性，只要 window 本身不被清理就不会消失
function setName() {
    name = "Jake";
}
```

```js
function setName() {
    let name = "Jake"; // var or const 也可以
}
```

* 定时器也可能会悄悄地导致内存泄漏
    - 只要定时器一直运行，回调函数中引用的 name 就会一直占用内存。
      垃圾回收程序当然知道这一点，因而就不会清理外部变量

```js
let name = 'Jake';

// 定时器的回调通过闭包引用了外部变量
setInterval(() => {
    console.log(name);
}, 100);
```

* 使用 JavaScript 闭包很容易在不知不觉间造成内存泄漏

```js
// 调用 outer()会导致分配给 name 的内存被泄漏。
// 以下代码执行后创建了一个内部闭包，
// 只要返回的函数存在就不能清理 name，
// 因为闭包一直在引用着它。
// 假如 name 的内容很大(不止是一个小字符 串)，
// 那可能就是个大问题了
let outer = function() {
    let name = 'Jake';
    return function() {
        return name;
    };
};
```

### 静态分配与对象池

为了提升 JavaScript 性能，最后要考虑的一点往往就是压榨浏览器了。
此时，一个关键问题就是如何减少浏览器执行垃圾回收的次数。

开发者无法直接控制什么时候开始收集垃圾，但可以间接控制触发垃圾回收的条件。
理论上，如果能够合理使用分配的内存，同时避免多余的垃圾回收，那就可以保住因释放内存而损失的性能

浏览器决定何时运行垃圾回收程序的一个标准就是对象更替的速度。
如果有很多对象被初始化，然后一下子又都超出了作用域，
那么浏览器就会采用更激进的方式调度垃圾回收程序运行，这样当然会影响性能

* 示例: 问题代码

调用这个函数时，会在堆上创建一个新对象，然后修改它，
最后再把它返回给调用者。如果这个矢量对象的生命周期很短，
那么它会很快失去所有对它的引用，成为可以被回收的值。
假如这个矢量加法函数频繁被调用，那么垃圾回收调度程序会发现这里对象更替的速度很快，
从而会更频繁地安排 垃圾回收。

```js
function addVector(a, b) {
    let resultant = new Vector();
    resultant.x = a.x + b.x;
    resultant.y = a.y + b.y;
    return resultant;
}
```

* 示例: 解决方案--不要动态创建矢量对象

```js
function addVector(a, b, resultant) {
    resultant.x = a.x + b.x;
    resultant.y = a.y + b.y;
    return resultant;
}
```

* 示例: 解决方法--使用对象池

在初始化的某一时刻，可以创建一个对象池，用来管理一组可回收的对象。
应用程序可以向这个对象池请求一个对象、设置其属性、使用它，
然后在操作完成后再把它还给对象池。由于没发生对象初始化，
垃圾回收探测就不会发现有对象更替，因此垃圾回收程序就不会那么频繁地运行

如果对象池只按需分配矢量(在对象不存在时创建新的，在对象存在时则复用存在的)，
那么这个实现本质上是一种贪婪算法，有单调增长但为静态的内存。
这个对象池必须使用某种结构维护所有对象，数组是比较好的选择。
不过，使用数组来实现，必须留意不要招致额外的垃圾回收

```js
// vectorPool 是已有的对象池
let v1 = vectorPool.allocate();
let v2 = vectorPool.allocate();
let v3 = vectorPool.allocate();

v1.x = 10;
v1.y = 5;
v2.x = -3;
v2.y = -6;

addVector(v1, v2, v3);
console.log([v3.x, v3.y]);  // [7, -1]

vectorPool.free(v1);
vectorPool.free(v2);
vectorPool.free(v3);

// 如果对象有属性引用了其他对象
// 则这里也需要把这些属性设置为null
v1 = null;
v2 = null;
v3 = null;
```