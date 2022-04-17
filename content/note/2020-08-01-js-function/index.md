---
title: 函数
author: 王哲峰
date: '2020-08-01'
slug: js-function
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

- [内容](#内容)
- [函数定义方式](#函数定义方式)
  - [函数声明](#函数声明)
  - [函数表达式](#函数表达式)
  - [箭头函数](#箭头函数)
  - [使用 Function 构造函数](#使用-function-构造函数)
- [箭头函数](#箭头函数-1)
  - [箭头函数的定义](#箭头函数的定义)
  - [嵌入函数](#嵌入函数)
  - [参数括号](#参数括号)
  - [函数体大括号](#函数体大括号)
  - [箭头函数的局限性](#箭头函数的局限性)
- [函数名](#函数名)
  - [函数名就是函数对象的指针](#函数名就是函数对象的指针)
  - [函数名属性](#函数名属性)
- [理解参数](#理解参数)
  - [非箭头函数参数](#非箭头函数参数)
  - [箭头函数中的参数](#箭头函数中的参数)
- [没有重载](#没有重载)
  - [同名函数覆盖](#同名函数覆盖)
  - [模拟函数重载](#模拟函数重载)
- [默认参数值](#默认参数值)
  - [ECMAScript5 默认参数](#ecmascript5-默认参数)
  - [ECMAScript6 默认参数](#ecmascript6-默认参数)
  - [箭头函数默认参数](#箭头函数默认参数)
  - [默认参数与暂时性死区](#默认参数与暂时性死区)
- [参数扩展与收集](#参数扩展与收集)
  - [扩展参数](#扩展参数)
  - [收集参数](#收集参数)
- [函数声明与函数表达式](#函数声明与函数表达式)
- [函数作为值](#函数作为值)
- [函数内部](#函数内部)
  - [arguments](#arguments)
  - [this](#this)
  - [caller](#caller)
  - [new.target](#newtarget)
- [函数属性与方法](#函数属性与方法)
- [函数表达式](#函数表达式-1)
- [递归](#递归)
- [尾调用优化](#尾调用优化)
  - [尾调用优化的条件](#尾调用优化的条件)
  - [尾调用优化的代码](#尾调用优化的代码)
- [闭包](#闭包)
  - [this 对象](#this-对象)
  - [内存泄漏](#内存泄漏)
- [立即调用的函数表达式](#立即调用的函数表达式)
- [私有变量](#私有变量)
  - [静态私有变量](#静态私有变量)
  - [模块模式](#模块模式)
  - [模块增强模式](#模块增强模式)
</p></details><p></p>


# 内容

* 函数表达式、函数声明及箭头函数
* 默认参数及扩展操作符
* 使用函数实现递归
* 使用闭包实现私有变量

# 函数定义方式

* 函数是对象。每个函数都是 `Function` 类型的实例，而 `Function` 也有属性和方法，跟其他引用类型一样
* 因为函数是对象，所以函数名就是指向函数对象的指针，而且不一定与函数本身紧密绑定


## 函数声明

```js
function sum(num1, num2) {
   return num1 + num2;
}
```

## 函数表达式

```js
let sum = function(num1, num2) {
   return num1 + num2;
};
```

## 箭头函数

```js
let sum = (num1, num2) => {
   return num1 + num2;
};
```

## 使用 Function 构造函数
    
* Function 构造函数接收任意多个字符串参数，最后一个参数始终会被当成函数体，
  而之前的参数都是新函数的参数，不推荐使用, 会影响性能

```js
let sum = new Function(
   "num1", 
   "num2", 
   "return num1 + num2"
); // 不推荐
```

# 箭头函数

ECMAScript 6 新增了使用胖箭头 (`=>`) 语法定义函数表达式的能力

## 箭头函数的定义

**箭头函数**实例化的函数对象与正式的**函数表达式**创建的函数对象行为是相同的。
任何可以使用函数表达式的地方，都可以使用箭头函数。


* 语法

```javascript
let func_name = (arg1, arg2) => {
    statement
}
```

* 示例

```js
// 箭头函数
let arrowSum = (a, b) => {
    return a + b;
};

// 函数表达式
let functionExpressionSum = function(a, b) {
    return a + b;
};

console.log(arrowSum(5, 8)); // 13
console.log(functionExpressionSum(5, 8)); // 13
```

## 嵌入函数

箭头函数简洁的语法非常适合**嵌入函数**的场景

* 函数声明嵌入函数

```js
let ints = [1, 2, 3];
console.log(
    ints.map(
        function(i) { 
            return i + 1; 
        }
    )
); // [2, 3, 4]
```

* 箭头函数嵌入函数

```js
console.log(
    ints.map(
        (i) => { 
            return i + 1 
        }
    )
); // [2, 3, 4]
```

## 参数括号

如果只有一个参数，可以不用括号，只有没有参数，或者多个参数的情况下，才需要使用括号

```js
// 一个参数可以使用括号也可以不使用括号
let double = (x) => { return 2 * x; };
let triple = x => { return 3 * x; };

// 没有参数需要括号
let getRandom = () => { return Math.random(); };

// 多个参数需要括号
let sum = (a, b) => { return a + b; };
```

## 函数体大括号

箭头函数也可以不用大括号，但这样会改变函数的行为。
使用大括号就说明包含“函数体”，可以在一个函数中包含多条语句，跟常规的函数一样。
如果不适用大括号，那么箭头后面就只能有一行代码，比如一个赋值操作，
或者一个表达式。而且，省略大括号会隐式返回这行代码的值

```js
// 一行代码可以省略大括号
let double = (x) => { return 2 * x; };
let triple = (x) => 3 * x;

// 可以赋值
let value = {};
let setName = (x) => x.name = "Matt";
setName(value);
console.log(value.name); // "Matt"
```

## 箭头函数的局限性

箭头函数虽然语法简洁，但也有很多场合不适用

* 不能使用 `arguments`、`super`、`new.target`
* 不能用作构造函数
* 没有 `prototype` 属性

# 函数名

## 函数名就是函数对象的指针

因为函数名就是指向函数的指针，所以它们跟其他包含对象指针的变量具有相同的行为，
这意味一个函数可以有个多个名称

* 使用不带括号的函数名会访问函数指针，而不会执行函数

```js
function sum(num1, num2) {
   return num1 + num2;
}
console.log(sum(10, 10)); // 20

// 使用不带括号的函数名会访问函数指针，而不是执行函数，anotherSum 和 sum 都指向同一个函数
let anothreSum = sum; 
console.log(anotherSum(10, 10)); // 20

// 把 sum 设置为 null 之后，就切断了它与函数之间的关联
sum = null;
console.log(anotherSum(10, 10)); // 20
```

## 函数名属性

ECMAScript 6 的所有函数对象都会暴露一个只读的 `name` 属性，其中包含关于函数的信息
  
* 多数情况下，这个属性中保存的就是一个函数标识符，或者说一个字符串化的变量名，
  即使函数没有名称，也会如实显示成空字符串
* 如果函数对象是使用 `Function` 构造函数创建的，则会标识成 `anonymous`
* 如果函数是一个获取函数、设置函数，或者使用 `bind()` 实例化，
  那么标识符前面会加上一个前缀 `get`、`set`、`bind`

```js
// 函数声明
function foo() {}

// 函数表达式
let bar = function() {};

// 箭头函数
let baz = () => {};

console.log(foo.name); // foo
console.log(bar.name); // bar
console.log(baz.name); // baz

console.log((() => {}).name);       // (空字符串)

// Function 构造函数
console.log((new Function()).name); // anonymous

// 使用 bind() 实例化
console.log(foo.bind(null).name);   // bound foo

// 对象
let dog = {
   years: 1,
   get age() {
       return this.years;
   },
   set age(newAge) {
       this.years = newAge;
   }
};
let propertyDescriptor = Object.getOwnPropertyDescriptor(dog, "age");
console.log(propertyDescriptor.get.name); // get age
console.log(propertyDescriptor.set.name); // set age
```

# 理解参数

ECMAScript 函数既不关心传入的参数个数，也不关心这些参数的数据类型。
定义函数时要接收两个参数，并不意味着调用时就传两个参数。可以传一个、三个，
甚至一个也不传，解释器都不会报错

之所以会这样，主要是因为 ECMAScript 函数的参数在内部表现为一个数组，
函数被调用时总会接收一个数组，但函数并不关心这个数组中包含什么。
如果数组中什么也没有，那没问题; 如果数组的元素超出了要求，那也没问题

## 非箭头函数参数

在使用 `function` 关键字定义(非箭头)函数时，可以在函数内部访问 `arguments` 对象，
从中取得传进来的每个参数值。

ECMAScript 函数的参数只是为了方便才写出来的，并不是必须写出来的。
与其他语言不同，在 ECMAScript 中的命名参数不会创建让之后的调用必须匹配的函数签名。
这是因为根本不存在验证命名参数的机制

`arguments` 对象是一个类数组对象(但不是 Array 的实例)，
因此可以使用中括号语法访问其中的元素。要确定传进来多少个参数，
可以访问 `arguments.length` 属性

```js
function sayHi(name, message) {
    console.log("Hello " + name + ", " + message);
}

// 等价于
function sayHi() {
    console.log("Hello " + arguments[0] + ", " + arguments[1]);
}

// arguments.length
function howManyArgs() {
    console.log(arguments.length);
}

howManyArgs("string", 45); // 2
howManyArgs(); // 0
howManyArgs(12); // 1


function doAdd() {
    if (arguments.length === 1) {
        console.log(arguments[0] + 10);
    } else if (arguments.length === 2) {
        console.log(arguments[0] + arguments[1]);
    }
}

doAdd(10); // 20
doAdd(30, 20); // 50
```

* `arguments` 对象可以跟命名参数一起使用

```js
function doAdd(num1, num2) {
    if (arguments.length === 1) {
        console.log(num1 + 10);
    } else if (arguments.length === 2) {
        console.log(arguments[0] + num2);
    }
}
```

* `arguments` 的值始终会与对应的命名参数同步
    - 因为 `arguments` 对象的值会自动同步到对应的命名参数，
      所以修改 `arguments[1]` 也会修改 `num2` 的值，因此两者的值都是 `10`。
      但这并不意味着它们都访问同一个内存地址，它们在内存中还是分开的，
      只不过会保持同步而已

    - 如果只传了一个参数，然后把 `arguments[1]` 设置为某个值，
      那么这个值并不会反映到第二个命名参数。这是因为 `arguments` 对象的长度是根据传入的参数个数，
      而非定义函数时给出的命名参数个数确定的

```js
function doAdd(num1, num2) {
    arguments[1] = 10;
    console.log(arguments[0] + num2);
}
```

## 箭头函数中的参数

如果函数是使用箭头语法定义的，那么传给函数的参数将不能使用 `arguments` 关键字访问，
而只能通过定义的命名参数访问

```js
function foo() {
    console.log(arguments[0]);
}

let bar = () => {
    console.log(arguments[0]);
}
bar(5);  // ReferenceError: arguments is not defined
```

虽然箭头函数中没有 `arguments` 对象，但可以在**包装函数**中把它提供给箭头函数

```js
function foo() {
    let bar = () => {
        console.log(arguments[0]); // 5
    }
    bar();
}
foo(5);
```

<div class="warning" 
     style='background-color:#E9D8FD; color: #69337A; border-left: solid #805AD5 4px; border-radius: 4px; padding:0.7em;'>
    <span>
        <p style='margin-top:1em; text-align:left'>
            <b>Note</b>
        </p>
        <p style='margin-left:1em;'>
            ECMAScript 中的所有参数都是按值传递的。不可能按引用传递参数。
            如果把对象作为参数传递，那么传递的值就是这个对象的引用
        </p>
        <p style='margin-bottom:1em; margin-right:1em; text-align:right; font-family:Georgia'> 
            <b></b> 
            <i></i>
        </p>
    </span>
</div>


# 没有重载

ECMAScript 函数不能像传统编程那样重载：一个函数可以有两个定义，
只要签名(接收参数的类型和数量)不同就行。

ECMAScript 函数没有签名，因为参数是由包含零个或多个值的数组表示的。
没有函数签名，自然也就没有重载

## 同名函数覆盖

如果 ECMAScript 中定义了两个同名函数，则后定义的会覆盖先定义的

```js
function addSomeNumber(num) {
    return num + 100;
}

function addSomeNumber(num) {
    return num + 200;
}

let result = addSomeNumber(100);  // 300
```

## 模拟函数重载

可以通过检查参数的类型和数量，然后分别执行不同的逻辑来模拟函数重载

```js
let addSomeNumber = function(num) {
    return num + 100;
};

// 在创建第二个函数时，变量 addSomeNumber 被重写成保 存第二个函数对象了
addSomeNumber = function(num) {
    return num + 200;
};

let result = addSomeNumber(100); // 300
```

# 默认参数值

## ECMAScript5 默认参数

在 ECMAScript 5 及之前，实现默认参数的一种常用方式就是检测某个参数是否等于 undefined，
如果是则意味着没有传这个参数，那就给它赋一个值

```js
function makeKing(name) {
    name = (typeof name !== 'undefined') ? name : "Henry";
    return `King ${name} VIII`;
}

console.log(makeKing()); // King Henry VIII
console.log(makeKing("Louis")); // King Louis VIII
```

## ECMAScript6 默认参数

* EMCAScript 6 之后就不用这么麻烦了，因为它支持显示定义默认参数了。
  只要在函数定义中的参数后面用 `=` 就可以为参数赋一个默认值

```js
function makeKing(name = "Henry") {
    return `King ${name} VIII`;
}

console.log(makeKing("Louis")); // King Louis VIII
console.log(makeKing()); // King Henry VIII
```

* 给参数传 `undefined` 相当于没有传值，不过这样可以利用多个独立的默认值

```js
function makeKing(name = "Henry", numerals = "VIII") {
    return `King ${name} ${numerals}`;
}

console.log(makeKing());  // 'King Henry VIII'
console.log(makeKing('Louis'));  // 'King Louis VIII'
console.log(makeKing(undefined, 'VI'));  // 'King Henry VI'
```

* 在使用默认参数时，`arguments` 对象的值不反映参数的默认值，只反映传给函数的参数，
  当然，跟 ES5 严格模式一样，修改命名参数也不会影响 `arguments` 对象，
  它始终以调用函数时传入的值为准

```js
function makeKing(name = "Henry") {
    name = "Louis";
    return `King ${arguments[0]}`;
}

console.log(makeKing()); // King undefined
console.log(makeKing("Louis")); // King Louis
```

* 默认参数值并不限于原始值或对象类型，也可以使用调用函数返回的值
* 函数的默认参数只有在函数被调用时才会求值，不会在函数定义时求值。
  而且，计算默认值的函数只有在调用函数但未传相应参数时才会被调用

```js
let romanNumerals = ["I", "II", "III", "IV", "V", "VI"];
let ordinality = 0;

function getNumerals() {
    // 每次调用后递增
    return romanNumerals[ordinality++];
}

function makeKing(name = "Henry", numerals = getNumerals()) {
    return `King ${name} ${numerals}`;
}

console.log(makeKing());                // 'King Henry I'
console.log(makeKing('Louis', 'XVI'));  // 'King Louis XVI'
console.log(makeKing());                // 'King Henry II'
console.log(makeKing());                // 'King Henry III'
```

## 箭头函数默认参数

箭头函数同样可以使用默认参数，只不过在只有一个参数时，就必须使用括号而不能省略了

```js
let makeKing = (name = "Henry") => `King ${name}`;
console.log(makeKing()); // King Henry
```

## 默认参数与暂时性死区

因为在求值默认参数时可以定义对象，也可以动态调用函数，
所以函数参数肯定是在某个作用域中求值的

* 给多个参数定义默认值实际上跟使用 `let` 关键字顺序声明变量一样，
默认参数会按照定义它们的顺序依次被初始化

```js
function makeKing(name = "Henry", numerals = "VIII") {
    return `King ${name} ${numerals}`;
}

// 等价于
function makeKing() {
    let name = "Henry";
    let numerals = "VIII";
    
    return `King ${name} ${numerals}`;
}
```




# 参数扩展与收集

- ECMAScript 6 新增了扩展操作符，使用它可以非常简洁地操作和组合集合数据
- 扩展操作符最有用的场景就是函数定义中的参数列表，在这里可以充分利用这门语言的弱类型及参数长度可变的特点。扩展操作符既可以用于调用函数时传参，也可以用于定义函数参数

## 扩展参数

- 在给函数传参时，有时候可能不需要传入一个数组，而是分别传入数组的元素

```js
let values = [1, 2, 3, 4];

// 这个函数希望将所有加数逐个传进来，然后通过迭代 arguments 对象来实现累加
function getSum() {
    let sum = 0;
    for (let i = 0; i < arguments.length; ++i) {
        sum += arguments[i];
    }
    return sum;
}

// 如果不适用扩展操作符，想把定义在这个函数这面的数组拆分，那么久得求助于 apply() 方法
console.log(getSum.apply(null, values)); // 10

// 但 在 ES6 中，可以通过扩展操作符极为简洁地实现这种操作，对可迭代对象应用扩展操作符，并将其作为一个参数传入，可以将可迭代对象拆分，并将迭代返回的每个值单独传入
console.log(getSum(...values)); // 10

// 因为数组的长度已知，所以在使用扩展操作符传参的时候，并不妨碍在其前面或后面再传其他的值，包括使用扩展操作符传其他参数
consoel.log(getSum(-1, ...value)); // 9
console.log(getSum(...values, 5)); // 15
console.log(getSum(-1, ..values, 5)); // 14
console.log(getSum(...values, ...[5, 6, 7])); // 28
```

- 对函数中的 arguments 对象而言，它并不知道扩展操作符的存在，而是调用函数时传入的参数接收每一个值

```js
let values = [1, 2, 3, 4];

function countArguments() {
    console.log(arguments.length);
}

countArguments(-1, ...values); // 5
countArguments(...values, 5); // 5
countArguments(-1, ...values, 5); // 6
countArguments(...values, ...[5, 6, 7]); // 7
```

- arguments 对象只是扩展操作符的一种方式。在普通函数和箭头函数中，也可以将扩展操作符用于命名参数，当然同时也可以使用默认参数

```js
function getProduct(a, b, c = 1) {
    return a * b * c;
}

let getSum = (a, b, c = 0) => {
    return a * b + c;
}

console.log(getProduct(...[1, 2])); // 2
console.log(getProduct(...[1, 2, 3])); // 6
console.log(getProduct(...[1, 2, 3, 4])); // 6

console.log(getSum(..[0, 1])); // 1
console.log(getSum(..[0, 1, 2])); // 3
console.log(getSum(..[0,1,2,3]))); // 3
```

## 收集参数

- 在构思函数定义时，可以使用扩展操作符把不同长度的独立参数组合为一个数组，这有点类似 arguments 对象的构造机制，只不过收集参数的结果会得到一个 Array 实例
    - 收集参数的前面如果还有命名参数，则会收集其余的参数
    - 如果没有则会得到空数组，因为收集的参数的结果可变，所以只能把它作为最后一个参数

```js
function getSum(...values) {
    // 顺序累加 values 中的所有值
    // 初始值的总和为 0
    return values.reduce((x, y) => x + y, 0); 
}

console.log(getSum(1, 2, 3)); // 6
```

```js
// 不可以
function getProduct(...values, lastValue) {}

// 可以
function ignoreFirst(firstValue, ...values) {
    console.log(values);
}
ignoreFirst(); // []
ignoreFirst(1); // []
ignoreFirst(1, 2); // [2]
ignoreFirst(1, 2, 3); // [2, 3]
```

- 箭头函数虽然不支持 arguments 对象，但支持收集参数的定义方式，因此可以实现与使用 arguments 一样的逻辑

```js
let getSum = (...values) => {
    return values.reduce((x, y) => x + y, 0);
}

console.log(getSum(1, 2, 3)); // 6
```

- 使用收集参数并不影响 arguments 对象，它仍然反映调用时传给函数的参数

```js
function getSum(...values) {
    console.log(arguments.length); // 3
    console.log(arguments); // [1, 2, 3]
    console.log(values); // [1, 2, 3]
}

console.log(getSum(1, 2, 3));
```

# 函数声明与函数表达式

- JavaScript 引擎在加载数据时对函数声明和函数表达式是区别对待的
    - JavaScript 引擎在任何代码执行之前，会先读取**函数声明**，并在执行上下文中生成函数定义，这个个过程叫做**函数声明提升(function declaration hoisting)**，即在执行代码时，JavaScript 引擎会先执行一遍扫描，把发现的函数声明提升到源代码树的顶部
    - 而**函数表达式**必须等到代码执行到它那一行，才会在执行上下文中生成函数定义

```js
// 没问题--函数声明
console.log(sum(10, 10));
function sum(num1, num2) {
    return num1 + num2;
}

// 会出错--函数表达式
console.log(sum(10, 10));
let sum = function(num1, num2) {
    return num1 + num2;
};

// 使用 var 也会碰到同样的问题--函数表达式
console.log(sum(10, 10));
var sum = function(num1, num2) {
    return num1 + num2;
};
```

- 在使用函数表达式初始化变量时，也可以给函数一个名称

```js
let sum = function sum() {};
```

# 函数作为值

- 因为函数名在 ECMAScript 中就是变量，所以函数可以用在任何可以使用变量的地方
    - 可以把函数作为参数传给一个函数
    - 可以在一个函数中返回另一个函数

```js
function callSomeFunction(someFunction, someArgument) {
    return someFunction(someArgument);
}

// 示例 1
function add10(num) {
    return num + 10;
}
let result1 = callSomeFunction(add10, 10);
console.log(result1); // 20


// 示例 2
function getGreeting(name) {
    return "Hello, " + name;
}
let result2 = callSomeFunction(getGreeting, "Nicholas");
console.log(result2); // Hello, Nicholas
```

```js
function createComparisonFunction(propertyName) {
    return function(object1, object2) {
        let value1 = object1[propertyName];
        let value2 = object2[propertyName];
        
        if (value1 < value2) {
            return -1;
        } else if (value1 > value2) {
            return 1;
        } else {
            return 0;
        }
    };
}

let data = [
    {
        name: "Zachary",
        age: 28
    },
    {
        name: "Nicholas",
        age: 29
    }
];
data.sort(createComparisonFunction("name"));
console.log(data[0].name); // Nicholas

data.sort(createaComparisonFunction("age"));
console.log(data[0].name); // Zachary
```

# 函数内部

- ECMAScript 5 中，函数内部存在两个特殊的对象
    - arguments
    - this
- ECMAScript 6 又新增了一个属性属性
    - new.target

## arguments

- arguments 对象有很多特征:

  - 一个类数组对象，包含调用函数时传入的所有参数

  - 只有以 function 关键字定义函数时才会有

  - 有个 callee 属性，是一个指向 arguments 对象所在函数的指针，因此可以在函数内部递归调用

    - 使用 `arguments.callee` 就可以让函数逻辑与函数名解耦
    - 在严格模式下运行的代码时不能访问 `arguments.callee` 的，因为访问会报错，此时可以使用命名函数表达式(named function expression)达到目的

  ```js
  // 阶乘函数一般定义成递归调用的，只要给函数一个名称，而且这个名称不会变，这样定义就没有问题。但是，这个函数要正确执行就必须保证函数名是 factorial，从而导致了紧密耦合
  
  // 经典的阶乘函数
  function factorial(num) {
      if (num <= 1) {
          return 1;
      } else {
          return num * factorial(num - 1);
      }
  }
  
  // 使用 arguments.callee
  // 这个重写之后的 factorial()函数已经用 arguments.callee 代替了之前硬编码的 factorial。 12 这意味着无论函数叫什么名称，都可以引用正确的函数
  function factorial(num) {
      if (num <= 1) {
          return 1;
      } else {
          return num * arguments.callee(num - 1);
      }
  }
  
  // 严格模式下使用 arguments.callee
  // 这里创建了一个命名函数表达式 f()，然后将它赋值给了变量 factorial。即使把函数赋值给另 一个变量，函数表达式的名称 f 也不变，因此递归调用不会有问题。这个模式在严格模式和非严格模式 下都可以使用
  "use strict";
  const factorial = (function f(num) {
      if (num < 1) {
          return 1;
      } else {
          return num * f(num - 1);
      }
  });
  ```

## this

- `this` 在标准函数和箭头函数中有不同的行为

    - 在标准函数中，this 引用的是把函数当成 **方法** 调用的上下文对象，这时候通常称其为 this 值(在网页的全局上下文中调用函数时，this 指向 window)

    ```js
    // window 全局对象
    window.color = "red";
    
    // 自定义对象
    let o = {
        color: "blue"
    };
    
    // 定义在全局上下文中的函数引用了this对象，这个this到底引用哪个对象必须到函数被调用时才能确定
    function sayColor() {
        console.log(this.color);
    }
    
    // 全局上下文中调用 sayColor(),this 指向 window,  this.color = window.color
    sayColor(); // "red"
    
    // sayColor() 赋值给了对象 O，sayColor() 作为 O 的方法调用，this 指向对象 o
    o.sayColor = sayColor;
    o.sayColor(); // "blue"
    ```

- 在箭头函数中，this 引用的是定义箭头函数的上下文

    ```js
    // window 全局对象
    window.color = "red";
    
    // 自定义对象
    let o = {
        color: "blue"
    };
    
    // 定义在 window 上下文中的箭头函数
    let sayColor = () => console.log(this.color);
    
    
    sayColor(); // "red"
    
    o.sayColor = sayColor;
    o.sayColor(); // "red"
    ```

- 在事件回调或定时回调中调用某个函数时，this 值指向的并非想要的对象。此时将回调函数写成箭头函数就可以解决问题，这是因为箭头函数中的 this 会保留定义该函数时的上下文

```js
function King() {
    this.royaltyName = "Henry";
    // this 引用 King 的实例
    setTimeout() => console.log((this.royaltyName), 10000);
}

function Queen() {
    this.royaltyName = "Elizabeth";
    // this 引用 window 对象
    setTimeout(function() { console.log(this.royaltyName); }, 1000);
}

new King(); // Henry
new Queen(); // undefined
```

## caller

- ECMAScript 5 会给函数对象上添加一个属性 `caller`，这个属性引用的是调用当前函数的函数，或者如果是在全局作用域中调用的则为 `null`

```js
function outer() {
    inner();
}

function inner() {
    console.log(inner.caller);
}

outer(); // outer() 函数的源代码

// 如果要降低耦合度，可以通过 arguments.callee.caller 来引用同样的值
function outer() {
    inner();
}

function inner() {
    console.log(arguments.callee.caller);
}
outer();
```

- 在严格模式下访问 arguments.callee 会报错
- ECMAScript 5 也定义了 arguments.caller，但在严格模式下访问它会报错，在非严格模式下则始终是 undefined。这是为了分清 arguments.caller 和函数的 caller 而故意为之的。而作为对这门语言的安全防护，这些改动也让第三方代码无法检测同 一上下文中运行的其他代码
- 严格模式下还有一个限制，就是不能给函数的 caller 属性赋值，否则会导致错误

## new.target

- ECMAScript 中的函数始终可以作为构造函数实例化一个新对象，也可以作为普通函数被调用
- EMCAScript 6 新增了检测函数是否使用 new 关键调用的 `new.target` 属性
    - 如果函数是正常调用的，则 new.target 的值是 undefined
    - 如果是使用 new 关键字调用的，则 new.target 将引用被调用的构造函数

```js
function King() {
    if (!new.target) {
        throw "King must be instantiated using 'new'"
    }
    console.log("King instantiated using 'new'");
}

new King(); // "King instantiated using 'new'"
King(); // Error: "King must be instantiated using 'new'"
```

# 函数属性与方法

- ECMAScript 中的函数是对象，因此有属性和方法，每个函数都有两个属性

    - `length`

        - 保存函数定义的命名参数的个数

    - `prototype`

        - 保存引用类型所有实例方法，这意味着 `toString()`、`valueOf()` 等方法实际上都保存在 `prototype` 上，进而由所有实例共享
        - 在 ECMAScript 5 中，prototype 属性是不可枚举的，因此使用 for-in 循环不会返回这个属性

    - 函数海还有两个方法，这两个方法都会以指定的 this 值来调用函数，即会设置调用函数时函数体内 this 对象的值

        - `apply()` 方法接收两个参数:
            - 函数内 this 的值
            - 一个参数数组，可以是 Array 的实例，但也可以是 arguments 对象

        ```js
        function sum(num1, num2) {
            return num1 + num2;
        }
        
        // callSum1()会调用 sum()函数，将 this 作为函数体内的 this 值(这里等于 window，因为是在全局作用域中调用的)传入，同时还传入了 arguments 对象
        function callSum1(num1, num2) {
            return sum.apply(this, arguments); // 传入 arguments 对象
        }
        
        // callSum2()也会调 用 sum()函数，但会传入参数的数组
        function callSum2(num1, num2) {
            return sum.apply(this, [num1, num2]); // 传入数组
        }
        
        console.log(callSum1(10, 10)); // 20
        console.log(callSum2(10, 10)); // 20
        ```

        - `call()` 方法与 `apply()` 的作用一样，只是传参的形式不同
            - 第一个参数跟 `apply()` 一样，也是 this 值
            - 剩下的要传给被调用函数的参数则是逐个传递的，换句话说，通过 call() 向函数传参时，必须将参数一个一个地列出来

        ```js
        function sum(num1, num2) {
            return num1 + num2;
        }
        
        function callSum(num1, num2) {
            return sum.call(this, num1, num2);
        }
        
        console.log(callSum(10, 10)); // 20
        ```

        - 到底是使用 `apply()` 还是 `call()` ，完全取决于怎么给要调用的函数传参更方便。如果想直接传 arguments 对象或者一个数组，那就用 `apply()`，否则，就用 `call()`。当然，如果不用给被调用的函数传参，则使用哪个方法都一样
        - `apply()` 和 `call()` 真正强大的地方不是给函数传参，而是控制函数调用上下文即函数体内 this 值的能力

        ```js
        window.color = "red";
        
        let o = {
            color: "blue"
        };
        
        function sayColor() {        // 全局函数
            console.log(this.color); // this.color 会求值为 window.color
        }
        
        sayColor(); 	       // red，this.color 会求值为 window.color
        sayColor.call(this);   // red，this.color 会求值为 window.color
        sayColor.call(window); // red，this.color 会求值为 window.color
        sayColor.call(o);      // blue，该调用把函数的执行上下文即 this 切换为对象 o
        ```

        

# 函数表达式

- 函数表达式看起来就像一个普通的变量定义和赋值，即创建一个函数再把它赋值给一个变量。这样创建的函数叫做**匿名函数(anonymous function)**，因为 function 关键字后面没有标识符。

    - 匿名函数有时候也被称为**兰姆达哈函数**
    - 未赋值给其他变量的匿名函数的 name 属性是空字符串
    - 函数表达式跟 JavaScript 中的其他表达式一样，需要先赋值再使用

    ```js
    // 千万别这样做
    if (condition) {
        function sayHi() {
            console.log("Hi!");
        }
    } else {
        function sayHi() {
            console.log("Yo!");
        }
    }
    
    // 没问题
    let sayHi;
    if (condition) {
        sayHi = function() {
            console.log("Hi!");
        };
    } else {
        sayHi = function() {
            console.log("Yo!");
        };
    }
    ```

- 创建函数并赋值给变量的能力也可以用在一个函数中把另一个函数当做返回值

    ```js
    function createComparisonFunction(propertyName) {
        return function (object1, object2) {
            let value1 = object1[propertyName];
            let value2 = object2[propertyName];
            
            if (value1 < value2) {
                return -1;
            } else if (value1 > value2) {
                return 1;
            } else {
                return 0;
            }
        };
    }
    ```

# 递归

- 递归函数通常的形式是一个函数通过名称调用自己
- 示例

```js
// 经典的递归阶乘函数
function factorial(num) {
    if (num < 1) {
        return 1;
    } else {
        return num * factorial(num - 1);
    }
}

// 如果把这个函数赋值给其他变量，就会出问题
let anotherFactorial = factorial;
factorial = null; 			      // factorial 已经不是函数了
console.log(anotherFactorial(4)); // 报错

// 使用 arguments.callee 可以避免上面的问题
function factorial(num) {
    if (num <= 1) {
        return 1;
    } else {
        return num * arguments.callee(num - 1);
    }
}

// 由于严格模式下是不能使用使用 arguments.callee 的，所以需要使用命名函数表达式(named function expression)来实现递归
// 这里创建了一个命名函数表达式 f()，然后将它赋值给了变量 factorial。即使把函数赋值给另 一个变量，函数表达式的名称 f 也不变，因此递归调用不会有问题。这个模式在严格模式和非严格模式 下都可以使用
"use strict";
const factorial = (function f(num) {
    if (num < 1) {
        return 1;
    } else {
        return num * f(num - 1);
    }
});
```

# 尾调用优化

- ECMAScript 6 新增了一项内存管理优化机制，让 JavaScript 引擎在满足条件时可以重用栈帧

    - 具体来说，这项优化非常适合 **尾调用**，即外部函数的返回值是一个内部函数的返回值

    ```js
    function outerFunction() {
        return innerFunction(); // 尾调用
    }
    ```

- 在 ES6 优化之前，这行这个例子会在内存中发生如下操作

    - (1) 执行到 outerFunction 函数体，第一个栈帧被推到栈上
    -  (2) 执行 outerFunction 函数体，到 return 语句。计算返回值必须先计算 innerFunction
    -  (3) 执行到 innerFunction 函数体，第二个栈帧被推到栈上
    -  (4) 执行 innerFunction 函数体，计算其返回值
    -  (5) 将返回值传回 outerFunction，然后 outerFunction 再返回值
    -  (6) 将栈帧弹出栈外

- 在 ES6 优化之后，执行这个例子会在内存中发生如下操作

    -  (1) 执行到 outerFunction 函数体，第一个栈帧被推到栈上
    -  (2) 执行 outerFunction 函数体，到达 return 语句。为求值返回语句，必须先求值 innerFunction
    - (3) 引擎发现把第一个栈帧弹出栈外也没问题，因为innerFunction的返回值也是outerFunction的返回值
    -  (4) 弹出 outerFunction 的栈帧
    -  (5) 执行到 innerFunction 函数体，栈帧被推到栈上
    -  (6) 执行 innerFunction 函数体，计算其返回值
    -  (7) 将 innerFunction 的栈帧弹出栈外

- 很明显，第一种情况下每多调用一次嵌套函数，就会多增加一个栈帧。而第二种情况下无论调用多

    少次嵌套函数，都只有一个栈帧。这就是 ES6 尾调用优化的关键: **如果函数的逻辑允许基于尾调用将其销毁，则引擎就会那么做**

## 尾调用优化的条件

- 尾调用优化的条件就是确定外部栈帧真的没有必要存在了。涉及的条件如下:  
    - 代码在严格模式下执行
    -  外部函数的返回值是对尾调用函数的调用
    - 尾调用函数返回后不需要执行额外的逻辑
    - 尾调用函数不是引用外部函数作用域中自由变量的闭包
- 差异化尾调用和递归尾调用是容易让人混淆的地方。无论是递归尾调用还是非递归尾调用，都可以 应用优化。引擎并不区分尾调用中调用的是函数自身还是其他函数。不过，这个优化在递归场景下的效 果是最明显的，因为递归代码最容易在栈内存中迅速产生大量栈帧

- 示例：违反尾调用优化条件

```js
// 无优化:尾调用没有返回
function outerFunction() {
    innerFunction();
}

// 无优化:尾调用没有直接返回
function outerFunction() {
    let innerFunctionResult = innerFunction();
    return innerFunctionResult;
}

// 无优化:尾调用返回后必须转型为字符串
function outerFunction() {
    return innerFunction().toString();
}

// 无优化:尾调用是一个闭包
function outerFunction() {
    let foo = "bar";
    function innerFunction() { return foo; }

    return innerFunction();
}
```

- 示例：符合尾调用优化条件

```js
"use strict";

// 有优化：栈帧销毁前执行参数计算
function outerFunction(a, b) {
    return innerFunction(a + b);
}

// 有优化：初始返回值不涉及栈帧
function outerFunction(a, b) {
    if (a < b) {
        return a;
    }
    return innerFunction(a + b);
}

// 有优化：两个内部函数都在尾部
function outerFunction(condition) {
    return condition ? innerFunctionA() : innerFunctionB();
}

```

## 尾调用优化的代码

- 可以通过把简单的递归函数转换为待优化的代码来加深对尾调用的理解
- 示例

```js
// 通过递归计算斐波那契数列的函数
// 显然这个函数不符合尾调用优化的条件，因为返回语句中有一个相加的操作。结果，fib(n)的栈 帧数的内存复杂度是 O(2n)
function fib(n) {
    if (n < 2) {
        return n;
    }
    return fib(n - 1) + fib(n - 2);
}
console.log(fib(0)); // 0
console.log(fib(1)); // 1
console.log(fib(2)); // 1 
console.log(fib(3)); // 2
console.log(fib(4)); // 3
console.log(fib(5)); // 5
console.log(fib(6)); // 8

// 因此，即使这么一个简单的调用也可以给浏览器带来麻烦
fib(1000);
```

```js
// 将其重构为满足优化条件的形式。为此可以使用两个嵌套的函数，外部函数作为基础框架，内部 函数执行递归
"use strict";

// 基础框架
function fib(n) {
    return fibImpl(0, 1, n);
}

// 执行递归
function fibImpl(a, b, n) {
    if (n === 0) {
        return a;
    }
    return fibImpl(b, a + b, n - 1);
}
```

# 闭包



## this 对象



## 内存泄漏

- 闭包（closure）指的是那些引用了另一个函数作用域中变量的函数，通常是在嵌套函数中实现的
    - 匿名函数经常被人误认为是闭包
- 

# 立即调用的函数表达式



# 私有变量



## 静态私有变量



## 模块模式



## 模块增强模式

