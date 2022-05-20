---
title: ES6 Cheet Sheet
author: 王哲峰
date: '2021-02-13'
slug: es6-cheat-sheet
categories:
  - javascript
tags:
  - tool
---

1. 模板字面量(template liternal)是允许嵌入表达式的字符串字面量: `${ninja}`
2. 块级作用域变量
   - 使用新的 `let` 关键字创建块级作用域变量：`let ninja = "Yoshi";`
   - 使用新的 `const` 关键字创建块级作用域常量，常量在创建后不能被重新赋值：`const ninja = "Yoshi";`
3. 函数参数
   - 剩余参数(rest parameter) 可以将未命中形参的参数创建为一个不定数量的数组

```js
function multiMax(first, ...remaining) {
   /* ... */
}

multiMax(2, 3, 4, 5); // first: 2;
```

   - 函数默认参数(default parameter) 允许在调用时没有值或 undefined 被传入时使用指定的默认参数值

```js
function do(ninja action = "skulk") {
   return ninja + " " + action;
}

do("Fuma"); // "Fuma skulk"
```

4. 扩展语法(spread operator) 允许一个表达式在期望多个参数(用于函数调用)或
   多个元素(用于数组字面量)或多个变量(用于解构赋值)的位置扩展：`[...items, 3, 4, 5]`
5. 箭头函数(arrow function) 可以创建语法更为简洁的函数。箭头函数不会创建自己的 `this` 参数，相反，它将继承使用执行上下文的 `this` 值：

```js
const values = [0, 3, 2, 5, 7, 4, 8, 1];
values.sort((v1, v2) => v1 - v2); /* OR */ 
values.sort((v1, v2) => {
   return v1 - v2;
});
values.forEach(value => console.log(value));
```

6. 生成器(generator) 函数能生成一组值的序列，但每个值的生成是基于每次请求，并不同于标准函数那样立即生成。
每当生成器函数生成了一个值，它都会暂停执行但不会阻塞后续代码执。使用 yield 来生成一个新的值：

```js
function *IdGenerator() {
   let id = 0;
   while(true) {
      yield ++id;
   }
}
```

7. `primise` 对象是对我们现在尚未得到但将来会得到值的占位符。它是对我们最终能够得知异步计算结果的一种保证。
promise 既可以成功也可以失败，并且一旦设定好了，就不能够有更多改变。通过调用传入的 resolve 函数，
一个 promise 就被成功兑现(resolve)(通过调用 reject 则 promise 被违背)。拒绝一个 promise 有两种方式：
显式拒绝，即在一个 promise 的执行函数中调用传入的 reject 方法；隐式拒绝，如果正处理一个 promise 的过程中抛出了一个异常。

   - 使用 `new Promise((resolve reject) => {});` 创建一个新的 promise 对象
   - 调用 `resolve` 函数来显式地兑现一个 `promise`。调用 `reject` 函数来显式地拒绝一个 `promise`，如果在处理的过程中发生异常则会隐式地拒绝该 promise
   - 一个 `promise` 对象拥有 `then` 方法，它接收两个回调函数(一个成功回调和一个失败回调)作为参数并返回一个 `promise`:

```js
myPromise.then(val => console.log("Success"), err => console.log("Error"));
```

   - 链式调用 `catch` 方法可以捕获 `promise` 的失败异常：`myPromise.catch(e => alert(e));`

8. 类(Class) 是 JavaScript 原型的语法糖

```js
class Person {
   constructor(name) {
      this.name = name;
   }
   dance() {
      return true;
   }
}

class Ninja extends Person {
   constructor(name, level) {
      super(name);
      this.level = level;
   }
   static compare(ninja1, ninja2) {
      return ninja1.level - ninja2.level;
   }
}
```

9. 代理(Proxy) 可对对象的访问进行控制。当与对象交互时(当获取对象的属性或调用函数时)可以执行自定义操作

```js
const p = new Proxy(target {
get: (target, key) => { /* Called when property accessed through proxy */ },
set: (target, key, value) => { /* Called when property set through proxy */ }
});
```

10.   映射(Map) 是键与值之间的映射关系

      - 通过 `new Map()` 创建一个新的映射
      - 使用 `set` 方法添加新映射
      - 使用 `get` 方法和获取映射
      - 使用 `has` 方法检测映射是否存在
      - 使用 `delete` 方法删除映射
11. `for ... of` 循环遍历集合或生成器
12. 集合(Set) 是一组非重复成员的集合
    - 通过 `new Set()` 创建一个新的集合
    - 使用 `add` 方法添加成员
    - 使用 `delete` 方法删除成员
    - 使用 `size` 属性获取集合中成员的个数
13. 对象与数组的解构(destructuring)
   - `const {name: ninjaName} = ninja;`
   - `const {firstNinja} = ["Yoshi"];`
14. 模块(Module) 是更大的代码组织单元，可以将程序划分为若干个小片段

```js
export class Ninja();         // 导出 Ninja 类
export default class Ninja{}; // 使用默认导出
export {ninja};               // 导出存在的变量
export {ninja as samurai};    // 导出时进行重命名

import Ninja from "Ninja.js";             // 导出默认值
import {ninja} from "Ninja.js";           // 导入单个导出
import * as Ninja from "Ninja.js";        // 导入整个模块的内容
import {ninja as iNinja} from "Ninja.js"; // 导入时重命名单个导出
```
