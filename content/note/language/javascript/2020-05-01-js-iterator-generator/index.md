---
title: 迭代器与生成器
author: 王哲峰
date: '2020-05-01'
slug: js-iterator-generator
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

- [迭代器模式](#迭代器模式)
	- [可迭代协议](#可迭代协议)
	- [迭代器协议](#迭代器协议)
	- [自定义迭代器](#自定义迭代器)
	- [提前终止迭代器](#提前终止迭代器)
- [生成器](#生成器)
	- [生成器基础](#生成器基础)
	- [通过 yield 中断执行](#通过-yield-中断执行)
	- [生成器作为默认迭代器](#生成器作为默认迭代器)
	- [提前终止生成器](#提前终止生成器)
</p></details><p></p>



- 循环是迭代机制的基础，这是因为它可以指定迭代的次数，以及每次迭代要执行什么操作。每次循环都会在下一次迭代开始之前完成，而每次迭代的顺序都是事先定义好的

- 迭代会在一个有序集合上进行，数组是 JavaScript 中有序集合的最典型例子
	- 因为数组有已知的长度，且数组每一项都可以通过索引获取，所以整个数组可以通过递增索引来遍历
	
```js
let collection = ["foo", "bar", "baz"];
for (let index = 0; index < collection.length; ++index) {
	console.log(collection[index]);
}
```

- 由于如下原因，通过循环来执行例程并不理想：
	- 迭代之前需要事先知道如何使用数据结构
		- 数组中的每一项都只能先通过引用取得数组对象，然后再通过 `[]` 操作符取得特定索引位置上的项。这种情况并不适用于所有数据结构
	- 遍历顺序并不是数据结构固有的
		- 通过递增索引来访问数据是特定与数组类型的方式，并不适用于其他具有隐式顺序的数据结构
	
- ES5 新增了 `Array.prototype.forEach()` 方法，向通用迭代需求迈进了一步，但仍然不够理想
	- 这个方法解决了单独记录索引和通过数组对象取得值的问题，不过，没有办法表示迭代何时终止。因为这个方法只适用于数组，而且回调结构也比较笨拙
	
```js
let collection = ["foo", "bar", "baz"];
collection.forEach(item) => console.log(item);
// foo
// bar
// baz
```

# 迭代器模式

- 迭代器模式描述了一个方案，即可以把有些结构称为 **可迭代对象(iterable)**，因为它们实现了正式的 `Iterable` 接口，而且可以通过迭代器 `Iterator` 消费

	- 可迭代对象
		- 可迭代对象是一种抽象的说法。基本上，可以把可迭代对象理解成数组或集合这样的集合类型的对象，它们包含的元素是有限的，而且都具有无歧义的遍历顺序
		- 可迭代对象不一定是集合对象，也可以是仅仅具有类似数组行为的其他数据结构
		- 临时性可迭代对象可以实现为生成器

	```js
	// 数组的元素是有限的
	// 递增索引可以访问每个元素
	let arr = [3, 1, 4];
	
	// 集合的元素是有限的
	// 可以按插入的顺序访问每个元素
	let set = new Set().add(3).add(1).add(4);
	```

	- `Iterable` 接口
		- 任何实现` Iterable` 接口的数据结构都可以被实现 `Iterator` 接口的结构消费(consume)
	- `Iterator` 迭代器
		- 按需创建的一次性对象
		- 每个迭代器都会关联一个可迭代对象，而迭代器会暴露迭代其关联可迭代对象的 AP
		- 迭代器无需了解与其关联的可迭代对象的结构，只需要知道如何取得连续的值

## 可迭代协议

## 迭代器协议

## 自定义迭代器

## 提前终止迭代器

# 生成器

## 生成器基础

## 通过 yield 中断执行

## 生成器作为默认迭代器

## 提前终止生成器

