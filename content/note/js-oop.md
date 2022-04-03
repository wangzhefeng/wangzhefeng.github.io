---
title: JavaScript 面向对象编程
author: 王哲峰
date: '2015-11-20'
slug: js-oop
categories:
  - 前端
tags:
  - tool
---

JavaScript 面向对象编程
=====================================

   JavaScript 的所有数据都就可以看成是对象，JavaScript 的面向对象编程和大多数其他语言如 Java、C# 的面向对象编程都不太一样，
   其他语言的面向对象的两个基本概念：

      - 类：类是对象的类型模板
      - 实例：实例是根据类创建的对象

   在 JavaScript 中，不区分类和实例的概念，而是通过 **原型(prototype)** 来实现面向对象编程。
   JavaScript 的原型链和 Java 的 Class 区别就在，它没有“Class”的概念，所有对象都是实例，
   所谓继承关系不过是把一个对象的原型指向另一个对象而已。

- 示例

   .. code-block:: javascript

      // 1.有一个 robot 对象
      var robot = {
         name: "Robot",
         height: 1.6,
         run: function () {
            console.log(this.name + ' is running...');
         }
      };
      
      // 2.创建 xiaoming 对象
      // 2.1 修改 robot 对象，创建 xiaoming
      var Student = {
         name: "Robot",
         height: 1.2,
         run: function () {
            console.log(this.name + ' is running...');
         }
      };
      var xiaoming = {
         name: "小明"
      };
      // 2.2 把 xiaoming 的原型指向了对象 Student，看上去 xiaoming 仿佛是从 Student 继承下来的
      xiaoming.__proto__ = Student;
      // 2.3 xiaoming 对象的调用
      xiaoming.name; // '小明'
      xiaoming.run(); // 小明 is running...

      // 3.把 xiaoming 的原型指向其他对象
      var Bird = {
         fly: function () {
            console.log(this.name + ' is flying...');
         }
      };
      xiaoming.__proto__ = Bird;
      xiaoming.fly(); // 小明 is flying...

   .. note:: 

      - 请注意，上述代码仅用于演示目的。在编写 JavaScript 代码时，
        不要直接用 ``obj.__proto__`` 去改变一个对象的原型，
        并且，低版本的IE也无法使用 ``__proto__``。 ``Object.create()`` 方法可以传入一个原型对象，
        并创建一个基于该原型的新对象，但是新对象什么属性都没有，因此，我们可以编写一个函数来创建 ``xiaoming``:

         .. code-block:: javascript

            // 原型对象
            var Student = {
               name: 'Robot',
               height: 1.2,
               run: function () {
                  console.log(this.name + 'is running...');
               }
            };

            // 创建基于 Student 的对象的函数
            function createStudent(name) {
               // 基于 Student 原型创建一个新对象
               var s = Object.create(Student);
               // 初始化新对象
               s.name = name;
               // s.height = height;
               return s;
            };

            // 创建 xiaoming
            var xiaoming = createStudent("小明");

            // 调用 xiaoming
            xiaoming.name;
            // xiaoming.height;
            xiaoming.run(); // 小明 is running....
            xiaoming.__proto__ === Student; // true


   ECMA-262 将对象定义为一组属性的无需集合。严格来说，这意味着对象就是一组没有特定顺序的值。
   对象的每个属性或方法都由一个名称来标识，这个名称映射到一个值。真因为如此(以及其他还未讨论的原因)，
   可以把 ECMAScript 的对象想象成一张散列表，其中的内容就是一组名/值对，值可以是数据或者函数。

      - 理解对象
      - 理解对象创建过程
      - 理解继承
      - 理解类

1.理解对象
-------------------------------------

   1.创建自定义对象的通常方式是创建 Object 的一个新实例，然后再给它添加属性和方法

      .. code-block:: javascript

         let person = new Object();
         person.name = "Nicholas";
         person.age = 29;
         person.job = "Software Engineer";
         person.sayName = function() {
            console.log(this.name);  // this.name == person.name
         };

   2.对象字面量更加流行

      .. code-block:: javascript

         let person = {
            name: "Nicholas",
            age: 29,
            job: "Software Engineer",
            sayName() {
               console.log(this.name);
            }
         };

1.1 属性的类型
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

   ECMA-262 使用一些内容特性来描述属性的特征。这些特性是由为 JavaScript 实现引擎的规范定义的。
   因此，开发者不能在 JavaScript 中直接访问这些特性。为了将某个特性标识位内部特性，
   规范会用两个中括号把特性的名称括起来，比如 ``[[Enumerable]]``.

   属性分两种：**数据属性** 和 **访问属性**

1.1.1 数据属性
^^^^^^^^^^^^^^

   - 数据属性包含一个保存数据值的位置。值会从这个位置读取，也会写入到这个位置。数据属性有 4 个描述他们的行为.
   - 要修改属性的默认特性，就必须使用 ``Object.defineProperty()`` 方法，这个方法接收 3 个参数：要给其添加属性的对象、
     属性的名称、一个描述符对象

      - ``[[Configurable]]``
      - ``[[Enumerable]]``
      - ``[[Writable]]``
      - ``[[Value]]``

   .. code-block:: javascript

      let person = {};
      Object.defineProperty(person, "name", {
         configurable: false,
         writable: false,
         value: "Nicholas"
      });

      console.log(person.name); // "Nicholas"

      person.name = "Greg";
      console.log(person.name); // "Nicholas"

      delete person.name;
      console.log(person.name); // "Nicholas"

      // 抛出错误
      Object.defineProperty(person, "name", {
         configurable: true,
         value: "Nicholas"
      });

   .. note:: 

      在调用 ``Object.defineProperty()`` 时，``configurable``、``enumerable`` 和 ``writable`` 的值如果不 指定，
      则都默认为 ``false``。多数情况下，可能都不需要 ``Object.defineProperty()`` 提供的这些强大的设置，
      但要理解 JavaScript 对象，就要理解这些概念。

1.1.2 访问属性
^^^^^^^^^^^^^^


1.2 定义多个属性
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~



1.3 读取属性的特性
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~



1.4 合并对象
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~




1.5 对象表示及相等判定
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


1.6 增强的对象语法
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


1.7 对象解构
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~





2.创建对象
-------------------------------------



3.继承
-------------------------------------





4.类
-------------------------------------



