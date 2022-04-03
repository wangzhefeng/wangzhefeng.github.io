---
title: JavaScript 数据类型
author: 王哲峰
date: '2015-11-20'
slug: js-data
categories:
  - 前端
tags:
  - tool
---

JavaScript 数据类型
===================

1.Number
------------------------------------------

   -  JavaScript 不区分整数和浮点数，统一用 Number 表示

   .. code:: javascript

      123;
      0.456;
      1.2345e3;
      -99;
      NaN               // NaN 表示 Not a Number，当无法计算结果时用 NaN 表示
      Infinity          // Infinity 表示无限大，当数值超过了 JavaScript 的 Number 所能表示的最大值时，就表示为 Infinity
      1 + 2; 
      (1 + 2) * 5 / 2;
      2 / 0;            // Infinity
      0 / 0;            // NaN
      10 % 3;
      10.5 % 3;
      0xff00;
      0xa5b4c3d2;       // 十六进制表示整数

2.字符串
------------------------------------------

   - 字符串是以单引号或双引号括起来的文本
   - 如果字符串内部既包括 ``'`` 又包含 ``"``，可以用转义字符 ``\`` 来标识
   - ASCII 字符可以以 ``\x##`` 形式的十六进制表示
   - Unicode 字符可以以 ``\u####`` 形式表示
   - 由于多行字符串用 ``\n`` 写起来比较费事，所以最新的 ES6 标准新增了一种多行字符串的表示方法，用反引号表示
   - 模板字符串：

      - 要把多个字符串连接起来，可以用 ``+`` 连接
      - 如果有很多变量需要连接，用 ``+`` 就比较麻烦，ES6 新增了一种模板字符串，类似于多行字符串的表示，
        但它会自动替换字符串中的变量

   -  字符串是不可变的，如果对字符串的某个索引赋值，不会有任何错误，但是，也没有任何效果

2.1 多行字符串
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

   .. code:: js

      ```这是一个
      多行
      字符串```;

2.2 模板字符串
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

   .. code:: javascript

      var name = "小明";
      var age = 20;
      var message = "你好,  " + name + ", 你今年" + age + "岁了！";
      alert(message);

      var name = "小明";
      var age = 20;
      var message = `你好, ${name}, 你今年${age}岁了!`;
      alert(message);

.. _header-n34:

2.3 操作字符串
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

   .. code:: javascript

      var s = "Hello, world!";
      s.length;

      s[0];
      s[1];
      s[13]; // undefined; 超出范围的索引不会报错，但一律返回 undefined

   .. code:: javascript

      var s = "Test";
      s[0] = "X";
      alert(s); // s 仍然为 "Test"

.. _header-n38:

2.4 字符串方法
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

   -  ``toUpperCase``
   -  ``toLowerCase``
   -  ``indexOf``
   -  ``substring``

   .. code:: javascript

      var s = "Hello";
      s.toUpperCase();
      s.toLowerCase();
      s.indexOf("e");
      s.substering(0, 5);

.. _header-n49:

3.布尔值
---------------------------------------------

   -  布尔值和布尔代数的表示完全一致，一个布尔值只有 ``true``、``flase`` 两种值
   -  可以直接用 ``true``、``false`` 表示布尔值，也可以通过布尔算计算出来
   -  ``&&`` 运算是 ``与`` 运算，只有所有都为 ``true``，``&&`` 运算结果才是 ``true``
   -  ``||`` 运算是 ``或`` 运算，只要其中有一个是 ``true``，``||`` 运算结果就是 ``true``
   -  ``!`` 运算是 ``非`` 运算，它是一个单目运算符，把 ``true`` 变成 ``false``，``false`` 变成 ``true``

.. code:: javascript

   true;
   false;
   2 > 1;
   2 >= 3;

.. code:: javascript

   true && true;
   true && false;
   false && true && false;

   false || false;
   true || false;
   false || true;

   !true
   !false
   !(2 > 5);

.. code:: javascript

   var age =  15;
   if (age >= 18) {
      alert("adult");
   } else {
      alert("teenager");
   }

.. _header-n65:

4.比较运算符
--------------------------------------------

   -  JavaScript 在设计时，有两种比较运相等算符：

      -  ``==``

         -  会自动转换数据类型再比较，很多时候，会得到非常诡异的结果

      -  ``===``

         -  不会自动转换数据类型，如果数据类型不一致，返回 ``false``，如果一致，再比较

      -  由于 JavaScript 这个设计缺陷，不要使用 ``==`` 比较，始终坚持使用 ``===`` 比较

   -  另一个例外是 ``NaN`` 这个特殊的 Number 与所有其他值都不相等，包括它自己，唯一能判断 ``NaN`` 的方法是通过 ``isNaN()`` 函数
   -  浮点数在运算过程中会产生误差，因为计算机无法精确表示无限循环小数。要比较两个浮点数是否相等，只能计算他们之差的绝对值，看是否小于某个阈值

   .. code:: javascript

      2 > 5;
      5 >= 2;
      7 === 7;

      flase == 0;
      false === 0;

      NaN === NaN;
      isNaN(NaN);

      1 / 3 === (1 -2 / 3);
      Math.abs(1 / 3 === (1 -2 / 3)) < 0.0000001;

.. _header-n88:

5. ``nul`` 和 ``undefined``
------------------------------------------

   - ``null`` 表示一个空的值，它和 ``0`` 以及空字符串 ``''`` 不同，``0`` 是一个数值，``''`` 表示长度为 0 的字符串，而 ``null`` 表示空。
     JavaScript 中的 ``null`` 类似于 Python 中的 ``None``
   - ``undefined`` 表示未定义
   - JavaScript 的设计者希望用 ``null`` 表示一个空的值，而 ``undefined`` 表示值未定义。事实证明，这并没有什么卵用，区分两者的意义不大。大多数情况下，我们都应该用
     ``null``。``undefined`` 仅仅在判断函数参数是否传递的情况下有用。

.. _header-n98:

6.数组
------------------------------------------

   -  JavaScript 数组是一组按顺序排列的集合，集合的每个值称为元素
   -  JavaScript 的 ``Array`` 可以包含任意数据类型，并通过索引来访问每个元素，索引的起始值为 ``0``
   -  数组可以用 ``[]`` 或 ``Array()`` 函数创建，出于可读性的考虑，强烈建议使用 ``[]``

.. _header-n107:

6.1 创建数组
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

   .. code:: javascript

      [1, 2, 3.14, "Hello", null, true];
      new Array([1, 2, 3]);

.. _header-n109:

6.2 数组索引
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

   .. code:: javascript

      var arr = [1, 2, 3.14, "Hello", null, true];
      arr[0];
      arr[1];
      arr[2];

.. _header-n111:

6.3 ``Array.length``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

   -  直接给 ``Array`` 的 ``length`` 赋一个新的值会导致 ``Array`` 大小的变化

   .. code:: javascript

      var arr = [1, 2, 3.14, "Hello", null, true];
      arr.length;

   .. code:: javascript

      var arr = [1, 2, 3];
      arr.length;
      arr.length = 6;
      arr; // [1, 2, 3, undefined, undefined, undefined]
      arr.length = 2;
      arr; // [1, 2]

.. _header-n117:

6.4 Array 索引赋值
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

   - ``Array`` 可以通过索引把对应的元素修改为新的值，因此对 ``Array`` 的索引进行复制会直接修改这个 ``Array``

.. _header-n119:

6.5 ``indexOf``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

   - ``Array`` 可以通过 ``indexOf()`` 来搜索一个指定的元素的位置：

   .. code:: javascript

      var arr = [10, 20, '30', 'xyz'];
      arr.indexOf(10);
      arr.indexOf(20);
      arr.indexOf(30);
      arr.indexOf('30');

.. _header-n124:

6.6 ``slice``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. _header-n130:

6.7 ``push``\ 、\ ``pop``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. _header-n132:

6.8 ``unshift``\ 、\ ``shift``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


.. _header-n134:

6.9 ``sort``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

   .. code:: javascript

      var arr = ['a', 'c', 'A'];
      arr.sort();
      arr;

.. _header-n136:

6.10 ``reverse``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

   .. code:: javascript

      var arr = ['one', 'two', 'three'];
      arr.reverse();
      arr;

.. _header-n138:

6.11 ``splice``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

   .. code:: javascript

      var arr = ['Microsoft', 'Apple', 'Yahoo', 'AOL', 'Excite', 'Oracle'];
      arr.splice(2, 3, 'Google', 'Facebook');
      arr;
      arr.splice(2, 2);
      arr;
      arr.splice(2, 0, 'Google', 'Facebook');
      arr;

.. _header-n140:

6.12 ``concat``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

   -  ``concat()`` 方法把当前的 ``Array`` 和另一个 ``Array`` 连接起来，并返回一个新的 ``Array``
   -  ``concat`` 方法并没有修改当前 ``Array``\ ，而是返回了一个新的 ``Array``
   -  ``concat`` 方法可以接收任意个元素和 ``Array``\ ，并且自动把 ``Array`` 拆开，然后全部添加到新的 ``Array`` 里

   .. code:: javascript

      var arr = ["A", "B", "C"];
      var added = arr.concat([1, 2, 3]);
      added;
      arr;

   .. code:: javascript

      var arr = ["A", "B", "C"];
      arr.concat([1, 2, [3, 4]]); // ['A', 'B', 'C', 1, 2, 3, 4]

.. _header-n150:

6.13 ``join``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

   -  ``join`` 方法把当前 ``Array`` 的每个元素都用指定的字符串连接起来，然后返回连接后的字符串
   -  如果 ``Array`` 的元素不是字符串，将自动转换为字符串后再连接

   .. code:: javascript

      var arr = ["A", "B", "C", 1, 2, 3];
      arr.join("-");

.. _header-n157:

6.14 多维数组
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

   .. code:: javascript

      var arr = [[1, 2, 3], [400, 500, 600], "-"];

.. _header-n160:

7.对象
------------------------------------------

   -  对象是一组由 ``键-值`` 组成的无需集合
   -  JavaScript 对象的键都是字符串类型，值可以是任意数据类型
   -  要获取一个对象的属性，用 ``object.attribute`` 的方式

   .. code:: javascript

      var person = {
         name: "Bob",
         age: 20,
         tags: ["js", "web", "mobile"],
         city: "Beijing",
         hasCar: ture,
         zipcode: null,
      }
      person.name;
      person.zipcode;

.. _header-n170:

8.变量
------------------------------------------

   -  声明一个变量用 ``var`` 语句
   -  在 JavaScript 中，使用等号 ``=`` 对变量进行赋值，可以把任意数据类型赋值给变量，同一个变量可以反复赋值，而且可以是不同类型的变量，但是要注意只能用 ``var`` 声明一次
   -  要显示变量的内容，可以用 ``console.log(x)`` ，打开 Chrome 的控制台就可以看到

   .. code:: javascript

      var a;
      var $b = 1;
      var s_001 = "007";
      var Answer = true;
      var t = null;

.. _header-n181:

9. ``strict`` 模式
-----------------------------------------

   -  JavaScript 在设计之初，为了方便初学者学习，并不强制要求用 ``var``
      声明变量。这个设计错误带来了严重的后果：如果一个变量没有通过
      ``var`` 声明就被使用，那么该变量就自动被声明为全局变量

      -  在同一个页面的不同的 JavaScript 文件中，如果都不用 ``var``
         声明，敲好都使用了变量 ``i``\ ，将造成变量 ``i``
         的互相影响，产生难以调试的错误结果

      -  使用 ``var`` 声明的变量则不是全局变量，它的范围被限制在改变量被声明的函数体内，同名变量在不同的函数体内互不冲突

   -  为了修补 JavaScript 这一严重设计缺陷，ECMA 在后续规范中推出了 ``strict`` 模式，在 ``strict`` 模式下运行的 JavaScript
      代码，强制通过 ``var`` 声明变量，未使用 ``var`` 声明变量就使用的，将导致错误
   -  启用 ``strict`` 模式的方法是在 JavaScipt 代码的第一行写上 ``use strict``\ 。这是一个字符串，不支持 ``strict``
      模式的浏览器会把它当做一个字符串语句执行，支持 ``strict`` 模式的浏览器将开启 ``strict`` 模式运行 JavaScript
   -  不用 ``var`` 声明的变量会被视为全局变量，为了避免这一缺陷，所有的 JavaScript 代码都应该使用 ``strict`` 模式

   .. code:: javascript

      'use strict';
      // 如果浏览器支持 strict 模式，下面的代码将报 ReferenceError 错误；

      abc = "Hello, world";
      console.log(abc);
