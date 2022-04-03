---
title: JavaScript 语句
author: 王哲峰
date: '2015-11-20'
slug: js-sentence
categories:
  - 前端
tags:
  - tool
---

JavaScript 语句
===============

1.条件判断
----------

   -  JavaScript 使用 ``if () {...} else {...}`` 进行条件判断

   -  JavaScript 中使用 ``if () {...} else {...}`` 永远都要加上 ``{}``

   -  JavaScript 把
      ``null``\ 、\ ``undefined``\ 、\ ``0``\ 、\ ``NaN``\ 、\ ``''``
      视为 ``false``\ ，其他值一概视为 ``true``

单条件判断：

.. code:: javascript

   var age  = 20;
   if (age >= 18) {
      alert("adult");
   } else {
      alert("teenager");
   }

多条件判断：

.. code:: javascript

   var age = 3;
   if (age >= 18) {
      alert("adult");
   } else if (age >= 6) {
      alert("teenager");
   } else {
      alert("kid");
   }

2.循环
------

   -  JavaScript 的循环有两种：

      -  ``for`` 循环：通过初试条件、结束条件和递增条件来循环执行语句块

      -  ``while``
         循环：只有一个判断条件，条件满足，就不断循环，条件不满足时则退出循环

      -  ``do ... while()``
         循环：不是在每次循环开始的时候判断条件，而是在每次循环完成的时候判断条件

   -  ``do ... while()`` 循环体至少会执行一次，而 ``for`` 和 ``while``
      循环则可能一次都不执行

   -  JavaScript的死循环会让浏览器无法正常显示或执行当前页面的逻辑，有的浏览器会直接挂掉，有的浏览器会在一段时间后提示你强行终止JavaScript的执行，因此，要特别注意死循环的问题

2.1 ``for`` 循环：
~~~~~~~~~~~~~~~~~~

一般形式：

.. code:: javascript

   var x = 0;
   var i;
   for (i = 1; i <= 10000; i++) {
      x = x + i;
   }
   x;

利用索引遍历数组：

.. code:: javascript

   var arr  = ["Apple", "Google", "Microsoft"];
   var i, x;
   for (i = 0; i < arr.lenght; i++) {
      x = arr[i];
      console.log(x);
   }

使用 ``break`` 语句退出循环：

.. code:: javascript

   var x = 0;
   for (;;) {
      if (x > 100) {
         break;
      }
      x ++;
   }

``for ... in`` 把一个对象的所有属性一次循环出来：

.. code:: javascript

   var o = {
      name: "Jack",
      age: 20,
      city: "Beijing",
   };
   for (var key in o) {
      console.log(key);
   }

.. code:: javascript

   var a = ["A", "B", "C"]
   for (var i in a) {
      console.log(i);
      console.log(a[i]);
   }

过滤掉对象继承的属性用 ``hasOwnProperty()``\ ：

.. code:: javascript

   var o = {
      name: "Jack",
      age: 20,
      city: "Beijing",
   }
   for (var key in o) {
      if (o.hasOwnProperty(key)) {
         console.log(key);
      }
   }

.. _header-n45:

2.2 ``while`` 循环
~~~~~~~~~~~~~~~~~~

一般形式：

.. code:: javascript

   var x = 0;
   var n = 99;
   while (n > 0) {
      x = x + n;
      n = n - 2;
   }
   x;

.. _header-n48:

2.3 ``do ... while``
~~~~~~~~~~~~~~~~~~~~

一般形式：

.. code:: javascript

   var n = 0;
   do {
      n = n + 1;
   } while (n < 100);
   n;

.. _header-n52:

3.函数
------

   -  函数内部语句在执行时，一旦遇到 ``return``
      时，函数就执行完毕，并将结果返回

   -  如果函数没有 ``return``
      语句，函数执行完毕后也会返回结果，只是结果是 ``undefined``

.. _header-n59:

3.1 JavaScript 函数定义方式：
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

方式一：

.. code:: javascript

   function abs(x) {
      if (x >= 0) {
         return x;
      } else {
         return -x;
      }
   }

方式二：

.. code:: javascript

   var abs = function (x) {
      if (x >= 0) {
         return x;
      } else {
         return -x;
      }
   };

.. _header-n65:

3.2 函数调用
~~~~~~~~~~~~

   -  由于 JavaScript
      允许传入任意个参数而不影响调用，因此传入的参数比定义的参数多也没有问题，虽然函数内部不需要这些参数，传入的参数比定义的少也没有问题

   -  如果没有传入定义的参数，函数将收到 ``undefined``\ ，计算结果为
      ``NaN``\ ，要避免函数收到 ``undefined``\ ，可以对参数进行检查

.. code:: javascript

   abs(10); // 10
   abs(-9); // 9
   abs(10, "blablaba"); // 10
   abs(-9, "haha", "hehe", null); // 9
   abs(); // Nan

.. code:: javascript

   function abs(x) {
      if (typeof x != "number") {
         throw "Not a number";
      } 
      if (x > 0) {
         return x;
      } else {
         return -x;
      }
   }

.. _header-n75:

3.3 ``arguments`` 关键字
~~~~~~~~~~~~~~~~~~~~~~~~

   -  JavaScript 还有一个免费赠送的关键字
      ``arguments``\ ，它只在函数内部起作用，并且永远指向当前函数的调用者传入所有的参数

   -  ``arguments`` 类似 ``Array``\ ，但是它不是一个 ``Array``

   -  利用
      ``arguments``\ ，你可以获得调用者传入的所有参数。也就是说，即使函数不定义任何参数，还是可以拿到函数值

   -  实际上 ``arguments`` 最常用于判断传入参数的个数

.. code:: javascript

   function foo(x) {
       console.log("x = " + x); //10
      for (var i = 0; i < arguments.length; i++) {
         console.log("arg " + i + " = " + arguments[i]);
      }
   }

   foo(10, 20, 30);

.. code:: javascript

   function abs() {
      if (arguments.length === 0) {
         return 0;
      }
      var x = arguments[0];
      return x > 0 ? x : -x;
   }

   abs();
   abs(10);
   abs(-9);

.. code:: javascript

   function foo(a, b, c) {
      if (arguments.length === 2) {
         c = b;
         b = null;
      }
   }

.. _header-n89:

3.4 ``rest`` 参数
~~~~~~~~~~~~~~~~~

   -  ``rest`` 参数只能写在最后，前面用 ``...``
      标识，从运行结果可知，传入的参数先绑定
      ``a``\ ，\ ``b``\ ，多余的参数以数组形式交给变量
      ``rest``\ ，所以，不再需要 ``arguments`` 就可以取到全部参数

   -  如果传入的参数连正常定义的参数都没有填满，也不要紧，\ ``rest``
      参数会接收一个空数组(注意不是 ``undefined``)

   -  因为 ``rest`` 参数是 ES65
      标准，所以使用前需要测试一个浏览器是否支持

由于 JavaScript 函数允许接收任意个参数，于是就不得不用 ``arguments``
来获取所有参数：

.. code:: javascript

   function foo(a, b) {
      var i, rest = [];
      if (arguments.length > 2) {
         for (i = 2; i < arguments.length; i++) {
               rest.push(arguments[i]);
         }
      }
      console.log("a = " + a)
      console.log("b = " + b)
      console.log(rest);
   }

为了获取除了已定义参数a、b之外的参数，我们不得不用arguments，并且循环要从索引2开始以便排除前两个参数，这种写法很别扭，只是为了获得额外的rest参数，有没有更好的方法，有，ES6
标准引入了 ``rest`` 参数，上面的函数可以改写成：

.. code:: python

   fucntion foo(a, b, ...rest) {
      console.log("a = " + a);
      console.log("b = " + b);
      console.log(rest);
   }

   foo(1, 2, 3, 4, 5);
   foo(1);

.. _header-n102:

3.5 ``return`` 语句
~~~~~~~~~~~~~~~~~~~

   -  JavaScript 引擎有一个在行末自动添加分号的机制，一次在写 ``reutrn``
      语句的时候需要注意

.. code:: javascript

   function foo() {
      return {name: "foo"};
   }
   foo(); // {name: "foo"}

.. code:: javascript

   function foo() {
      return 
         {name: "foo"};
   }
   foo(); // undefined

.. code:: javascript

   function foo() {
      return {
         name: "foo"
      };
   }
