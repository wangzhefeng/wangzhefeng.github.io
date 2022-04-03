---
title: JavaScript DOM 编程艺术
author: 王哲峰
date: '2015-11-20'
slug: js-DOM
categories:
  - 前端
tags:
  - tool
---

JavaScript DOM 编程艺术
===========================

1.DOM
---------------------------

如果没有文档(document)，DOM 也就无从谈起。
当创建一个网页并把它加载到 Web 浏览器中时，
DOM 就在幕后悄然而生。它把编写的网页文档转换为一个文档对象。

对象是一种自足的数据集合，与某个特定对象相关联的变量被称为这个对象的属性；
只能通过某个特定对象去调用的函数被称为这个对象的方法。

   - D: document
   - O: object(host object)
      
      - window object、BOM、Window Object Model
      - document object 

   - M: Model

      - DOM 代表着加载到浏览器窗口的当前网页
      - DOM 把一份文档(document)表示为一棵树，更具体地说，DOM 把文档表示为一棵节点树

.. note:: JavaScript 对象类型：

   - 用户自定义对象(user-defined object) 由程序员自行创建的对象
   - 内建对象(native object) 内建在 JavaScript 语言里的对象，如 Array、Math 和 Date
   - 宿主对象(host object) 由浏览器提供的对象

2.node--节点
---------------------------

文档是由节点构成的集合，在 DOM 里有许多不同类型的节点：

   - 元素节点(element node)
   - 文本节点(text node)
   - 属性节点(attribute node)

2.1 元素节点(element node)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

   - DOM 的原子是元素节点，元素节点在文档中的布局形成了文档的结构，标签的名字就是元素的名字
   - 元素可以包含其他元素

2.2 文本节点(text node)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

   - 在 XHTML 文档里，文本节点总是被包含在元素节点的内部，但并非所有的元素节点都包含有文本节点。

2.3 属性节点(attribute node)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

   - 属性节点用来对元素做出更加具体的描述
   - 属性总是被放在起始标签里，所以属性节点总是被包含在元素节点中
   - 并非所有的元素都包含着属性，但所有的属性都被元素包含

2.4 CSS
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

   - 除了DOM与网页结构打交道外，还可以通过 CSS 告诉浏览器应该如何显示一份文档的内容
   - CSS 语法(类似于 JavaScript 函数的定义)

      .. code-block:: css

         selector {
            property: value;
         }

   -  继承(inheritance) 是 CSS 中的一项强大的功能，类似于 DOM，
      CSS 也把文档的内容视为一颗节点树。节点树上的各个元素将继承其父元素的样式属性。
      在某些场合，档把样式应用于一份文档时，其实只是想让那些样式作用于特定的元素。
      为了获得如此精细的控制，需要在文档里插入一些能够把这段文本与其他段落区别开来的特殊标志。

      1. class 属性

         - 可以在所有元素上任意应用 class 属性
         - 在 CSS 里，可以为 class 属性值相同的所有元素定义同一种样式，也可以为 class 属性为一种特定类型的元素定义一种特定的样式

         .. code-block:: html
            
            <p class="sepcial">This paragraph has the special class</p>
            <h2 class="special">So dose this headline</h2>

         .. code-block:: css

            .special {
               font-style: italic;
            }

            h2.special {
               text-transform: uppercase;
            }

      2. id 属性

         - id 属性的用途是给网页里的某个元素加上一个独一无二的标识符
         - 在 CSS 里，可以为特定 id 属性的元素定义一种独享的样式

         .. code-block:: html

            <ul id="pruchase">
               <li>A tin of beans</li>
               <li class="sale">Cheese</li>
               <li class="sale important">Milk</li>
            </ul>

         .. code-block:: css

            #pruchase {
               border: 1px solid white;
               background-color: #333;
               color: #ccc;
               padding: 1em;
            }

         - 尽管 id 本身只能使用一次，CSS 还是可以利用 id 属性为包含该特定元素里的其他元素定义样式

         .. code-block:: css

            #pruchase li {
               font-weight: blod;
            }

2.5 获取元素
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

有三种 DOM 方法可以获取元素节点，分别是通过元素 id、标签名字、class名字

1. ``document.getElementById(idName)``

   - 这个调用将返回一个对象，这个对象对应着 document 对象里的一个独一无二的元素，可以使用 typeof 操作符来验证这一点
   - 事实上，document 中的每一个元素都是一个对象，利用 DOM 提供的方法能得到任何一个对象

2. ``document.getElementByTagName(tagName)``

   - 一般来说，用不着为文档里的每一个元素都定义一个独一无二的 id 值，DOM 提供了一个方法来获取那些没有 id 属性的对象
   - getElementByTagName 方法返回一个对象数组，每个对象分别对应着文档里有着给定标签的一个元素

   .. code-block:: js

      document.getElementByTagName("li");

3. ``document.getElementByClass(className)``

   - 这个方法的返回值与 getElementByTagName 类似，都是一个具有相同类名的元素的数组
   - 使用这个方法还可以查找那些带有多个 className 的元素，要指定多个类名，只要在字符串参数中用空格分隔列名即可


2.6 获取和设置属性
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

得到需要的元素后，可以设法获取它的各个属性，``getArrtibute`` 和 ``setAttribute`` 方法可以分别获取属性和更改属性节点的值

1. ``getAttribute``

   - 语法：getAttribute 是一个函数，它只有一个参数，就是查询的属性的名字：

      .. code-block:: js

      object.getAttribute(attribute)

   - 与此前介绍的方法不同，getAttribute 方法不属于 document 对象，所以不能通过 document 对象调用，它只能通过元素节点对象调用

2. ``setAttribute``

   - 语法：setAttribute 允许对属性节点的值做出修改，与 getAttribute 一样，setAttribute 也只能用于元素节点

      .. code-block:: js

         object.setAttribute(attribute, value)

   -  通过 setAttribute 对文档做出修改之后，在通过浏览器的查看源代码选项去查看文档的源代码时看到的仍将是改变前的属性值，
      也就是说，setAttribute 做出的修改不会反映在文档本身的源代码里。这种表里不一的现象源自于 DOM 的工作模式，
      先加载文档的静态内容，再动态刷新，动态刷新不影响文档的静态内容。这正是 DOM 的真正威力：对页面内容进行刷新却不需要在浏览器里刷新页面

2.7 DOM 其他属性和方法
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

   - nodeName
   - nodeValue
   - childNodes
   - nextSibling
   - parentNode