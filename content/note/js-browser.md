---
title: JavaScript 浏览器
author: 王哲峰
date: '2015-11-20'
slug: js-browser
categories:
  - 前端
tags:
  - tool
---

JavaScript 浏览器
====================================

1.JavaScript 浏览器简介
------------------------------------

目前主流的浏览器分这么几种：

   - IE 6~11：国内用得最多的 IE 浏览器，历来对 W3C 标准支持差。从 IE10 开始支持 ES6 标准
   - Chrome：Google 出品的基于 Webkit 内核浏览器，内置了非常强悍的 JavaScript 引擎——V8。
     由于 Chrome 一经安装就时刻保持自升级，所以不用管它的版本，最新版早就支持 ES6 了
   - Safari：Apple 的 Mac 系统自带的基于 Webkit 内核的浏览器，从 OS X 10.7 Lion 自带的 6.1 版本开始支持 ES6，
     目前最新的 OS X 10.11 El Capitan 自带的 Safari 版本是 9.x，早已支持 ES6
   - Firefox：Mozilla 自己研制的 Gecko 内核和 JavaScript 引擎 OdinMonkey。早期的 Firefox 按版本发布，
     后来终于聪明地学习 Chrome 的做法进行自升级，时刻保持最新
   - 移动设备上目前 iOS 和 Android 两大阵营分别主要使用 Apple 的 Safari 和 Google 的 Chrome，由于两者都是 Webkit 核心，
     结果 HTML5 首先在手机上全面普及(桌面绝对是 Microsoft 拖了后腿)，对 JavaScript 的标准支持也很好，最新版本均支持 ES6
   - 其他浏览器如 Opera 等由于市场份额太小就被自动忽略了
   - 另外还要注意识别各种国产浏览器，如某某安全浏览器，某某旋风浏览器，它们只是做了一个壳，其核心调用的是 IE，
     也有号称同时支持 IE 和 Webkit 的“双核”浏览器

.. note:: 

   - 不同的浏览器对 JavaScript 支持的差异主要是，有些 API 的接口不一样，比如 AJAX，File 接口。
     对于 ES6 标准，不同的浏览器对各个特性支持也不一样
   - 在编写 JavaScript 的时候，就要充分考虑到浏览器的差异，尽量让同一份 JavaScript 代码能运行在不同的浏览器中

2.浏览器对象
------------------------------------

   JavaScript 可以获取浏览器提供的很多对象，并进行操作:

      - window
      - navigator
      - screen
      - location
      - document
      - history

2.1 window
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

   - ``window`` 对象不但充当全局作用域，而且表示浏览器窗口
   - ``window`` 对象有 ``innerWidth`` 和 ``innerHeight`` 属性，可以获取浏览器窗口的内部宽度和高度
      
      - 内部宽高是指除去菜单栏、工具栏、边框等占位元素后，用于显示网页的净宽高

      .. code-block:: javascript

               // 调整浏览器窗口大小
               'use strict';
               console.log('window inner size: ' + window.innerWidth + ' x ' + window.innerHeight);

   - ``window`` 还有一个 ``outerWidth`` 和 ``outerHeight`` 属性，可以获取浏览器窗口的整个宽高
   - 兼容性: IE <= 8 不支持
   - window 对象的 open 方法来创建新的浏览器窗口

      .. code-block:: js

         window.open(url, name, features)

      .. code-block:: js

         // 调用 popUp 函数的一个办法是使用 伪协议(pseudo-protocol)
         function popUp(winURL) {
            window.open(winURL, "popup", "width=320,height=480");
         }



2.3 navigator
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

   - ``navigator`` 对象表示浏览器的信息，最常用的属性包括:

      - ``navigator.appName``: 浏览器名称
      - ``navigator.appVersion``: 浏览器版本
      - ``navigator.language``: 浏览器设置的语言
      - ``navigator.platform``: 操作系统类型
      - ``navigator.userAgent``: 浏览器设定的 User-Agent 字符串

   - navigator 的信息可以很容易地被用户修改，所以 JavaScript 读取的值不一定是正确的，
     所以可以充分利用 JavaScript 对不存在属性返回 ``undefined`` 的特性, 直接用短路运算符 ``||`` 计算

      .. code-block:: javascript

         var width = window.innerWidth || document.body.clientWidth;

示例:

   .. code-block:: javascript

      console.log('appName = ' + navigator.appName);
      console.log('appVersion = ' + navigator.appVersion);
      console.log('language = ' + navigator.language);
      console.log('platform = ' + navigator.platform);
      console.log('userAgent = ' + navigator.userAgent);

2.4 screen
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

   - ``screen`` 对象表示屏幕的信息，常用的属性有:

      - screen.width: 屏幕宽度，以像素为单位
      - screen.height: 屏幕高度，以像素为单位
      - screen.colorDepth: 返回颜色位数，如：8、16、24

示例:

   .. code-block:: javascript

      'use strict';
      console.log('Screen size = ' + screen.width + ' x ' + screen.height);

2.5 location
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

   - ``location`` 对象表示当前页面的 URL 信息
   - 页面的 URL 可以通过 ``location.href`` 获取，要获取 URL 各个部分的值：

      .. code-block:: javascript

         location.protocol;   // 'http'
         location.host;       // 'www.example.com'
         location.port;       // '8080'
         location.pathname;   // '/path/index.html'
         location.search;     // '?a=1&b=2'
         location.hash;       // 'TOP'
   - 要加载一个新页面，可以调用 ``location.assign()`` 方法
   - 要加载当前页面，调用 ``location.reload()`` 方法

      .. code-block:: javascript

         'use strict';
         if (confirm('重新加载当前项' + location.href + '?')) {
            location.reload();
         } else {
            location.assign('/'); // 设置一个新的 URL 地址
         }

2.6 document
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

   - ``document`` 对象表示当前页面，由于 HTML 在浏览器中以 DOM 形式表示树形结构的，
     ``document`` 对象就是整个 DOM 树的根节点
   - ``document`` 的 ``title`` 属性是从 HTML 文档中的 ``<title>xxx</title>`` 读取的，但是可以动态改变

      .. code-block:: javascript

         'use strict';
         document.title = "努力学习 JavaScript!"

   - 要查找 DOM 树的某个节点，需要从 ``document`` 对象开始查找，最常见的查找是 ID 和 Tag Name

      .. code-block:: html

         <dl id="drink-menu" style="border:solid 1px #ccc;padding:6px;">
            <dt>摩卡</dt>
            <dd>热摩卡咖啡</dd>
            <dt>酸奶</dt>
            <dd>北京酸奶</dd>
            <dt>果汁</dt>
            <dd>鲜榨苹果汁</dd>
         </dl>
      
      .. code-block:: javascript

         'use strict';
         var menu = document.getElementById('drink-menu');
         var drinks = document.getElementsByTagName('dt');
         var i, s;
         s = '提供的饮料有:';
         for (i=0; i<drinks.length; i++) {
            s = s + drinks[i].innerHTML + ',';
         }

   - ``document`` 对象还有一个 ``cookie`` 属性，可以获取当前页面的 Cookie

      - Cookie 是由服务器发送的 key-value 标示符。因为 HTTP 协议是无状态的，
        但是服务器要区分到底是哪个用户发过来的请求，就可以用 Cookie 来区分。
        当一个用户成功登录后，服务器发送一个 Cookie 给浏览器，例如 ``user=ABC123XYZ(加密的字符串)...``，
        此后，浏览器访问该网站时，会在请求头附上这个 Cookie，服务器根据 Cookie 即可区分出用户
      - Cookie 还可以存储网站的一些设置，例如，页面显示的语言等等
      - JavaScript 可以通过 ``doucment.cookie`` 读取到当前页面的 Cookie

      .. code-block:: javascript

         document.cookie;

      .. note:: 

         - 由于 JavaScript 能读取到页面的 Cookie，而用户的登录信息通常也存在 Cookie 中，这就造成了巨大的安全隐患，
           这是因为在 HTML 页面中引入第三方的 JavaScript 代码是允许的：

            .. code-block:: html
            
               <!-- 当前页面在wwwexample.com -->
               <html>
                  <head>
                     <script src="http://www.foo.com/jquery.js"></script>
                  </head>
                  ...
               </html>

            - 如果引入的第三方的 JavaScript 中存在恶意代码，则 www.foo.com 网站将直接获取到 www.example.com 网站的用户登录信息
            - 为了解决这个问题，服务器在设置 Cookie 时可以使用 ``httpOnly``，设定了 ``httpOnly`` 的 Cookie 将不能被 JavaScript 读取。
              这个行为由浏览器实现，主流浏览器均支持 ``httpOnly`` 选项，IE 从 IE6 SP1 开始支持
            - 为了确保安全，服务器端在设置 Cookie 时，应该始终坚持使用 ``httpOnly``


2.7 history
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

   - ``history`` 对象保存了浏览器的历史记录，JavaScript 可以调用 ``history`` 对象的 ``back()`` 或 ``forward()``，
     相当于用户点击了浏览器的后腿或前进按钮
   - ``history`` 对象属于历史遗留对象，对于现代 Web 页面来说，由于大量使用 AJAX 和页面交互，
     简单粗暴地调用 ``history.back()`` 可能会让用户感到非常愤怒
   - 任何情况下，都不应该使用 ``history`` 对象

3.操作 DOM
------------------------------------


4.操作表单
------------------------------------


5.AJAX
------------------------------------



6.Promise
------------------------------------


7.Canvas
------------------------------------

   Canvas 是 HTML5 新增的组件，它就像一块幕布，可以用 JavaScript 在上面绘制各种图表、动画等。
   在没有 Canvas 的年代，绘图只能借助 Flash 插件实现，页面不得不用 JavaScript 和 Flash 进行交互。
   有了 Canvas，我们就再也不需要 Flash 了，直接使用 JavaScript 完成绘制。

7.1 Canvas 简介
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

   -  一个 Canvas 定义了一个指定尺寸的矩形框，在这个范围内我们可以随意绘制：

      .. code-block:: html

         <canvas id="test-canvas" width="300" height="200"></canvas>

   -  由于浏览器对 HTML5 标准支持不一致，所以，通常在 ``<canvas>`` 内部添加一些说明性 HTML 代码，
      如果浏览器支持 Canvas，它将忽略 ``<canvas>`` 内部的 HTML，如果浏览器不支持 Canvas，
      它将显示 ``<canvas>`` 内部的 HTML：

      .. code-block:: html

         <canvas id="test-stock" width="300" height="200">
            <p>Current Price: 25.51</p>
         </canvas>

   - 在使用 Canvas 前，用 ``canvas.getContext`` 来测试浏览器是否支持 Canvas:

      .. code-block:: html

         <canvas id="test-canvas" width="200" height="100">
            <p>你的浏览器不支持 Canvas</p>
         </canvas>

      .. code-block:: js

         'use strict';
         var canvas = document.getElementById("test-canvas");
         if (canvas.getContext) {
            console.log("你的浏览器支持 Canvas");
         } else {
            console.log("你的浏览器不支持 Canvas!");
         }

   - ``getContext("2d")`` 方法让我们拿到一个 ``CanvasRenderingContext2D`` 对象，所有的绘图操作都需要通过这个对象完成：

      .. code-block:: html

         <canvas id="test-canvas" width="300" height="200"></canvas>

      .. code-block:: js

         var canvas = document.getElementById("test-canvas");
         var ctx = canvas.getContext("2d");

   - 如果需要绘制 3D，HTML5 还有一个  WebGL 规范，允许在 Canvas 中绘制 3D 图形：

      .. code-block:: html

         <canvas id="test-canvas" width="300" height="200"></canvas>

      .. code-block:: js

         var canvas = document.getElementById("test-canvas");
         var gl = canvas.getContext("webgl");

7.2 绘制形状
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

   - 可以在 Canvas 上绘制各种形状，在绘制前，需要先了解一下 Canvas 的坐标系统

      - Canvas 的坐标以左上角为原点，水平向右为 X 轴，垂直向下为 Y 轴，以像素为单位，所以每个点都是非负整数

      .. image:: ../images/canvas_shape.png

   - ``CanvasRenderingContext2D`` 对象有若干方法来绘制图形：

      .. code-block:: html

         <canvas id="test-shape-canvas" width="300" height="200"></canvas>

      .. code-block:: js
         
         'use strict';
         var canvas = document.getElementById("test-shape-canvas");
         var ctx = canvas.getContext("2d");

         // 擦除 (0, 0) 位置大小为 200x200 的矩形，擦除的意思是把该区域变为透明
         ctx.clearRect(0, 0, 200, 200);

         // 设置颜色
         ctx.fillStyle = "#dddddd";

         // 把 (10, 10) 位置大小为 130x130 的矩形涂色
         ctx.fillRect(10, 10, 130, 130);

         // 利用 Path 绘制复杂路径
         var path = new Path2D();
         path.arc(75, 75, 50, 0, Math.PI * 2, true);
         path.moveTo(110, 75);

         path.arc(75, 75, 35, 0, Math.PI, false);
         path.moveTo(65, 65);

         path.arc(60, 65, 5, 0, Math.PI * 2, true);
         path.moveTo(95, 65);

         path.arc(90, 65, 5, 0, Math.PI * 2, true);
         ctx.strokeStyle = "#0000ff";
         ctx.stroke(path);

7.3 绘制文本
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

   绘制文本就是在指定的位置输出文本，可以设置文本字体、样式、阴影等，与 CSS 完全一致：

      .. code-block:: html

         <canvas id="test-text-canvas" width="300" height="200"></canvas>

      .. code-block:: js

         'use strict';
         var canvas = document.getElementById("test-text-canvas");
         ctx.canvas.getContext("2d");

         ctx.clearRect(0, 0, canvas.width, canvas.height);
         ctx.shadowOffsetX = 2;
         ctx.shadowOffsetY = 2;
         ctx.shadowBlur = 2;
         ctx.shadowColor = "#666666";
         ctx.font = "24px Arial";
         ctx.fillStyle = "#333333";
         ctx.fillText("带阴影的文字", 20, 40);

7.4 其他
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

   Canvas 除了绘制基本的形状和文本，还可以实现动画、缩放、各种滤镜和像素转换等高级操作。
   如果要实现非常复杂的操作，考虑以下优化方案：

      - 通过创建一个不可见的 Canvas 来绘图，然后将最终绘制结果复制到页面的可见 Canvas 中
      - 尽量使用整数坐标而不是浮点数
      - 可以创建多个重叠的 Canvas 绘制不同的层，而不是在一个 Canvas 中绘制非常复杂的图
      - 背景图片如果不变可以直接用 ``<img>`` 标签并放到最底层
