---
title: Linux Shell
author: 王哲峰
date: '2022-05-07'
slug: linux-shell
categories:
  - Linux
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

- [Linux Shell](#linux-shell)
  - [1.什么是Shell](#1什么是shell)
  - [2.搭建运行环境](#2搭建运行环境)
- [运行Bash环境](#运行bash环境)
- [切换到Bash运行环境](#切换到bash运行环境)
- [确认命令行中运行的是Bash](#确认命令行中运行的是bash)
- [确保执行的命令具有可执行权限](#确保执行的命令具有可执行权限)
- [运行脚本](#运行脚本)
  - [3.Shell基本语法](#3shell基本语法)
- [helloworld.sh](#helloworldsh)
- [helloworld.sh](#helloworldsh-1)
- [helloworld.sh](#helloworldsh-2)
- [查找字符串`i`或`o`的位置(哪个字母先出现就计算哪个)](#查找字符串i或o的位置哪个字母先出现就计算哪个)
- [method 1](#method-1)
- [method 2](#method-2)
- [method 3](#method-3)
- [method 4](#method-4)
- [params.sh](#paramssh)
- [取得数组元素的个数](#取得数组元素的个数)
- [or](#or)
- [取得数组单个元素的长度](#取得数组单个元素的长度)
- [-e: 开启转义](#-e-开启转义)
- [-e: 开启转义, \c不换行](#-e-开启转义-c不换行)
- [格式只指定了一个参数, 但多出的参数仍然会按照该格式输出, format-string 被重用](#格式只指定了一个参数-但多出的参数仍然会按照该格式输出-format-string-被重用)
- [如果没有 arguments, 那么 %s 用NULL代替, %d 用 0 代替](#如果没有-arguments-那么-s-用null代替-d-用-0-代替)
- [author: zfwang](#author-zfwang)
- [file: demo.sh](#file-demosh)
- [函数调用](#函数调用)
- [author: zfwang](#author-zfwang-1)
- [file: funWithReturn.sh](#file-funwithreturnsh)
- [函数调用](#函数调用-1)
- [author: zfwang](#author-zfwang-2)
- [file: funWithParam.sh](#file-funwithparamsh)
- [函数调用](#函数调用-2)
- [将命令的完整的输出重定向到users文件中](#将命令的完整的输出重定向到users文件中)
- [查看users文件中的内容](#查看users文件中的内容)
- [将输出重定向覆盖users文件中的内容](#将输出重定向覆盖users文件中的内容)
- [查看users文件中的内容](#查看users文件中的内容-1)
- [将输出重定向追加到users文件的末尾](#将输出重定向追加到users文件的末尾)
- [统计users文件中的行数(会输出文件名users)](#统计users文件中的行数会输出文件名users)
- [统计users文件中的行数(不会输出文件名users)](#统计users文件中的行数不会输出文件名users)
- [同时替换输入和输出,执行command,从文件infile读取内容, 然后将输出写入到outfile中](#同时替换输入和输出执行command从文件infile读取内容-然后将输出写入到outfile中)
- [将两个delimiter之间的内容(document)作为输入传递给command](#将两个delimiter之间的内容document作为输入传递给command)
- [通过wc -l 命令计算Here Document的行数](#通过wc--l-命令计算here-document的行数)
- [hereDocument.sh](#heredocumentsh)
- [屏蔽stdout和stderr](#屏蔽stdout和stderr)
- [or](#or-1)
- [file: test1.sh](#file-test1sh)
- [file: test2.sh](#file-test2sh)
- [使用`.`来引用test1.sh文件](#使用来引用test1sh文件)
- [使用`source`来引用test1.sh文件](#使用source来引用test1sh文件)
  - [4.Shell程序设计过程](#4shell程序设计过程)
  - [5.Shell数值运算](#5shell数值运算)
- [test](#test)
  - [6.Shell 布尔运算](#6shell-布尔运算)
  - [7.文件操作](#7文件操作)
  - [8.Shell程序调试方法](#8shell程序调试方法)
  - [9.用户管理](#9用户管理)
- [创建家目录、指定登录 Shell](#创建家目录指定登录-shell)
- [创建家目录、指定登录 Shell、加入所属组](#创建家目录指定登录-shell加入所属组)
- [删除用户以及家目录](#删除用户以及家目录)
- [常常用来修改默认的 Shell](#常常用来修改默认的-shell)
- [把用户加入某个新安装软件所属的组](#把用户加入某个新安装软件所属的组)
- [修改登录用户名并搬到新家](#修改登录用户名并搬到新家)
- [禁用某个账号](#禁用某个账号)
- [让用户 test 无需密码登录(密码为空), 这个很方便某些安全无关紧要的条件下(比如已登录主机中的虚拟机), 可避免每次频繁输入密码](#让用户-test-无需密码登录密码为空-这个很方便某些安全无关紧要的条件下比如已登录主机中的虚拟机-可避免每次频繁输入密码)
</p></details><p></p>


Linux Shell
===============

1.什么是Shell
-------------

   -  操作系统的核心: Linux Kernel;

   -  用户与操作系统之间的接口: Shell, GUI;

   -  GUI 提供了一种图形化的用户接口,使用起来非常简便易学;

   -  Shell
      提供了一种命令行的接口, 接收用户的键盘输入, 并分析和执行输入字符串中的命令, 然后给用户返回执行结果,使用起来可能会复杂一些, 但是由于占用的资源少, 而且在操作熟练以后可能会提高工作效率, 而且具有批处理的功能, 因此在某些应用场合还非常流行;

   -  **Shell**
      作为一种用户接口, 它实际上是一个能够解释和分析用户键盘输入, 执行输入中的命令, 然后返回结果的一个解释程序(Interpreter,
      Linux下比较常用的是Bash);

   -  **脚本**:
      该解释程序不仅能够解释简单的命令, 而且可以解释一个具有特定语法结构的文件, 这种文件被称作\ ``脚本(Script)``;

**查看当前的Shell:**

.. code:: shell

   $ echo $SHELL
   $ ls -l /bin/bash
   $ ls -l /bin/zsh

2.搭建运行环境
--------------

   在一个Linux操作系统中, 有一个运行有Bash的命令行在等待键入命令, 这个命令行可以是图形界面下的
   ``终端(Terminal)``\ , 例如: Ubuntu 下的 Terminator.

2.1 搭建Bash运行环境
~~~~~~~~~~~~~~~~~~~~

.. code:: shell

   # 运行Bash环境
   $ chsh $USER -s /bin/bash

   $ su $USER

或者

.. code:: shell

   # 切换到Bash运行环境
   $ bash

   # 确认命令行中运行的是Bash
   $ echo $SHELL

2.2 运行Shell脚本
~~~~~~~~~~~~~~~~~

   -  正确使用 ``source`` 和 ``.``:

   -  仅使用 ``source`` 和 ``.``
      来执行环境配置等功能, 建议不要用于其他用途. 在 Shell
      中使用脚本时, 使用 ``bash your_script.sh``\ . 

   -  当使用 ``bash`` 的时候, 当前的 Shell
      会创建一个新的子进程执行脚本; 当使用 ``source`` 和 ``.``
      时, 当前的 Shell 会直接解释执行 ``your_script.sh`` 中的代码. 如果
      ``your_script.sh`` 中包含了类似 ``exit 0`` 这样的代码, 使用
      ``source`` 和 ``.`` 执行会导致当前 Shell 意外地退出

-  方法 1:确保执行的命令具有可执行权限: ``chmod +x``

.. code:: shell

   # 确保执行的命令具有可执行权限
   $ chmod +x ./shell/helloworld.sh

   # 运行脚本
   $ ./shell/helloworld.sh

-  方法 2: 直接把脚本作为Bash解释器的参数传入: ``bash``

.. code:: shell

   $ bash ./shell/helloworld.sh

-  方法 3: ``source``

.. code:: shell

   $ source ./shell/helloworld.sh

-  方法 4: ``.``

.. code:: shell

   $ . ./shell/helloworld.sh

3.Shell基本语法
---------------

3.1 Shell 的 Hello, World 程序
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

   -  永远的 Hello, World 程序;

   -  ``#!``\ 是一个约定的标记,它告诉系统这个脚本需要什么解释器来执行,即使用哪一种Shell;

   -  ``echo``\ 命令用于向窗口输出文本;

   -  分析 Shell 程序的基本结构;

**两种 Shell 程序:**

.. code:: shell

   #!/bin/bash -v
   # helloworld.sh

   echo "Hello, World"

或

.. code:: shell

   #!/bin/bash
   # helloworld.sh

   echo "Hello, World"

**分析 Shell 程序:**

上面两种程序结构对于两种不同的运行方式输出内容有差别: 

-  对于第一种程序结构

``chmod +x ./shell/helloworld.sh``,
``./shell/helloworld.sh``\ 运行结果如下(多打印了脚本文件本身的内容):

.. code:: 

   #!/bin/bash -v
   # helloworld.sh

   echo "Hello World"
   Hello World

``bash ./shell/helloworld.sh``\ 运行结果如下: 

.. code:: 

   Hello World

-  对于第二种程序结构

``chmod +x ./shell/helloworld.sh``,
``./shell/helloworld.sh``\ 和\ ``bash ./shell/helloworld.sh``\ 的运行结果一样, 如下:

.. code:: 

   Hello World

造成上面的结果的原因是在第一种程序结构中的第一行, 当直接运行该脚本文件时, 该行告诉操作系统使用\ ``#!``\ 符号之后的解释器以及相应的参数来解释该脚本文件, 通过分析第一行, 发现对应的解释器以及参数是
``/bin/bash -v`` , 而 ``-v`` 刚好就是要打印程序的源代码;
但是我们在用第二种方法时没有给 Bash
传递任何额外的参数, 因此, 它仅仅解释了脚本文件本身.



3.2 Shell 变量
~~~~~~~~~~~~~~

**定义变量:**

.. code:: shell

   $ var_name="wangzhefeng"

-  变量名和等号之间不能有空格, 任何赋值语句中,等号前后都不能有空格;

-  依然是字母,数字,下划线; 数字不能作为开头;

**使用变量:**

-  ``${var_name}`` or ``$var_name``

-  变量可以重新定义;

.. code:: shell

   $ var_name="wangzhefeng"
   $ var_name="zfwang"

   $ echo $var_name
   $ echo ${var_name}

**只读变量:**

.. code:: shell

   $ var_name="read only"
   $ readonly var_name

**删除变量:**

-  可以使用\ ``unset``\ 命令删除变量;

-  ``unset``\ 命令不能删除只读(readonly)变量;

.. code:: shell

   $ var_name="wanzhefeng"
   $ unset var_name

**变量类型:**

运行Shell时, 会同时存在 3 种变量:

1. **局部变量**:
   局部变量在脚本或命令中定义,仅在当前Shell实例中有效,其他Shell启动的程序不能访问局部变量;

2. **环境变量**:

   -  所有的程序, 包括 Shell 启动的程序都能访问环境变量;

   -  有些程序需要环境变量来保证其正常运行, 必要的时候 Shell
      脚本也可以定义环境变量;

3. **Shell 变量**: Shell 变量是由 Shell 程序设置的特殊变量. Shell
   变量中有一部分是环境变量,有一部分是局部变量,这些变量保证了 Shell
   的正常运行;



3.3 Shell 字符串
~~~~~~~~~~~~~~~~

   Shell字符串可以用单引号, 也可以用双引号, 也可以不用引号;

**单引号:**

   -  单引号字符串的限制:

   -  单引号里的任何字符都会原样输出, 单引号字符串中的变量是无效的;

   -  单引号字符串中不能出现单独一个的单引号(对单引号使用转义符也不行),但可以成对出现,作为字符串拼接的使用;

.. code:: shell

   $ str='this is a string'

**双引号:**

   -  双引号里可以有变量;

   -  双引号里可以出现转义字符;

.. code:: shell

   $ var="wangzhefeng"
   $ str="Hello, I know you are \"$var\"! \n"
   $ echo -e $str

输出:

.. code:: 

   Hello, I know you are "wangzhefeng"!

**拼接字符串:**

-  使用双引号拼接

.. code:: shell

   $ var1="wangzhefeng"
   $ greeting1="hello, "$var1" !"
   $ greeting2="hello, ${var1} !"
   $ echo $greeting1 $greeting2

输出:

.. code:: 

   hello, wangzhefeng ! hello, wangzhefeng !

-  使用单引号拼接

.. code:: shell

   $ var2="wangzhefeng"
   $ greet1='hello, '$var2' !' # 单引号可以成对出现,作为字符串拼接的使用;
   $ greet2='hello, ${var2} !' # 单引号中的变量无效
   $ echo greet1 greet2

输出:

.. code:: 

   hello, wangzhefeng ! hello, ${var2} !

**获取字符串长度:**

.. code:: shell

   string="abcd"
   echo ${#string}

**提取字符串:**

.. code:: shell

   string="wangzhefeng is a man!"
   echo ${string:0:11}

**查找子字符串:**

.. code:: shell

   # 查找字符串`i`或`o`的位置(哪个字母先出现就计算哪个)
   string="wangzhefeng is a man!"
   echo `expr index "$string" io`

输出:

.. code:: 

   13



3.3.1 字符串操作
^^^^^^^^^^^^^^^^

-  字符串的属性

   -  字符串的类型

   -  字符串的长度

-  字符串的显示

-  字符串的存储

-  字符串常规操作

   -  取子串

   -  查询子串

   -  子串替换

   -  插入子串

   -  删除子串

   -  子串比较

   -  子串排序

   -  子串进制转换

   -  子串编码转换

-  字符串操作进阶

   -  正则表达式

   -  处理格式化的文本

字符串的属性: 

字符串的类型: 

字符可能是数字、字符、空格、其他特殊字符, 而字符串有可能是它们中的一种或多种的组合, 在组合之后还可能形成具有特定意义的字符串, 诸如邮件地址、URL
地址等. 

字符串的长度: 

计算某个字符串的长度

.. code:: shell

   var="get the length of me"
   echo ${var}

   # method 1
   echo ${#var}

   # method 2
   expr length "$var"

   # method 3
   echo $var awk '{printf("%d\n", length($0));}'

   # method 4 
   echo -n $var | wc -c

计算某些指定字符或者字符组合的个数

.. code:: shell

   echo $var | tr -cd g | wc -c
   echo -n $var | sed -e 's/[^g]//g' | wc -c
   echo -n $var | sed -e 's/[^gt]//g' | wc -c

统计单词个数

.. code:: shell

   echo $var | wc -w
   echo "$var" | tr " " "\n" | grep get | uniq -c 
   echo "$var" | tr " " "\n" | grep get | wc -l



3.4 Shell 注释
~~~~~~~~~~~~~~

   -  单行注释: Shell 注释以\ ``#``\ 开头;

   -  多行注释:

   -  ``:<<EOF comment EOF``

   -  ``:<<' comment '``

   -  ``:<<! comment !``

   -  ``function fun_comment(){ comment }``

.. code:: shell

   :<<EOF 
   注释
   EOF

.. code:: shell

   :<<'
   注释
   '

.. code:: shell

   :<<!
   注释
   !

.. code:: shell

   function func_comment() {
   	注释
   }



3.3 Shell传递参数
~~~~~~~~~~~~~~~~~

在执行Shell脚本时, 向脚本传递参数, 脚本内获取参数的格式为: ``$n``;

+-----------------------------------+-----------------------------------------------------------------+
| 参数处理格式                        | 说明                                                            |
+===================================+=================================================================+
| ``$#``                            | 传递到脚本的参数个数                                               |
+-----------------------------------+-----------------------------------------------------------------+
| ``$*``                            | 以一个单字符串显示所有向脚本传递的参数. 如 ``$*`` 用 ``""`` 括起来的情况 |
+-----------------------------------+-----------------------------------------------------------------+
| ``$@``                            | 与 ``$*`` 相同,但是使用时加引号,并在引号中返回每个参数.                 |
+-----------------------------------+-----------------------------------------------------------------+
| ``$$``                            | 脚本运行的当前进程ID号                                             |
+-----------------------------------+-----------------------------------------------------------------+
| ``$!``                            | 后台运行的最后一个进程的ID号                                        |
+-----------------------------------+-----------------------------------------------------------------+
| ``$-``                            | 显示Shell使用的当前选项,与set命令功能相同.                           |
+-----------------------------------+-----------------------------------------------------------------+
| ``$?``                            | 显示最后命令退出的状态,0表示没有错误,其他任何值表示又错误.               |
+-----------------------------------+-----------------------------------------------------------------+


.. code:: shell

   #!/bin/bash
   # params.sh

   echo "Shell 传递参数!"
   echo "第一个参数: $1"
   echo "第二个参数: $2"
   echo "第三个参数: $3"
   echo "传递的参数个数: $#"
   echo "传递的参数以一个字符串显示: $*"
   echo "传递的参数以多个字符串显示: $@"
   echo "Shell使用的当前选项: $-"
   echo "Shell最后命令的退出状态: $?"
   echo "脚本运行的当前进程ID号: $$"
   echo "脚本运行的最后一个进程ID号: $!"

   echo "=======\$*的示例========"
   for i in "$*"
   do 
   	echo $i
   done

   echo "=======\$@的示例========"
   for i in "$@"
   do 
   	echo $i
   done

.. code:: 

   Shell 传递参数!
   第一个参数: param_1
   第二个参数: param_2
   第三个参数: param_3
   传递的参数个数: 3
   传递的参数以一个字符串显示: param_1 param_2 param_3
   传递的参数以多个字符串显示: param_1 param_2 param_3
   Shell使用的当前选项: hB
   Shell最后命令的退出状态: 0
   脚本运行的当前进程ID号: 21869
   脚本运行的最后一个进程ID号: 
   =======$*的示例========
   param_1 param_2 param_3
   =======$@的示例========
   param_1
   param_2
   param_3



3.4 Shell数组
~~~~~~~~~~~~~

   bash支持一维数组,不支持多维数组,并且没有限定数组的大小;
   数组元素的下表由0开始编号,获取数组中的元素要利用下标,
   下标可以是整数或算术表达式,其值应大于或等于0;

**定义数组:**

   在Shell中,用圆括号来表示数组,数组元素用"空格"分割开;

.. code:: shell

   arrayName=(elem_1 elem_2 ... elem_n)

.. code:: shell

   arrayName=(
   elem_1
   elem_2
   ...
   elem_n
   )

.. code:: shell

   arrayName[0]=elem_1
   arrayName[1]=elem_2
   arrayName[n]=elem_n

**读取数组:**

格式:

.. code:: shell

   ${数组名[下标]}

示例:

-  获取数组中第n个元素:

.. code:: shell

   value_n=${arrayName[n]}

-  获取数组中的所有元素:

.. code:: shell

   echo ${arrayName[@]}

**获取数组的长度:**

.. code:: shell

   # 取得数组元素的个数
   length=${#arrayName[@]}

   # or
   length=${#arrayName[*]}

   # 取得数组单个元素的长度
   length_n=${#arrayName[n]}



3.5 Shell运算符
~~~~~~~~~~~~~~~

-  Shell运算符: 

   -  算术运算符

   -  关系运算符

   -  布尔运算符

   -  字符串运算符

   -  文件测试运算符

-  原生bash不支持简单的数学运算, 但是可以通过其他命令实现, 例如\ ``awk``,
   ``expr``\ ; 

   -  ``expr``\ 是一款表达式计算工具, 使用它能完成表达式的求值操作; 

      -  表达式和运算符之间要有空格

      -  完整的表达式要被反引号包住, 而不是单引号

.. code:: shell

   #!/bin/bash

   val=`expr 2 + 2`
   echo "两个之和为: $val"

**算术运算符: **

+--------+--------------------------+---------------------------+
| 参数   | 说明                      | 举例                       |
+========+==========================+===========================+
| ``+``  | 加                       | expr :math:`a + b`        |
+--------+--------------------------+---------------------------+
| ``-``  | 减                       | expr :math:`a - b`        |
+--------+--------------------------+---------------------------+
| ``*``  | 乘                       | expr :math:`a * b`        |
+--------+--------------------------+---------------------------+
| ``/``  | 除                       | expr :math:`b \ a`        |
+--------+--------------------------+---------------------------+
| ``%``  | 求余                      | expr :math:`b % a`        |
+--------+--------------------------+---------------------------+
| ``=``  | 赋值                      | :math:`a=b`               |
+--------+--------------------------+---------------------------+
| ``=``  | 相等, 用于比较两个数字       | [ :math:`a == b`]         |
+--------+--------------------------+---------------------------+
| ``!=`` | 不相等, 用于比较两个数字     | [ :math:`a != b` ]        |
+--------+--------------------------+---------------------------+

**关系运算符: **

+---------+-------------+-------------------+
| 参数    | 说明        | 举例              |
+=========+=============+===================+
| ``-eq`` | 等于,=      | ``[ $a -eq $b ]`` |
+---------+-------------+-------------------+
| ``ne``  | 不等于,!=   | ``[ $a -ne $b ]`` |
+---------+-------------+-------------------+
| ``-gt`` | 大于,>      | ``[ $a -gt $b ]`` |
+---------+-------------+-------------------+
| ``-ge`` | 大于等于,>= | ``[ $a -ge $b ]`` |
+---------+-------------+-------------------+
| ``-lt`` | 小于,<      | ``[ $a -lt $b ]`` |
+---------+-------------+-------------------+
| ``-le`` | 小于等于,<= | ``[ $a -le $b ]`` |
+---------+-------------+-------------------+

**布尔运算符: **

+--------+------+---------------------------------+
| 参数   | 说明 | 举例                            |
+========+======+=================================+
| ``!``  | 非   | ``[ !false ]``                  |
+--------+------+---------------------------------+
| ``-o`` | 或   | ``[ $a -lt 20 -o $b -gt 100 ]`` |
+--------+------+---------------------------------+
| ``-a`` | 与   | ``[ $a -lt 20 -a $b -gt 100 ]`` |
+--------+------+---------------------------------+

**逻辑运算符: **

+----------+---------+------------------------------------+
| 参数     | 说明    | 举例                               |
+==========+=========+====================================+
| ``&&``   | 逻辑AND | ``[[ $a -lt 100 && $b -gt 100 ]]`` |
+----------+---------+------------------------------------+
| 两个竖杆 | 逻辑OR  |                                    |
+----------+---------+------------------------------------+

**字符串运算符: **

+-----------------------+-----------------------+-----------------------+
| 运算符                | 说明                  | 举例                  |
+=======================+=======================+=======================+
| =                     | 检测两个字符串是否相等, 相等返回 | [ :math:`a = `\ b ] |
|                       |                       | 返回 false.           |
|                       | true.                 |                       |
+-----------------------+-----------------------+-----------------------+
| !=                    | 检测两个字符串是否相等, 不相等返回 | [ :math:`a != `\ b ] |
|                       |                       | 返回 true.            |
|                       | true.                 |                       |
+-----------------------+-----------------------+-----------------------+
| -z                    | 检测字符串长度是否为0, 为0返回 | [ -z $a ] 返回 |
|                       |                       | false.                |
|                       | true.                 |                       |
+-----------------------+-----------------------+-----------------------+
| -n                    | 检测字符串长度是否为0, 不为0返回 | [ -n "$a" ] 返回 |
|                       |                       | true.                 |
|                       | true.                 |                       |
+-----------------------+-----------------------+-----------------------+
| $                     | 检测字符串是否为空, 不为空返回 | [ $a ] 返回 true.  |
|                       |                       |                       |
|                       | true.                 |                       |
+-----------------------+-----------------------+-----------------------+

**文件测试运算符: **

+-----------------------+-----------------------+-----------------------+
| 操作符                | 说明                  | 举例                  |
+=======================+=======================+=======================+
| ``-b file``           | 检测文件是否是块设备文件, 如果是, 则返回 | [ -b $file ] 返回 |
|                       |                       | false.                |
|                       | true.                 |                       |
+-----------------------+-----------------------+-----------------------+
| ``-c file``           | 检测文件是否是字符设备文件, 如果是, 则返回 | [ -c $file ] 返回 |
|                       |                       | false.                |
|                       | true.                 |                       |
+-----------------------+-----------------------+-----------------------+
| ``-d file``           | 检测文件是否是目录, 如果是, 则返回 | [ -d $file ] 返回 |
|                       |                       | false.                |
|                       | true.                 |                       |
+-----------------------+-----------------------+-----------------------+
| ``-f file``           | 检测文件是否是普通文件(既不是目录, 也不是 | [ -f $file ] 返回 |
|                       | 设备文件), 如果是, 则返回 | true.            |
|                       |                       |                       |
|                       | true.                 |                       |
+-----------------------+-----------------------+-----------------------+
| ``-g file``           | 检测文件是否设置了    | [ -g $file ] 返回     |
|                       | SGID                  | false.                |
|                       | 位, 如果是, 则返回    |                       |
|                       | true.                 |                       |
+-----------------------+-----------------------+-----------------------+
| ``-k file``           | 检测文件是否设置了粘着位(Sticky | [ -k $file ] 返回 |
|                       |                       | false.                |
|                       | Bit), 如果是, 则返回  |                       |
|                       | true.                 |                       |
+-----------------------+-----------------------+-----------------------+
| ``-p file``           | 检测文件是否是有名管道, 如果是, 则返回 | [ -p $file ] 返回 |
|                       |                       | false.                |
|                       | true.                 |                       |
+-----------------------+-----------------------+-----------------------+
| ``-u file``           | 检测文件是否设置了    | [ -u $file ] 返回     |
|                       | SUID                  | false.                |
|                       | 位, 如果是, 则返回    |                       |
|                       | true.                 |                       |
+-----------------------+-----------------------+-----------------------+
| ``-r file``           | 检测文件是否可读, 如果是, 则返回 | [ -r $file ] 返回 |
|                       |                       | true.                 |
|                       | true.                 |                       |
+-----------------------+-----------------------+-----------------------+
| ``-w file``           | 检测文件是否可写, 如果是, 则返回 | [ -w $file ] 返回 |
|                       |                       | true.                 |
|                       | true.                 |                       |
+-----------------------+-----------------------+-----------------------+
| ``-x file``           | 检测文件是否可执行, 如果是, 则返回 | [ -x $file ] 返回 |
|                       |                       | true.                 |
|                       | true.                 |                       |
+-----------------------+-----------------------+-----------------------+
| ``-s file``           | 检测文件是否为空(文件大小是否大于0), 不 | [ -s $file ] 返回 |
|                       | 为空返回              | true.                 |
|                       | true.                 |                       |
+-----------------------+-----------------------+-----------------------+
| ``-e file``           | 检测文件(包括目录)是否存在, 如果是, 则返 | [ -e $file ] 返回 |
|                       | 回                    | true.                 |
|                       | true.                 |                       |
+-----------------------+-----------------------+-----------------------+



3.6 Shell命令之echo, printf, test
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~



3.6.1 echo
^^^^^^^^^^

   echo用于字符串的输出, 可以使用echo实现复杂的输出格式控制;

**显示普通字符串:**

.. code:: shell

   echo "It is a test."

   echo It is a test.

**显示转义字符:**

.. code:: shell

   echo "\"It is a test\""

   echo \"It is a test\"

**显示变量: **

.. code:: shell

   #!/bin/bash

   read name
   echo "$name It is a test"

**显示换行: **

.. code:: shell

   # -e: 开启转义
   echo -e "OK! \n"
   echo "It is a test"

**显示不换行: **

.. code:: shell

   #!/bin/bash

   # -e: 开启转义, \c不换行
   echo -e "OK! \c"

**显示结果定向至文件: **

.. code:: shell

   echo "It is a test" > myfile

**原样输出字符串, 不进行转义或取变量(用单引号)**

.. code:: shell

   echo '$name\'

**显示命令执行结果: **

.. code:: shell

   echo `date`



3.6.2 printf
^^^^^^^^^^^^

   -  Shell中的printf命令模仿C程序库中的printf(); 

   -  printf 由POSIX标准定义, 因此使用printf的脚本比使用echo移植性要好; 

   -  printf
      使用引用文本或空格分隔的参数, 可以在printf中使用格式化字符串, 还可以指定字符串的宽度、左右对齐等方式; 

   -  printf 默认不会像echo自动添加换行符, 可以手动进行添加; 

   -  printf 格式化替代符: 

   -  ``%-ns``\ : 左对齐, 宽度为n, 字符

   -  ``%-nc``\ : 左对齐, 宽度为n, 

   -  ``%-nd``\ : 左对齐, 宽度为n, 整数

   -  %-n.mf: 左对齐, 宽度n, 保留m为小数, 小数

   -  printf 转义字符: 

   -  ``\a``\ : 警告字符, 通常为ASCII的BEL字符

   -  ``\b``\ : 后退

   -  ``\c``\ : 抑制(不显示)输出结果中任何结尾的换行字符(只在%b格式指示符控制下的参数字符串中有效), 而且, 任何留在参数里的字符、任何接下来的参数以及任何留在格式字符串中的字符, 都被忽略

   -  ``\f``\ : 换页

   -  ``\n``\ : 换行

   -  ``\r``\ : 回车

   -  ``\t``\ : 水平制表符

   -  ``\v``\ : 垂直制表符

   -  ``\\``\ : \\

   -  ``\ddd``\ : 表示1到3位数8进制的字符

   -  ``0ddd``\ : 表示1到3位8进制字符

**格式: **

.. code:: shell

   printf format-string args

**示例: **

手动添加换行符: 

.. code:: shell

   $ echo "wangzhefeng"

   $ printf "wangzhefeng\n"

格式化打印字符串: 

.. code:: shell

   #!/bin/bash

   printf "%-10s %-8s %-4s\n" 姓名 性别 体重kg
   printf "%-10s %-8s %-4.2s\n" name1 gender1 66.1234
   printf "%-10s %-8s %-4.2s\n" name2 gender2 77.2234
   printf "%-10s %-8s %-4.2s\n" name3 gender3 88.3234

   printf "%d %s\n" 1 "abc"
   printf '%d %s\n' 1 "abc"
   printf %s abcdef

   # 格式只指定了一个参数, 但多出的参数仍然会按照该格式输出, format-string 被重用
   printf %s abc def
   printf "%s\n" abc def
   printf "%s %s %s\n" a b c d e f g h i j k

   # 如果没有 arguments, 那么 %s 用NULL代替, %d 用 0 代替
   printf "%s and %d \n"

转义字符: 

.. code:: shell

   printf "a string, no processing:<%s>\n" "A\nB"
   printf "a string, no processing:<%b>\n" "A\nB"



3.6.3 test
^^^^^^^^^^

   Shell中的test命令用于检查某个条件是否成立, 可以进行多种测试: 

   -  数值测试

   -  字符测试

   -  文件测试



3.6.3.1 数值测试
''''''''''''''''

-  使用\ ``[]``\ 执行基本的算数运算; 

+---------+-------------+
| 参数    | 说明        |
+=========+=============+
| ``-eq`` | 等于,=      |
+---------+-------------+
| ``-ne`` | 不等于,!=   |
+---------+-------------+
| ``-gt`` | 大于,>      |
+---------+-------------+
| ``-ge`` | 大于等于,>= |
+---------+-------------+
| ``-lt`` | 小于,<      |
+---------+-------------+
| ``-le`` | 小于等于,<= |
+---------+-------------+

**示例: **

.. code:: shell

   num1=100
   num2=100

   if test $[num1] -eq $[num2]
   then
   	echo "两个数字相等"
   else
   	echo "两个数字不相等"
   if

.. code:: shell

   #!/bin/bash

   a=5
   b=6

   result=$[a+b]
   echo "result 是: $result"



3.6.3.2 字符测试
''''''''''''''''

+---------------+--------------------------+
| 参数          | 说明                     |
+===============+==========================+
| ``=``         | 等于                     |
+---------------+--------------------------+
| ``!=``        | 不等于                   |
+---------------+--------------------------+
| ``-z string`` | 字符串的长度为零则为真   |
+---------------+--------------------------+
| ``-n string`` | 字符串的长度不为零则为真 |
+---------------+--------------------------+

**示例: **

.. code:: shell

   str1="wangzhefeng"
   str2="tinker"

   if test $str1 = $str2
   then
   	echo "两个字符串相等"
   else
   	echo "两个字符串不相等"
   fi



3.6.3.3 文件测试
''''''''''''''''

+-----------------+--------------------------------------+
| 参数            | 说明                                 |
+=================+======================================+
| ``-e filename`` | 如果文件存在则为真                   |
+-----------------+--------------------------------------+
| ``-r filename`` | 如果文件存在且可读则为真             |
+-----------------+--------------------------------------+
| ``-w filename`` | 如果文件存在且可写则为真             |
+-----------------+--------------------------------------+
| ``-x filename`` | 如果文件存在且可执行则为真           |
+-----------------+--------------------------------------+
| ``-s filename`` | 如果文件存在且至少有一个字符则为真   |
+-----------------+--------------------------------------+
| ``-d filename`` | 如果文件存在且为目录则为真           |
+-----------------+--------------------------------------+
| ``-f filename`` | 如果文件存在且为普通文件则为真       |
+-----------------+--------------------------------------+
| ``-c filename`` | 如果文件存在且为字符型特殊文件则为真 |
+-----------------+--------------------------------------+
| ``-b filename`` | 如果文件存在且为特殊文件则为真       |
+-----------------+--------------------------------------+

**示例: **

.. code:: shell

   cd /bin

   if test -e ./bash
   then
   	echo "文件已存在"
   else
   	echo "文件不存在"
   fi



3.6.3.4 逻辑操作符
''''''''''''''''''

-  Shell提供了与, 或, 非逻辑操作符用于将测试条件连接起来, 优先级为:
   ``！`` > ``-a`` > ``-o``\ ; 

+--------+--------+
| 参数   | 说明   |
+========+========+
| ``-a`` | 与,and |
+--------+--------+
| ``-o`` | 或,or  |
+--------+--------+
| ``!``  | 非,not |
+--------+--------+

.. code:: shell

   cd /bin

   if test -e ./notfile -o -e ./bash
   then 
   	echo "至少有一个文件存在"
   else
   	echo "两个文件都不存在"
   fi



3.7 Shell流程控制
~~~~~~~~~~~~~~~~~



3.7.1 if...else
^^^^^^^^^^^^^^^

**if语句:**

.. code:: shell

   if [test] condition
   then 
   	command1
   	command2
   	...
   	commandN
   fi

or

.. code:: shell

   if condition then commands fi

**if...else...语句:**

.. code:: shell

   if [test] condition
   then 
   	command1
   	command2
   	...
   	commandN
   else
   	commandM
   fi

**if...elif...else...语句:**

.. code:: shell

   if [test] condition1
   then 
   	command1
   elif [test] condition2
   then 
   	command2
   else
   	command3
   fi



3.7.2 case语句
^^^^^^^^^^^^^^

   -  Shell case 语句为多选语句, 可以用case语句匹配一个值与一个模式,
      如果匹配成功,执行相匹配的命令;

   -  case 取值后面必须为\ ``in``, 每一模式必须以有括号结束;

   -  取值可以为变量常数

   -  匹配发现取值符合某一模式后, 期间所有命令开始执行, 直至\ ``;;``

   -  取值将检测匹配的每一模式,一旦模式匹配,则执行完匹配模式相应命令后不再继续其他模式;如果无一匹配模式,使用\ ``*``\ 捕获该值,再执行后面的命令

格式:

.. code:: shell

   case value in
   mode1)
   	command1
   	command2
   	...
   	commandN
   	;;
   mode2)
   	command1
   	command2
   	...
   	commandN
   	;;
   esac

示例:

.. code:: shell

   echo "输入1到4之间的数字:"
   echo "你输入的数字为:"
   read aNum

   case $aNum in
   	1) echo "你选择了 1"
   	;;
   	2) echo "你选择了 2"
   	;;
   	3) echo "你选择了 3"
   	;;
   	4) echo "你选择了 4"
   	;;
   	*) echo "你没有输入1到4之间的数字"
   	;;
   esac



3.7.3 for循环
^^^^^^^^^^^^^

**普通格式:**

.. code:: shell

   for var in item1 item2 ... itemN
   do 
   	command1
   	command2
   	...
   	commandN
   done

or

.. code:: shell

   for var in item1 item2 ... itemN do command1 command2 ... commandN done

**无限循环格式:**

.. code:: shell

   for (( ; ;))



3.7.4 while语句
^^^^^^^^^^^^^^^

**普通格式:**

.. code:: shell

   while condition
   do 
   	command
   done

**无限循环格式:**

.. code:: shell

   while :
   do
   	command
   done

or

.. code:: shell

   while true
   do
   	command
   done

示例:

.. code:: shell

   #!/bin/bash

   int=1
   while (($int<=5))
   do 
   	echo $int
   	let "int++"
   done



3.7.5 until循环
^^^^^^^^^^^^^^^

   until循环执行一系列命令直至条件为true时停止,
   condition一般为条件表达式,如果返回值为false,则继续执行循环体内的语句,否则跳出循环;
   until循环与while循环在处理方式上刚好相反;
   一般while循环优于until循环,但在某些时候until循环更加有用;

格式:

.. code:: shell

   until condition
   do 
   	command
   done

示例:

.. code:: shell

   #!/bin/bash

   a=0
   until [!$a -lt 10]
   do 
   	echo $a
   	a=`expr $a + 1`
   done



3.7.6 跳出循环
^^^^^^^^^^^^^^

   -  break

   -  continue



3.7.6.1 break
'''''''''''''

   break命令允许跳出所有循环(终止执行后面的所有循环);

示例:

.. code:: shell

   #!/bin/bash

   while :
   do
   	echo -n "输入1到5之间的数字:"
   	read aNum
   	case $aNum in
   		1|2|3|4|5) echo "你输入的数字是: $aNum"
   		;;
   		*) echo "你输入的数字不是1到5之间的; "
   			break
   		;;
   	esac
   done



3.7.6.2 continue
''''''''''''''''

   continue不会跳出所有循环,仅仅跳出当前循环;

示例:

.. code:: shell

   #!/bin/bash

   while :
   do 
   	echo -n "输入1到5之间的数字:"
   	read aNum
   	case $aNum in
   		1|2|3|4|5) echo "你输入的数字为: $aNum !"
   		;;
   		*) echo "你输入的数字不是1到5之间的; "
   			continue
   			echo "游戏结束"
   		;;
   	esac
   done



3.8 Shell函数
~~~~~~~~~~~~~

   -  定义形式:

   -  ``function fun_name()``

   -  ``fun_name()``

   -  参数返回:

   -  ``return``: 可以显式添加

   -  不显式加\ ``return``\ 则以最后一条命令运行结果作为返回值;

   -  函数返回值在调用之后通过\ ``$?``\ 来获得;

   -  函数参数

   -  调用函数时可以向其传递参数;

   -  在函数体内部,通过\ ``$n``\ 的形式来获取参数的值,
      当n>10时,需要使用\ ``${n}``\ 来获取参数;

   -  其他特殊字符处理参数:

      -  ``$#``:传递到脚本的参数个数

      -  ``$*``:以一个单字符串显示所有向脚本传递的参数

      -  ``$$``:脚本运行的当前进程ID号

      -  ``$!``:后台运行的最后一个进程ID号

      -  ``$@``:与``$*``\ 相同,但是使用引号,并在引号中返回每个参数

      -  ``$-``:显示Shell使用的当前选项,与``set``\ 命令功能相同

      -  ``$?``:显示最后命令的退出状态,0表示没有错误,其他值表示有错误

**函数定义形式:**

-  函数定义:

.. code:: shell

   function fun_name(){
   	action;
   	
   	return
   }

.. code:: shell

   fun_name(){
   	action;
   	return
   }

-  函数调用:

.. code:: shell

   fun_name param1, param2, param3, ...

**函数定义示例:**

-  Example 1:

.. code:: shell

   #!/bin/bash
   # author: zfwang
   # file: demo.sh

   demoFun(){
   	echo "This is my first Shell function!"
   }

   # 函数调用
   echo "-----函数开始执行-----"
   demoFun
   echo "-----函数执行完毕-----"

-  Example 2:

.. code:: shell

   #!/bin/bash
   # author: zfwang
   # file: funWithReturn.sh

   funWithReturn(){
   	echo "这个函数会对输入的两个数字进行相加运算..."
   	echo "输入第一个数字: "
   	read aNum
   	echo "输入第二个数字: "
   	read anothreNum
   	echo "两个数字分别位 $aNum 和 $anotherNum !"
   	return ${$aNum+$anotherNum}
   }


   # 函数调用
   funWithReturn
   echo "输入的两个数字之和为: $? !"

-  Example 3:

.. code:: shell

   #!/bin/bash
   # author: zfwang
   # file: funWithParam.sh

   funWithParam(){
   	echo "第一个参数为 $1 !"
       echo "第二个参数为 $2 !"
       echo "第十个参数为 $10 !"
       echo "第十个参数为 ${10} !"
       echo "第十一个参数为 ${11} !"
       echo "参数总数有 $# 个!"
       echo "作为一个字符串输出所有参数 $* !"
   }


   # 函数调用
   funWithParam 1 2 3 4 5 6 7 8 9 34 73



3.9 Shell输入输出重定向
~~~~~~~~~~~~~~~~~~~~~~~

   -  一般情况下, 每个Unix/Linux命令运行时都会打开三个文件:

   -  标准输入文件(stdin):stdin的文件描述符位0, Unix程序默认从stdin读取数据,stdin默认为终端;

   -  标准输出文件(stdout):stdout的文件描述符位1, Unix程序默认向stdout输出数据,stdout默认为终端;

   -  标准错误文件(stderr):stderr的文件描述符位2, Unix程序默认向stderr流中写入错误信息,stderr默认为终端;

**重定向命令:**

+----------------------------+------------------------------------------------+
| 命令                       | 说明                                           |
+============================+================================================+
| command ``>`` file         | 将输出重定向到file                             |
+----------------------------+------------------------------------------------+
| command ``>>`` file        | 将输出以追加的方式重定向到file                 |
+----------------------------+------------------------------------------------+
| command ``2 >`` file       | 将stderr重定向到file                           |
+----------------------------+------------------------------------------------+
| command ``2 >>`` file      | 将stderr以追加的方式重定向到file               |
+----------------------------+------------------------------------------------+
| command ``<`` file         | 将输入重定向到file                             |
+----------------------------+------------------------------------------------+
| command < infile > outfile | 对stdin和stdout同时重定向                      |
+----------------------------+------------------------------------------------+
| n >& m                     | 将输出文件m和n合并                             |
+----------------------------+------------------------------------------------+
| n <& m                     | 将输入文件m和n合并                             |
+----------------------------+------------------------------------------------+
| command > file 2>&1        | 将stdout和stderr合并后重定向到file             |
+----------------------------+------------------------------------------------+
| command >> file 2>&1       | 将stdout和stderr合并后以追加的方式重定向到file |
+----------------------------+------------------------------------------------+
| n ``>`` file               | 将文件描述符为n的文件重定向到file              |
+----------------------------+------------------------------------------------+
| n ``>>`` file              | 将文件描述符为n的文件以追加的方式重定向到file  |
+----------------------------+------------------------------------------------+
| ``<<tag``                  | 将开始标记tag和结束标记tag之间的内容作为输入   |
+----------------------------+------------------------------------------------+



3.9.1 输出重定向
^^^^^^^^^^^^^^^^

**语法:**

.. code:: shell

   command > file
   command >> file

**示例:**

.. code:: shell

   # 将命令的完整的输出重定向到users文件中
   who > users

   # 查看users文件中的内容
   cat users

.. code:: shell

   # 将输出重定向覆盖users文件中的内容
   echo "This is a test command by wangzhefeng" > users

   # 查看users文件中的内容
   cat users

.. code:: shell

   # 将输出重定向追加到users文件的末尾
   echo "This is another test command by wangzhefeng" >> users



3.9.2 输入重定向
^^^^^^^^^^^^^^^^

   本来需要从键盘获取输入的命令会转移到文件读取内容;

**语法:**

.. code:: shell

   command < file

**示例:**

.. code:: shell

   # 统计users文件中的行数(会输出文件名users)
   wc -l users

   # 统计users文件中的行数(不会输出文件名users)
   wc -l < users

.. code:: shell

   # 同时替换输入和输出,执行command,从文件infile读取内容, 然后将输出写入到outfile中
   command < infile > outfile



3.9.3 Here Document
^^^^^^^^^^^^^^^^^^^

   Here Document将输入重定向到一个交互式Shell脚本或程序;

**基本形式:**

.. code:: shell

   # 将两个delimiter之间的内容(document)作为输入传递给command
   command << delimiter
   	document
   delimiter

**示例:**

.. code:: shell

   # 通过wc -l 命令计算Here Document的行数
   wc -l << EOF
   	test line 1
   	test line 2
   	test line 3
   EOF

.. code:: shell

   #!/bin/bash
   # hereDocument.sh

   cat << EOF
   	test line 1
   	test line 2
   EOF



3.9.4 /dev/null 文件
^^^^^^^^^^^^^^^^^^^^

   -  如果希望执行某个命令, 但不希望在终端显示输出结果, 可以将输出重定向到\ ``/dev/null``;

   -  ``/dev/null``\ 是一个特殊的文件, 写入到它的内容都会被丢弃;如果从该文件读取内容, 什么也读不到; 

**格式:**

.. code:: shell

   command > /dev/null

**示例:**

.. code:: shell

   # 屏蔽stdout和stderr
   command > /dev/null 2>&1



3.10 Shell文件包含
~~~~~~~~~~~~~~~~~~

   Shell可以包含外部脚本,可以很方便的封装一些公用的代码作为一个独立的文件;

格式:

.. code:: shell

   . fileName

   # or

   source fileName

示例:

创建两个Shell脚本文件:\ ``test1.sh``,
``test2.sh``.并在\ ``test2.sh``\ 中调用\ ``test1.sh``

.. code:: shell

   #!/bin/bash
   # file: test1.sh

   string1="wangzhefeng in test1.sh"

.. code:: shell

   #!/bin/bash
   # file: test2.sh

   # 使用`.`来引用test1.sh文件
   . ./shell/test1.sh

   # 使用`source`来引用test1.sh文件
   source ./shell/test1.sh

   echo "在test1.sh中的字符串为: $string1"

执行test2.sh脚本:

.. code:: shell

   chmod +x ./shell/test2.sh
   ./shell/test2.sh

输出

.. code:: 

   在test1.sh中的字符串为: wangzhefeng in test.sh



4.Shell程序设计过程
-------------------

Shell
语言作为解释型语言, 它的程序设计过程跟编译型语言有些区别, 其基本过程如下: 

-  设计算法

-  用 Shell 编写脚本程序实现算法

-  直接运行脚本程序

可见它没有编译型语言的"麻烦的"编译和链接过程, 不过正是因为这样, 它出错时调试起来不
是很方便, 因为语法错误和逻辑错误都在运行时出现.



5.Shell数值运算
---------------

   -  Shell编程中的基本数值运算:

   -  数值(整数,浮点数)间的加,减,乘,除,求余,求幂;

   -  产生指定范围的随机数;

   -  产生指定范围的数列;

   -  Shell本身可以做整数运算,复杂一些的运算要通过外部命令实现,比如:\ ``expr``,
      ``bc``,
      ``awk``\ 等;另外,可以通过\ ``RANDOM``\ 环境变量产生一个从0到32767的随机数;

   -  ``awk``\ 可以通过\ ``rand()``\ 函数产生随机数;

   -  ``seq``\ 命令可以用来产生一个数列;



5.1整数运算
~~~~~~~~~~~

**对某个数加1:**

.. code:: shell

   # test

**从1加到某个数:**

**求余数:**

**求幂:**



6.Shell 布尔运算
----------------



常规的布尔运算
~~~~~~~~~~~~~~



在 Shell 下进行逻辑运算
^^^^^^^^^^^^^^^^^^^^^^^

1.\ ``true`` 和 ``false``

.. code:: shell

   $ if true;then echo "YES"; else echo "NO"; fi
   $ if false;then echo "YES"; else echo "NO"; fi

2.与运算、或运算、非运算

.. code:: shell

   if true && true;then echo "YES"; else echo "NO"; fi
   if true && false;then echo "YES"; else echo "NO"; fi
   if false && false;then echo "YES"; else echo "NO"; fi
   if false && true;then echo "YES"; else echo "NO"; fi

.. code:: shell

   if true || true;then echo "YES"; else echo "NO"; fi
   if true || false;then echo "YES"; else echo "NO"; fi
   if false || false;then echo "YES"; else echo "NO"; fi
   if false || true;then echo "YES"; else echo "NO"; fi

.. code:: shell

   if ! false;then echo "YES"; else echo "NO"; fi
   if ! true;then echo "YES"; else echo "NO"; fi



7.文件操作
----------



文件的各种属性
~~~~~~~~~~~~~~

通过文件的结构体来看看文件到底有哪些属性: 

.. code:: 

   struct stat {
       dev_t st_dev;              /* 设备   */
       ino_t st_ino;     		   /* 节点   */
       mode_t st_mode;   		   /* 模式   */
       nlink_t st_nlink; 		   /* 硬连接 */
       uid_t st_uid;     		   /* 用户ID */
       gid_t st_gid;     		   /* 组ID   */
       dev_t st_rdev;             /* 设备类型 */
       off_t st_off;              /* 文件字节数 */
       unsigned long  st_blksize; /* 块大小 */
       unsigned long st_blocks;   /* 块数   */
       time_t st_atime;           /* 最后一次访问时间 */
       time_t st_mtime;           /* 最后一次修改时间 */
       time_t st_ctime;           /* 最后一次改变时间(指属性) */
   };

查看某个文件的属性: 

-  如果需要查看某个文件属性, 用 ``stat`` 命令就好

-  ``ls`` 命令在跟上一定参数后可以显示文件的相关属性, 比如 ``-l`` 参数

.. code:: shell

   stat file_name
   ls -l file_name



文件类型
^^^^^^^^

文件类型对应上面的
``st_mode``\ , 文件类型有很多, 比如: 常规文件、符号链接(硬链接、软连接)、管道文件、设备文件(符号设备、块设备)、socket
文件等, 不同的文件类型对应不同的功能和作用

在命令行简单地区分各类文件: 

.. code:: shell
   
   ls -l

简单比较文件的异同: 

.. code:: shell
   
   test


普通文件再分类: 

.. code:: shell
   
   test




文件属主
^^^^^^^^



文件权限
^^^^^^^^



文件大小
^^^^^^^^



文件访问、更新、修改时间
^^^^^^^^^^^^^^^^^^^^^^^^



文件名
^^^^^^



文件的基本操作
~~~~~~~~~~~~~~

-  创建文件

-  删除文件

-  复制文件



创建文件
^^^^^^^^

.. code:: shell

   $ touch regular_file
   $ mkdir directory_file

.. code:: shell

   $ ln regular_file regular_file_hard_link
   $ ln -s regular_file regular_file_soft_link

.. code:: shell

   $ mkfifo fifo_pipe
   $ mknod hda1_block_dev_file b 3 1
   $ mknod null_char_dev_file c 1 3



删除文件
^^^^^^^^

.. code:: shell

   rm regular_file
   rmdir directory_file
   rm -r directory_file_not_empty



复制文件
^^^^^^^^

.. code:: shell

   $ cp regular_file regular_file_copy
   $ cp -r directory_file directory_file_copy



8.Shell程序调试方法
-------------------

`Bash的调试手段 <http://tinylab.org/bash-debugging-tools/>`__
`Shell脚本调试技术 <https://www.ibm.com/developerworks/cn/linux/l-cn-shell-debug/index.html>`__



9.用户管理
----------

   -  在实际使用中, Linux
      系统首先是面向用户的系统, 所有值钱介绍的内容全部是提供给不同的用户使用的. 实际使用中常常碰到各类用户操作. 

   -  Linux
      支持多用户, 也就是说允许不同的人使用同一个系统, 每个人有一个属于自己的账号. 而且允许大家设置不同的认证密码, 确保大家的私有信息得到保护. 另外, 为了确保整个系统的安全, 用户权限又做了进一步划分, 包括普通用户和系统管理员. 普通用户只允许访问自己账户授权下的信息, 而系统管理员才能访问所有资源. 普通用户
      如果想行使管理员的职能, 必须获得系统管理员的许可. 

   -  查看用户相关的命令帮助

      -  ``man 5 passwd``

      -  ``man shadow``

      -  ``man group``

      -  ``man gshadow``



7.1 Linux 用户账号
~~~~~~~~~~~~~~~~~~

   -  账号操作主要是: 增、删、改、禁

   -  Linux 系统提供了底层的
      ``useradd``\ 、\ ``userdel``\ 、\ ``usermod``
      来完成相关操作, 也提供了进一步的简化封装
      ``adduser``\ 、\ ``deluser``

   -  由于只有系统管理员才能创建新用户, 请确保以 ``root``
      账号登录或者可以通过 ``sudo`` 切换为管理员账号



添加用户
^^^^^^^^

.. code:: shell

   # 创建家目录、指定登录 Shell
   $ useradd -s /bin/bash -m test
   $ groups test

   # 创建家目录、指定登录 Shell、加入所属组
   $ useradd -s /bin/bash -m -G docker test
   $ groups test



删除用户
^^^^^^^^

.. code:: shell

   # 删除用户以及家目录
   $ userdel -r test



修改用户
^^^^^^^^

.. code:: shell

   # 常常用来修改默认的 Shell
   $ usermod -s /bin/bash test

   # 把用户加入某个新安装软件所属的组
   $ usermod -a -G docker test

   # 修改登录用户名并搬到新家
   $ usermod -d /home/new_test -m -l new_test test



禁用用户
^^^^^^^^

.. code:: shell

   # 禁用某个账号
   $ usermod -L test
   $ usermod -expiredate 1 test



7.2 用户口令
~~~~~~~~~~~~

   -  口令操作主要是设置、删除、修改、禁用

   -  Linux 系统提供了 ``passwd`` 命令来管理用户口令



设置口令
^^^^^^^^

.. code:: shell

   $ passwd test



删除口令
^^^^^^^^

.. code:: shell

   # 让用户 test 无需密码登录(密码为空), 这个很方便某些安全无关紧要的条件下(比如已登录主机中的虚拟机), 可避免每次频繁输入密码
   $ passwd -d test



修改口令
^^^^^^^^

.. code:: shell

   $ passwd test



禁用口令
^^^^^^^^

.. code:: shell

   $ passwd -l user



7.2 Mac 用户、用户组操作
~~~~~~~~~~~~~~~~~~~~~~~~

使用 Mac 的时候需要像 Linux 一样对用户和群组进行操作, 但是 Linux 使用的
``gpasswd`` 和 ``usermod`` 在 Mac 上都不可以使用, Mac 使用 ``dscl`` 来对
``group`` 和 ``user`` 操作

查看用户组、用户: 

.. code:: shell

   $ dscl . list /Groups
   $ dscl . list /Users

   $ sudo dscl . -list /Groups GroupMembership
   $ sudo dscl . -append /Groups/groupname GroupMembership username
   $ sudo dscl . -delete /Groups/groupname GroupMembership username

添加用户组、添加用户: 

.. code:: shell

   $ sudo dscl . -create /Groups/test

   $ sudo dscl . -create /Users/redis

删除用户组、用户: 

.. code:: shell

   $ sudo dscl . -delete /Groups/test

   $ sudo dscl . -delete /Users/redis
