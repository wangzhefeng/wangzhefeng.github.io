---
title: Pyhton 异常处理
author: 王哲峰
date: '2022-07-26'
slug: python-exception
categories:
  - python
tags:
  - tool
---

<style>
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
img {
    pointer-events: none;
}
</style>

<details><summary>目录</summary><p>

- [异常的角色](#异常的角色)
- [默认异常处理器](#默认异常处理器)
- [捕获异常](#捕获异常)
- [引发异常](#引发异常)
- [用户定义的异常](#用户定义的异常)
- [终止行为](#终止行为)
- [异常层次结构](#异常层次结构)
- [语法错误(Syntax Errors)](#语法错误syntax-errors)
- [捕获异常](#捕获异常-1)
  - [捕获所有异常](#捕获所有异常)
  - [创建自定义异常](#创建自定义异常)
- [traceback](#traceback)
</p></details><p></p>

为什么使用异常? 

- 异常可以改变程序中控制流程的事件
- 在 Python 中, 异常会根据错误自动地被触发, 也能由代码触发和截获
- 异常让我们从一个程序中任意大的代码中跳出来
- 异常可以在一个步骤内跳至异常处理器, 终止开始的所有函数调用而进入异常处理器, 在异常处理器中编写代码, 
  来响应在适当时候引发的异常

异常处理器会留下标识, 并可执行一些代码. 程序前进到某处代码时, 产生异常, 因而会使 Python 立即跳到那个标识, 
而放弃留下该标识之后所调用的任何激活的函数. 这个协议提供了一种固有的方式响应不寻常的事件. 再者, 
因为 Python 会立即跳到处理器的语句代码更简单——对于可能会发生失败的函数的每次调用, 通常就没有必要检查这些函数的状态码. 

异常由四个语句处理:

- ``try/except``
    - 捕获由 Python 或你引起的异常并恢复
- ``try/finally``
    - 无论异常是否发生, 执行清理行为
- ``raise``
    - 手动在代码中触发异常
- ``assert``
    - 有条件地在程序代码中触发异常
- ``with/as``
    - 在 Python 2.6 和后续版本中实现环境管理

# 异常的角色

在 Python 中, 异常通常可以用于各种用途, 下面是它最常见的几种角色: 

- **错误处理**
    - 每当在运行时检测到程序错误时, Python 就会引发异常. 可以在程序代码中捕捉和响应错误, 
      或者忽略已发生的异常
    - 如果忽略错误, Python 默认的异常处理行为将启动: 停止程序, 打印出错误信息; 
    - 如果不想启动这种默认行为, 就要写 try 语句来捕捉异常并从异常中恢复: 当检测到错误时, 
      Python 会跳到 try 处理器, 而程序在 try 之后会重新继续执行; 
- **事件通知**
    - 异常可用于发出有效状态的信号, 而不需在程序间传递结果标志位, 或者刻意对其进行测试. 
- **特殊情况处理**
    - 有时发生了某种很罕见的情况, 很难调整代码去处理, 
      通常会在异常处理器中处理这些罕见的情况, 从而省去编写应对特殊情况的代码. 
- **终止行为**
    - try/finally 语句可确保一定会进行需要的结束运算, 无论程序中是否有异常
- **非常规控制流程**
    - 因为异常是一种高级的 "goto", 它可以作为实现非常规的控制流程的基础

# 默认异常处理器

默认的异常处理器: 就是打印标准出错消息. 消息包括引发的异常还有堆栈跟踪, 
也就是异常发生时激活的程序行和函数清单. 

通过交互模式编写代码时, 文件名就是 "stdin" (标准输入流), 表示标准的输入流. 
当在 IDLE GUI 的交互 shell 中工作的时候, 文件名就是 "pyshell", 
并且会显示出源行. 

在交互模式提示符环境外启动的更为现实的程序中, 顶层的默认处理器也会立刻终止程序. 
对简单的脚本而言, 这种行为很有道理. 错误通常应该是致命错误, 而当其发生时, 所能做的就是
查看标准出错消息. 


# 捕获异常

如果不想要默认的异常行为, 就需要把调用包装在 try 语句内, 自行捕捉异常. 当 try 代码块执行时触发异常, 
Python 会自动跳至处理器(指出引发的异常名称的 except 分句下面的代码块), 并会从中恢复执行. 

# 引发异常



# 用户定义的异常



# 终止行为




# 异常层次结构

内置异常的类层级结构如下: 

```python
BaseException
+-- SystemExit
+-- KeyboardInterrupt
+-- GeneratorExit
+-- Exception
    +-- StopIteration
    +-- StopAsyncIteration
    +-- ArithmeticError
    |    +-- FloatingPointError
    |    +-- OverflowError
    |    +-- ZeroDivisionError
    +-- AssertionError
    +-- AttributeError
    +-- BufferError
    +-- EOFError
    +-- ImportError
    |    +-- ModuleNotFoundError
    +-- LookupError
    |    +-- IndexError: 序列检测到超出边界的索引运算
    |    +-- KeyError
    +-- MemoryError
    +-- NameError
    |    +-- UnboundLocalError
    +-- OSError
    |    +-- BlockingIOError
    |    +-- ChildProcessError
    |    +-- ConnectionError
    |    |    +-- BrokenPipeError
    |    |    +-- ConnectionAbortedError
    |    |    +-- ConnectionRefusedError
    |    |    +-- ConnectionResetError
    |    +-- FileExistsError
    |    +-- FileNotFoundError
    |    +-- InterruptedError
    |    +-- IsADirectoryError
    |    +-- NotADirectoryError
    |    +-- PermissionError
    |    +-- ProcessLookupError
    |    +-- TimeoutError
    +-- ReferenceError
    +-- RuntimeError
    |    +-- NotImplementedError
    |    +-- RecursionError
    +-- SyntaxError
    |    +-- IndentationError
    |         +-- TabError
    +-- SystemError
    +-- TypeError
    +-- ValueError
    |    +-- UnicodeError
    |         +-- UnicodeDecodeError
    |         +-- UnicodeEncodeError
    |         +-- UnicodeTranslateError
    +-- Warning
        +-- DeprecationWarning
        +-- PendingDeprecationWarning
        +-- RuntimeWarning
        +-- SyntaxWarning
        +-- UserWarning
        +-- FutureWarning
        +-- ImportWarning
        +-- UnicodeWarning
        +-- BytesWarning
        +-- ResourceWarning
```

# 语法错误(Syntax Errors)

- 默认异常
- 捕获异常
- 引发异常
- 创建自定义异常

```python
def action():
    pass

try:
    action()
except:
    print('something')
except NameError:
    print('statements')
except IndexError as data:
    print('statements')
except KeyError, value2:
    print('statements')
except (AttributeError, TypeError):
    print('statements')
except (AttributeError, TypeError, SyntaxError), value3:
    print('statements')
else:
    print('statements')
finally:
    print('statements')
```

# 捕获异常

## 捕获所有异常

- 想要捕获所有的异常, 可以直接捕获 ``Exception``
    即可. 这样将会捕获除了
    ``SystemExit``\ 、\ ``KeyboardInterrupt``\ 、\ ``GeneratorExit``
    之外的所有异常. 如果想捕获这三个异常, 将 ``Exception`` 改成
    ``BaseException`` 即可. 

- 自定义异常类应该总是继承自内置的 ``Exception`` 类, 
    或者是继承自那些本身就是从 ``Exception`` 继承而来的类. 
    尽管所有类同时也继承自 ``BaseException``
    , 但你不应该使用这个基类来定义新的异常.  ``BaseException``
    是为系统退出异常而保留的, 比如 ``KeyboardInterrupt`` 或
    ``SystemExit`` 以及其他那些会给应用发送信号而退出的异常. 
    因此, 捕获这些异常本身没什么意义.  这样的话, 假如你继承
    ``BaseException`` 可能会导致你的自定义异常不会被捕获而直接发送信号退出程序运行. 

```python
def action():
    pass

try:
    action()
except Exception as e:
    print("Reason:", e)
```

## 创建自定义异常

- 创建新的异常很简单, 定义一个新的 class, 并让它继承自 ``Exception``
    (或者是任何一个已存在的异常类型). 

- 在程序中引入自定义异常可以使得你的代码更具可读性, 能清晰显示谁应该阅读这个代码. 
    还有一种设计是将自定义异常通过继承组合起来. 在复杂应用程序中, 
    使用基类来分组各种异常类也是很有用的. 它可以让用户捕获一个范围很窄的特定异常.

```python
# 创建新的异常类
class Error_1(Exception):
    pass

class Error_2(Error_1):
    pass

class Error_3(Error_2):
    pass

# 使用自定义的异常
try:
    action()
except Error_1 as e:
    print("Reason:", e)
except Error_2 as e:
    print("Reason:", e)
except Error_3 as e:
    print("Reason:", e)
```

# traceback

```python
#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import sys
import traceback as traceback

print(sys.exc_info())
print(sys.exc_info()[2])

try:
    a = 1 / 0
except ZeroDivisionError:
    print(sys.exc_info())
    print(sys.exc_info()[2])
    tb.print_exc(file = sys.stdout)
```



