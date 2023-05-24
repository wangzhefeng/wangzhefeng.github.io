---
title: Python Processing 和 Threading
author: 王哲峰
date: '2021-12-30'
slug: python-processing-threading
categories:
  - python
tags:
  - note
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
img {
    pointer-events: none;
}
</style>

<details><summary>目录</summary><p>

- [进程、线程、并发、并行、高并发](#进程线程并发并行高并发)
  - [多任务执行](#多任务执行)
  - [进程(Process)、线程(Threading)](#进程process线程threading)
  - [并发](#并发)
  - [并行](#并行)
  - [高并发](#高并发)
- [Python 多进程](#python-多进程)
  - [multiprocessing--基于进程的并行](#multiprocessing--基于进程的并行)
    - [概述](#概述)
    - [Process 类](#process-类)
    - [Pool](#pool)
    - [子进程](#子进程)
    - [进程间通信](#进程间通信)
- [Python 多线程](#python-多线程)
  - [调用 Thread 类的构造器创建线程](#调用-thread-类的构造器创建线程)
  - [继承Thread类创建线程类](#继承thread类创建线程类)
- [IO 编程](#io-编程)
  - [IO 编程](#io-编程-1)
  - [异步 IO](#异步-io)
  - [协程](#协程)
  - [asyncio、async/await、aiohttp](#asyncioasyncawaitaiohttp)
    - [asyncio](#asyncio)
    - [async/await](#asyncawait)
    - [aiohttp](#aiohttp)
- [参考资料](#参考资料)
</p></details><p></p>


# 进程、线程、并发、并行、高并发

- `知乎解释 <https://www.zhihu.com/question/307100151/answer/894486042>`_ 
- `简单解释 <http://www.ruanyifeng.com/blog/2013/04/processes_and_threads.html>`_ 

## 多任务执行

- 单核 CPU 多任务模式
    - 现在, 多核 CPU已经非常普及了, 但是, 即使过去的 **单核 CPU**, 也可以执行 **多任务**. 
      由于 CPU 执行代码都是顺序执行的, 因此, 单核 CPU 执行多任务就是操作系统轮流让各个任务 **交替执行**, 
      任务 1 执行 0.01 秒, 切换到任务 2, 任务 2 执行 0.01秒, 再切换到任务 3, 
      执行 0.01 秒...这样反复执行下去. 表面上看, 每个任务都是交替执行的, 
      但是, 由于 CPU 的执行速度实在是太快了, 我们感觉就像所有任务都在同时执行一样
- 多核 CPU 多任务模式
    - 真正的 **并行执行多任务** 只能在 **多核 CPU** 上实现, 
      但是, 由于任务数量远远多于CPU的核心数量, 
      所以, 操作系统也会自动把很多任务轮流调度到每个核心上执行

## 进程(Process)、线程(Threading)

- 进程、多进程

  - 几乎所有的 **操作系统** 都支持同时运行多个 *任务* , 每个任务通常是一个 *程序* , 
    每一个运行中的程序就是一个 **进程**, 即进程是应用程序的执行实例
  - 现在的 **操作系统** 几乎都支持 **多进程并发** 执行. 但事实的真相是, 对于一个 CPU 而言, 
    在某个时间点它只能执行一个程序. 也就是说, 只能运行一个进程, CPU 不断地在这些进程之间轮换执行. 
    那么, 为什么用户感觉不到任何中断呢?  这是因为相对人的感觉来说, 
    CPU 的执行速度太快了(如果启动的程序足够多, 则用户依然可以感觉到程序的运行速度下降了). 
    所以, 虽然 CPU 在多个进程之间轮换执行, 但用户感觉到好像有多个进程在同时执行. 
- 线程、多线程
  - **线程** 是进程的组成部分, 一个进程可以拥有多个线程. 在多个线程中, 
    会有一个 **主线程** 来完成整个进程从开始到结束的全部操作, 
    而其他的线程会在主线程的运行过程中被创建或退出. 当进程被初始化后, 
    主线程就被创建了, 对于绝大多数的应用程序来说, 
    通常仅要求有一个主线程, 但也可以在进程内创建多个 **顺序执行流**, 
    这些顺序执行流就是线程
  - 当一个进程里只有一个线程时, 叫做 **单线程**, 超过一个线程就叫做 **多线程**

    > 由于每个进程至少要干一件事, 所以, 一个进程至少有一个线程. 多线程的执行方式和多进程是一样的, 
      也是由操作系统在多个线程之间快速切换, 让每个线程都短暂地交替运行, 看起来就像同时执行一样. 
      当然, 真正地同时执行多线程需要多核CPU才可能实现

  - 每个线程必须有自己的 **父进程**, 且它可以拥有自己的 **堆栈**、 **程序计数器** 和 **局部变量**, 
    但不拥有系统资源, 因为它和父进程的其他线程共享该进程所拥有的全部资源. 
    线程可以完成一定的任务, 可以与其他线程共享父进程中的共享变量及部分环境, 
    相互之间协同完成进程所要完成的任务
    
    > 多个线程共享父进程里的全部资源, 会使得编程更加方便, 
      需要注意的是, 要确保线程不会妨碍同一进程中的其他线程

  - 线程是独立运行的, 它并不知道进程中是否还有其他线程存在. 线程的运行是 **抢占式** 的, 
    也就是说, 当前运行的线程在任何时候都可能被挂起, 以便另外一个线程可以运行
  - 一个线程可以创建和撤销另一个线程, 同一个进程中的多个线程之间可以 **并发** 运行, 
    即同一时刻, 主程序只允许有一个线程执行
  - 从逻辑的角度来看, 多线程存在于一个应用程序中, 让一个应用程序可以有多个执行部分同时执行, 
    但操作系统无须将多个线程看作多个独立的应用, 对多线程实现调度和管理以及资源分配, 
    线程的调度和管理由进程本身负责完成
- 进程与线程的关系
    - 简而言之, 进程和线程的关系是这样的: 操作系统可以同时执行多个任务, 每一个任务就是一个进程, 
      进程可以同时执行多个任务, 每一个任务就是一个线程
- Python 既支持多进程编程, 又支持多线程编程
    - 多进程编程
        - 所谓多进程编程, 即将整个程序划分为多个子任务, 这些任务在多核 CPU 上可以实现并行执行, 
         反之, 在单核 CPU 上, 只能并发执行
    - 多线程编程
        - 在此基础上, 我们还可以对每个任务进行更细致地划分, 将其分为多个线程, 
          和多进程不同, 每个任务的多个线程, 只能利用某一个 CPU 并发执行
    - Python 多进程、多线程编程的实现方式
        - 创建进程(线程)
        - 启动进程(线程)
        - 管理多进程(多线程)

## 并发

并发是指在同一时刻只能有一条指令执行, 但多个进程指令被快速轮换执行, 使得在宏观上具有多个 **进程/线程** 执行的效果

- 多进程并发
- 多线程并发

## 并行

并行指在同一时刻有多条指令(任务)在多个 **CUP 处理器** 上同时执行

## 高并发

# Python 多进程

在使用 `multiprocessing` 库实现多进程之前, 我们先来了解一下操作系统相关的知识. 

- Unix/Linux 实现多进程
    - Unix/Linux 操作系统提供了一个 `fork()` 系统调用, 它非常特殊. 普通的函数调用, 调用一次, 返回一次, 
      但是 `fork()` 调用一次, 返回两次, 因为操作系统自动把当前父进程复制了一份子进程, 然后, 
      分别在父进程和子进程内返回.
    - 子进程永远返回 0, 而父进程返回子进程的 ID. 这样, 一个父进程可以 fork 出很多子进程, 所以, 
      父进程要记下每个子进程的 ID, 而子进程只需要调用 `getppid()` 就可以拿到父进程的 ID.
    - 有了 fork 调用, 一个进程在接到新任务时就可以复制出一个子进程来处理新任务, 常见的 Apache 服务器就是由父进程监听端口, 
      每当有新的 http 请求时, 就 fork 出子进程来处理新的 http 请求.
    - Python 的 `os` 模块封装了常见的系统调用, 其中就包括 `fork`, 可以在 Python 程序中轻松创建子进程.

```python
import os
print("Process (%s) start..." % os.getpid())
# Only works on Unix/Linux/Mac
pid = os.fock()
if pid == 0:
    print(f"I am child process ({os.getpid()}) and my parent is {os.getppid()}.")
else:
    print(f"I ({os.getpid()}) just created a child process ({pid}).")
```

- Windows的多进程
    - 由于 Windows 没有 fork 调用, 而如果我们需要在 Windows 上用 Python 编写多进程的程序, 就需要使用到 `multiprocessing` 模块

## multiprocessing--基于进程的并行

### 概述

由于 Python 是跨平台的, 自然也应该提供一个跨平台的多进程支持. `multiprocessing` 模块就是跨平台版本的多进程模块. 
`multiprocessing` 模块提供了一个 `Process` 类来代表一个进程对象. 

```python
from multiprocessing import Process
import os

# 子进程要执行的代码
def run_proc(name):
   print("Run child process %s (%s)..." % (name, os.getpid()))

if __name__ == "__main__":
   print("Parent process %s." % os.getpid())
   p = Process(target = run_proc, args = ("test",))
   print("Child process will start.")
   p.start()
   p.join()
   print("Child process end.")
```

### Process 类

在 `multiprocessing` 中, 通过创建一个 Process 对象然后调用它的 `start()` 方法来生成进程. 
`Process` 和 `threading.Thread API` 相同. 一个简单的多进程程序示例是: 

```python
from multiprocessing import Process

def f(name):
   print("hello", name)

if __name__ == "__main__":
   p = Process(target = f, args = ("bob",))
   p.start()
   p.join()
```

要显示所涉及的各个进程 ID, 这是一个扩展示例: 

```python
from multiprocessing import Process
import os

def info(title):
   print(title)
   print("module name:", __name__)
   print("parent process:", os.getppid())
   print("process id:", os.getpid())

def f(name):
   info("function f")
   print("hello", name)

if __name__ == "__main__":
   info("main line")
   p = Process(target = f, args = ("bob",))
   p.start()
   p.join()
```

### Pool

如果要启动大量的子进程, 可以用进程池的方式批量创建子进程.

```python
from multiprocessing import Pool
import os, time, random

def long_time_task(name):
   print("Run task %s (%s)..." % ())
```

### 子进程

### 进程间通信

Process 之间肯定是需要通信的, 操作系统提供了很多机制来实现进程间的通信. Python 的 multiprocessing 模块包装了底层机制, 
提供了 Queue、Pipes 等多种方式来交换数据. 

以 Queue 为例, 在父进程中创建两个子进程, 一个往 Queue 里写数据, 一个从 Queue 里读数据.

```python
from multiprocessing import Process, Queue
import os, time, random

# 写数据进程执行的代码:
def write(q):
   print('Process to write: %s' % os.getpid())
   for value in ['A', 'B', 'C']:
      print('Put %s to queue...' % value)
      q.put(value)
      time.sleep(random.random())

# 读数据进程执行的代码:
def read(q):
   print('Process to read: %s' % os.getpid())
   while True:
      value = q.get(True)
      print('Get %s from queue.' % value)

if __name__=='__main__':
   # 父进程创建Queue, 并传给各个子进程: 
   q = Queue()
   pw = Process(target=write, args=(q,))
   pr = Process(target=read, args=(q,))
   # 启动子进程pw, 写入:
   pw.start()
   # 启动子进程pr, 读取:
   pr.start()
   # 等待pw结束:
   pw.join()
   # pr进程里是死循环, 无法等待其结束, 只能强行终止:
   pr.terminate()
```

```
Put A to queue...
Process to read: 50564
Get A from queue.
Put B to queue...
Get B from queue.
Put C to queue...
Get C from queue.
```

# Python 多线程

- Python3 线程中常用的两个模块为: 
  - `_thread` 是 Pyton3 以前版本中 thread 模块的重命名, 此模块提供了低级别的、
    原始的线程以及一个简单的锁, 它相比 `threading` 模块的功能还是比较有限的, 一般不建议使用
  - `threading` 是 Python3 之后的线程模块, 提供了丰富的多线程支持, 推荐使用
- Python 主要通过两种方式来创建线程
  - 使用 `threading` 模块中 `Thread` 类的构造器创建线程. 
    即直接对类 `threading.Thread` 进行实例化创建线程, 
    并调用实例化对象的 `start()` 方法启动线程
  - 继承 `threading` 模块中的 `Thread` 类创建线程类. 
    即用 `threading.Thread` 派生出一个新的子类, 
    将新建类实例化创建线程, 并调用其 `start()` 方法启动线程. 

## 调用 Thread 类的构造器创建线程

## 继承Thread类创建线程类

# IO 编程

## IO 编程

- **IO**: 
    - IO 在计算机中指 Input/Output, 也就是输入和输出
    - 由于程序和运行时数据是在内存中驻留, 由 CPU 这个超快的计算核心来执行, 涉及到数据交换的地方通常是磁盘、网络等, 
      就需要 IO 接口
- **Steam**: 
    - IO 编程中, Stream(流)是一个很重要的概念, 可以把流想象成一个水管, 数据就是水管里的水, 但是只能单向流动, 
      Input Stream 就是数据从外面(磁盘、网络)流进内存, Output Stream 就是数据从内存流到外面去
- **同步/异步 IO**: 
    - 由于 CPU 和内存的速度远远高于外设的速度, 所以在 IO 编程中, 就存在速度严重不匹配的问题
    - 举个例子: 比如要把 100M 的数据写入磁盘, CPU 输出 100M 的数据只需要 0.01s, 可是磁盘接收这 100M 数据可能需要 10s, 怎么办呢? 有两种办法: 
        - (1)CPU 等着, 也就是程序暂停执行后续代码, 等 100M 的数据在 10s 后写入磁盘, 再接着往下执行, 这种模式成为 **同步 IO**
        - (2)CPU 不等待, 只是告诉磁盘, “您老慢慢写, 不着急, 我接着干别的事去了”, 于是, 后续代码可以立刻接着执行, 这种模式称为 **异步 IO**
    - 同步和异步 IO 的区别就在于是否等待 IO 执行的结果. 很明显使用异步 IO 来编写程序性能会远远高于同步 IO, 
      但是异步 IO 的缺点是编程复杂, 异步 IO 通知的方式有两种: 
        - 回调模式
        - 轮询模式
    - 操作 IO 的能力都是由操作系统提供的, 每一种编程语言都会把操作系统提供的低级 C 接口封装起来方便使用, Python 也不例外

## 异步 IO

- CPU 的速度远远快于磁盘、网络等 IO, 在一个线程中, CPU 执行代码的速度极快, 
  然而, 一旦遇到 IO 操作, 如读写文件、发送网络数据时, 就需要等待 IO 操作完成, 
  才能进行下一步操作. 这种情况称为同步 IO. 
- 在 IO 操作的过程中, 当前线程被挂起, 而其他需要 CPU 执行的代码就无法被当前线程执行了. 
  因为一个 IO 操作就阻塞了当前线程, 导致其他代码无法执行, 所以我们必须使用多线程或者多进程来并发执行代码, 
  为多个用户服务, 每个用户都会分配一个线程, 如果遇到 IO 导致线程被挂起, 其他用户的线程不受影响. 
- 多线程和多进程的模型虽然解决了并发问题, 但是系统不能无上限地增加线程. 由于系统切换线程的开销也很大, 
  所以, 一旦线程数量过多, CPU 的时间就花在线程切换上了, 真正运行代码的时间就少了, 结果导致性能严重下降. 
- 针对 CPU 高速执行能力和 IO 设备的龟速严重不匹配问题, 有两种方式可以解决: 
    - 多线程、多进程
    - 异步 IO
        - 当代码需要执行一个耗时的 IO 操作时, 它只发出 IO 指令, 并不等待 IO 结果, 然后就去执行其他代码了, 
          一段时间后, 当 IO 返回结果时, 再通知 CPU 进行处理
        - 异步 IO 模型需要一个消息循环, 在消息循环中, 主线程不断地重复 `读取消息--处理消息` 这一过程
- 消息模型是如何解决同步 IO 必须等待 IO 操作这一问题的呢? 当遇到 IO 操作时, 代码只负责发出 IO 请求, 
  不等待 IO 结果, 然后直接结束本轮消息处理, 进入下一轮消息处理过程. 当 IO 操作完成后, 将收到一条“IO 完成”的消息, 
  处理该消息时就可以直接获取 IO 操作结果. 在“发出 IO 请求”到收到“IO 完成”的这段时间里, 同步 IO 模型下, 
  主线程只能挂起, 但异步 IO 模型下, 主线程并没有休息, 而是在消息循环中继续处理其他消息. 这样, 在异步 IO 模型下, 
  一个线程就可以同时处理多个 IO 请求, 并且没有切换线程的操作. 对于大多数 IO 密集型的应用程序, 
  使用异步 IO 将大大提升系统的多任务处理能力. 


消息模型其实早在应用在桌面应用程序中了. 一个GUI程序的主线程就负责不停地读取消息并处理消息. 
所有的键盘、鼠标等消息都被发送到GUI程序的消息队列中, 然后由GUI程序的主线程处理. 

由于GUI线程处理键盘、鼠标等消息的速度非常快, 所以用户感觉不到延迟. 某些时候, 
GUI线程在一个消息处理的过程中遇到问题导致一次消息处理时间过长, 此时, 用户会感觉到整个GUI程序停止响应了, 
敲键盘、点鼠标都没有反应. 这种情况说明在消息模型中, 处理一个消息必须非常迅速, 否则, 主线程将无法及时处理消息队列中的其他消息, 
导致程序看上去停止响应. 


老张爱喝茶, 废话不说, 煮开水.  出场人物: 老张, 水壶两把(普通水壶, 简称水壶; 会响的水壶, 简称响水壶).  

- 1.老张把水壶放到火上, 立等水开
    - 【同步阻塞】老张觉得自己有点傻
- 2.老张把水壶放到火上, 去客厅看电视, 时不时去厨房看看水开没有
    - 【同步非阻塞】老张还是觉得自己有点傻, 于是变高端了, 买了把会响笛的那种水壶. 水开之后, 能大声发出嘀~~~~的噪音
- 3.老张把响水壶放到火上, 立等水开
    - 【异步阻塞)】老张觉得这样傻等意义不大
- 4.老张把响水壶放到火上, 去客厅看电视, 水壶响之前不再去看它了, 响了再去拿壶
    - 【异步非阻塞】老张觉得自己聪明了

所谓同步异步, 只是对于水壶而言:

- 普通水壶, 同步
- 响水壶, 异步

虽然都能干活, 但响水壶可以在自己完工之后, 提示老张水开了. 这是普通水壶所不能及的. 
同步只能让调用者去轮询自己(情况2中), 造成老张效率的低下. 

- 所谓阻塞非阻塞, 仅仅对于老张而言:
   - 立等的老张, 阻塞
   - 看电视的老张, 非阻塞

情况 1 和情况 3 中老张就是阻塞的, 媳妇喊他都不知道. 虽然 3 中响水壶是异步的, 可对于立等的老张没有太大的意义. 
所以一般异步是配合非阻塞使用的, 这样才能发挥异步的效用. 


## 协程

- 协程, 又称微线程、迁程、Coroutine. 
   - 协程的概念很早就提出来了, 但知道最近几年才在某些语言(如 Lua)中得到广泛应用. 
- 子程序, 或者称为函数, 在所有语言中都是层级调用的
   - 子程序调用是通过栈实现的, 一个线程就是执行一个子程序

子程序调用总是一个入口, 一次返回, 调用顺序是明确的, 而协程的调用和子程序不同. 
协程看上去也是子程序, 但执行过程中, 在子程序内部可中断, 然后转而执行别的子程序, 在适当的时候再返回来接着执行. 

- 协程最大的优势就是极高的执行效率. 
   - 因为子程序切换不是线程切换, 而是由程序自身控制, 因此, 没有线程切换的开销, 和多线程比, 线程数量越多, 协程的性能优势就越明显. 
   - 第二大优势就是不需要多线程的锁机制, 因为只有一个线程, 也不存在同时写变量冲突, 在协程中控制共享资源不加锁, 只需要判断状态就好了, 
     所以执行效率比多线程高很多. 

因为协程是一个线程执行, 那怎么利用多核CPU呢? 最简单的方法是多进程+协程, 既充分利用多核, 又充分发挥协程的高效率, 可获得极高的性能. 

Python 对协程的支持是通过 generator 实现的, 在 generator 中, 不但可以通过 `for` 循环来迭代, 
还可以不断调用 `next()` 函数获取由 `yield` 语句返回的下一个值. 但是 Python 的 `yield` 不但可以返回一个值, 
它还可以接收调用者发出的参数

## asyncio、async/await、aiohttp

### asyncio

asyncio 是 Python3.4 一如的标准库, 直接内置了对异步 IO 的支持. asyncio 的编程模型就是一个消息循环. 
从 asyncio 模块中直接获取一个 EventLoop 的引用, 然后把需要执行的协程扔到 EventLoop 中执行, 
就实现了异步 IO.

- asyncio 提供了完善的异步 IO 支持
- 异步 IO 操作需要在 coroutine 中通过 yield from 完成
- 多个 coroutine 可以封装成一组 Task 然后并发执行

- 示例 1: 用asyncio实现Hello world代码如下

```python
import asyncio

@asyncio.coroutine
def hello():
    print("Hello, world!")
    # 异步调用 asyncio.sleep(1)
    r = yield from asyncio.sleep(1)
    print("Hello, again!")

# 获取 EventLoop
loop = asyncio.get_event_loop()
# 执行 coroutine
loop.run_until_complete(hello())
loop.close()
```

> @asyncio.coroutine把一个generator标记为coroutine类型, 然后, 我们就把这个coroutine扔到EventLoop中执行. 
> 
> hello()会首先打印出Hello world!, 然后, yield from语法可以让我们方便地调用另一个generator. 
由于asyncio.sleep()也是一个coroutine, 所以线程不会等待asyncio.sleep(), 而是直接中断并执行下一个消息循环. 
当asyncio.sleep()返回时, 线程就可以从yield from拿到返回值(此处是None), 然后接着执行下一行语句. 
> 
> 把asyncio.sleep(1)看成是一个耗时1秒的IO操作, 在此期间, 主线程并未等待, 而是去执行EventLoop中其他可以执行的coroutine了, 
因此可以实现并发执行. 

- 示例 2: 用Task封装两个 coroutine

```python
import threading
import asyncio

@asyncio.coroutine
def hello():
   print("Hello, world! (%s)" % threading.currentThread())
   yield from asyncio.sleep(1)
   print("Hello again! (%s)" % threading.currentThread())

loop = asyncio.get_event_loop()
tasks = [hello(), hello()]
loop.run_until_complete(asyncio.wait(tasks))
loop.close()
```

```
Hello world! (<_MainThread(MainThread, started 140735195337472)>)
Hello world! (<_MainThread(MainThread, started 140735195337472)>)
(暂停约1秒)
Hello again! (<_MainThread(MainThread, started 140735195337472)>)
Hello again! (<_MainThread(MainThread, started 140735195337472)>)
```

- 由打印的当前线程名称可以看出, 两个coroutine是由同一个线程并发执行的. 
- 如果把asyncio.sleep()换成真正的IO操作, 则多个coroutine就可以由一个线程并发执行. 

### async/await

用 asyncio 提供的 @asyncio.coroutine 可以把一个 generator 标记为 coroutine 类型, 
然后在 coroutine 内部用 yield from 调用另一个 coroutine 实现异步操作. 

为了简化并更好地标识异步 IO, 从 Python3.5 开始引入了新的语法 async 和 await, 可以让 coroutine 的代码更简洁易读. 

请注意, async 和 await 是针对 coroutine 的新语法, 要使用新的语法, 只需要两步简单的替换: 

- (1)把 @asyncio.coroutine 替换为 async
- (2)把 yield from 替换为 await

示例: 

```python
@asyncio.coroutine
def hello():
    print("Hello world!")
    r = yield from asyncio.sleep(1)
    print("Hello again!")
```

```python
async def hello():
    print("Hello world!")
    r = await asyncio.sleep(1)
    print("Hello again!")
```

### aiohttp

`asyncio` 可以实现单线程并发 IO 操作. 如果仅用在客户端, 发挥的威力不大. 
如果把 `asyncio` 用在服务器端, 例如 Web 服务器, 
由于 HTTP 连接就是 IO 操作, 因此可以用单线程 + `coroutine` 实现多用户的高并发支持. 

`asyncio` 实现了 TCP、UDP、SSL 等协议, `aiohttp` 则是基于 `asyncio` 实现的 HTTP 框架. 

- `aiohttp` 安装

```bash
$ pip install aiohttp
```

- `aiohttp` 使用: 编写一个 HTTP 服务器, 分别处理以下 URL:
   - `/`
      - 首页返回 `b'<h1>Index</h1>'`
   - `/hello/{name}`
      - 根据 URL 参数返回文本 `hello, %s!`

代码

```python
import asyncio
from aiohttp import web
import async

async def index(request):
await asyncio.sleep(0.5)
return web.Response(body = b"<h1>Index</h1>")

async def hello(request):
await asyncio.sleep(0.5)
text = f"<h1>hello, {request.match_info["name"]}!</h1>"
return web.Response(body = text.encode("utf-8"))

async def init(loop):
app = web.Application(loop = loop)
app.router.add_router("GET", "/", index)
app.router.add_router("GET", "/hello/{name}", hello)
srv = await loop.create_server(app.make_handler(), "127.0.0.1", 8000)
print("Server started at http://127.0.0.1:8000...")
return srv

loop = asyncio.get_event_loop()
loop.run_until_complete(init(loop))
loop.run_forever()
```

- 注意:
    - `aiohttp` 的初始化函数 `init()` 也是一个 `coroutine`
    - `loop.create_server()` 则利用 `asyncio` 创建 TCP 服务. 



# 参考资料

https://www.liaoxuefeng.com/wiki/1016959663602400/1017627212385376