---
title: Hadoop
author: 王哲峰
date: '2022-05-06'
slug: database-hadoop
categories:
  - database
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
</style>


<details><summary>目录</summary><p>

- [Hadoop](#hadoop)
  - [Hadoop 介绍](#hadoop-介绍)
  - [Hadoop 构造模块](#hadoop-构造模块)
    - [NameNode](#namenode)
    - [DataNode](#datanode)
    - [Secondary NameNode](#secondary-namenode)
    - [JobTracker](#jobtracker)
    - [TaskTracker](#tasktracker)
- [HDFS](#hdfs)
  - [HDFS 介绍](#hdfs-介绍)
  - [HDFS 备份](#hdfs-备份)
  - [NameNode](#namenode-1)
    - [editlog](#editlog)
    - [fsimage](#fsimage)
    - [SecondNameNode](#secondnamenode)
    - [ZooKeeper](#zookeeper)
    - [JournalNode](#journalnode)
  - [DataNode](#datanode-1)
  - [HDFS 文件操作](#hdfs-文件操作)
  - [Hadoop 基本文件命令](#hadoop-基本文件命令)
    - [指定文件和目录确切位置的URI](#指定文件和目录确切位置的uri)
    - [基本形式](#基本形式)
    - [基本文件命令](#基本文件命令)
- [MapReduce 程序](#mapreduce-程序)
  - [MapReduce 程序通过操作键值对来处理数据](#mapreduce-程序通过操作键值对来处理数据)
  - [Hadoop 数据类型](#hadoop-数据类型)
  - [Mapper](#mapper)
  - [Reducer](#reducer)
  - [Partitioner: 重定向Mapper的输出](#partitioner-重定向mapper的输出)
  - [Combiner: 本地reduce](#combiner-本地reduce)
- [YARN](#yarn)
- [参考](#参考)
</p></details><p></p>

# Hadoop

## Hadoop 介绍

Google 率先提出了 MapReduce 用来应对数据处理需求的系统, 
Doug Cutting 领导开发了一个开源版本的 MapReduce, 称为 Hadoop

Hadoop 是一个开源框架,可编写和运行分布式应用处理大规模数据;

* 方便: Hadoop 运行在由一般商用机器构成的大型集群上, 或者如 Amazon 弹性计算云(EC2)等云服务器之上;
* 健壮: Hadoop 致力于在一般商用硬件上运行, 其架构假设硬件会频繁地出现失效. 它可以从容地处理大多数此类故障
* 可扩展: Hadoop 通过增加集群节点, 可以线性地扩展以处理更大的数据集
* 简单: Hadoop 允许用户快速编写出高效的并行代码

Hadoop 集群是在同一地点用网络互联的一组通用机器. 数据存储和处理都发生在这个机器"云"中。
不同的用户可以从独立的客户端提交计算作业到 Hadoop, 这些客户端可以是远离 Hadoop 集群的个人台式机

通常在一个 Hadoop 集群中的机器都是相对同构的 x86 Linux 服务器.
而且它们几乎位于同一个数据中心,并且通常在同一组机架里;

Hadoop 强调把代码向数据迁移,而不是相反的过程; Hadoop 的集群环境中既包含数据又包含计算环境, 
客户端仅需发送待执行的 MapReduce 程序, 而这些程序一般都很小(通常为几千字节)

## Hadoop 构造模块

在一个全配置的集群上, "运行 Hadoop" 意味着在网络分布的不同服务器上运行一组守护进程(daemons). 
这些守护进程有着特殊的角色, 一些仅存在单个服务器上, 一些则运行在多个服务器上: 

* NameNode(名字节点)
* DataNode(数据节点)
* Secondary NameNode(次名字节点)
* JobTracker(作业跟踪节点)
* TaskTracker(任务跟踪节点)

### NameNode

Hadoop 在分布式计算机与分布式存储中都采用了 **主/从(master/slave)** 结构。
分布式存储系统被称为 Hadoop 文件系统, 或简称 HDFS 

* **NameNode** 位于 HDFS 的主端, 它指导 **从端的 DataNode** 执行底层的 I/O 任务; 
  NameNode 是 HDFS 的书记员, 他跟踪文件如何被分割成文件块, 而这些块又被哪些节点存储, 
  以及分布式文件系统的整体运行状态是否正常; 
* 运行 NameNode 会消耗大量的内存和 I/O 资源, 因此, 为了减轻机器的负载, 
  驻留在 NameNode 的服务器通常不会存储用户数据或者执行 MapReduce 程序的计算任务, 
  这意味着 NameNode 服务器不会同时是 DataNode 或者 TaskTracker; 

### DataNode

### Secondary NameNode

### JobTracker

JobTracker 守护进程是应用程序和 Hadoop 之间的纽带。
每个 Hadoop 集群只有一个 JobTracker 守护进程, 
通常位于服务器集群的主节点上；一旦提交代码到集群上, JobTracker 就会确定执行计划, 
包括决定处理哪些文件、为不同的任务分配节点以及监控所有任务的运行; 
如果任务失败, JobTracker 将自动重启任务, 但分配的节点可能会不同, 
同时受到预定义的重试次数限制

### TaskTracker


# HDFS

## HDFS 介绍

随着数据量越来越大，在一台机器上已经无法存储所有的数据了，那我们会将这些数据分配到不同的机器来进行存储，
但是这就带来一个问题：不方便管理。所以，希望有一个系统可以将这些分布在不同操作服务器上的数据进行统一管理，
这就有了分布式文件系统。HDFS 就是分布式文件系统的其中一种(目前用得最广泛的一种)

在使用 HDFS 的时候是非常简单的：虽然 HDFS 是将文件存储到不同的机器上，
但是使用的时候是把这些文件当做是存储在一台机器的方式去使用(背后却是多台机器在执行)。
好比，调用了一个 RPC 接口，给它参数，它返回了一个 response。RPC 接口做了什么事情其实不知道，
可能这个 RPC 接口又调用了其他的 RPC 接口，屏蔽掉实现细节，对用户友好。
HDFS 就是一个分布式文件系统，用来存储数据

HDFS 作为一个分布式文件系统，那么它的数据是保存在多个系统上的。例如：一个 1GB 的文件，
会被切分成几个小文件，每个服务器都会存放一部分，默认以 128MB 的大小来切分，每个 128MB 的文件，
在 HDFS 叫做块(block)

> 显然，这个 128MB 大小是可配的。如果设置为太小或者太大都不好。如果切分的文件太小，
> 那一份数据可能分布到多台的机器上(寻址时间就很慢)。如果切分的文件太大，
> 那数据传输时间的时间就很慢
> 
> PS：老版本默认是 64MB

一个用户发出了一个 1GB 的文件请求给 HDFS 客户端，HDFS 客户端会根据配置对这个文件进行切分，
所以 HDFS 客户端会切分为 8 个文件(也叫作 block)，然后每个服务器都会存储这些切分后的文件(block)。
现在假设每个服务器都存储两份，这些存放真实数据的服务器，在 HDFS 领域叫做 DataNode

现在问题来了，HDFS 客户端按照配置切分完以后，怎么知道往哪个服务器(DataNode)放数据呢？
这个时候，就需要另一个角色了，管理者 NameNode。
NameNode 实际上就是管理文件的各种信息(这种信息专业点我们叫做 MetaData 元数据)，
其中包括：文文件路径名，每个 Block 的 ID 和存放的位置等等。所以，无论是读还是写，
HDFS 客户端都会先去找 NameNode，通过 NameNode 得知相应的信息，再去找 DataNode

* 如果是写操作，HDFS 切分完文件以后，会询问 NameNode 应该将这些切分好的 block 往哪几台 DataNode 上写
* 如果是读操作，HDFS 拿到文件名，也会去询问 NameNode 应该往哪几台 DataNode 上读数据

![img](images/HDFS.jpeg)

## HDFS 备份

作为一个分布式系统(把大文件切分为多个小文件，存储到不同的机器上)，如果没有备份的话，
只要有其中的一台机器挂了，那就会导致数据是不可用状态的。Kafka 对 partition 备份，
ElasticSearch 对分片进行备份，而到 HDFS 就是对 Block 进行备份。

尽可能将数据备份到不同的机器上，即便某台机器挂了，那就可以将备份数据拉出来了。
这里的备份并不需要 HDFS 客户端去写，只要 DataNode 之间互相传递数据就好了

## NameNode

NameNode 是需要处理 HDFS 客户端请求的。因为它是存储元数据的地方，无论读写都需要经过它。
现在问题就来了，NameNode 是怎么存放元数据的呢？

* 如果 NameNode 只是把元数据放到内存中，那如果 NameNode 这台机器重启了，那元数据就没了
* 如果 NameNode 将每次写入的数据都存储到硬盘中，那如果只针对磁盘查找和修改又会很慢(因为这个是纯 IO 的操作)

### editlog

说到这里，又想起了 Kafka。Kafka 也是将 partition 写到磁盘里边的，
但人家是怎么写的？顺序 IO

NameNode 同样也是做了这个事：修改内存中的元数据，
然后把修改的信息 append(追加)到一个名为 `editlog` 的文件上。
由于 append 是顺序 IO，所以效率也不会低。现在我们增删改查都是走内存，
只不过增删改的时候往磁盘文件 `editlog` 里边追加一条。这样我们即便重启了 NameNode，
还是可以通过 `editlog` 文件将元数据恢复

### fsimage

现在也有个问题：如果 NameNode 一直长期运行的话，
那 `editlog` 文件应该会越来越大(因为所有的修改元数据信息都需要在这追加一条)。
重启的时候需要依赖 `editlog` 文件来恢复数据，如果文件特别大，那启动的时候不就特别慢了吗？
的确是如此的，那 HDFS 是怎么做的呢？

为了防止 `editlog` 过大，导致在重启的时候需要较长的时间恢复数据，
所以 NameNode 会有一个内存快照，叫做 `fsimage`。
这样一来，重启的时候只需要加载内存快照 fsimage + 部分的 editlog 就可以了。
想法很美好，现实还需要解决一些事：我什么时候生成一个内存快照 fsimage？
我怎么知道加载哪一部分的 editlog？

### SecondNameNode

问题看起来好像复杂，其实我们就只需要一个定时任务。如果让我自己做的话，
我可能会想：我们加一份配置，设置个时间就 OK 了

* 如果 `editlog` 大到什么程度或者隔了多长时间，我们就把 `editlog` 文件的数据跟内存快照 `fsiamge` 给合并起来。
  然后生成一个新的 fsimage，把 editlog 给清空，覆盖旧的 fsimage 内存快照
* 这样一来，NameNode 每次重启的时候，拿到的都是最新的 fsimage 文件，editlog 里边的都是没合并到 fsimage的。
  根据这两个文件就可以恢复最新的元数据信息了

HDFS 也是类似上面这样干的，只不过它不是在 NameNode 起个定时的任务跑，
而是用了一个新的角色：SecondNameNode。至于为什么？可能 HDFS 觉得合并所耗费的资源太大了，
不同的工作交由不同的服务器来完成，也符合分布式的理念

![img](images/secondnamenode.jpeg)

### ZooKeeper

现在问题还是来了，此时的架构 NameNode 是单机的。
SecondNameNode 的作用只是给 NameNode 合并 editlog 和 fsimage文件，
如果 NameNode 挂了，那 client 就请求不到了，而所有的请求都需要走 NameNode，
这导致整个 HDFS 集群都不可用了

于是我们需要保证 NameNode 是高可用的。一般现在我们会通过 Zookeeper 来实现。架构图如下：

![img](images/zookeeper.jpeg)

### JournalNode

主 NameNode 和从 NameNode 需要保持元数据的信息一致，因为如果主 NameNode 挂了，
那从 NameNode 需要顶上，这时从 NameNode 需要有主 NameNode 的信息

所以，引入了 Shared Edits 来实现主从 NameNode 之间的同步，Shared Edits 也叫做 JournalNode。
实际上就是主 NameNode 如果有更新元数据的信息，它的 editlog 会写到 JournalNode，
然后从 NameNode 会在 JournalNode 读取到变化信息，然后同步。
从 NameNode 也实现了上面所说的 SecondNameNode 功能(合并 editlog 和 fsimage）

![img](images/journanode.png)

稍微总结一下：

* NameNode 需要处理 client 请求，它是存储元数据的地方
* NameNode 的元数据操作都在内存中，会把增删改以 editlog 持续化到硬盘中(因为是顺序 IO，所以不会太慢)
* 由于 editlog 可能存在过大的问题，导致重新启动 NameNode 过慢(因为要依赖 editlog 来恢复数据)，
  引出了 fsimage 内存快照。需要跑一个定时任务来合并 fsimage 和 editlog，引出了 SecondNameNode
* 又因为 NameNode 是单机的，可能存在单机故障的问题。所以我们可以通过 Zookeeper 来维护主从 NameNode，
  通过 JournalNode(Share Edits) 来实现主从 NameNode 元数据的一致性。最终实现 NameNode 的高可用

## DataNode

从上面我们就知道，我们的数据是存放在 DataNode 上的，还会备份。
如果某个 DataNode 掉线了，那 HDFS 是怎么知道的呢？
DataNode 启动的时候会去 NameNode 上注册，他俩会维持心跳，
如果超过时间阈值没有收到 DataNode 的心跳，那 HDFS 就认为这个 DataNode 挂了

还有一个问题就是：我们将 Block 存到 DataNode 上，那还是有可能这个 DataNode 的磁盘损坏了部分，
而我们 DataNode 没有下线，但我们也不知道损坏了。
一个 Block 除了存放数据的本身，还会存放一份元数据，包括数据块的长度，块数据的校验和，以及时间戳。
DataNode 还是会定期向 NameNode 上报所有当前所有 Block 的信息，
通过元数据就可校验当前的 Block 是不是正常状态

## HDFS 文件操作

HDFS 是一种文件系统, 专为 MapReduce 这类框架下的大规模分布式数据处理设计, 
可以把一个大数据集在 HDFS 中存储位单个文件, 而大多数其它的文件系统无力实现这一点 

HDFS 并不是一个天生的 Unix 文件系统, 不支持标准的 Unix 文件命令和操作; 
Hadoop 提供了一套与 Linux 文件命令类似的命令行工具, 即 Hadoop 操作文件的 Shell 命令, 
它们是与 HDFS 系统的主要接口

一个典型的 Hadoop 工作流会:  

* 在别的地方生成数据文件(如日志文件), 再将其复制到 HDFS 中 
* 接着由 MapReduce 程序处理这个数据, 但它们通常不会直接读任何一个 HDFS 文件, 
  相反, 它们依靠 MapReduce 框架来读取 HDFS 文件, 并将其解析为独立的记录(键值对),
  这些记录才是 MapReduce 程序所处理的数据单元, 除非需要定制数据的导入与导出, 
  否则几乎不必编程来读写 HDFS 文件 






## Hadoop 基本文件命令

1. 添加目录和文件
2. 获取文件
3. 删除文件

### 指定文件和目录确切位置的URI

- Hadoop的文件命令既可以与HDFS文件系统交互, 也可以和本地文件系统交互; 
- URI精确地定位一个特定文件或目录的位置, 完整的URI格式为 `scheme//authority/path`
- `scheme` 类似于一个协议, 它可以是 `hdfs` 或 `file` , 分别指定HDFS文件系统或本地文件系统; 
- `authority` 是HDFS中NameNode的主机名; 
- `path` 是文件或目录的路径; 
- 例如, 对于在本地机器的9000端口上, 以标准伪分布式模型运行的HDFS, 访问用户目录/user/chunk中文件example.txt的URI:  `hdfs://localhost:9000/user/chuck/example.txt`
- 大多数设置不需要制定URI中的 `scheme://authority` 部分; 
- 当在本地文件系统和HDFS之间复制文件时, Hadoop中的命令会分别吧本地文件熊作为源和目的, 而不需要制定scheme为file; 
- 对于其他命令, 如果未设置URI中的 `scheme://authority` , 就会采用Hadoop的默认设置; 
   - 假如 `conf/core-site.xml` 文件已经更改为伪分布式配置, 则文件中的 `fs.default.name` 属性应为 `hdfs://localhost:9000` . 在此配置下, URI `hdfs://localhost:9000/user/chuck/example.txt` 可以缩短为 `/user/chuck/example.txt`
   - HDFS默认当前工作目录为 `/user/$USER` , 其中 `$USER` 是登录用户名, 如果作为 `chuck` 登录, 则URI `hdfs://localhost:9000/user/chuck/example.txt` 就缩短为 `example` ; 

### 基本形式

```bash
# method 1
$ hadoop fs -cmd <args>

# method 2
$ hadoop dfs -cmd <args>

# method 3
$ alias hdfs="hadoop dfs"
$ hdfs -cmd <args>
```

其中

* `cmd` : 特定的文件命令
* `<args>` : 一个数目可变的参数 

### 基本文件命令

1.查看帮助

```bash
$ hadoop dfs -help <cmd>
```

2.查看目录

查看某个目录: 

```bash
$ hadoop dfs -ls <path>
```

查看当前目录: 

```bash
hadoop dfs -ls /
```

查看当前目录下的文件和子目录, 类似于 `ls -r` : 

```bash
$ hadoop dfs -lsr /
```

3.创建、删除文件夹

- 删除一般是删除到 `.Trash` 中, 一般有一定的时效清空的, 如果误删可以找回; 

```bash
# 创建
$ hadoop dfs -mkdir </hadoop dir path/dir name>
$ hadoop dfs -mkdir -p </hadoop dir path/dir name>

# 删除
$ hadoop dfs -rmr <hadoop dir path>
```

4.创建、删除空文件

- 删除一般是删除到 `.Trash` 中, 一般有一定的时效清空的, 如果误删可以找回; 

```bash
# 创建
$ hadoop dfs -touchz </hadoop file path/file name>

# 删除
$ hadoop dfs -rm <hadoop file path>
```

5.检索文件

- 复制; 
- 移动或重命名; 
- 从HDFS中下载文件到本地文件系统; 
- 从本地文件系统复制文件到HDFS中; 

```bash
# 复制
$ hadoop dfs -cp <hadoop source file path > <hadoop target file path>
$ hadoop dfs -cp -r <hadoop source path> <hadoop target path>

# 移动、重命名
$ hadoop dfs -mv <hadoop source path> <hadoop target path>

# HDFS => local(从HDFS上把文件或文件夹下载到本地)
$ hadoop dfs -get <hadoop source path> <local target path>
$ hadoop dfs -copyToLocal <hadoop source path> <local target path>

# 将HDFS上一个目录下的所有文件合并成一个文件下载到本地
$ hadoop dfs -getmerge <hadoop dir path> <local file path>

# local => HDFS(上传本地文件或文件夹到HDFS)
$ hadoop dfs -put <local source path> <hadoop target path>
$ hadoop dfs -copyFromLocal <local source path> <hadoop target path>
$ hadoop dfs -moveToLocal <local source path> <hadoop target path>
```

6.查看文件内容

```bash
$ hadoop dfs -cat <hadoop file path>
$ hadoop dfs -text <hadoop file path>

$ hadoop dfs -tail <hadoop file path>
$ hadoop dfs -tail -f <hadoop file path>
```

7.查看文件、文件夹大小

```bash
# 字节为单位展示
$ hadoop dfs -du <hadoop file path>

# GB为单位展示
$ hadoop dfs -du -s -h <hadoop file path>

# 查看文件夹下每个文件大小
$ hadoop dfs -du -s -h <hadoop dir path/*>
```

8.判断文件、目录、大小

```bash
# 检查文件是否存在, 存在返回0
$ hadoop dfs -test -e filename

# 检查文件是否是0字节, 是返回0
$ hadoop dfs -test -z filename

# 检查文件是否是目录, 是返回1, 否则返回0
$ hadoop dfs -test -d filename
```

# MapReduce 程序

## MapReduce 程序通过操作键值对来处理数据

一般形式:

```
map: (K1, V1) -> (K2, V2)
reduce: (K2, list(V2)) -> (K3, V3)
```

高阶视图:

## Hadoop 数据类型


## Mapper

一个类要作为mapper,需要继承MapReduceBase基类,并且实现Mapper接口;
MapReduceBase基类包含了类的构造与解构方法;

- void configure (JobConf job)
   - 该函数提取XML配置文件或应用程序主类中的参数,在数据处理之前调用该函数;
- void close()
   - 作为map任务结束前的最后一个操作,该函数完成所有的结尾工作,如关闭数据库连接,关闭文件等;

Mapper接口负责数据处理阶段. 它采用的形式为 `Mapper<K1, V1, K2, V2>` 的 java 泛型, 
这里的键类和值类分别实现 `WritableComparable` 和 `Writable` 接口. 
Mapper 只有一个方法: `map`,用于处理一个单独的键/值对:

```java
void map(
   K1 key, 
   V1 value,
   OutputCollector<K2, V2> output,
   Reporter reporter
) throws IOException
```

- 该函数处理一个给定的键/值对(K1, V1),生成一个键/值对(K2,V2)的列表(该列表可能为空);
- OutputCollector接收上面的映射过程的输出;
- Reporter可提供对mapper相关附加信息的记录,形成任务进度;

Hadoop提供了一些有用的mapper实现:

   - `IdentityMapper<K, V>`
      - 实现Mapper<K, V, K, V>, 将输入直接映射到输出
   - `InverseMapper<K, V>`
      - 实现Mapper<K, V, K, V>, 反转键/值对
   - `RegexMapper<K>`
      - 实现Mapper<K, Text, Text, LongWritable>,为每个常规表达式的匹配项生成一个(match, 1)对
   - `TokenCountMapper<K>`
      - 实现Mapper<K, Text, Text, LongWritable>,当输入的值为分词时,生成一个(token, 1)对

## Reducer

一个类要作为reducer,需要继承MapReduceBase基类,
允许配置和清理.并且实现Reducer接口;
MapReduceBase基类包含了类的构造与解构方法;

Reducer只有一个方法: `reduce`:

```java
void reduce(
   K2 key,
   Iterator<V2> values,
   OutputCollector<K3, V3> output,
   Reporter reporter,
) throws IOException
```

- 当 reducer 任务接收来自各个 mapper 的输出时,它按照键/值对中的键对输入数据进行排序,
  并将相同键的值归并. 然后调用 reduce() 函数, 并通过迭代处理那些与指定键相关联的值, 
  生成一个(可能为空)的列表 (K3,V3).
- OutputCollecotr 接收 reduce 阶段的输出, 并写入输出文件;
- Reporter 可提供对 reducer 相关附加信息的记录, 形成任务进度;

Hadoop提供了一些基本的reducer实现

- `IdentityReducer<K, V>`
  - 实现Reducer<K, V, K, V>,将输入直接映射到输出
- `LongSumReducer<K>`
  - 实现<K, LongWritable, K, LongWritable>,计算与给定键相对应的所有值的和

## Partitioner: 重定向Mapper的输出

虽然将 Hadoop 程序称为 MapReduce 应用, 但是在 map 和 reduce 两个阶段之间还有一个及其重要的步骤:将 mapper 的结果输出给不同的reducer.这就是partitioner的工作.
一个MapReduce应用需要使用多个reducer,但是,当使用多个reducer时,就需要采取一些办法来确定mapper应该把键值对输出给谁.

- 默认的做法是对键进行散列来确定 reducer.Hadoop通过 `HashPartitioner` 类强制执行这个策略.但有时HashPartitioner会让你出错.
- 一个定制的partitioner只需要实现 `configure()` 和 `getPartition()` 两个函数.

  - configure()将Hadoop对作业的配置应用在partition上;
  - getPartition()返回一个介于0和reduce任务数之间的整数,指向键/值对将要发送到的reducer;

在 map 和 reduce 阶段之间,一个 MapReduce 应用必然从 mapper 任务得到输出结果,
并把这些结果发布给 reducer 任务. 该过程通常被称为洗牌(shuffling), 
因为在单节点上的 mapper 输出可能被送往分布在集群多个节点上的 reducer.

## Combiner: 本地reduce

在许多MapReduce应用场景中,不妨在分发mapper结果之前做一下"本地Reduce".

预定义Mapper和Reducer类的WordCount:

```java
public class WordCount2 {
   public static void main(String[] args) {
      JobClient client = new JobClient();
      JobConf conf = new JobConf(WordCount2.class);

      FileInputFormat.addInputPath(conf, new Path(args[0]));
      FileOutputFormat.setOutputPath(conf, new Path(args[1]));

      conf.setOutputKeyClass(Text.class);
      conf.setOutputValueClass(LongWritable.class);
      conf.setMapperClass(TokenCountMapper.class);
      conf.setCombinerClass(LongSumReducer.class);
      conf.setReducerClass(LongSumReducer.class);

      client.setConf(conf);
      try {
         JobClient.runJob(conf);
      } catch (Exception e) {
         e.printStackTrace();
      }
   }
}
```


# YARN

1.查看帮助

```bash
$ yarn application --help
```

2.查看提交到集群上的所有任务

```bash
$ yarn application -list
```

3.杀死某个任务

```bash
$ yarn application -kill <applicationId>
```


# 参考

* [什么是HDFS](https://mp.weixin.qq.com/s?__biz=MzI4Njg5MDA5NA==&mid=2247486743&idx=1&sn=658d90686b4b7e80d3042f4208bf07eb&chksm=ebd74c16dca0c5009f6e12750306ea55803b6e9d02d21017a429ac4e65bcbd12dbf8c925f1ec&token=1109491988&lang=zh_CN#rd)

