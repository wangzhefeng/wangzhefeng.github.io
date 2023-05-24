---
title: MapReduce
author: 王哲峰
date: '2023-03-13'
slug: mapreduce
categories:
  - hadoop
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

- [MapReduce 程序](#mapreduce-程序)
  - [MapReduce 程序通过操作键值对来处理数据](#mapreduce-程序通过操作键值对来处理数据)
  - [Hadoop 数据类型](#hadoop-数据类型)
  - [Mapper](#mapper)
  - [Reducer](#reducer)
  - [Partitioner: 重定向Mapper的输出](#partitioner-重定向mapper的输出)
  - [Combiner: 本地reduce](#combiner-本地reduce)
</p></details><p></p>

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