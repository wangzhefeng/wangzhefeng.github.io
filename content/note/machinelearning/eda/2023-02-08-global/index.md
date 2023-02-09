---
title: 全局数据探索分析
author: 王哲峰
date: '2023-02-08'
slug: global
categories:
  - data analysis
tags:
  - machinelearning
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

- [数据整体观测](#数据整体观测)
- [数据类型概览](#数据类型概览)
- [数据大小概览](#数据大小概览)
- [数据整体缺失情况观测](#数据整体缺失情况观测)
  - [简单数值观测](#简单数值观测)
  - [可视化观测](#可视化观测)
- [字段 nunique 观测](#字段-nunique-观测)
- [参考](#参考)
</p></details><p></p>

# 数据整体观测

先对数据进行简单的观测，对数据有一个简单的了解

```python
import pandas as pd

df = pd.read_csv("sample.csv", header = None)
df[1] = df[1].astype(str)
df[10] = df[10].astype(float)
df.head()
```

# 数据类型概览

通过 `.dtypes` 属性了解所有字段的含义，这么做就可以知道每个数据的类型，
进一步加深对于数据的理解

```python
df.dtypes
```

# 数据大小概览

通过 pandas 的 `info` 函数，可以拿到每个字段的样本个数，以及数据集所占据的空间大小，
对数据的大小有了一定的了解之后，就需要考虑：

* 如果数据很大的话，是否需要性能更强的服务器
* 如果测试数据集太小的话，需要考虑可能会出现模型波动非常大的情况，不太会产生较好的效果

```python
df.info()
```

# 数据整体缺失情况观测

## 简单数值观测

一般使用下面的方法来观测每个字段的缺失情况

```python
df.isnull().sum(axis = 0)
```

## 可视化观测

全局可视化：数据集的缺失情况

```python
import pandas as pd
import missingno as msno

df = pd.read_csv("kamyr-digester.csv")
msno.matrix(df)
```

全局可视化：整体缺失情况

```python
msno.bar(df)
```

# 字段 nunique 观测

还有一个在全局探索分析时需要重点观测的就是 nunique 分布，通过 nunique 的观测，
可以知道每个字段中不同的个数。就可以直接对 nunique 为 1 的字段直接删除，
因为这些字段是没有任何信息的

```python
df.nunique()
```

# 参考

* [数据探索分析-全局数据探索分析](https://mp.weixin.qq.com/s?__biz=Mzk0NDE5Nzg1Ng==&mid=2247493165&idx=1&sn=6ac408ad026e963cd1f68c2fb94d3b59&chksm=c32affa2f45d76b45815114992e9aa51f735fbccef89dae9d0fd65e5928a011c639bb1436772&scene=21#wechat_redirect)

