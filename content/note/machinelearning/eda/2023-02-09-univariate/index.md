---
title: 单变量数据探索分析
author: 王哲峰
date: '2023-02-09'
slug: univariate
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

- [数值变量](#数值变量)
  - [基于数值观察](#基于数值观察)
  - [可视化](#可视化)
- [类别变量](#类别变量)
  - [基于数值观察](#基于数值观察-1)
  - [可视化](#可视化-1)
- [时间变量](#时间变量)
- [字符串](#字符串)
  - [wordcloud 可视化](#wordcloud-可视化)
  - [scattertext 可视化](#scattertext-可视化)
- [图像](#图像)
- [参考](#参考)
</p></details><p></p>

针对单变量观测分析，可以将数据按照类型拆分成数值型、类别型、时间类型、字符串(`object` 型)、图像

# 数值变量

关于数值变量分析，一般会有下面几点：

1. 是否存在异常值
2. 数据的整体分布情况

```python
import seaborn as sns
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

plt.figure(figsize = (10, 6))
%matplotlib inline

df = pd.read_csv("")
df.head()
```

## 基于数值观察

可以直接通过 pandas 的 `describe` 函数去观测数值数据的分位数，
基于分位数判断这些数据是否符合预期

```python
df[""].describe(percentiles = np.array(list(range(10))) * 0.1)
```

## 可视化

```python
plt.figure(figsize = (10, 8))

ax = plt.subplot(221)
sns.boxplot(data = df[""], ax = ax)
plt.ylabel("")

plt.subplot(222)
sns.distplot(data = df[""])
plt.xlabel("")
```

# 类别变量

关于类别变量，一般可以分为两类：

* 一类为无相对大小的类别变量，例如花瓣的形状、颜色
* 另外一类为有相对大小的类别变量，例如年龄分段等

而关于单个类别变量的分析，需要重点观测其分布情况，包括：

* 类别变量的 nunique 情况
* 类别变量在数据集的占比分布，比如：出现次数、占比

## 基于数值观察

使用 pandas 的 `nunique`、`value_counts` 等函数进行观察即可

```python
df[""].nunique()
df[""].value_counts()
df[""].value_counts(normalize = True)
```

## 可视化

同样地，可以通过可视化的方式来看数据的分布，此处我们可以直接用柱状图图来进行观测。
在数据的 nunique 值非常大的时候，一般会选择摘取出现次数最多的 Top N 个数据进行观测

```python
plt.figure(figsize = (10, 8))

ax = plt.subplot(221)
sns.countplot(x = "", data = df)
plt.xlabel("")

plt.subplot(222)
df[""].value_counts(normalize = True).plot(kind = "bar")
plt.xlabel("")
```

# 时间变量

关于时间类型的数据，需要重点观测下面的几点内容：

* 数据在每个时间段的频次，可以当做类别变量分析
* 尤其需要注意一些突变的点，这些点一般都会存在某些特殊的信息

因为时间变量的特殊性，存在非常多特殊的问题需要思考：

* 拆分为月/天/日/小时，然后当做类别变量进行观测
* 抽取周期性信息，例如：工作日七天，然后观察周中和周末的分布信息

```python
dts = np.random.randint(low = 1, high = 30, size = df.shape[0])
df['dt'] = dts
df['dt'] = '2020-11-' + df['dt'].astype(str)
df['dt'] = pd.to_datetime(df['dt'])

df['day'] = df['dt'].apply(lambda x: x.day)
df['week_day'] = df['dt'].apply(lambda x: x.day % 7) 
```

# 字符串

字符串类型的数据是最为复杂的一类数据，所有的数值、类别、时间等信息全部可以设置为字符串类型，
因而第一时间，需要判断字符串是不是真正意义上的字符串类型：

* 当前的字符串数据是不是类型误标记了，它本质可能是其它类型的变量，例如时间类型的变量等等
* 如果是简单的字符串，例如国家等信息，就可以将其作为无相对大小的类别变量进行分析

此处，重点介绍稍微复杂的文本类的字符串的分析。因为文本信息非常多，里面包含的信息也十分复杂，
这从早期的文本相关的数据竞赛的特征工程技巧中就可以发现，典型的包括：

* 标点符号的信息，例如感叹号、问号
* 错误的单词的个数
* 情感词汇的信息
* 等

## wordcloud 可视化

```python
# !pip install wordcloud

from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator  

plt.figure(figsize = [10, 8])
text = 'I love dog dog dog, mikey. My brother likes dog too. But his brother likes cats.' 

wordcloud = WordCloud().generate(text)  
plt.imshow(wordcloud, interpolation = 'bilinear')
plt.axis("off")
plt.show() 
```

## scattertext 可视化

```python
# !pip install scattertext

import scattertext as st

df = st.SampleCorpora.ConventionData2012.get_data().assign(
    parse = lambda df: df.text.apply(st.whitespace_nlp_with_sentences)
)

corpus = st.CorpusFromParsedDocuments(
    df, 
    category_col = 'party', 
    parsed_col = 'parse'
).build().get_unigram_corpus().compact(st.AssociationCompactor(2000))

html = st.produce_scattertext_explorer(
    corpus,
    category = 'democrat', 
    category_name = 'Democratic', 
    not_category_name = 'Republican',
    minimum_term_frequency = 0, 
    pmi_threshold_coefficient = 0,
    width_in_pixels = 1000, 
    metadata = corpus.get_df()['speaker'],
    transform = st.Scalers.dense_rank
)
open('./demo_compact.html', 'w').write(html)
```

# 图像

图像数据和文本数据都属于特殊的数据，暂时将其放在单变量一起介绍。
图像的数据可以直接进行可视化，有些朋友会加入不同的算子或者预处理之类的，
然后再进行可视化看效果。而关于图像的可视化方式有非常多，
可以基于 PIL，基于 matplotlb.image，cv2，scipy，skimage 等等，
下面是一个基于 PIL 的图像可视化案例

```python
from PIL import Image
import numpy as np 

img_path = './pic/picture.jpg'
I = np.array(Image.open(path) )
plt.imshow(I) 
```

# 参考

* [Text Data Visualization in Python]()
* [Generating WordClouds in Python]()
* [word_cloud](https://github.com/amueller/word_cloud)
* [scattertext](https://github.com/JasonKessler/scattertext)
* [text-visualization](https://kanoki.org/2019/03/17/text-data-visualization-in-python/)
* [数据探索分析-单变量数据分析](https://mp.weixin.qq.com/s?__biz=Mzk0NDE5Nzg1Ng==&mid=2247493208&idx=1&sn=0b78caad1b06fe2b18da50c84cea4f23&chksm=c32affd7f45d76c19af81f32e2bc730e33f436c956ffa024098d9ad6857a617c90eddd9ba307&cur_album_id=1701045138849906691&scene=189#wechat_redirect)

