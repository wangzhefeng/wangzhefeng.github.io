---
title: FastText
author: wangzf
date: '2022-04-05'
slug: fasttext
categories:
  - nlp
tags:
  - model
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

- [fasttext 算法简介](#fasttext-算法简介)
- [fasttext 算法实现](#fasttext-算法实现)
- [词向量模型 FastText](#词向量模型-fasttext)
    - [模型](#模型)
        - [Skip-gram](#skip-gram)
        - [FastText](#fasttext)
    - [模型总结](#模型总结)
- [fastText 库](#fasttext-库)
    - [fastText 是什么](#fasttext-是什么)
    - [fastText 环境依赖](#fasttext-环境依赖)
    - [下载预训练模型](#下载预训练模型)
    - [fastText 工具库构建](#fasttext-工具库构建)
    - [使用 fastText 进行文本分类](#使用-fasttext-进行文本分类)
        - [准备文本数据](#准备文本数据)
        - [构建分类器](#构建分类器)
    - [使用 fastText 进行词表示](#使用-fasttext-进行词表示)
        - [获取文本语料数据](#获取文本语料数据)
        - [训练词向量模型](#训练词向量模型)
        - [使用 CBOW 训练词向量模型](#使用-cbow-训练词向量模型)
- [参考](#参考)
</p></details><p></p>

# fasttext 算法简介

fasttext 的模型与 CBOW 类似，实际上，fasttext 的确是由 CBOW 演变而来的。
CBOW 预测上下文的中间词，fasttext 预测文本标签。与 Word2Vec 算法的衍生物相同，
稠密词向量也是训练神经网路的过程中得到的

![img](images/fasttext.png)

1. fasttext 的输入是一段词的序列，即一篇文章或一句话，输出是这段词序列属于某个类别的概率，所以，fasttext 是用来做文本分类任务的
2. fasttext 中采用层级 Softmax 做分类，这与 CBOW 相同。fasttext 算法中还考虑了词的顺序问题，即采用 N-gram，与之前介绍的离散表示法相同，如:
    - 今天天气非常不错，Bi-gram 的表示就是:今天、天天、天气、气非、非常、常不、不错

fasttext 做文本分类对文本的存储方式有要求:

```
__label__1, It is a nice day.
__label__2, I am fine, thank you.
__label__3, I like play football.
```

其中:

* `__label__`:为实际类别的前缀，也可以自己定义

# fasttext 算法实现

GitHub：

* https://github.com/facebookresearch/fastText

示例:

```python
import fasttext

classifier = fasttext.supervised(
    input_file, 
    output, 
    label_prefix = "__label__"
)
result = classifier.test(test_file)
print(result.precision, result.recall)
```

其中:

* `input_file`：是已经按照上面的格式要求做好的训练集 txt
* `output`：后缀为 `.model`，是保存的二进制文件
* `label_prefix`：可以自定类别前缀


# 词向量模型 FastText

FastText 是由 Facebook 提出的，一般有两个含义：

* 第一个是指用于句子分类的 FastText
* 第二个则是指用于获取词向量的 FastText

这里主要介绍用于获取词向量的 FastText 的内容及含义。
FastText 可以认为是从 Word2Vec 进化而来的，其结构类似。
其主要改进在于引入了 sub-word(也表述为 Character N-Grams)，
能够有效的解决 OOV 问题。

## 模型

### Skip-gram

FastText 算是 Word2Vec 的一个扩展，因此在了解 FastText 之前，先回顾下 Skip-gram。
给定一个词典其大小为 `$W$`，单词的索引为： `$w \in \{1, 2, \cdots, W\}$`，
学习的目标是每个单词 `$w$` 的向量表征。其目标为极大化如下的对数似然函数：

`$$\sum_{t=1}^{T}\sum_{c \in C_{t}}\log p(w_{c}|w_{t})$$`

其中，`$C_{t}$` 为单词 `$w_{t}$` 的上下文单词索引集合。

在一系列优化之后，可以得到最终的目标函数为：

`$$\underset{x_{\theta}}{\text{argmax}}\sum_{(w,c)\in D} \log \sigma(u_{c}, u_{w}) +\sum_{(w, c) \in \bar{D}} \log \sigma(-u_{c}, v_{w})$$`

其中，`$D$` 是正样本，`$\bar{D}$` 是负采样后的负样本。

### FastText

由于 Skip-gram 模型忽略了单词的内部结构。FastText 中最大的改进就是拆词，
将 word 拆成 sub-word，特殊的约束符号 `<` 和 `>` 分别加在单词前和单词后，
使得能够将前缀和后缀与其他字符序列区分开来。

原文中以 `where` 为例，如果 Character N-Grams 中的 `N` 为 3，则 `where` 可以拆成：

```
<wh, whe, her, ere, re>
```

另外，`where` 这个词本身也会包含在 `where` 的 Character N-Grams 之中。
因此 `where` 的表征则会变成其 N-Grams 中的子词和本身的表征之和:

```
<wh, whe, her, ere, re> + <where>
```

因此模型要学习的不仅是 `<where>`，另外还需要学习 `wh, whe, her, ere, re` 的向量，
最后通过求和得到 `where` 最终的表征向量：

`$$s(w, c) = \sum_{g \in G_{w}}z_{g}^{t}v_{c}$$`

这个简单的模型允许跨单词共享表示，从而允许学习罕见单词的可靠表示。

由于 sub-word 带来的词表远大于 word 带来的此表，因此会带来非常大的存储消耗，
所以文中使用了哈希函数对文本进行处理并存储。

## 模型总结

FastText 是从 Word2Vec 进化而来的，最大的改变就是拆词，
将 word 拆成 Character N-Grams。此时带来的两个优点：

* 解决了 OOV 的问题；
* 对罕见单词能够学习到较为可靠的表示。

# fastText 库

## fastText 是什么

`fastText` is a library for efficient **text classification** and **representation learning**.

FastText is an open-source, free, lightweight library that allows users to learn text representations 
and text classifiers. It works on standard, generic hardware. 
Models can later be reduced in size to even fit on mobile devices.

## fastText 环境依赖

* 计算机系统
    - macOS
    - Linux
* C++11 编译器
    - `(gcc-4.6.3 or newer)` or `(clang-3.3 or newer)`
    - `make`
* Python 依赖
    - `>=python 2.6`
    - `numpy`
    - `scipy`

## 下载预训练模型

* English word vectors: Pre-trained on English webcrawl and Wikipedia.
* Multi-lingual word vectors: Pre-trained models for 157 different languages.

## fastText 工具库构建

1. 构建 fastText 为一个命令行工具(CLT)

```bash
$ git clone https://github.com/facebookresearch/fastText.git
$ cd fastText
$ make
```

或者:

```bash
$ wget https://github.com/facebookresearch/fastText/archive/v0.9.2.zip
$ unzip v0.9.2.zip
$ cd fastText-0.9.2
$ make
```

2. 构建 fastText 为一个 Python 模块

```bash
$ git clone https://github.com/facebookresearch/fastText.git
$ cd fastText
$ sudo pip install .
# or
$ sudp python setup.py install
```

或者:

```bash
$ wget https://github.com/facebookresearch/fastText/archive/v0.9.2.zip
$ unzip v0.9.2.zip
$ cd fastText-0.9.2
$ pip install
```

```python
>>> import fasttext
>>>
```

3. 获取帮助:

```bash
./fasttext
```

```python
>>> import fasttext
>>> help(fasttext.FastText)
```

## 使用 fastText 进行文本分类

文本分类可以应用在许多方面:

* 垃圾邮件检测(spam detection)
* 情感分析(sentiment analysis)
* 智能回复(smart replies)

### 准备文本数据

数据来源：

* https://cooking.stackexchange.com/

数据描述：

* building a classifier to automatically recognize the topic of a stackexchange question about cooking

数据下载：

```bash
$ wget https://dl.fbaipublicfiles.com/fasttext/data/cooking.stackexchange.tar.gz
$ tar xvzf cooking.stackexchange.tar.gz
$ head cooking.stackexchange.txt
$ wc cooking.stackexchange.txt
```

数据格式预览：

| Label                                                                         |   document | 
|-------------------------------------------------------------------------------|---------------------------------------------------------------|
| __label__sauce __label__cheese                                                |  How much does potato starch affect a cheese sauce recipe? | 
| __label__food-safety __label__acidity                                         |  Dangerous pathogens capable of growing in acidic environments | 
| __label__cast-iron __label__stove                                             |  How do I cover up the white spots on my cast iron stove? | 
| __label__restaurant                                                           |  Michelin Three Star Restaurant; but if the chef is not there | 
| __label__knife-skills __label__dicing                                         |  Without knife skills, how can I quickly and accurately dice vegetables? | 
| __label__storage-method __label__equipment __label__bread                     |  What's the purpose of a bread box? | 
| __label__baking __label__food-safety __label__substitutions __label__peanuts  |  how to seperate peanut oil from roasted peanuts at home? | 
| __label__chocolate                                                            |  American equivalent for British chocolate terms | 
| __label__baking __label__oven __label__convection                             |  Fan bake vs bake | 
| __label__sauce __label__storage-lifetime __label__acidity __label__mayonnaise |  Regulation and balancing of readymade packed mayonnaise and other sauces | 

数据集分割：

* Training dataset

```bash
$ head -n 12404 cooking.stackexchange.txt > cooking.train
$ wc cooking.train
```

* validation dataset

```bash
$ tail -n 3000 cooking.stackexchange.txt > cooking.valid
$ wc cooking.valid
```

### 构建分类器

- 基本模型

```python
import fasttext

# 模型训练    
model = fasttext.train_supervised(input = "cooking.train")

# 模型保存
model.save_model("model_cooking.bin")

# 模型测试
model.predict("Which baking dish is best to bake a banana bread ?")
model.predict("Why not put knives in the dishwater?")
model.test("cooking.valid")
model.test("cooking.valid", k = 5)
```

- precision 和 recall

```python
# Top 5 预测标签，用来计算 precision 和 recall
model.predict("Why not put knives in the dishwater?", k = 5)
```

- 增强模型预测能力

- (2)数据预处理

    - 将单词中的大写字母转换为小写字母
    - 处理标点符号

```bash
$ cat cooking.stackexchange.txt | sed -e "s/\([.\!?,'/()]\)/ \1 /g" | tr "[:upper:]" "[:lower:]" > cooking.preprocessed.txt
$ head -n 12404 cooking.preprocessed.txt > cooking_preprocessed.train
$ tail -n 3000 cooking.preprocessed.txt > cooking_preprocessed.valid
```

```python
import fasttext

model = fasttext.train_supervised(input = "cooking_preprocessed.train")
model.test("cooking_preprocessed.valid")
```

- (2)增多 epochs

```python
import fasttext

model = fasttext.train_supervised(input = "cooking.train", epoch = 25)
model.test("cooking.valid")
```

- (3)增大 learning_rate

```python
import fasttext

model = fasttext.train_supervised(input = "cooking.train", lr = 1.0)
model.test("cooking.valid")
```

```python
import fasttext

model = fasttext.train_supervised(input = "cooking.train", lr = 1.0, epoch = 25)
model.test("cooking.valid")
```

- (4)word n-grams

```python
model = fasttext.train_supervised(
    input = "cooking.train", 
    lr = 1.0, 
    epoch = 25, 
    wordNgrams = 2
)
model.test("cooking.valid")
```

- Bigram


- Scaling thing up

```python
model = fasttext.train_supervised(
    input = "cooking.train", 
    lr = 1.0, 
    epoch = 25, 
    wordNgrams = 2, 
    bucket = 200000, 
    dim = 50,
    loss = "hs"
)
```

- 多标签分类(Multi-label classification)

```python
import fasttext

model = fasttext.train_supervised(
    input = "cooking.train", 
    lr = 0.5, 
    epoch = 25, 
    wordNgrams = 2, 
    bucket = 200000, 
    dim = 50, 
    loss = "ova"
)

model.predict(
    "Which baking dish is best to bake a banana bread ?", 
    k = -1,
    threshold = 0.5
)
model.test("cooking.valid", k = -1)
```

## 使用 fastText 进行词表示

> word representations

### 获取文本语料数据

为了计算 word vectors，需要一个大型的文本语料(text corpus)。
基于这个文本语料，word vectors 将会学习到到不同的信息。

下载文本语料数据：

```bash
$ wget https://dumps.wikimedia.org/enwiki/latest/enwiki-latest-pages-articles.xml.bz2
```

为了节省时间，选择一个较小的语料库：

```bash
$ mkdir data
$ wget -c http://mattmahoney.net/dc/enwik9.zip -P data
$ unzip data/enwik9.zip -d data
```

原始数据中包含了大量的 HTML/XML 数据。所以需要使用 fastText 的 `wikifil.pl` 脚本进行处理。

```bash
$ perl wikifil.pl data/enwik9 > data/fil9
```

运行一下命令检查文件：

```bash
$ head -c 80 data/fil9
```

文本已经很好地预处理过，可以用来学习我们的词向量。

### 训练词向量模型

在文本语料数据上学习词向量：

```python
import fasttext

model = fasttext.train_unsupervised("data/fil9")
```

一旦训练完成，`model` 变量包含关于训练模型的信息，并可用于查询：

```python
# 它返回词汇表中的所有单词，按频率递减排序
model.words
```

我们可以通过以下方式获取单词向量：

```python
model.get_word_vector("the")
```

保存模型：

```python
model.save_model("result/fil9.bin")
```

重新加载模型：

```python
import fasttext

model = fasttext.load_model("result/fil9.bin")
```

### 使用 CBOW 训练词向量模型

fastText 提供了两种模型来计算单词表示：skipgram 和 cbow（continuous bag of words）。

skipgram 模型通过附近的单词学习预测目标单词。而 CBOW 模型根据目标单词的上下文来预测目标单词。
上下文表示为围绕目标单词固定大小窗口内包含的单词的集合。

```python
import fasttext

model = fasttext.train_unsupervised("data/fil9", "cbow")
```

在实践中，skipgram 模型在处理子词信息时比 cbow 模型表现更好。

# 参考

* [fastText 官网](https://fasttext.cc/)
* [fastText](https://fasttext.cc/docs/en/supervised-tutorial.html)
* [详解词向量模型 FastText](https://zhuanlan.zhihu.com/p/556223472)
