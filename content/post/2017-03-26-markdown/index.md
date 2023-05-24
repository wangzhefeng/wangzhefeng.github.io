---
title: RMarkdown、Rmd、Markdown
author: 王哲峰
date: '2017-03-26'
slug: markdown
categories:
  - blog
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

- [Markdown](#markdown)
  - [TODO list](#todo-list)
  - [上标、下标](#上标下标)
  - [文档内引用](#文档内引用)
  - [脚注尾注](#脚注尾注)
  - [插入表情(Emogi)](#插入表情emogi)
  - [插入视屏](#插入视屏)
  - [插入图片](#插入图片)
  - [生成目录](#生成目录)
    - [TOC](#toc)
    - [DocToc](#doctoc)
    - [blogdown](#blogdown)
    - [html](#html)
  - [插入Note、Important](#插入noteimportant)
  - [代码高亮](#代码高亮)
  - [Blog Markdown](#blog-markdown)
    - [数学公式](#数学公式)
  - [插入 HTML](#插入-html)
- [RMarkdwon](#rmarkdwon)
- [参考资料](#参考资料)
</p></details><p></p>


# Markdown

Markdown 的维基百科[^Markdown维基百科]介绍是这样的：

> Markdown是一种轻量级标记语言，创始人为约翰·格鲁伯。
  它允许人们使用易读易写的纯文本格式编写文档，
  然后转换成有效的XHTML（或者HTML）文档。
  这种语言吸收了很多在电子邮件中已有的纯文本标记的特性。
> 
> 由于Markdown的轻量化、易读易写特性，并且对于图片，
  图表、数学式都有支持，目前许多网站都广泛使用 Markdown 
  来撰写帮助文档或是用于论坛上发表消息。如GitHub、Reddit、
  Diaspora、Stack Exchange、OpenStreetMap 、SourceForge、
  简书等，甚至还能被用来撰写电子书。

[^Markdown维基百科]: 百度百科也可以看看


## TODO list

Markdown 的 To-Do List 的内容是由无序列表，复选框，图标几种功能的组合

- 复选框(未选中)语法

```markdown
# 未选中复选框列表

* [ ] todo item 1
* [ ] todo item 2
* [ ] todo item 3
```

- 未选中复选框列表效果展示

  * [ ] todo item 1
  * [ ] todo item 2
  * [ ] todo item 3

- 复选框(选中)语法

```markdown
# 选中复选框列表

* [x] todo item 1
* [x] todo item 2
* [x] todo item 3
```

- 选中复选框列表效果展示

  * [x] todo item 1
  * [x] todo item 2
  * [x] todo item 3


## 上标、下标

上标、下标是一种文字的特殊写法，常用于化学式、数学公式、引用文字的脚注等。

Markdown 的原生语法不支持上、下标的写法。需要通过 html 标签实现此类效果。
在 Markdown 中，我们可以通过 `<sub>` 和 `<sup>` 标签来实现上标和下标

- 语法示例

```markdown
a<sup>2</sup> + b<sup>2</sup> =c<sup>2</sup>
```

- 效果展示

  - a<sup>2</sup> + b<sup>2</sup> =c<sup>2</sup>


## 文档内引用

```markdown
[text](#header-label)
[text](path)
```


## 脚注尾注

脚注和尾注都是对文章的补充说明。

* 脚注通常与被注释内容出现在同一页，并位于该页面的最下方，一般用来解释专有名词、数据来源等
* 尾注通常出现在文章的最后一页，写在文章全部正文之后，一般用来列明引用的文章列表等

1. 添加引用的描述

要增加脚注/尾注，首先需要在文章的适当位置增加引用的描述

- 脚注声明语法

```markdown
[^引用ID]: 说明文字
```

2. 引用部位添加引用注释

在需要增加引用标记的内容后面增加引用注释

- 脚注引用注释语法

```markdown
`[^引用ID]`
```

3. 完整示例

```markdown
<!-- 脚本引用 -->
- 这里是一个脚注[^脚注ID1]
- 这里是一个脚注[^脚注ID2]

<!-- 引用注释 -->
- [脚注ID1]: 此处是 **脚注** 的 *文本内容*
- [脚注ID2]: 此处是 **脚注** 的 *文本内容*
```

<!-- 脚本引用 -->
- 这里是一个脚注[^脚注ID1]
- 这里是一个脚注[^脚注ID2]

<!-- 引用注释 -->
[^脚注ID1]: 此处是 **脚注** 的 *文本内容*
[^脚注ID2]: 此处是 **脚注** 的 *文本内容*

## 插入表情(Emogi)

* 表情

  - \:smile\: => :smile:
  - \:joy\: => :joy:

* 十二星座都可以

  - \:aries\: => :aries:
  - \:taurus\: => :taurus:
  - \:gemini\: => :gemini:
  - \:cancer\: => :cancer:
  - \:le\o => :leo:
  - \:virgo\: => :virgo:
  - \:libra\: => :libra:
  - \:scorpius\: => :scorpius:
  - \:sagittarius\: => :sagittarius:
  - \:capricorn\: => :capricorn:
  - \:aquarius\: => :aquarius:
  - \:pisces\: => :pisces:
  - \:ophiuchus\: => :ophiuchus:
  - \:six_pointed_star\: => :six_pointed_star:

* 钟表时间

  - \:clock930\: => :clock930:

* 其他好用的

  - :white_check_mark:
  - :heavy_check_mark:
  - :heavy_multiplication_x:
  - :black_square_button:
  - :heavy_exclamation_mark:
  - :x:
  - :bangbang:
  - :link:
  - :recycle:
  - :negative_squared_cross_mark:
  - :cn:
  - :mag:
  - :octocat:


## 插入视屏

- HTML script

```markdown
<div class="plyr__video-embed" id="player" width="100%">
  <iframe
    src="https://www.youtube.com/embed/bTqVqk7FSmY?origin=https://plyr.io&amp;iv_load_policy=3&amp;modestbranding=1&amp;playsinline=1&amp;showinfo=0&amp;rel=0&amp;enablejsapi=1"
    allowfullscreen
    allowtransparency
    allow="autoplay"
    width="100%"
    height="400px"
  ></iframe>
</div>
```

<div class="plyr__video-embed" id="player" width="100%">
  <iframe
    src="https://www.youtube.com/embed/bTqVqk7FSmY?origin=https://plyr.io&amp;iv_load_policy=3&amp;modestbranding=1&amp;playsinline=1&amp;showinfo=0&amp;rel=0&amp;enablejsapi=1"
    allowfullscreen
    allowtransparency
    allow="autoplay"
    width="100%"
    height="400px"
  ></iframe>
</div>

- 图片加视频连接

```markdown
[![Machine Learning Meets Fashion](images/ae143b2d.png)](https://youtu.be/RJudqel8DVA)
```

[![Machine Learning Meets Fashion](https://github.com/zalandoresearch/fashion-mnist/blob/master/doc/img/ae143b2d.png?raw=true)](https://youtu.be/RJudqel8DVA)


## 插入图片

- markdown 方法

```markdown
![text](/path/image.png)
```


- html 方法

```markdown
<image src="/path/image.png" width=100%>
<image src="/path/image.png" width=50%><image src="/path/image.png" width=50%>
```


## 生成目录

### TOC

```markdown
[TOC]

# Header 1

## Header 2

### Header 3
```

### DocToc

```bash
npm install doctoc -g
```

```bash
cd project
doctoc file.md
```

### blogdown

```yaml/toml
---
title: R Markdown 与 Rmd 与 Markdown 的测试
author: 王哲峰
date: '2022-03-26'
slug: rmarkdown-rmd-markdown
categories:
  - Markdown
tags:
  - note
output:
  blogdown::html_page:
    toc: true
    fig_width: 6
    dev: "svg"
---
```

### html

```markdown
<details><summary>Table of Contents</summary><p>

* [Header1-1](#header1-label)
* [Header1-2](#header2-label)
  - [Header2-1](##header2-1-label)
</p></details><p></p>
```

## 插入Note、Important

```markdown
* method 1

***
**Note**
This is a note.
***

* method 2

> **_Note:_** 
> 
> The note content.

* method 3

<div class="warning" style='padding:0.1em; background-color:#E9D8FD; color:#69337A'>
<span>
<p style='margin-top:1em; text-align:center'>
<b>On the importance of sentence length</b></p>
<p style='margin-left:1em;'>
This is a note.<br><br>
This ia another note.
</p>
<p style='margin-bottom:1em; margin-right:1em; text-align:right; font-family:Georgia'> <b>- Gary Provost</b> <i>(100 Ways to Improve Your Writing, 1985)</i>
</p></span>
</div>

* method 3

| | |
|-|-|
|`NOTE` | This is a note.|

* method 4

|`NOTE` | This is a note.|
|-|-|

* method 5

| | |
|-|-|
|`NOTE` | This is a note.|

* method 6

<div class="warning" style='background-color:#E9D8FD; color: #69337A; border-left: solid #805AD5 4px; border-radius: 4px; padding:0.7em;'>
<span>
<p style='margin-top:1em; text-align:center'>
<b>On the importance of sentence length</b></p>
<p style='margin-left:1em;'>
This is a note.<br>
This ia another note.
</p>
<p style='margin-bottom:1em; margin-right:1em; text-align:right; font-family:Georgia'> <b>- Gary Provost</b> <i>(100 Ways to Improve Your Writing, 1985)</i>
</p></span>
</div>
```

* method 1

***
**Note**
This is a note.
***

* method 2

> **_Note:_** 
> 
> This is a note.

* method 3

<div class="warning" style='padding:0.1em; background-color:#E9D8FD; color:#69337A'>
    <span>
        <p style='margin-top:1em; text-align:center'>
            <b>On the importance of sentence length</b>
        </p>
        <p style='margin-left:1em;'>
            This is a note.<br>
            This ia another note.
        </p>
        <p style='margin-bottom:1em; margin-right:1em; text-align:right; font-family:Georgia'> 
          <b>- Gary Provost</b> 
          <i>(100 Ways to Improve Your Writing, 1985)</i>
        </p>
    </span>
</div>

* method 3

| | |
|-|-|
|`NOTE` | This is a note.|

* method 4

|`NOTE` | This is a note.|
|-|-|

* method 5

| | |
|-|-|
|`NOTE` | This is a note.|

* method 6

<div class="warning" style='background-color:#E9D8FD; color: #69337A; border-left: solid #805AD5 4px; border-radius: 4px; padding:0.7em;'>
    <span>
        <p style='margin-top:1em; text-align:left'>
            <b>On the importance of sentence length</b>
        </p>
        <p style='margin-left:1em;'>
            This is a note.<br>
            This ia another note.
        </p>
        <p style='margin-bottom:1em; margin-right:1em; text-align:right; font-family:Georgia'> 
            <b>- Gary Provost</b> 
            <i>(100 Ways to Improve Your Writing, 1985)</i>
        </p>
    </span>
</div>


## 代码高亮

Markdown 的代码高亮是对代码块语法的扩展。
即通过对代码块进行语法标注，对其在渲染输出时匹配不同的样式。

代码高亮模块是 Markdown 的一种扩展语法，通常通过第三方的高亮插件完成支持。
常见的高亮插件实现如 Typora 使用的 codemirror，
还有在网页中应用较多的 highlightjs 等。
大部分的 Markdown 编辑器或者编辑环境都已经集成好，
只要按照其语法规范，在文档完成渲染后即可得到带有高亮样式的代码块了。

- https://highlightjs.org/


## Blog Markdown

### 数学公式

```markdown
<!-- markdown/RMarkdown -->
`$a^{2}+b^{2} = c^{2}$`

<!-- RMarkdown -->
# markdown/RMarkdown
$a^{2}+b^{2} = c^{2}$
```

- markdown/RMarkdown

  - `$a^{2}+b^{2} = c^{2}$`

- RMarkdown

  - $a^{2}+b^{2} = c^{2}$

## 插入 HTML

<iframe allowtransparency="yes" frameborder="0" width="100%" height="300" src="click.html"></iframe>

# RMarkdwon

- https://rpubs.com/Wangzf/RMarkdown

# 参考资料

- [Markdown 维基百科](https://zh.wikipedia.org/wiki/Markdown) | [:link:](https://zh.wikipedia.org/wiki/Markdown)
- [Markdown 官网](https://daringfireball.net/projects/markdown/syntax)
- [插入表情](https://www.webfx.com/tools/emoji-cheat-sheet/)
- [插入 Note](https://stackoverflow.com/questions/25654845/how-can-i-create-a-text-box-for-a-note-in-markdown)
- [插入目录 DocToc](https://github.com/thlorenz/doctoc)

