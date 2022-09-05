---
title: 玩转 HTML
author: 王哲峰
date: '2021-04-03'
slug: html
categories:
  - html
tags:
  - tool
---

# HTML 介绍

超文本标记语言(HyperText Markup Language, HTML）是构成 Web 世界的一砖一瓦。
它定义了网页内容的含义和结构。

HTML 元素通过“标签”（tag）将文本从文档中引出，标签由在“<”和“>”中包裹的元素名组成，
HTML 标签里的元素名不区分大小写。

# HTML 关键概念

* HTML 元素
  - 开始标签(opening tag)
  - 结束标签(closing tag)
  - 内容(content)
  - 元素(element)
    + 属性(attribute)
* 嵌套元素
* 空元素
* HTML 文档
* 图像
* 标记文本
  - 标题
  - 段落
  - 列表
  - 链接

# HTML 文档

```html
<!DOCTYPE html>
<html>
  <head>
    <meta charset="utf-8">
    <title>测试页面</title>
  </head>
  <body>
    <img src="images/firefox-icon.png" alt="测试图片">
  </body>
</html>
```

# HTML header 中的元数据




# HTML 文字处理

## 基础文字处理

### 标题和段落

* 标签
    - `<h1></h1>`
    - `<h2></h2>`
    - `<h3></h3>`
    - `<h4></h4>`
    - `<h5></h5>`
    - `<h6></h6>`
    - `<p></p>`
* 最佳实践
    - 应该最好只对每个页面使用一次 `<h1>`，这是顶级标题，所有其他标题位于层次结构中的下方
    - 请确保在层次结构中以正确的顺序使用标题。不要使用 `<h3>` 来表示副标题，
      后面跟 `<h2>` 来表示副副标题，这是没有意义的，会导致奇怪的结果
    - 在可用的六个标题级别中，您应该只在每页使用不超过三个，除非您认为有必要使用更多。
      具有许多级别的文档(即较深的标题层次结构)变得难以操作并且难以导航。
      在这种情况下，如果可能，建议将内容分散在多个页面上

### 列表

* 无需列表
    - `<ul><li></li></ul>`
* 有序列表
    - `<ol><li></li></ol>`
* 嵌套列表
    
    ```html
    <ol>
        <li></li>
        <li></li>
        <li>
            <ul>
                <li></li>
                <li></li>
            </ul>
        </li>
    </ol>`
    ```

### 重点强调

* 强调
    - `<em></em>` (emphasis): 斜体
    - `<span class=""></span>` + CSS
* 非常重要
    - `<strong></strong>` (strong importance): 粗体字
    - `<span class=""></span>` + CSS
* 表象元素(presentational elements)
    - 斜体字
        + `i`: 被用来传达传统上用斜体表达的意义：外国文字，分类名称，技术术语，一种思想……
    - 粗体字
        + `b`: 被用来传达传统上用粗体表达的意义：关键字，产品名称，引导句……
    - 下划线
        + `u`: 用来传达传统上用下划线表达的意义：专有名词，拼写错误……

***

**最佳实践**

- 如果没有更合适的元素，那么使用 `<b>`、`<i>` 或 `<u>` 来表达传统上的粗体、斜体或下划线表达的意思是合适的。
  然而，始终拥有可访问性的思维模式是至关重要的。斜体的概念对人们使用屏幕阅读器是没有帮助的，
  对使用其他书写系统而不是拉丁文书写系统的人们也是没有帮助的

***

## 高级文本排版


# HTML 超链接

# HTML 文档和网站结构

HTML 不仅能够定义网页的单独部分(例如“段落”或“图片”)，
还可以使用 **块级元素** (例如 “标题栏”、“导航栏”、“主内容列”)来定义网站中的复合区域。

## HTML 文档的基本组成部分

网页的外观多种多样，但是除了全屏视频或游戏，或艺术作品页面，
或只是结构不当的页面以外，都倾向于使用类似的标准组件:

* 页眉 `<header></header>`
    - 通常横跨于整个页面顶部有一个大标题 和/或 一个标志。 
    这是网站的主要一般信息，通常存在于所有网页
* 导航栏 `<nav></nav>`
    - 指向网站各个主要区段的超链接。通常用菜单按钮、链接或标签页表示。
      类似于标题栏，导航栏通常应在所有网页之间保持一致，否则会让用户感到疑惑，
      甚至无所适从。许多 web 设计人员认为导航栏是标题栏的一部分，而不是独立的组件，
      但这并非绝对；还有人认为，两者独立可以提供更好的 无障碍访问特性，
      因为屏幕阅读器可以更清晰地分辨二者
* 主内容 `<main></main>`
    - 中心的大部分区域是当前网页大多数的独有内容，
      例如视频、文章、地图、新闻等。这些内容是网站的一部分，
      且会因页面而异
* 侧边栏 `<aside></aside>`
    - 一些外围信息、链接、引用、广告等。通常与主内容相关(例如一个新闻页面上，
      侧边栏可能包含作者信息或相关文章链接)，还可能存在其他的重复元素，如辅助导航系统
* 页脚 `<footer></footer>`
    - 横跨页面底部的狭长区域。和标题一样，页脚是放置公共信息(比如版权声明或联系方式)的，
    一般使用较小字体，且通常为次要内容。 还可以通过提供快速访问链接来进行搜索引擎优化(SEO)

HTML 代码中可根据功能来为区段添加标记。可使用元素来无歧义地表示上文所讲的内容区段，
屏幕阅读器等辅助技术可以识别这些元素，并帮助执行“找到主导航 “或”找到主内容“等任务。
为了实现语义化标记，HTML 提供了明确这些区段的专用标签，例如:


* 导航栏 `<nav></nav>`
* 页眉 `<header></header>`
* 主内容 `<main></main>`
    - `<article></article>`
    - `<section></section>`
    - `<div></div>`
* 侧边栏 `<aside></aside>`
* 页脚 `<footer></footer>`


## HTML 文档示例

```html
<!DOCTYPE html>
<html>
    <head>
        <meta charset="utf-8">
        <title>页面</title>
    </head>
    <body>
        <!-- 页眉 -->
        <header> <!-- 本站所有网页的统一主标题 -->
            <h1>Header 1</h1>
        </header>
        
        <!-- 导航栏 -->
        <nav> <!-- 本站统一的导航栏 -->
            <ul>
                <li><a href="#">主页</a></li>
                <!-- 共n个导航栏项目，省略…… -->
            <ul>
            
            <form> <!-- 搜索栏是站点内导航的一个非线性的方式。 -->
                <input type="search" name="q" placeholder="要搜索的内容">
                <input type="submit" value="搜索">
             </form>
        </nav>
        
        <!-- 主内容 -->
        <main> <!-- 网页主体内容 -->
            <article>
                <!-- 此处包含一个 article（一篇文章），内容略…… -->
            </article>
            
            <section></section>
            
            <div></div>
        
        
            <!-- 侧边栏 -->
            <aside> <!-- 侧边栏在主内容右侧 -->
            </aside>
        </main>
        
        <!-- 页脚-->
        <footer> <!-- 本站所有网页的统一页脚 -->
            <p>所有权利</p>
        </footer>
    </body>
</html>
```

## HTML 布局元素

### 主要元素

* `<header></header>`
    - 是简介形式的内容。如果它是 <body> 的子元素，那么就是网站的全局页眉。
      如果它是 <article> 或<section> 的子元素，那么它是这些部分特有的页眉
* `<nav></nav>`
    - 包含页面主导航功能。其中不应包含二级链接等内容
* `<main></main>`
    - 存放每个页面独有的内容。每个页面上只能用一次 <main>，
      且直接位于 <body> 中。最好不要把它嵌套进其它元素
* `<article></article>`
    - 包围的内容即一篇文章，与页面其它部分无关(比如一篇博文)
* `<section></section>`
    - 与 <article> 类似，但 <section> 更适用于组织页面使其按功能(比如迷你地图、一组文章标题和摘要)分块。
       一般的最佳用法是：以标题作为开头；也可以把一篇 <article> 分成若干部分并分别置于不同的 <section> 中，
       也可以把一个区段 <section> 分成若干部分并分别置于不同的 <article> 中，取决于上下文
* `<aside></aside>`
    - 包含一些间接信息(术语条目、作者简介、相关链接，等等)
* `<footer></footer>`
    - 包含了页面的页脚部分

### 无语义元素

对于一些要组织的项目或要包装的内容，现有的语义元素均不能很好对应。
有时候你可能只想将一组元素作为一个单独的实体来修饰来响应单一的用 CSS 或 JavaScript。
为了应对这种情况，HTML 提供了 `<div>` 和 `<span>` 元素。
应配合使用 `class` 属性提供一些标签，使这些元素能易于查询。

- `<div class=""></div>` 是一个块级(block)无语义元素，
  应仅用于找不到更好的块级元素时，或者不想增加特定的意义时
- `<span class=""></span>` 是一个内联的(inline)无语义元素，
  最好只用于无法找到更好的语义元素来包含内容时，或者不想增加特定的含义时

### 换行与水平分割线

* `<br>`
    - 可在段落中进行换行, 是唯一能够生成多个短行结构(例如邮寄地址或诗歌)的元素
* `<hr>`
    - 元素在文档中生成一条水平分割线，表示文本中主题的变化(例如话题或场景的改变)。一般就是一条水平的直线

### 网站规划

在完成页面内容的规划后，一般应按部就班地规划整个网站的内容：

* 要可能带给用户最好的体验
* 需要哪些页面
* 如何排列组合这些页面、
* 如何互相链接等问题不可忽略

这些工作称为信息架构。在大型网站中，大多数规划工作都可以归结于此，
而对于一个只有几个页面的简单网站，规划设计过程可以更简单，更有趣。

下面是一个简单的网站规划步骤

1. 大多数页面会使用一些相同的元素，例如导航菜单以及页脚内容
    - 通用内容
        - 页眉: 标题、Logo
        - 页脚: 联系方式、版权声明
2. 接下来，可为页面结构绘制草图，记录每一块的作用
3. 下面，对于期望添加近站点的所有其他(通用内容以外的)内容展开头脑风暴，直接罗列出来
4. 下一步，试着对这些内容进行分组，这样可以让你了解哪些内容可以放在统一页面上，这种做法和卡片分类非常相似
5. 接下来，试着绘制一个站点地图的草图，使用一个气泡代表网站的一个页面，
   并绘制连线来表示页面间的一般工作流。主页面一般置于中心，
   且链接到其他大多数页面；小型网站的大多数页面都可以从主页的导航栏中链接跳转。
   也可记录下内容的显示方式

# HTML 调试


# HTML 多媒体与嵌入内容

* HTML 中的图片
* HTML 中的视频和音频
* HTML 中的嵌入技术
* HTML 中的矢量图像
* HTML 中的响应式图片

# HTML 表格

## 表格知识点

* 表格标签 `<table></table>`: table
* 表格标题 `<th></th>`: table head
* 表格行标签 `<tr></tr>`: table row
* 单元格标签 `<td></td>`: table data
* 单元格跨越多个列和行: `colspan` 属性、`rowspan` 属性
  - `<th colspan="n"></th>`
  - `<th rowspan="n"></th>`
  - `<td colspan="n"></td>`
  - `<td rowspan="n"></td>`
* 表格样式化
  - 为列提供样式 `<colgroup><col><col style=""></colgroup>`
  
## 表格示例

```html
<table>  <!-- 表格标签 -->
  <colgroup>
    <col>                                           <!-- 第一列样式 -->
    <col style="background-color: yellow" span="">  <!-- 第一列样式 -->
    <col>                                           <!-- 第一列样式 -->
  </colgroup>
  
  <tr>  <!-- 第一行(标题行) -->
    <th colspan="2">Column Header 1</th>
    <th>Column Header 2</th>
  </tr>
  <tr>  <!-- 第二行 -->
    <td>row1 col1</td>
    <td>row1 col2</td>
    <td>row1 col3</td>
  </tr>
  <tr>  <!-- 第三行 -->
    <td>row2 col1</td>
    <td>row2 col2</td>
    <td>row2 col3</td>
  </tr>
  <tr>  <!-- 第四行 -->
    <td>row3 col1</td>
    <td>row3 col2</td>
    <td>row3 col3</td>
  </tr>
</table>
```

## 表格样式化




# HTML 表单





# CORS 处理跨域图片

# CORS 设置属性

# 使用 rel="preload" 预加载页面内容







# 相关教程、文章

- https://developer.mozilla.org/zh-CN/docs/Web/HTML
- https://en.wikipedia.org/wiki/List_of_XML_and_HTML_character_entity_references
- https://html.spec.whatwg.org/multipage/
- https://validator.w3.org/#validate_by_uri
- https://ogp.me/
- https://www.w3.org/International/articles/language-tags/
- https://search.google.com/search-console/welcome?hl=zh-CN&utm_source=wmx&utm_medium=deprecation-pane&utm_content=home
