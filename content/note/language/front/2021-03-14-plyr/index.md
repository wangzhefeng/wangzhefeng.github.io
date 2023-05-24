---
title: 视屏播放库 Plyr
author: 王哲峰
date: '2021-03-14'
slug: plyr
categories:
  - javascript
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
img {
    pointer-events: none;
}
</style>

# 简介

最近看到一个前端的视屏播放器库，这个库的名字叫做 Plyr(Player)，主页大概长这样：

![plyr.io](images/plyrio.jpg)

据介绍，Plyr 是一个简单的、轻量级的、可访问和可定制的 HTML5、YouTube 和 Vimeo 媒体播放器，支持大多数的现代浏览器。

- 官方网站：https://plyr.io/
- GiuHub地址：https://github.com/sampotts/plyr
- 一个优美的官方 Demo 如下

<div class="plyr__video-embed" id="player">
  <iframe
    src="https://www.youtube.com/embed/bTqVqk7FSmY?origin=https://plyr.io&amp;iv_load_policy=3&amp;modestbranding=1&amp;playsinline=1&amp;showinfo=0&amp;rel=0&amp;enablejsapi=1"
    allowfullscreen
    allowtransparency
    allow="autoplay">
  </iframe>
</div>

# 优点

- 不仅美观优雅，而且功能十分丰富
- 各种控制条的 UI 都挺好看的
- 进度拖动流畅
- 音量条
- 字幕控制，可以点击来开启或者关闭字幕，支持嵌入字幕文件
- 分辨率支持
- 播放速度控制，还支持各种自定义速度，比如 1.25 倍，4倍等等
- 支持画中画模式播放

# 功能

GitHub 的介绍如下：

- 📼 HTML 视频和音频、YouTube 和 Vimeo - 支持主要格式
- 💪 无障碍- 完全支持 VTT 字幕和屏幕阅读器
- 🔧 可定制- 我们可以自定义各种选项来让播放器呈现不同的 UI。
- 😎 干净的 HTML - 使用正确的元素，比如 `<input type="range">` 控制音量和使用 `<progress>` 控制进度，
  以及 `<button>` 按钮，没有 `<span>` 或 `<a href="#">` 按钮
    - `<input type="range">` => <input type="range">
    - `<progress>` => <progress>
    - `<button>` => <button>
- 📱 响应式- 适用于任何屏幕尺寸
- 💵 获利- 从您的视频中赚钱
- 📹 流式传输- 支持 hls.js、Shaka 和 dash.js 流式播放
- 🎛 API - 通过标准化 API 切换播放、音量、搜索等
- 🎤 事件- 不用搞乱 Vimeo 和 YouTube API，所有事件都是跨格式标准化的
- 🔎 全屏- 支持原生全屏并回退到“全窗口”模式
- ⌨️ 快捷键- 支持键盘快捷键
- 🖥 画中画- 支持画中画模式
- 📱 Playsinline - 支持playsinline属性
- 🏎 速度控制- 即时调整速度
- 📖 多个字幕- 支持多个字幕轨道
- 🌎 i18n 支持- 支持控件的国际化
- 👌 预览缩略图- 支持显示预览缩略图
- 🤟 没有框架- 用“vanilla” ES6 JavaScript 编写，不需要 jQuery
- 💁‍♀️ SASS - 包含在您的构建过程中


# 使用

## 引用

- 客户端引用

    要使用 Plyr，可以直接引用 Plyr 的 CDN 文件，添加如下引用即可：
    
    ```html
    <script src="https://cdn.plyr.io/3.6.12/plyr.js"></script>
    <link rel="stylesheet" href="https://cdn.plyr.io/3.6.12/plyr.css" />
    ```

- 服务端引用

    当然，Plyr 还支持 Node.js 项目直接引用，安装方式如下：
    
    ```bash
    $ yarn add plyr
    ```
    
    然后这样引用
    
    ```js
    import Plyr from 'plyr';
    
    const plyr = new Plyr('#player')
    ```


Plyr 有一个非常强的功能，那就是它扩展了原生 HTML5 中 Media 相关标签的功能，
比如我们现在可以给 video 标签添加一些自定义的功能，
比如添加一个 data-poster 属性来当作视频预览封面，
比如添加一个 track 标签来添加字幕文件，写法如下：

```html
<video id="player" playsinline controls data-poster="/path/to/poster.jpg">
  <source src="/path/to/video.mp4" type="video/mp4" />
  <source src="/path/to/video.webm" type="video/webm" />

  <!-- Captions are optional -->
  <track kind="captions" label="English captions" src="/path/to/captions.vtt" srclang="en" default />
</video>
```

同时 Plyr 还支持嵌入一些外链网站，比如 Youtube、Vimeo (如果支持中国的一些视频网站就好了)

如果要引用 Youtube，那么只需要给 div 添加一些 class 即可，比如：


```html
<div class="plyr__video-embed" id="player">
  <iframe
    src="https://www.youtube.com/embed/bTqVqk7FSmY?origin=https://plyr.io&amp;iv_load_policy=3&amp;modestbranding=1&amp;playsinline=1&amp;showinfo=0&amp;rel=0&amp;enablejsapi=1"
    allowfullscreen
    allowtransparency
    allow="autoplay"
  ></iframe>
</div>
```

## 样式自定义

另外 Plyr 还支持我们添加一些自定义样式，我们需要使用 CSS Custom Properties 
即可轻松实现样式复写。

比如说，我们想要把默认的按钮颜色由蓝色改成红色，那就可以直接添加 CSS 样式：

```css
:root {
  --plyr-color-main: red
}
```
这样 Plyr 就可以读取这个 CSS 属性，然后实现样式控制了。

更多的自定义样式名称可以参考：https://github.com/sampotts/plyr#customizing-the-css

## 配置自定义

刚才我们还提到了，Plyr 支持我们配置一些 Options 选项来实现一些自定义的功能，这里功能也非常全面，比如：

* settings：是一个列表，我们可以控制 settings 的功能列表，比如配置成 ['captions', 'quality', 'speed', 'loop'] 即可控制设置功能里面出现字幕、分辨率、速度、循环播放等控制。
* i18n：可以控制多语言配置。
* blankVideo：如果是空的视频的话，默认播放什么。
* autoplay：是否自动播放。

等等，还有很多，大家可以参考 https://github.com/sampotts/plyr#options 来查看更多功能，总之能想到的几乎都有了。

## JavaScrip API

另外 Play 还暴露了很多 API，比如 play、pause、stop、restart 等方法可以控制播放、
暂停、停止、重新播放等等，甚至还有 airplay 都支持。

具体的功能大家可以参考 https://github.com/sampotts/plyr#methods 查看


