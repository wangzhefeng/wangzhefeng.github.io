---
title: 本站构建之路
author: 王哲峰
date: '2022-02-27'
slug: blog-build-deploy
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

- [软件/知识](#软件知识)
- [1.创建 GitHub 仓库](#1创建-github-仓库)
- [2.创建项目](#2创建项目)
- [3.新建文章](#3新建文章)
- [4.评论功能](#4评论功能)
- [5.页面自定义](#5页面自定义)
- [6.部署前检查](#6部署前检查)
- [7.提交代码到 GitHub](#7提交代码到-github)
- [8.部署](#8部署)
  - [8.1 Netlify](#81-netlify)
    - [8.1.1 部署](#811-部署)
    - [8.1.2 设置私有域名](#812-设置私有域名)
  - [8.2 Vercel](#82-vercel)
    - [8.2.1 部署](#821-部署)
    - [8.2.2 设置私有域名](#822-设置私有域名)
- [9.最终效果](#9最终效果)
- [参考资料](#参考资料)
</p></details><p></p>

> 早在 2016 年左右，我就一直在研究 R 相关的包构建 Blog。
  之前使用过 Hexo，部署在 GitHub Pages 上，但之后断断续续删了重建。
  后来就没有投入太多的精力搞这个，把各种内容写在了本地，
  有些存储在 GitHub 代码仓库中.
> 
> 后来慢慢发现了 R 包 blogdown，重新激起我建站的动力，
  经过断断续续的折腾，2022.02.27 终于连上了 Internet，不容易啊，慢慢维护

# 软件/知识

- R
- RStudio IDE
- blogdown
- Hugo
- Markdown
- R Markdown
- Github
- Netlify
- Vercel
- Disqus
- Utterances

# 1.创建 GitHub 仓库

1. 在 GitHub 上新建一个仓库，并在仓库中初始化一个 `README.md` 文件，
   但不要添加 `.gitignore` 文件(一会再处理)
2. 克隆仓库到本地

# 2.创建项目

1. 在 RStudio 中创建一个新项目
2. 使用 `blogdown` 创建一个带有 Hugo Ivy 主题的网站项目

```r
> blogdown::new_site(theme = "yihui/hugo-ivy")
```

3. 启动 blogdown 的本地服务器

```r
blogdown::serve_site()
```

# 3.新建文章

```r
blogdown::new_post(title = "Hi Hugo", 
                   ext = ".md",
                   subdir = "post")
```

# 4.评论功能

- Disqus
- Utterances

# 5.页面自定义

- HTML
- CSS
- JS
- Markdown
- R
- R Markdown
- Hugo
- blogdown

# 6.部署前检查

- 检查 `.gitignore` 文件

```r
blogdown::check_gitignore()
```

- 检查内容

```r
blogdown::check_content()
```

# 7.提交代码到 GitHub

```r
file.edit(".gitignore")
```

在 `.gitignore` 中添加以下内容:

```
.Rproj.user
.Rhistory
.RData
.Ruserdata
.DS_Store
Thumbs.db
```

# 8.部署

## 8.1 Netlify

### 8.1.1 部署

1. 注册、登录 [Netlify](https://vercel.com/login)
2. 导入 GitHub 仓库 [wangzhefeng.github.io](https://github.com/wangzhefeng/wangzhefeng.github.io)
3. 点击部署
4. 查看网站 [wangzhefeng.com](https://wangzhefeng.com/)

### 8.1.2 设置私有域名

- 腾讯域名: wangzhefeng.com

## 8.2 Vercel

### 8.2.1 部署

1. 注册、登录 [Vercel](https://vercel.com/login)
2. 导入 GitHub 仓库 [wangzhefeng.github.io](https://github.com/wangzhefeng/wangzhefeng.github.io)
3. 点击部署

### 8.2.2 设置私有域名

- 腾讯域名: wangzhefeng.com

# 9.最终效果

- [wangzhefeng.com](https://wangzhefeng.vercel.app/)

# 参考资料

- [谢益辉的网站](https://yihui.org/)
- [统计之都--用 R 语言的 blogdown+hugo+netlify+github 建博客](https://cosx.org/2018/01/build-blog-with-blogdown-hugo-netlify-github/)
- [blogdown GitHub](https://github.com/rstudio/blogdown)
- [教程](https://www.apreshill.com/blog/2020-12-new-year-new-blogdown/#step-5-publish-site)
- [好的博客](https://robjhyndman.com/)
- [Hugo Ivy 主题 GitHub](https://github.com/yihui/hugo-ivy)
- [Hugo Ivy 主题示例网站](https://ivy.yihui.org/)
- [Hugo 主题库](https://themes.gohugo.io/?search-input=)
- [R Markdown](https://rmarkdown.rstudio.com/)
- [Google 搜索](https://www.google.com)“如何使用 Vercel 部署个人网站”
