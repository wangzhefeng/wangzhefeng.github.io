---
title: 
author: 王哲峰
date: '2022-02-27'
categories:
  - blogdown
tags:
  - note
slug: blog-build-deploy
---

> 经过断断续续的折腾，2020.02.27 终于连上了 Internet，不容易啊，慢慢维护

# 软件/知识

- R
- RStudio IDE
- Markdown
- blogdown
- Hugo
- Github
- Netlify
- Vercel


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



# 4.提交代码到 GitHub

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

# 4.部署前检查

- 检查 `.gitignore` 文件

```r
blogdown::check_gitignore()
```

- 检查内容

```r
blogdown::check_content()
```


# 5.部署

## 5.1 Netlify

### 部署

1. 注册、登录 [Netlify](https://vercel.com/login)
2. 导入 GitHub 仓库 [wangzhefeng.github.io](https://github.com/wangzhefeng/wangzhefeng.github.io)
3. 点击部署
4. 查看网站 [wangzhefeng.com](https://wangzhefeng.com/)

### 设置私有域名



## 5.2 Vercel

### 部署

1. 注册、登录 [Vercel](https://vercel.com/login)
2. 导入 GitHub 仓库 [wangzhefeng.github.io](https://github.com/wangzhefeng/wangzhefeng.github.io)
3. 点击部署

### 设置私有域名

4. 查看网站 [wangzhefeng.com](https://wangzhefeng.vercel.app/)

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
