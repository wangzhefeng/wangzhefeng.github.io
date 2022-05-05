---
title: MkDocs Doc
author: 王哲峰
date: '2022-05-06'
slug: pydocs-mkdocs
categories: 
  - doc
tags:
  - tool
---


MkDocs is a documentation generator that focuses on speed and
simplicity. It has many great features including:

- Preview your documentation as you write it
- Easy customization with themes and extensions
- Writing documentation with Markdown

# 安装环境

```bash
$ pip3 install mkdocs
```

# 创建文档

```bash
$ mkdocs new project_doc
```

# 编译文档

```bash
$ mkdocs serve
```

# 打开离线文档: 

- http://127.0.0.1:8000/


# GitHub 代码托管

```bash
$ touch .gitignore
$ git init
$ git add .
$ git remote add origin git:/github.git
$ git push -u origin master
```

# 绑定 Read the Docs

1. [Import your docs.](https://docs.readthedocs.io/en/stable/intro/import-guide.html)
2. [Read the Docs dashboard](https://readthedocs.org/dashboard/)
3. [Import](https://readthedocs.org/dashboard/import/?__cf_chl_captcha_tk__=f51d0fd05a6dd27a26845c9bd923a6f42ecfded4-1588260812-0-AVHp7xZY-MfpUWYf-sWQgn7MpabCmi2Dzc_tn4_f3tGxMObBh87mGw19KwybY3HkO9EzmoByZ_vpqhjdGT6oOoXXPt714nvln3sxrf6vsoIa_Q8wQ0aHNgzPEhBiO7u0LyHFxtYsg8cbCFpUY-Y_HPZ-Th-S6BmRj6pZIZPh4ieiR6nrWAmQEqnhPeCl79jRC11MMwJ5Gao4xji5JEufhc98l4D-okayG_5A1B8W2kCEXPaENPFiBc113EpO3E70G03ibg25CfezRwD7jXAG5Sc86TZ_u35SRkn7e_IySD-yEkUec8NRFQRPH6uEhP8RPVXdjKzhFrD7D6s19Uevg8eDXqTCO-y8TjdSTQ_28xcDeBz_jMRyveeYFNp5QgGbXRox5WxdaiMFCGaufD4Aqfc)

# 版本管理

- [Version Doc](https://docs.readthedocs.io/en/stable/versions.html)

# 资源

- [MkDocs documentation](https://www.mkdocs.org/)
- [Markdown syntax guide](https://daringfireball.net/projects/markdown/syntax)
- [Writing your docs with MkDocs](https://www.mkdocs.org/user-guide/writing-your-docs/)
