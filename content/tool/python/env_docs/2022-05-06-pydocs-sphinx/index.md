---
title: Sphinx Doc
author: wangzf
date: '2022-05-06'
slug: pydocs-sphinx
categories:
  - Python
tags:
  - tool
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

- [安装环境](#安装环境)
- [创建文档](#创建文档)
- [修改配置文件](#修改配置文件)
  - [更改主题](#更改主题)
  - [支持 markdown 语法](#支持-markdown-语法)
- [编译文档](#编译文档)
- [GitHub 代码托管](#github-代码托管)
- [绑定 Read the Docs](#绑定-read-the-docs)
- [版本管理](#版本管理)
- [资源](#资源)
- [config.py 模板](#configpy-模板)
- [reStructuredText Markup 语法](#restructuredtext-markup-语法)
</p></details><p></p>


Sphinx is a powerful documentation generator that has many great
features for writing technical documentation including:

- Generate web pages, printable PDFs, documents for e-readers (ePub),
   and more all from the same sources
- You can use reStructuredText or Markdown to write documentation
- An extensive system of cross-referencing code and documentation
- Syntax highlighted code samples
- A vibrant ecosystem of first and third-party [extensions](https://www.sphinx-doc.org/en/master/usage/extensions/index.html#builtin-sphinx-extensions)


# 安装环境

(1)激活 Python 虚拟环境

```bash
workon doc_env
```

(2)安装 Sphinx 及其依赖库

```bash
$ pip3 install sphinx 
$ pip3 install sphinx-autobuild 
$ pip3 install sphinx_rtd_theme
```

# 创建文档

(1)创建 Sphinx Doc 项目目录

```bash
$ mkdir project
$ cd project
$ mkdir docs
$ mkdir src
$ cd docs
```

(2)创建 Sphinx 项目

```bash
$ sphinx quickstart
```

# 修改配置文件

## 更改主题

```python
# ./project/docs/source/conf.py

import sphinx_rtd_theme

html_theme = "sphinx_rtd_theme"
html_theme_path = [sphinx_rtd_theme.get_html_theme_path()]
```


## 支持 markdown 语法

(1)安装扩展库: 

```bash
$ pip3 install recommonmark
```

(2)修改配置: 

```python
# ./project/doca/source/conf.py

from recommonmark.parser import CommonMarkParser

extensions = [
      "recommonmark",
]

source_parsers = {
      '.md': CommonMarkParser,
}

source_suffix = ['.rst', '.md']
```


# 编译文档

```bash
cd ./project/docs/
make html
```


# GitHub 代码托管

```bash
cd ./project/
touch .gitignore

git init
git add doc
git remote add origin git:/github.git
git push -u origin master
```


# 绑定 Read the Docs

1. [Import your docs.](https://docs.readthedocs.io/en/stable/intro/import-guide.html)
2. [Read the Docs dashboard](https://readthedocs.org/dashboard/)
3. [Import](https://readthedocs.org/dashboard/import/?__cf_chl_captcha_tk__=f51d0fd05a6dd27a26845c9bd923a6f42ecfded4-1588260812-0-AVHp7xZY-MfpUWYf-sWQgn7MpabCmi2Dzc_tn4_f3tGxMObBh87mGw19KwybY3HkO9EzmoByZ_vpqhjdGT6oOoXXPt714nvln3sxrf6vsoIa_Q8wQ0aHNgzPEhBiO7u0LyHFxtYsg8cbCFpUY-Y_HPZ-Th-S6BmRj6pZIZPh4ieiR6nrWAmQEqnhPeCl79jRC11MMwJ5Gao4xji5JEufhc98l4D-okayG_5A1B8W2kCEXPaENPFiBc113EpO3E70G03ibg25CfezRwD7jXAG5Sc86TZ_u35SRkn7e_IySD-yEkUec8NRFQRPH6uEhP8RPVXdjKzhFrD7D6s19Uevg8eDXqTCO-y8TjdSTQ_28xcDeBz_jMRyveeYFNp5QgGbXRox5WxdaiMFCGaufD4Aqfc)



# 版本管理

- [Version Doc](https://docs.readthedocs.io/en/stable/versions.html)



# 资源

- [Sphinx documentation](https://www.sphinx-doc.org/en/master/)
- [RestructuredText primer](https://www.sphinx-doc.org/en/master/usage/restructuredtext/basics.html)
- [An introduction to Sphinx and Read the Docs for technical writers](https://www.ericholscher.com/blog/2016/jul/1/sphinx-and-rtd-for-writers/)



# config.py 模板

```python
# -*- coding: utf-8 -*-
#
# Configuration file for the Sphinx documentation builder.
#
# This file does only contain a selection of the most common options. For a
# full list see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Path setup --------------------------------------------------------------

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
#
# import os
# import sys
# sys.path.insert(0, os.path.abspath('.'))


# -- Project information -----------------------------------------------------

project = 'OpenCV'
copyright = '2018, Hunag Xinyuan'
author = 'Hunag Xinyuan'

# The short X.Y version
version = '1.0'
# The full version, including alpha/beta/rc tags
release = '1.0'


# -- General configuration ---------------------------------------------------

# If your documentation needs a minimal Sphinx version, state it here.
#
# needs_sphinx = '1.0'

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
      'sphinx.ext.autodoc',
      'sphinx.ext.viewcode',
      'sphinx.ext.todo',
      'sphinx.ext.mathjax',
      'sphinx.ext.apidoc',
      'sphinx.ext.extlinks',
      'nbsphinx',
      'sphinx_markdown_tables',
      'sphinx.ext.githubpages',
]

# Add any paths that contain templates here, relative to this directory.
templates_path = ['_templates']

# The suffix(es) of source filenames.
# You can specify multiple suffix as a list of string:
#
# source_suffix = ['.rst', '.md']
# source_suffix = '.rst'

# The master toctree document.
master_doc = 'index'

# The language for content autogenerated by Sphinx. Refer to documentation
# for a list of supported languages.
#
# This is also used if you do content translation via gettext catalogs.
# Usually you set "language" from the command line for these cases.
language = 'zh_CN'

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = [
      '**.ipynb_checkpoints',
]

# The name of the Pygments (syntax highlighting) style to use.
pygments_style = None


# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
import sphinx_rtd_theme
html_theme = 'sphinx_rtd_theme'
html_theme_path = [sphinx_rtd_theme.get_html_theme_path()]

# Theme options are theme-specific and customize the look and feel of a theme
# further.  For a list of options available for each theme, see the
# documentation.
#
# html_theme_options = {}

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ['_static']

# Custom sidebar templates, must be a dictionary that maps document names
# to template names.
#
# The default sidebars (for documents that don't match any pattern) are
# defined by theme itself.  Builtin themes are using these templates by
# default: ``['localtoc.html', 'relations.html', 'sourcelink.html',
# 'searchbox.html']``.
#
# html_sidebars = {}


# Markdown support
import recommonmark
from recommonmark.transform import AutoStructify
from recommonmark.parser import CommonMarkParser
source_parsers = {
      # '.md': CommonMarkParser,
      '.md': 'recommonmark.parser.CommonMarkParser',
}
source_suffix = ['.rst', '.md']


def setup(app):
      app.add_config_value('recommonmark_config', {
         # 'url_resolver': lambda url: github_doc_root + url,
         'enable_math': False,
         'enable_inline_math': False,
      }, True)
      app.add_transform(AutoStructify)


# math support
# TODO


# -- Options for HTMLHelp output ---------------------------------------------

# Output file base name for HTML help builder.
htmlhelp_basename = 'OpenCVdoc'


# -- Options for manual page output ------------------------------------------

# One entry per manual page. List of tuples
# (source start file, name, description, authors, manual section).
# man_pages = [
#     (master_doc, 'opencv', u'OpenCV Documentation',
#     [author], 1)
# ]


# -- Options for Texinfo output ----------------------------------------------

# Grouping the document tree into Texinfo files. List of tuples
# (source start file, target name, title, author,
#  dir menu entry, description, category)
# texinfo_documents = [
#     (master_doc, 'OpenCV', u'OpenCV Documentation',
#      author, 'OpenCV', 'One line description of project.',
#      'Miscellaneous'),
# ]


# -- Options for Epub output -------------------------------------------------

# Bibliographic Dublin Core info.
# epub_title = project

# The unique identifier of the text. This can be a ISBN number
# or the project homepage.
#
# epub_identifier = ''

# A unique identification for the text.
#
# epub_uid = ''

# A list of files that should not be packed into the epub file.
# epub_exclude_files = ['search.html']


# -- Extension configuration -------------------------------------------------


# -- Options for LaTeX output ------------------------------------------------

# latex_elements={
#     # The paper size ('letterpaper' or 'a4paper').
#     'papersize': 'a4paper', # The font size ('10pt', '11pt' or '12pt').
#     'pointsize': '12pt',
#     'classoptions': ',oneside',
#     'babel': '',    #必須
#     'inputenc': '', #必須
#     'utf8extra': '',#必須
#     # Additional stuff for the LaTeX preamble.
#     'preamble': r"""
#         \usepackage{xeCJK}
#         \usepackage{indentfirst}
#         \setlength{\parindent}{2em}
#         \setCJKmainfont{WenQuanYi Micro Hei}
#         \setCJKmonofont[Scale=0.9]{WenQuanYi Micro Hei Mono}
#         \setCJKfamilyfont{song}{WenQuanYi Micro Hei}
#         \setCJKfamilyfont{sf}{WenQuanYi Micro Hei}
#         \XeTeXlinebreaklocale "zh"
#         \XeTeXlinebreakskip = 0pt plus 1pt
#     """
# }
```

# reStructuredText Markup 语法

[reStructured](https://docutils.sourceforge.io/rst.html)
[A ReStructuredText Primer](https://docutils.sourceforge.io/docs/user/rst/quickstart.html)
[Quick reStructuredText](https://docutils.sourceforge.io/docs/user/rst/quickref.html)
[reStructuredText Markup Specification](https://docutils.sourceforge.io/docs/ref/rst/restructuredtext.html#literal-blocks)
[Docutils](https://docutils.sourceforge.io/docs/ref/rst/restructuredtext.html)

