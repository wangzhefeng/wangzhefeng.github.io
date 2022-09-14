---
title: NLP-utils-spaCy
author: 王哲峰
date: '2022-04-05'
slug: nlp-utils-spacy
categories:
  - NLP
tags:
  - tool
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

- [spaCy 安装](#spacy-安装)
  - [官网安装](#官网安装)
  - [pip 安装](#pip-安装)
</p></details><p></p>

# spaCy 安装

## 官网安装

```bash
# Install spaCy
$ pip install spacy
$ pip install spacy-nightly --pre

# Download model
$ python -m spacy en

# EN
$ python -m spacy download en_core_web_lg
$ python -m spacy download en_core_web_md
$ python -m spacy download en_core_web_sm

# ZH
$ python -m spacy download en_core_web_lg
$ python -m spacy download en_core_web_md
$ python -m spacy download en_core_web_sm

# Install textacy which will also be useful
$ pip3 install -U textacy
```

## pip 安装

- (1)从 GitHub 上安装

```bash
# EN
pip install https://github.com/explosion/spacy-models/releases/download/en_core_web_md-1.2.0/en_core_web_md-1.2.0.tar.gz
pip install https://github.com/explosion/spacy-models/releases/download/en_core_web_md-1.2.0/en_core_web_md-1.2.0.tar.gz
pip install https://github.com/explosion/spacy-models/releases/download/en_core_web_md-1.2.0/en_core_web_md-1.2.0.tar.gz

# ZH
pip install https://github.com/explosion/spacy-models/releases/download/en_core_web_md-1.2.0/en_core_web_md-1.2.0.tar.gz
pip install https://github.com/explosion/spacy-models/releases/download/en_core_web_md-1.2.0/en_core_web_md-1.2.0.tar.gz
pip install https://github.com/explosion/spacy-models/releases/download/en_core_web_md-1.2.0/en_core_web_md-1.2.0.tar.gz
```

- (2)先从 Github 上下载安装包，再用 pip 或 ``python setup.py install`` 安装

    - `GitHub 下载地址 <https://github.com/explosion/spacy-models/releases/>`_ 
    - 本地 pip 安装

```bash
# EN
$ pip install /your_path/en_core_web_lg-x.x.x.tar.gz
$ pip install /your_path/en_core_web_md-x.x.x.tar.gz
$ pip install /your_path/en_core_web_sm-x.x.x.tar.gz

# ZH
$ pip install /your_path/zh_core_web_lg-x.x.x.tar.gz
$ pip install /your_path/zh_core_web_md-x.x.x.tar.gz
$ pip install /your_path/zh_core_web_sm-x.x.x.tar.gz
```

- 本地 ``python3 setup.py install`` 安装

```bash
# (1)解压到 setup.py 路径
# (2)安装
# EN
$ python3 setup.py install /your_path/en_core_web_lg-x.x.x.tar.gz
$ python3 setup.py install /your_path/en_core_web_md-x.x.x.tar.gz
$ python3 setup.py install /your_path/en_core_web_sm-x.x.x.tar.gz

# ZH
$ python3 setup.py install /your_path/zh_core_web_lg-x.x.x.tar.gz
$ python3 setup.py install /your_path/zh_core_web_md-x.x.x.tar.gz
$ python3 setup.py install /your_path/zh_core_web_sm-x.x.x.tar.gz
```

