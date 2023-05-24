---
title: Python Jupyter
author: 王哲峰
date: '2022-07-13'
slug: python-jupyter
categories:
  - python
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

- [Install Jupyter](#install-jupyter)
  - [安装 Jupyter 相关库](#安装-jupyter-相关库)
  - [Jupyter kernel 设置](#jupyter-kernel-设置)
- [JupyterLab 1.0](#jupyterlab-10)
- [Jupyter 插件](#jupyter-插件)
</p></details><p></p>

# Install Jupyter

- `jupyter`
- `notebook`
- `jupyterlab`
- `ipykernel`
- `jupyter-client`
- `jupyter-console`
- `jupyter-core`
- `jupyter-server`
- `jupyterlab-pygments`
- `jupyterlab-server`
- `voila`

## 安装 Jupyter 相关库


- Jupyter

```bash
$ pip install jupyter
```

- Jupyter Notebook

```bash
$ pip install notebook
```

- Jupyter Lab

```bash
$ pip install jupyterlab
```

- Voila

```bash
$ pip install voila
```

## Jupyter kernel 设置


- 安装 `ipykernel` 在当前环境：

```bash
$ pip instll ipykernel
```

- 查看 kernel

```bash
$ jupyter kernelspec list
```
- 将环境加入 Jupyter Lab

```bash
$ workon pysci
$ python -m ipykernel install --prefix=/Users/zfwang/.virtualenv/pysci/ --name pysci
$ ipykernel install --name env_name --user
```
- 删除 kernel

```bash
$ jupyter kernelspec remove python3
```

# JupyterLab 1.0

- JupyterLab 帮助

```bash
$ jupyter lab -h
```

- 登录 JupyterLab
    - `--port`
        - 指定端口号
    - `--ip`
        - 指定 IP
    - `--notebook`

```bash
$ jupyter lab --port="8080" --ip="*" --notebook-dir="/path/..."
```

- 配置 JupyterLab 密码

```bash
$ jupyter lab --generate-config
$ jupyter lab password
```

> - 可以使用 `--port` 参数指定端口号
>     - 部分云服务(如GCP)的实例默认不开放大多数网络端口，如果使用


# Jupyter 插件