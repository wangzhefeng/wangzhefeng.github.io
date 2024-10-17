---
title: Python conda
author: wangzf
date: '2022-07-13'
slug: python-conda
categories:
  - Python
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
img {
    pointer-events: none;
}
</style>

<details><summary>目录</summary><p>

- [Conda 下载](#conda-下载)
- [Conda 安装](#conda-安装)
- [Conda 使用](#conda-使用)
    - [conda 管理](#conda-管理)
    - [Python 管理](#python-管理)
</p></details><p></p>

# Conda 下载

- [Anaconda3](https://www.anaconda.com/products/individual) 
    - [Anaconda 老版本](https://repo.anaconda.com/archive/) 
- [Miniconda3](https://docs.conda.io/en/latest/miniconda.html) 
    - [Miniconda 老版本](https://repo.anaconda.com/miniconda/) 

# Conda 安装

1. 如何检查当前环境中是否安装了 Conda?
   - Windows(Anaconda Prompt)
        - `echo %PATH%`    
   - macOS 和 Linux
        - `echo $PAHT`
2. 如何检查当前环境中默认的 Python 环境？
   - Windows(Anaconda Prompt)
        - `where python`   
    - macOS 和 Linux
        - `which python`

# Conda 使用

Package, dependency and environment management for any 
language—Python, R, Ruby, Lua, Scala, Java, JavaScript, 
C/ C++, FORTRAN, and more.

- conda 管理
- packages 管理
- virtual packages 管理
- environment 管理
- channels 管理
- Python 管理

## conda 管理

1. 验证 conda 是否已经安装

```bash
conda --version
```

2. 确定 conda 版本

```bash
conda info
conda -V
```

3. 将 conda 更新到当前版本

```bash
conda update conda
```

4. 禁止显示有关更新 conda 的警告消息

```bash
conda update -n base conda
```

```bash
conda config --set notify_outdated_conda false
```

```bash
# ~/.conda
notify_updated_conda: false
```

## Python 管理

Conda treats Python the same as any other package, so it is easy to manage and update multiple installations.

1. 查看可供 conda 下载的 Python 版本列表

```bash
conda search python
conda search --full-name python
```

2. 安装其他版本的 Python

安装其他版本的 Python 并不覆盖目前已经存在的版本.

(1)创建新环境

```bash
conda create -n py36 python=3.6 anaconda
conda create -n py36 python=3.7 miniconda
conda create -n py27 python=2.7 anaconda
```

(2)激活新环境

- 参考 **切换其他版本的 Python**

(3)验证新环境是否为当前环境
(4)验证当前环境

```bash
python --version
```

3. 切换其他版本的 Python
    - (1)如果当前环境是 conda 环境:
        - 1.激活环境

        ```bash
        conda activate myenv
        ```

        - 2.停用环境

        ```bash
        conda deactivate
        ```

    - (2)如果当前环境不是 conda 环境:
        - Windows 

        ```bash
        D:\Miniconda\Script\acitvate base
        ```

        - macOS 或 Linux

        ```bash
        ~/opt/miniconda3/bin/activate base
        ```
      
        - 嵌套环境

        ```bash
        (doc)$ codna activate --stack myenv
        $ conda config -set auto_stack 1
        ```
    
        - Conda init 

        ```bash
        conda init 
        auto_activate_base: bool
        ```

4. 更新或升级 Python

```bash
conda update python
conda install python=3.6
```