---
title: Python conda
author: 王哲峰
date: '2022-07-13'
slug: python-conda
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
</style>

<details><summary>目录</summary><p>

- [Conda 介绍](#conda-介绍)
- [Conda 安装](#conda-安装)
  - [Conda 下载](#conda-下载)
  - [Conda 安装](#conda-安装-1)
- [Conda 使用](#conda-使用)
  - [conda 管理](#conda-管理)
  - [packages 管理](#packages-管理)
  - [virtual packages 管理](#virtual-packages-管理)
  - [environment 管理](#environment-管理)
  - [channels 管理](#channels-管理)
  - [Python 管理](#python-管理)
</p></details><p></p>


# Conda 介绍

系统要求:

- 32- or 64-bit computer.
- For Miniconda---400 MB disk space.
- For Anaconda---Minimum 3 GB disk space to download and install.
- Windows, macOS, or Linux.

下载安装方式：

- Miniconda
    - conda
    - conda dependencies
- Anaconda
    - conda
    - 7500+ open-source packages
- Silent mode
    - Windows
    - macOS
    - Linux
- 多 Python 环境中安装
    - 为了安装 Conda 不需要卸载其他 Python 环境，其他 Python 环境如下：
        - 系统中自带安装的 Python
        - 从 macOS Homebrew 包管理工具中安装的 Python
        - 从 pip 安装的 Python 包



> 1.如何检查当前环境中是否安装了 Conda?
> 
>    - Windows(Anaconda Prompt)
>         - `echo %PATH%`    
>    - macOS 和 Linux
>         - `echo $PAHT`
> 
> 2.如何检查当前环境中默认的 Python 环境？
>     
>    - Windows(Anaconda Prompt)
>         - `where python`   
>     - macOS 和 Linux
>         - `which python`

# Conda 安装

## Conda 下载

- [Anaconda3](https://www.anaconda.com/products/individual) 
    - [Anaconda 老版本](https://repo.anaconda.com/archive/) 
- [Miniconda3](https://docs.conda.io/en/latest/miniconda.html) 
    - [Miniconda 老版本](https://repo.anaconda.com/miniconda/) 
- [Anaconda Enterprise](https://www.anaconda.com/products/enterprise) 

## Conda 安装

- Windows
- macOS
- Linux

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

1.验证 conda 是否已经安装

```bash
conda --version
```

2.确定 conda 版本

```bash
conda info
conda -V
```

3.将 conda 更新到当前版本

```bash
conda update conda
```

4.禁止显示有关更新 conda 的警告消息

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


## packages 管理





## virtual packages 管理





## environment 管理





## channels 管理





## Python 管理


Conda treats Python the same as any other package, so it is easy to manage and update multiple installations.

- Anaconda supports Python 2.7, 3.6, and 3.7. 
  The default is Python 2.7 or 3.7,
  depending on which installer you used:
    - For the installers "Anaconda" and "Miniconda," the default is 2.7.
    - For the installers "Anaconda3" or "Miniconda3," the default is 3.7.

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