---
title: TensorFlow Env
author: 王哲峰
date: '2022-07-15'
slug: dl-tensorflow-env
categories:
  - deeplearning
  - tensorflow
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

- [TensorFlow 支持的硬件平台](#tensorflow-支持的硬件平台)
- [TensorFlow 系统要求](#tensorflow-系统要求)
- [安装 TensorFlow 2](#安装-tensorflow-2)
  - [使用 pip 安装 TensorFlow 2](#使用-pip-安装-tensorflow-2)
  - [使用 conda 安装 TensorFlow 2](#使用-conda-安装-tensorflow-2)
  - [使用 Docker 安装 TensorFlow 2](#使用-docker-安装-tensorflow-2)
  - [GPU 版本的 TensorFlow 安装](#gpu-版本的-tensorflow-安装)
    - [GPU 硬件的准备](#gpu-硬件的准备)
    - [NVIDIA 驱动程序的安装](#nvidia-驱动程序的安装)
    - [CUDA Toolkit 和 cuDNN 的安装](#cuda-toolkit-和-cudnn-的安装)
- [Google Colab](#google-colab)
</p></details><p></p>


# TensorFlow 支持的硬件平台

* PC
    - CPU
    - GPU
        - 一块主流的 NVIDIA GPU 会大幅提高训练速度
        - 显卡的 CUDA 核心数和显存大小是决定机器学习性能的两个关键参数
            - 显卡的 CUDA 核心数决定训练速度
            - 显卡的 显存大小 是决定能够训练多大的模型以及训练时的最大批次大小(batch size)
        - 对于前沿的机器学习研究，尤其是计算机视觉和自然语言处理领域而言，对 GPU 并行训练是标准配置
    - Cloud TPU(张量)
* Mobile
    - Android 
    - iOS
    - Embedded Devices


# TensorFlow 系统要求

* Python 3.5-3.7
* pip >= 19.0(需要 manylinux2010 支持)
* Ubuntu 16.04 or later(64位)
* Windows 7 or later(64位, 仅支持 Python 3)
* macOS 10.12.6(Sierra) or later(64位, no GPU support)
* Raspbian 9.0 or later
* GPU 支持需要使用支持 CUDA® 的显卡（适用于 Ubuntu 和 Windows）

# 安装 TensorFlow 2

> * tensorflow: 支持 CPU 和 GPU 的最新稳定版（适用于 Ubuntu 和 Windows）
> * tf-nightly: 预览 build（不稳定）。Ubuntu 和 Windows 均包含 GPU 支持
> * tensorflow-gpu: Current release with GPU support (Ubuntu and Windows)
> * tf-nightly-gpu: Nightly build with GPU support (unstable, Ubuntu and Windows)

## 使用 pip 安装 TensorFlow 2

> 必须使用最新版本的 pip, 才能安装 TensorFlow 2

* Virtualenv 安装 

```bash        
# Requires the latest pip
(venv) $ pip install --upgrade pip

# Current stable release for CPU and GPU
(venv) $ pip install --upgrade tensorflow
(venv) $ python -c "import tensorflow as tf;print(tf.reduce_sum(tf.random.normal([1000, 1000])))"
```

* 系统安装

```bash
# Requires the latest pip
$ pip3 install --upgrade pip

# Current stable release for CPU and GPU
$ pip3 install --user --upgrade tensorflow
$ python3 -c "import tensorflow as tf; print(tf.reduce_sum(tf.random.normal([1000, 1000])))"
```

```bash
$ pip3 install pydot
```

- graphviz
    - https://graphviz.gitlab.io/download/

## 使用 conda 安装 TensorFlow 2

1. 安装 Python 环境
    - 建议安装 Anaconda 的 Python 3.7 64 位版本
2. 使用 Anaconda/Miniconda 自带的 conda 包管理器建立一个 conda 虚拟环境，并进入虚拟环境

```bash
$ conda create --name tf2 python=3.7
conda activate tf2
```

3.使用 Python 包管理器 pip 安装 TensorFlow

```bash
pip install tensorflow
```

## 使用 Docker 安装 TensorFlow 2

- TensorFlow Docker 映像已经过配置，可运行 TensorFlow。
  Docker 容器可在虚拟环境中运行，是设置 GPU 支持的最简单方法。

```bash 
# Download latest stable image
$ docker pull tensorflow/tensorflow:latest-py3

# Start Jupyter server
$ docker run -it -p 8888:8888 tensorflow/tensorflow:latest-py3-jupyter
```


* TensorFlow Docker Image
  - https://hub.docker.com/r/tensorflow/tensorflow/


小技巧

- 也可以使用 ``conda install tensorflow`` 或者 ``conda install tensorflow-gpu`` 命令安装 TensorFlow，
  不过 conda 源的版本往往更新较慢，难以在第一时间获得最新的 TensorFlow 版本
- 从 TensorFlow 2.1 开始，pip 包 tensorflow 同时包含 GPU 支持，无须通过特定的 pip 包 tensorflow-gpu 安装 GPU 版本。
  如果对 pip 包的大小敏感，可使用 tensorflow-cpu 包安装仅支持 CPU 的 Tensorflow 版本
- 在 Windows 系统下，需要打开 “开始” 菜单中的 “Anaconda Prompt” 进入 Anaonda 的命令行环境


## GPU 版本的 TensorFlow 安装

GPU 版本的 TensorFlow 可以利用 NVIDIA GPU 强大的加速计算能力，
使 TensorFlow 运行更加高效，尤其是可以成倍提升模型的训练速度.
在安装 GPU 版本的 TensorFlow 前，需要有一块“不太旧”的 NVIDIA 显卡，
并正确安装 NVIDIA 显卡驱动程序、CUDA Toolkit 和 cuDNN.

### GPU 硬件的准备

TensorFlow 对 NVIDIA 显卡的支持较为完备。对于 NVIDIA 显卡，
要求其 CUDA 的算力(compute capability) 不低于 3.5。
可以到 NVIDIA 的官网查询自己所用显卡的 CUDA 算力。

目前 AMD 显卡也开始对 TensorFlow 提供支持。

### NVIDIA 驱动程序的安装

* Windows 下安装 NVIDIA 驱动程序:
    - 在 Windows 系统中，如果系统具有 NVIDIA 显卡，那么系统内往往已经自动安装了 NVIDIA 显卡驱动程序。
      如果未安装，直接访问 NVIDIA 官网，下载并安装对应型号的最新标准版驱动程序即可。
* Linux 下安装 NVIDIA 驱动程序:
    - 服务器版 Linux 系统
        1. 访问 NVIDIA 官网下载驱动程序(.run 文件)
        2. 安装驱动

            ```bash
            sudo apt-get install build-essential # 安装之前，可能需要安装合适的编译环境
            sudo bash DRIVER_FILE_NAME.run
            ```
    
    - 具有图形界面的桌面版 Linux 系统(Ubuntu为例)
        1. 禁用系统自带的开源显卡驱动 Nouveau(在 ``/etc/modprobe.d/blacklist.conf``)文件中添加如下内容，并更新内核、重启

            ```bash
            cd /etc/modprobe.d/blacklist.conf
            blacklist nouveau
            sudo update-initramfs -u
            ```
            
        2. 禁用主板的 Secure Boot 功能
        3. 停用桌面环境

            ```bash
            sudo service lightdm stop
            ```

        4. 删除原有 NVIDIA 驱动程序

            ```bash
            sudo apt-get purge nvidia*
            ```

- NVIDIA 驱动程序安装完成后，可以在命令行下使用 ``nvidia-smi`` 命令检查是否安装成功，若成功，则会打印当前系统安装的 NVIDIA 驱动信息:

```bash
$ nvidia-sim
```

- ``nvidia-sim`` 命令可以产看机器上现有的 GPU 及使用情况

### CUDA Toolkit 和 cuDNN 的安装

- 在 Anaconda/Miniconda 环境下，推荐使用 conda 安装 CUDA Toolkit 和 cuDNN

1. 搜索 conda 源中可用的 CUDA Toolkit 和 cuDNN 版本号

```bash
conda search cudatoolkit
conda search cudnn
```

2. 安装 CUDA Toolkit 和 cuDNN

```bash
conda install cudatoolkit=X.X
conda install cudnn=X.X.X
```

- 在 使用 Python pip 安装 时:
- 按照 TensorFlow 官网的说明手动下载 CUDA Toolkit 和 cuDNN 并安装，不过过程比较繁琐。

# Google Colab

* https://colab.research.google.com/notebooks/welcome.ipynb?hl=zh_cn

