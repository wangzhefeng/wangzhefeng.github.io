---
title: PyTorch Env
author: 王哲峰
date: '2022-07-11'
slug: dl-pytorch-env
categories:
  - deeplearning
  - pytorch
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

- [PyTorch 支持的硬件平台](#pytorch-支持的硬件平台)
- [PyTorch 系统要求](#pytorch-系统要求)
- [macOS 安装 PyTorch](#macos-安装-pytorch)
  - [使用 pip 安装 PyTorch](#使用-pip-安装-pytorch)
  - [使用 Anaconda 安装 PyTorch](#使用-anaconda-安装-pytorch)
  - [使用 Docker 安装 PyTorch](#使用-docker-安装-pytorch)
- [Ubuntu 安装 PyTorch](#ubuntu-安装-pytorch)
  - [使用 pip 安装 PyTorch](#使用-pip-安装-pytorch-1)
  - [使用 Anaconda 安装 PyTorch](#使用-anaconda-安装-pytorch-1)
  - [使用 Docker 安装 PyTorch](#使用-docker-安装-pytorch-1)
- [Building from source](#building-from-source)
  - [macOS](#macos)
  - [Ubuntu](#ubuntu)
- [Verification](#verification)
</p></details><p></p>


# PyTorch 支持的硬件平台

- PC
	- CPU
	- GPU
	- Cloud TPU(张量)
- Mobile
	- Android 
	- iOS
	- Embedded Devices

# PyTorch 系统要求

- Linux distributions that use glibc >= v2.17
    - Arch Linux, minimum version 2012-07-15
    - CentOS, minimum version 7.3-1611
    - Debian, minimum version 8.0
    - Fedora, minimum version 24
    - Mint, minimum version 14
    - OpenSUSE, minimum version 42.1
    - PCLinuxOS, minimum version 2014.7
    - Slackware, minimum version 14.2
    - Ubuntu, minimum version 13.04
- macOS 10.10(Yosemite) or above
- Windows
    - Windows 7 and greater; Windows 10 or greater recommended.
    - Windows Server 2008 r2 and greater

# macOS 安装 PyTorch

## 使用 pip 安装 PyTorch

```bash
$ pip install numpy
$ pip install torch torchvision
```

## 使用 Anaconda 安装 PyTorch

```bash
$ conda install pytorch torchvision -c pytorch
```

## 使用 Docker 安装 PyTorch

```bash
$ 
```

# Ubuntu 安装 PyTorch

## 使用 pip 安装 PyTorch

```bash
$ pip install torch==1.8.0+cpu torchvision==0.9.0+cpu torchaudio==0.8.0 -f https://download.pytorch.org/whl/torch_stable.html
```

## 使用 Anaconda 安装 PyTorch

```bash
$ conda install pytorch torchvision torchaudio cpuonly -c pytorch
```

## 使用 Docker 安装 PyTorch

```bash
$ 
```

# Building from source

## macOS

- Prerequisites
    - Install Anaconda
    - Install CUDA, if your machine has a CUDA-enabled GPU.
    - Install optional dependencies:

```bash
$ export CMAKE_PREFIX_PATH=[anaconda root directory]
$ conda install numpy pyyaml mkl mkl-include setuptools cmake cffi typing
```

```bash
$ git clone --recursive https://github.com/pytorch/pytorch
$ cd pytorch
$ MACOSX_DEPLOYMENT_TARGET=10.9 CC=clang CXX=clang++ python setup.py install
```

- 当前，仅可以通过从源码构建 PyTorch 来获得 macOS 的 CUDA 支持

## Ubuntu

```bash
$ git clone 
```

# Verification

- Torch 使用:

```bash
>>> from __future__ import print_function
>>> import torch

>>> x = torch.rand(5, 3)
>>> print(x)

tensor([[0.3380, 0.3845, 0.3217],
	[0.8337, 0.9050, 0.2650],
	[0.2979, 0.7141, 0.9069],
	[0.1449, 0.1132, 0.1375],
	[0.4675, 0.3947, 0.1426]])
```

- GPU dirver 和 CUDA:

```bash
import torch
torch.cuda.is_available()
```

