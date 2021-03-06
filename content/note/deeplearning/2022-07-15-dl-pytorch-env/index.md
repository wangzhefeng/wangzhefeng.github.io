---
title: PyTorch Env
author: 王哲峰
date: '2022-07-15'
slug: dl-pytorch-env
categories:
  - deeplearning
tags:
  - tool
---



# 1.PyTorch 支持的硬件平台

- PC
	- CPU
	- GPU
	- Cloud TPU(张量)
- Mobile
	- Android 
	- iOS
	- Embedded Devices

# 2.PyTorch 系统要求

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

# 3.macOS 安装 PyTorch

## 3.1 使用 pip 安装 PyTorch

```bash
$ pip install numpy
$ pip install torch torchvision
```

## 3.2 使用 Anaconda 安装 PyTorch

```bash
$ conda install pytorch torchvision -c pytorch
```

3.3 使用 Docker 安装 PyTorch

```bash
$ test test
```

# 4.Ubuntu 安装 Pytorch


## 4.1 使用 pip 安装 PyTorch

```bash
$ pip install torch==1.8.0+cpu torchvision==0.9.0+cpu torchaudio==0.8.0 -f https://download.pytorch.org/whl/torch_stable.html
```

## 4.2 使用 Anaconda 安装 PyTorch

```bash
$ conda install pytorch torchvision torchaudio cpuonly -c pytorch
```

## 4.3 使用 Docker 安装 PyTorch

```bash
$ test test
```

# 5.Building from source

## 5.1 macOS

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

## 5.2 Ubuntu

```bash
$ git clone 
```

# 6.Verification

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
