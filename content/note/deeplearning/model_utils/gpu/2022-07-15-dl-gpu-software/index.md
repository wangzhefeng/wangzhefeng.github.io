---
title: GPU 软件
author: 王哲峰
date: '2022-07-15'
slug: dl-gpu-software
categories:
  - deeplearning
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

- [Ubuntu](#ubuntu)
  - [安装 Nvidia 显卡驱动](#安装-nvidia-显卡驱动)
  - [安装 CUDA](#安装-cuda)
    - [CUDA 介绍](#cuda-介绍)
    - [CUDA 安装](#cuda-安装)
  - [安装 cuDNN](#安装-cudnn)
  - [安装 PyTorch](#安装-pytorch)
- [Windows](#windows)
  - [查看系统信息](#查看系统信息)
  - [安装 Nvidia 显卡驱动](#安装-nvidia-显卡驱动-1)
  - [安装 CUDA 和 cuDNN](#安装-cuda-和-cudnn)
    - [安装 CUDA](#安装-cuda-1)
    - [安装 cuDNN](#安装-cudnn-1)
  - [安装 PyTorch](#安装-pytorch-1)
- [参考](#参考)
</p></details><p></p>
   
# Ubuntu

## 安装 Nvidia 显卡驱动

最简单的方式是通过系统的软件与更新来安装:

1. 进入系统的图形桌面，打开 ``Software & Updates`` 软件，可以看到标签栏有一个 ``Additional Drivers``:
      - NVIDIA Corporation: Unknown
          - Using NVIDIA dirver metapackage from nvidia-driver-455(proprietary, tested)
          - Using X.Org x server -- Nouveau display driver from xserver-xorg-video-nouveau(open source)
      - 选择第一个安装 Nvidia 官方驱动(第二个是开源驱动)即可，根据网络情况稍等大概十分钟，安装完重启服务器。
2. 重启完之后更新一下软件

```bash
sudo apt update
sudo apt upgrade
```

这里会连带 Nvidia 的驱动一起神级一遍，更新到最新的驱动；
更新完可能会出现 nvidia-smi 命令报错，再重启一遍就解决了

## 安装 CUDA

### CUDA 介绍

NVIDIA® CUDA® 工具包提供了开发环境，可供创建经 GPU 加速的高性能应用。
借助 CUDA 工具包，可以在经 GPU 加速的嵌入式系统、台式工作站、企业数据中心、
基于云的平台和 HPC 超级计算机中开发、优化和部署应用。
此工具包中包含多个 GPU 加速库、多种调试和优化工具、
一个 C/C++ 编译器以及一个用于在主要架构(包括 x86、Arm 和 POWER)
上构建和部署应用的运行时库。

借助多 GPU 配置中用于分布式计算的多项内置功能，
科学家和研究人员能够开发出可从单个 GPU 工作站扩展到配置数千个 GPU 的云端设施的应用。

### CUDA 安装

1. 如果之前安装了旧版本的 CUDA 和 cudnn 的话，需要先卸载后再安装, 卸载 CUDA:

```bash
# sudo apt-get remove --purge nvidia
```

然后重新安装显卡驱动，安装好了之后开始安装 CUDA

2. 下载 CUDA 安装包--CUDA Toolkit 11.0 Download|NVIDIA Developer
    - https://developer.nvidia.com/cuda-11.0-download-archive

![images](images/cuda.png)

3. 安装 CUDA

```bash
chmod +x cuda_11.0.2_450.51.05_linux.run
sudo sh ./cuda_11.0.2_450.51.05_linux.run
```

4. 配置 UDA 环境变量

```bash
vim ~/.bashrc
# or
vim ~/.zsh

export CUDA_HOME=/usr/local/cuda-11.0
export LD_LIBRARY_PATH=${CUDA_HOME}/lib64
export PATH=${CUDA_HOME}/bin/${PATH}
```

```bash
source ~/.bashrc
```

5. 查看安装的版本信息

```bash
nvcc -V
```

可以编译一个程序测试安装是否成功，执行以下几条命令:

```bash
cd ~/Softwares/cuda/NVIDIA_CUDA-11.0_Samples/1_Utilities/deviceQuery
make
./deviceQuery
```

## 安装 cuDNN

1. 下载 cuDNN 安装包--cuDNN Download|NVIDIA Developer
    - https://developer.nvidia.com/rdp/cudnn-download
    - 选择与 CUDA 版本对应的 cuDNN 版本
2. 安装 cuDNN

```bash
tar -xzvf cudnn-11.0-linux-x64-v8.0.5.39.tag
sudo cp cuda/lib64/* /usr/local/cuda-11.0/lib64/
sudo cp cuda/include/* /usr/local/cuda-11.0/include/
```

3. 查看 cuDNN 的版本信息

```bash
$ cat /usr/local/cuda/include/cudnn_version.h | grep CUDNN_MAJOR -A 2
```

## 安装 PyTorch

[PyTorch Org 安装](https://pytorch.org/get-started/locally/)

```bash
$ pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
$ conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia
```

PyTorch GPU 支持测试

```bash
import torch
torch.cuda.is_available()
```

# Windows

## 查看系统信息

* 系统：win10 64 位操作系统
* 安装组合：Miniconda + PyTorch(GPU version) + RTX 4070

## 安装 Nvidia 显卡驱动

* [显卡驱动程序安装](https://blog.csdn.net/A_Small_Man/article/details/126945715)

## 安装 CUDA 和 cuDNN

### 安装 CUDA

* [CUDA 与 cuDNN 安装教程](https://blog.csdn.net/anmin8888/article/details/127910084)

### 安装 cuDNN

* [CUDA 与 cuDNN 安装教程](https://blog.csdn.net/anmin8888/article/details/127910084)

## 安装 PyTorch

* [PyTorch Org 安装](https://pytorch.org/get-started/locally/)

```bash
$ pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
$ conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia
```

# 参考

* [深度学习环境配置指南(Windows、Mac、Ubuntu 全讲解)](https://mp.weixin.qq.com/s/ZTzfC7xp8PVMvOONVIiK6g)
* https://developer.ridgerun.com/wiki/index.php?title=Xavier/Processors/HDAV_Subsystem/Audio_Engine
* https://developer.nvidia.com/zh-cn/blog/bringing-cloud-native-agility-to-edge-ai-with-jetson-xavier-nx/
* https://jingyan.baidu.com/article/fdbd4277a447ebb89e3f48ca.html
* https://mp.weixin.qq.com/s/TsETgLLNWRskYbmh2wdiLg
* https://developer.nvidia.com/zh-cn/CUDA-toolkit
* https://developer.nvidia.com/zh-cn/CUDA-downloads
* https://docs.nvidia.com/CUDA/CUDA-quick-start-guide/index.html
* https://docs.nvidia.com/CUDA/CUDA-installation-guide-linux/
