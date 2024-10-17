---
title: PyTorch 环境
author: wangzf
date: '2022-07-11'
slug: dl-pytorch-env
categories:
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
img {
    pointer-events: none;
}
</style>

<details><summary>目录</summary><p>

- [Windows 安装 PyTorch](#windows-安装-pytorch)
    - [CPU version](#cpu-version)
        - [使用 pip 安装 PyTorch](#使用-pip-安装-pytorch)
        - [使用 conda 安装 PyTorch](#使用-conda-安装-pytorch)
        - [使用 Docker 安装 PyTorch](#使用-docker-安装-pytorch)
    - [GPU-CUDA version](#gpu-cuda-version)
        - [使用 pip 安装 PyTorch](#使用-pip-安装-pytorch-1)
        - [使用 conda 安装 PyTorch](#使用-conda-安装-pytorch-1)
        - [使用 Docker 安装 PyTorch](#使用-docker-安装-pytorch-1)
- [macOS 安装 PyTorch](#macos-安装-pytorch)
    - [使用 pip 安装 PyTorch](#使用-pip-安装-pytorch-2)
    - [使用 conda 安装 PyTorch](#使用-conda-安装-pytorch-2)
    - [使用 Docker 安装 PyTorch](#使用-docker-安装-pytorch-2)
- [Linux 安装 PyTorch](#linux-安装-pytorch)
    - [CPU version](#cpu-version-1)
        - [使用 pip 安装 PyTorch](#使用-pip-安装-pytorch-3)
        - [使用 conda 安装 PyTorch](#使用-conda-安装-pytorch-3)
        - [使用 Docker 安装 PyTorch](#使用-docker-安装-pytorch-3)
    - [GPU-CUDA version](#gpu-cuda-version-1)
        - [使用 pip 安装 PyTorch](#使用-pip-安装-pytorch-4)
        - [使用 conda 安装 PyTorch](#使用-conda-安装-pytorch-4)
        - [使用 Docker 安装 PyTorch](#使用-docker-安装-pytorch-4)
- [Verification](#verification)
- [官网安装介绍](#官网安装介绍)
</p></details><p></p>


# Windows 安装 PyTorch

## CPU version

### 使用 pip 安装 PyTorch

```bash
$ pip3 install torch torchvision torchaudio
```

### 使用 conda 安装 PyTorch

```bash
$ conda install pytorch torchvision torchaudio cpuonly -c pytorch
```

### 使用 Docker 安装 PyTorch

```bash
$ TODO
```

## GPU-CUDA version

### 使用 pip 安装 PyTorch

```bash
# CUDA 11.3
$ pip3 install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu113
# CUDA 11.6
$ pip3 install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu116
# CUDA 11.8
$ pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
# CUDA 12.1
$ pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

### 使用 conda 安装 PyTorch

```bash
# CUDA 11.3
$ conda install pytorch torchvision torchaudio cudatoolkit=11.3 -c pytorch
# CUDA 11.6
$ conda install pytorch torchvision torchaudio cudatoolkit=11.6 -c pytorch -c conda-forge
# CUDA 11.8
$ conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia
# CUDA 12.1
$ conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia
```

### 使用 Docker 安装 PyTorch

```bash
$ TODO
```

# macOS 安装 PyTorch

## 使用 pip 安装 PyTorch

```bash
$ pip3 install torch torchvision torchaudio
```

## 使用 conda 安装 PyTorch

```bash
$ conda install pytorch::pytorch torchvision torchaudio -c pytorch
```

## 使用 Docker 安装 PyTorch

```bash
$ TODO
```

# Linux 安装 PyTorch

## CPU version

### 使用 pip 安装 PyTorch

```bash
$ pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
```

### 使用 conda 安装 PyTorch

```bash
$ conda install pytorch torchvision torchaudio cpuonly -c pytorch
```

### 使用 Docker 安装 PyTorch

```bash
$ TODO
```

## GPU-CUDA version

### 使用 pip 安装 PyTorch

```bash
# CUDA 10.2
$ pip3 install torch torchvision torchaudio
# CUDA 11.3
$ pip3 install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu113
# CUDA 11.6
$ pip3 install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu116
# ROCm 5.1.1
$ pip3 install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/rocm5.1.1

# CUDA 11.8
$ pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
# CUDA 12.1
$ pip3 install torch torchvision torchaudio
# ROCm 5.6
$ pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/rocm5.6
```

### 使用 conda 安装 PyTorch

```bash
# CUDA 10.2
$ conda install pytorch torchvision torchaudio cudatoolkit=10.2 -c pytorch
# CUDA 11.3
$ conda install pytorch torchvision torchaudio cudatoolkit=11.3 -c pytorch
# CUDA 11.6
$ conda install pytorch torchvision torchaudio cudatoolkit=11.6 -c pytorch -c conda-forge
# CUDA 11.8
$ conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia
# CUDA 12.1
$ conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia
```

### 使用 Docker 安装 PyTorch

```bash
$ TODO
```

# Verification

* Torch 使用:

```bash
>>> import torch
>>> x = torch.rand(5, 3)
>>> print(x)

tensor([[0.3380, 0.3845, 0.3217],
	[0.8337, 0.9050, 0.2650],
	[0.2979, 0.7141, 0.9069],
	[0.1449, 0.1132, 0.1375],
	[0.4675, 0.3947, 0.1426]])
```

* GPU dirver 和 CUDA:

```python
>>> import torch
>>> torch.cuda.is_available()
```

# 官网安装介绍

* [Install PyTorch](https://pytorch.org/get-started/locally/)
* [Download Pytorch](https://download.pytorch.org/whl/torch)
