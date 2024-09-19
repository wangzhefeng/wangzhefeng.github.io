---
title: LightGBM 安装
author: 王哲峰
date: '2024-09-18'
slug: ml-gbm-lightgbm-install
categories:
  - machinelearning
tags:
  - model
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

- [pip 安装](#pip-安装)
- [从源码安装](#从源码安装)
  - [Linux](#linux)
  - [Windows](#windows)
  - [macOS](#macos)
</p></details><p></p>

LightGBM 的安装非常简单，在 Linux 下很方便的就可以开启 GPU 训练。可以优先选用从 pip 安装，如果失败再从源码安装。

# pip 安装

```bash
pip install lightgbm

pip install --no-binary :all: lightgbm # 从源码编译安装

pip install lightgbm --install-option=--mpi # 从源码编译安装 MPI 版本

pip install lightgbm --install-option=--gpu # 从源码编译安装 GPU 版本

pip install lightgbm --install-option=--gpu --install-option="--opencl-include-dir=/usr/local/cuda/include/" --install-option="--opencl-library=/usr/local/cuda/lib64/libOpenCL.so" # 从源码编译安装，指定配置
# 可选的配置有：
#    boost-root
#    boost-dir
#    boost-include-dir
#    boost-librarydir
#    opencl-include-dir
#    opencl-library
```

# 从源码安装

## Linux

1. 下载源码

```bash
$ git clone --recursive https://github.com/microsoft/LighGBM
```

2. 编译 `lib_lightgbm.so`

```bash
$ cd LightGBM
$ mkdir build
$ cd build
$ cmake ..

# 开启 MPI 通信机制，训练更快：
$ cmake -DUSE_MPI=ON ..

# GPU 版本，训练更快：
$ cmake -DUSE_GPU=1 ..

# 如果安装了 NVIDIA OpenGL，则使用
$ cmake -DUSE_GPU=1 -DOpenCL_LIBRARY=/usr/local/cuda/lib64/libOpenCL.so -DOpenCL_INCLUDE_DIR=/usr/local/cuda/include/ ..

# 如果想要 protobuf 来保存和加载模型，则先安装 `protobuf c++` 版本，然后使用
$ cmake -DUSE_PROTO=ON ..

$ make -j4
```

3. 安装 Python 支持

```bash
$ cd python-package
$ sudo python setup.py install --precompile
```

## Windows

1. 下载源码

```bash
$ git clone --recursive https://github.com/microsoft/LighGBM
$ git submodule init
$ git submodule update
```

2. 编译 `lib_lightgbm.dll`

```bash
$ cd LightGBM
$ mkdir build
$ cd build
$ cmake ..

# 开启 MPI 通信机制，训练更快：
$ cmake -DUSE_MPI=ON ..

# GPU 版本，训练更快：
$ cmake -DUSE_GPU=1 ..

# 如果安装了 NVIDIA OpenGL，则使用
$ cmake -DUSE_GPU=1 -DOpenCL_LIBRARY=/usr/local/cuda/lib64/libOpenCL.so -DOpenCL_INCLUDE_DIR=/usr/local/cuda/include/ ..

# 如果想要 protobuf 来保存和加载模型，则先安装 `protobuf c++` 版本，然后使用
$ cmake -DUSE_PROTO=ON ..

$ make -j4
```

3. 安装 Python 支持

```bash
$ cd python-package
$ python setup.py install --precompile
```

## macOS
