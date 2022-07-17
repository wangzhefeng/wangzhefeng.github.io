---
title: DL 模型部署
author: 王哲峰
date: '2022-07-15'
slug: dl-model-deploy
categories:
  - deeplearning
tags:
  - note
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
</style>

# 模型部署简介

模型部署内容

- 中间表示 ONNX 的定义标准
- PyTorch 模型转换到 ONNX 模型的方法
- 推理引擎 ONNX Runtime、TensorRT 的使用方法
- 部署流水线 PyTorch => ONNX => ONNX Runtime/TensorRT 的示例及常见部署问题的解决方法
- MMDeploy C/C++ 推理 SDK







# 部署一个模型

> 用 PyTorch 实现一个模型，并把模型部署到 ONNX Runtime 推理引擎上

## 创建 PyTorch 模型

1. 首先，需要创建一个有 PyTorch 库的 Python 编程环境。

- conda [CPU only]

```bash
# 创建预安装 Python3.7 的名叫 deploy 的虚拟环境
$ conda create -n deploy python=3.7 -y
# 进入虚拟环境
$ conda activate deploy
$ conda install pytorch torchvision cpuonly -c pytorch
```

- conda [GPU]

```bash
# 安装 cuda 11.3 的 PyTorch
$ conda install pytorch torchvision cudatoolkit=11.3 -c pytorch
```

2. 安装其他第三方库

```bash
# 安装 ONNX Runtime, ONNX, OpenCV
$ conda install onnxruntime onnx opencv-python
```

3. 在一切配置完毕后，创建一个经典的超分辨率模型 SRCNN

```python
import os
import cv2
import numpy as np
import requests
import torch
import torch.onnx
from torch import nn


class SuperResolutionNet(nn.Module):
    """
    经典的超分辨率模型 SRCNN
    """
    def __init__(self, upscale_factor):
        super().__init__()
        # TODO
        self.upscale_factor = upscale_factor
        # TODO
        self.img_upsampler = nn.Upsample(
        	scale_factor = self.upscale_factor,
            mode = "bicuibc",
            align_corners = False
        )
        # CNN layers
        self.conv1 = nn.Conv2d(3, 64, kernel_size = 9, padding = 4)
        self.conv2 = nn.Conv2d(64, 32, kernel_size = 1, padding = 0)
        self.conv3 = nn.Conv2d(32, 3, kernel_size = 5, padding = 2)
        # ReLU layers
        self.relu = nn.ReLU()
        
    def forward(self, x):
        x = self.img_upsampler(x)
        out = self.relu(self.conv1(x))
        out = self.relu(self.conv2(out))
        out = self.conv3(out)
        return out


# download checkpoint and test image
urls = [
    'https://download.openmmlab.com/mmediting/restorers/srcnn/srcnn_x4k915_1x16_1000k_div2k_20200608-4186f232.pth',
    'https://raw.githubusercontent.com/open-mmlab/mmediting/master/tests/data/face/000001.png',
]
names = [
    "srcnn.pth",
    "face.png",
]

for url, name in zip(urls, names):
    if not os.path.exists(name):
        open(name, "wb").write(requests.get(url).content)

def init_torch_model():
    """
    初始化模型
    """
    torch_model = SuperResolutionNet(upscale_factor = 3)
    state_dict = torch.load("srcnn.pth")["state_dict"]
    # Adapt the checkpoint
    for old_key in list(state_dict.keys()):
        new_key = ".".join(old_key.split(".")[1:])
        state_dict[new_key] = state_dict.pop(old_key)
    torch_model.load_state_dict(state_dict)
    torch_model.eval()
    return torch_model


model = init_torch_model()
input_img = cv2.imread("face.png").astype(np.float32)

# HWC to NCHW
input_img = np.transpose(input_img, [2, 0, 1])
input_img = np.expand_dims(input_img, 0)

# Inference
torch_output = model(torch.from_numpy(input_img)).detach().numpy()

# NCHW to HWC
torch_output = np.squeeze(torch_output, 0)
torch_output = np.clip(torch_output, 0, 255)
torch_output = np.transpose(torch_output, [1, 2, 0]).astype(np.unit8)

# Show image
cv2.imwrite("face_torch.png", torch_output)
```

4. 在 PyTorch 模型测试正确后，接下来就是正式部署这个模型，所以下一步的任务就是把 PyTorch 模型转换成中间表示 ONNX 描述的模型

## 中间表示 -- ONNX

在介绍 ONNX 之前，先认识一下神经网络的结构。神经网络实际上只是描述了数据计算的过程，其结构可以用计算图表示。为了加速计算，一些框架会使用对神经网络 “先编译，后执行” 的静态图来描述网络。静态图的缺点是难以描述控制流，直接对其引入控制语句会导致产生不同的计算图。

ONNX (Open Neural Network Exchange) 是 Facebook 和微软在 2017 年共同发布的，用于标准描述计算图的一种格式。目前，在数家机构的共同维护下，ONNX 已经对接了多种深度学习框架和多种推理引擎。因此，ONNX 被当成了深度学习框架到推理引擎的桥梁，就像编译器的中间语言一样。由于各框架兼容性不一，通常只用 ONNX 表示更容易部署的静态图。





## 推理引擎 -- ONNX Runtime





# 总结

- 模型部署，指把训练好的模型在特定额环境中运行的过程。模型部署要解决模型框架兼容性差和模型运行速度慢这两个大问题
- 模型部署的常见流水线是：**深度学习框架 => 中间表示 => 推理引擎**。其中比较常用的一个中间表示是 ONNX
- 深度学习模型实际上就是一个计算图。模型部署时通常把模型转换成静态的计算图，即没有控制流（分支语句、循环语句）的计算图
- PyTorch 框架自带对 ONNX 的支持，只需要构造一组随机的输入，并对模型调用 `torch.onnx.export` 即可完成 PyTorch 到 ONNX 的转换
- 推理引擎 ONNX Runtime 对 ONNX 模型有原生的支持。给定一个 `.onnx` 文件，只需要简单使用 ONNX Runtime 的 Python API 就可以完成模型推理
