---
title: PyTorch-模型部署
author: wangzf
date: '2022-09-15'
slug: model-deploy
categories:
  - deeplearning
tags:
  - note
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

- [模型部署简介](#模型部署简介)
- [模型训练-PyTorch](#模型训练-pytorch)
- [中间表示-ONNX](#中间表示-onnx)
    - [ONNX 简介](#onnx-简介)
    - [ONNX 安装](#onnx-安装)
    - [ONNX 基础概念](#onnx-基础概念)
        - [计算图](#计算图)
        - [operator](#operator)
    - [PyTorch 导出 ONNX](#pytorch-导出-onnx)
        - [TorchDynamo-based ONNX Exporter](#torchdynamo-based-onnx-exporter)
        - [TorchScript-based ONNX Exporter](#torchscript-based-onnx-exporter)
- [推理引擎-ONNX Runtime](#推理引擎-onnx-runtime)
    - [ONNX Runtime 简介](#onnx-runtime-简介)
    - [ONNX Runtime 安装](#onnx-runtime-安装)
        - [CPU 版本](#cpu-版本)
        - [GPU 版本](#gpu-版本)
    - [ONNX Runtime 使用](#onnx-runtime-使用)
        - [ONNX Runtime 模型推理](#onnx-runtime-模型推理)
        - [ONNX Runtime 推理速度评估](#onnx-runtime-推理速度评估)
    - [ONNX Runtime 进阶使用](#onnx-runtime-进阶使用)
- [TensorRT](#tensorrt)
    - [TensorRT 简介](#tensorrt-简介)
- [参考](#参考)
</p></details><p></p>

# 模型部署简介

* 模型部署，指把训练好的模型在特定额环境中运行的过程。
  模型部署要解决**模型框架兼容性差**和**模型运行速度慢**这两个大问题
* 模型部署的常见流水线是：**深度学习框架 => 中间表示 => 推理引擎**，
  其中比较常用的一个中间表示是 ONNX
* 深度学习模型实际上就是一个计算图。模型部署时通常把模型转换成静态的计算图，
  即没有控制流（分支语句、循环语句）的计算图
* PyTorch 框架自带对 ONNX 的支持，只需要构造一组随机的输入，
  并对模型调用 `torch.onnx.export` 即可完成 PyTorch 到 ONNX 的转换
* 推理引擎 ONNX Runtime 对 ONNX 模型有原生的支持。给定一个 `.onnx` 文件，
  只需要简单使用 ONNX Runtime 的 Python API 就可以完成模型推理

# 模型训练-PyTorch

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

4. 在 PyTorch 模型测试正确后，接下来就是正式部署这个模型，
   所以下一步的任务就是把 PyTorch 模型转换成中间表示 ONNX 描述的模型。

# 中间表示-ONNX

在介绍 ONNX 之前，先认识一下神经网络的结构。神经网络实际上只是描述了数据计算的过程，
其结构可以用计算图表示。为了加速计算，一些框架会使用对神经网络 “先编译，后执行” 的静态图来描述网络。
静态图的缺点是难以描述控制流，直接对其引入控制语句会导致产生不同的计算图。

ONNX (Open Neural Network Exchange) 是 Facebook 和微软在 2017 年共同发布的，
用于标准描述计算图的一种格式。目前，在数家机构的共同维护下，ONNX 已经对接了多种深度学习框架和多种推理引擎。
因此，ONNX 被当成了深度学习框架到推理引擎的桥梁，就像编译器的中间语言一样。
由于各框架兼容性不一，通常只用 ONNX 表示更容易部署的静态图。

PyTorch 具有原生 ONNX 导出支持。

## ONNX 简介

在深度学习算法开发过程中，模型训练与部署是两个环节，PyTorch 通常只用于训练，获得模型权重文件，
而最终部署还有专门的部署平台，例如 TensorRT、NCNN、OpenVINO 等几十种部署推理平台。
如何将 PyTorch 模型文件让几十种部署推理平台能接收与读取是个大问题。
即使各推理平台都适配 PyTorch，那还有其他训练框架也要适配，是非常麻烦的。
假设有 N 个训练框架，M 个推理框架，互相要适配，那就是 `$O(NM)$` 的复杂度。
如果能有一种中间格式作为一个标注，能被所有框架所适配，那复杂度顺便降低为 `$O(N+M)$`。
ONNX 就是为了降低深度学习模型从训练到部署的复杂度，由微软和 Meta 在 2017 年提出的一种开放神经网络交换格式，
目的在于方便的将模型从一个框架转移到另一个框架。

ONNX(Open Neural Network Exchange，开放神经网络交换格式)是一种开放的、跨平台的深度学习模型交换格式，
可以方便地将模型从一个框架转移到另一个框架。ONNX 最初由微软和 meta 在 2017 年联合发布，
后来亚马逊也加入进来，目前已经成为行业共识，目前已经有 50 多个机构的产品支持 ONNX。

ONNX 最大的优点是简化了模型部署之间因框架的不同带来的繁琐事，这就像普通话。
在中国 129 种方言之间要互相通信是很困难的，解决办法就是设计一种可以与 129 种语言进行转换的语言——普通话。
ONNX 就是一个支持绝大多数主流机器学习模型格式之间转换的格式。

采用 PyTorch 进行模型开发时，部署环节通常将 PyTorch 模型转换为 ONNX 模型，
然后再进行其他格式转换，或者直接采用 ONNX 文件进行推理。

## ONNX 安装

```bash
$ pip install onnx
```

## ONNX 基础概念 

### 计算图

ONNX 文件是一种计算图，用于描述数据要进行何种计算，它就像是数学计算的语言，
可以进行计算的操作称之为操作符(operator)，一系列操作符构成一个计算图。
计算图中包含了各节点、输入、输出、属性的详细信息，有助于开发者观察模型结构。

下面通过一个线性回归模型的计算图来了解 ONNX 的计算图，可以采用 Python 代码构建 ONNX 计算图，
运行配套代码，构建了一个线性回归模型。

```python
from onnx import TensorProto
from onnx.helper import (
    make_model, 
    make_node,
    make_graph,
    make_tensor_value_info,
)

# tensor value info
# 'X' is the name, TensorProto.FLOAT the type, [None, None] the shape
X = make_tensor_value_info("X", TensorProto.FLOAT, [None, None])
A = make_tensor_value_info("A", TensorProto.FLOAT, [None, None])
B = make_tensor_value_info("B", TensorProto.FLOAT, [None, None])
Y = make_tensor_value_info("Y", TensorProto.FLOAT, [None])

# node
node1 = make_node("MatMul", ["X", "A"], ["XA"])
node2 = make_node("Add", ["XA", "B"], ["Y"])

# graph
graph = make_graph(
    [node1, node2],  # nodes
    "lr",  # name
    [X, A, B],  # inputs
    [Y]  # output
)

# model
onnx_model = make_model(graph)

with open("linear_regression.onnx", "wb") as f:
    f.write(onnx_model.SerializeToString())
```

导出的模型 `linear_regression.onnx` 可以在 [https://netron.app/](https://netron.app/) 中进行可视化：

![img](images/onnx-lr.png)

* `A`、`B`、`X`、`Y` 表示输入、输出数据
* 黑色的 `MatMul` 和 `Add` 是 Node，表示具体的操作
* `format`：表示生成该 ONNX 文件的 ONNX 版本
* `imports`：`operator`(算子)的版本。算子是 ONNX 中最重要的一个概念，
  大多数模型不成功是因为没有对应的算子，因此算子集的版本选择很重要
* `inputs` 和 `outputs`：是输入和输出，其中 `type` 是数据类型以及 `shape`

### operator

上面介绍了 ONNX 文件主要定义了计算图，计算图中的每个操作称为算子，
算子库的丰富程度，直接决定了 ONNX 可以表示模型的种类，
[ONNX 支持的算子有很多](https://onnx.ai/onnx/operators/index.html)。

对于普通用户，需要关注使用时的 `opset_version` 是哪个版本，目前最新版本是 `20`。
算子库可通过以下函数查看：

```python
import onnx

print(f"onnx_version = {onnx.version()}, opset={onnx.defs.onnx_opset_version()}")
```

## PyTorch 导出 ONNX

PyTorch 模型导出为 ONNX 调用 `torch.onnx.export` 函数即可，该函数包含很多参数，
这里只介绍几个常用的，更多的参考官方文档。

```python
torch.onnx.export(
    model, 
    args, 
    f, 
    export_params=True, 
    verbose=False, 
    training=, 
    input_names=None, 
    output_names=None, 
    operator_export_type=, 
    opset_version=None, 
    do_constant_folding=True, 
    dynamic_axes=None, 
    keep_initializers_as_inputs=None, 
    custom_opsets=None, 
    export_modules_as_functions=False
)
```

* `model`: 需要被转换的模型，可以有三种类型：`torch.nn.Module`、`torch.jit.ScriptModule`、`torch.jit.ScriptFunction`
* `args`：`model` 输入时所需要的参数，这里要传参时因为构建计算图过程中，
  需要采用数据对模型进行一遍推理，然后记录推理过程需要的操作，
  然后生成计算图。`args` 要求是 tuple 或者是 Tensor 的形式。
  一般只有一个输入时，直接传入 Tensor，多个输入时要用 tuple 包起来
* `export_params`: 是否需要保存参数。默认为 `True`，通常用于模型结构迁移到其它框架时用 `False`
* `input_names`：输入数据的名字，(list of str, default empty list) ，
  在使用 ONNX 文件时，数据的传输和使用，都是通过 `name: value` 的形式
* `output_names`：同上
* `opset_version`：使用的算子集版本
* `dynamic_axes`：
    - 动态维度的指定，例如 `batch_size` 在使用时随时会变，则需要把该维度指定为动态的。
      默认情况下计算图的数据维度是固定的，这有利于效率提升，但缺乏灵活性
    - 用法是，对于动态维度的输入、输出，需要设置它哪个轴是动态的，并且为这个轴设定名称。
      这里有 3 个要素，数据名称、轴序号、轴名称。因此是通过 dict 来设置的。
      例如 `dynamic_axes={"x": {0: "my_custom_axis_name"}}`，表示名称为 `x` 的数据，第 `0` 个轴是动态的，
      动态轴的名字叫 `my_custom_axis_name`。通常用于 `batchsize` 或者是对于 `h`，`w` 是不固定的模型要设置动态轴

下面以 ResNet50 为例，导出一个在 ImageNet 上训练好的分类模型进行观察：

```python
import torch
import torchvision

model = torchvision.models.resnet50(
    weights = torchvision.models.ResNet50_Weights.IMAGENET1K_V1
)

op_set = 13
dummy_data = torch.randn((1, 3, 224, 224))
dummdy_data_128 = torch.randn((128, 3, 224, 224))

# 固定 batch = 1
torch.onnx.export(
    model,
    (dummy_data),
    "resnet50_bs_1.onnx",
    opset_version = op_set,
    input_names = ["input"],
    output_names = ["output"],
)
# 固定 batch = 128
torch.onnx.export(
    model,
    (dummdy_data_128),
    "resnet50_bs_128.onnx",
    opset_version = op_set,
    input_names = ["input"],
    output_names = ["output"],
)
# 动态 batch
torch.onnx.export(
    model,
    (dummy_data),
    "resnet50_bs_dynamic.onnx",
    opset_version = op_set,
    input_names = ["input"],
    output_names = ["output"],
    dynamic_axes = {
        "input": {0: "batch_axes"},
        "output": {0: "batch_axes"},
    },
)
```

### TorchDynamo-based ONNX Exporter

```python
import torch
import torch.onnx as ONNX
```

### TorchScript-based ONNX Exporter

# 推理引擎-ONNX Runtime

## ONNX Runtime 简介

ONNX 是一个开放式的格式，它还需要放到推理框架（推理引擎）上运行才可以，
支持运行 ONNX 文件的框架有 ONNX Rruntime、TensorRT、PyTorch、TensorFlow 等等。
其中，ONNX Runtime 是 ONNX 官方的推理框架，它与 ONNX 库是两个东西。

## ONNX Runtime 安装

ONNX 官方的推理框架，它与 ONNX 库是两个东西。
安装了 `onnx` 库并没有安装上 `onnxruntime`，它需要额外安装。
`onnxruntime` 分为 CPU 版和 GPU 版，两个版本的安装又分别是两个库，
分别是 `onnxruntime`、`onnxruntime-gpu`。
`onnxruntime-gpu` 的安装，又要求 cuda、cudnn 版本的严格匹配，否则会无法运行。

### CPU 版本

```bash
$ pip install onnxruntime
```

### GPU 版本

对于 GPU 版本的安装，通常不能直接 `pip install onnxruntime-gpu`，而是要设置指定版本，
因为 cuda 和 cudnn 版本会限制 `onnxruntime` 的版本。
版本的对应关系见[官网](https://onnxruntime.ai/docs/execution-providers/CUDA-ExecutionProvider.html#requirements)。
例如，cuda 版本是 11.4，且 cudnn 是 8.2.2.26，则可以 `pip install onnxruntime-gpu==1.10.0`。
如果不是，那就需要配置对应版本的 cuda、cudnn 了。

通常来说，系统上 cuda 和 cudnn 的安装比较麻烦，并且更换版本也不方便。
这里推荐直接在 Python 虚拟环境中安装指定版本的 cuda、cudnn，
这样就不会与系统的 cuda、cudnn 冲突了。

```bash
# cuda version=12.1
# $ conda install cudatoolkit=12.1 -c pytorch -c conda-forge
# cudnn version=8.9.2
# $ conda install cudnn==8.9.2
# ONNX Runtime versio=1.18.0
$ pip install onnxruntime-gpu==1.18.0
```

需要注意的是，`onnxruntime` 和 `onnxruntime-gpu` 不可并存，
装了 `onnxruntime-gpu` 也是可以调用 CPU 版本的，
这里建议把 `onnxruntime` 卸载，只保留 `onnxruntime-gpu` 即可。

## ONNX Runtime 使用

### ONNX Runtime 模型推理

`onnxruntime` 中使用 ONNX 文件，只需要将其加载到 `InferenceSession` 中，
然后调用 `InferenceSession.run()`就可以完成推理。
相比于 PyTorch，不需要在代码中保留如何定义模型的的 `class`，
也不用加载权重了，这一切都存储在 ONNX 的计算图中。

`InferenceSession` 的初始化细节如下所示：

```python
class InferenceSession(Session):
    """
    This is the main class used to run a model.
    """

    def __init__(self, path_or_bytes, sess_options=None, providers=None, provider_options=None, **kwargs):
        """
        :param path_or_bytes: filename or serialized ONNX or ORT format model in a byte string
        :param sess_options: session options
        :param providers: Optional sequence of providers in order of decreasing
            precedence. Values can either be provider names or tuples of
            (provider name, options dict). If not provided, then all available
            providers are used with the default precedence.
        :param provider_options: Optional sequence of options dicts corresponding
            to the providers listed in 'providers'.
```

在这里，需要关注的是 `providers`，它的作用是指定可用的设备，
如 `["CUDAExecutionProvider", "CPUExecutionProvider", "ROCMExecutionProvider"]`。

```python
ort_session_bs1 = ort.InferenceSession(
    'resnet50_bs_1.onnx', 
    providers = ['CUDAExecutionProvider']
)
inp = np.random.randn(1, 3, 224, 224).astype(np.float32)

output = model.run(
    ['output'], 
    {'input': inp}
)
```

完整的 ResNet50 实现图像分类推理见下面的代码，需要注意的是要与模型训练时的前处理、后处理保持一致。

```python
import json
from PIL import Image
import time
import numpy as np
import onnxruntime as ort
from tqdm import tqdm
import matplotlib.pyplot as plt

print(ort.get_device())


def preprocess(input_data):
    # convert the input data into the float32 input
    img_data = input_data.astype('float32')

    # normalize
    mean_vec = np.array([0.485, 0.456, 0.406])
    stddev_vec = np.array([0.229, 0.224, 0.225])
    norm_img_data = np.zeros(img_data.shape).astype('float32')
    for i in range(img_data.shape[0]):
        norm_img_data[i, :, :] = (img_data[i, :, :] / 255 - mean_vec[i]) / stddev_vec[i]

    # add batch channel
    norm_img_data = norm_img_data.reshape(1, 3, 224, 224).astype('float32')
    return norm_img_data


def softmax(x):
    x = x.reshape(-1)
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum(axis=0)


def postprocess(result):
    return softmax(np.array(result))


def load_class_names(p_clsnames, p_clsnames_cn):
    """
    加载标签名
    :param p_clsnames:
    :param p_clsnames_cn:
    :return:
    """
    with open(p_clsnames, "r") as f:
        class_names = json.load(f)
    with open(p_clsnames_cn, encoding='UTF-8') as f:  # 设置文件对象
        class_names_cn = f.readlines()
    return class_names, class_names_cn


if __name__ == '__main__':

    path_img = r'G:\deep_learning_data\coco128\images\train2017\000000000081.jpg'
    path_classnames = "imagenet1000.json"
    path_classnames_cn = "imagenet_classnames.txt"

    # load class names
    cls_n, cls_n_cn = load_class_names(path_classnames, path_classnames_cn)
    # 初始化模型
    ort_session = ort.InferenceSession('resnet50_bs_1.onnx', providers=['CUDAExecutionProvider'])

    # 图片读取
    image = Image.open(path_img).resize((224, 224))
    img_rgb = np.array(image)
    image_data = img_rgb.transpose(2, 0, 1)
    input_data = preprocess(image_data)

    # 推理
    raw_result = ort_session.run([], {'input': input_data})
    res = postprocess(raw_result)  # 后处理 softmax

    def topk(array, k=1):
        index = array.argsort()[::-1][:k]
        return index

    top5_idx = topk(res, k=5)

    # 结果可视化
    pred_str, pred_cn = cls_n[top5_idx[0]], cls_n_cn[top5_idx[0]]
    print("img: {} is: {}, {}".format(path_img, pred_str, pred_cn))
    plt.imshow(img_rgb)
    plt.title("predict:{}".format(pred_str))
    text_str = [cls_n[t] for t in top5_idx]
    for idx in range(len(top5_idx)):
        plt.text(5, 15+idx*15, "top {}:{}".format(idx+1, text_str[idx]), bbox=dict(fc='yellow'))
    # plt.savefig("tmp.png")
    plt.show()
```

### ONNX Runtime 推理速度评估

通常说推理速度，只看一次推理的耗时是不足以反应模型在生产时的效率的，
因为推理并行的存在，因此可以采用大的 batch size 来提高单位时间内处理样本的数量。

通常评估模型的推理的时间效率会将时延(latency)和吞吐量(throughout)一起观察。
这里简单介绍时延(latency)和吞吐量(throughout)的意义。

* 时延(latency)：通常用于评估用户需要等待多长时间，根据业务场景，
  需要针对性保障时延，约等于平时说的耗时。
* 吞吐量(throughout)：用于评估服务器一定时间内能处理的量，通常是为了提高单位时间内，
  能处理更多的用户请求。

时延和吞吐量通常是矛盾的，即想要高吞吐的时候，时延就会提高。
这个就像深夜的大排档，你到店里点一份炒河粉，需要等待多久？
这取决于老板的策略是低延时，还是高吞吐。

* 低延时策略：来一个处理一个，尽快把你的一份河粉炒出来，需要 3 分钟。
* 高吞吐策略：稍微等等，等到 3 个炒河粉订单一次性炒出来，等了 3 分钟，
  炒粉 3 分钟，总共 6 分钟，算下来，每分钟可以炒 0.5 份。
  而低时延策略的吞吐量显然低了，每分钟可以炒 0.33 份。

计算机的运行也是一样的，可以通过 batch size 来权衡时延与吞吐量。

为了观察 batch size 对推理效率的影响，这里设计了三个模型的对比实验，
分别是 `bs=1`，`bs=128`, `bs` 为动态时，从 1 到 256 的推理时延与吞吐量的对比。

```python
# -*- coding:utf-8 -*-
"""
resnet50 固定bs1， bs128， 动态bs的推理速度评估
使用说明：需要将bs1, bs128 与 动态bs的实验分开跑，即需要运行两次。（在下边手动注释，切换模型）
原因：onnxruntime不会自动释放显存，导致显存不断增长，6G的显存扛不住。
"""

import time
import numpy as np
import onnxruntime as ort
from tqdm import tqdm
import matplotlib.pyplot as plt

print(ort.get_device())


def speed_test(bs, model, model_name):
    print(f"start: bs {bs}, model_name {model_name}")
    inp = np.random.randn(bs, 3, 224, 224).astype(np.float32)
    loop_times = datasize / bs

    # warmup
    _ = model.run(['output'], {'input': inp})

    time_s = time.time()
    for i in tqdm(range(int(loop_times))):
        _ = model.run(['output'], {'input': inp})
    time_consumed = time.time() - time_s

    latency = time_consumed / loop_times * 1000
    throughput = 1 / (time_consumed / datasize)

    print("model_name: {} bs: {} latency: {:.1f} ms, throughput: {:.0f} frame / s".format(
        model_name, bs, latency, throughput))
    return latency, throughput


if __name__ == '__main__':

    datasize = 1280

    # Load the ONNX model

    ort_session_bs1 = ort.InferenceSession('resnet50_bs_1.onnx', providers=['CUDAExecutionProvider'])
    ort_session_bs128 = ort.InferenceSession('resnet50_bs_128.onnx', providers=['CUDAExecutionProvider'])
    ort_session_dynamic = ort.InferenceSession('resnet50_bs_dynamic.onnx', providers=['CUDAExecutionProvider'])

    # 测试固定 batch size, 由于onnx不会释放显存，所以把3个模型拆开推理
    bs_list = [1, 128]
    model_names = ['bs1', 'bs128']
    model_list = [ort_session_bs1, ort_session_bs128]
    model_container = dict(zip(model_names, model_list))

    # 测试动态 batch size
    # bs_list = list(map(lambda x: 2**x,  range(0, 8)))
    # model_names = ['bs_dynamic']
    # model_list = [ort_session_dynamic]
    # model_container = dict(zip(model_names, model_list))

    info_dict = {}
    for model_name in model_names:
        for bs in bs_list:
            if bs != 1 and model_name == 'bs1':
                continue
            if bs != 128 and model_name == 'bs128':
                continue
            latency, throughput = speed_test(bs, model_container[model_name], model_name)

            info_dict[model_name + str(bs)] = (latency, throughput)

    throughput_list = [v[1] for v in info_dict.values()]
    plt.plot(bs_list, throughput_list, marker='o', linestyle='--')
    for a, b in zip(bs_list, throughput_list):
        plt.text(a, b, f'{b:.2f}', ha='center', va='bottom', fontsize=10)
    plt.title('Throughput frame/s')
    plt.show()
```

结论：

1. 随着 batch size 的增加，吞吐量逐步提高，在 `bs=128` 时，吞吐量增长平缓；
2. cpu 上推理，batch size 的增加，吞吐量差别不大，这也符合逻辑，
   毕竟 cpu 不是计算型处理器，无法批量处理大规模矩阵运算；
3. 不定 batch size 的模型与动态 batch size 的模型，在相同 batch size 下，
   效率并没有什么变化（注：由于变化没差别，表格中没有展示）；
4. 在 `onnruntime` 有一些奇怪的 `bs`，当 `bs=16`，`bs=256` 时，运行效率出现异常，详情看表格；

建议：模型上线前，实际评测一下模型不同输入时的效率，选择合适的 batch size，可以最大化服务器利用率。

## ONNX Runtime 进阶使用

# TensorRT

## TensorRT 简介

TensorRT 是 NVIDIA 公司针对 N 卡推出的高性能深度学习推理框架，
TensorRT 采用 C++ 编写底层库，并提供 C++/Python 应用接口，
实现了高吞吐、低时延的优点。TensorRT 应用量化、图优化、层融合等优化技术，
同时利用高度优化的内核找到该模型的最快实现。
TensorRT 是官方推理引擎，优化效果自然靠谱，因此使用 TensorRT 进行工程化部署已经成为主流方案。




# 参考

* [ONNX 教程](https://github.com/ONNX/tutorials)
* [PyTorch ONNX](https://pytorch.org/docs/stable/ONNX.html#)
* [ONNX 算子](https://zhuanlan.zhihu.com/p/479290520%EF%BC%9A%E8%AE%B2%E8%A7%A3%E4%BA%86pytorch%E8%BD%AConnx%E6%97%B6%EF%BC%8C%E6%AF%8F%E4%B8%80%E4%B8%AA%E6%93%8D%E4%BD%9C%E6%98%AF%E5%A6%82%E4%BD%95%E8%BD%AC%E6%8D%A2%E5%88%B0onnx%E7%AE%97%E5%AD%90%E7%9A%84%EF%BC%9B%E4%BB%8B%E7%BB%8D%E4%BA%86%E7%AE%97%E5%AD%90%E6%98%A0%E5%B0%84%E5%85%B3%E7%B3%BB)
* [ONNX 算子](https://zhuanlan.zhihu.com/p/498425043%EF%BC%9A%E8%AE%B2%E8%A7%A3%E4%BA%86pytorch%E8%BD%AConnx%E6%97%B6%EF%BC%8Ctrace%E5%92%8Cscript%E4%B8%A4%E7%A7%8D%E6%A8%A1%E5%BC%8F%E4%B8%8B%E7%9A%84%E5%8C%BA%E5%88%AB%EF%BC%9B%E4%BB%A5%E5%8F%8Atorch.onnx.export()%E5%87%BD%E6%95%B0%E7%9A%84%E4%BD%BF%E7%94%A8%EF%BC%9B)
* [ONNX 算子](https://zhuanlan.zhihu.com/p/513387413%EF%BC%9A%E8%AE%B2%E8%A7%A3%E4%BA%86%E4%B8%89%E7%A7%8D%E6%B7%BB%E5%8A%A0%E7%AE%97%E5%AD%90%E7%9A%84%E6%96%B9%E6%B3%95)

