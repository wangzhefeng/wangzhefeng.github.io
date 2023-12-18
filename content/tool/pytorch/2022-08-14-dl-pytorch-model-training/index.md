---
title: PyTorch 模型训练
author: 王哲峰
date: '2022-08-14'
slug: dl-pytorch-model-compile
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

- [模型训练简介](#模型训练简介)
  - [MNIST 数据集](#mnist-数据集)
  - [神经网络示例](#神经网络示例)
  - [损失函数、优化器、评价指标](#损失函数优化器评价指标)
- [脚本风格](#脚本风格)
- [函数风格](#函数风格)
- [类风格](#类风格)
  - [KerasModel](#kerasmodel)
  - [LightModel](#lightmodel)
</p></details><p></p>

# 模型训练简介

PyTorch 通常需要用户编写自定义训练循环，训练循环的代码风格因人而异。有三类典型的训练循环代码风格：

* 脚本形式训练循环
* 函数形式训练循环
* 类形式训练循环

还有一种第三方用户编写的风格，通用的 Keras 风格的脚本形式训练循环，需要安装相应的第三方库：

```bash
$ pip install torchkeras
```

## MNIST 数据集

```python
import os
import sys
import time
import datetime
from tqdm import tqdm
from copy import deepcopy

import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt

import torch
from torch import nn
import torchvision
from torchvision import datasets
from torchvision import transforms
from torchmetrics import Accuracy

print(f"torch.__version__ = {torch.__version__}")
device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

# 注：
#   多分类使用 torchmetrics 中的评估指标
#   二分类使用 torchkeras.metrics 中的评估指标

# log
def printlog(info):
    nowtime = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    print("\n" + "==========" * 8 + "%s" % nowtime)
    print(str(info) + "\n")
```

```python
# 数据转换
transform = transforms.Compose([
    transforms.ToTensor()
])
# 数据集
ds_train = datasets.MNIST(
    root = "./data/minist/", 
    train = True,
    download = True,
    transform = transform
)
ds_val = datasets.MNIST(
    root = "./data/minist/",
    train = False,
    download = True,
    transform = transform
)
# 数据加载器
dl_train =  torch.utils.data.DataLoader(
    ds_train, 
    batch_size = 128, 
    shuffle = True, 
    num_workers = 4
)
dl_val =  torch.utils.data.DataLoader(
    ds_val, 
    batch_size = 128, 
    shuffle = False, 
    num_workers = 4
)
```

```python
# %matplolib inline
# %config InlineBackend.figure_format = "svg"
plt.figure(figsize = (8, 8))
for i in range(9):
    # data
    img, label = ds_train[i]
    img = torch.squeeze(img).numpy()
    # plot
    ax = plt.subplot(3, 3, i + 1)
    ax.imshow(img)
    ax.set_title(f"label = {label}")
    ax.set_xticks([])
    ax.set_yticks([])
plt.show()
```

![img](images/mnist.png)

## 神经网络示例

```python
class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        self.layers = nn.ModuleList([
            nn.Conv2d(
                in_channels = 1, 
                out_channels = 32, 
                kernel_size = 3
            ),
            nn.MaxPool2d(kernel_size = 2, stride = 2),
            nn.Conv2d(
                in_channels = 32, 
                out_channels = 64, 
                kernel_size = 5
            ),
            nn.MaxPool2d(kernel_size = 2, stride = 2),
            nn.Dropout2d(p = 0.1),
            nn.AdaptiveMaxPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 10),
        ])
    
    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x
    
net = Net()
print(net)
```

## 损失函数、优化器、评价指标

```python
# 损失函数
loss_fn = nn.CrossEntropyLoss()
# 优化器
optimizer = torch.optim.Adam(net.parameters(), lr = 0.01)
# 评价指标
metrics_dict = {
    "acc": Accuracy(task = "binray"),
}
```

# 脚本风格

脚本风格的训练循环最为常见

```python
# epochs 相关设置
epochs = 20 
ckpt_path = 'checkpoint.pt'

# early_stopping 相关设置
monitor = "val_acc"
patience = 5
mode = "max"

# 训练循环
history = {}
for epoch in range(1, epochs + 1):
    printlog(f"Epoch {epoch} / {epochs}")
    # ------------------------------
    # 1.train
    # ------------------------------
    net.train()

    total_loss, step = 0, 0    
    loop = tqdm(enumerate(dl_train), total = len(dl_train))
    train_metrics_dict = deepcopy(metrics_dict) 
    for i, batch in loop: 
        features, labels = batch
        # forward
        preds = net(features)
        loss = loss_fn(preds, labels)
        # backward
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()    
        # metrics
        step_metrics = {
            "train_" + name: metric_fn(preds, labels).item()
            for name, metric_fn in train_metrics_dict.items()
        }
        step_log = dict({
            "train_loss": loss.item()
        }, **step_metrics)
        # 总损失和训练迭代次数更新
        total_loss += loss.item()
        step += 1

        if i != len(dl_train) - 1:
            loop.set_postfix(**step_log)
        else:
            epoch_loss = total_loss / step
            epoch_metrics = {
                "train_" + name: metric_fn.compute().item() 
                for name, metric_fn in train_metrics_dict.items()
            }
            epoch_log = dict({
                "train_loss": epoch_loss
            }, **epoch_metrics)
            loop.set_postfix(**epoch_log)
            for name, metric_fn in train_metrics_dict.items():
                metric_fn.reset()
    for name, metric in epoch_log.items():
        history[name] = history.get(name, []) + [metric]
    # ------------------------------
    # 2.validate
    # ------------------------------
    net.eval()    

    total_loss, step = 0, 0
    loop = tqdm(enumerate(dl_val), total =len(dl_val))    
    val_metrics_dict = deepcopy(metrics_dict)     
    with torch.no_grad():
        for i, batch in loop: 
            features, labels = batch            
            #forward
            preds = net(features)
            loss = loss_fn(preds, labels)
            # metrics
            step_metrics = {
                "val_" + name: metric_fn(preds, labels).item() 
                for name,metric_fn in val_metrics_dict.items()
            }
            step_log = dict({
                "val_loss": loss.item()
            }, **step_metrics)
            # 总损失和训练迭代次数更新
            total_loss += loss.item()
            step9 += 1
            if i != len(dl_val) - 1:
                loop.set_postfix(**step_log)
            else:
                epoch_loss = (total_loss / step)
                epoch_metrics = {
                    "val_" + name: metric_fn.compute().item() 
                    for name, metric_fn in val_metrics_dict.items()
                }
                epoch_log = dict({
                    "val_loss": epoch_loss
                }, **epoch_metrics)
                loop.set_postfix(**epoch_log)
                for name, metric_fn in val_metrics_dict.items():
                    metric_fn.reset()
    epoch_log["epoch"] = epoch           
    for name, metric in epoch_log.items():
        history[name] = history.get(name, []) + [metric]
    # ------------------------------
    # 3.early-stopping
    # ------------------------------
    arr_scores = history[monitor]
    best_score_idx = np.argmax(arr_scores) if mode == "max" else np.argmin(arr_scores)
    
    if best_score_idx == len(arr_scores) - 1:
        torch.save(net.state_dict(), ckpt_path)
        print(f"<<<<<< reach best {monitor} : {arr_scores[best_score_idx]} >>>>>>", file = sys.stderr)
    
    if len(arr_scores) - best_score_idx > patience:
        print(f"<<<<<< {monitor} without improvement in {patience} epoch, early stopping >>>>>>", file = sys.stderr)
        break
    net.load_state_dict(torch.load(ckpt_path))

# 模型训练结果
dfhistory = pd.DataFrame(history)
```

# 函数风格

函数风格是在脚本风格形式的基础上做了进一步的函数封装

```python
class StepRunner:

    def __init__(self,
                 net, 
                 loss_fn,
                 stage = "train",
                 metrics_dict = None,
                 optimizer = None,
                 lr_scheduler = None,
                 accelerator = None):
        self.net = net
        self.loss_fn = loss_fn
        self.metrics_dict = metrics_dict
        self.stage = stage
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler
        self.accelerator = accelerator

    def setp(self, features, labels):
        # loss
        preds = self.net(features)
        loss = self.loss_fn(preds, labels)

        # backward
        if self.optimizer is not None and self.stage == "train":
            # backward
            if self.accelerator is None:
                loss.backward()
            else:
                self.accelerator.backward(loss)
            # optimizer
            self.optimizer.step()
            # learning rate scheduler
            if self.lr_scheduler is not None:
                self.lr_scheduler.step()
            # zero grad
            self.optimizer.zero_grad()
        
        # metrics
        step_metrics = {
            self.stage + "_" + name: metric_fn(preds, labels).item() 
            for name, metric_fn in self.metrics_dict.items()
        }

        return loss.item(), step_metrics

    def train_step(self, features, labels):
        """
        训练模式，dropout 层发生作用
        """
        self.net.train()
        return self.step(features, labels)

    @torch.no_grad()
    def eval_step(self, features, labels):
        """
        预测模式，dropout 层不发生作用
        """
        self.net.eval()
        return self.step(features, labels)

    def __call__(self, features, labels):
        if self.stage == "train":
            return self.train_step(features, labels)
        else:
            return self.eval_step(features, labels)


class EpochRunner:

    def __init__(self, steprunner):
        self.steprunner = steprunner
        self.stage = steprunner.stage
        self.steprunner.net.train() if self.stage == "train" else self.steprunner.net.eval()

    def __call__(self, dataloader):
        total_loss = 0
        step = 0
        loop = tqdm(enumerate(dataloader), total = len(dataloader))
        for i, batch in loop:
            if self.stage == "train":
                loss, step_metrics = self.steprunner(*batch)
            else:
                with torch.no_grad():
                    loss, step_metrics = self.steprunner(*batch)
            step_log = dict({
                self.stage + "_loss": loss
            }, **step_metrics)

            total_loss += loss
            step += 1
            if i != len(dataloader) - 1:
                loop.set_postfix(**step_log)
            else:
                epoch_loss = total_loss / step
                epoch_metrics = {
                    self.stage + "_" + name: metric_fn.compute().item()
                                       for name, metric_fn in self.steprunner.metrics_dict.items()
                }
                epoch_log = dict({
                    self.stage + "_loss": epoch_loss
                }, **epoch_metrics)
                loop.set_postfix(**epoch_log)

                for name, metric_fn in self.steprunner.metrics_dict.items():
                    metric_fn.reset()

        return epoch_log


def train_model(net, 
                optimizer, 
                loss_fn, 
                metrics_dict, 
                train_data, 
                val_data = None, 
                epochs = 10, 
                ckpt_path = 'checkpoint.pt',
                patience = 5, 
                monitor = "val_loss", 
                mode = "min"):
    history = {}
    for epoch in range(1, epochs+1):
        printlog("Epoch {0} / {1}".format(epoch, epochs))
        # ------------------------------
        # 1.train
        # ------------------------------
        train_step_runner = StepRunner(
            net = net,
            stage="train",
            loss_fn = loss_fn,
            metrics_dict = deepcopy(metrics_dict),
            optimizer = optimizer
        )
        train_epoch_runner = EpochRunner(train_step_runner)
        train_metrics = train_epoch_runner(train_data)
        for name, metric in train_metrics.items():
            history[name] = history.get(name, []) + [metric]
        # ------------------------------
        # 2.validate
        # ------------------------------
        if val_data:
            val_step_runner = StepRunner(
                net = net,
                stage = "val",
                loss_fn = loss_fn,
                metrics_dict = deepcopy(metrics_dict)
            )
            val_epoch_runner = EpochRunner(val_step_runner)
            with torch.no_grad():
                val_metrics = val_epoch_runner(val_data)
            val_metrics["epoch"] = epoch
            for name, metric in val_metrics.items():
                history[name] = history.get(name, []) + [metric]
        # ------------------------------
        # 3.early-stopping
        # ------------------------------
        arr_scores = history[monitor]
        best_score_idx = np.argmax(arr_scores) if mode=="max" else np.argmin(arr_scores)
        if best_score_idx == len(arr_scores) - 1:
            torch.save(net.state_dict(), ckpt_path)
            print(f"<<<<<< reach best {monitor} : {arr_scores[best_score_idx]} >>>>>>", file = sys.stderr)
        if len(arr_scores) - best_score_idx > patience:
            print(f"<<<<<< {monitor} without improvement in {patience} epoch, early stopping >>>>>>", file = sys.stderr)
            break 
        net.load_state_dict(torch.load(ckpt_path))

    return pd.DataFrame(history)
```

```python
df_history = train_model(
    net = net, 
    optimizer = optimizer, 
    loss_fn = loss_fn, 
    metrics_dict = metrics_dict, 
    train_data = dl_train,
    val_data = dl_val,
    epochs = 10,
    patience = 3,
    monitor = "val_acc",
    mode = "max"
)
```

# 类风格

`torchkeras` 安装:

```bash
$ pip install torchkeras
```

## KerasModel

使用 `torchkeras.KerasModel` 高阶 API 接口中的 `fit` 方法训练模型

```python
from torchkeras import KerasModel

# 模型实例化
model = KerasModel(
    net = net, 
    loss_fn = loss_fn,
    metrics_dict = metrics_dict,
    optimizer = optimizer,
)

# 启动训练循环
model.fit(
    trian_data = dl_train,
    val_data = dl_val,
    epochs = 10,
    patience = 3,
    monitor = "val_acc",
    mode = "max",
)
```

## LightModel

`torchkeras` 还提供了 `torchkeras.LightModel` 来支持跟多的功能。
`torchkeras.LightModel` 借鉴了 `pytorch_lightning` 库中的很多功能，
并演示了对 `pytorch_lightning` 的一种最佳实践

```python
from torchkeras import LightModel
import pytorch_lightning as pl

# 模型实例化
model = LightModel(
    net = net, 
    loss_fn = loss_fn,
    metrics_dict = metrics_dict,
    optimizer = optimizer,
)

# 设置回调函数
model_ckpt = pl.callbacks.ModelCheckpoint(
    monitor = "val_acc",
    save_top_k = 1,
    mode = "max",
)

# 设置早停
early_stopping = pl.callbacks.EarlyStopping(
    monitor = "val_acc",
    patience = 3,
    mode = "max",
)

# 设置训练参数
trainer = pl.Trainer(
    logger = True,
    min_epochs = 10,
    max_epochs = 20,
    gpus = 0,
    callbacks = [model_ckpt, early_stopping],
    enable_progress_bar = True,
)

# 启动训练循环
trianer.fit(
    model,
    dl_train,
    dl_val,
)
```
