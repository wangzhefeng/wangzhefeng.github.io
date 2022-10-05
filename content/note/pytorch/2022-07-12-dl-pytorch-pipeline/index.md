---
title: PyTorch Pipeline
author: 王哲峰
date: '2022-07-12'
slug: dl-pytorch-pipeline
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

- [Libraries](#libraries)
- [Data](#data)
  - [Data Download](#data-download)
  - [Data Loader](#data-loader)
- [Model](#model)
- [Model Parameters Optimizing](#model-parameters-optimizing)
  - [Loss and Optimizer](#loss-and-optimizer)
  - [Model Training](#model-training)
- [Model Saving](#model-saving)
- [Model Loading](#model-loading)
- [PyTorch Cheat Sheet](#pytorch-cheat-sheet)
  - [Imports](#imports)
  - [Tensors](#tensors)
  - [Deep Learning](#deep-learning)
  - [Data Utilities](#data-utilities)
</p></details><p></p>

# Libraries

```python
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader

from torchvision import datasets
from torchvision.transforms import ToTensor
```

# Data

## Data Download

```python
training_data = datasets.FashionMNIST(
    root = "data",
    train = True,
    download = True,
    transform = ToTensor(),
)
test_data = datasets.FashionMNIST(
    root = "data",
    train = False,
    download = True,
    transform = ToTensor(),
)
```

## Data Loader

```python
batch_size = 64

train_dataloader = DataLoader(
    training_data, 
    batch_size = batch_size
)
test_dataloader = DataLoader(
    test_data, 
    batch_size = batch_size
)

for X, y in test_dataloader:
    print(f"Shape of X [N, C, H, W]: {X.shape}")
    print(f"Shape of y: {y.shape}, {y.dtype}")
    break
```

# Model

```python
# get cpu or gpu device for training.
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using {device} device")


# Define model
class NeuralNetwork(nn.Module):
    def __init__(self):
        super(NeuralNetwork, self).__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(28 * 28, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 10),
        )
    
    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits

model = NeuralNetwork().to(device)
print(model)
```

# Model Parameters Optimizing

## Loss and Optimizer

```python
loss_fn = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(mode.parameters(), lr = 1e-3)
```

## Model Training

```python
def train(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)
    model.train()
    for batch, (X, y) in enumerate(dataloader):
        X, y = X.to(devcie), y.to(device)

        # Compute prediction error
        pred = model(X)  # 前向传播
        loss = loss_fn(pred, y)  # 损失函数

        # Backpropagation
        optimizer.zero_grad()  # 梯度
        loss.backward()  # 反向传播
        optimizer.step()  # 优化

        if batch % 100 == 0:
            loss, current = loss.item(), batch * len(X)
            print(f"loss: {loss:>7f} [{current:>5d}/{size:>5d}]")
```

```python
def test(dataloader, model, loss_fn):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    model.eval()    
    test_loss, correct = 0, 0
    with torch.no_grad():
        for X, y in dataloader:
            # data
            X, y = X.to(device), y.to(device)
            # Compute prediction error
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y) \
                            .type(torch.float) \
                            .sum() \
                            .item() \
    # 计算评价指标
    test_loss /= num_batches
    correct /= size
    print(f"Test Error: \n")
    print(f"Accuracy: {(100 * correct):>0.1f}")
    print(f"Avg loss: {test_loss:>8f}\n")
```

```py
epochs = 5
for t in range(epochs):
    print(f"Epoch {t+1}\n-----------------------------")
    train(train_dataloader, model, loss_fn, optimizer)
    test(test_dataloader, model, loss_fn)
print("Done!")
```

# Model Saving

```python
torch.save(model.state_dict(), "model.pth")
print("Saved PyTorch Model State to model.pth")
```

# Model Loading

```python
model = NeuralNetwork()
model.load_state_dict(torch.load("model.pth"))
model.eval()
```

```python
classes = [
    "T-shirt/top",
    "Trouser",
    "Pullover",
    "Dress",
    "Coat",
    "Sandal",
    "Shirt",
    "Sneaker",
    "Bag",
    "Ankle boot",
]
x, y = test_data[0][0], test_data[0][1]
with torch.no_grad():
    pred = model(x)
    predicted, actual = classes[pred[0].argmax(0)], classes[y]
    print(f"Predicted: {predicted}, Actual: {actual}")
```


# PyTorch Cheat Sheet

## Imports

```python
# General
import torch # root package
from torch.utils.data import Dataset, Dataloader # dataset representation and loading

# Neural Network API
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F
import torch.autograd as autograd
import torch.optim as optim
from torch.jit import script, trace

# Torchscript and JIT
torch.jit.trace()
@script

# ONNX
torch.onnx.export(model, dummy data, xxxx.proto)
model = onnx.load("alexnet.proto")
onnx.checker.check_model(model)
onnx.helper.printable_graph(model.graph)

# Vision
from torchvision import datasets, models, transforms
import torchvision.transforms as transforms

# Distributed Training
import torch.distributed as dist
from torch.multiprocessing import Process
```

## Tensors

```python   
import torch

# Creation
x = torch.randn(*size)
x = torch.[ones|zeros](*size)
x = torch.tensor(L)
y = x.clone()
with torch.no_grad():
requires_grad = True

# Dimensionality
x.size()
x = torch.cat(tensor_seq, dim = 0)
y = x.view(a, b, ...)
y - x.view(-1,a)
y = x.transpose(a, b)
y = x.permute(*dims)
y = x.unsqueeze(dim)
y = x.unsqueeze(dim = 2)
y = x.squeeze()
y = x.squeeze(dim = 1)

# Algebra
ret = A.mm(B)
ret = A.mv(x)
x = x.t()

# GUP Usage
torch.cuda.is_available
x = x.cuda()
x = x.cpu()
if not args.disable_cuda and torch.cuda.is_available():
   args.device = torch.device("cuda")
else:
   args.device = torch.device("cpu")
net.to(device)
x = x.to(device)
```

## Deep Learning

```python
import torch.nn as nn

nn.Linear(m, n)
nn.ConvXd(m, n, s)
nn.MaxPoolXd(s)
nn.BatchNormXd
nn.RNN
nn.LSTM
nn.GRU
nn.Dropout(p = 0.5, inplace = False)
nn.Dropout2d(p = 0.5, inplace = False)
nn.Embedding(num_embeddings, embedding_dim)

# Loss Function
nn.X

# Activation Function
nn.X

# Optimizers
opt = optim.x(model.parameters(), ...)
opt.step()
optim.X

# Learning rate scheduling
scheduler = optim.X(optimizer, ...)
scheduler.step()
optim.lr_scheduler.X
```

## Data Utilities

```python
# Dataset
Dataset
TensorDataset
Concat Dataset

# Dataloaders and DataSamplers
DataLoader(dataset, batch_size = 1, ...)
sampler.Sampler(dataset, ...)
sampler.XSampler where ...
```

