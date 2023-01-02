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
optimizer = torch.optim.SGD(model.parameters(), lr = 1e-3)
```

## Model Training

```python
def train(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)
    model.train()
    for batch, (X, y) in enumerate(dataloader):
        # data
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

