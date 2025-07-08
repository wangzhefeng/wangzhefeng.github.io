---
title: 模型分布式训练
author: wangzf
date: '2025-01-19'
slug: distributed-parallel-training
categories:
  - tool
tags:
  - machinelearning
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

- [训练方法介绍](#训练方法介绍)
- [分布式训练框架](#分布式训练框架)
- [单机单卡训练](#单机单卡训练)
- [数据并行](#数据并行)
    - [Data Prallel-DP](#data-prallel-dp)
    - [Distributed Data Parallel-DDP](#distributed-data-parallel-ddp)
    - [Fully Sharded Data Parallel-FSDP](#fully-sharded-data-parallel-fsdp)
        - [FSDP2 原理](#fsdp2-原理)
        - [FSDP2 使用](#fsdp2-使用)
            - [Model Initialization](#model-initialization)
            - [Forward/Backward with Prefetching](#forwardbackward-with-prefetching)
            - [Mixed Precision](#mixed-precision)
            - [Gradient Clipping and Optimizer with State Dicts with DTensor APIs](#gradient-clipping-and-optimizer-with-state-dicts-with-dtensor-apis)
            - [State Dict with DCP APIs](#state-dict-with-dcp-apis)
- [张量并行](#张量并行)
- [数据并行 \& 张量并行](#数据并行--张量并行)
- [流水线并行](#流水线并行)
- [torchrun](#torchrun)
    - [torchrun 使用](#torchrun-使用)
        - [单节点多进程](#单节点多进程)
        - [堆叠式单节点多工作进程](#堆叠式单节点多工作进程)
        - [容错](#容错)
        - [Elastic](#elastic)
    - [torchrun 重要提示](#torchrun-重要提示)
- [DeepSpeed 使用](#deepspeed-使用)
    - [DeepSpeed Model](#deepspeed-model)
        - [Training](#training)
        - [Model Checkpointing](#model-checkpointing)
    - [DeepSpeed 配置](#deepspeed-配置)
    - [启动 DeepSpeed 训练](#启动-deepspeed-训练)
    - [资源配置](#资源配置)
        - [资源配置-多节点](#资源配置-多节点)
            - [hostfile](#hostfile)
            - [num\_nodes 和 num\_gpus](#num_nodes-和-num_gpus)
            - [include 和 exclude](#include-和-exclude)
            - [不使用无密码 SSH 启动](#不使用无密码-ssh-启动)
            - [多节点环境变量](#多节点环境变量)
        - [资源配置-单节点](#资源配置-单节点)
    - [DeepSpeed 资料](#deepspeed-资料)
- [参考](#参考)
</p></details><p></p>

# 训练方法介绍

并行方法：

* 数据并行
* 张量并行
* 数据并行和张量并行
* 模型并行
* 流水线并行

硬件分类方法：

* 单机单卡
* 单机多卡
* 多机多卡（集群）

# 分布式训练框架

常用的分布式训练框架：

* PyTorch 分布式训练框架:
    - DP
    - DDP
    - FSDP
    - torchrun
* DeepSpeed 分布式训练框架


# 单机单卡训练

* 单线程

```bash
export CUDA_VISIBLE_DEVICES="0"

python -u YOUR_TRAINING_SCRIPT.py \
    --num_workders 4 \
    --use_gpu 1 \
    --gpu_type cuda \
    --use_multi_gpu 0 \
    --devices 0,1,2,3,4,5,6,7
```

# 数据并行

## Data Prallel-DP

建议使用 `torch.nn.parallel.DistributedDataParallel` 而不是 `torch.nn.DataParallel` 进行多 GPU 训练，即使只有一个节点。

`DistributedDataParallel` 和 `DataParallel` 的区别是：`DistributedDataParallel` 在多进程(multiprocessing)中为每个 GPU 创建一个进程，
而 `DataParallel` 使用多线程(multithreading)。通过使用多进程，每个 GPU 都有自己的专用进程，这避免了 Python 解释器 GIL 带来的性能开销。

## Distributed Data Parallel-DDP

## Fully Sharded Data Parallel-FSDP

### FSDP2 原理

在 DistributedDataParallel (DDP)训练中，每个 rank 拥有一个模型副本并处理一批数据，
最后使用 all-reduce 来跨 rank 同步梯度。

与 DDP 相比，FSDP 通过分片模型参数、梯度和优化器状态来减少 GPU 内存占用。
这使得训练无法在单个 GPU 上运行的模型成为可能。如下图所示，

![img](images/fsdp_workflow.png)

* 在正向和反向传播计算之外，参数(parameters)是完全分片的(fully sharded)
* 在正向和反向传播之前，分片参数(sharded parameters)会全部聚合(all-gathered)为未分片参数(unsharded parameters)
* 在反向传播过程中，局部的未分片梯度(unsharded gradients)会通过 reduce-scatter 聚合为分片梯度(sharded gradients)
* 优化器(optimizer)使用分片梯度(sharded parameters)更新分片参数(sharded gradients)，
  从而产生分片优化器状态(sharded optimizer stats)

FSDP 可以被视为将 DDP 的全归约(all-reduce)操作分解为：归约散播(reduce-scater)和全收集(all-gather)操作：

![img](images/fsdp_sharding.png)

### FSDP2 使用

#### Model Initialization

* **在子模块上应用 `fully_shard`**：与 DDP 不同，我们不仅应该在根模型上应用 fully_shard，
  还应该在子模块上应用。

```bash
$ torchrun --nproc_per_node 2 train.py
```

```python
from torch.distributed.fsdp import fullly_shard, FSDPModule

# model
model = Transformer()

# 首先对每一层应用 fully_shard，然后对根模型应用
for layer in model.layers:
    fullly_shard(layer)
fully_shard(layer)

assert isinstance(model, Transformer)
assert isinstance(model, FSDPModule)
print(model)
```

* `model.parameters() as DTensor`：`fully_shard` 在不同 rank 之间分片参数，
  并将 `model.parameters()` 从普通的 `torch.Tensor` 转换为 `DTensor` 来表示分片参数。
  FSDP2 默认在 `dim-0` 上分片，因此 DTensor 的放置是 `Shard(dim=0)`。
  假设我们有 N 个 rank，并且分片前参数有 N 行。分片后，每个 rank 将拥有参数的 1 行。
  我们可以使用 `param.to_local()` 检查分片参数。

```python
from torch.distributed.tensor import DTensor

for param in model.parameters():
    assert isinstance(param, DTensor)
    assert param.placements == (Shard(0),)
    # inspect sharded parameters with param.to_local()

# optimizer
optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)
```

#### Forward/Backward with Prefetching

#### Mixed Precision

#### Gradient Clipping and Optimizer with State Dicts with DTensor APIs

#### State Dict with DCP APIs

# 张量并行


# 数据并行 & 张量并行


# 流水线并行


<!-- 
# 模型分布式训练

## 单机多卡训练

### 原理

**单机多卡模型训练** 就是用多张显卡训练一个模型，对于训练速度的提升有极为明显的效果。
相比较多机多卡，单机多卡实现极其简易，只需要使用 `torch` 自带的通讯通道和端口，
即可快速将单线程的代码转换成多线程。

`torch 1.11.0` 提供了启动命令 `torchrun`，对于 `local rank` 的使用有了优化。
理解了分布式训练的基础原理，遵循着 PyTorch 官方文档，就能写出非常优秀的分布式训练代码。

首先我们要思考一个问题，**分布式训练** 相比较 **单线程训练** 需要做出什么改变呢？

1. 第一个是 **启动的命令行**。以前我们使用 `python train.py` 启动一个线程进行训练，现在我们需要一个新的启动方式，
   从而让机器知道现在我们要启动八个线程了。这八个线程之间的通讯方式完全由 `torch` 帮我们解决。
   而且我们还可以知道，`torch` 会帮助我们产生一个 `local rank` 的标志，如果是八个线程，
   每个线程的 `local rank` 编号会是 `0~7`。这样我们就可以通过 `local rank` 确定当前的线程具体是哪一个线程，
   从而将其绑定至对应的显卡。
2. 第二个是 **模型** 的改变。八个线程如果各自训练各自的模型，那也是白搭。
   所以我们要将模型改成支持分布式的模型。`torch` 为我们提供的解决思路是，
   分布式的 `model` 在多个线程中始终保持参数一致。任意一个线程梯度传播造成的参数改变会影响所有模型中的参数。
3. 第三个是 **数据分配** 的改变。假设我们仍然使用传统的 `dataloader` 去处理数据，会发生什么？
   八个线程，拿到了一样的训练数据集，然后进行训练。虽然这样训练一个周期相当于训练了八个周期，但是总感觉怪怪的。
   我们拿到一个任务，肯定希望它被拆分成多份给不同的工人去做，而不是每个工人都做一遍全部任务。
   所以我们希望 `dataloader` 可以把训练数据集拆成八份，每个线程训练自己获得的那一份数据集。
4. 第四个是 **evaluation**。训练完了模型我们肯定要进行性能测试。由于八个线程上的模型参数是一样的，
   所以我们在任意一个线程上 evaluation 就可以了。这个时候有人可能要说，我可不可以把 evaluation 也变成多线程的呢？
   我觉得可以，但是会有一个问题。我们通常希望获得模型在 test set 上的表现比如 `mae loss`。
   我们去看第三点，分布式的操作会把数据集分成多块，如果 evaluation 变成分布式会让 8 个线程得到 8 个 `mae loss`，
   而这 8 个 `mae loss` 是对于 8 个被拆分的测试数据集的。
   怎么把这 8 个 `mae loss` 结合成对于整体 test set 的 `loss` 是我们需要考虑的问题。
   `mae loss` 还好，直接取平均看起来还比较合理，但是我面对的任务是要去评估 `precision_recall 曲线` 的。
   我没有想好怎么分布式地评估 `P_R 曲线`，同时注意到 test set 相比较 train set 规模小了太多，
   我直接单线程的进行 evaluation 是一个比较方便的做法。当然我相信一定有方法可以分布式地进行 `evaluation`，
   但是我面临的问题不需要这么做。

### 示例

1. 第一步，要让代码支持分布式，也就是引入一些支持分布式通讯的代码。
   具体操作就是需要在代码最前端添加：

```python
import torch.distributed as dist

local_rank = int(os.environ["LOCAL_RANK"])
dist.init_process_group(backend = "gloo|nccl")
```

`local_rank` 是每个进程的标签，对于八个进程，`local_rank` 这个变量最后会被分配 `0~7` 的整数。

在 torch 1.10 之前是用 `argparse` 拿 `local rank` 这个参数的。
但是新版的 api 已经改成了用 `os.environ["LOCAL_RANK"]`的写法。

2. 第二步，让模型支持分布式

```python
import os
import torch

local_rank = os.environ["LOCAL_RANK"]

model = torch.nn.parallel.DistributedDataParallel(
    model, 
    device_ids=[local_rank],
    output_device=local_rank,
)
```

第二句代码很有意思，`device_ids=[local_rank]`。假设我们有八个 GPU，
我们同时知道对于不同的进程 `local_rank` 被赋的值是不一样的。
所以这一步在不同的线程中，会把 `model` 放在不同的 GPU 上，
这样就非常简洁地完成了 GPU 和进程之间的对应操作。

3. 要让 dataset 被分布式地 sample 到不同的进程上

```python
import torch

def load_data(train_file, test_file, batch_size=32):
    train_dataset = SentenceDataset(train_file)
    test_dataset = SentenceDataset(test_file)

    train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)

    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, sampler=train_sampler)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_dataloader, test_dataloader
```

如同我前面提到的，想分布式地处理 train 的过程，但是不想分布式地处理 evaluation，
所以这边对于 train set 我使用了 distributed sampler，对于 test set 用了传统的 load 方式。

4. 另外在 evaluation 的地方，加了对于 local rank 的判断，保证只有主线程在指定的时间会做 evaluation。

```python
import torch

if local_rank == 0:
    MSE = nn.MSELoss()
    with torch.no_grad():
         model.eval()
         for test_batch in test_dataloader:
            ...
```

这里 `with torch.no_grad()` 必须要加。我们都知道这句话的作用是在代码块中不进行梯度的追踪，
一般用于在 evaluation 中加速运算和节省资源。但是在本实验中如果不加这行代码会报错。
推测的原因是：在多个进程中，`model` 的参数是一直保持一致的，
也就是 `torch distributed` 使用了某种机制控制了多个进程在运算中产生的梯度，再进行反向传播的调度。
但是现在限制了只在线程 `0` 进行了一次额外的 evaluation 运算，虽然 `eval` 过程中不会进行梯度反向传播，
但是如果不加 `no_grad` 依然会有梯度的计算，可能会影响到 `torch` 的调度过程。
总之加上这句既避免了 error 又加速了 eval。

5. 到这里，写代码的部分基本就结束了，进入 Debug 环节。先分布式运行 Python 文件：

```bash
$ OMP_NUM_THREADS=12 torchrun \
    --standalone \
    --nnodes=1 \
    --nproc_per_node=8 \
    train.py \
    ...
```

在 `torchrun` 和 `.py` 文件之间又三个关于分布式的参数，如果是单机多卡前两个都是不变的，
最后一个 `nproc_per_node` 填显卡个数即可。如果是多机多卡这边会有不同的写法，建议查询官方文档。

> torchrun前面还有一个 `OMP_NUM_THREAD` 的参数，如果不写，程序也可以执行，
> 但是会报一个警告：
> ```
> warning: Setting OMP_NUM_THREADS environment variable for each process to be 1 in default, 
> to avoid your system being overloaded, please further tune the variable for optimal performance 
> in your application as needed
> ```
> 我的理解是由于使用了多进程运行 `.py` 文件，intel 怕使用太多线程把 CPU 过载了，
> 默认将每个进程可以使用的线程数调成了 `1`，最大程度地避免过载。
> 可以直接不写 OMP 参数并且忽略这个 warning，因为我认为机器学习的主要时间耗费在 GPU 运算上而不是 CPU 的调度。
> 但是如果设备说明书上写了可以支持超过一百个线程，那可以设置成 12，同时使用 8 个进程也只会耗费 96 个线程不会过载。

6. 在加载模型时，也有一个很重要的问题。正常来说，当我们在 train 完 model 之后会用 `torch.save` 储存模型。
   日后要用的时候再用 `model.load` 或者 `model.load_state_dict` 加载模型。
   由于 `eval` 一直是单线程的设计，所以在 `eval.py` 中并没有使用任何有关多线程的代码。
   这个时候在使用 `load` 方法的时候，会报错。在 `train.py` 中，虽然也有 `eval` 的操作，
   但是是直接用 `train.py` 中多线程的 `model` 做 `eval` 的，自然不会报错。
   但是在 `eval.py` 使用单线程的模型去读多线程的参数，自然是不可以的。
   那怎么解决这个问题呢？其实思路很简单，多线程不是加了个前缀嘛，
   我们读参数的时候把那个多出来的前缀删掉不就好了。

    ```python
    model.load_state_dict(
        {
            k.replace('module.', ''): v 
            for k, v in torch.load(args.model_path, map_location=device)['model'].items()
        },
        strict=True
    )
    ```

    所以在 `eval.py` 文件中，`load` 这么改写了一下。`torch.load` 本质上就是 load 了一个词典进来，
    我把词典的每一个 item 都读一下，如果 key 里面有 module.，我们就把他换成空字符串。
    这种解决方案意思是我存下来的 model 文件是多线程的，但是我读的时候按单线程的读。
    所以我们也可以在存的时候就用类似的方法改写字典，这样存下来的就直接是单线程的 model 了。 -->

# torchrun

> Elastic Launch

`torch.distributed.run` 是一个在每台训练节点上启动多个分布式训练进程的模块。

`torchrun` 是一个 Python 控制台脚本，
对应于在 `setup.py` 中 `entry_points` 配置中声明的 `torch.distributed.run` 主模块。
它等同于调用 `python -m torch.distributed.run`。

> torchrun 将 `--local-rank=<rank>` 参数传递给您的脚本。从 PyTorch 2.0.0 开始，
> 推荐使用连字符 `--local-rank` 而不是之前使用的下划线 `--local_rank`。

## torchrun 使用

### 单节点多进程

> Single-node multi-worker

```shell
torchrun
    --standalone
    --nnodes=1
    --nproc-per-node=$NUM_TRAINERS
    YOUR_TRAINING_SCRIPT.py (--arg1 ... train script args...)
```

* `--nproc-per-node`
    - `gpu`：每个 GPU 启动一个进程
    - `cup`：每个 CPU 启动一个进程
    - `auto`：如果 CUDA 可用则等效于 `"gpu"`，否则等效于 `"cpu"`，或指定进程数的整数

### 堆叠式单节点多工作进程

> Stacked single-node multi-worker

要在同一主机上运行多个单节点多工作进程实例（分离的任务），我们需要确保每个实例（任务）在不同的端口上设置，
以避免端口冲突（或者更糟，两个任务被合并为一个任务）。
为此，你必须使用 `--rdzv-backend=c10d` 并通过设置 `--rdzv-endpoint=localhost:$PORT_k` 指定不同的端口。
对于 `--nodes=1`，通常让 torchrun 自动选择一个空闲的随机端口更方便，而不是手动为每次运行分配不同的端口。

```shell
torchrun
    --rdzv-backend=c10d
    --rdzv-endpoint=localhost:0
    --nnodes=1
    --nproc-per-node=$NUM_TRAINERS
    YOUR_TRAINING_SCRIPT.py (--arg1 ... train script args...)
```

### 容错

> Fault tolerant (fixed sized number of workers, no elasticity, tolerates 3 failures)
> 
> 容错（固定数量的工作进程，无弹性，可容忍 3 次失败）

```bash
torchrun
    --nnodes=$NUM_NODES
    --nproc-per-node=$NUM_TRAINERS
    --max-restarts=3
    --rdzv-id=$JOB_ID
    --rdzv-backend=c10d
    --rdzv-endpoint=$HOST_NODE_ADDR
    YOUR_TRAINING_SCRIPT.py (--arg1 ... train script args...)
```

### Elastic

> Elastic (min=1, max=4, tolerates up to 3 membership changes or failures)
> 
> 弹性 ( min=1 , max=4 , 可容忍最多 3 次成员变更或故障)

```bash
torchrun
    --nnodes=1:4
    --nproc-per-node=$NUM_TRAINERS
    --max-restarts=3
    --rdzv-id=$JOB_ID
    --rdzv-backend=c10d
    --rdzv-endpoint=$HOST_NODE_ADDR
    YOUR_TRAINING_SCRIPT.py (--arg1 ... train script args...)
```

## torchrun 重要提示

1. `torchrun` 和多进程分布式（单节点或多节点）GPU 训练目前仅在使用 NCCL 分布式后端时能达到最佳性能。
   因此，NCCL 后端是 GPU 训练推荐使用的后端。
2. 初始化 Torch 进程组所需的环境变量由该模块提供，无需手动传递 `RANK`。
   要在训练脚本中初始化进程组，只需运行：

   ```python
   import torch.distributed as dist
    dist.init_process_group(backend="gloo|nccl")
   ```

3. 在训练程序中，可以使用常规的分布式函数，或者使用 `torch.nn.parallel.DistributedDataParallel()` 模块。
   如果训练程序使用 GPU 进行训练，并且想使用 `torch.nn.parallel.DistributedDataParallel()` 模块，
   以下是配置方法：

    ```python
    local_rank = int(os.environ["LOCAL_RANK"])
    model = torch.nn.parallel.DistributedDataParallel(
        model, device_ids=[local_rank], output_device=local_rank
    )
    ```

    - 请确保 `device_ids` 参数设置为将操作的唯一 GPU 设备 ID。这通常是进程的本地排名。换句话说，
      `device_ids` 需要是 [`int(os.environ("LOCAL_RANK"))`] ，
      并且 `output_device` 需要是 `int(os.environ("LOCAL_RANK"))`，才能使用这个工具。

4. 在失败或成员资格变更时，所有存活的进程将立即被终止。确保保存您的进度。
   检查点的频率应根据您的工作对丢失工作的容忍度来决定。
5. 该模块仅支持同构的 `LOCAL_WORLD_SIZE`。也就是说，假设所有节点运行相同数量的本地进程（按角色划分）。
6. `RANK` 是不稳定的。在重启之间，节点上的本地进程可能被分配与之前不同的排名范围。
   永远不要硬编码任何关于排名稳定性的假设，或 `RANK` 和 `LOCAL_RANK` 之间某些关联的假设。
7. 在使用弹性（`min_size!=max_size`）时，不要硬编码关于 `WORLD_SIZE` 的假设，因为当节点允许离开和加入时，
   世界大小可能会改变。
8. 建议您的脚本具有以下结构：

    ```python
    def main():
        load_checkpoint(checkpoint_path)
        initialize()
        train()

    def train():
        for batch in iter(dataset):
            train_step(batch)

            if should_checkpoint:
                save_checkpoint(checkpoint_path)
    ```

9. (推荐) 当工作进程出错时，该工具将总结错误详情（例如时间、排名、主机、进程 ID、堆栈跟踪等）。
    在每个节点上，按时间戳排序的第一个错误会被启发式地报告为“根本原因”错误。
    要获取作为此错误总结输出的一部分的堆栈跟踪，您必须在训练脚本中的主入口函数上添加装饰器，
    如下面的示例所示。如果没有添加装饰器，则总结将不包含异常的堆栈跟踪，而只包含退出码。

    ```python
    from torch.distributed.elastic.multiprocessing.errors import record

    @record
    def main():
        # do train
        pass


    if __name__ == "__main__":
        main()
    ```

# DeepSpeed 使用

## DeepSpeed Model

DeepSpeed 模型训练是通过 DeepSpeed 引擎完成的。
该引擎可以包装任何类型为 torch.nn.module 的任意模型，
并具有一套最小的 API 用于模型训练和检查点保存。

初始化 DeepSpeed 引擎：

```python
import deepspeed

model_engine, optimizer, _, _ = deepspeed_initialize(
    args = cmd_args,
    model = model,
    model_parameters = params
)
```

> 如果你已经设置了分布式环境，
> 你需要将：`torch.distributed.init_process_group()` 替换为：`deepspeed.init_distributed()`。
>
> 但如果不需要在 `deepspeed.initialize()` 之后设置分布式环境，则不必使用此函数，
> 因为 DeepSpeed 将在其 `initialize` 自动初始化分布式环境。无论如何，
> 如果已经设置好了 `torch.distributed.init_process_group`，需要将其移除。

### Training

一旦 DeepSpeed 引擎被初始化，它就可以使用三个简单的 API 来训练模型：前向传播（可调用对象）、
反向传播（ backward ）和权重更新（ step ）。

```python
for step, batch in enumerate(data_loader):
    # forward
    loss = model_engine(batch)
    # backward
    model_engine.backward(loss)
    # weight update
    model_engine.step()
```

在底层，DeepSpeed 自动执行分布式数据并行训练所需的操作，以混合精度和预定义的学习率调度器进行：

* Gradient Averaging，梯度平均
    - 在分布式数据并行训练中，`backward` 确保在训练一个 `train_batch_size` 后，
      梯度在数据并行过程中被平均。
* Loss Scaling，损失缩放
    - 在 FP16/混合精度训练中，DeepSpeed 引擎自动处理损失缩放，以避免梯度中的精度损失。
* Learning Rate Scheduler，学习率调度器
    - 当使用 DeepSpeed 的学习率调度器（在 `ds_config.json` 文件中指定）时，
      DeepSpeed 在每个训练步骤（当 `model_engine.step()` 执行时）调用调度器的 step() 方法。
    - 当不使用 DeepSpeed 的学习率调度器时：
        - 如果调度器需要在每个训练步骤执行，
          那么用户可以在初始化 DeepSpeed 引擎时将调度器传递给 `deepspeed.initialize`，
          并让 DeepSpeed 管理其更新或保存/恢复。
        - 如果计划表需要在其他任何间隔执行（例如训练轮次），则用户不应在初始化时将计划表传递给 DeepSpeed，
          而必须显式管理它。

### Model Checkpointing

保存和加载训练状态是通过 DeepSpeed 中的 `save_checkpoint` 和 `load_checkpoint` API 处理的，
该 API 需要两个参数来唯一标识一个检查点：

* `ckpt_dir`：检查点将保存的目录
* `ckpt_di`：一个唯一标识目录中检查点的标识符。在以下代码片段中，我们使用损失值作为检查点标识符。

DeepSpeed 可以自动保存和恢复模型、优化器和学习率调度器的状态，同时将这些细节隐藏起来。
然而，用户可能希望保存特定于给定模型训练的额外数据。为了支持这些项目，
`save_checkpoint` 接受客户端状态字典 `client_sd` 进行保存。
这些项目可以作为返回参数从 `load_checkpoint` 中检索。
在下述示例中，`step` 值被存储为 `client_sd` 的一部分。

```python
# load checkpoint
_, client_sd = model_engine.load_checkpoint(args.load_dir, args.ckpt_id)
step = client_sd['step']

# advance data loader to ckpt step
dataloader_to_step(data_loader, step + 1)

for step, batch in enumerate(data_loader):
    # forward() method
    loss = model_engine(batch)

    # runs backpropagation
    model_engine.backward(loss)

    # weight update
    model_engine.step()

    # save checkpoint
    if step % args.save_interval:
        client_sd['step'] = step
        ckpt_id = loss.item()
        model_engine.save_checkpoint(args.save_dir, ckpt_id, client_sd = client_sd)
```

## DeepSpeed 配置

DeepSpeed 的功能可以通过一个 config JSON 文件来启用、禁用或配置，
该文件应指定为 `args.deepspeed_config`。下面是一个示例配置文件。

```json
// ds_config.json
{
    "train_batch_size": 8,
    "gradient_accumulation_steps": 1,
    "optimizer": {
        "type": "Adam",
        "params": {
            "lr": 0.00015
        }
    },
    "fp16": {
        "enabled": true
    },
    "zero_optimization": true
}
```

## 启动 DeepSpeed 训练

DeepSpeed 安装入口 `deepspeed` 来启动分布式训练。我们通过以下假设来展示 DeepSpeed 的一个使用示例：

1. 已经将 DeepSpeed 集成到你的模型中
2. `client_entry.py` 是你的模型的入口脚本
3. `client args` 是 `argparse` 命令行参数
4. `ds_config.json` 是 DeepSpeed 的配置文件

## 资源配置

### 资源配置-多节点

#### hostfile

DeepSpeed 使用与 OpenMPI 和 Horovod 兼容的 `hostfile` 配置多节点计算资源。
`hostfile` 是一个主机名（或 SSH 别名）列表，这些是可以通过无密码 SSH 访问的机器，
以及槽位数量(slot counts)，这些指定了系统上可用的 GPU 数量。

例如：下面的 `hostfile` 指定了名为 `worker-1` 和 `worker-2` 的两台机器，每台机器都有四个 GPU 用于训练。

```
worker-1 slots=4
worker-2 slots=4
```

Hostfiles 通过 `--hostfile` 命令行选项指定。如果没有指定 `hostfile`，
DeepSpeed 会搜索 `/job/hostfile`。如果没有指定或找到 `hostfile`，
DeepSpeed 会查询本地机器上的 GPU 数量，以发现可用的本地插槽数量。

以下命令将在 `myhostfile` 中指定的所有可用节点和 GPU 上启动一个 PyTorch 训练作业：

```bash
$ deepspeed --hostfile=myhostfile \
    <client_entry.py> <client args> \
    --deepspeed --deepspeed_config ds_config.json
```

#### num_nodes 和 num_gpus

或者，DeepSpeed 允许你将模型的分布式训练限制在可用的节点和 GPU 的子集上。
这一功能通过两个命令行参数 `--num_nodes` 和 `--num_gpus` 启用。
例如，可以使用以下命令将分布式训练限制在仅使用两个节点：

```bash
$ deepspeed --num_nodes=2 --num_gpus 8 \
    <client_entry.py> <client args> \
    --deepspeed --deepspeed_config ds_config.json
```

#### include 和 exclude

可以使用 `--include` 和 `--exclude` 标志来包含或排除特定资源。
例如，要在节点 `worker-2` 上使用除 GPU `0` 以外的所有可用资源，
并在 `worker-3` 上使用 GPU `0` 和 GPU `1`：

```bash
$ deepspeed --exclude="worker-2:0@worker-3:0,1" \
    <client_entry.py> <client args> \
    --deepspeed --deepspeed_config ds_config.json
```

同样地，你也可以在 `worker-2` 上仅使用 GPU `0` 和 `1`：

```bash
$ deepspeed --include="worker-2:0,1" \
    <client_entry.py> <client args> \
    --deepspeed --deepspeed_config ds_config.json 
```

#### 不使用无密码 SSH 启动

DeepSpeed 现在支持在不使用无密码 SSH 的情况下启动训练作业。
这种模式在 Kubernetes 等云环境中特别有用，这些环境允许灵活的容器编排，
而使用无密码 SSH 设置 leader-worker 架构会增加不必要的复杂性。

要使用此模式，您需要在所有节点上分别运行 DeepSpeed 命令。命令应按以下结构运行：

```bash
deepspeed --hostfile=myhostfile \
    --no_ssh \
    --node_rank=<n> \
    --master_addr=<addr> --master_port=<port> \
    <client_entry.py> <client args> \
    --deepspeed --deepspeed_config ds_config.json
```

每个节点必须使用唯一的 `node_rank` 启动，并且所有节点都需要提供领导节点的地址和端口（`rank 0`）。
这种模式使启动器表现得类似于 PyTorch 文档中描述的 `torchrun` 启动器。

#### 多节点环境变量

在跨多个节点进行训练时，我们发现支持传播用户定义的环境变量很有用。
默认情况下，DeepSpeed 会传播所有已设置的 NCCL 和 PYTHON 相关的环境变量。
如果您想传播额外的变量，可以在名为 `.deepspeed_env` 的点文件中指定它们，
该文件包含用换行符分隔的 `VAR=VAL` 条目列表。
DeepSpeed 启动器将检查您正在执行的本地路径以及您的家目录（`~/`）。
如果您想覆盖此文件的默认名称或路径并用自己的名称指定，可以使用环境变量 `DS_ENV_FILE`。
这主要适用于您启动多个作业，而所有作业都需要不同的变量的情况。

作为一个具体的例子，某些集群需要在训练之前设置特殊的 `NCCL` 变量。
用户只需将这些变量添加到其主目录中的一个 `.deepspeed_env` 文件中，该文件看起来像这样：

```bash
NCCL_IB_DISABLE=1
NCCL_SOCKET_IFNAME=eth0
```

DeepSpeed 将确保在训练作业中，在每个节点上启动每个进程时都设置这些环境变量。

### 资源配置-单节点

如果我们只在一个节点上运行（该节点有一个或多个 GPU），DeepSpeed 就不需要像上面描述的那样使用主机文件。
如果未检测到或未传入主机文件，DeepSpeed 将查询本地机器上的 GPU 数量，以发现可用的插槽数量。 
`--include` 和 `--exclude` 参数按正常方式工作，但用户应将 `localhost` 指定为主机名。

此外，`CUDA_VISIBLE_DEVICES` 可以与 `deepspeed` 一起使用，以控制在一个节点上应使用哪些设备。
因此，以下任一方式都可以用于仅在当前节点的设备 `0` 和 `1` 上启动：

```bash
$ deepspeed --include localhost:0,1 ...
```

```bash
$ CUDA_VISIBLE_DEVICES=0,1 deepspeed ...
```

## DeepSpeed 资料

* [deepspeed.ai](https://www.deepspeed.ai/)
* [deepspeedai/DeepSpeed](https://github.com/deepspeedai/DeepSpeed)
* [deepspeed 知乎](https://www.zhihu.com/people/deepspeed/posts)

# 参考

* [torchrun](https://docs.pytorch.org/docs/stable/elastic/run.html)
* [大模型分布式并行训练范式原理篇](https://mp.weixin.qq.com/s/xtYqNQJBb5vJ11NWzbrAFg)
* [pytorch单机多卡训练](https://zhuanlan.zhihu.com/p/510718081)
* [PyTorch examples distributed](https://github.com/pytorch/examples/tree/main/distributed)
* [DataParallel](https://docs.pytorch.org/docs/stable/generated/torch.nn.DataParallel.html)
* [使用 `nn.parallel.DistributedDataParallel` 而不是 `multiprocessing` 或 `nn.DataParallel`](https://docs.pytorch.org/docs/stable/notes/cuda.html#cuda-nn-ddp-instead)
* [Distributed Data Parallel](https://docs.pytorch.org/docs/stable/notes/ddp.html#ddp)
* [pytorch/examples/distributed/FSDP2/](https://github.com/pytorch/examples/tree/main/distributed/FSDP2)
