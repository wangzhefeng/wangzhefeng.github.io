---
title: PyTorch Distributed Data Parallel
author: wangzf
date: '2025-01-19'
slug: pytorch-distributed-data-parallel
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
- [单机单卡训练](#单机单卡训练)
- [模型分布式训练](#模型分布式训练)
    - [单机多卡训练](#单机多卡训练)
        - [原理](#原理)
        - [示例](#示例)
    - [多机多卡训练](#多机多卡训练)
- [torchrun](#torchrun)
    - [torchrun 使用](#torchrun-使用)
        - [单节点多进程](#单节点多进程)
        - [堆叠式单节点多工作进程](#堆叠式单节点多工作进程)
        - [容错](#容错)
        - [Elastic](#elastic)
    - [torchrun 重要提示](#torchrun-重要提示)
- [分布式训练框架](#分布式训练框架)
- [参考](#参考)
</p></details><p></p>

# 训练方法介绍

并行方法：

* 数据并行
* 模型并行
* 流水线并行

硬件分类方法：

* 单机单卡
* 单机多卡
* 多机多卡（集群）

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
    所以我们也可以在存的时候就用类似的方法改写字典，这样存下来的就直接是单线程的 model 了。

## 多机多卡训练


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

# 分布式训练框架

常用的分布式训练框架：

* PyTorch 分布式训练框架 DDP
* DeepSpeed 分布式训练框架

# 参考

* [torchrun](https://docs.pytorch.org/docs/stable/elastic/run.html)
* [大模型分布式并行训练范式原理篇](https://mp.weixin.qq.com/s/xtYqNQJBb5vJ11NWzbrAFg)
* [pytorch单机多卡训练](https://zhuanlan.zhihu.com/p/510718081)
