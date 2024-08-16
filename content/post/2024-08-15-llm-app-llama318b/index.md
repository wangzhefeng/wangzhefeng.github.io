---
title: LLM-Llama 3.1 8B
author: 王哲峰
date: '2024-08-15'
slug: llm-app-llama318b
categories:
  - llm
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

- [环境](#环境)
    - [本地环境](#本地环境)
    - [云服务器](#云服务器)
- [模型下载](#模型下载)
- [构建 LLM 应用](#构建-llm-应用)
    - [模型构建](#模型构建)
    - [调用模型](#调用模型)
- [LoRA 微调](#lora-微调)
    - [微调数据集准备](#微调数据集准备)
    - [指令集收集](#指令集收集)
    - [数据格式化](#数据格式化)
    - [加载 tokenizer 和半精度模型](#加载-tokenizer-和半精度模型)
    - [定义 LoraConfig](#定义-loraconfig)
    - [自定义 TrainingArguments 参数](#自定义-trainingarguments-参数)
    - [使用 Trainer 训练](#使用-trainer-训练)
    - [加载 LoRA 权重权重推理](#加载-lora-权重权重推理)
- [FastAPI 部署和调用](#fastapi-部署和调用)
- [Instruct WebDemo 部署](#instruct-webdemo-部署)
- [资料](#资料)
</p></details><p></p>

# 环境

## 本地环境

* Ubuntu 22.04
* Python 3.12
* CUDA 12.1
* PyTorch 2.3.0
* Python Libs

    ```bash
    # 升级 pip
    $ pip install --upgrade pip
    # 更换 pypi 源加速库的安装
    $ pip config set global.index.url https://pypi.tuna.tsinghua.edu.cn/simple

    # 安转依赖库
    $ pip install modelscope==1.11.0
    $ pip install langchain==0.2.3
    $ pip install transformers==4.43.1
    $ pip install accelerate==0.33.0
    $ pip install peft==0.11.1
    $ pip install datasets==2.20.0
    ```

## 云服务器

* [AutoDL 平台 LlaMA3.1 镜像](https://www.codewithgpu.com/i/datawhalechina/self-llm/self-llm-llama3.1)

# 模型下载

Llama-3.1-8B-Instruct 模型大小为 16 GB，下载模型大概需要 12 分钟。

新建 `model_download.py` 脚本如下：

```python
# model_download.py
import os
import torch
from modelscope import snapshot_download, AutoModel, AutoTokenizer

# 模型下载
model_dir = snapshot_download(
    "LLM-Research/Meta-Llama-3.1-8B-Instruct", 
    cache_dir = "/root/autodl-tmp",  # win10: downloaded_models
    revision = "master",
)
```

# 构建 LLM 应用

为便捷构建 LLM 应用，需要基于本地部署的 LLaMA3_1_LLM，自定义一个 `LLM` 类，
将 LLaMA3.1 接入到 LangChain 框架中。完成自定义 LLM 类之后，
可以以完全一致的方式调用 LangChain 的接口，而无需考虑底层模型调用的不一致。

## 模型构建

基于本地部署的 LLaMA3.1 自定义 LLM 类并不复杂，只需从 `langchain.llms.base.LLM` 类继承一个子类，
并重写构造函数与 `_call` 函数即可。

新建 `LLM.py` 脚本如下：

```python
from typing import Any, List, Optional
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from langchain.llms.base import LLM
from langchain.callbacks.manager import CallbackManagerForLLMRun


class LLaMA3_1_LLM(LLM):
    """
    基于本地 llama3.1 自定义 LLM 类
    """
    tokenizer: AutoTokenizer = None
    model: AutoModelForCausalLM = None

    def __init__(self, model_name_or_path: str):
        super().__init__()
        print("正在从本地加载模型...")
        # tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name_or_path, 
            use_fast = False
        )
        # model
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name_or_path, 
            torch_dtype = torch.bfloat16, 
            device_map = "auto"
        )
        self.tokenizer.pad_token = self.tokenizer.eos_token
        print("完成本地模型的加载")

    def _call(self, 
              prompt : str, 
              stop: Optional[List[str]] = None,
              run_manager: Optional[CallbackManagerForLLMRun] = None,
              **kwargs: Any):
        messages = [
            {
                "role": "system",
                "content": "You are a helpful assistant.",
            },
            {
                "role": "user",
                "content": prompt,
            }
        ]
        input_ids = self.tokenizer.apply_chat_template(
            messages, 
            tokenize = False, 
            add_generation_prompt = True
        )
        model_inputs = self.tokenizer(
            [input_ids],
            return_tensors = "pt",
        ).to(self.model.device)
        generated_ids = self.model.generate(
            model_inputs.input_ids, 
            max_new_tokens = 512
        )
        generated_ids = [
            output_ids[len(input_ids):] 
            for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
        ]
        response = self.tokenizer.batch_decode(
            generated_ids, 
            skip_special_tokens = True
        )[0]
        
        return response

    @property
    def _llm_type(self) -> str:
        return "LLaMA3_1_LLM" 
```

## 调用模型

```python
# 测试代码 main 函数
def main():
    from LLM import LLaMA3_1_LLM

    llm = LLaMA3_1_LLM(model_name_or_path = "D:\projects\llms_proj\llm_proj\downloaded_models\LLM-Research\Meta-Llama-3.1-8B-Instruct")
    print(llm("你好"))

if __name__ == "__main__":
    main()
```

# LoRA 微调

## 微调数据集准备

```

```

## 指令集收集

LLM 微调一般指 **指令微调** 的过程。所谓指令微调，是说使用的微调数据形如：

```
{
    "instruction": "回答以下用户问题，仅输出答案。",
    "input": "1+1等于几？",
    "output": "2"
}
```

其中：

* `instruction`: 是用户指令，告知模型其需要完成的任务
* `input`: 是用户输入，是完成用户指令所必须的输入内容
* `output`: 是模型应该给出的输出

微调的核心训练目标是让模型具有理解并遵循用户指令的能力。因此，在指令集构建时，
应针对我们的目标任务，针对性构建任务指令集。

## 数据格式化

LoRA 训练的数据是需要经过格式化、编码之后再输入给模型进行训练的。就像 PyTorch 模型训练过程中，
一般需要将输入文本编码为 `input_ids`，将输出文本编码为 `labels`，编码之后的结果都是多维的向量。
下面定义一个预处理函数，这个函数用于对每一个样本，编码其输入、输出文本并返回一个编码后的字典。

```python
def process_func(example):
    """
    数据格式化
    """
    # LlaMA 分词器会将一个中文字切分为多个 token，
    # 因此需要放开一些最大长度，保证数据的完整性
    MAX_LENGTH = 384
    input_ids, attention_mask, labels = [], [], []

    instruction = tokenizer(
        f"<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n现在你要扮演皇帝身边的女人--甄嬛<|eot_id|><|start_header_id|>user<|end_header_id|>\n\n{example['instruction'] + example['input']}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n", 
        add_special_tokens = False
    )  # add_special_tokens 不在开头加 special_tokens
    response = tokenizer(
        f"{example["output"]}<|eot_id|>", 
        add_special_tokens = False
    )

    input_ids = instruction["input_ids"] + response["input_ids"] + [tokenizer.pad_token_id]
    attention_mask = instruction["attention_mask"] + response["attention_mask"] + [1]  # 因为 eos token 咱们也是要关注的，所以补充为 1
    labels = [-100] * len(instruction["input_ids"]) + response["input_ids"] + [tokenizer.pad_token_id]

    if len(input_ids) > MAX_LENGTH:
        input_ids = input_ids[:MAX_LENGTH]
        attention_mask = attention_mask[:MAX_LENGTH]
        labels = labels[:MAX_LENGTH] 
    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "label": labels,
    }
```

LlaMA 3.1 采用的 Prompt Tempalte 格式如下：

```
<|begin_of_text|>
<|start_header_id|>system<|end_header_id|>现在你要扮演皇帝身边的女人--甄嬛<|eot_id|>
<|start_header_id|>user<|end_header_id|>你好呀<|eot_id|>
<|start_header_id|>assistant<|end_header_id|>你好，我是甄嬛，你有什么事情要问我吗？<|eot_id|>
<|start_header_id|>assistant<|end_header_id|>
```

## 加载 tokenizer 和半精度模型

模型以 **半精度** 形式加载，如果显卡比较新的话，可以用 `torch.bfloat` 形式加载。
对于自定义的模型一定要指定 `trust_remote_code = True`。

```python
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

tokenizer = AutoTokenizer.from_pretrained(
    "/root/autodl-tmp/LLM-Research/Meta-Llama-3.1-8B-Instruct",
    use_fast = False,
    trust_remote_code = True,
)
model = AutoModelForCausalLM.from_pretrained(
    "/root/autodl-tmp/LLM-Research/Meta-Llama-3.1-8B-Instruct",
    device_map = "auto",
    torch_dtype = torch.bfloat16,
)
```

## 定义 LoraConfig

`LoarConfig` 这个类中可以设置很多参数，但主要的参数没多少：

* `task_type`: 模型类型
* `target_modules`: 需要训练的模型层的名字，主要就是 `attention` 部分的层，
  不同的模型对应的层的名字不同，可以传入数组，也可以是字符串，也可以是正则表达式
* `r`: LoRA 的秩
* `lora_alpha`: LoRA alpha

LoRA 的缩放是 `lora_alpha/r`，在这个 `LoraConfig` 中缩放就是 4 倍。

```python
config = LoraConfig(
    task_type = TaskType.CAUSAL_LM,
    target_modules = [
        "q_proj", "k_proj", "v_proj", 
        "o_proj", "gate_proj", 
        "up_proj", "down_proj"
    ],
    inference_mode = False,  # 训练模式
    r = 8,  # LoRA 秩
    lora_alpha = 32,  # LoRA alpha
    lora_dropout = 0.1,  # dropout 比例
)
```

## 自定义 TrainingArguments 参数

介绍一下 `TrainingArguments` 这个类的源码每个参数的具体作用：

* `output_dir`: 模型的输出路径
* `per_device_train_batch_size`: 顾名思义 `batch_size`
* `gradient_accumulation_steps`: 梯度累加，如果显存比较小，
  可以把 `batch_size` 设置小一点，梯度累加增大一些
* `logging_steps`: 多少步，输出一次 `log`
* `num_train_epochs`: 顾名思义 `epoch`
* `gradient_checkpointing`: 梯度检查，这个一旦开启，
  模型就必须执行 `model.enable_input_require_grads()`

```python
args = TrainingArguments(
    output_dir = "./output/llama3_1_instruct_lora",
    per_device_train_batch_size = 4,
    gradient_accumulation_steps = 4,
    logging_steps = 10,
    num_train_epochs = 3,
    save_steps = 100,
    learning_rate = 1e-4,
    save_on_each_node = True,
    gradient_checkpointing = True,
)
```

## 使用 Trainer 训练

```python
trainer = Trainer(
    model = model,
    args = args,
    train_dataset = tokenized_id,
    data_collator = DataCollatorForSeq2Seq(tokenizer = tokenizer, padding = True),
)
trainer.train()
```

## 加载 LoRA 权重权重推理

训练好之后可以使用如下方式加载 LoRA 权重进行推理。

```python
from torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

# 模型地址
mode_path = '/root/autodl-tmp/LLM-Research/Meta-Llama-3.1-8B-Instruct'
# lora 输出对应 checkpoint 地址
lora_path = '/root/autodl-tmp/output/llama3_1_instruct_lora/checkpoint-100'

# 加载 tokenizer
tokenizer = AutoTokenizer.from_pretrained(
    mode_path, 
    trust_remote_code = True
)
# 加载模型
model = AutoModelForCausalLM.from_pretrained(
    mode_path, 
    device_map = "auto",
    torch_dtype = torch.bfloat16, 
    trust_remote_code = True
).eval()
# 加载 loRA 权重
model = PeftModel.from_pretrained(
    model, 
    model_id = lora_path
)
# 输入
prompt = "你是谁？"
messages = [
    {
        "role": "system",
        "content": "假设你是皇帝身边的女人--甄嬛。",
    },
    {
        "role": "user",
        "content": prompt,
    }
]
input_ids = tokenizer.apply_chat_template(
    messages, 
    tokenize = False
)
model_inputs = tokenizer(
    [input_ids], 
    return_tensors = "pt"
).to('cuda')
generated_ids = model.generate(
    model_inputs.input_ids, 
    max_new_tokens = 512
)
generated_ids = [
    output_ids[len(input_ids):]
    for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
]
response = tokenizer.batch_decode(
    generated_ids, 
    skip_special_tokens = True
)[0]
print(response)
```

# FastAPI 部署和调用

1. 新建 `api.py` 脚本如下：

```python

```

2. 启动 API 服务

```bash
$ python api.py
```

3. API 调用

服务默认部署在 `6006` 端口，通过 POST 方法进行调用，可以使用 `curl` 调用，如下所示：

```bash
$ curl -X POST "http://127.0.0.1:6006" \
    -H 'Content-Type: application/json' \
    -d '{"prompt": "你好"}'
```

也可以使用 Python 的 `requests` 库进行调用：

```python
import json
import requests

def get_completion(prompt):
    headers = {
        "Content-Type": "application/jons",
    }
    data = {
        "prompt": prompt,
    }
    response = requests.post(
        url = "http://127.0.0.1:6006", 
        headers = headers, 
        data = json.dumps(data)
    )

    return response.json()["response"]




def main():
    print(get_completion(prompt = "你好"))

if __name__ == "__main__":
    main()
```



# Instruct WebDemo 部署


# 资料

* [AutoDL 平台 LlaMA3.1 镜像](https://www.codewithgpu.com/i/datawhalechina/self-llm/self-llm-llama3.1)
