---
title: Linux 后台执行命令
author: 王哲峰
date: '2023-07-23'
slug: linux-hangup
categories:
  - Linux
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

- [查看当前 Linux 后台运行的程序](#查看当前-linux-后台运行的程序)
- [命令 `&`](#命令-)
- [命令 nohup](#命令-nohup)
- [杀死后台运行的任务](#杀死后台运行的任务)
</p></details><p></p>

# 查看当前 Linux 后台运行的程序

- `jobs` 命令可以查看当前有多少任务在后台运行
- `jobs -l` 可以查看到当前所有在后台运行任务的 PID，任务状态等信息

```bash
jobs
jobs -l
```

# 命令 `&`

- 在程序的后面加上一个 `&` 命令后，程序就可以在后台运行了。

```bash
# test.py 在后台运行，但是程序的 log 会输出到当前终端
./test.py &

# 将 log 重定向到指定的文件
./test.py >> test.log 2>&1 &
```

- 其中： `2>&1` 是将标准错误重定向到标准输出，于是标准错误和标准输出都重定向到指定的 `test.log` 文件中；
- 其中：最后一个 `&`的意思是 `test.py`在后台运行；

# 命令 nohup

- 在命令的末尾加上一个 `&` 符号后，程序可以在后台运行，但是一旦当前终端关闭（即退出当前帐户），该程序就会停止运行。那假如说我们想要退出当前终端，但又想让程序在后台运行，该如何处理呢？
- 这个需求在现实中很常见，比如想远程到服务器编译程序，但网络不稳定，一旦掉线就编译就中止，就需要重新开始编译，很浪费时间。
- 可以使用 `nohup` 命令。`nohup` 就是不挂起的意思( no hang up)。该命令的一般形式如下。

```bash
nohup ./test.py &
exit

nohup ./test.py > test.log 2>&1 &
exit
```

- 使用了 `nohup` 之后，很多人就不管了，其实这样有可能在当前账户非正常退出或者结束的时候，命令还是自己结束了。所以在使用 `nohup` 命令后台运行命令之后，需要使用 `exit` 正常退出当前账户，这样才能保证命令一直在后台运行。

# 杀死后台运行的任务
