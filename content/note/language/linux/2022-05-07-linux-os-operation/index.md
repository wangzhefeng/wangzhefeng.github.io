---
title: Linux 系统操作
author: 王哲峰
date: '2022-05-07'
slug: linux-os-operation
categories:
  - Linux
tags:
  - tool
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

- [Linux Help](#linux-help)
  - [`man`: 有问题找男人帮忙](#man-有问题找男人帮忙)
  - [`help`](#help)
  - [`info`](#info)
  - [其他](#其他)
- [文件查看](#文件查看)
  - [`pwd` 显示当前的目录名称](#pwd-显示当前的目录名称)
  - [`cd` 更改当前的操作目录](#cd-更改当前的操作目录)
  - [`ls` 文件查看](#ls-文件查看)
- [目录文件操作](#目录文件操作)
  - [创建目录](#创建目录)
  - [删除目录](#删除目录)
  - [复制目录](#复制目录)
  - [移动、重命名目录](#移动重命名目录)
- [通配符](#通配符)
- [文件操作](#文件操作)
  - [用户与权限管理](#用户与权限管理)
    - [新建用户](#新建用户)
    - [修改用户密码](#修改用户密码)
    - [删除用户](#删除用户)
    - [修改用户属性](#修改用户属性)
    - [修改用户生命周期](#修改用户生命周期)
    - [组管理命令](#组管理命令)
    - [用户切换](#用户切换)
  - [打包压缩和解压缩](#打包压缩和解压缩)
    - [打包](#打包)
- [文本内容查看](#文本内容查看)
  - [文件内容查看](#文件内容查看)
  - [强大的文本编辑器 Vi/Vim](#强大的文本编辑器-vivim)
    - [多模式文本编辑器](#多模式文本编辑器)
    - [正常模式](#正常模式)
    - [插入模式](#插入模式)
    - [可视模式](#可视模式)
    - [命令模式](#命令模式)
- [其他命令](#其他命令)
</p></details><p></p>




**Linux 中一切皆文件**

- 文件查看
- 目录文件操作创建、删除、复制、移动
- 通配符
- 文件操作
- 文本内容查看

# Linux Help

为什么要学习帮助命令?

- Linux 的基本操作方式是命令行
- 海量的命令不适合"死记硬背"
- 你要升级你的大脑

常用帮助方式:

- man 帮助
- help 帮助
- info 帮助

## `man`: 有问题找男人帮忙

- `man` 是 manual 的缩写
- `man` 帮助用法演示

```bash
$ man 1 ls
$ man ls
```

- `man` 也是一条命令, 分为 9 章, 可以用 `man` 命令获得 `man` 的帮助
    - 获取 `man` 的帮助

```bash
$ man man
```

- 获取 `man` 章节的帮助

```bash
$ man 7 man
```

- 1 Command
- 2 System calls(编程函数)
- 3 Library calls(编程函数)
- 4 Special files(文件)
- 5 File formats and conventions(文件)
- 6 Games
- 7 Macro packages and conventions
- 8 System management commands
- 9 Kernel routines(废弃)

- 使用示例:

```bash
# 命令帮助
$ man 1 password

# 配置文件帮助
$ man 5 password

# 只知道名字获取帮助
$ man -a password
```

- 其他
    - 根据命令部分关键字搜索

```bash
$ man date
$ man -k date
```

- 命令参数及使用方法

```bash
$ man ls
```

## `help`

- shell(命令解释器) 自带的命令称为内部命令, 其他的是外部命令
- 内部命令使用 `help` 帮助

```bash
$ help cd
```

- 外部命令使用 `help` 帮助

```bash
$ ls --help
```

- 查看命令的类型

```bash
$ type cd 
$ type ls
```

## `info`

- `info` 帮助比 `help` 帮助更详细, 作为 `help` 帮助的补充

```bash
$ info ls
```

## 其他

- 命令简要说明

```bash
$ whatis ls
$ whatis -w "mkd*"
```

- 查看程序的binary文件所在路径

```bash
$ which python
```

- 查看程序的搜索路径

```bash
$ whereis python
```

# 文件查看

- `pwd`
- `cd`
- `ls`

## `pwd` 显示当前的目录名称

- `pwd` 显示当前的目录名称

```bash
$ man pwd
$ pwd
```

- `/root`: root 用户的家目录
- `/`: 根目录

## `cd` 更改当前的操作目录

- `cd` 更改当前的操作目录
    - 绝对路径
        - `cd /path/to/...`
    - 相对路径
        - `cd ./path/to/...`
            - `./` 代表当前目录,可省略
        - `cd ../path/to/...`
            - `../` 代表上级目录

- 常用命令

```bash
# 回到之前的一个目录
$ cd -
$ cd ./
$ cd .
$ cd ../
$ cd ..
```

## `ls` 文件查看

- `ls` 查看当前目录下的文件
    - `ls [选项, 选项...] 参数...`
    - `ls [选项] [文件夹...]`

- 常用参数
    - `-l` 长格式显示文件
    - `-a` 显示隐藏文件
    - `-r` 逆序显示
    - `-t` 按照时间顺序显示
    - `-R` 递归显示

- 使用示例

```bash
# 使用
$ ls -l /root /
$ ls -l .
$ ls -l 
$ ls -l -r
$ ls -l -r -t
$ ls -lrt
$ ls -lartR
$ clear(or Ctrl + l)
```

- `-rw-------.`
    - `-` 第一个 `-` 代表普通文件
- `drwxr-xr-x.`
    - `d` 第一个 `d` 代表目录

# 目录文件操作

- 创建
- 删除
- 复制
- 移动

## 创建目录

- `mkdir` 新建目录
- `mkdir -p` 建立多级目录
- `touch` 新建文件
- 使用方法

```bash
$ man mkdir

# 相对目录
$ mkdir ./dir_name1 dir_name2 dir_name3 ...

# 绝对目录
$ mkdir /path/dir_name1 dir_name2 dir_name3 ...

# 创建多级目录
$ mkdir ./a
$ mkdir ./a/b
$ mkdir ./a/b/c
$ mkdir ./a/b/c/d
# 与上面等效
$ mkdir -p ./a/b/c/d

$ ls -R ./a

# 新建文件
$ touch file1 file2 file3 ...
```

如果目录已经存在, 创建目录会导致报错. 

## 删除目录

- `rmdir` 删除空目录
- `rm -r` 删除非空目录
- `rm -rf` 删除非空目录
- 使用方法

```bash
$ rmdir ./a

# 删除非空目录目录, 需要确认删除
$ rm -r ./dir1 ./dir2 ...

# 删除非空目录目录, 需要确认删除
$ rm -rf ./dir1 ./dir2 ...
```

- `rmdir` 只能删除空目录

## 复制目录

- `cp` 复制目录
- 使用方法

```bash
# 复制文件
$ cp 源文件 目的文件目录

# 复制目录
$ cp -r 源文件目录 目的文件目录

# 复制目录并显示复制信息
$ cp -v 源文件目录 目的文件目录

# 复制目录并保持属主
$ cp -p 源文件目录 目的文件目录

# 复制目录并保持属主、时间...
$ cp -a 源文件目录 目的文件目录
```


- `/tmp` 目录为临时文件目录


## 移动、重命名目录

- `mv` 移动、重命名目录
- 使用方法

```bash
# 移动文件
$ mv 源文件 目的文件目录

# 重命名文件名
$ mv 源文件 重命名后的文件
$ mv 源文件 目的文件目录/重命名后的文件

# 移动目录
$ mv 源文件目录 目的文件目录

# 重命名目录
$ mv 源文件 重命名后的文件 
```

# 通配符

- 定义: shell 内建的符号
- 用途: 操作多个相似(有简单规律)的文件
- 常用通配符
  - `*` 匹配任何字符串
  - `?` 匹配 1 个字符串
  - `[xyz]` 匹配 xyz 任意一个字符
  - `[a-z]` 匹配一个范围
  - `[!xyz]` 或 `[^xyz]` 不匹配


# 文件操作

## 用户与权限管理

用户管理常用命令:

- `useradd username` 新建用户
- `userdel username` 删除用户
- `passwd` 修改用户密码
- `usermod` 修改用户属性
- `chage` 修改用户属性(声明周期属性)

### 新建用户

1.切换到 root 用户

```bash
$ su - root
```

2.新建用户

```bash
$ useradd user_test
```

3.查看用户信息

```bash
$ id

$ id root
$ id user_test
```

4. 查看用户家目录

```bash
$ ls /root
$ ls /home/user_test/
$ ls -a /home/user_test/
```

5.查看用户配置文件

```bash
# 用户记录
$ tail -10 /etc/passwd

# 用户密码记录
$ tail -10 /etc/shadow
```

### 修改用户密码

1.切换到 root 用户

```bash
$ su - root
```

2.修改用户密码

```bash
# 修改自己密码
$ passwd

# 修改其他用户密码
$ passwd user_test
```

### 删除用户

1.切换到 root 用户

```bash
$ su - root
```

2.删除用户

```bash
$ userdel user_test
$ userdel -r user_test  # 彻底删除, 包块/home目录下的用户目录
$ id user_test
$ tail -10 /etc/passwd
```

### 修改用户属性

1.切换到 root 用户

```bash
$ su - root
```

2.修改用户账号

```bash
# 修改用户家目录
$ usermod -d /home/user_test_2 user_test
```

### 修改用户生命周期

1.切换到 root 用户

```bash
$ su - root
```

### 组管理命令

组管理命令:

- `groupadd` 新建用户组
- `groupdel` 删除用户组


1.切换到 root 用户

```bash
$ su - root
```

2.新建用户组

```bash
# 新建组、新建用户、修改用户组
$ groupadd group1 
$ useradd user1
$ usermod -g group1 user1

# 新建用户并加入组
$ useradd -g group1 user2
```

3.删除用户组

```bash
$ groupdel group1
```

### 用户切换

用户切换命令: 

- `su` 切换用户
    - `su - USERNAME` 使用 login shell 方式切换用户
- `sudo` 以其他用户身份执行命令
    - `visudo` 设置需要使用 `sudo` 的用户(组)

1.普通用户 => root用户(需要输入 root 密码)

```bash
$ su - root
$ 
```

2.root用户 => 普通用户

```bash
# 完全切换
$ su - user_test

# 不完全切换
$ su user_test
$ cd /home/user_test
```

## 打包压缩和解压缩

- 最早的 Linux 备份介质是磁带, 使用的命令是 `tar`
- 可以打包后的磁带文件进行压缩存储, 压缩的命令是 `gzip` 和 `bzip2`
- 经常使用的扩展名是 `.tar.gz`、`.tar.bz2`、`.tgz`
- `tar` 打包命令
    - `c` 打包
    - `x` 解包
    - `f` 指定操作类型为文件


### 打包

- 打包
        
```bash
$ tar cf /new_path/new_file.tar /path_to_be_tar
```

- 打包并压缩

```bash
# 压缩更快 (.tgz == .tar.gz)
$ tar czf /new_path/new_file.tar.gz /path_to_be_tar_zip

# 更高的压缩比例 (.tbz2 == .tar.bz2)
$ tar cjf /new_path/new_file.tar.bz2 /path_to_be_tar_zip
```

- 解包并解压缩

```bash
$ tar xf /new_path/new_file.tar -C /path_to_save
$ tar zxf /new_path/new_file.tar.gz -C /path_to_save 
$ tar jxf /new_path/new_file.tar.bz2 -C /path_to_save
```

- 压缩、解压缩
    - gzip new_file.tar
    - bzip2 new_file.tar




# 文本内容查看

## 文件内容查看

- 文本查看命令
    - `cat` 文本内容显示到终端
        - `cat filename`
    - `head` 查看文件开头
        - `head -n filename`
    - `tail` 查看文件结尾
        - `tail -3 filename`
        - 常用参数 `-f` 文件内容更新后, 显示信息同步更新            
            - `tail -f filename`
    - `wc` 统计文件内容信息(文件长度)
        - `wc -l filename`    
    - `less`
    - `more`


## 强大的文本编辑器 Vi/Vim

### 多模式文本编辑器

- 四种模式

    - 正常模式(Normal-mode)
    - 插入模式(Insert-mode)
    - 命令模式(Command-mode)
    - 可视模式(Visual-mode)

- 进入编辑器

```bash
$ vi
$ vim
$ vim file_name
```

### 正常模式

- 进入正常模式        
    - `Esc`
- 光标移动
    - `h`: 左
    - `j`: 上
    - `k`: 下
    - `l`: 右
- 复制文本
    - `yy`: 复制当前整行
    - `[n]yy`: 复制当前行下面的多行
    - `y$`: 复制光标位置到当前行的结尾
- 粘贴文本
    - `p`    
- 剪切文本
    - `dd`: 剪切光标所在的行
    - `[n]dd`: 剪切当前行下面的多行
    - `d$`: 剪切光标位置到当前行的结尾    
- 撤销
    - `u`: 撤销
    - `Ctrl + r`: 重做(撤销撤销)    
- 单个字符删除
    - `x`: 光标移动到要删除的字符上
- 单个字符替换
    - `r`: 光标移动到要删除的字符上
- 移动到指定的行
    - `[n]G`: 移动到第 n 行
    - `g`: 移动到第一行
    - `G`: 移动到最后一行
    - `^`: 移动到当前行的开头
    - `$`: 移动到当前行的结尾


### 插入模式

- `I`: 进入插入模式, 光标处于插入之前行的开头
- `i`: 进入插入模式, 光标处于插入之前的位置
- `A`: 进入插入模式, 光标处于插入之前的行结尾
- `a`: 进入插入模式, 光标处于插入之前的行的下一位
- `O`: 进入插入模式, 光标处于插入之前的行的上一行
- `o`: 进入插入模式, 光标处于插入之前的行的下一行

### 可视模式

- `v`: 字符可视模式
- `V`: 行可视模式
- `Ctrl + V`: 块可视模式
    - `d`: 多行删除
    - `I` + 连续两次按 `Esc`: 多行插入

### 命令模式

- (1)进入命令模式、末行模式

    - `:`

- (2)保存文件

    - 保存
        - `:w /zfwang/filename.sh`: 保存到指定文件
        - `:w`: 保存到当前文件
    - 退出
        - `:q`
    - 保存退出
        - `:wq` 
    - 不保存退出
        - `:q!` 

- (3)执行 Linux 命令

    - `:![command]`        
    - `:!ifconfig`: 查看ip地址

- (4)查看、查找字符

    - 查找字符
        - `/[str]`
        - `/[str]` + `n` 查找到的字符下移光标
        - `/[str]` + `N`: 查找到的字符上移光标
    - 替换查找到的字符
        - `:s/old_str/new_str`: 只替换光标所在行的目标字符
        - `:%s/old_str/new_str`: 替换整个文件的第一个目标字符
        - `:%s/old_str/new_str/g`: 替换整个文件的目标字符
        - `:[n],[m]s/old_str/new_str/g`: 替换第n行到第m行的目标字符
    
- (5)vim 编辑器配置项设置

    - 显示/不显示行号
        - `:set nu`
        - `:set nonu`
    - 去掉高亮显示
        - `:set nohlsearch` 
    - 设置 vim 的配置文件

```bash
# 打开 /etc/vimrc
$ vim /etc/vimrc

# /etc/vimrc 文件修改
set nu
```

# 其他命令

- 系统关机

```bash
# 当前用户 30 分钟后关闭系统
$ shutdown -h 30

# 当前用户取消关机
$ shutdown -c
```