---
title: Git Merge
author: 王哲峰
date: '2022-07-10'
slug: git-merge
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
</style>


<details><summary>目录</summary><p>

- [Git Merge 本仓库的分支](#git-merge-本仓库的分支)
  - [使用场景](#使用场景)
  - [操作步骤](#操作步骤)
- [Git Merge 其他仓库的分支](#git-merge-其他仓库的分支)
  - [使用场景](#使用场景-1)
  - [操作步骤](#操作步骤-1)
- [Git Merge 其他使用场景](#git-merge-其他使用场景)
</p></details><p></p>




# Git Merge 本仓库的分支

## 使用场景

仓库 repos 的分支 master 需要合并到其 dev 分支上

## 操作步骤

1. 获取 master 分支最新代码

```bash
$ cd repos
$ git checkout master
$ git pull
```

2. 获取 dev 分支最新代码

```bash
$ git checkout dev
$ git pull
```

3. Merge

```bash
$ git merge master
```

4. 提交 Merge  后的修改
  - 如果有冲突，解决冲突，提交
  - 如果没有冲突，提交

```bas
$ git add .
$ git commit -m "merge master"
$ git push
```

# Git Merge 其他仓库的分支

## 使用场景

仓库 A_repos 的分支 branch_A 分支 需要合并到本地仓库 B_repos 的 branch_B 分支

远程仓库 A_repos: A_repos:branch_A
本地仓库 B_repos: B_repos:branch_B

## 操作步骤

1. 将远程仓库 A_repos 的地址添加到本地仓库 B_repos 的远程仓库中

```bash
$ cd path/to/B_repos
$ git checkout branch_B

$ git remote add A_respoBranch_A git@github.com:username/A_repos.git
// git remote add 仓库B_repos的另一个远程仓库名称 仓库A_repos远程地址
```

现在可以看到本地仓库 B_repos 有两个远程仓库:

```bash
$ git remote
A_respoBranch_A
origin
```

2. 抓取远程仓库 A_respo 数据到本地仓库 A_respoBranch_A 中

```bash
$ git fetch A_respoBranch_A
// git fetch 本地仓库A名称
```

3. 为仓库 A_repos 创建一个新的分支 branch_A_B

这一步是将远程仓库 A_repos 的代码在本地新建一个分支 branch_A_B，
稍后会将这个 A_respo/branch_A_B 分支的代码和本地 B_repos/branch_B 代码 merge，
这样也就是将仓库 A_repos 代码合并到 B_repos 代码仓库

```bash
git checkout -b branch_A_B A_respoBranch_A/branch_A
// git checkout -b 远程仓库A_repos本地新分支名称 远程仓库A_repos本地名称/远程仓库A_repos远程分支名称
```

4. 切换到本地分支

现在本地有两个分支: 一个是之前的 branch_B，这个分支就是 B_repos 仓库的代码; 
一个新增的分支 branch_A_B，这个是远程仓库 A_repos 的代码

```bash
$ git checkout branch_B
```

5. 合并两个分支

```bash
$ git merge branch_a
// git merge 分支名称
```

# Git Merge 其他使用场景

* TODO