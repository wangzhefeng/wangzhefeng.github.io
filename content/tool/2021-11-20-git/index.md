---
title: Git
author: 王哲峰
date: '2021-11-20'
slug: git
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

- [Git Install](#git-install)
  - [Ubuntu](#ubuntu)
    - [Install](#install)
    - [Config](#config)
  - [macOS](#macos)
    - [Install](#install-1)
    - [Config](#config-1)
- [Git Repository](#git-repository)
  - [ZIP Install](#zip-install)
  - [HTTP/HTTPS Clone](#httphttps-clone)
    - [Clone](#clone)
    - [SSH Clone](#ssh-clone)
  - [SSH Clone](#ssh-clone-1)
    - [SSH Key](#ssh-key)
    - [Clone](#clone-1)
  - [GitHub CLI](#github-cli)
    - [安装](#安装)
    - [使用](#使用)
- [Git flow](#git-flow)
  - [Create a Branch](#create-a-branch)
  - [Add Commit](#add-commit)
  - [Open a Pull Request](#open-a-pull-request)
  - [Discuss and Review your code](#discuss-and-review-your-code)
  - [Deploy](#deploy)
  - [Merge](#merge)
- [Git Branch](#git-branch)
- [Git Commit](#git-commit)
  - [commit message](#commit-message)
    - [为什么要写好 commit message](#为什么要写好-commit-message)
    - [基本要求](#基本要求)
- [Git Pull Request](#git-pull-request)
  - [Fork](#fork)
  - [Clone](#clone-2)
  - [Branch](#branch)
    - [确认分支](#确认分支)
    - [创建特性(feature)分支](#创建特性feature分支)
  - [Change](#change)
    - [修改项目代码](#修改项目代码)
    - [提交修改](#提交修改)
    - [创建远程分支](#创建远程分支)
  - [Pull Request](#pull-request)
- [Git Discuss and Review](#git-discuss-and-review)
- [Git Deploy](#git-deploy)
- [Git Merge](#git-merge)
  - [Git Merge 本仓库的分支](#git-merge-本仓库的分支)
    - [使用场景](#使用场景)
    - [操作步骤](#操作步骤)
  - [Git Merge 其他仓库的分支](#git-merge-其他仓库的分支)
    - [使用场景](#使用场景-1)
    - [操作步骤](#操作步骤-1)
  - [Git Merge 其他使用场景](#git-merge-其他使用场景)
- [Git .gitignore](#git-gitignore)
  - [设置 .gitignore](#设置-gitignore)
  - [更新 .gitignore](#更新-gitignore)
- [Git Fetch and Pull](#git-fetch-and-pull)
  - [git fetch](#git-fetch)
  - [git pull](#git-pull)
- [Git Status](#git-status)
- [Git Diff](#git-diff)
- [Git rm](#git-rm)
- [Git Checkout](#git-checkout)
- [Git Reset](#git-reset)
- [Git blame](#git-blame)
  - [简单使用](#简单使用)
  - [高阶使用](#高阶使用)
- [Git Large File Storage](#git-large-file-storage)
  - [安装](#安装-1)
  - [为账户设置 Git LFS](#为账户设置-git-lfs)
  - [为仓库设置 Git LFS](#为仓库设置-git-lfs)
  - [下一步](#下一步)
</p></details><p></p>

* [22 张图，我发现了 git 秘密](https://mp.weixin.qq.com/s/xrjru4jfzfIyl_p-84e1gA)

# Git Install

## Ubuntu

### Install

```bash
$ sudo apt-get install git
```

### Config

* 配置用户信息

```bash
$ git config --global user.name "wangzhefeng"
$ git config --global user.email "wangzhefengr@163.com"
```

* 配置 SSH、GitHub

```bash
$ ssh-keygen -t rsa -b 4096 -C "wangzhefengr@163.com"
```

## macOS

### Install

```bash
$ brew install git
```

### Config

> Git 自带一个 `git config` 的工具来帮助设置控制 Git 外观和行为的配置变量。
> 这些变量存储在三个不同的位置:
> 
> 1. `/etc/gitconfig` 文件
>    - 包含系统上每一个用户及他们仓库的通用配置。
>      如果在执行 `git config` 时带上 `--system` 选项，
>      那么它就会读写该文件中的配置变量。由于它是系统配置文件，
>      因此你需要管理员或超级用户权限来修改它
> 2. `~/.gitconfig` 或 `~/.config/git/config` 文件
>    - 只针对当前用户。你可以传递 `--global` 选项让 Git 读写此文件，
>    这会对你系统上 **所有** 的仓库生效
> 3. 当前使用仓库的 Git 目录中的 `config` 文件，即 `.git/config`
>    - 针对该仓库你可以传递 `--local` 选项让 Git 强制读写此文件，虽然默认情况下用的就是它。
>      当然，你需要进入某个 Git 仓库中才能让该选项生效
> 
> 每一个级别会覆盖上一级别的配置，所以 `.git/config` 的配置变量会覆盖 `/etc/gitconfig` 中的配置变量


* 用户信息

    - 针对系统的配置:

    ```bash
    # vim /etc/gitconfig
    $ sudo git config --system user.name "wangzhenfeng"
    $ sudo git config --system user.name "wangzhefengr@163.com"
    ```

    - 针对当前用户的配置：

    ```bash
    # vim ~/.gitconfig
    $ git config --global user.name "wangzhefeng"
    $ git config --global user.email "wangzhefengr@163.com"
    ```

    - 针对某个项目的配置，需要在该项目目录下进行配置：

    ```bash
    # vim project/.git/config
    $ git config user.name "wangzhefeng"
    $ git config user.email "wangzhefeng@163.com"
    ```

* 文本编辑器

    ```bash
    $ git config --global core.editor vim
    ```

* 检查配置信息

    ```bash
    $ git config --list
    $ git config user.name
    $ git config user.email
    $ git config core.editor
    $ git config --show-origin rerere.autoupdate
    ```

* 获取 Git 帮助

    ```bash
    $ git help <verb>
    $ git <verb> --help/-h
    $ man git-<verb>
    ```

* 配置 SSH、GitHub

    ```bash
    $ ssh-keygen -t rsa -b 4096 -C "wangzhefengr@163.com"
    ```


# Git Repository

## ZIP Install

1. 下载 repo.zip
2. 解压 repo.zip
3. 进入解压后的 repo

```bash
$ cd repo
```

4. 将 `repo` 关联到 GitHub remote

```bash
$ git init
$ git remote add origin <remote repository URL>
$ git remote -v
$ git pull origin master
```

## HTTP/HTTPS Clone

### Clone

```bash
$ git clone https://github.com/username/repos.git
```

### SSH Clone

将使用 HTTP/HTTPS 克隆的仓库转换为 SSH 方式验证

1. 查看现在的仓库远程地址配置

```bash
git remote -v
```

2. 删除现在的远程地址配置

```bash
git remote rm origin
```

3. 添加新的远程地址配置

```bash
git remote add origin git@github.com:example.git
git remote set-url origin git@github.com:example.git
```

4. 现在可以使用 SSH 方式进行验证

```bash
git pull
...
# 第一次 git push 时需要执行以下操作
git push
git push --set-upstream origin master
```

## SSH Clone

### SSH Key

1. 检查本地是否已有 SSH keys

```bash
$ ls -al /.ssh
```

2. 生成一个新的 SSH key

```bash
$ ssh-keygen -t rsa -b 4096 -C "your_email@example.com"
```

3. 把生成的 SSH key 加入 ssh-agent

Start the ssh-agent in the background

```bash
$ eval "$(ssh-agent -s)"
```

Modify your `/.ssh/config` file to automatically load keys
into the ssh-agent and store passphrases in your keychain

```bash
$ open /.ssh/config
$ touch /.ssh/config
$ vim /.ssh/config

Host alia_name
AddKeysTzoAgent yes
UseKeychain yes
IdentityFile /.ssh/id_rsa
```

Add your SSH private key to the ssh-agent and store your passphrase in the keychain

```bash
# mac
$ ssh-add -K /.ssh/id_rsa

# Linux
$ ssh-add /.ssh/id_rsa
```

4. 将 SSH Public key 加入 GitHub 账号

### Clone

```bash
$ git clone git@github.com:username/repos.git
```

## GitHub CLI

### 安装

```bash
$ brew install gh
```

### 使用

- https://cli.github.com/

# Git flow

GitHub flow 是一个轻量级的、基于分支(branch)的工作流, 它支持定期进行部署的团队和项目. 

## Create a Branch

![git](images/git1.png)

- When you're working on a project, you're going to have a bunch of
    different features or ideas in progress at any given time – some of
    which are ready to go, and others which are not. Branching exists to
    help you manage this workflow.

- When you create a branch in your project, you're creating an
    environment where you can try out new ideas. Changes you make on a
    branch don't affect the `master` branch, so you're free to
    experiment and commit changes, safe in the knowledge that your branch
    won't be merged until it's ready to be reviewed by someone you're
    collaborating with.

- **ProTip**

    - Branching is a core concept in Git, and the entire GitHub flow is
        based upon it. There's only one rule: anything in the `master`
        branch is always deployable. Because of this, it's extremely
        important that your new branch is created off of master when
        working on a feature or a fix. Your branch name should be
        descriptive (e.g., `refactor-authentication`,
        `user-content-cache-key`, `make-retina-avatars`), so that
        others can see what is being worked on.

## Add Commit

![git](images/git2.png)

- Once your branch has been created, it's time to start making changes.
    Whenever you add, edit, or delete a file, you're making a commit, and
    adding them to your branch. This process of adding commits keeps
    track of your progress as you work on a feature branch.

- Commits also create a transparent history of your work that others
    can follow to understand what you've done and why. Each commit has an
    associated commit message, which is a description explaining why a
    particular change was made. Furthermore, each commit is considered a
    separate unit of change. This lets you roll back changes if a bug is
    found, or if you decide to head in a different direction.

- **ProTip**

    - Commit messages are important, especially since Git tracks your
        changes and then displays them as commits once they're pushed to
        the server. By writing clear commit messages, you can make it
        easier for other people to follow along and provide feedback.

## Open a Pull Request

![git](images/git3.png)


- Pull Requests initiate discussion about your commits. Because they're
    tightly integrated with the underlying Git repository, anyone can see
    exactly what changes would be merged if they accept your request.

- You can open a Pull Request at any point during the development
    process: when you have little or no code but want to share some
    screenshots or general ideas, when you're stuck and need help or
    advice, or when you're ready for someone to review your work. By
    using GitHub's @mention system in your Pull Request message, you can
    ask for feedback from specific people or teams, whether they're down
    the hall or ten time zones away.

- **ProTip**

    - Pull Requests are useful for contributing to open source projects
        and for managing changes to shared repositories. If you're using a
        Fork & Pull Model, Pull Requests provide a way to notify project
        maintainers about the changes you'd like them to consider. If
        you're using a Shared Repository Model, Pull Requests help start
        code review and conversation about proposed changes before they're
        merged into the master branch.

## Discuss and Review your code

![git](images/git4.png)

- Once a Pull Request has been opened, the person or team reviewing
    your changes may have questions or comments. Perhaps the coding style
    doesn't match project guidelines, the change is missing unit tests,
    or maybe everything looks great and props are in order. Pull Requests
    are designed to encourage and capture this type of conversation.

- You can also continue to push to your branch in light of discussion
    and feedback about your commits. If someone comments that you forgot
    to do something or if there is a bug in the code, you can fix it in
    your branch and push up the change. GitHub will show your new commits
    and any additional feedback you may receive in the unified Pull
    Request view.

- **ProTip**

    - Pull Request comments are written in Markdown, so you can embed
        images and emoji, use pre-formatted text blocks, and other
        lightweight formatting.

## Deploy

![git](images/git5.png)
   
 - With GitHub, you can deploy from a branch for final testing in
    production before merging to master.
 - Once your pull request has been reviewed and the branch passes your
    tests, you can deploy your changes to verify them in production. If
    your branch causes issues, you can roll it back by deploying the
    existing master into production.

## Merge


![git](images/git6.png)
   

- Now that your changes have been verified in production, it is time to
  merge your code into the master branch.

- Once merged, Pull Requests preserve a record of the historical
  changes to your code. Because they're searchable, they let anyone go
  back in time to understand why and how a decision was made.

- **ProTip**

    - By incorporating certain keywords into the text of your Pull
    Request, you can associate issues with code. When your Pull
    Request is merged, the related issues are also closed. For
    example, entering the phrase `Closes #32` would close issue
    number 32 in the repository. For more information, check out our
    [help article](https://help.github.com/articles/closing-issues-via-commit-messages).

# Git Branch

- [Git branching model](https://nvie.com/files/Git-branching-model.pdf)

* 创建并切换分支

```bash
$ git checkout -b tinker
```

```bash
$ git branch tinker
$ git checkout tinker
```

* 查看当前分支

```bash
$ git branch
```

* 切换分支

```bash
git checkout master
```

* 合并分支

```bash
$ git checkout master
$ git merge tinker
```

* 删除分支


```bash
$ git branch -d tinker
```

# Git Commit




## commit message

### 为什么要写好 commit message

* 加快 Reviewing Code 的过程
* 帮助我们写好 release note
* 5 年后帮你快速想起来某个分支，tag 或者 commit 增加了什么功能，改变了哪些代码
* 让其他的开发者在运行 git blame 的时候想跪谢
* 总之一个好的提交信息，会帮助你提高项目的整体质量

### 基本要求

* 永远不要在 `git commit` 上增加 `-m <msg>` 或 `--message=<msg>` 参数，
  要单独编写 commit message
    - 一个不好的例子: `git commit -m "Fix login bug"`
    - 一个推荐的 commit message
    
    ```bash
    $ git commit
    ```

    ```
    Redirect user to the requested page after login

    https://trello.com/path/to/relevant/card

    Users were being redirected to the home page after login, which is less
    useful than redirecting to the page they had originally requested before
    being redirected to the login form.

    * Store requested path in a session variable
    * Redirect to the stored location after successfully logging in the user
    ```

* 第一行应该少于 50 个字，随后是一个空行
* 用空行来分割 commit message，让它在某些软件里面更容易读
* 使用 fix, add, change 而不是 fixed, added, changed
* 注释最好包含一个连接指向项目的 `issue/story/card`，一个完整的 issue numbers 更好
* commit message 中包含一个简短的故事，能让别人更容易理解你的项目
* 请将每次提交限定于完成一次逻辑功能。并且可能的话，适当地分解为多次小更新，以便每次小型提交都更易于理解
* 喜欢用 vim 的可以把下面这行代码加入到 `.vimrc` 文件中，来检查和自动换行

```js
autocmd Filetype gitcommit setlocal spell textwidth=72
```



# Git Pull Request

pull request 是社会化编程的象征, 通过这个功能, 你可以参与到别人开发的项目中, 并作出自己的贡献. 
pull request 是自己修改源代码后, 请求对方仓库采纳的一种行为.

## Fork

找到想要 `pull request` 的项目 `test` , 然后点击 `fork`
按钮, 此时自己的仓库中就会有一个别人的项目仓库, 名字为:  `you_github_name/test` . 

## Clone

在把想要 `pull request` 的项目克隆到本地环境: 

```bash
git clone https://github.com/wangzhefeng/test.git
```

## Branch

### 确认分支

通过在终端运行命令查看当前项目所在的分支, 通常都是在查看分支后再进行代码的修改, 这是个好习惯. 

```bash
git branch -a
```

### 创建特性(feature)分支


在应用 GitHub 修改代码时, 常常采用的策略是在主分支(`master`)下再创建一个特性分支(`feature_a`), 
在该特性分支下进行代码的修改, 然后通过该分支执行 `pull request` 操作. 
通过命令:  `git checkout -b feature_a master` (其中 `feature_a` 为新建的特性分支,  `master`
为当前所在的分支)创建新的特性分支并自动切换. 

```bash
git checkout -b feature_a master
```

## Change

### 修改项目代码

在创建的 `feature_a` 分支下对 `fork` 的项目内容进行修改. 

### 提交修改

```bash
# 查看修改的内容是否正确
git diff

# 增加修改内容说明文档
git add README.md

# 提交修改
git commit -m "add README.md"
```

### 创建远程分支

要从 GitHub 发送 `pull request` , GitHub
端的仓库中必须有一个包含了修改后的代码的分支, 
所以需要创建一个与刚刚创建的特性分支（修改所在的分支）相对应的远程分支, 
执行命令: 

```bash
git push origin work1
```

其中:  `origin` 为当时 `fork` 的远程主分支的名称, 
一般默认为 `origin`, `feature_a` 为本地工作的特性分支.

然后进行查看是否创建成功: 

```bash
$ git branch -a
```

## Pull Request

进入到自己的 GitHub 账户下, 并切换到创建的特性分支(`feature_a`)下, 
点击 `create pull request` , 确定没问题, 并填写相关修改内容, 
点击 `send pull request`.


# Git Discuss and Review


# Git Deploy


# Git Merge

## Git Merge 本仓库的分支

### 使用场景

仓库 repos 的分支 master 需要合并到其 dev 分支上

### 操作步骤

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

## Git Merge 其他仓库的分支

### 使用场景

仓库 A_repos 的分支 branch_A 分支 需要合并到本地仓库 B_repos 的 branch_B 分支

远程仓库 A_repos: A_repos:branch_A
本地仓库 B_repos: B_repos:branch_B

### 操作步骤

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
$ git merge branch_A_B
// git merge 分支名称
```

如果在 merge 的过程中报错: `fatal: refusing to merge unrelated histories`，
则需要使用如下命令及参数

```bash
$ git merge branch_A_B --allow-unrelated-histories
```




## Git Merge 其他使用场景

* TODO

# Git .gitignore

- [.gitignore template](https://github.com/github/gitignore)

## 设置 .gitignore

## 更新 .gitignore

Git 更新 ignore 文件直接修改 `.gitignore` 是不不会生效的, 
需要先去掉已经托管的文件, 修改完成之后再重新添加并提交.

1. 删除已经托管的文件

```bash
$ git rm -r --cached path_to_file
```

2. 修改 `.gitignore` 文件的内容
3. 提交修改

```bash
$ git add .
$ git commit -m "clear cached"
$ git push
```

# Git Fetch and Pull

从远程分支拉取最新的版本到本地

## git fetch

- 从远程获取最新版本到本地, 不会自动merge

```bash
# 从远程的origin的master主分支下载最新的版本到origin/master分支上
$ git fetch origin master

# 比较本地的master分支和origin/master分支的差别
$ git log -p master..origin/master

# 合并
$ git merge origin/master
```

or

```bash
$ git fetch origin master:tmp
$ git diff tmp
$ git merge tmp
```

## git pull

从远程获取最新版本到本地, 自动merge

```bash
$ git pull origin master
```

# Git Status

查看当前文件状态

```bash
$ git status
$ git status -s
$ git status --short
```

# Git Diff

* 查看已暂存和未暂存的修改

```bash
$ git diff 
$ git diff --cached
$ git diff --staged
```

* 查看工作区和版本库里最新版本的区别

```bash
$ git diff HEAD --file.txt
```

# Git rm

移除文件(从暂存区域移除)

* 确定要从版本库中删除文件

```bash
$ git rm
$ git commit -m "message"
```

or

```bash
$ git add .
```

# Git Checkout

误删工作区文件, 将版本库中的文件替换工作取得文件

```bash
$ git checkout -- test.txt
```

# Git Reset

* 提交日志

```bash
$ git log --pretty=oneline
```

* 查看命令历史, 记录每次命令

```bash
$ git reflog
```

* 版本回退

回退到上一个版本(HEAD: 当前版本)

```bash
$ git log
$ git reset --hard HEAD^
$ git reset --hard HEAD10
$ git reset --hard commit_id
```

前进到某个版本

```bash
$ git reflog
$ git reset --hard commit_id
```

* 撤销修改
    - 丢弃工作区的修改 
        - 自修改后还未被放到暂存区(还未进行git add) => 回到和版本库一样的状态
        - 添加到暂存区后又作了修改(进行了git add) => 添加到暂存区后的状态

```bash
$ git checkout -- file.txt
```

* 把暂存区的修改回退到工作区(丢弃暂存区的修改)

```bash
$ git reset HEAD file.txt
$ git checkout --file.txt
```

# Git blame

查看某个文件的某行代码的修改历史

## 简单使用

```bash
$ git blame <filename>
$ git blame -L 100,100 <filename>
$ git blame -L 100,+10 <filename>
```

## 高阶使用

* 使用 log 来查看某一行的所有 commit

```bash
git log -L start,end:file
git log -L 155,155:git-web-browse.sh
```

* 找到某一 commit，在 GitHub 上查看修改的那一次 commit
    - https://github.com/<user_name>/<project_name>/commit/<commit_hash_id>

# Git Large File Storage

- https://git-lfs.github.com/

## 安装 

```bash
$ brew install git-lfs
```

## 为账户设置 Git LFS

每个账户只需设置一次

```bash
git lfs install
```

## 为仓库设置 Git LFS

在每个要使用 Git LFS 的 Git repository 中选择要让 Git LFS 管理的文件格式.
可以随时进行设置

```bash
$ git lfs track "*.psd"
```

现在要确保 `.gitattributes` 是被跟踪的

```bash
$ git add .gitattributes
```

## 下一步

```bash
$ git add file.psd
$ git commit -m "add file.psd"
$ git push origin main
```
