---
title: 玩转 SSH
author: 王哲峰
date: '2021-01-25'
slug: ssh
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

- [安装 SSH 服务(Debina)](#安装-ssh-服务debina)
  - [安装 SSH](#安装-ssh)
  - [SSH 无法连接的场景排查](#ssh-无法连接的场景排查)
    - [检查网络](#检查网络)
    - [检查端口](#检查端口)
    - [检查服务](#检查服务)
- [什么是 SSH](#什么是-ssh)
- [SSH 工作原理](#ssh-工作原理)
- [SSH 存在的问题--中间人攻击](#ssh-存在的问题--中间人攻击)
- [SSH 功能](#ssh-功能)
  - [操作场景](#操作场景)
  - [操作系统](#操作系统)
  - [鉴权方式](#鉴权方式)
  - [前提条件](#前提条件)
  - [SSH 口令登录](#ssh-口令登录)
    - [操作步骤](#操作步骤)
    - [问题处理](#问题处理)
  - [SSH 公钥登录](#ssh-公钥登录)
  - [SSH 常用软件](#ssh-常用软件)
  - [SSH 端口转发](#ssh-端口转发)
    - [端口转发的参数](#端口转发的参数)
    - [本地转发](#本地转发)
    - [远程转发](#远程转发)
    - [动态端口转发](#动态端口转发)
  - [SSH 远程操作](#ssh-远程操作)
  - [利用远程转发实现代理功能](#利用远程转发实现代理功能)
- [内网、外网穿透](#内网外网穿透)
  - [cpolar 内网穿透服务](#cpolar-内网穿透服务)
</p></details><p></p>

# 安装 SSH 服务(Debina)

## 安装 SSH

```bash
# 更新软件源
$ sudo apt-get update

# 安装 SSH
$ sudo apt-get install ssh

# 修改配置文件
$ vim /etc/ssh/sshd_config
PasswordAuthentication yes
PermitRootLogin yes

# 重启 SSH 服务
$ /etc/init.d/ssh restart
```

此时我们就可以使用 SSH 对服务器进行远程连接, 但也不排除无法连接的情况

## SSH 无法连接的场景排查

### 检查网络

很多情况下, 无法使用 SSH 进行连接都是因为网络或者 iptables 限制导致的, 
所以可以通过 `ping` 命令来检查网络的连通性

当然, 有时候也会遇到防火墙拦截了 ICMP 和 SSH, 
所以检查网络最好是在需要 SSH 连接的服务下进行

```bash
$ ping xxx.xxx.xxx.xxx
```

### 检查端口

SSH 使用的是 22 端口, 所以可以使用 `telnet` 命令检测该端口

```bash
$ telnet xxx.xxx.xxx.xxx 22
```

* 如果正常能通, 一般情况下 SSH 服务器也就没有问题了
* 如果不能通, 检查 iptables；因为为了服务器的安全性, 
  经常会通过 iptables 去限制特定的网段才允许连接服务器

### 检查服务

如果网络能通, 且端口未被屏蔽, 依旧无法使用 SSH 连接到服务器, 
那么检查 SSH 服务, 以及配置文件是否正确配置

```bash
$ /etc/init.d/ssh status
```

# 什么是 SSH

![ssh](images/Linux_ssh2.png)

SSH 是 Secure Shell 的缩写, 也叫安全外壳协议, 是一种网络协议, 
用于计算机之间的安全远程加密登录. 

最早的时候, 互联网通信都是明文通信, 一旦被截获, 内容就暴露无疑. 

1995年, 芬兰学者 Tatu YIonen 设计了 SSH 协议, 
将登录信息全部加密, 成为互联网安全的一个基本解决方案, 迅速在全世界获得推广, 
目前已经成为 Linux 系统的标准配置.

# SSH 工作原理

![ssh](images/Linux_ssh1.png)

SSH 的安全性比较好, 其对数据进行加密的方式主要有两种: 

- 对称加密(密钥加密, Symmetrical encryption)
    - 对称加密指加密、解密使用的是同一套密钥
    - Client 端把密钥加密后发送给 Server 端, Server 用同一套密钥解密
    - 对称加密的加密强度比较高, 很难破解. 但是, Client 数量庞大, 
      很难保证密钥不泄露, 如果有一个 Client 端的密钥泄露, 
      那么整个系统的安全性就存在严重的漏洞. 为了解决对称加密的漏洞, 
      于是就产生了非对称加密
- 非对称加密(公钥加密, Asymmetrical encryption)
    - 非对称加密有两个密钥: 公钥和私钥, 公钥加密后的密文, 只能通过对应私钥进行解密. 
      想从公钥解密出私钥几乎不可能, 所以非对称加密的安全性比较高. 
    - Hasing 加密: SSH 的加密原理中, 使用了 RSA 非对称加密算法, 整个过程是这样的: 
        1. 远程主机收到用户的登录请求, 把自己的公钥发送给用户
        2. 用户使用这个公钥, 将登录密码加密后, 发送给远程主机
        3. 远程主机用自己的私钥解密登录密码, 如果密码正确, 就同意用户登录

# SSH 存在的问题--中间人攻击

SSH 之所以能够保证安全, 原因在于它采用了公钥加密, 这个过程本身是安全的, 但是实际用的时候存在一个风险: 
如果有人截获了登录请求, 然后冒充远程主机, 将伪造的公钥发送给用户, 那么用户很难辨别真伪. 
因为不像 HTTPS 协议, SSH 协议的公钥是没有证书中心(CA)公证的, 也就是说, 都是自己签发的. 

可以设想, 如果攻击者插在用户与远程主机之间(比如公共的 WIFI 区域), 用伪造的公钥获取用户的登录密码. 
再用这个密码登录远程主机, 那么 SSH 的安全机制就荡然无存了. 
这种风险就是著名的 "中间人攻击(Man-in-the-middle attack)". 

# SSH 功能

## 操作场景

- 如何在 Linux 或者 Mac OS 系统的本地电脑中通过 SSH 登录远程 Linux 实例

## 操作系统

* Linux 或 Mac OS

## 鉴权方式

* 密码
* 密钥

## 前提条件

- 已获取登录实例的管理员账号及密码(或密钥)
    - 使用系统默认密码登录
    - 使用密钥登录，需完成密钥的创建，并已将密钥绑定至该云服务器中
- 云服务器实例已购买公网 IP，且该实例已开通云服务器实例的 22 号端口(对于通过快速配置购买的云服务器实例已默认开通)

## SSH 口令登录

### 操作步骤

1. 执行以下命令，连接 Linux 云服务器。

```bash
# 如果没有修改默认端口可以省略端口参数 `-p 22`
ssh -p 22 <username>@<hostname or IP address>
```

- username 即为前提条件中获得的默认账号
- hostname or IP address 为 Linux 实例公网 IP 或自定义域名

2. 输入已经获取的密码，按 Enter，即可完成登录。

### 问题处理

如果是第一次登录远程机, 会出现以下提示: 

```bash
$ ssh user@host
# The authenticity of host 'host (12.18.429.21)' can't be established.
# RSA key fingerprint is 98:2e:d7:e0:de:9f:ac:67:28:c2:42:2d:37:16:58:4d.
# Are you sure you want to continue connecting (yes/no)? 
```

当用户发送登录请求 `ssh user@host` 给远程主机后, 远程主机发送自己的公钥给用户. 
因为公钥长度较长(采用 RSA 算法, 长达 1024 位), 很难比对, 所以对其进行 MD5 计算, 
将它变成一个 128 位的指纹. 如: 98:2e:d7:e0:de:9f:ac:67:28:c2:42:2d:37:16:58:4d, 
这样比对就容易多了. 

经过对比后, 如果用户接受这个远程主机的公钥, 系统会出现一句提示语, 表示 host 主机已经得到认可, 
然后输入登录密码就可以登录了: 

```bash
Warning: Permanently added 'host,12.18.429.21' (RSA) to the list of known hosts.
```

当远程主机的公钥被接受以后, 它就会被保存在文件 `~/.ssh/known_hosts` 之中. 
下次再连接这台主机, 系统就会认出它的公钥已经保存在本地了, 从而跳过警告部分, 
直接提示输入密码. 每个 SSH 用户都有自己的 `known_hosts` 文件, 此外系统也有一个这样的文件, 
一般是 `/etc/ssh/ssh_known_hosts`, 保存一些对所有用户都可信赖的远程主机的公钥. 

## SSH 公钥登录

使用密码登录, 每次都必须输入密码, 非常麻烦. 好在 SSH 还提供了公钥登录, 可以省去输入密码的步骤. 
所谓"公钥登录", 原理很简单, 就是用户将自己的公钥储存在远程主机上. 登录的时候, 
远程主机会向用户发送一段随机字符串, 用户用自己的私钥加密后, 再发回来. 
远程主机用事先储存的公钥进行解密, 如果成功, 就证明用户是可信的, 直接允许登录 shell, 不再要求密码. 
这种方法要求用户必须提供自己的公钥. 

如果没有现成的, 可以直接用 `ssh-keygen` 生成一个。
下面创建本地 SSH 密钥，如果生成密钥匙不输入密码，则可以实现免密登录

```bash
$ ssh-keygen
ssh-keygen -t rsa -b 4096 -C "email@example.com"
```

运行上面的命令以后, 系统会出现一系列提示, 可以一路回车. 其中有一个问题是, 
要不要对私钥设置口令(passphrase), 如果担心私钥的安全, 这里可以设置一个. 
运行结束以后, 在 `~/.ssh/` 目录下, 会新生成两个文件: `id_rsa.pub` 和 `id_rsa`. 
前者是公钥, 后者是私钥. 


这时再输入下面的命令, 将本地 SSH 公钥上传到远程服务器: 

```bash
$ ssh-copy-id user@host
# or
$ scp ~/.ssh/id_rsa.pub <username>@<hostname or IP address>:/home/username/.ssh/
```

远程主机将用户的公钥保存在登录后的用户主目录的 `~/.ssh/authorized_keys` 服务的默认验证文件中. 
这样, 以后就登录远程主机不需要输入密码了. 

```bash
cd ~/.ssh
cat /home/username/.ssh/id_rsa.pub >> authorized_keys
```

如果还是不行, 就用 vim 打开远程主机的 `/etc/ssh/sshd_config` 这个文件, 将以下几行的注释去掉. 

```bash
RSAAuthentication yes 
PubkeyAuthentication yes 
AuthorizedKeysFile 
.ssh/authorized_keys
```

然后, 重启远程主机的ssh服务: 

```bash
# Redhat6系统
$ service ssh restart

# Redhat7系统
$ systemctl restart sshd

# ubuntu系统
$ service ssh restart

# debian系统
$ /etc/init.d/ssh restart
```

配置本地 ssh config 文件

```bash
vim ~/.ssh/config
Host cssor_server                       # 别名，域名缩写
    HostName cssor.com                  # <hostname or IP address>
    User cssor                          # <username>
    PreferredAuthentications publickey  # 有些情况或许需要加入此句，优先验证类型ssh
    IdentityFile ~/.ssh/id_rsa          # 本地私钥文件的路径
```

最后, 登录远程主机: 

* 远程主机

```bash
$ ifconfig                 # 查看服务器 IP 地址
$ netstat -ntlp | grep ssh # 查看服务器是否启用了 SSH
```

* 本地用户

```bash
$ ssh -p 22 user@host
$ ssh user@host            # 默认端口为 22, 当端口为 22时, 可以省略端口
$ ssh host                 # 本地使用的用户名与远程登录的用户名一致, 登录的用户名可以省略
```

退出远程服务器:

```bash
$ exit                     # 退出服务器， 或者 `Control D`
```

## SSH 常用软件

- FinalShell

下载

```bash
# 下载脚本：
curl -o finalshell_install.sh www.hostbuf.com/downloads/finalshell_install.sh
# 授权+执行：
chmod +x finalshell_install.sh
sudo ./finalshell_install.sh
```

- Cyberduck
- SSH Shell
- Termius

## SSH 端口转发

SSH 不仅能够自动加密和解密 SSH 客户端与服务端之间的网络数据, 
同时, SSH 还能够提供一个非常有用的功能, 端口转发, 
即将 TCP 端口的网络数据转发到指定的主机的某个端口上. 
在转发的同时会对数据进行相应的加密和解密. 如果工作环境中防火墙限制了一些网络端口的使用, 
但是允许 SSH 的连接, 那么也是能够使用 SSH 转发后的端口进行通信. 

SSH 端口转发有三种: 动态端口转发、本地端口转发、远程端口转发

举例说明: 假设有三台主机: host1、host2、host3

- 动态端口转发是找一个代理端口, 然后通过代理端口去连相应的端口. 
  动态端口转发的好处在于通过代理端口可以去找很多需要连接的端口, 
  提高了工作效率. 比如 host1 本来是连不上 host2 的, 而 host3 却可以连上 host2. 
  host1 可以找到 host3 作代理, 然后通过 host3 去连接 host2 的相应端口
- 本地端口转发也是找到第三方, 通过第三方再连接想要连接的端口, 
  但这种方式的端口转发是固定的, 是点对点的. 比如假定 host1 是本地主机, 
  host2 是远程主机. 由于种种原因, 这两台主机之间无法连通. 
  但是, 另外还有一台 host3, 可以同时连上 host1 和 host2 这两台主机. 
  通过 host3, 将 host1 连上 host2. host1 找到 host3, 
  host1 和 host3 之间就像有一条数据传输的道路, 通常被称为 "SSH 隧道", 
  通过这条隧道 host1 就可以连上 host2
- 远程端口转发和本地端口转发就是反过来了. 假如 host1 在外网, host2 在内网, 
  正常情况下, host1 不能访问 host2. 通过远程端口转发, host2 可以反过来访问 host1. 
  host2 和 host1 之间形成了一条道路, host1 就可以通过这条道路去访问 host2

### 端口转发的参数

- `-c`: 压缩数据
- `-f`: 后台认证用户/密码, 通常和 `-N` 连用, 不用登录到远程主机
- `-N`: 不执行脚本或命令, 通常与 `-f` 连用
- `-g`: 在 `-L`/`-R`/`-D` 参数中, 允许远程主机连接到建立的转发的端口上, 如果不加这个参数, 只允许本地主机建立连接
- `-L`: 本地网卡地址:本地端口:目标IP:目标端口
- `-D`: 动态端口转发
- `-R`: 远程端口转发
- `-T`: 不分配 TTY 只做代理用
- `-q`: 安静模式, 不输出错误/警告信息

### 本地转发

有本地网络服务器的某个端口, 转发到远程服务器的某个端口. 说白了就是, 将发送到本地端口的请求, 
转发到目标端口, 格式如下: 

```bash
$ ssh -L 本地网卡地址:本地端口:目标地址:目标端口 user@host
$ ssh -L 本地端口:目标地址:目标端口 user@host            # 本地网卡地址可以省略
$ ssh -L 本地网卡地址:本地端口:目标地址:目标端口           # 如果本地用户名与远程用户名相同, 用户名可以省略
```

举例说明: 

- 有两台机器
    - centOS A(192.168.13.139)
    - centOS B(192.168.13.142) => 安装了 MySQL
- 如果 centOS B 上设置了允许任何主机连接 MySQL, 
此时, 在 centOS A 上面是可以连上 centOS B 的 MySQL 的
- 如果设置 centOS B 限制不允许外部 IP 连接, 仅仅让 127.0.0.1 连接, 
此时 centOS A 需要通过本地端口转发的方式连接 centOS B 上的 MySQL 服务, 
即将 centOS A 本地的某个端口, 映射到 centOS B 上, 然后在 centOS A 上连接
centOS B 上的 MySQL 服务

```bash
# current on centOS A
$ ssh -L 127.0.0.1:3306:127.0.0.1:3306 root@192.168.13.142
$ bin/mysql -h127.0.0.1 -uroot -p
$ lsof -i:3306 # 查看 SSH 转发监听的进程
$ netstat -apn |grep 3306
$ ps -ef |grep ssh
```

### 远程转发

远程转发是指由远程服务器的某个端口转发到本地网络服务器的某个端口. 说白了, 
就是将发送到远程端口的请求转发到目标端口, 格式如下: 

```bash
$ ssh -R 远程网卡地址:远程端口:目标地址:目标端口
```

举例说明: 

- 有三台机器
    - centOS A(192.168.13.139)
    - centOS B(192.168.13.142)
    - win7(10.18.78.135)

- 假设 win7 与 centOS B 不能直接连接, 但是 win7 与 centOS A 可以连接, 
centOS B 也可以与 centOS A 连接, 那么, 就可以在 centOS A 上使用远程端口转发了, 
让 win7 与 centOS B 进行通信, 即 centOS B 监听自己的 80 端口, 
然后将所有数据由 centOS A 发给 win7. 

```bash
# current on centOS A
$ ssh -R 127.0.0.1:80:10.18.78.135:80 root@192.168.13.142
```

### 动态端口转发

对于 SSH 的本地转发和远程转发, 都需要将本地端口和远程端口一一绑定, 格式如下: 

```bash
$ ssh -D [本地地址:]本地端口 user@host
```

## SSH 远程操作

SSH 远程操作, 主要用于在远程服务器上面执行某个操作, 格式如下: 

```bash
$ ssh user@host 'command'
```

示例: 

```bash
user_a $ ssh user_b@host_b 'uname -a' # 在 user_a 上查看 user_b 的操作系统
user_a $ tar -cz test | ssh user_b@host_b 'tar -xz'
user_a $ ssh user_b@host_b 'netstat -tln |grep 1080' # 在 user_a 上查看 user_b 是否监听了 1080 端口
```

## 利用远程转发实现代理功能

目前 B 机器, 只能在自己 127.0.0.1 的 80 端口监听并转发, 如何让 B 机器作为代理, 
转发其他机器的请求到 A 机器上面呢？比如, 现在有一台机器 C(192.168.13.143), 
C 不能访问 A, 但是能够访问 B. 如何让 C 利用 B 来访问 A 呢？

此时, 需要将 B 的监听, 由 127.0.0.1:8081, 改为 0:0.0.0:8081, 修改 
`sshd` 的配置 `/etc/ssh/sshd_config`:

```bash
vim /etc/ssh/sshd_config
#如果有
GatewayPorts no
#改为
GatewayPorts yes
#没有, 添加即可
```

然后重启 `sshd`

```bash
$ sudo service sshd restart
```

然后重新, 设置动态转发, 如下: 

```bash
$ ssh -f -g  -N -R 8081:127.0.0.1:80 dequan@192.168.13.149
```

可以看到, 此时 B 机器, 已经监听了 0:0.0.0:8081

在C机器上面, 我们通过curl模拟请求, 利用B机器做代理, 如下: 

```bash
$ curl -x 192.168.13.149:8081 127.0.0.1
```

当然, 如果还有其他机器, 也可以使用类似的方式, 来请求A机器. 

# 内网、外网穿透

## cpolar 内网穿透服务

```bash
#软件安装
curl -L https://www.cpolar.com/static/downloads/install-release-cpolar.sh | sudo bash

#查看版本
cpolar version

#添加个人账户token认证
cpolar authtoken xxxxxxx

#开启穿透http端口服务
cpolar http 8080
#指定二级域名 -subdomain=YOUR_SUBDOMAIN 

#开启穿透ssh端口服务
cpolar tcp 22
#指定二级域名 -remote-addr=1.tcp.vip.cpolar.cn:xxxxx

#向系统添加服务
sudo systemctl enable cpolar

#启动服务
sudo systemctl start cpolar

#查看状态
sudo systemctl status cpolar
```

