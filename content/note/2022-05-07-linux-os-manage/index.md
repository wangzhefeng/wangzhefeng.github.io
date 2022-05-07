---
title: linux-os-manage
author: 王哲峰
date: '2022-05-07'
slug: linux-os-manage
categories:
  - Linux
tags:
  - tool
---


Linux 系统管理

- 网络管理
- 软件包安装
- 内存管理
- 进程管理
- 磁盘管理

# 网络管理

- 网络状态查看
- 网络配置
- 路由命令
- 网络故障排除
- 网络服务管理
- 常用网络配置文件

## 网络状态查看

- 网络状态查看工具:
    - **net-tools** (CentOS 7以前)
        - `ifconfig`
        - `route`
        - `netstat`
    - **iproute2** (CentOS 7以后)
        - `ip`
        - `ss`
- 网络状态查看命令
    - 查看网络接口(网卡信息)
        - `ifconfig`
        - `eth0` 第一块网卡(网络接口)
        - 第一个网络接口可能叫做下面的名字
            - `eno1` 板载网卡
            - `ens33` PCI-E 网卡
            - `enp0s3` 无法获取物理信息的 PCI-E 网卡
            - CentOS 7 使用了一致性网络设备命名, 以上都不匹配则使用 `eth0`    
    - 查看网络情况
        - 查看网卡连接状态
        - `mii-tool eth0`    
    - 查看网关(路由)命令
        - `route -n`
        - 使用 `-n` 参数不解析主机名, 加快速度
- 网络接口命名修改
    - 将网络接口名字转换为原始的 `eth0`, 方便对服务器网络接口进行批量操作
    - 网卡命名规则受 `biosdevname` 和 `net.ifnames` 两个参数影响
    - 编辑 `/etc/default/grub` 文件, 增加 `biosdevname=0 net.ifnames=0`
    - 更新 `grub`

```bash
$ grub2-mkconfig -o /boot/grub2/grub.cfg
```

    - 重启

```bash
$ reboot
```

    - 修改参数的显示组合
        - ![img](images/network.png)
    - 使用示例

```bash
# root 用户
$ ifconfig

# 普通用户
$ /sbin/ifconfig

# 编辑 /etc/default/grub, 增加 `biosdevname=0 net.ifnames=0`
$ vim /etc/default/grub
$ grub2-mkconfig -o /boot/grub2/grub.cfg
$ reboot
$ ifconfig
$ ifconfig eth0
```

- 网络配置命令
    - 设置网卡 IP 地址        
        - `ifconfig <接口> <IP地址>[netmask 子网掩码]`
    - 启动网卡        
        - `ifup <接口>`
    - 关闭网卡        
        - `ifdown <接口>`
- 网关配置命令
    - 添加网关
        - `route add default gw <网关ip>`
        - `route add -host <指定ip> gw <网关ip>`
        - `route add -net <指定网段> netmask <子网掩码> gw <网关ip>`
- 网络命令集合: ip命令(iproute2工具)
    - `ip addr ls`
        - `ifconfig`
    - `ip link set dev eth0 up`
        - `ifup eth0`
    - `ip addr add 10.0.0.1/24 dev eth1`
        - `ifconfig eth1 10.0.0.1 netmask 255.255.255.0`
    - `ip route add 10.0.0/24 via 192.168.0.1`        
        - `route add -net 10.0.0.0 netmask 255.255.255.0 gw 192.168.0.1`

# 软件包管理器的使用
