---
title: Java 环境配置
author: 王哲峰
date: '2020-10-01'
slug: java-env
categories:
  - java
tags:
  - tool
---

Java 程序必须运行在 JVM 上，需要安装 JDK

# 安装 JDK, Idea

- JDK
- Idea

# 设置环境变量

安装完 JDK 后，需要设置一个 `JAVA_HOME` 的环境变量，它指向 JDK 的安装目录: 

```bash
# ~/.zshrc(~/.zprofile)
export JAVA_HOME=`/usr/libexec/java_home -v 15`
```

把 `JAVA_HOME` 的 `bin` 目录附加到系统环境变量 `PATH` 上。
把 `JAVA_HOME` 的 `bin` 目录添加到 `PATH` 中是为了在任意文件夹下都可以运行 Java:

```bash
# ~/.zshrc(~/.zprofile)
export PATH=$JAVA_HOME/bin:$PATH
```

参考：`如何配置 Java 环境变量 <https://www.java.com/zh_CN/download/help/path.xml>`_ 

# 验证安装

```bash
java -version
```


# JDK

在JAVA_HOME的bin目录下找到很多可执行文件：

- `java`：这个可执行程序其实就是 JVM，运行 Java 程序，就是启动 JVM，然后让 JVM 执行指定的编译后的代码；
- `javac`：这是 Java 的编译器，它用于把 Java 源码文件（以.java后缀结尾）编译为 Java 字节码文件（以.class后缀结尾）；
- `jar`：用于把一组 .clas s文件打包成一个 .jar 文件，便于发布；
- `javadoc`：用于从 Java 源码中自动提取注释并生成文档；
- `jdb`：Java 调试器，用于开发阶段的运行调试。
