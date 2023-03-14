---
title: Maven
author: 王哲峰
date: '2020-10-01'
slug: java-maven
categories:
  - java
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
</style>

<details><summary>目录</summary><p>

- [Maven 介绍](#maven-介绍)
  - [一个 Java 项目需要的东西](#一个-java-项目需要的东西)
  - [Maven 作用](#maven-作用)
  - [Maven 项目结构](#maven-项目结构)
    - [pom.xml 项目描述文件](#pomxml-项目描述文件)
  - [Maven 安装](#maven-安装)
- [Maven 使用](#maven-使用)
  - [依赖管理](#依赖管理)
- [test](#test)
  - [3.1](#31)
  - [3.2](#32)
  - [3.3](#33)
</p></details><p></p>

# Maven 介绍

Maven 是一个 Java 项目的管理和构建工具：

- Maven 使用 pom.xml 定义项目内容，并使用预设的目录结构
- 在 Maven 中声明一个依赖项可以自动下载并导入 classpath
- Maven 使用 groupId，artifactId 和 version 唯一定位一个依赖


## 一个 Java 项目需要的东西

- 依赖包
    - 将依赖包的 jar 包放入 classpath
- 项目的目录结构
    - `src` 目录存放 Java 源码
    - `resources` 目录存放配置文件
    - `bin` 目录存放编译生成的 `.class 文件`
- 配置环境
    - JDK 的版本
    - 编译打包的流程
    - 当前代码的版本号
- 除了使用 IDE 进行编译，还必须能通过命令行工具进行编译，才能够让项目在一个独立的服务器上编译、测试、部署


> 这些工作难度不大，但是非常琐碎且耗时。如果每一个项目都自己搞一套配置，肯定会一团糟。我们需要的是一个标准化的Java项目管理和构建工具。

## Maven 作用

Maven就是是专门为Java项目打造的管理和构建工具，它的主要功能有：

- 提供了一套标准化的项目结构；
- 提供了一套标准化的构建流程（编译，测试，打包，发布……）；
- 提供了一套依赖管理机制。


## Maven 项目结构

一个使用Maven管理的普通的Java项目，它的目录结构默认如下：

``` 
a-maven-project
├── pom.xml
├── src
│   ├── main
│   │   ├── java
│   │   └── resources
│   └── test
│       ├── java
│       └── resources
└── target
```

其中：

- `a-maven-project` 是项目的根目录，也是项目名
- `pox.xml` 项目描述文件
- `src/main/java` 存放 Java 源码的目录
- `src/main/resources` 存放资源文件的目录
- `src/test/java` 存放测试源码的目录
- `src/test/resources` 存放测试资源的目录
- `target` 存放所有编译、打包生成的文件


> 所有的目录结构都是约定好的标准结构，我们千万不要随意修改目录结构。使用标准结构不需要做任何配置，Maven就可以正常使用。


### pom.xml 项目描述文件

使用 `<dependency>` 声明一个依赖后，Maven就会自动下载这个依赖包并把它放到 `classpath` 中。

```xml
<project ...>
    <modelVersion>4.0.0</modelVersion>
    <groupId>com.itranswarp.learnjava</groupId>
    <artifactId>hello</artifactId>
    <version>1.0</version>
    <packaging>jar</packaging>

    <properties>
        ...
    </properties>

    <dependencies>
        <dependency>
            <groupId>commons-logging</groupId>
            <artifactId>commons-logging</artifactId>
            <version>1.2</version>
        </dependency>

    </dependencies>
</project>
```

- `groupId` 类似于 Java 的包名，通常是公司或组织名称
- `artifactId` 类似于 Java 的类名，通常是项目名称，再加上version
- 一个 Maven 工程就是由 `groupId`，`artifactId` 和 `version` 作为唯一标识。
  在引用其他第三方库的时候，也是通过这3个变量确定。


## Maven 安装

要安装 Maven，可以从 Maven 官网下载最新的 Maven 3.6.x，然后在本地解压，设置几个环境变量：

```bash
$ M2_HOME=/path/to/maven-3.6.x
$ PATH=$PATH:$M2_HOME/bin
```

Windows 可以把 `%M2_HOME%\bin` 添加到系统 Path 变量中。

然后，打开命令行窗口，输入 `mvn -version`，应该看到 Maven 的版本信息：

```bash
$ mvn -version
```

# Maven 使用

## 依赖管理

# test

## 3.1

```bash
cd ~/project/java_proj/
mvn archetype:generate -DgroupId=com.mycompany.helloworld -DartifactId=helloworld -Dpackage=com.mycompany.helloworld -Dversion=1.0-SNAPSHOT
```

- `archetype:`
- `-DgroupId`
- `-DartifactId`
- `-Dpackage`
- `-Dversion=`

```
.
├── pom.xml
└── src
   ├── main
   │   └── java
   │       └── com
   │           └── mycompany
   │               └── helloworld
   │                   └── App.java
   └── test
       └── java
           └── com
               └── mycompany
                   └── helloworld
                       └── AppTest.java

11 directories, 3 files
```

## 3.2 

```bash
cd helloworld
mvn package
```

- Linux: `~/.m2/repository/`
- Win7: `%USER_HOME%\.m2\repository`

```
 .
 ├── pom.xml
 ├── src
 │   ├── main
 │   │   └── java
 │   │       └── com
 │   │           └── mycompany
 │   │               └── helloworld
 │   │                   └── App.java
 │   └── test
 │       └── java
 │           └── com
 │               └── mycompany
 │                   └── helloworld
 │                       └── AppTest.java
 └── target
     ├── classes
     │   └── com
     │       └── mycompany
     │           └── helloworld
     │               └── App.class
     ├── helloworld-1.0-SNAPSHOT.jar
     ├── maven-archiver
     │   └── pom.properties
     ├── maven-status
     │   └── maven-compiler-plugin
     │       ├── compile
     │       │   └── default-compile
     │       │       ├── createdFiles.lst
     │       │       └── inputFiles.lst
     │       └── testCompile
     │           └── default-testCompile
     │               ├── createdFiles.lst
     │               └── inputFiles.lst
     ├── surefire-reports
     │   ├── com.mycompany.helloworld.AppTest.txt
     │   └── TEST-com.mycompany.helloworld.AppTest.xml
     └── test-classes
         └── com
             └── mycompany
                 └── helloworld
                     └── AppTest.class

 28 directories, 13 files
```

## 3.3 


```bash
$ java -cp target/helloworld-1.0-SNAPSHOT.jar com.mycompany.helloworld.App
```

