---
title: Java Web 开发
author: 王哲峰
date: '2020-10-01'
slug: java-web
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
img {
    pointer-events: none;
}
</style>

<details><summary>目录</summary><p>

- [什么是 JavaEE](#什么是-javaee)
- [Web 基础](#web-基础)
  - [HTTP 协议](#http-协议)
  - [HTTP 编程](#http-编程)
- [Web 编程](#web-编程)
  - [编写 HTTP Server](#编写-http-server)
- [Servlet 开发](#servlet-开发)
  - [为什么使用 Servlet 开发](#为什么使用-servlet-开发)
  - [最简单的 Servlet](#最简单的-servlet)
  - [Servlet API](#servlet-api)
- [JSP 开发](#jsp-开发)
- [MVC](#mvc)
- [Filter](#filter)
- [Listener](#listener)
- [Deploy](#deploy)
</p></details><p></p>

# 什么是 JavaEE

- JavaEE 是Java Platform Enterprise Edition 的缩写，即 Java 企业平台
    - 基于标准 JDK 的开发都是 JavaSE，即 Java Platform Standard Edition
    - 此外，还有一个小众不太常用的 JavaME：Java Platform Micro Edition，
      是 Java 移动开发平台(非Android)
    - 它们三者关系如下：

```
┌────────────────┐
│     JavaEE     │
│┌──────────────┐│
││    JavaSE    ││
││┌────────────┐││
│││   JavaME   │││
││└────────────┘││
│└──────────────┘│
└────────────────┘
```

- JavaEE 并不是一个软件产品，它更多的是一种软件架构和设计思想。我们可以把 JavaEE 看作是在 
  JavaSE 的基础上，开发的一系列 **基于服务器的组件**、**API 标准** 和 **通用架构**
- JavaEE 最核心的组件就是 **基于 Servlet 标准的 Web 服务器**，开发者编写的应用程序是基于 
  Servlet API 并运行在 Web 服务器内部的：

```
┌─────────────┐
│┌───────────┐│
││ User App  ││
│├───────────┤│
││Servlet API││
│└───────────┘│
│ Web Server  │
├─────────────┤
│   JavaSE    │
└─────────────┘
```

- 此外，JavaEE还有一系列技术标准：
    - EJB：Enterprise JavaBean，企业级JavaBean，早期经常用于实现应用程序的业务逻辑，现在基本被轻量级框架如Spring所取代；
    - JAAS：Java Authentication and Authorization Service，一个标准的认证和授权服务，常用于企业内部，Web程序通常使用更轻量级的自定义认证；
    - JCA：JavaEE Connector Architecture，用于连接企业内部的EIS系统等；
    - JMS：Java Message Service，用于消息服务；
    - JTA：Java Transaction API，用于分布式事务；
    - JAX-WS：Java API for XML Web Services，用于构建基于XML的Web服务；
    - ...


> - JavaEE 也不是凭空冒出来的，它实际上是完全基于 JavaSE，只是多了一大堆服务器相关的库以及 API 接口。
  所有的 JavaEE 程序，仍然是运行在标准的 JavaSE 的虚拟机上的。
> - 目前流行的基于 Spring 的轻量级 JavaEE 开发架构，使用最广泛的是 Servlet 和 JMS，
  以及一系列开源组件。


# Web 基础

## HTTP 协议


- HTTP 目前有多个版本，1.0是早期版本，浏览器每次建立TCP连接后，只发送一个HTTP请求并接收一个HTTP响应，然后就关闭TCP连接。由于创建TCP连接本身就需要消耗一定的时间，因此，HTTP 1.1允许浏览器和服务器在同一个TCP连接上反复发送、接收多个HTTP请求和响应，这样就大大提高了传输效率。
- HTTP 协议是一个请求-响应协议，它总是发送一个请求，然后接收一个响应。能不能一次性发送多个请求，然后再接收多个响应呢？HTTP 2.0可以支持浏览器同时发出多个请求，但每个请求需要唯一标识，服务器可以不按请求的顺序返回多个响应，由浏览器自己把收到的响应和请求对应起来。可见，HTTP 2.0进一步提高了传输效率，因为浏览器发出一个请求后，不必等待响应，就可以继续发下一个请求。
- HTTP 3.0为了进一步提高速度，将抛弃TCP协议，改为使用无需创建连接的UDP协议，目前HTTP 3.0仍然处于实验阶段。

## HTTP 编程

# Web 编程

- HTTP 编程是以客户端的身份去请求服务器资源
- 以服务器的身份响应客户端请求，编写服务器程序来处理客户端请求通常就称之为 Web 开发

## 编写 HTTP Server

一个 HTTP Server 本质上是一个 TCP 服务器, 我们先用 TCP 编程的多线程实现的服务器端框架:

```java
public class Server {
    public static void main(String[] args) throws IOException {
        ServerSocket ss = new ServerSocket(8080); // 监听指定端口
        System.out.println("server is running...");
        for (;;) {
            Socket sock = ss.accept();
            System.out.println("connected from " + sock.getRemoteSocketAddress());
            Thread t = new Handler(sock);
            t.start();
        }
    }
}

class Handler extends Thread {
    Socket sock;

    public Handler(Socket sock) {
        this.sock = sock;
    }

    public void run() {
        try (InputStream input = this.sock.getInputStream()) {
            try (OutputStream output = this.sock.getOutputStream()) {
                handle(input, output);
            }
        } catch (Exception e) {
            try {
                this.sock.close();
            } catch (IOException ioe) {
            }
            System.out.println("client disconnected.");
        }
    }

    private void handle(InputStream input, OutputStream output) throws IOException {
        var reader = new BufferedReader(new InputStreamReader(input, StandardCharsets.UTF_8));
        var writer = new BufferedWriter(new OutputStreamWriter(output, StandardCharsets.UTF_8));
        // TODO: 处理HTTP请求
    }
}
```


```java
private void handle(InputStream input, OutputStream output) throws IOException {
    System.out.println("Process new http request...");
    var reader = new BufferedReader(new InputStreamReader(input, StandardCharsets.UTF_8));
    var writer = new BufferedWriter(new OutputStreamWriter(output, StandardCharsets.UTF_8));
    // 读取HTTP请求:
    boolean requestOk = false;
    String first = reader.readLine();
    if (first.startsWith("GET / HTTP/1.")) {
        requestOk = true;
    }
    for (;;) {
        String header = reader.readLine();
        if (header.isEmpty()) { // 读取到空行时, HTTP Header读取完毕
            break;
        }
        System.out.println(header);
    }
    System.out.println(requestOk ? "Response OK" : "Response Error");
    if (!requestOk) {
        // 发送错误响应:
        writer.write("HTTP/1.0 404 Not Found\r\n");
        writer.write("Content-Length: 0\r\n");
        writer.write("\r\n");
        writer.flush();
    } else {
        // 发送成功响应:
        String data = "<html><body><h1>Hello, world!</h1></body></html>";
        int length = data.getBytes(StandardCharsets.UTF_8).length;
        writer.write("HTTP/1.0 200 OK\r\n");
        writer.write("Connection: close\r\n");
        writer.write("Content-Type: text/html\r\n");
        writer.write("Content-Length: " + length + "\r\n");
        writer.write("\r\n"); // 空行标识Header和Body的分隔
        writer.write(data);
        writer.flush();
    }
}
```

# Servlet 开发

## 为什么使用 Servlet 开发

为了高效而可靠地开发 Web 应用，在 JavaEE 平台上，处理 TCP 连接，解析 HTTP 协议这些底层工作统统
扔给现成的 Web 服务器去做，我们只需要把自己的应用程序跑在 Web 服务器上。

为了实现这一目的，JavaEE 提供了 Servlet API，我们使用 Servlet API 编写自己的 Servlet 来处理
HTTP 请求，Web 服务器实现 Servlet API 接口，实现底层功能：

``` 
                 ┌───────────┐
                 │My Servlet │
                 ├───────────┤
                 │Servlet API│
┌───────┐  HTTP  ├───────────┤
│Browser│<──────>│Web Server │
└───────┘        └───────────┘
```

## 最简单的 Servlet

```java
import java.WebServlet;
import java.HttpServlet;

// WebServlet 注解表示这是一个 Servlet,并映射到地址 "/:"

@WebServlet(urlPatterns = "/")
public class HelloServlet extends HttpServlet {

    // 覆写 doGet() 方法
    protected void doGet(HttpServletRequest req, HttpServletResponse resp) throws ServletException, IOException {
        //设置响应类型
        resp.setContentType("text/html");
        // 获取输出流
        PrintWriter pw = resp.getWriter();
        // 写入响应
        pw.write("<h1>Hello, world!</h1>");
        // 最后不要忘记 flush 强制输出
        pw.flush();
    }

    // 覆写 doPost() 方法
    protected void doPost(HttpServletRequest req, HttpServletResponse resp) throws ServletException, IOException {
        ...
    }

}
```


> - 一个 Servlet 总是继承自 `HttpServlet`，然后覆写 `doGet()` 或 `doPost()` 方法。
注意到 `doGet()` 方法传入了 `HttpServletRequest` 和 `HttpServletResponse` 两个对象，
分别代表 HTTP 请求和响应。
> - 我们使用 Servlet API 时，并不直接与底层 TCP 交互，也不需要解析 HTTP 协议，因为 `HttpServletRequest` 
和 `HttpServletResponse` 就已经封装好了请求和响应。以发送响应为例，我们只需要设置正确的响应类型，
然后获取 `PrintWriter`，写入响应即可。

## Servlet API

Servlet API 是一个 jar 包，需要通过 Maven 来引入才能正常编译。

- 编写 `pox.xml` 文件如下：

``` xml
<project xmlns="http://maven.apache.org/POM/4.0.0"
    xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance"
    xsi:schemaLocation="http://maven.apache.org/POM/4.0.0 http://maven.apache.org/maven-v4_0_0.xsd">
    <modelVersion>4.0.0</modelVersion>
    <groupId>com.itranswarp.learnjava</groupId>
    <artifactId>web-servlet-hello</artifactId>
    # ----------------------------
    # Java Web Application Archive
    # ----------------------------
    <packaging>war</packaging>
    <version>1.0-SNAPSHOT</version>

    # ----------------------------
    # properties
    # ----------------------------
    <properties>
        <project.build.sourceEncoding>UTF-8</project.build.sourceEncoding>
        <project.reporting.outputEncoding>UTF-8</project.reporting.outputEncoding>
        <maven.compiler.source>11</maven.compiler.source>
        <maven.compiler.target>11</maven.compiler.target>
        <java.version>11</java.version>
    </properties>

    # ----------------------------
    # Servlet API dependencies
    # ----------------------------
    <dependencies>
        <dependency>
            <groupId>javax.servlet</groupId>
            <artifactId>javax.servlet-api</artifactId>
            <version>4.0.0</version>
            <scope>provided</scope>                     # provided 表示编译时使用
        </dependency>
    </dependencies>

    # ----------------------------
    # build
    # ----------------------------
    <build>
        <finalName>hello</finalName>
    </build>
</project>
```

- 创建一个 `web.xml` 描述文件

``` 
<!DOCTYPE web-app PUBLIC "-//Sun Microsystems, Inc.//DTD Web Application 2.3//EN" "http://java.sun.com/dtd/web-app_2_3.dtd">
<web-app>
    <display-name>Archetype Created Web Application</display-name>
</web-app>
```

- 过程结构

``` 
web-servlet-hello
├── pom.xml
└── src
    └── main
        ├── java
        │   └── com
        │       └── itranswarp
        │           └── learnjava
        │               └── servlet
        │                   └── HelloServlet.java
        ├── resources
        └── webapp
            └── WEB-INF
                └── web.xml
```

```bash
// Maven 打包, 在 target 目录下得到一个 hello.war 文件, 这个文件就是编译打包后的 Web 应用程序
$ mvn clean package
```

# JSP 开发

# MVC 

# Filter

# Listener

# Deploy

