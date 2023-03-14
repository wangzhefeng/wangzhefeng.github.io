---
title: Spring
author: 王哲峰
date: '2020-10-01'
slug: java-spring-boot
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

- [IoC 容器](#ioc-容器)
  - [IoC 原理](#ioc-原理)
    - [通常的 Java 组件协作示例](#通常的-java-组件协作示例)
  - [IoC 方案](#ioc-方案)
  - [依赖注入方式](#依赖注入方式)
  - [无侵入容器](#无侵入容器)
  - [装配 Bean](#装配-bean)
- [使用 AOP](#使用-aop)
  - [OOP 编程](#oop-编程)
  - [AOP 编程](#aop-编程)
  - [拦截器类型](#拦截器类型)
- [访问数据库](#访问数据库)
  - [使用 JDBC](#使用-jdbc)
  - [使用声明式事务](#使用声明式事务)
- [开发 Web 应用](#开发-web-应用)
- [集成第三方组件](#集成第三方组件)
- [Spring Boot](#spring-boot)
  - [微服务阶段](#微服务阶段)
  - [环境配置](#环境配置)
    - [开发者工具](#开发者工具)
    - [打包 Spring Boot 应用](#打包-spring-boot-应用)
    - [使用 Actuator](#使用-actuator)
    - [使用 Profiles](#使用-profiles)
    - [加载配置文件](#加载配置文件)
  - [Spring Boot 应用](#spring-boot-应用)
    - [Spring Boot 应用结构](#spring-boot-应用结构)
    - [application.java](#applicationjava)
    - [logback-spring.xml](#logback-springxml)
    - [环境变量](#环境变量)
    - [Open API](#open-api)
</p></details><p></p>

Spring 是一个支持快速开发 Java EE 应用程序的框架。它提供了一系列底层容器和基础设施，
并可以和大量常用的开源框架无缝集成，可以说是开发 Java EE 应用程序的必备。

随着 Spring 越来越受欢迎，在 Spring Framework 基础上，又诞生了 Spring Boot、Spring 
Cloud、Spring Data、Spring Security 等一系列基于 Spring Framework 的项目。

Spring Framework主要包括几个模块：

- 支持 IoC 和 AOP 的容器；
- 支持 JDBC 和 ORM 的数据访问模块；
- 支持声明式事务的模块；
- 支持基于 Servlet 的 MVC 开发；
- 支持基于 Reactive 的 Web 开发；
- 以及集成 JMS、JavaMail、JMX、缓存等其他模块。

# IoC 容器

Spring 的核心就是提供了一个 IoC 容器，它可以管理所有轻量级的 JavaBean 组件，
提供的底层服务包括组件的生命周期管理、配置和组装服务、AOP支持，以及建立在 AOP 
基础上的声明式事务服务等。

Spring 提供的容器又称为 IoC 容器，什么是 IoC？IoC 全称 Inversion of Control，
直译为控制反转。

## IoC 原理

### 通常的 Java 组件协作示例

假定一个在线书店：

- 通过 `BookService` 获取书籍
    - 为了从数据库查询书籍，`BookService` 持有一个 `DataSource`(通过实例化 `HikariDataSource` 得到)
    - 为了实例化一个 `HikariDataSource`，又不得不示例化一个 `HikariConfig`
- 通过 `UserService` 获取用户
    - 为了使用户查看书籍，`UserService` 也需要访问数据库，不得不也实例化一个 `HikariDataSource`
    - 在处理用户购买的 `CartServlet` 中，需要实例化 `UserService` 和 `BookService`
- 在购买历史 `HistoryServlet` 中，也需要实例化 `UserService` 和 `BookService`

```java
// 通过 BookService 获取在线书店的书籍
public class BookService {
    private HikariConfig config = new HikariConfig();
    private DataSource dataSource = new HikariDataSource(config);

    public Book getBook(long bookId) {
        try (Connection conn = dataSource.getConnection()) {
            ...
            return book;
        }
    }
}

// 获取用户
public class UserService {
    private HikariConfig config = new HikariConfig();
    private DataSource dataSource = new HikariDataSource(config);

    public User getUser(long userId) {
        try (Connection conn = dataSource.getConnection()) {
            ...
            return user;
        }
    }
}

// 在处理用户购买的 CartServlet 中，需要实例化 UserService 和 BookService
public class CartServlet extends HttpServlet {
    private BookService bookService = new BookService();
    private UserService userService = new UserService();

    protected void doGet(HttpServletRequest req, HpptServletResponse resp) throws ServletException, IOException {
        long currentUserId = getFromCookie(req);
        User currentUser = userService.getUser(currentUserId);
        Book book = bookService.getBook(req.getParameter("bookId"));
        cartService.addToCart(currentUser, book);
        ...
    }
}

// 在购买历史 HistoryServlet 中，也需要实例化 UserService 和 bookService
public class HistoryServlet extends HttpServlet {
    private BookService bookService = new BookService();
    private UserService userService = new UserService();
}
```

> 上述每个组件都采用了一种简单的通过new创建实例并持有的方式。仔细观察，会发现以下缺点：
> 
- 实例化一个组件其实很难，例如，BookService和UserService要创建HikariDataSource，实际上需要> 读取配置，才能先实例化HikariConfig，再实例化HikariDataSource。
- 没有必要让BookService和UserService分别创建DataSource实例，完全可以共享同一个DataSource，但谁负责创建DataSource，谁负责获取其他组件已经创建的DataSource，不好处理。类似> 的，CartServlet和HistoryServlet也应当共享BookService实例和UserService实例，但也不好处理。
- 很多组件需要销毁以便释放资源，例如DataSource，但如果该组件被多个组件共享，如何确保它的使用方都> 已经全部被销毁？
- 随着更多的组件被引入，例如，书籍评论，需要共享的组件写起来会更困难，这些组件的依赖关系会越来越复> 杂。
> - 测试某个组件，例如BookService，是复杂的，因为必须要在真实的数据库环境下执行。
> 
从上面的例子可以看出，如果一个系统有大量的组件，其生命周期和相互之间的依赖关系如果由组件自身来维护，不但大大增加了系统的复杂度，而且会导致组件之间极为紧密的耦合，继而给测试和维护带来了极大的困> 难。

## IoC 方案

核心问题：

- 谁负责创建组件？
- 谁负责根据依赖关系组装组件？
- 销毁时，如何按依赖顺序正确销毁？


传统的应用程序中，控制权在程序本身，程序的控制流程完全由开发者控制。在 IoC 模式下，控制权发生了反转，
即从应用程序转移到了 IoC 容器，所有组件不再由应用程序自己创建和配置，而是由 IoC 容器负责，这样，
应用程序只需要直接使用已经创建好并且配置好的组件。为了能让组件在 IoC 容器中被“装配”出来，需要某种“注入”机制。

```java
// BookService 自己并不会创建 DataSource，而是等待外部通过 setDataSource() 方法来注入一个 DataSource
public class BookService {

    // 创建 dataSource 实例
    private DataSource dataSource;

    // 注入 DataSource
    public void setDataSource(DataSource dataSource) {
        this.dataSource = dataSource;
    }

}
```

IoC 又称为依赖注入(DI：Dependency Injection)，它解决了一个最主要的问题：将组件的创建+配置与组件的使用相分离，
并且，由 IoC 容器负责管理组件的生命周期。

因为 IoC 容器要负责实例化所有的组件，因此，有必要告诉容器如何创建组件，以及各组件的依赖关系。
一种最简单的配置是通过 XML 文件来实现.

``` xml
// XML 配置文件指示 IoC 容器创建 3 个 JavaBean 组件
// 并把 id 为 dataSource 的组件通过属性 dataSource(即调用setDataSource()方法)注入到另外两个组件中
<beans>
    <bean id="dataSource" class="HikariDataSource" />
    
    <bean id="bookService" class="BookService">
        <property name="dataSource" ref="dataSource" />
    </bean>

    <bean id="userService" class="UserService">
        <property name="dataSource" ref="dataSource" />
    </bean>
</beans>
```


> 在 Spring 的 IoC 容器中，我们把所有组件统称为 JavaBean，即配置一个组件就是配置一个 Bean。

## 依赖注入方式

- 依赖注入可以通过 `set()` 方法实现，依赖注入也可以通过构造方法实现。
- Spring 的 IoC 容器同时支持属性注入和构造方法注入，并允许混合使用。
- 示例：构造方法注入

```java
// BookService 通过构造方法注入
public class BookService {
    private DataSource dataSource;

    public BookService(DataSource dataSource) {
        this.dataSource = dataSource;
    }
}
```

## 无侵入容器

在设计上，Spring 的 IoC 容器是一个高度可扩展的无侵入容器。所谓无侵入，
是指应用程序的组件无需实现 Spring 的特定接口，或者说，组件根本不知道自己在 Spring 的容器中运行。
这种无侵入的设计有以下好处：

- 应用程序组件既可以在 Spring 的 IoC 容器中运行，也可以自己编写代码自行组装配置；
- 测试的时候并不依赖 Spring 容器，可单独进行测试，大大提高了开发效率。

## 装配 Bean

用户注册登录示例：

(1)创建 Maven 工程并引入 `spring-context` 依赖

``` xml
<dependencies>
    <dependency>
        <groupId>org.springframework</groupId>
        <artifactId>spring-context</artifactId>
        <version>${spring.version}</version>
    </dependency>
</dependencies>
```

(2) `MailService` 类

- 用于在用户登录和注册成功后发送邮件通知

```java
public class MailService {
    private ZoneId zoneId = ZoneId.systemDefault();

    public void setZoneId(ZoneId zoneId) {
        this.zoneId = zoneId;
    }

    public String getTime() {
        return ZonedDateTime.now(this.zoneId).format(DateTimeFormatter.ISO_ZONED_DATE_TIME);
    }

    public void sendLoginMail(User user) {
        System.err.prinln(String.format("Hi, %s! You are logged in at %s", user.getName(), getTime()));
    }

    public void sendRegistrationMail(User user) {
        System.err.println(String.format("Welcome, %s!", user.getName()));
    }
}
```

# 使用 AOP

AOP 是 Aspect Oriented Programming，即面向切面编程。而 AOP 是一种新的编程方式，它和 OOP 不同，
OOP 把系统看作多个对象的交互，AOP 把系统分解为不同的关注点，或者称之为切面(Aspect)。

## OOP 编程

一个业务组件BookService，它有几个业务方法：

- `createBook`：添加新的Book；
- `updateBook`：修改Book；
- `deleteBook`：删除Book。

```java
public class BookService {
    // ================================
    // 业务方法 createBook()
    // ================================
    // 业务逻辑, 安全检查, 日志记录, 事务处理
    public void createBook(Book book) {
        // 安全检查
        securityCheck();
        // 事务处理
        Transaction tx = startTransaction();
        try {
            // 核心业务逻辑

            // 事务处理
            tx.commit();
        } catch (RuntimeException e) {
            // 事务处理
            tx.rollback();
            throw e;
        }
        // 日志记录
        log("created book: " + book);
    }
    // ================================
    // 业务方法 updateBook()
    // ================================
    // 业务逻辑, 安全检查, 日志记录, 事务处理
    public void updateBook(Book book) {
        // 安全检查
        securityCheck();
        // 事务处理
        Transaction tx = startTransaction();
        try {
            // 核心业务逻辑

            // 事务处理
            tx.commit();
        } catch (RuntimeException e) {
            // 事务处理
            tx.rollback();
            throw e;
        }
        log("update book: " + book);
    }
}
```

Proxy 模式：

```java
public class SecurityCheckBookService implements BookService {
    private final BookService target;

    public SecurityCheckBookService(BookService target) {
        this.target = target;
    }

    public void createBook(Book book) {
        securityCheck();
        target.createBook(book);
    }

    public void updateBook(Book book) {
        securityCheck();
        target.updateBook(book);
    }

    public void deleteBook(Book book) {
        securityCheck();
        target.deleteBook(book);
    }

    private void securityCheck() {
        ...
    }
}
```

## AOP 编程

AOP 编程原理:

AOP本质上只是一种代理模式的实现方式.

如果我们以 AOP 的视角来编写上述业务，可以依次实现：

- 核心逻辑，即BookService；
- 切面逻辑，即：
- 权限检查的 Aspect；
- 日志的 Aspect；
- 事务的 Aspect。

示例：

以 UserService 和 MailService 为例，这两个属于核心业务逻辑，现在，我们准备给 UserService 的每个业务方法执行前添加日志，
给 MailService 的每个业务方法执行前后添加日志，在 Spring 中，需要以下步骤：

(1)通过 Maven 引入 Spring 对 AOP 的支持

```   
// 自动引入 AspectJ，使用 Aspect 实现 AOP 比较方便
<dependency>
    <groupId>org.springframework</groupId>
    <artifactId>spring-aspects</artifactId>
    <version>${spring.version}</version>
</dependency>
```

(2)定义一个 LoggingAspect

```java
@Aspect
@Component
public class LoggingAspect {
    // 在执行 UserService 的每个方法前执行
    @Before("execution(public * com.itranswarp.learnjava.service.UserService.*(..))")
    public void doAccessCheck() {
        System.err.println("[Before] do access check...");
    }

    // 在执行 MailService 的每个方法前后执行
    @Around("execution(public * com.itranswarp.learnjava.service.MailService.*(..))")
    public Object doLogging(ProceedingJoinPoint pip) throws Throwable {
        System.err.println("[Around] start " + pjp.getSignature());
        Object retVal = pjp.proceed();
        System.err.println("[Around] done " + pjp.getSignature());
        return retVal;
    }
}
```

(3)给 @Configuration 类加上一个 @EnableAspectJAutoProxy 注解

```java
@Configuration
@ComponentScan
@EnableAspectJAutoProxy
public class AppConfig {
    ...
}
```

使用 AOP 需要三步：

- 1.定义执行方法，并在方法上通过 AspectJ 的注解告诉 Spring 应该在何处调用此方法；
- 2.标记 @Component 和 @Aspect；
- 3.在 @Configuration 类上标注 @EnableAspectJAutoProxy

## 拦截器类型

- @Before：这种拦截器先执行拦截代码，再执行目标代码。如果拦截器抛异常，那么目标代码就不执行了；
- @After：这种拦截器先执行目标代码，再执行拦截器代码。无论目标代码是否抛异常，拦截器代码都会执行；
- @AfterReturning：和@After不同的是，只有当目标代码正常返回时，才执行拦截器代码；
- @AfterThrowing：和@After不同的是，只有当目标代码抛出了异常时，才执行拦截器代码；
- @Around：能完全控制目标代码是否执行，并可以在执行前后、抛异常后执行任意拦截代码，可以说是包含了上面所有功能。

# 访问数据库

Java 程序访问数据库的标准接口 JDBC，它的实现方式非常简洁，即：Java 标准库定义接口，
各数据库厂商以“驱动”的形式实现接口。应用程序要使用哪个数据库，就把该数据库厂商的驱动以 
jar 包形式引入进来，同时自身仅使用 JDBC 接口，编译期并不需要特定厂商的驱动。

使用 JDBC 虽然简单，但代码比较繁琐。Spring 为了简化数据库访问，主要做了以下几点工作：

- 提供了简化的访问 JDBC 的模板，不必手动释放资源；
- 提供了一个统一的 DAO 类以实现 Data Access Object 模式；
  - 把 `SQLException` 封装为 `DataAccessException`，这个异常是一个 `RuntimeException`，并且让我们能区分 SQL 异常的原因，例如，`DuplicateKeyException` 表示违反了一个唯一约束；
- 能方便地集成 Hibernate、JPA 和 **MyBatis** 这些数据库访问框架；

## 使用 JDBC

Java 程序使用 JDBC 接口访问关系数据库的时候，需要以下几步：

- 1.创建全局 `DataSource` 实例，表示数据库连接池；
- 2.在需要读写数据库的方法内部，按如下步骤访问数据库：
    - (1)从全局 `DataSource` 实例获取 `Connection` 实例；
    - (2)通过 `Connection` 实例创建 `PreparedStatement` 实例；
    - (3)执行 SQL 语句
        - 如果是查询，则通过 `ResultSet` 读取结果集
        - 如果是修改，则获得 `int` 结果


> 正确编写 JDBC 代码的关键是使用 `try ... finally` 释放资源，涉及到事务的代码需要正确提交或回滚事务。

在 Spring 使用 JDBC 的步骤：

- 首先我们通过 IoC 容器创建并管理一个 `DataSource` 实例
- 然后，Spring 提供了一个 `JdbcTemplate`，可以方便地让我们操作 JDBC，因此，
  通常情况下，我们会实例化一个 `JdbcTemplate`。顾名思义，这个类主要使用了 Template 模式。

## 使用声明式事务

# 开发 Web 应用

# 集成第三方组件


# Spring Boot

## 微服务阶段

- JavaSE: OOP
- MySQL: 持久化
- HTML+CSS+JS+JQuery+框架: 视图、框架不熟练、CSS 不好
- JavaWeb: 独立开发 MVC 三层架构的网站
- SSM 框架: 简化了我们的开发流程，配置也开始较为复杂
- War: tomcat 运行
- Spring再简化: SpringBoot Jar: 内嵌 tomcat; 微服务架构！

Spring Boot 是一个基于 Spring 的套件，它帮我们预组装了 Spring 的一系列组件，
以便以尽可能少的代码和配置来开发基于 Spring 的 Java 应用程序。

Spring Boot 和 Spring 的关系就是整车和零部件的关系，它们不是取代关系，
试图跳过 Spring 直接学习 Spring Boot 是不可能的。

Spring Boot 的目标就是提供一个开箱即用的应用程序架构，我们基于 Spring Boot 
的预置结构继续开发，省时省力。


## 环境配置

### 开发者工具

在开发阶段，我们经常要修改代码，然后重启Spring Boot应用。经常手动停止再启动，比较麻烦。

Spring Boot 提供了一个开发者工具，可以监控classpath路径上的文件。只要源码或配置文件发生修改，
Spring Boot 应用可以自动重启。在开发阶段，这个功能比较有用。

要使用这一开发者功能，我们只需添加如下依赖到pom.xml

```
<dependency>
    <groupId>org.springframework.boot</groupId>
    <artifactId>spring-boot-devtools</artifactId>
</dependency>
```

> 默认配置下，针对 `/static`、`/public` 和 `/templates` 目录中的文件修改，不会自动重启，
因为禁用缓存后，这些文件的修改可以实时更新

### 打包 Spring Boot 应用

我们在 Maven 的使用插件一节中介绍了如何使用 `maven-shade-plugin` 打包一个可执行的jar包。
在 Spring Boot 应用中，打包更加简单，因为 Spring Boot 自带一个更简单的 `spring-boot-maven-plugin` 
插件用来打包，我们只需要在 `pom.xml` 中加入以下配置：

```
<project ...>
    ...
    <build>
        <plugins>
            <plugin>
                <groupId>org.springframework.boot</groupId>
                <artifactId>spring-boot-maven-plugin</artifactId>
            </plugin>
        </plugins>
    </build>
</project>
```

无需任何配置，Spring Boot 的这款插件会自动定位应用程序的入口 `Class`，我们执行以下 Maven 
命令即可打包：

```bash
$ mvn clean package
```

### 使用 Actuator


### 使用 Profiles

### 加载配置文件



## Spring Boot 应用

### Spring Boot 应用结构

创建的 Spring Boot Project 标准 Maven 目录结构：

```
springboot-hello
    ├── pom.xml
    ├── src
    │   └── main
    │       ├── java
    │       └── resources
    │           ├── application.yml
    │           ├── logback-spring.xml
    │           ├── static
    │           └── templates
    └── target
```

创建的 Spring Boot Project 源码目录结构：

``` 
src/main/java
└── com
    └── itranswarp
        └── learnjava
            ├── Application.java
            ├── entity
            │   └── User.java
            ├── service
            │   └── UserService.java
            └── web
                └── UserController.java
```

其中，在 `src/main/resources` 目录下，注意到几个文件，下面进行详细解释。

### application.java

`application.yml` 是 Spring Boot 默认的配置文件

- 它采用 `YAML` 格式而不是 `.properties` 格式，
  因为 `YAML` 格式比 `key=value` 格式的 `.properties` 文件更易读
- 文件名必须是 `application.yml` 而不是其他名称

> 也可以使用application.properties作为配置文件，但不如YAML格式简单。

### logback-spring.xml

```java
@SpringBootApplication
public class Application {
    public static void main(String[] args) throws Exception }{
        SpringApplication.run(Applicaton.class, args)
    }
}
```

### 环境变量

### Open API

