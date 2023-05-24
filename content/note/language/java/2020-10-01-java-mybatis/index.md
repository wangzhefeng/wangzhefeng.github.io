---
title: MyBatis
author: 王哲峰
date: '2020-10-01'
slug: java-mybatis
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

- [什么是 MyBatis?](#什么是-mybatis)
- [安装 MyBatis](#安装-mybatis)
- [什么是 MyBatis Plus](#什么是-mybatis-plus)
  - [前提条件](#前提条件)
  - [数据库表](#数据库表)
  - [建立项目](#建立项目)
    - [初始化工程](#初始化工程)
    - [添加依赖](#添加依赖)
    - [配置](#配置)
    - [编码](#编码)
    - [开始使用](#开始使用)
- [安装、配置 MyBatis-Plus](#安装配置-mybatis-plus)
  - [安装](#安装)
    - [Release](#release)
    - [Snapshot](#snapshot)
  - [配置](#配置-1)
- [注解](#注解)
- [代码生成器](#代码生成器)
- [CRUD 接口](#crud-接口)
- [条件构造器](#条件构造器)
- [Sequence 主键](#sequence-主键)
- [自定义ID生成器](#自定义id生成器)
- [插件扩展](#插件扩展)
  - [逻辑删除](#逻辑删除)
    - [使用方法](#使用方法)
    - [常见问题](#常见问题)
- [MyBatis Plus Examples](#mybatis-plus-examples)
  - [自动添加数据库中表中的创建时间、创建者、更新时间、更新者](#自动添加数据库中表中的创建时间创建者更新时间更新者)
    - [数据库表](#数据库表-1)
    - [配置](#配置-2)
    - [MetaHandler 类](#metahandler-类)
    - [执行正常的增删改操作](#执行正常的增删改操作)
    - [SysDeptEntity Entity 类](#sysdeptentity-entity-类)
    - [BaseEntity 类](#baseentity-类)
</p></details><p></p>

# 什么是 MyBatis?

- MyBatis 是一款优秀的持久层框架，它支持自定义 SQL、存储过程以及高级映射。
- MyBatis 免除了几乎所有的 JDBC 代码以及设置参数和获取结果集的工作。
- MyBatis 可以通过简单的 **XML** 或 **注解** 来配置和映射 **原始类型**、**接口** 和 **Java POJO** (Plain Old Java Objects，普通老式 Java 对象)为数据库中的记录。

# 安装 MyBatis



# 什么是 MyBatis Plus

- 学习前提
    - MyBatis
    - Spring
    - SpringMVC
- MySQL 数据库驱动配置
    - MySQL 5

    ```
    spring.datasource.username=root
    spring.datasource.password=123456
    spring.datasource.url=jdbc:mysql://localhost:3306/mybatis_plus?userSSL=false&useUnicode=true&characterEncoding=utf-8
    spring.datesource.driver-class-name=com.mysql.cj.jdbc.Driver
    ```

    - MySQL 8 
        - 驱动与 MySQL 5 不同，需要增加时区的配置

    ```
    spring.datasource.username=root
    spring.datasource.password=123456
    spring.datasource.url=jdbc:mysql://localhost:3306/mybatis_plus?userSSL=false&useUnicode=true&characterEncoding=utf-8&serverTimezone=GMT%2B8
    spring.datesource.driver-class-name=com.mysql.cj.jdbc.Driver
    ```

MyBatis-Plus 是一个 MyBatis 的增强工具，在 MyBatis 的基础上只做增强不做改变，为简化开发、提高效率而生。

## 前提条件

- 拥有 Java 开发环境以及相应 IDE
- 熟悉 Spring Boot
- 熟悉 Maven

## 数据库表

- `User` 表结构

| id | name    | age | email |
|----|---------|-----|--------|
| 1  | Jone    | 18  | test1@baomidou.com |
| 2  | Jack    | 20  | test2@baomidou.com |
| 3  | Tom     | 28  | test3@baomidou.com |
| 4  | Sandy   | 21  | test4@baomidou.com |
| 5  | Billie  | 24  | test5@baomidou.com |

- `User` 对应的数据库 Schema 脚本

```sql
DROP TABLE IF EXISTS user;

CREATE TABLE user (
    id BIGINT(20) NOT NULL COMMENT '主键ID',
    name VARCHAR(30) NULL DEFAULT NULL COMMENT '姓名',
    age INT(11) NULL DEFAULT NULL COMMENT '年龄',
    email VARCHAR(50) NULL DEFAULT NULL COMMENT '邮箱',
    PRIMARY KEY (id)
);
```

- `User` 对应的数据库 Data 脚本

```sql
DELETE FROM user;

INSERT INTO user (id, name, age, email) VALUES
(1, 'Jone', 18, 'test1@baomidou.com'),
(2, 'Jack', 20, 'test2@baomidou.com'),
(3, 'Tom', 28, 'test3@baomidou.com'),
(4, 'Sandy', 21, 'test4@baomidou.com'),
(5, 'Billie', 24, 'test5@baomidou.com');
```

## 建立项目

### 初始化工程

- 创建一个空的 Spring Boot 工程
- 快速创建工具 `Spring Initializer <https://start.spring.io/>`_ 

### 添加依赖

- (1)在 `pox.xml` 中引入 Spring Boot Starter 父工程
- (2)在 `pox.xml` 中引入如下依赖 
    - `spring-boot-starter`
    - `sprint-boot-starter-test`
    - `mybatis-plus-boot-starter`
    - `lombok`
    - `h2`

``` xml
<?xml version="1.0" encoding="UTF-8"?>
<project xmlns="http://maven.apache.org/POM/4.0.0" xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance"
    xsi:schemaLocation="http://maven.apache.org/POM/4.0.0 https://maven.apache.org/xsd/maven-4.0.0.xsd">
    <modelVersion>4.0.0</modelVersion>
    // =======================================
    // Spring Boot Starter Parent Project
    // -------
    <parent>
        <groupId>org.springframework.boot</groupId>
        <artifactId>spring-boot-starter-parent</artifactId>
        <version>2.3.4.RELEASE</version>
        <relativePath/> <!-- lookup parent from repository -->
    </parent>
    // =======================================
    <groupId>com.example</groupId>
    <artifactId>demo</artifactId>
    <version>0.0.1-SNAPSHOT</version>
    <name>demo</name>
    <description>Demo project for Spring Boot</description>

    // =======================================
    // Java 1.8
    // -------
    <properties>
        <java.version>1.8</java.version>
    </properties>
    // =======================================

    // =======================================
    // MyBatis-Plus 依赖
    // -------
    <dependencies>
        <dependency>
            <groupId>org.springframework.boot</groupId>
            <artifactId>spring-boot-starter</artifactId>
        </dependency>
        <dependency>
            <groupId>org.springframework.boot</groupId>
            <artifactId>spring-boot-starter-test</artifactId>
            <scope>test</scope>
        </dependency>
        <dependency>
            <groupId>org.projectlombok</groupId>
            <artifactId>lombok</artifactId>
            <optional>true</optional>
        </dependency>
        <dependency>
            <groupId>com.baomidou</groupId>
            <artifactId>mybatis-plus-boot-starter</artifactId>
            <version>3.4.0</version>
        </dependency>
        <dependency>
            <groupId>com.h2database</groupId>
            <artifactId>h2</artifactId>
            <scope>runtime</scope>
        </dependency>
    </dependencies>
    // =======================================

    // =======================================
    // maven build
    // -------
    <build>
        <plugins>
            <plugin>
                <groupId>org.springframework.boot</groupId>
                <artifactId>spring-boot-maven-plugin</artifactId>
            </plugin>
        </plugins>
    </build>
    // =======================================

</project>
```

### 配置

- 在 `./project/src/main/resources/application.yml` 配置文件中添加数据库的相关配置

```
# -------------------
# H2 database
# -------------------
# DataSource Config
spring:
datasource:
    driver-class-name: org.h2.Driver
    schema: classpath:db/schema-h2.sql
    data: classpath:db/data-h2.sql
    url: jdbc:h2:mem:test
    username: root
    password: test

# -------------------
# MySQL database
# -------------------
spring:
datasource:
    type: com.alibaba.druid.pool.DruidDataSource
    url: jdbc:mysql://0.0.0.0:3306/db_name?useUnicode=true&characterEncoding=UTF-8&serverTimezone=Asia/Shanghai&allowMultiQueries=true&autoReconnect=true&failOverReadOnly=false
    username: username
    password: password
    driver-class-name: com.mysql.jdbc.Driver

# -------------------
# logging config
# -------------------
logging:
file:
    name: ./logs/logger.log
level:
    furnace: info
    com.example.demo.mapper: debug
```

- 在 Spring Boot 启动类 `./project/src/main/java/com.projet_name/projectApplication.java` 中添加 `@MapperScan` 注解，扫描 `Mapper` 文件夹

```java
@SpringBootApplication
@MapperScan({"com.projet_name.**.mapper"})
public class projectApplication {

    public static void main(String[] args) {
        SpringApplication.run(projectApplication.class, args);
    }

}
```

### 编码

- 编写实体类(entity class) `./project/src/main/java/com.project_name/user/entity/User.java`

```java
package com.project_name.basic.entity;

import com.baomidou.mybatisplus.annotation.TableName;
import com.baomidou.mybatisplus.annotation.IdType;
import com.baomidou.mybatisplus.annotation.TableId;
import java.time.LocalDateTime;
import java.io.Serializable;
import java.util.Date;

import io.swagger.annotations.ApiModel;
import io.swagger.annotations.ApiModelProperty;
import lombok.Data;
import lombok.EqualsAndHashCode;

/**
* <p>
* table_name
* </p>
*
* @author admin
* @since 2020-09-22
*/
@Data
@EqualsAndHashCode(callSuper = false)
@TableName("table_name")
@ApiModel(value="Table对象", description="table_name")
public class User implements Serializable {

    private static final long serialVersionUID = 1L;

    @ApiModelProperty(value = "id")
    @TableId(value = "id", type = IdType.AUTO)
    private Long id;

    @ApiModelProperty(value = "姓名")
    private String name;

    @ApiModelProperty(value = "年龄")
    private String age;

    @ApiModelProperty(value = "邮箱")
    private String email;

}
```

- 编写 Mapper 类 `./project/src/main/java/com.project_name/user/mapper/UserMapper.java`

```java
public interface UserMapper extends BaseMapper<User> {

}
```

### 开始使用

- 添加测试类，进行功能测试

```java
@RunWith(SpringRunner.class)
@SpringBootTest
public class SampleTest {

    @Autowired
    private UserMapper userMapper;

    @Test
    public void testSelect() {
        System.out.println(("---- selectAll method test -----"));
        // selectList() 方法的参数为 MP 内置的条件封装器 Wrapper，所以不填写就是无任何条件
        List<User> userList = userMapper.selectList(null);
        Assert.assertEquals(5, userList.size());
        userList.forEach(System.out::println);
    }

}

// 控制台输出
User(id=1, name=Jone, age=18, email=test1@baomidou.com)
User(id=2, name=Jack, age=20, email=test2@baomidou.com)
User(id=3, name=Tom, age=28, email=test3@baomidou.com)
User(id=4, name=Sandy, age=21, email=test4@baomidou.com)
User(id=5, name=Billie, age=24, email=test5@baomidou.com)
```

# 安装、配置 MyBatis-Plus

全新的 MyBatis-Plus 3.0 版本基于 JDK8，提供了 `lambda` 形式的调用，所以安装继承 MP3.0 要求如下：

- JDK 8+
- Maven or Gradle

## 安装

### Release

- Spring Boot Maven

``` xml
<dependency>
    <groupId>com.baomidou</groupId>
    <artifactId>mybatis-plus-boot-starter</artifactId>
    <version>3.4.0</version>
</dependency>
```

- Spring MVC Maven

``` xml
<dependency>
    <groupId>com.baomidou</groupId>
    <artifactId>mybatis-plus</artifactId>
    <version>3.4.0</version>
</dependency>
```


> 引入 MyBatis-Plus 之后请不要再次引入 MyBatis 以及 MyBatis-Spring，以避免因版本差异导致的问题。

### Snapshot

- 快照 SNAPSHOT 版本需要添加仓库，且版本号为快照版本

``` xml
<repository>
    <id>snapshots</id>
    <url>https://oss.sonatype.org/content/repositories/snapshots/</url>
</repository>
```

## 配置

在配置 MyBatis-Plus 之前，应确保已经安装了 MyBatis-Plus

Spring Boot 工程

- 配置 `MapperScan` 注解

```java
@SpringBootApplication
@MapperScan("com.baomidou.mybatisplus.samples.quickstart.mapper")
public class Application {

    public static void main(String[] args) {
        SpringApplication.run(Application.class, args);
    }

}
```

Spring MVC 工程

- 配置 MapperScan

```xml
<bean class="org.mybatis.spring.mapper.MapperScannerConfigurer">
    <property name="basePackage" value="com.baomidou.mybatisplus.samples.quickstart.mapper"/>
</bean>
```

- 调整 `SqlSessionFactory` 为 MyBatis-Plus 的 `SqlSessionFactory`

```xml
<bean id="sqlSessionFactory" class="com.baomidou.mybatisplus.extension.spring.MybatisSqlSessionFactoryBean">
    <property name="dataSource" ref="dataSource"/>
</bean>
```

# 注解

`MybatisPlus` 注解包相关类

- `@TableName`
- `@TableId`
- `@TableField`
- `@Version`
- `@EnumValue`
- `@TableLogic`
- `@SqlParser`
- `@KeySequence`

# 代码生成器

AutoGenerator 是 MyBatis-Plus 的代码生成器，通过 AutoGenerator 可以快速生成 Entity、Mapper、Mapper XML、
Service、Controller 等各个模块的代码，极大的提升了开发效率。

# CRUD 接口

# 条件构造器


# Sequence 主键

# 自定义ID生成器


# 插件扩展

## 逻辑删除

### 使用方法

- 步骤1：配置
    - `com.baomidou.mybatisplus.core.config.GlobalConfig$DbConfig`
    - `application.yml`

```
mybatis-plus:
    global-config:
        db-config:
        logic-delete-field: flag  # 全局逻辑删除的实体字段名(since 3.3.0,配置后可以忽略不配置步骤2)
        logic-delete-value: 1     # 逻辑已删除值(默认为 1)
        logic-not-delete-value: 0 # 逻辑未删除值(默认为 0)
```

- 步骤2：实体类字段上加上 `@TableLogic` 注解

```java
@TableLogic
private Integer deleted;
```

### 常见问题

- 1.如何 insert ?
    - (1)字段在数据库定义默认值(推荐)
    - (2)insert 前自己 set 值
    - (3)使用自动填充功能
- 2.删除接口自动填充功能失效
    - (1)使用 update 方法并 UpdateWrapper.set(column, value)(推荐)
    - (2)使用 update 方法并 UpdateWrapper.setSql("column=value")
    - (3)使用 Sql 注入器注入
        - `com.baomidou.mybatisplus.extension.injector.methods.LogicDeletedByIdWithFill` 并使用(推荐)


# MyBatis Plus Examples

## 自动添加数据库中表中的创建时间、创建者、更新时间、更新者

### 数据库表

```sql
CREATE TABLE `sys_dept` (
    `dept_id` bigint(20) NOT NULL AUTO_INCREMENT COMMENT '部门id',
    `parent_id` bigint(20) DEFAULT '0' COMMENT '父部门id',
    `dept_name` varchar(30) DEFAULT '' COMMENT '部门名称',
    `order_num` int(4) DEFAULT '0' COMMENT '显示顺序',
    `status` tinyint(1) DEFAULT '0' COMMENT '部门状态（0：正常 1：停用）',
    `create_by` varchar(64) DEFAULT '' COMMENT '创建者',
    `create_time` datetime DEFAULT NULL COMMENT '创建时间',
    `update_by` varchar(64) DEFAULT '' COMMENT '更新者',
    `update_time` datetime DEFAULT NULL COMMENT '更新时间',
    `remark` varchar(200) DEFAULT '' COMMENT '备注',
        PRIMARY KEY (`dept_id`)
) ENGINE=InnoDB AUTO_INCREMENT=27 DEFAULT CHARSET=utf8;
```

### 配置

```java
    @Configuration
    public class MyBatisPlusConfig {
        /**
         * 自动填充功能
         * @return
         */
        @Bean
        public GlobalConfig globalConfig() {
            GlobalConfig globalConfig = new GlobalConfig();
            globalConfig.setMetaObjectHandler(new MetaHandler());
            return globalConfig;
        }
    }
```

### MetaHandler 类

```java
/**
    * 处理新增和更新的基础数据填充，配合BaseEntity和MyBatisPlusConfig使用
    */
@Component
public class MetaHandler implements MetaObjectHandler {

    private static final Logger logger = LoggerFactory.getLogger(MetaHandler.class);

    /**
        * 新增数据执行
        * @param metaObject
        */
    @Override
    public void insertFill(MetaObject metaObject) {
        SysUserEntity userEntity = ShiroUtil.getUser();
        this.setFieldValByName("createTime", new Date(), metaObject);
        this.setFieldValByName("createBy", userEntity.getLoginName(), metaObject);
        this.setFieldValByName("updateTime", new Date(), metaObject);
        this.setFieldValByName("updateBy", userEntity.getLoginName(), metaObject);
    }

    /**
        * 更新数据执行
        * @param metaObject
        */
    @Override
    public void updateFill(MetaObject metaObject) {
        SysUserEntity userEntity = ShiroUtil.getUser();
        this.setFieldValByName("updateTime", new Date(), metaObject);
        this.setFieldValByName("updateBy", userEntity.getLoginName(), metaObject);
    }

}
```

### 执行正常的增删改操作

```java
@RequiresPermissions("sys:dept:add")
@PostMapping("/add")
@ResponseBody
public R add(@RequestBody SysDeptEntity deptEntity) {
    logger.info("添加信息={}", deptEntity);
    sysDeptService.save(deptEntity); // 不再需要设置setCreateBy、setCreateTime、setUpdateBy、setUpdateTime操作，代码更优美
    return R.ok();
}
```

### SysDeptEntity Entity 类

```java
@Data
@TableName("sys_dept")
public class SysDeptEntity extends BaseEntity {

    private static final long serialVersionUID = 1L;

    /**
        * 部门ID
        **/
    @TableId
    private Long deptId;

    /**
        * 部门父节点ID
        **/
    private Long parentId;

    /**
        * 部门名称
        **/
    private String deptName;

    /**
        * 显示顺序
        **/
    private Integer orderNum;

    /**
        * 用户状态（0：正常 1：禁用）
        **/
    private Integer status;

    @TableField(exist = false)
    private List<SysDeptEntity> children;

}
```

### BaseEntity 类

```java
/**
    * 基础Bean
    */
@Data
public class BaseEntity implements Serializable {

    @TableField(value = "create_by", fill = FieldFill.INSERT) // 新增执行
    private String createBy;

    @TableField(value = "create_time", fill = FieldFill.INSERT)
    private Date createTime;

    @TableField(value = "update_by", fill = FieldFill.INSERT_UPDATE) // 新增和更新执行
    private String updateBy;

    @TableField(value = "update_Time", fill = FieldFill.INSERT_UPDATE)
    private Date updateTime;

    @TableField(value = "remark")
    private String remark;

}
```
