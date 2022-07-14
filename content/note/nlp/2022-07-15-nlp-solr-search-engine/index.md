---
title: NLP--Solr 搜索引擎
author: 王哲峰
date: '2022-04-05'
slug: nlp-utils-spacy
categories:
  - NLP
tags:
  - tool
---

NLP--Solr 搜索引擎
=============================

1.Solr 简介
-----------------------------

   - 基于关键字查询的搜索引擎
      - 缺点:难以构造出准确表达用户需求的查询请求, 返回的结果冗余甚至无用的信息多
   - 基于 NLP 处理的搜索引擎
      - 优点:能够更好地理解用户的查询意图, 更准确地推荐相关查询请求, 并且高效地返回更相关的查询结果
   - Lucene
      - Lucene 是一个基于 Java 的全文信息检索工具包, 它不是一个完整的搜索应用程序, 而为应用程序提供索引和搜索功能
      - Lucene 是 Apache Jakarta(雅加达)家族中的一个开源项目, 也是目前最为流行的基于 Java 开源全文检索工具包
      - 目前已经有很多应用程序的搜索功能是基于 Lucene, 比如 Eclipse 帮助系统的搜索功能
      - Lucene 能够为文本类型的数据建立索引, 所以只要把要索引的数据格式转化为文本格式, Lucene 就能对文档进行索引和搜索
   - Solr
      - Solr 与 Lucene 并不是竞争关系, Solr 依存于 Lucene, 因为 Solr 底层的核心技术是使用 Lucene 来实现的
      - Solr 和 Lucene 的本质区别有三点:

         - 搜索服务器
            - Lucene 本质上是搜索库, 不是独立的应用程序, 而 Solr 是
         - 企业级
            - Lucene 专注于搜索底层的建设, 而 Solr 专注于企业应用
         - 管理
            - Lucene 不负责支撑搜索服务所必需的管理, 而 Solr 负责

      - Solr 是 Lucene 面向企业搜索应用的扩展
   - Solr 与 NLP 的关系
      - 在 NLP 的处理过程中, 有一些场景, 比如人机交互, 是需要实时或近似实时的。
        在人机对话中, 用户所关心的一些常用问题、答案会尽可能预存在 Solr 中做检索, 
        当用户提问机器人的时候, NLP 算法会先理解问题的语义, 之后将“翻译”后的语言推送给 Solr, 
        由 Solr 负责检索预存的问题, 将最匹配用户提问的那个答案返回给用户; 

2.全文检索的原理
-----------------------------

   - 全文检索
      
      - 比如, 电脑里有一个文件夹, 文件夹中存储了很多文件, 例如 Word、Excel、PPT, 希望根据搜索关键字的方式搜索到相应的文档, 
        比如输入 Solr, 所有内容含有 Solr 这个关键字的文件就会被筛选出来, 这个就是全文检索
   
   - 搜索方法(非结构化数据)
      
      - 顺序扫描法(Serial Scanning)
         
         - 比如要找内容包含某一个字符串的文件, 就是一个文档一个文档地看, 对于每一个文档, 从头看到尾, 
           如果该文档包含此字符串, 则此文档为我们要找的文件, 接着看下一个文件, 知道扫描完所有的文件
         - Linux 下的 `grep` 命令就是利用了顺序扫描法来寻找包含某个字符串的文件, 这种方法针对小数据量的文件来说最直接, 
           但是对于大量文件, 查询效率就很低了
   
      - 全文检索
         
         - 对非结构化数据顺序扫描很慢, 对结构化数据的搜索却相对较快, 把非结构化数据转化为结构化数据构成了全文检索的基本思路, 
           也就是将非结构化数据中的一部分信息提取出来, 重新组织, 使其变得有一定结构, 然后对这些有一定结构的数据进行搜索, 
           从而达到搜索相对较快的目的
           
            - 从非结构化数据中提取出然后重新组织的信息, 称为 **索引**
            - 先建立索引, 再对索引进行搜索的过程, 就叫做 **全文检索(full-text search)**

      - 顺序扫描 vs 全文检索
         
         - 创建索引之后进行检索与顺序扫描的区别在于, 顺序扫描时每次都要扫描, 而创建索引的过程仅仅需要一次, 以后便是一劳永逸了, 
           每次搜索, 创建索引的过程不必经过, 仅仅搜索创建好的索引就可以了。这也是全文搜索相对于顺序扫描的优势之一:
            
            - 一次索引, 多次使用

3.Solr 简介和部署
-----------------------------

3.1 Solr 简介
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

   - Solr 是一种开发源码的、基于 Lucene Java 的搜索服务器, 易于加入 Web 应用程序中
   - Solr 提供了层面搜索(就是统计)、命中醒目显示并且支持多种输出格式
      
      - XML
      - XSLT
      - JSON
   
   - Solr 易于安装和配置, 而且附带了一个基于 HTTP 的管理页面
   - 可以使用 Solr 的基本搜索功能, 也可以对它进行扩展从而满足企业的需要
   - Solr 的特性:
   
      - (1)高级的全文搜索功能
      - (2)转为高质量的网络流量进行的优化
      - (3)基于开放接口(XML、HTTP)的标准
      - (4)综合的 HTML 管理页面
      - (5)可伸缩性--能够有效地复制到另一个 Solr 搜索服务器
      - (6)使用 XML 配置达到灵活性和适配性
      - (7)可扩展的插件体系

3.2 Solr 部署
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

   - 1.下载 Tomcat、Solr、JDK 安装包
      
      - Tomcat 8.5.24
         - http://tomcat.apache.org/download-80.cgi
      - Solr 6.5.1
         - http://archive.apache.org/dist/lucene/solr/
      - JDK 1.8
         - TODO
   
   - 2.规划安装目录
      
      - Solr 安装在 Linux 的 `/opt/bigdata/` 目录中

         .. code-block:: shell

            $ mkdir -p /opt/bigdata

   - 3.将下载好的 Tomcat、Solr、JDK 包移动到 `/opt/bigdata/` 下, 解压、安装

      - (1)解压 Tomcat、Solr

         .. code-block:: shell

            $ cd /opt/bigdata
            
            $ tar -zxvf apache-tomcat-8.5.24.tar.gz
            $ tar -zxvf solr-6.5.1.gz
            
            $ mv /opt/bigdata/apache-tomcat-8.5.24 /opt/bigdata/tomcat
            $ mv /opt/bigdata/solr-6.5.1 /opt/bigdata/solr
      
      - (2)JDK 安装
         
         .. code-block:: shell
         
            $ rpm -ivh jdk-8u144-linux-x64.rpm
            $ java -version

      - (3)目录结构

         .. code-block:: 

            - /opt/bigdata
               - tomcat
               - solr

   - 4.Solr 集成 Tomcat

      - Solr 需要运行在一个 Servlet 容器中, Solr 6.5.1 要求 JDK 使用 1.8 以上版本, Solr 默认提供 Jetty(Java 写的 Servlet 容器), 
        这里使用 Tomcat 作为 Servlet 容器
      
      - (1)创建 Tomcat 的 Solr 工程

         .. code-block:: shell

            # 复制 Solr 中的 webapp 到 Tomcat webapps
            $ cp /opt/bigdata/solr/server/solr-webapp/webapp /opt/bigdata/tomcat/webapps/
            $ mv /opt/bigdata/tomcat/webapps/webapp /opt/bigdata/tomcat/webapps/solr

            # 复制 Solr 的 jar 包到 Tomcat 的 Solr 工程中
            $ cp /opt/bigdata/solr/server/lib/ext/* /opt/bigdata/tomcat/webapps/solr/WEB-INF/lib
            $ cp /opt/bigdata/solr/server/lib/metric*.jar /opt/bigdata/tomcat/webapps/solr/WEB-INF/lib
            $ cp /opt/bigdata/solr/dist/*.jar /opt/bigdata/tomcat/webapps/solr/WEB-INF/lib
      
      - (2)创建 `solrhome` 文件夹, `solrhome` 是存放 Solr 服务器所有配置文件的目录

         .. code-block:: shell

            $ mkdir /opt/bigdata/solrhome
            $ cp -r /opt/bigdata/solr/server/solr/* /opt/bigdata/solrhome

      - (3)添加 `log4j` 的配置文件

         .. code-block:: shell

            # 创建 classes 目录
            $ mkdir /opt/bigdata/tomcat/webapps/solr/WEB-INF/classes
            # 复制 log4j.properties 文件
            $ cp /opt/bigdata/solr/server/resources/log4j.properties /opt/bigdata/tomcat/webapps/solr/WEB-INF/classes
      
      - (4)在 Tomcat 的 Solr 工程的 `web.xml` 文件中指定 `solrhome` 的位置

         .. code-block:: shell

            $ cd /opt/bigdata/tomcat/webapps/solr/WEB-INF/
            $ vim web.xml
         
         .. code-block:: xml
         
            <env-entry>
               <env-entry-name>solr/home</env-entry-name>
               <env-entry-value>/opt/bigdata/solrhome</env-entry-value>
               <env-entry-type>java.lang.String</env-entry-type>
            </env-entry>

      - (5)启动 Tomcat

         .. code-block:: shell

            $ /opt/bigdata/tomcat/bin/startup.sh

      - (6)建立 solrcore

         .. code-block:: shell

            $ cd /opt/bigdata/solrhome
            $ mkdir my_core
            $ cp /opt/bigdata/solr/server/solr/configsets/sample_techproducts_config/conf /opt/bigdata/solrhome/my_core

         - 访问 Solr 地址

            - http://xxx.xxx.xxx.xxx:8080/solr/index.html

      - (7)访问 Solr 控制台界面主页

         - http://xxx.xxx.xxx.xxx:8000/solr/admin.html
         - 在 Solr 管理控制台界面, 添加一个 core

4.Solr 后台管理描述
-----------------------------


5.Solr 管理索引库
-----------------------------


