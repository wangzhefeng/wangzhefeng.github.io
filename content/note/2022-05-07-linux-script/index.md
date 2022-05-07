---
title: linux-script
author: 王哲峰
date: '2022-05-07'
slug: linux-script
categories:
  - Linux
tags:
  - tool
---



Linux Script
=================



1.Bash快捷键
--------------------


.. code:: 

   Ctl-U   删除光标到行首的所有字符,在某些设置下,删除全行
   Ctl-W   删除当前光标到前边的最近一个空格之间的字符
   Ctl-H   backspace,删除光标前边的字符
   Ctl-R   匹配最相近的一个文件, 然后输出



2.Linux基础
-----------

-  文件及目录管理

-  文本处理

-  磁盘管理

-  进程管理工具

-  性能监控

-  网络工具

-  用户管理工具

-  系统管理及IPC资源管理



2.1 文件及目录管理
~~~~~~~~~~~~~~~~~~



2.1.1 创建目录和文件\ ``mkdir``,\ ``touch``
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

创建目录: 

.. code:: shell

   $ mkdir dirName

创建文件: 

.. code:: shell

   $ touch fileName

   # or 
   $ >fileName



2.1.2 删除目录和文件\ ``rm``
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code:: shell

   # 删除非空目录
   $ rm -rf dir_name

   # 删除日志文件
   $ rm *log
   $ find ./ -name "*log" -exec rm {} \;



2.1.3 移动目录及文件
^^^^^^^^^^^^^^^^^^^^

.. code:: shell

   $ mv sourec_dir dest_dir



2.1.4 复制目录及文件
^^^^^^^^^^^^^^^^^^^^

.. code:: shell

   # 复制目录
   $ cp -r source_dir dest_dir



2.1.5 目录切换
^^^^^^^^^^^^^^

.. code:: shell

   # 找到文件/目录位置
   $ cd dirName

   # 切换到home目录
   $ cd
   $ cd ~


   # 切换到上一级/上上级工作目录
   $ cd .
   $ cd ..

   # 切换到上一个工作目录
   $ cd -

   # 显示当前路径
   $ pwd

   # 更改当前工作路径为"path"
   $ cd path



2.1.6 列出目录项
^^^^^^^^^^^^^^^^

.. code:: shell

   # 显示当前目录下的文件
   $ ls

   $ ls -lrt
   # or 
   $ ll

   $ ls -al|more
   # or 
   $ lm

   # 给每项文件前面加一个id编号
   $ ls | cat -n
   # or
   $ lcn



2.1.7 查找目录及文件\ ``$find``,\ ``locate``
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code:: shell

   # 查看当前目录下文件的个数
   $ find ./ | wc -l

   # 搜寻当前目录下的文件或目录中是否有core开头的内容
   $ find ./ -name "core*" | xargs file

   # 查看当前目录下是否有obj文件
   $ find ./ -name "*.o"

   # 递归当前目录及子目录删除所有.o文件
   $ find ./ -name "*.o" -exec rm {} \;

-  find是实时查找, 如果需要更快的查询, 需要使用\ ``locate``\ ; 

-  locate不是实时查找, locate为文件系统建立索引数据库, 如果有文件更新, 需要定期执行更新命令来更新索引库, 以获得最新的文件索引信息; 

.. code:: 

   # 寻找系统中包含有string的所有路径
   $ locate string

   # 更新索引库
   $ updatedb



2.1.8 查看文件内容
^^^^^^^^^^^^^^^^^^

-  cat

-  vi

-  head

-  tail

-  more

.. code:: shell

   # 显示文件时同时显示行号
   $ cat -n filename

   # 按页显示列表内容
   $ ls -al | more filename

   # 只看前10行
   $ head -10 filename

   # 只看倒数10行
   $ tail -10 filename

   # 查看两个文件间的差别
   $ diff file1 file2

   # 动态显示文本最新信息
   $ tail -f crawler.log



给文件创建别名
^^^^^^^^^^^^^^

-  创建符号链接/硬链接

   -  硬链接

.. code:: shell

   $ ln



在.bashrc(/home/wangzhefeng/.bashrc)中设置命令别名
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code:: shell

   alias ll='ls - lrt'
   alias lm=ls -al|more



3.环境变量
----------

   -  Linux环境变量按照变量的生存周期来划分有两类: 

      -  永久的: 需要修改配置文件, 变量永久生效; 

      -  临时的: 使用\ ``export``\ 命令声明即可, 变量在关闭shell时失效; 



Linux环境变量设置
~~~~~~~~~~~~~~~~~

**1.在\ ``/etc/.profile``\ 文件中添加变量**

-  变量对Linux下所有用户生效, 并且是永久有效的; 

.. code:: shell

   $ sudo gedit /etc/profile

   # spark path
   $ export SPARK_HOME=/usr/lib/spark/spark-2.3.0-bin-hadoop2.7
   $ export PATH=${SPARK_HOME}/bin:$PATH

   # 使环境变量马上生效
   $ source /etc/profile

**2.在用户目录下的\ ``/home/wangzhefeng/.bash_profile``\ 文件中添加变量**

-  变量对当前用户生效, 并且是永久有效的; 

.. code:: shell

   $ sudo gedit /home/wnagzhefeng/.bash_profile

   # spark path
   $ sudo gedit /etc/profile
   $ export SPARK_HOME=/usr/lib/spark/spark-2.3.0-bin-hadoop2.7
   $ export PATH=${SPARK_HOME}/bin:$PATH

   # 使环境变量马上生效
   $ source /home/wangzhefeng/.bash_profile

**3.直接运行\ ``export``\ 命令定义变量**

-  变量只对当前shell(bash)及其子shell(bash)生效, 临时有效; 

.. code:: shell

   $ export var=value



Linux环境变量查看
~~~~~~~~~~~~~~~~~

.. code:: shell

   # 查看某个环境变量
   $ echo $SPARK_HOME

.. code:: shell

   # 查看所有环境变量
   $ env

.. code:: shell

   # 查看所有本地定义的环境变量
   $ set



Linux环境变量删除
~~~~~~~~~~~~~~~~~

.. code:: shell

   $ export VAR=value
   $ unset VAR
   $ env|grep VAR



Linux常用环境变量
~~~~~~~~~~~~~~~~~

-  PATH

   -  决定了shell将到哪些目录中寻找命令或程序

-  HOME

   -  当前用户主目录

-  HISTSIZE

   -  历史记录数

-  LOGNAME

   -  当前用户的登录名

-  HOSTNAME

   -  指主机的名称

-  SHELL

   -  当前用户Shell类型

-  LANGUGE

   -  语言相关的环境变量, 多语言可以修改此环境变量

-  MAIL

   -  当前用户的邮件存放目录

-  PS1

   -  基本提示符, 对于root用户是#, 对于普通用户是$



2.2 文本处理
~~~~~~~~~~~~

-  find文件查找

-  ``grep``\ 文本搜索

-  ``xargs``\ 命令行参数转换

-  ``sort``\ 排序

-  ``uniq``\ 消除重复行



2.3 磁盘管理
~~~~~~~~~~~~

-  查看磁盘空间

-  压缩包

   -  打包/压缩

   -  解包/解压缩



2.3.1 查看磁盘空间
^^^^^^^^^^^^^^^^^^

-  命令行参数

   -  ``-h``:人性化显示

   -  ``-s`` 递归整个目录的大小

**1.查看磁盘空间利用大小:**

.. code:: shell

   df -h

**2.查看当前目录所占空间大小:**

.. code:: shell

   du -sh

.. code:: shell

   du -h

**3.查看当前目录下所有子文件夹(按目录名字排序)所占空间大小: **

.. code:: shell

   $ for i in `ls`
   $ do 
   $   du -sh $i
   $ done [| sort]

or

.. code:: shell

   du -sh `ls` [| sort]



2.3.2 
^^^^^^

**基本概念:**

-  打包: 将一些文件或目录变成一个总的文件; 

-  压缩: 将一个大的问津通过压缩算法变成一个小文件; 

-  解包: 

-  解压缩: 压缩的反过程, 将一个通过软件压缩的文档、文件等各种东西恢复到压缩之前的样子; 

-  打包与压缩: 在Linux中很多

**压缩包文件格式:**

+----------+-----------------------------------+
| 文件格式 | 说明                              |
+==========+===================================+
| demo.zip | WIN,\ ``zip程序``\ 打包压缩的文件 |
+----------+-----------------------------------+
| demo.rar | WIN,\ ``rar程序``\ 压缩的文件     |
+----------+-----------------------------------+
| demo.7z  | WIN,\ ``7zip程序``\ 压缩的文件    |
+----------+-----------------------------------+

+--------------+--------------------------------------------------------+
| 文件格式     | 说明                                                   |
+==============+========================================================+
| demo.tar     | LINUX,\ ``tar程序``\ 打包,未压缩的文件                 |
+--------------+--------------------------------------------------------+
| demo.gz      | LINUX,\ ``gunzip(GUN zip)程序``\ 压缩的文件            |
+--------------+--------------------------------------------------------+
| demo.xz      | LINUX,\ ``xz程序``\ 压缩的文件                         |
+--------------+--------------------------------------------------------+
| demo.bz2     | LINUX,\ ``bzip2程序``\ 压缩的文件                      |
+--------------+--------------------------------------------------------+
| demo.tar.gz  | LINUX,\ ``tar程序``\ 打包,\ ``gunzip程序``\ 压缩的文件 |
+--------------+--------------------------------------------------------+
| demo.tar.xz  | LINUX,\ ``tar程序``\ 打包,\ ``xz程序``\ 压缩的文件     |
+--------------+--------------------------------------------------------+
| demo.tar.bz2 | LINUX,\ ``tar程序``\ 打包,\ ``bzip2程序``\ 压缩的文件  |
+--------------+--------------------------------------------------------+
| demo.tar.7z  | LINUX,\ ``tar程序``\ 打包,\ ``7zip程序``\ 压缩的文件   |
+--------------+--------------------------------------------------------+



2.3.2.1 打包/压缩
'''''''''''''''''

-  打包命令行\ ``tar -cvf``\ 参数

   -  ``-c``: 打包选项

   -  ``-v``: 显示打包进度

   -  ``-f``: 使用档案文件

-  压缩命令行参数

   -  ``gzip``: 压缩为\ ``.gz``\ 文件

**打包:**

.. code:: shell

   tar -cvf demo.tar /dir

**压缩:**

-  生成\ ``demo.txt.gz``

.. code:: shell

   gzip demo.txt



2.3.2.2 解包/解压缩
'''''''''''''''''''

-  解包命令行\ ``tar -xvf``\ 参数

   -  ``-x``: 解包选项

   -  ``-v``: 显示打包进度

   -  ``-f``: 使用档案文件

   -  ``-zxvf``: 解压gz文件

   -  ``-jxvf``: 解压bz2文件

   -  ``-Jxvf``: 解压xz文件

-  解压缩命令行参数

   -  ``bzip2 -d``: decompose解压缩

   -  ``gunzip -d``: 解压缩

1.对格式\ ``.tar``\ 的包进行解包: 

.. code:: shell

   $ tar -xvf demo.tar

2.对格式\ ``.gz``\ 的压缩文件解压缩

.. code:: shell

   $ tar -zxvf demo.gz

3.对格式\ ``.xz``\ 的压缩文件解压缩

.. code:: shell

   $ tar -Jxvf demo.xz

4.对格式\ ``.bz2``\ 的压缩文件解压缩

.. code:: shell

   $ tar -jxvf demo.bz2

5.对格式\ ``.tar.gz``\ 的包进行解压缩、解包

.. code:: shell

   # 先对".tar.gz"解压缩,生成".tar"
   $ gunzip demo.tar.gz

   # 再解包
   $ tar -xvf demo.tar

6.对格式\ ``.tar.xz``\ 的包进行解压缩、解包

.. code:: shell

   $ xz demo.tar.xz
   $ tar -xvf demo.tar

7.对格式\ ``.tar.bz2``\ 的包进行解压缩、解包

.. code:: shell

   tar -jxvf demo.tar.bz2

.. code:: shell

   # 如果tar不支持`j`, 需要先对".tar.bz2"解压缩,生成".tar"
   $ bzip2 -d demo.tar.bz2

   # 再解包
   $ tar -xvf demo.tar

8.对格式\ ``.tar.7z``\ 的包进行解压缩、解包

.. code:: shell

   $ 7zip demo.tar.7z
   $ tar -xvf demo.tar



4.运行代码时常用信息查看命令
----------------------------------------

-  ``cat``: 显示文本的内容

.. code:: shell

   cat <filename>

-  ``wc``: 查看文本行数、词数、字节数

.. code:: shell

   wc <filename>
   wc -l <filename>

-  ``vi/vim``: 编辑文本

.. code:: shell

   vim <filename>

   :q  # 表示直接退出
   :q! # 强制退出
   :wq # 表示写入内存, 再退出, 即保存退出

-  ``more/less``: 从文本的前面/后面显示

.. code:: shell

   more <filename>
   less <filename>

-  ``head/tail``: 显示最前/后面的内容

.. code:: shell

   head <filename>
   tail -f <filename> # 显示不断更新的内容

-  ``file``: 显示文本的编码

.. code:: shell

   file <filename>

-  ``doc2unix``: 将 windows 的换行 ``/r/n`` 变换为 ``/n``

.. code:: shell

   doc2unix <filename>

-  ``grep``: 用于查找文件里符合条件的字符串

.. code:: shell

   grep 正则字符串 <filename>

-  ``awk``: 按指定分隔符列输出

默认按空格或 TAB 键为分隔符

.. code:: shell

   awk '{print 想要输出的列}' <filename>
   awk -F, 'print 想要的列' <filename>

-  ``nohup`` 后台挂起

   -  如果运行很长时间的代码, 一般都会放在后台运行

.. code:: shell

   nohup command > out.file 2>&1 &

``2>1&``: 是将标准错误流重定向到标准输出流 ``2>``: 标准错误重定向
``&1``: 标准输出 ``&`` 让前面的命令在后台执行

-  ``top``: 查看 CPU, 进程, 内存

-  ``kill/pkill``

.. code:: shell

   kill <ID>  # 根据 `top` 得到想杀的进程 ID

.. code:: shell

   pkill <name> # nohup 挂起的进程, 根据 nohup 给的 ID 也能直接杀掉

-  ``>/>>``: 输出到文件

   -  如果文件不存在, 则创建

   -  如果文件存在, 则

      -  ``>`` 表示覆盖写入

      -  ``>>`` 表示 append 写入



5.工具
------



5.1 crontab定时任务
~~~~~~~~~~~~~~~~~~~



5.1.1 命令格式
^^^^^^^^^^^^^^

-  ``$crontab [-u user] file crontab [-u user] [-e|-l|-r|-i]``



5.1.2 crontab文件格式
^^^^^^^^^^^^^^^^^^^^^

-  ``分 时 日 月 星期 要运行的命令``

-  设置crontab文件编辑器环境变量

   -  cd ~

   -  sudo gedit .profile

   -  EDITOR=gedit; export EDITOR

-  创建crontab文件

   -  crontab文件存放位置: "/var/spool/cron/wangzhefengcron"

   -  ``* * * * * /bin/echo 'date' > /dev/console``

-  提交crontab任务进程(新创建文件的一个副本放在/var/spool/cron中)

   -  ``crontab tinkercron``

-  列出crontab文件

   -  ``crontab -l``

   -  在$HOME目录中对crontab文件做一备份

      -  ``cron -l > $HOME/mycron``

-  编辑crontab文件

   -  ``crontab -e``

-  删除crontab文件

   -  ``crontab -r``



5.2 examples
~~~~~~~~~~~~

1.  每分钟执行一次myCommand

    -  ``$* * * * * myCommand``

2.  每小时的第3和第15分钟执行

    -  ``$3,15 * * * * myCommand``

3.  在上午8点到11点的第三和第15分钟执行

    -  ``$3,15 8-11 * * * myCommand``

4.  每隔两天的上午8点到11点的第3和第15分钟执行

    -  ``$3,15 8-11 */2 * * myCommand``

5.  每周一上午8点到11点的第3和第15分钟执行

    -  ``$3,15 8-11 * * 1 myCommand``

6.  每晚的21:30重启smb

    -  ``$30 21 * * * /etc/init.d/smb restart``

7.  每月1、10、22日的4 : 45重启smb

    -  ``$45 4 1,10,22 * * /etc/init.d/smb restart``

8.  每周六、周日的1 : 10重启smb

    -  ``$10 1 * * 6,0 /ect/init.d/smb restart``

9.  每天18 : 00至23 : 00之间每隔30分钟重启smb

    -  ``$0,30 18-23 * * * /etc/init.d/smb restart``

10. 每星期六的晚上11 : 00 pm重启smb

    -  ``$0 23 * * 6 /etc/init.d/smb restart``

11. 每一小时重启smb

    -  ``$* */1 * * * /etc/init.d/smb restart``

12. 晚上11点到早上7点之间, 每隔一小时重启smb

    -  ``$0 23-7 * * * /etc/init.d/smb restart``



5.3 系统级任务调度和用户级任务调度
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

-  可以将用户级任务调度放到系统级任务调度来完成(不建议这么做), 但是反过来却不行

-  root用户任务调度

   -  ``$crontab -uroot -e``



5.4 log位置
~~~~~~~~~~~

-  ``/etc/init.d/crond restart``

-  ``$tail -f /var/log/cron``



5.5 特殊字符
~~~~~~~~~~~~

-  ``%``

-  转义

   -  ``%%``



5.6 重启cron
~~~~~~~~~~~~

-  ``$service cron restart``



5.7 ubuntu启动, 停止, 重启cron
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

-  ``$sudo /etc/init.d/cron start``

-  ``$sudo /etc/init.d/cron stop``

-  ``$sudo /etc/init.d/cron restart``
