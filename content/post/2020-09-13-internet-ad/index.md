---
title: 互联网广告
author: 王哲峰
date: '2020-09-13'
slug: internet-ad
categories:
  - machinelearning
tags:
  - article
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

- [互联网广告简史](#互联网广告简史)
  - [广告售卖方式历史](#广告售卖方式历史)
  - [计费方式历史](#计费方式历史)
- [互联网广告类型](#互联网广告类型)
  - [搜索广告](#搜索广告)
  - [定向广告](#定向广告)
  - [实时广告竞价](#实时广告竞价)
  - [广告系统架构](#广告系统架构)
</p></details><p></p>

# 互联网广告简史

## 广告售卖方式历史

- 显示广告(Display Advertising)
   - 互联网页面中的条幅(banner)广告
      - 工程师把HTML的代码嵌入到网站的某个位置, 没有复杂的在线计算
   - 合约广告(Agreement-based Advertising)
      - 采用合同约束的方式, 让某一广告位在某一段时间被特定广告主所独占
   - 位置、时间售卖
- 定向广告(Targetd Advertising)
   - 通过受众定向技术(Audience Targeting)分析用户的属性(User
      Profile)标签(性别、年龄...), 通过广告投放服务(Ad
      Server), 将原来静态的广告HTML代码转变为根据算法实时变换广告HTML代码, 返回给浏览器
- 担保式投放(Guarnateed Delivery)
   - 网站作为媒体方, 需要跟广告主保证一定量的投放量(页面或者时长)
   - 计费方式: CPM(Cost Per Mile)
- 竞价广告(Auction-based Advertising)
   - 广义一阶价格拍卖(GFP) - Overture(GoTo)
   - 广义二阶价格拍卖(GSP) - Google
      - 搜索广告(Search Ad)
         - 根据用户的搜索词进行定向, 然后投放相关的广告
      - 上下文广告(Contextual Ad)
         - 根据网页的上下文内容进行定向, 这种定向方式是假定用户的兴趣点跟当前的网页内容是相关的
         - 产生了根据用户历史行为进行定向的技术(Behavior Targeting)
   - 实时竞价(Real Time Bidding)
      - 改变了预先由广告主出价的模式, 改为每次请求的时候, 实时出价, 改变自己的流量
      - 广告交易平台(Ad Exchange)
         - 交易平台, 处于中间的角色, 一方面接入了流量, 另一方面接入了竞价的广告主
         - 需求方: 需求方平台(Demand Side Platform, DSP)
            - 广告主, 专门的技术公司
         - 提供方: 提供方平台(Supply Side Platform, SSP)
            - (小型)网站

## 计费方式历史

- CPM(Cost Per Mile)
   - 广告主要求: 大面积、稳定的曝光
- CPT(Cost Per Time)
   - 广告主要求增加曝光度
- CPC(Cost per Click)
   - 按照点击付费方式；能够让一些中小网站媒体获得了可以变现的方式
   - 广告主只想把流量引入到自己的网站
- CPS(Cost per Sale)
   - 只有在发生了销售的情况下, 广告主才付费；有力于广告主, 不用担心投资回报率的问题
   - 只想在成交的时候才付费

# 互联网广告类型

- 条幅广告(Banner Ad)
- 邮件直接营销广告(Email Direct Marketing Ad, EDMA)
- 富媒体广告(Rich Media Ad)
- 视屏广告(Video Ad)
- 文字链广告(Textual Ad)
- 社交广告(Social Ad)
- 移动端广告(Mobile Ad)


## 搜索广告


## 定向广告


## 实时广告竞价


## 广告系统架构
