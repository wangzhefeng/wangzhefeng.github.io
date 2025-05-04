---
title: MCP
author: wangzf
date: '2025-04-03'
slug: mcp
categories:
  - AI
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

- [什么是 MCP](#什么是-mcp)
    - [MCP 架构](#mcp-架构)
- [MCP 环境配置](#mcp-环境配置)
    - [Stdio 配置](#stdio-配置)
    - [SSE 配置](#sse-配置)
    - [获取 MCP](#获取-mcp)
    - [配置 MCP-Windsurf](#配置-mcp-windsurf)
- [支持 MCP 服务的客户端](#支持-mcp-服务的客户端)
</p></details><p></p>

# 什么是 MCP

MCP（Model Context Protocol，模型上下文协议）是由 Anthropic 推出的开源协议，
旨在实现大型语言模型（LLM）与外部数据源和工具的无缝集成，用来在大模型和数据源之间建立安全双向的链接。

MCP 的目标是成为 AI 领域的“HTTP 协议”，推动 LLM 应用的标准化和去中心化。

> 简单来说 LLM 使用不同工具时，以前需要同时修改模型和工具，因为各工具的 API 数据格式不统一，
> 导致适配成本高、功能添加慢。
> 
> MCP 协议统一了数据格式标准，规定了应用向 LLM 传输数据的方式。任何模型只要兼容 MCP 协议，
> 就能与所有支持 MCP 的应用交互。这将适配工作从双向简化为单向（仅应用端），且对于已有 API 的应用，
> 第三方开发者也可基于其 API 进行 MCP 封装适配，无需官方支持。

## MCP 架构





# MCP 环境配置

MCP 现在一共有两种模式：

* Stdio：主要用在本地服务上，操作本地的软件或者本地的文件，比如 Blender 这种就只能用 Stdio 因为他没有在线服务；
* SSE ：主要用在远程服务上，这个服务本身就有在线的 API，比如访问谷歌邮件，谷歌日历等。

## Stdio 配置

* `uvx`: `uv`
    - Windows:

    ```bash
    $ powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
    ```

    - MacOS:

    ```bash
    $ curl -LsSf https://astral.sh/uv/install.sh | sh
    ```

* `npx`: [Node.js](https://nodejs.org/)

## SSE 配置

SEE 的配置方式非常简单基本上就一个链接就行，如果找到的是 SEE 的直接复制链接填上就行，
而且现在使用 SEE 配置的 MCP 非常少，基本上都是 Stdio 的方式。

## 获取 MCP 

MCP 聚合网站：

1. https://mcp.so/
2. https://smithery.ai

## 配置 MCP-Windsurf


# 支持 MCP 服务的客户端

* 聊天客户端
    - Claude
    - Chatwise
    - CherryStudio
* AI 编程 IDE
    - Cursor
    - Windsurf

