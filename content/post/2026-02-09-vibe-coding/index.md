---
title: vibe coding
author: wangzf
date: '2026-02-09'
slug: vibe-coding
categories:
  - AI
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
img {
    pointer-events: none;
}
</style>


<details><summary>目录</summary><p>

- [OpenAI: codex](#openai-codex)
- [Anthropic: Claude Code](#anthropic-claude-code)
- [Anthropic: Cowork](#anthropic-cowork)
- [Skywork](#skywork)
</p></details><p></p>

# OpenAI: codex

karpathy 形容 LLM 像是一把威力巨大的外星武器，但它没有说明书，每个人都得摸索着如何去使用它。
vibe coding 更是如此，发挥得好的时候，不需要干预就完成了一个复杂需求；发挥得不好的时候，像是在跟杠精对话。
尤其是有时候 LLM 根本也不听你在提示词里强调的那些东西。

1. 模型

最喜欢的还是 ChatGPT，最开始先跟 5.2 thinking 对需求、整理思路，有时候会直接让它给出 PRD 、需求清单和实现步骤。

实际 coding 推荐使用：GPT-5.2-codex（High）。GPT-5.2是模型版本，codex 意味着代码调优，推理等级设为 High；
只有当碰到疑难杂症 High 解决不了的时候，我才会尝试使用 extra-high 来解决。

ChatGPT 和 codex 最大的优点是订阅制，你不怎么需要去计算 token，放心大胆地去用。

2. 关键 `Agents.md` 

上来直接干也不是不行，但我建议先调教一下 agent，其实我推荐开始 vibe 之前先看一遍官方文档。
如果懒得看，可以照着我下面的教程操作：

* 基本上所有的 agent 都有自己的配置文件，codex 是 `AGENTS.md`，gemini 是 `GEMINI.md`，claude code 是 `CLAUDE.md`；
* 配置文件分为全局级别和项目级别，codex 的全局级别通常放在 `~/.codex` 目录，项目级别则在当前项目的根目录。
* 你也可以用 `AGENTS.override.md` 来覆盖项目级别的 `AGENTS.md`，但说实话我从来没这么做过。

接下来，`AGENTS.md` 里写什么是关键：

* 2.1 全局级别
    - 我强烈推荐 xuanwo 大佬的配置，定义了清晰的交互边界。
    - 当然，我也结合自己的习惯，开源了我自己的全局配置（https://github.com/tangwz/dotfiles），请随意取用。
* 2.2 项目级别
    - 这就要看具体的项目，可以看看 Github 的文章[《How to write a great agents.md: Lessons from over 2,500 repositories》](https://github.blog/ai-and-ml/github-copilot/how-to-write-a-great-agents-md-lessons-from-over-2500-repositories/)，
      学习怎么写。
    - 建议不要自己瞎写，先照着模板改一改，然后把官方文档、Best Practice 丢进去，根据不同项目拟定不同的规则。
      上面我的开源配置中也有一些我的前后端配置。
* 2.3 迭代 `AGENTS.md` 
    - 持续优化 `AGENTS.md` 也很关键，这里最简单的办法就是，把你在和大模型沟通过程中发现它犯的错，直接添加到 AGENTS.md 里去，避免下次再重复。

3. 原子化工作流

有了模型、配置文件和你的需求，剩下的就是 vibe。基本上 Vibe Coding 大家都遵守着一些最佳实践：

* 3.1 及时 Commit
    - AI 随时可能发疯，如果它在下一轮对话中把代码改崩了，及时 commit 可以保证“存档点”。
* 3.2 一轮对话 等于 一个需求
    - 每个需求，开启一轮新的对话。不要一下子塞给它一堆问题。让 AI 聚焦于当下的任务。上下文越短，
      注意力越集中，出错率越低。完成一个 Feature，Close 掉，开启下一轮对话。
* 3.3 该断则断（Reset）
    - 当你发现 AI 开始来回说车轱辘话、逻辑鬼打墙、或者死活改不对一个 Bug 时——不要纠缠！
      不要试图说服它！重新开启一轮新的对话。
* 3.4 测试驱动开发
    - 说实话，一行行看 vibe coding 出来的代码不现实，但是你总要把控软件质量，这时候可以回归传统的软件工程：TDD。
    - 当然，测试用例也可以丢给 codex 写，但最好自己把把关。

4. 写在最后

目前 vibe coding 和代码自动完成之类的东西，最大的风险和担忧在于，人类会逐渐失去对代码的理解力、洞察力和掌控力。
但实际上，随着软件复杂度的提高，后者在 vibe coding 大爆发之前就已经在发生了，只是现在会更严重更极端。

Vibe Coding 本质上是一种人机协作的范式。

* 用模型代替你的大脑去思考细节；
* 用 `agents.md` 代替你的记忆去规范细节；
* 用原子化工作流代替你的双手去控制质量。

无论如何，拥抱它而不是拒绝它，用 Redis 之父的话来说，Don't fall into the anti-AI hype，
LLM 已经能在多数场景下高效完成编程工作，从修 bug、重构到实现中型项目，效率远超人类。

# Anthropic: Claude Code

Claude Code 官方推出的[入门课程](https://anthropic.skilljar.com/claude-code-in-action)，免费！有空补学习一下

内容包括：

1. 使用 Claude Code 的核心工具进行文件操作、命令执行和代码分析。
2. 通过 `/init`、`Claude.md` 文件以及 (`@`) 引用来高效管理上下文。
3. 利用多种快捷键和命令来控制对话流程。
4. 在需要更深入分析的复杂任务中启用 Plan Mode 和 Thinking Mode。
5. 创建自定义命令，用于自动化重复性的开发工作流。
6. 通过 MCP 服务器扩展 Claude Code，引入浏览器自动化等能力。
7. 配置 GitHub 集成，实现自动化的 PR 审查和 Issue 处理。
8. 编写 hooks，为 Claude Code 添加额外的行为和能力。

# Anthropic: Cowork

* [Getting started with Cowork](https://support.claude.com/en/articles/13345190-getting-started-with-cowork)

# Skywork

* [Skywork 官网](https://skywork.ai/desktop?utm_source=google&utm_medium=919-995-8647&utm_campaign=23061180190&utm_term=skywork&utm_group=186186660939&utm_creative=775867784835)
