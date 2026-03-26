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
- [Anthropic: Claude Skills](#anthropic-claude-skills)
    - [Skill 介绍](#skill-介绍)
        - [Skills 是什么：从概念来源到运作原理](#skills-是什么从概念来源到运作原理)
        - [如何理解 Skill](#如何理解-skill)
    - [Skill 使用](#skill-使用)
        - [安装 Claude Code](#安装-claude-code)
        - [Claude Code 模型](#claude-code-模型)
            - [MiniMax-M2 模型](#minimax-m2-模型)
        - [安装并使用 Skills](#安装并使用-skills)
        - [如何找到好用的 Skills ?](#如何找到好用的-skills-)
    - [制作 Skill](#制作-skill)
    - [什么时候应该用 Skills](#什么时候应该用-skills)
        - [1.发现自己在向 AI 反复解释同一件事](#1发现自己在向-ai-反复解释同一件事)
        - [某些任务需要特定知识、模板、材料才能做好](#某些任务需要特定知识模板材料才能做好)
        - [发现一个任务要多个流程协同完成](#发现一个任务要多个流程协同完成)
    - [资料](#资料)
- [Skywork](#skywork)
</p></details>

## OpenAI: codex

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

## Anthropic: Claude Code

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

## Anthropic: Cowork

* [Getting started with Cowork](https://support.claude.com/en/articles/13345190-getting-started-with-cowork)

## Anthropic: Claude Skills

### Skill 介绍

> 巧借通用 Agent 内核，只靠 Skills 设计，就能低成本创造具有通用 AI 智能上限的垂直 Agent 应用。
> 
> 一个好 Skill 能发挥的智能效果，甚至能轻松等同、超越完整的 AI 产品。任何不懂技术的人，都能开发属于自己的 Skills。

#### Skills 是什么：从概念来源到运作原理

2025 年 10 月中旬，Anthropic 正式发布 Claude Skills。
两个月后，Agent Skills 作为开放标准被进一步发布，意在引导一个新的 AI Agent 开发生态。
OpenAI、Github、VS Code、Cursor 均已跟进。

为了更好的理解，你可以把 Skills 理解为“通用 Agent 的扩展包”：
Agent 可通过加载不同的 Skills 包，来具备不同的专业知识、工具使用能力，稳定完成特定任务。

最常见的疑惑是：这和 MCP 有什么区别？

* MCP 是一种开放标准的协议，关注的是 AI 如何以统一方式调用外部的工具、数据和服务，本身不定义任务逻辑或执行流程。
* Skill 则教 Agent 如何完整处理特定工作，它将执行方法、工具调用方式以及相关知识材料，
  封装为一个完整的能力扩展包，使 Agent 具备稳定、可复用的做事方法。

以 Anthropic 官方 Skills 为例：

* `PDF`：包含 PDF 合并、拆分、文本提取等代码脚本，教会 Agent 如何处理 PDF 文件 - 提取文本，
  创建新的 PDF、合并或拆分文档。
* `Brand-guidelines`：包含品牌设计规范、Logo 资源等，Agent 设计网站、海报时，
  可参考 Skill 内的设计资源，自动遵循企业设计规范。
* `Skill-Creator`：把创建 Skill 的方法打包成元 Skill，让 AI 发起 Skill 创建流程，
  引导用户创建出符合需求的高水准 Skill。

但 Skills 的价值上限，远不止于此。它应该是一种极其泛用的新范式，从垂直 Agent 到 AI 产品开发：
借用通用 Agent 内核，零难度创造具备通用 AI 智能的垂直 Agent 应用。

#### 如何理解 Skill

Anthropic 说：Skills 是模块化的能力，扩展了 Agent 的功能。
每个 Skill 都打包了 LLM 指令、元数据、可选资源（脚本、模板等），
Agent 会在需要时自动使用他们。

更直观的解释：Skill 就像给 Agent 准备的工作交接 SOP 大礼包。想象你要把一项工作交给新同事，若不准口口相传，
只靠文档交接（而且你想一次性交接完成，以后不被打扰），你会准备什么？

* 任务的执行 SOP 与必要背景知识（这件事大致怎么做）
* 工具的使用说明（用什么软件、怎么操作）
* 要用到的模板、素材（历史案例、格式规范）
* 可能遇到的问题、规范、解决方案（细节指引补充）

Skill 的设计架构，几乎是交接大礼包的数字版本：

![img](images/skill2.png)

在 Skill 中，指令文档用于灵活指导，代码用于可靠性调用，资源用于事实查找与参考。

当 Agent 运行某个 Skill 时，就会：

1. 以 `SKILL.md` 为第一指引
2. 结合任务情况，判断何时需要调用代码脚本（`scripts`）、翻阅参考文档（`references`）、使用素材资源（`assets`）
3. 通过“规划-执行-观察”的交错式反馈循环，完成任务目标

当然，Skill 也可以用来扩展 Agent 的工具、MCP 使用边界，通过文档与脚本，
也可以教会 Agent 连接并使用特定的外部工具、MCP 服务。

### Skill 使用

#### 安装 Claude Code



```bash
$ claude --version
```

#### Claude Code 模型

现在大部分国产模型都已经支持了 Skill 的使用与创建。

##### MiniMax-M2 模型

替换 MiniMax-M2 模型，可以在终端内输入：

```
# minimax-m2.sh
export ANTHROPIC_BASE_URL=https://api.minimaxi.com/anthropic
export ANTHROPIC_AUTH_TOKEN=【换成你的 API KEY】
export ANTHROPIC_SMALL_FAST_MODEL=MiniMax-M2
export ANTHROPIC_DEFAULT_SONNET_MODEL=MiniMax-M2
export ANTHROPIC_DEFAULT_OPUS_MODEL=MiniMax-M2
export ANTHROPIC_DEFAULT_HAIKU_MODEL=MiniMax-M2
claude
```

该操作在当前终端窗口中，将要用的模型临时改为目标模型。关掉该窗口后，则需再次发送该命令，重新指定模型 API 与 Key。

> MiniMax Coding Plan 的 API Key，
> 可以到 https://platform.minimaxi.com/user-center/payment/coding-plan 获取。
> 
> Coding Plan 总共有 3 档，分别是 9.9（首月）/49/119 ，每 5 小时提供 40/100/300 次 Prompt 额度。
> 大概是 Claude 原本模型的 8% 价格，整体 TPS > 100，实际体感生成速度很快。
> 
> 开始前记得在这里订阅套餐 https://platform.minimaxi.com/subscribe/coding-plan
>
> 轻量任务选 Starter 就行。
>
> M2 宣称有极强的长文本处理，以及在复杂任务中的自我纠错和任务恢复能力。


#### 安装并使用 Skills

> - (1) 正式使用 Claude Code 之前，建议在任意目录下创建一个空文件夹，比如叫test，再在终端内切换到对应文件目录；
> - (2) 然后在终端输入 `claude`，就可以启动 CC 了；
> - (3) 这一步能把 Claude Code 的后续 AI 行为，都局限在该目录，减小对本地电脑其他文件的影响。

1. 在安装 Skill 之前，你需要先获取需要的 Skill 文件包。
    - 比如官方 Skills 仓库：https://github.com/anthropics/skills/tree/main，里面就有很多已经做好的 Skills。
    - 你可以让 Claude Code 替你自动安装 Skill，比如在 CC 中发送 `安装 skill，skill 项目地址为：<skill 项目地址>`。
2. 也可以手动下载 Skill，把文件包解压后，放在 skills 安装目录下：
    - 可以在当前项目文件夹的 `/.claude/skills/` 目录下，放入要安装的 skill 文件包

    ![img](images/skill1.png)

    - 也可以选择全局目录 `~/.claude/skills/`（所有项目都能共享放在全局目录的 Skill）
3. 完成安装后，记得重启 Claude Code 退出终端再打开就行，或者双击 `ctrl+c` 终止 Claude Code 进程）
4. 要使用 Skill 时，
    - 只要在装好后的 Claude Code 中，发送 `开始使用 <skill 名称>`；
    - 或者用户消息与 skill 元数据的描述匹配，就能自动调用 Skills，执行任务。

#### 如何找到好用的 Skills ?

在面向 to C 用户（也就是自己日常使用）时，以上的方法有两个问题：

1. 使用步骤确实比日常的 APP 复杂不少
2. 比较难找到想用的 Skills

**常规方法** 是找规模比较大的第三方 Skills 市场：https://skillsmp.com/zh。
但不难发现，现有大部分的 Skills 公开市场，没有完善的评价和精选体系，
所有 Skill 缺少合理的分类与排序机制，导致很难找到需要的 Skills。
可以看到仅靠 star 排序，是非常难找到合适的精选 Skill 的。

**Mulerun** 最近就在研究解决这个问题，

* 在打造全球性的 Agent 市场，支持创作者在平台上开发并上架 Skill、N8N 等形式的 AI Agent
* 会帮助 Agent 创作者做全球分发、增长（类似 Agent 向的 APP Store），且上架 Mulerun 后，
  Agent 能被其他用户付费使用
* Mulerun 也即将支持 Agent Skills 生态，还会有个好功能：
  一键运行并测试 github 上公开的 skill repo（也就是省掉了 Claude Code 那样配置 Skill 的步骤）
* 另外，还会引入自动评分、精选的 Skills 发现机制，帮助用户能够更好地找到自己需要的优质 Skills

### 制作 Skill

如果你按照上文，学会了 Skill 安装与使用，那制作第一个 Skill 将会无比容易。
我们需要用到 Anthropic 官方的一个 skill：`skill-creator`
顾名思义，用来帮你自动开发 Skill 的 Skill。

1. 首先是安装 `skill-creator`，
   skill 项目地址在：https://github.com/anthropics/skills/tree/main/skills/skill-creator，
   安装过程请 Claude Code 来帮忙自动安装：

```bash
$ 安装 skill, skill 项目地址为：https://github.com/anthropics/skills/tree/main/skills/skill-creator
```

2. 安装完成后，即可调用 skill-creator 自动创建需要的 skill。比如，发送创建需求给 Claude Code：
    - `创建新的 skill，能自动吧用户指定的 pdf 转成 word 文档`，
      Claude Code 自动调用 `skill-creator`，编写 `SKILL.md` 与 `pdf2word 脚本`
    - `创建 skill，能按照我写文章的行文风格写文章`
    - `创建 skill，能自动整理近期 XX 领域的新闻日报`
3. 安装自己做好的 skill
    - 上述方式做出来的 skill，会默认是 `xx.skill` 格式，会与 zip 或文件夹格式略有区别。
      是 `skill-creator` 创建的 skill 压缩格式。直接使用 Claude Code 安装即可；
        - `安装 skill，地址：...`
    - 如果是文件夹或者 zip，那就按上文的介绍，手动解压放到对应 skills 目录即可。

### 什么时候应该用 Skills

什么场景值得“用 Skill 来解决”、“开发一个 Skill”？
这个问题对于普通用户优化 AI 工作流程，开发者找 Skills Agent 创业机会，同样重要。
根据 Anthropic 官方博客建议，与实际理解，梳理了 3 种明显的时机：

#### 1.发现自己在向 AI 反复解释同一件事

最典型的信号是：为了完成某个任务，在多轮对话中，需要不断向 AI 解释一件事应该怎么做。

比如：

```
“帮我写一份技术文档”
“不对，我们公司的技术文档格式是这样的……”
“还有，代码示例要按这个模板来……”
“上次不是说了吗，章节标题要三级标题……”
“帮我分析这个数据”
“先把 ＞ XX 的异常值筛掉”
“不对，应该用中位数，不是平均值”
“图表要按我们公司文档的配色方案……”
```

这时候就该想到：与其每次都解释一遍，不如把这些规则打包成一个 Skill，一次创建永久复用。

#### 某些任务需要特定知识、模板、材料才能做好

有时候是 AI 的通用能力够了，但缺“特定场景的知识材料”。

典型场景：

* 技术文档写作：需要参考代码规范、术语表，使用文档模板
* 品牌设计：需要参考品牌手册、色彩规范，使用 Logo 资源
* 数据分析：需要参考指标定义、计算公式，使用报表模板……

这些都是 `通用 Agent + 垂直知识` 的典型场景：人提供材料，Agent 才能具备场景 Context。

在 Skill 包里放对应的知识材料，比如把模板、规范、案例放到 Skill 的 `assets/`、`reference/` 目录，
或者直接描述在 `SKILL.md` 中，Agent 就能一次性输出符合任务需要的精准结果。

#### 发现一个任务要多个流程协同完成

有些任务更加复杂，往往需要“组合多个流程”才能完成。

* 竞品分析报告：检索竞品数据 + 数据分析 + 制作 PPT
* 内容生产：收集参考资料 + 学习风格 + 大纲协作 + 正文写作

把这类任务中每个环节的指令文档、可执行脚本、参考材料、可用资源打包成单个或多个 Skill 也是不错的 AI 解决方法。

让Agent 根据任务描述，智能调用不同的 Skill 模块，通过“规划-执行-观察”的交错式行动，
一次性完成原本需要多个流程协同完成的复杂任务。

### 资料

* [Agent Skills 终极指南：入门、精通、预测](https://zhuanlan.zhihu.com/p/1992272492392380044?share_code=2ubO4NqsZxWB&utm_psn=2000838846183674669)
    * https://skillsmp.com/zh
    * https://github.com/anthropics/skills/tree/main/skills/skill-creator
    * https://github.com/JimLiu/baoyu-skills/tree/main
* [A complete guide to building skills for Claude](https://claude.com/blog/complete-guide-to-building-skills-for-claude)
* [The Complete Guide to Building Skills for Claude](https://resources.anthropic.com/hubfs/The-Complete-Guide-to-Building-Skill-for-Claude.pdf?hsLang=en)
* [Anthropic 首个公开的 Skills 构建指南](https://mp.weixin.qq.com/s/PcyKi5q8zT-tJ_9rzgKSqg)





## Skywork

* [Skywork 官网](https://skywork.ai/desktop?utm_source=google&utm_medium=919-995-8647&utm_campaign=23061180190&utm_term=skywork&utm_group=186186660939&utm_creative=775867784835)
