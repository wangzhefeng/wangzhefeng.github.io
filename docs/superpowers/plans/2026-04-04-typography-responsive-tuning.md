# Typography Responsive Tuning Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 在不改变站点整体视觉方向的前提下，优化中文/英文字体搭配、字号层级、窄屏自适应和手机端阅读体验。

**Architecture:** 仅调整主题样式层，集中修改 `themes/hugo-claudecode/assets/css/custom.css` 中的字体变量、正文与标题排版、阅读区宽度和关键断点规则。保持现有模板结构与组件布局不变，用变量和局部断点覆盖实现低风险优化。

**Tech Stack:** Hugo, CSS, npm, Pagefind

---

### Task 1: 字体与字号基础变量

**Files:**
- Modify: `themes/hugo-claudecode/assets/css/custom.css`

- [ ] 调整字体栈，优先适配中文长阅读与英文混排一致性。
- [ ] 微调 `body` 基础字号、行高、字重渲染和标题比例。

### Task 2: 正文阅读区与内容排版

**Files:**
- Modify: `themes/hugo-claudecode/assets/css/custom.css`

- [ ] 收紧阅读区宽度，优化正文、标题、副标题、列表、表格和代码块的阅读节奏。
- [ ] 保持首页与列表页视觉稳定，只做与阅读体验直接相关的微调。

### Task 3: 窄屏与手机端断点

**Files:**
- Modify: `themes/hugo-claudecode/assets/css/custom.css`

- [ ] 补强 `980px / 720px / 560px` 三段断点下的主内容宽度、区块内边距和按钮点击尺寸。
- [ ] 优化手机端正文、卡片、搜索结果、归档条目和代码块展示密度。

### Task 4: 验证

**Files:**
- Verify: `themes/hugo-claudecode/assets/css/custom.css`

- [ ] 运行 `npm run build` 验证 Hugo 构建与 Pagefind 索引生成。
- [ ] 检查是否出现 CSS 语法错误或构建中断。
