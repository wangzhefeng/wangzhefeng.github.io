# Article Layout And Table Fix Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 修复所有 Markdown 文章中正文与右侧目录重叠的问题，并统一表格与其他正文内容的宽度约束和渲染风格。

**Architecture:** 保持现有单篇文章模板结构不变，只在 `themes/hugo-claudecode/assets/css/custom.css` 中调整文章页三栏网格、正文列约束、目录列约束，以及 Markdown 表格/长内容的溢出规则。通过 CSS 约束解决布局重叠，并强化表格在正文中的卡片式渲染。

**Tech Stack:** Hugo, CSS, npm, Pagefind

---

### Task 1: 文章页双栏约束

**Files:**
- Modify: `themes/hugo-claudecode/assets/css/custom.css`

- [ ] 约束单篇文章页正文列和目录列宽度，避免正文被大内容撑向右侧目录。
- [ ] 收紧 `article-frame` 与 `page-rail` 的宽度和最小宽度规则。

### Task 2: Markdown 内容溢出治理

**Files:**
- Modify: `themes/hugo-claudecode/assets/css/custom.css`

- [ ] 统一正文里的表格、代码块、图片、iframe、链接和长文本的宽度与换行策略。
- [ ] 保持表格在正文阅读区内横向滚动，并沿用 Claude Code 风格的低对比浅面板。

### Task 3: 验证

**Files:**
- Verify: `themes/hugo-claudecode/assets/css/custom.css`

- [ ] 运行 `npm run build`，确认 Hugo 与 Pagefind 构建通过。
