# CLAUDE.md

本文件为 Claude Code 在本仓库中的工作说明；如与根目录 `AGENTS.md` 冲突，以 `AGENTS.md` 为准。

## 项目现状

这是一个部署到 Netlify 的 Hugo 中文站点，当前主题固定为 `hugo-claudecode`。仓库不再维护多主题切换逻辑，主题方向与协作约束统一收敛到根目录 `AGENTS.md`。站点定位不是传统时间流博客，而是更接近文档站和个人知识库：强调结构化浏览、搜索、长期写作和轻量交互。

## 关键目录

- `content/post/`：日志文章
- `content/note/`：知识库笔记
- `content/tool/`：工具、教程、参考页
- `content/pomes/`、`content/resume/`：独立栏目
- `themes/hugo-claudecode/layouts/`：主题模板
- `themes/hugo-claudecode/assets/css/`：主题样式
- `themes/hugo-claudecode/assets/js/site.js`：站点前端交互
- `static/`：favicon、媒体、验证文件、`rmarkdown-libs`
- `scripts/`：辅助脚本，如图片检查和图标生成

## 构建与运行

常用命令：

```bash
npm install
npm run build
npm run search:local
npm run build:local-search
npm run check:images
```

- `npm run build`：正式构建，执行 `hugo` 并为 `public/` 生成 Pagefind 索引。
- `npm run search:local`：将搜索索引写入 `static/pagefind`，用于本地预览。
- `npm run build:local-search`：同时生成正式和本地两套搜索输出。
- `npm run check:images`：检查内容图片引用是否存在。

Netlify 使用 `npm ci && npm run build`；分支预览和 deploy preview 使用 `hugo -F -b $DEPLOY_PRIME_URL && npx pagefind --site public`。Hugo 版本在 `netlify.toml` 和 `.Rprofile` 中统一为 `0.134.1`。本地 R/blogdown 预览通常在 `http://localhost:4321/`。

## 主题实现要点

- 主题模板集中在 `themes/hugo-claudecode/layouts/`，主页与栏目页主要由 `_default/list.html` 驱动，文章页由 `_default/single.html` 驱动。
- `themes/hugo-claudecode/assets/js/site.js` 负责 Pagefind 搜索、明暗主题切换、界面文案中英切换、代码块交互、视图切换高亮等前端行为。
- 主题样式集中在 `themes/hugo-claudecode/assets/css/custom.css`。
- 设计方向保持文档站式阅读体验，不要为追求“更炫”引入重型前端框架、额外状态管理或恢复多主题兼容层。

## 内容与配置约定

- 新内容默认使用中文。
- 内容目录一般采用 `content/<section>/<yyyy-mm-dd-slug>/index.md`，配图放在同级 `images/`。
- 新增需要进入顶层导航的栏目时，同步更新 `config.yaml` 的 `menu.main`。
- 搜索由 Pagefind 提供，评论系统使用 Utterances，界面支持中英切换，但正文并未实现完整 Hugo 多语言。

## 历史 Session 沉淀

- 主题已从历史主题体系中抽离并收敛为单主题维护。
- `.gitattributes` 已修复图片二进制规则：`png/jpg/jpeg/gif/webp/bmp/ico/pdf` 必须按二进制处理。
- 仓库中仍可能存在历史损坏图片；遇到异常图片时，优先恢复原始资源或重新导出，其次才考虑 SVG 示意替代。
- `themes/hugo-claudecode/`、`config.yaml`、`netlify.toml`、`.Rprofile` 是高影响区域，修改后要补做完整构建验证。

## 工作规则

- 优先直接编辑现有主题和内容，不做无关重构。
- 不覆盖用户已有未提交改动。
- 内容改动至少运行 `npm run check:images`；主题、模板、样式或脚本改动至少运行 `npm run build`。
- 如果只修改协作文档，可不跑站点构建，但要明确说明未做运行验证。
