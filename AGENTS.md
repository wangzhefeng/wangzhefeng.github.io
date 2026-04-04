# Repository Guidelines

## Project Structure & Module Organization

本仓库是一个基于 Hugo 的中文个人知识库与博客，当前只维护 `hugo-claudecode` 单一主题。核心内容位于 `content/`：`content/post/` 为日志文章，`content/note/` 为知识库笔记，`content/tool/` 为工具与教程，`content/pomes/` 和 `content/resume/` 为独立栏目。主题模板位于 `themes/hugo-claudecode/layouts/`，样式与脚本位于 `themes/hugo-claudecode/assets/`，站点级静态资源位于 `static/`，辅助脚本位于 `scripts/`。

## Build, Test, and Development Commands

- `npm install`：安装本地依赖；CI 和 Netlify 使用 `npm ci`。
- `npm run build`：执行 `hugo && npx pagefind --site public`，用于正式构建。
- `npm run search:local`：将 Pagefind 本地索引输出到 `static/pagefind`。
- `npm run build:local-search`：同时生成正式索引与本地索引。
- `npm run check:images`：检查内容中的图片引用。

内容改动至少运行一次图片检查；主题、模板、样式或脚本改动至少运行一次 `npm run build`。

## Coding Style & Naming Conventions

保持小改动，避免无关重排。内容页通常采用 `content/<section>/<yyyy-mm-dd-slug>/index.md` 结构，配图放在同级 `images/`。模板、partials、CSS、JS 文件沿用当前命名方式：小写、语义明确，例如 `page_render_data.html`、`site.js`。新内容默认使用中文。主题方向保持“文档站式阅读体验”，不要引入重型前端框架或额外多主题兼容逻辑。

## Testing Guidelines

仓库没有独立单元测试，验证以构建和资源检查为主。重点检查 front matter、相对资源路径、Pagefind 搜索、主题切换、语言切换、代码块交互和评论区样式。涉及图片修复时，优先恢复原始资源或重新导出；仅在原图不可恢复时，才使用 SVG 替代。

## Commit & Pull Request Guidelines

近期提交同时存在 `update` 和更清晰的约定式写法，例如 `feat(theme): extract hugo-claudecode theme and refresh site assets`。建议优先使用 `<type>(<scope>): <summary>`。PR 需说明影响范围、验证命令；涉及视觉改动时附截图或预览链接。

## Agent-Specific Notes

历史 Codex 会话已沉淀出几条稳定约束：主题决策统一维护在本文件；仓库已经收敛为单主题维护；`.gitattributes` 已显式将 `png/jpg/jpeg/gif/webp/bmp/ico/pdf` 标记为二进制，遇到坏图先判断是否为历史损坏资源；`themes/hugo-claudecode/`、`config.yaml`、`netlify.toml`、`.Rprofile` 属于高影响区域，修改后应补做完整构建验证。优先读取本地文件，不覆盖用户已有未提交改动。
