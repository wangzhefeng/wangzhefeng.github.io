# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Chinese-language personal knowledge base and blog built with **Hugo**, deployed on **Netlify** at https://wangzhefeng.com/. Covers statistics, time series analysis, machine learning, deep learning, NLP, LLM, computer vision, and operations research. Also includes blog posts, Chinese poetry, tool guides, a resume, and a TODO list.

Built with R package **blogdown** (Hugo-based static site generator). The `hugo-ivy` base theme has been heavily customized into an Anthropics-style documentation layout (design style: `anthropic-docs`).

## Development

The dev server is started via R/blogdown and runs at **http://localhost:4321/**. The site auto-rebuilds on file changes — open this URL to preview rendering in real time.

```bash
# Full production build (Hugo + Pagefind search index)
npm run build

# Build + generate search index for local dev preview
npm run build:local-search
```

Hugo version is **0.134.1** (pinned in `netlify.toml` and aligned in `.Rprofile`). Local `blogdown` preview should use the same Hugo major/minor version as production builds.

## Architecture

### Template Layer
`layouts/` overrides `themes/hugo-ivy/layouts/`. Key templates:
- `layouts/_default/list.html` -- Section listing with docs-layout (sidebar navigation + section cards + archive list)
- `layouts/_default/single.html` -- Single page with docs-layout (sidebar + TOC rail + comments)
- `layouts/partials/header.html` -- Site header (brand, nav, search trigger with Cmd+K, theme/locale toggles, mobile nav)
- `layouts/partials/docs_sidebar.html` -- Left sidebar with section tree navigation
- `layouts/partials/footer.html` -- Footer with Pagefind search modal

### Design System
CSS custom properties in `static/css/custom.css` with light/dark theme variants. Configured via `params.design.style = "anthropic-docs"` in `config.yaml`.

### Client-Side Features
All in `static/js/site.js`:
- Pagefind static search (built post-Hugo from `public/`)
- Light/dark theme toggle with localStorage persistence
- Chinese/English locale switching (UI strings only, not content)
- Scroll-reveal animations via `[data-reveal]` attributes
- Code copy buttons, heading anchor links, prev/next keyboard navigation

### External Integrations
- **Comments**: Utterances (GitHub issues), config in `params.utteranc`
- **Syntax highlighting**: highlight.js 11.9.0 from bootcss CDN
- **Math rendering**: MathJax via CDN
- **Analytics**: Google Analytics via Hugo internal template

## Content Structure

- `content/post/` -- Blog posts (date-prefixed dirs, e.g. `2022-03-14-hugo-learning/index.md`)
- `content/note/` -- Technical knowledge base (subdirs: `algorithms`, `analysis`, `control_algorithms`, `cv`, `deeplearning`, `llm`, `machinelearning`, `nlp`, `operationsresearch`, `timeseries`)
- `content/tool/` -- Tool guides (python, pytorch, tensorflow, docker, matlab, etc.)
- `content/pomes/` -- Chinese classical poetry
- `content/resume/` -- Personal resume
- `content/about.md`, `content/todo.md` -- Standalone pages
- `content/Ivy/` -- Theme design documentation and changelog

## Content Conventions

- Each content piece is a date-prefixed directory containing an `index.md`
- Frontmatter: `title`, `author`, `date`, `slug`, `categories`, `tags`
- Set `comment: false` in frontmatter to disable comments on a page
- Set `disable_highlight: true` or `disable_mathjax: true` to disable those features per-page
- The `<!--# ON_HOLD -->` marker truncates content display for old posts
- Permalinks: `/post/:year/:month/:day/:slug/` and `/note/:year/:month/:day/:slug/`
- All content is in Chinese (zh-cn); UI has partial English locale support

## Key Rules

- **Do NOT modify files inside `themes/hugo-ivy/`**. Always use `layouts/` overrides.
- New content should be written in Chinese (zh-cn).
- When adding new content sections, add the corresponding entry to `menu.main` in `config.yaml`.
- `public/`, `resources/`, and `static/pagefind/` are git-ignored (build artifacts).
- Run `npm run build` to verify changes before committing.
- The dev server at http://localhost:4321/ is always available for live preview — check it after making changes.
