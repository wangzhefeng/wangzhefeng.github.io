[![Netlify Status](https://api.netlify.com/api/v1/badges/72676f40-4e3e-4b9d-81d9-843759ddd081/deploy-status)](https://app.netlify.com/sites/wangzf/deploys)

Personal website

## Theme

The site uses a single Hugo theme:

- `hugo-claudecode`: the docs-style theme used by the current site

Theme-specific templates and assets for the current site design now live in:

- [themes/hugo-claudecode/layouts](/Users/wangzf/wangzhefeng.github.io/themes/hugo-claudecode/layouts)
- [themes/hugo-claudecode/assets](/Users/wangzf/wangzhefeng.github.io/themes/hugo-claudecode/assets)

Theme-specific shortcodes also live under `themes/hugo-claudecode/layouts/shortcodes/`.

Implementation notes for the current theme:

- CSS and JS are built through Hugo Pipes from `themes/hugo-claudecode/assets/`
- Markdown code fences are highlighted by Hugo/Chroma instead of runtime `highlight.js`
- Navigation/sidebar and page render cleanup use cached partials such as `docs_sidebar_data.html` and `page_render_data.html`

Root-level `static/` now keeps only site-wide assets such as favicons, verification files, media, and `rmarkdown-libs`.
