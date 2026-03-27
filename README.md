[![Netlify Status](https://api.netlify.com/api/v1/badges/72676f40-4e3e-4b9d-81d9-843759ddd081/deploy-status)](https://app.netlify.com/sites/wangzf/deploys)

Personal website

## Theme switching

The site now ships with two Hugo themes:

- `hugo-claudecode`: the default docs-style theme used by the current site
- `hugo-ivy`: the original upstream theme kept as a fallback/reference

To switch themes, edit [config.yaml](/Users/wangzf/wangzhefeng.github.io/config.yaml) and change:

```yaml
theme: hugo-claudecode
```

to:

```yaml
theme: hugo-ivy
```

Theme-specific templates and assets for the current site design now live in:

- [themes/hugo-claudecode/layouts](/Users/wangzf/wangzhefeng.github.io/themes/hugo-claudecode/layouts)
- [themes/hugo-claudecode/static](/Users/wangzf/wangzhefeng.github.io/themes/hugo-claudecode/static)

Root-level `static/` now keeps only site-wide assets such as favicons, verification files, media, and `rmarkdown-libs`.
