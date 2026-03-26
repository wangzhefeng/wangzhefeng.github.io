(() => {
  const body = document.body;
  const root = document.documentElement;
  const header = document.querySelector("[data-site-header]");
  const menuToggle = document.querySelector("[data-menu-toggle]");
  const mobileNav = document.querySelector("[data-mobile-nav]");
  const searchModal = document.querySelector("[data-search-modal]");
  const searchInput = document.querySelector("[data-search-input]");
  const searchState = document.querySelector("[data-search-state]");
  const searchResults = document.querySelector("[data-search-results]");
  const themeColorMeta = document.querySelector("[data-theme-color]");
  const utterancesRoot = document.querySelector("[data-utterances-root]");
  const openSearchButtons = document.querySelectorAll("[data-open-search]");
  const closeSearchButtons = document.querySelectorAll("[data-close-search]");
  const themeToggles = document.querySelectorAll("[data-theme-toggle]");
  const localeToggles = document.querySelectorAll("[data-locale-toggle]");
  const localeLabels = document.querySelectorAll("[data-locale-label]");
  const themeIcons = document.querySelectorAll("[data-theme-toggle-icon]");
  const themeLabels = document.querySelectorAll("[data-theme-toggle-label]");
  const tocLinks = Array.from(document.querySelectorAll(".toc-shell a[href^='#']"));
  const revealNodes = document.querySelectorAll("[data-reveal]");
  const themeKey = "wzf-theme";
  const localeKey = "wzf-locale";
  let pagefindModule = null;
  let searchSequence = 0;

  const messages = {
    "zh-CN": {
      localeLabel: "简体中文",
      localeSwitch: "切换语言",
      searchDocs: "搜索...",
      menu: "菜单",
      switchDark: "切换到深色模式",
      switchLight: "切换到浅色模式",
      searchSite: "搜索站内内容",
      close: "关闭",
      searchPlaceholder: "搜索...",
      escToClose: "",
      searchIntro: "",
      searching: "",
      noResults: "没有找到相关结果。你可以尝试更短的关键词，或者从栏目页继续浏览。",
      foundResults: () => "",
      searchUnavailable: "搜索资源暂时不可用。请先执行完整构建以生成 Pagefind 索引。",
      openPage: "打开页面",
      untitledPage: "未命名页面",
      copy: "复制",
      copied: "已复制",
      failed: "失败",
      headingAnchor: "链接到此标题",
      onThisPage: "在此页面",
      pageInfo: "页面信息",
      updatedOn: "更新于",
      backTo: "返回",
      homeHeroTitle: "统计学、机器学习、深度学习、时间序列、运筹学与大模型的个人知识库。",
      homeHeroLead: "这个站点是一个持续维护的个人知识库，而不是只按时间滚动的博客首页。内容围绕统计、时间序列、机器学习与大模型展开，强调结构化浏览、快速检索和长期复用。",
      homeStoryIntro: "2017 年我从上海大学数学系毕业后，开始了在上海工作、生活的日子。工作之余一般在家听音乐、看书、睡觉，偶尔在这里写文章、写笔记，记录生活，记录学习。",
      homeGithubIntro: "这是我的 GitHub 贡献图，看看我在忙些啥：",
      topicTimeseriesTitle: "时间序列",
      topicTimeseriesDesc: "从模型、评估到深度学习专题的完整入口",
      topicMlTitle: "机器学习",
      topicMlDesc: "围绕监督学习、特征工程和集成方法组织",
      topicLlmTitle: "大模型",
      topicLlmDesc: "从基础概念、模型、微调到工程接口",
      browseByType: "按内容类型浏览",
      browseByTypeDesc: "保留接近文档站的入口逻辑：先选内容类型，再进入具体专题或页面。",
      notesDesc: "系统化整理统计、时序、机器学习、运筹优化与 LLM 内容。",
      toolsDesc: "偏向文档、教程和速查，适合直接作为工作时的参考页。",
      postsDesc: "以时间为线索的文章输出，适合浏览最近的思考、项目和记录。",
      aboutDesc: "作者说明、主题方向与站点结构，适合第一次进入时快速了解。",
      profile: "Profile",
      deepDiveCore: "从核心专题继续深入",
      deepDiveDesc: "专题页会继续展开子栏目、按主题阅读路径和时间归档。",
      openTopic: "Open topic",
      recentUpdates: "最近更新",
      recentUpdatesDesc: "更像 docs changelog 的阅读入口，优先看标题、栏目、更新时间和摘要。",
      startWithSubsection: "先从子专题进入",
      openSubsection: "Open subsection",
      menuHome: "首页",
      menuAbout: "关于",
      menuPosts: "日志",
      menuNotes: "笔记",
      menuTools: "工具",
      menuPoems: "诗词",
      menuTodo: "事务",
      menuResume: "简历",
      menuTheme: "主题",
      menuPage: "页面",
      archiveSuffix: "归档",
      countItems: (count) => `${count} 篇内容`,
      countSubsections: (count) => `${count} 个子栏目`,
      countPages: (count) => `${count} pages`,
      countTopicPages: (count) => `${count} pages available for this topic.`,
      countSubsectionPages: (count) => `${count} pages in this subsection.`
    },
    en: {
      localeLabel: "English",
      localeSwitch: "Switch language",
      searchDocs: "Search...",
      menu: "Menu",
      switchDark: "Switch to dark mode",
      switchLight: "Switch to light mode",
      searchSite: "Search the site",
      close: "Close",
      searchPlaceholder: "Search...",
      escToClose: "",
      searchIntro: "",
      searching: "",
      noResults: "No results found. Try a shorter keyword or continue from a section page.",
      foundResults: () => "",
      searchUnavailable: "Search assets are unavailable. Run a full build to generate the Pagefind index.",
      openPage: "Open page",
      untitledPage: "Untitled page",
      copy: "Copy",
      copied: "Copied",
      failed: "Failed",
      headingAnchor: "Link to this heading",
      onThisPage: "On this page",
      pageInfo: "Page info",
      updatedOn: "Updated",
      backTo: "Back to",
      homeHeroTitle: "A personal knowledge base for statistics, time series, machine learning, and LLMs.",
      homeHeroLead: "This site works more like a maintained personal knowledge base than a blog homepage ordered only by time. It focuses on structured browsing, fast retrieval, and long-term reuse across statistics, time series, machine learning, and LLM topics.",
      homeStoryIntro: "After graduating from the Department of Mathematics at Shanghai University in 2017, I began working and living in Shanghai. Outside work, I usually spend time listening to music, reading, sleeping, and occasionally writing posts and notes here to record life and learning.",
      homeGithubIntro: "This is my GitHub contribution chart, showing what I have been working on:",
      topicTimeseriesTitle: "Time series",
      topicTimeseriesDesc: "A full entry point from models and evaluation to deep learning topics",
      topicMlTitle: "Machine learning",
      topicMlDesc: "Organized around supervised learning, feature engineering, and ensemble methods",
      topicLlmTitle: "LLMs",
      topicLlmDesc: "From fundamentals and models to fine-tuning and engineering interfaces",
      browseByType: "Browse by content type",
      browseByTypeDesc: "Keep the docs-style entry flow: choose a content type first, then open a topic or page.",
      notesDesc: "Systematic notes on statistics, time series, machine learning, operations research, and LLMs.",
      toolsDesc: "Reference-style docs, tutorials, and quick lookups for day-to-day work.",
      postsDesc: "Time-ordered writing for recent ideas, projects, and working notes.",
      aboutDesc: "Author profile, site direction, and structure for first-time visitors.",
      profile: "Profile",
      deepDiveCore: "Continue from core topics",
      deepDiveDesc: "Topic pages expand into subsections, reading paths, and time-based archives.",
      openTopic: "Open topic",
      recentUpdates: "Recent updates",
      recentUpdatesDesc: "A docs-like changelog entry point: scan titles, sections, dates, and excerpts first.",
      startWithSubsection: "Start with subsections",
      openSubsection: "Open subsection",
      menuHome: "Home",
      menuAbout: "About",
      menuPosts: "Posts",
      menuNotes: "Notes",
      menuTools: "Tools",
      menuPoems: "Poems",
      menuTodo: "Tasks",
      menuResume: "Resume",
      menuTheme: "Theme",
      menuPage: "Page",
      archiveSuffix: "Archive",
      countItems: (count) => `${count} items`,
      countSubsections: (count) => `${count} subsections`,
      countPages: (count) => `${count} pages`,
      countTopicPages: (count) => `${count} pages available for this topic.`,
      countSubsectionPages: (count) => `${count} pages in this subsection.`
    }
  };

  const currentLocale = () => root.getAttribute("data-locale") || "zh-CN";
  const normalizeKey = (key) => key.replace(/_([a-z])/g, (_, char) => char.toUpperCase());
  const message = (key, locale = currentLocale()) => messages[locale]?.[key] ?? messages["zh-CN"][key] ?? "";
  const routeKeyFromUrl = (url, exact = false) => {
    try {
      const path = new URL(url, window.location.origin).pathname.toLowerCase();
      if (path === "/") return "menuHome";
      if (path === "/about/" || (!exact && path.startsWith("/about/"))) return "menuAbout";
      if (path === "/post/" || (!exact && path.startsWith("/post/"))) return "menuPosts";
      if (path === "/note/" || (!exact && path.startsWith("/note/"))) return "menuNotes";
      if (path === "/tool/" || (!exact && path.startsWith("/tool/"))) return "menuTools";
      if (path === "/pomes/" || (!exact && path.startsWith("/pomes/"))) return "menuPoems";
      if (path === "/todo/" || (!exact && path.startsWith("/todo/"))) return "menuTodo";
      if (path === "/resume/" || (!exact && path.startsWith("/resume/"))) return "menuResume";
      if (path === "/ivy/" || (!exact && path.startsWith("/ivy/"))) return "menuTheme";
    } catch (error) {}
    return "";
  };
  const routeLabelFromUrl = (url, locale = currentLocale(), exact = false) => {
    const key = routeKeyFromUrl(url, exact);
    return key ? message(key, locale) : "";
  };
  const formatCount = (key, count, locale = currentLocale()) => {
    const formatter = message(key, locale);
    return typeof formatter === "function" ? formatter(count) : `${count}`;
  };

  const updateLocaleUi = (locale) => {
    root.setAttribute("data-locale", locale);
    root.lang = locale === "en" ? "en" : "zh-CN";
    localeLabels.forEach((label) => {
      label.textContent = message("localeLabel", locale);
    });
    localeToggles.forEach((toggle) => {
      toggle.setAttribute("aria-label", message("localeSwitch", locale));
      toggle.setAttribute("title", message("localeSwitch", locale));
    });
    document.querySelectorAll("[data-i18n]").forEach((node) => {
      const key = normalizeKey(node.getAttribute("data-i18n") || "");
      const value = message(key, locale);
      if (value) node.textContent = value;
    });
    document.querySelectorAll("[data-i18n-placeholder]").forEach((node) => {
      const key = normalizeKey(node.getAttribute("data-i18n-placeholder") || "");
      const value = message(key, locale);
      if (value) node.setAttribute("placeholder", value);
    });
    document.querySelectorAll("[data-i18n-prefix]").forEach((node) => {
      const prefix = message(normalizeKey(node.getAttribute("data-i18n-prefix") || ""), locale);
      const section = routeLabelFromUrl(node.getAttribute("href") || "", locale, true) || node.textContent.replace(/^(返回|Back to)\s*/, "");
      node.textContent = `${prefix}${locale === "en" ? " " : ""}${section}`;
    });
    document.querySelectorAll("[data-i18n-count]").forEach((node) => {
      const key = node.getAttribute("data-i18n-count");
      const count = Number(node.getAttribute("data-count") || 0);
      const map = {
        items: "countItems",
        subsections: "countSubsections",
        pages: "countPages",
        topic_pages: "countTopicPages",
        subsection_pages: "countSubsectionPages"
      };
      const formatterKey = map[key];
      if (formatterKey) node.textContent = formatCount(formatterKey, count, locale);
    });
    document.querySelectorAll("[data-i18n-archive-title]").forEach((node) => {
      const title = node.getAttribute("data-i18n-archive-title") || "";
      const translatedTitle = routeLabelFromUrl(window.location.pathname, locale, true) || title;
      node.textContent = locale === "en" ? `${translatedTitle} ${message("archiveSuffix", locale)}` : `${translatedTitle}${message("archiveSuffix", locale)}`;
    });
    document.querySelectorAll("[data-route-title]").forEach((node) => {
      const translated = routeLabelFromUrl(node.getAttribute("data-route-title") || "", locale, true);
      if (translated) node.textContent = translated;
    });
    document.querySelectorAll(".site-search-trigger__text").forEach((node) => {
      node.textContent = message("searchDocs", locale);
    });
    document.querySelectorAll("[data-menu-toggle]").forEach((node) => {
      node.textContent = message("menu", locale);
    });
    closeSearchButtons.forEach((node) => {
      node.setAttribute("aria-label", message("close", locale));
    });
    document.querySelectorAll(".site-nav__item a, .docs-tabs__item a, .docs-sidebar__title a").forEach((node) => {
      const translated = routeLabelFromUrl(node.getAttribute("href") || "", locale, true);
      if (translated) node.textContent = translated;
    });
    document.querySelectorAll(".code-copy").forEach((node) => {
      node.textContent = message("copy", locale);
    });
  };

  const currentTheme = () => root.getAttribute("data-theme") || "light";

  const updateThemeUi = (theme) => {
    const locale = currentLocale();
    root.setAttribute("data-theme", theme);
    root.style.colorScheme = theme;
    if (themeColorMeta) {
      themeColorMeta.setAttribute("content", theme === "dark" ? "#171513" : "#f7f6f2");
    }
    themeIcons.forEach((icon) => {
      icon.textContent = theme === "dark" ? "☀" : "☾";
    });
    themeLabels.forEach((label) => {
      label.textContent = theme === "dark" ? message("switchLight", locale) : message("switchDark", locale);
    });
    themeToggles.forEach((toggle) => {
      toggle.setAttribute("aria-pressed", String(theme === "dark"));
      toggle.setAttribute("data-theme-state", theme);
      toggle.setAttribute("aria-label", theme === "dark" ? message("switchLight", locale) : message("switchDark", locale));
      toggle.setAttribute("title", theme === "dark" ? message("switchLight", locale) : message("switchDark", locale));
    });
    updateUtterancesTheme(theme);
  };

  const updateUtterancesTheme = (theme = currentTheme()) => {
    if (!utterancesRoot) return;
    const lightTheme = utterancesRoot.getAttribute("data-utterances-theme-light");
    const darkTheme = utterancesRoot.getAttribute("data-utterances-theme-dark");
    const nextTheme = theme === "dark" ? darkTheme : lightTheme;
    if (!nextTheme) return;
    const frame = document.querySelector("iframe.utterances-frame");
    if (!frame || !frame.contentWindow) return;
    frame.contentWindow.postMessage(
      { type: "set-theme", theme: nextTheme },
      "https://utteranc.es"
    );
  };

  const setTheme = (theme, persist = true) => {
    updateThemeUi(theme);
    if (persist) {
      try {
        localStorage.setItem(themeKey, theme);
      } catch (error) {}
    }
  };

  const setLocale = (locale, persist = true) => {
    updateLocaleUi(locale);
    updateThemeUi(currentTheme());
    if (persist) {
      try {
        localStorage.setItem(localeKey, locale);
      } catch (error) {}
    }
    if (searchInput) {
      searchInput.setAttribute("lang", locale === "en" ? "en" : "zh-CN");
    }
    if (searchInput && !searchInput.value.trim() && searchState) {
      searchState.textContent = message("searchIntro", locale);
    }
  };

  setLocale(currentLocale(), false);
  updateThemeUi(currentTheme());

  themeToggles.forEach((toggle) => {
    toggle.addEventListener("click", () => {
      const nextTheme = currentTheme() === "dark" ? "light" : "dark";
      setTheme(nextTheme);
    });
  });

  localeToggles.forEach((toggle) => {
    toggle.addEventListener("click", () => {
      const nextLocale = currentLocale() === "zh-CN" ? "en" : "zh-CN";
      setLocale(nextLocale);
    });
  });

  if (window.matchMedia) {
    const media = window.matchMedia("(prefers-color-scheme: dark)");
    const syncTheme = (event) => {
      try {
        if (!localStorage.getItem(themeKey)) {
          updateThemeUi(event.matches ? "dark" : "light");
        }
      } catch (error) {
        updateThemeUi(event.matches ? "dark" : "light");
      }
    };
    if (typeof media.addEventListener === "function") {
      media.addEventListener("change", syncTheme);
    } else if (typeof media.addListener === "function") {
      media.addListener(syncTheme);
    }
  }

  const updateScrolledState = () => {
    if (!header) return;
    header.classList.toggle("is-scrolled", window.scrollY > 12);
  };

  const handleScroll = () => {
    updateScrolledState();
  };

  window.addEventListener("scroll", handleScroll, { passive: true });
  handleScroll();

  if (menuToggle && mobileNav) {
    menuToggle.addEventListener("click", () => {
      const expanded = menuToggle.getAttribute("aria-expanded") === "true";
      menuToggle.setAttribute("aria-expanded", String(!expanded));
      mobileNav.hidden = expanded;
    });
  }

  const closeSearch = () => {
    if (!searchModal) return;
    searchModal.hidden = true;
    body.classList.remove("is-search-open");
  };

  const openSearch = async () => {
    if (!searchModal) return;
    searchModal.hidden = false;
    body.classList.add("is-search-open");
    if (searchInput && !searchInput.value.trim()) {
      if (searchState) searchState.textContent = "";
      if (searchResults) searchResults.innerHTML = "";
    }
    window.setTimeout(() => {
      if (searchInput) searchInput.focus();
    }, 20);
  };

  openSearchButtons.forEach((button) => {
    button.addEventListener("click", openSearch);
  });

  closeSearchButtons.forEach((button) => {
    button.addEventListener("click", closeSearch);
  });

  document.addEventListener("keydown", (event) => {
    const key = event.key.toLowerCase();
    if ((event.metaKey || event.ctrlKey) && key === "k") {
      event.preventDefault();
      openSearch();
    } else if (key === "escape") {
      closeSearch();
      if (menuToggle && mobileNav && !mobileNav.hidden) {
        menuToggle.setAttribute("aria-expanded", "false");
        mobileNav.hidden = true;
      }
    }
  });

  const ensurePagefind = async () => {
    if (pagefindModule) return pagefindModule;
    pagefindModule = await import("/pagefind/pagefind.js");
    return pagefindModule;
  };

  const sectionLabelFromUrl = (url) => {
    return routeLabelFromUrl(url) || message("menuPage");
  };

  const normalizedPath = (url) =>
    url.replace(/^\/+|\/+$/g, "").split("/").slice(0, 4).join(" / ") || "home";

  const renderSearchResults = async (query) => {
    if (!searchResults || !searchState) return;
    const currentSequence = ++searchSequence;
    const trimmed = query.trim();
    const locale = currentLocale();
    if (!trimmed) {
      searchState.textContent = "";
      searchResults.innerHTML = "";
      return;
    }

    searchState.textContent = message("searching", locale);
    try {
      const pagefind = await ensurePagefind();
      const result = await pagefind.search(trimmed);
      if (currentSequence !== searchSequence) return;
      const items = await Promise.all(
        result.results.slice(0, 10).map(async (item) => item.data())
      );

      if (!items.length) {
        searchState.innerHTML = `<div class="search-empty">${message("noResults", locale)}</div>`;
        searchResults.innerHTML = "";
        return;
      }

      searchState.textContent = "";
      searchResults.innerHTML = items
        .map((item, index) => {
          const excerpt = item.excerpt ? item.excerpt.replace(/<[^>]+>/g, "") : message("openPage", locale);
          const section = sectionLabelFromUrl(item.url);
          return `
            <a class="search-result" href="${item.url}">
              <div class="search-result__top">
                <span class="search-result__index">${String(index + 1).padStart(2, "0")}</span>
                <div class="search-result__meta">
                  <span class="search-result__section">${section}</span>
                  <span>${normalizedPath(item.url)}</span>
                </div>
              </div>
              <div class="search-result__body">
                <h3>${item.meta.title || message("untitledPage", locale)}</h3>
                <span class="search-result__cta">${message("openPage", locale)}</span>
              </div>
              <p class="search-result__excerpt">${excerpt}</p>
            </a>
          `;
        })
        .join("");
    } catch (error) {
      searchState.innerHTML = `<div class="search-empty">${message("searchUnavailable", locale)}</div>`;
      searchResults.innerHTML = "";
      console.error(error);
    }
  };

  if (searchInput) {
    let timeoutId = null;
    searchInput.addEventListener("input", (event) => {
      window.clearTimeout(timeoutId);
      timeoutId = window.setTimeout(() => {
        renderSearchResults(event.target.value);
      }, 120);
    });
  }

  const languageName = (className = "") => {
    const match = className.match(/(?:language|lang)-([a-z0-9#+_-]+)/i);
    const raw = match ? match[1].toLowerCase() : "";
    const names = {
      bash: "Bash",
      shell: "Shell",
      sh: "Shell",
      zsh: "Shell",
      console: "Shell",
      python: "Python",
      py: "Python",
      r: "R",
      yaml: "YAML",
      yml: "YAML",
      json: "JSON",
      toml: "TOML",
      ini: "INI",
      latex: "TeX",
      tex: "TeX",
      cpp: "C++",
      c: "C",
      javascript: "JavaScript",
      js: "JavaScript",
      typescript: "TypeScript",
      ts: "TypeScript",
      html: "HTML",
      xml: "XML",
      css: "CSS",
      scss: "SCSS",
      sql: "SQL",
      go: "Go",
      rust: "Rust",
      java: "Java",
      kotlin: "Kotlin",
      swift: "Swift",
      matlab: "MATLAB",
      mat: "MATLAB",
      plaintext: "Text",
      text: "Text"
    };
    return names[raw] || (raw ? raw.toUpperCase() : "");
  };

  document.querySelectorAll("pre").forEach((pre) => {
    const code = pre.querySelector("code");
    if (!code || pre.closest(".code-block-shell")) return;
    const wrapper = document.createElement("div");
    wrapper.className = "code-block-shell";
    const meta = document.createElement("div");
    meta.className = "code-block__meta";
    const label = document.createElement("span");
    label.className = "code-block__language";
    label.textContent = languageName(code.className);
    const actions = document.createElement("div");
    actions.className = "code-block__actions";
    const button = document.createElement("button");
    button.type = "button";
    button.className = "code-copy";
    button.textContent = message("copy");
    button.addEventListener("click", async () => {
      try {
        await navigator.clipboard.writeText(code.innerText);
        button.textContent = message("copied");
        window.setTimeout(() => {
          button.textContent = message("copy");
        }, 1200);
      } catch (error) {
        button.textContent = message("failed");
        window.setTimeout(() => {
          button.textContent = message("copy");
        }, 1200);
      }
    });
    actions.appendChild(button);
    if (!label.textContent) label.hidden = true;
    meta.append(label, actions);
    pre.parentNode.insertBefore(wrapper, pre);
    wrapper.append(meta, pre);
  });

  const headingSelector = ".article-content h2[id], .article-content h3[id], .article-content h4[id]";
  const headings = Array.from(document.querySelectorAll(headingSelector));
  headings.forEach((heading) => {
    if (heading.querySelector(".heading-anchor")) return;
    const anchor = document.createElement("a");
    anchor.className = "heading-anchor";
    anchor.href = `#${heading.id}`;
    anchor.setAttribute("aria-label", message("headingAnchor"));
    anchor.textContent = "#";
    heading.appendChild(anchor);
  });

  if (tocLinks.length && headings.length && "IntersectionObserver" in window) {
    const lookup = new Map();
    tocLinks.forEach((link) => {
      lookup.set(link.getAttribute("href"), link);
    });

    const observer = new IntersectionObserver(
      (entries) => {
        entries.forEach((entry) => {
          const link = lookup.get(`#${entry.target.id}`);
          if (!link) return;
          if (entry.isIntersecting) {
            tocLinks.forEach((item) => item.classList.remove("is-active"));
            link.classList.add("is-active");
          }
        });
      },
      { rootMargin: "-25% 0px -60% 0px", threshold: 0 }
    );

    headings.forEach((heading) => observer.observe(heading));
  }

  if ("IntersectionObserver" in window) {
    const revealObserver = new IntersectionObserver(
      (entries) => {
        entries.forEach((entry) => {
          if (entry.isIntersecting) {
            entry.target.classList.add("is-visible");
            revealObserver.unobserve(entry.target);
          }
        });
      },
      { rootMargin: "0px 0px -10% 0px", threshold: 0.08 }
    );
    revealNodes.forEach((node) => revealObserver.observe(node));
  } else {
    revealNodes.forEach((node) => node.classList.add("is-visible"));
  }

  const utterancesFrame = document.querySelector("iframe.utterances-frame");
  if (utterancesFrame) {
    utterancesFrame.addEventListener("load", () => {
      window.setTimeout(() => updateUtterancesTheme(currentTheme()), 80);
    });
  } else if (utterancesRoot) {
    let attempts = 0;
    const syncUtterances = window.setInterval(() => {
      const frame = document.querySelector("iframe.utterances-frame");
      if (frame) {
        frame.addEventListener("load", () => {
          window.setTimeout(() => updateUtterancesTheme(currentTheme()), 80);
        }, { once: true });
        updateUtterancesTheme(currentTheme());
        window.clearInterval(syncUtterances);
      }
      attempts += 1;
      if (attempts > 20) window.clearInterval(syncUtterances);
    }, 300);
  }
})();
