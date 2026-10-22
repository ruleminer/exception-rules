"""Build the reviewer supplement independently of notebooks and API imports."""

project = "exception-rules — Supplementary materials"
author = "Dawid Macha, Łukasz Wróbel, Marek Sikora"
copyright = "2026, the authors"
root_doc = "supplementary_materials"
extensions = ["sphinx.ext.mathjax"]
exclude_patterns = [
    "index.rst",
    "api.rst",
    "quick_start.rst",
    "quick_start/**",
    "tutorials.rst",
    "tutorials/**",
]
html_theme = "alabaster"
html_title = "exception-rules: Scientific background and experimental evaluation"
html_show_sourcelink = False
html_theme_options = {"page_width": "1120px", "body_max_width": "900px"}
highlight_language = "text"
