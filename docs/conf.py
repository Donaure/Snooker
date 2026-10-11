"""Sphinx configuration for the Markdown documentation."""

project = "Snooker MuJoCo 项目"
copyright = "2026, Snooker 课程项目贡献者"
language = "zh_CN"

extensions = ["myst_parser"]
source_suffix = {".md": "markdown", ".rst": "restructuredtext"}
root_doc = "index"
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]
myst_heading_anchors = 3

html_theme = "sphinx_rtd_theme"
html_title = project
html_context = {
    "display_github": True,
    "github_user": "Donaure",
    "github_repo": "Snooker",
    "github_version": "master",
    "conf_py_path": "/docs/",
}
