"""One-time build helper: render Jinja templates to static HTML for the site.

Run from the project root with the venv Python:
    .venv/Scripts/python site/build_templates.py

url_for('static', ...) calls are rewritten to absolute /static/... paths in
the template SOURCE first, then Jinja renders {% extends %}/{% block %}.
"""
import os
import re

from jinja2 import Environment, FileSystemLoader

TEMPLATES = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "backend", "app", "templates"))
OUT = os.path.abspath(os.path.join(os.path.dirname(__file__), "public"))

PAGES = {
    "home.html": "index.html",
    "dashboard.html": "dashboard.html",
    "predict.html": "predict.html",
    "bulk.html": "bulk.html",
    "history.html": "history.html",
    "comparison.html": "comparison.html",
    "explorer.html": "explorer.html",
    "irrigation.html": "irrigation.html",
    "login.html": "login.html",
}

URL_FOR_RE = re.compile(
    r"\{\{\s*url_for\(\s*'static'\s*,\s*filename\s*=\s*([^)]+?)\s*\)\s*\}\}")


def rewrite_url_for(m):
    # filename arg is like 'css/style.css' (quoted literal in our templates)
    arg = m.group(1).strip().strip("'\"")
    return "/static/" + arg


def main():
    env = Environment(loader=FileSystemLoader(TEMPLATES))
    env.filters["__noop"] = lambda x: x
    os.makedirs(OUT, exist_ok=True)

    # Patch the loader source so url_for is rewritten before parsing.
    orig = env.loader.get_source

    def patched(loader, name):
        src, path, uptodate = orig(loader, name)
        return URL_FOR_RE.sub(rewrite_url_for, src), path, uptodate
    env.loader.get_source = patched

    for src, dst in PAGES.items():
        html = env.get_template(src).render()
        with open(os.path.join(OUT, dst), "w", encoding="utf-8") as f:
            f.write(html)
        print(f"[build] {src} -> {dst} ({len(html)} bytes)")


if __name__ == "__main__":
    main()
