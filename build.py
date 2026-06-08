#!/usr/bin/env python3
"""Build the static blog: convert Kaggle notebooks -> branded HTML + landing index."""
import json, re, html, pathlib, datetime

import mistune
from pygments import highlight
from pygments.lexers import PythonLexer, BashLexer
from pygments.formatters import HtmlFormatter

ROOT = pathlib.Path(__file__).parent
SRC  = ROOT / "_src"
POSTS_DIR = ROOT / "posts"
ASSETS = ROOT / "assets"
POSTS_DIR.mkdir(exist_ok=True); ASSETS.mkdir(exist_ok=True)

# ---------------------------------------------------------------- post registry
POSTS = [
    dict(slug="all-the-segmentation-metrics",
         title="All the Segmentation Metrics",
         subtitle="A visual catalogue of the metrics that show up in segmentation "
                  "challenges — IoU, Dice, Hausdorff and many more — explained and illustrated.",
         tags=["Computer Vision", "Segmentation", "Metrics"]),
    dict(slug="efficientdet-meets-pytorch-lightning",
         title="EfficientDet meets PyTorch Lightning",
         subtitle="Object detection with EfficientDet, wrapped cleanly in a "
                  "PyTorch Lightning training loop.",
         tags=["Object Detection", "PyTorch Lightning", "EfficientDet"]),
    dict(slug="ffmpeg-101",
         title="FFmpeg 101",
         subtitle="Practical FFmpeg recipes for video and audio preprocessing in ML pipelines.",
         tags=["FFmpeg", "Video", "Tooling"]),
    dict(slug="roberta-meets-tpus",
         title="RoBERTa meets TPUs",
         subtitle="Training RoBERTa efficiently on TPUs with PyTorch/XLA and Lightning.",
         tags=["NLP", "TPU", "Transformers"]),
]
KAGGLE_USER = "yassinealouini"
# Public URL of the computer-vision book. Set to "#" to render "here" as plain
# (un-linked) text until a public link is available.
BOOK_URL = "https://alouinimohamedyass.gumroad.com/l/computer_vision_with_pytorch"

# external posts that live in their own repos (linked, not rebuilt)
EXTERNAL = [
    dict(title="LeCun's World Models: JEPA, LeJEPA & LeWorldModel",
         subtitle="A technical walkthrough of the JEPA world-model line — the maths, "
                  "a PyTorch reconstruction, and the AMI Labs startup.",
         tags=["World Models", "JEPA", "Research"],
         url="https://yassineAlouini.github.io/lecun-world-models/"),
]

md = mistune.create_markdown(escape=False,
                             plugins=["math", "table", "strikethrough", "url"])

def lexer_for(src: str):
    s = src.lstrip()
    if s.startswith("!") or s.startswith("%%bash") or s.startswith("%%sh"):
        return BashLexer()
    return PythonLexer()

def render_markdown_cell(cell) -> str:
    text = "".join(cell["source"])
    # inline notebook attachments (pasted images) as data URIs
    for name, mimes in (cell.get("attachments") or {}).items():
        for mime, data in mimes.items():
            b64 = data if isinstance(data, str) else "".join(data)
            text = text.replace(f"attachment:{name}", f"data:{mime};base64,{b64}")
    return f'<div class="md">{md(text)}</div>'

def render_code_cell(cell) -> str:
    src = "".join(cell["source"]).rstrip("\n")
    if not src.strip():
        return ""
    body = highlight(src, lexer_for(src), HtmlFormatter(nowrap=True))
    parts = [f'<div class="code-cell"><pre class="highlight"><code>{body}</code></pre>']
    # render any text/markdown/image outputs that happen to be saved
    for out in cell.get("outputs", []):
        data = out.get("data", {})
        if "image/png" in data:
            b64 = data["image/png"];  b64 = b64 if isinstance(b64, str) else "".join(b64)
            parts.append(f'<div class="out"><img src="data:image/png;base64,{b64}"></div>')
        elif "text/html" in data:
            parts.append(f'<div class="out">{"".join(data["text/html"])}</div>')
        elif "text/plain" in data:
            parts.append(f'<div class="out"><pre>{html.escape("".join(data["text/plain"]))}</pre></div>')
        elif out.get("output_type") == "stream":
            parts.append(f'<div class="out"><pre>{html.escape("".join(out.get("text", [])))}</pre></div>')
    parts.append("</div>")
    return "\n".join(parts)

def notebook_body(nb_path: pathlib.Path) -> str:
    nb = json.loads(nb_path.read_text())
    chunks = []
    for cell in nb.get("cells", []):
        if cell["cell_type"] == "markdown":
            chunks.append(render_markdown_cell(cell))
        elif cell["cell_type"] == "code":
            chunks.append(render_code_cell(cell))
    return "\n".join(c for c in chunks if c)

# ---------------------------------------------------------------- HTML shells
def page_shell(title, head_extra, body, rel="../"):
    return f"""<!DOCTYPE html>
<html lang="en"><head>
<meta charset="UTF-8"><meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>{html.escape(title)}</title>
<link rel="stylesheet" href="{rel}assets/style.css">
<link rel="stylesheet" href="{rel}assets/pygments.css">
{head_extra}
</head><body>
{body}
</body></html>"""

MATHJAX = """<script>window.MathJax={tex:{inlineMath:[['\\\\(','\\\\)']],displayMath:[['\\\\[','\\\\]']]},svg:{fontCache:'global'}};</script>
<script src="https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-svg.js" async></script>"""

def build_post(p):
    nb = SRC / p["slug"] / f"{p['slug']}.ipynb"
    body_inner = notebook_body(nb)
    tags = "".join(f'<span class="tag">{html.escape(t)}</span>' for t in p["tags"])
    kaggle = f"https://www.kaggle.com/code/{KAGGLE_USER}/{p['slug']}"
    body = f"""
<a class="back" href="../index.html">← All posts</a>
<article>
  <header class="post-head">
    <div class="tags">{tags}</div>
    <h1>{html.escape(p['title'])}</h1>
    <p class="subtitle">{html.escape(p['subtitle'])}</p>
    <p class="kaggle-link"><a href="{kaggle}" target="_blank" rel="noopener">▶ View original on Kaggle</a></p>
  </header>
  <div class="note">Ported from the Kaggle notebook source (narrative + code). Cell outputs are
  not re-executed here — run it on Kaggle for live results.</div>
  <div class="nb">{body_inner}</div>
</article>
<footer class="post-foot">Ported from <a href="{kaggle}">kaggle.com/code/{KAGGLE_USER}/{p['slug']}</a> ·
built {datetime.date.today().isoformat()}</footer>
"""
    out = page_shell(p["title"], MATHJAX, body, rel="../")
    (POSTS_DIR / f"{p['slug']}.html").write_text(out)
    print("post  ->", POSTS_DIR / f"{p['slug']}.html", f"({len(out)//1024}KB)")

def card(title, subtitle, tags, href, external=False):
    chips = "".join(f'<span class="tag">{html.escape(t)}</span>' for t in tags)
    ext = '<span class="ext">↗ external</span>' if external else ""
    return f"""<a class="card" href="{href}"{' target="_blank" rel="noopener"' if external else ''}>
  <div class="tags">{chips}{ext}</div>
  <h3>{html.escape(title)}</h3>
  <p>{html.escape(subtitle)}</p>
</a>"""

def build_index():
    cards = [card(p["title"], p["subtitle"], p["tags"], f"posts/{p['slug']}.html") for p in POSTS]
    cards += [card(e["title"], e["subtitle"], e["tags"], e["url"], external=True) for e in EXTERNAL]
    book = "here" if BOOK_URL == "#" else f'<a href="{BOOK_URL}">here</a>'
    body = f"""
<header class="hero">
  <div class="hero-inner">
    <div class="eyebrow">Yassine Alouini · Notebooks &amp; Research</div>
    <h1>The Notebook Blog</h1>
    <p class="lede">I am a <strong>computer vision expert</strong> — at least on the subset of
    deep-learning image applications. I have written a computer vision book; you can get it {book}.</p>
    <p class="lede bio">Focusing now on some <strong>multi-modal applications</strong>
    (video, text and images, …) and on <strong>LLMs for specific tasks</strong>
    (coding, mathematical reasoning, …).</p>
  </div>
</header>
<main class="grid-wrap">
  <div class="grid">
    {''.join(cards)}
  </div>
</main>
<footer class="site-foot">Built with Claude Code · {datetime.date.today().isoformat()} ·
sources on <a href="https://www.kaggle.com/{KAGGLE_USER}/code">Kaggle</a></footer>
"""
    out = page_shell("The Notebook Blog — Yassine Alouini", "", body, rel="")
    (ROOT / "index.html").write_text(out)
    print("index ->", ROOT / "index.html")

def write_assets():
    (ASSETS / "pygments.css").write_text(HtmlFormatter(style="friendly").get_style_defs(".highlight"))
    print("assets-> pygments.css")

if __name__ == "__main__":
    write_assets()
    for p in POSTS:
        build_post(p)
    build_index()
    print("done.")
