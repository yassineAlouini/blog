# The Notebook Blog

High-scoring Kaggle notebooks and research write-ups by
[Yassine Alouini](https://www.kaggle.com/yassinealouini/code) — computer vision,
multi-modal applications and LLMs — ported to standalone, shareable HTML in a
hand-made *Mediterranean watercolor* reading layout.

**Live site → https://yassinealouini.github.io/blog/**

---

## What's on the blog

The landing page (`index.html`) is organized into themed sections. Each post is a
self-contained HTML page under `posts/` with the original narrative, math and code.

### 🎨 Computer Vision
*Segmentation, detection, and the metrics that score them.*
- **[All the Segmentation Metrics](posts/all-the-segmentation-metrics.html)** — a visual
  catalogue of IoU, Dice, Hausdorff and many more, explained and illustrated.
  ([Kaggle](https://www.kaggle.com/code/yassinealouini/all-the-segmentation-metrics))
- **[EfficientDet meets PyTorch Lightning](posts/efficientdet-meets-pytorch-lightning.html)** —
  object detection with EfficientDet wrapped in a clean Lightning training loop.
  ([Kaggle](https://www.kaggle.com/code/yassinealouini/efficientdet-meets-pytorch-lightning))

### 🎬 Video Processing
*Working with video — codecs, frames, and the preprocessing plumbing.*
- **[FFmpeg 101](posts/ffmpeg-101.html)** — practical FFmpeg recipes for video and audio
  preprocessing in ML pipelines.
  ([Kaggle](https://www.kaggle.com/code/yassinealouini/ffmpeg-101))

### 🤖 NLP & LLMs
*Language models and how to train them efficiently.*
- **[RoBERTa meets TPUs](posts/roberta-meets-tpus.html)** — training RoBERTa efficiently on
  TPUs with PyTorch/XLA and Lightning.
  ([Kaggle](https://www.kaggle.com/code/yassinealouini/roberta-meets-tpus))

### 🔬 Research
*Deeper dives and paper write-ups.*
- **[LeCun's World Models: JEPA, LeJEPA & LeWorldModel](https://yassineAlouini.github.io/lecun-world-models/)**
  *(external)* — a technical walkthrough of the JEPA world-model line, the maths, a PyTorch
  reconstruction, and the AMI Labs startup. Lives in its own
  [repo](https://github.com/yassineAlouini/lecun-world-models).

---

## How it's built

`build.py` is a small static-site generator (no framework). For each notebook in `_src/` it:

- renders **markdown** cells with [mistune](https://github.com/lepture/mistune) — including
  `$…$` math (handed to **MathJax**), tables, and notebook image attachments inlined as data URIs;
- renders **code** (both notebook code cells and fenced code blocks inside markdown) with
  [Pygments](https://pygments.org/) syntax highlighting;
- wraps everything in a branded page shell and shared CSS.

It then generates `index.html`, grouping posts into sections by their `category` field.
Styling lives in `assets/style.css` (watercolor theme: Fraunces / DM Sans / Caveat fonts,
an Aegean palette, paper-grain texture) and `assets/pygments.css`.

### Repository layout

```
build.py              # the generator
index.html            # generated landing page (sections + cards)
posts/                # generated post pages (one .html per notebook)
assets/
  style.css           # watercolor theme (shared)
  pygments.css        # code syntax-highlighting theme
_src/<slug>/<slug>.ipynb   # source notebooks pulled from Kaggle
```

### Rebuild

```bash
python3 build.py        # regenerates posts/ and index.html
```

Dependencies: `mistune`, `pygments` (MathJax and fonts load from CDNs at view time).

### Add a new post

1. Pull the notebook into `_src/<slug>/<slug>.ipynb`
   (`kaggle kernels pull yassinealouini/<slug> -p _src/<slug>`).
2. Add an entry to the `POSTS` list in `build.py` with a `slug`, `title`, `subtitle`,
   `tags`, and a `category` (`"Computer Vision"`, `"Video Processing"`, `"NLP & LLMs"`, …).
   New sections appear automatically; empty ones are skipped.
3. `python3 build.py`, then commit and push.

---

## Deployment

Served by **GitHub Pages** from the `master` branch root. Pushing to `master`
triggers a rebuild; the live site updates within a minute.

> Code snippets are faithful reconstructions of the Kaggle notebook sources (narrative + code);
> cell outputs are not re-executed here — run a notebook on Kaggle for live results.

Built with [Claude Code](https://claude.com/claude-code), styled in a custom watercolor theme.
