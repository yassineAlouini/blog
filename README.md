# The Notebook Blog

High-scoring Kaggle notebooks and research write-ups by
[Yassine Alouini](https://www.kaggle.com/yassinealouini/code), ported to standalone,
shareable HTML in a clean reading layout.

**Live site:** https://yassineAlouini.github.io/blog/

## Posts
- **All the Segmentation Metrics** — IoU, Dice, Hausdorff and more, illustrated.
- **EfficientDet meets PyTorch Lightning** — object detection, Lightning-style.
- **FFmpeg 101** — practical video/audio preprocessing recipes.
- **RoBERTa meets TPUs** — training RoBERTa on TPUs.
- **LeCun's World Models** (external) — JEPA / LeJEPA / LeWorldModel research digest.

## Rebuilding
Source notebooks live in `_src/` (pulled via the Kaggle API). To regenerate the HTML:

```bash
python3 build.py
```

`build.py` converts each notebook to a branded page (markdown via mistune + MathJax,
code via Pygments) and regenerates `index.html`. Styling is shared in `assets/`.

Built with [Claude Code](https://claude.com/claude-code), styled in Anthropic's brand design language.
