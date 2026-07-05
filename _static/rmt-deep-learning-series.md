<!-- DRAFT / SERIES ROADMAP — planning doc, not yet registered in build.py.
     Organizes a multi-post series on RMT × deep learning. Each post has an
     explicit "unexpected link" as its spine. The free-probability post
     (_static/free-probability-deep-learning.md) is Post 1. -->

> **Status: series roadmap (draft).** A planning document for a series of blog
> posts. The thesis, the per-post hooks, and the dependency order live here; each
> post gets its own `_static/<slug>.md` as it's written.

# Random Matrices Meet Deep Learning — a series

**The thesis.** The central objects of deep learning are *large random matrices*:
weight matrices, input–output Jacobians, loss Hessians, kernels (CK/NTK), data
covariances. Random Matrix Theory (RMT) is the physics of their **eigenvalue
spectra**. The fun isn't that RMT *applies* — it's that it keeps producing results
that are genuinely **surprising**: phenomena practitioners discovered empirically
(double descent, scaling laws, trainability of very deep nets, "spikes" in the
Hessian) turn out to be *predicted*, sometimes exactly, by a spectral law written
down decades earlier for nuclear physics or statistics.

**Editorial rule for the series.** Every post is built around *one* unexpected link:
*"you'd never guess that [DL phenomenon] is really [RMT object]."* Lead with the
surprise, then earn it. Keep an honest "where it breaks" note in each.

---

## The arc (each post = one unexpected link)

### Post 1 — The semicircle hiding in your loss landscape
*RMT idea:* **free probability & free convolution** (R-transform / ⊞, S-transform / ⊠).
*DL objects:* the **Hessian** (a sum → ⊞) and the **Jacobian** (a product → ⊠).
*The unexpected link:* you can train a **10,000-layer** vanilla network by *tuning a
free-multiplicative-convolution spectrum* (dynamical isometry) — and the loss
Hessian's shape (convex bowl vs. saddle-riddled) is a **Wishart ⊞ Wigner** density
that morphs with the loss value.
*Draft:* `_static/free-probability-deep-learning.md` (already scaffolded & reviewed).
*Status:* outline complete, prose pending. **This is the theory backbone for the series.**

### Post 2 — Heavy tails know your test accuracy
*RMT idea:* **heavy-tailed spectra / power laws** (vs. the clean Marchenko–Pastur bulk).
*DL object:* trained **weight matrices**.
*The unexpected link:* fit a power law to a weight matrix's eigenvalue density and
its **exponent predicts generalization** — *without ever touching the test set, the
training labels, or even the data.* Heavy-Tailed Self-Regularization (Martin &
Mahoney); the `weightwatcher` tool operationalizes it. Also: the "5+1 phases" a
weight spectrum passes through during training (random → bulk+spikes → heavy-tailed).
*Why it's surprising:* generalization, allegedly a data-dependent quantity, leaves a
fingerprint in the *weights alone*.

### Post 3 — Double descent is a Marchenko–Pastur edge
*RMT idea:* the **Marchenko–Pastur law** and the divergence of the **smallest
eigenvalue** as the aspect ratio → 1.
*DL phenomenon:* the **double-descent** test-error curve.
*The unexpected link:* the notorious error *explosion at the interpolation threshold*
($n \approx p$) is just the MP edge — the sample-covariance / feature-Gram matrix
becomes **ill-conditioned** exactly there, and *over*-parameterizing past it
**improves conditioning** and lowers error. The bump isn't mysterious; it's a
vanishing eigenvalue. (Belkin et al.; Hastie–Montanari–Rosset–Tibshirani;
Mei–Montanari; Nakkiran et al.)

### Post 4 — Scaling laws fall out of a power-law spectrum
*RMT idea:* **spectra of random-feature / kernel matrices** when the **data
covariance is power-law**.
*DL phenomenon:* **neural scaling laws** (loss ∝ (data or params)$^{-\alpha}$).
*The unexpected link:* a *solvable* random-feature model reproduces the empirical
scaling exponents analytically — and the exponent $\alpha$ is **set by the power-law
tail of the data covariance spectrum**. The "magic numbers" in scaling-law plots are
an RMT readout of the dataset's spectrum. (Maloney, Roberts & Sully, *A Solvable
Model of Neural Scaling Laws*, arXiv:2210.16859; Bahri et al., *Explaining Neural
Scaling Laws*.)

### Post 5 — Universality: the spectral edge ignores your architecture
*RMT idea:* **universality** — Wigner semicircle, Tracy–Widom edge, local eigenvalue
spacing — laws that don't depend on microscopic details.
*DL object:* Hessian / Jacobian / weight spectra across architectures & nonlinearities.
*The unexpected link:* swap the nonlinearity, the init, the data — and the
**limiting bulk and the microscopic edge statistics stay the same**. Deep nets
inherit RMT's universality, which is *why* RMT predictions transfer at all.
(Pennington–Schoenholz–Ganguli spectral universality; Tracy–Widom at the Hessian edge.)

### Post 6 — Signal vs. noise: spikes, BBP, and what the network learned
*RMT idea:* **spiked models & the BBP phase transition** (a low-rank perturbation
pops outliers out of the bulk past a threshold).
*DL object:* the few **outlier** eigenvalues of the Hessian / Gram matrices.
*The unexpected link:* the number and structure of Hessian **"spikes" mirrors the
label structure** — e.g. roughly one outlier cluster per class; class/cross-class
geometry is legible in the spectrum. Signal (what's learned) literally separates from
noise (the bulk) at a sharp threshold. (Papyan, *Traces of class/cross-class
structure*; BBP.)

### Post 7 (capstone / optional) — Where RMT breaks, and why that's interesting too
*The honest chapter:* Gaussian-equivalence assumptions failing, finite-width
corrections, correlated/structured weights after training, and the unresolved
"empirical Hessian ≠ clean Wishart" puzzle. The frontier, not a footnote.

---

## Dependency / reading order
- **Post 1** is the toolbox (free convolution, MP, semicircle) — write/publish first.
- **Posts 2–6** are *independent* and can ship in any order; each re-introduces only
  the one RMT tool it needs (link back to Post 1 for depth).
- **Post 7** is a capstone — write last.
- Recommended publish order by "hook strength / shareability": **3 (double descent)**
  or **2 (heavy tails predict accuracy)** make the punchiest standalone second post.

## Cross-cutting assets to build once, reuse everywhere
- A small **`numpy/scipy` RMT toolkit**: sample Wigner/Wishart, compute empirical
  spectral density, Marchenko–Pastur & semicircle reference curves, a free-convolution
  (subordination fixed-point) routine. Reused in Posts 1, 3, 5, 6.
- A **PyHessian** harness (SLQ spectral density on a small net) — Posts 1, 5, 6.
- A consistent **watercolor-styled spectral-density plot** template (matplotlib →
  styled SVG) so every post's spectra look like one family.

## Series-wide resources (beyond Post 1's list)
- Martin & Mahoney, *Implicit Self-Regularization in Deep Neural Networks: Evidence
  from Random Matrix Theory* (HT-SR); `weightwatcher`:
  <https://github.com/CalculatedContent/WeightWatcher>
- Belkin, Hsu, Ma, Mandal, *Reconciling modern ML practice and the bias–variance
  trade-off* (double descent), PNAS 2019.
- Hastie, Montanari, Rosset, Tibshirani, *Surprises in High-Dimensional Ridgeless
  Least Squares Interpolation*. <https://arxiv.org/abs/1903.08560>
- Mei & Montanari, *The generalization error of random features regression*.
  <https://arxiv.org/abs/1908.05355>
- Nakkiran et al., *Deep Double Descent*. <https://arxiv.org/abs/1912.02292>
- Maloney, Roberts, Sully, *A Solvable Model of Neural Scaling Laws*.
  <https://arxiv.org/abs/2210.16859>
- Bahri, Dyer, Kaplan, Lee, Sharma, *Explaining Neural Scaling Laws*, PNAS 2024.
  <https://arxiv.org/abs/2102.06701>
- Papyan, *Traces of Class/Cross-Class Structure Pervade Deep Learning Spectra*.
  <https://arxiv.org/abs/2008.11865>
- Pastur, *Appearance of Random Matrix Theory in Deep Learning* (survey).
  <https://arxiv.org/abs/2102.06740>

## Publishing mechanics
- Category for all: `["Research"]` (consider adding a new **"Math / Theory"** section
  in `build.py`, or a dedicated **"RMT × Deep Learning"** section so the series groups
  together on the landing page).
- Each post links to the next/prev to read as a series.
- Keep slugs uniform: `rmt-<topic>` (e.g. `rmt-free-probability`, `rmt-heavy-tails`,
  `rmt-double-descent`, `rmt-scaling-laws`, `rmt-universality`, `rmt-spikes`).
  *(Post 1 currently lives at `free-probability-deep-learning` — rename to
  `rmt-free-probability` on publish for consistency.)*
