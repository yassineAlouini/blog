# `rmtdl` — runnable companion code for *Random Matrices Meet Deep Learning*

Small, dependency-light (numpy / scipy / matplotlib) code so readers can **run
each example and watch the RMT ↔ deep-learning connections appear**. Every figure
is styled to match the blog's watercolor theme.

## Setup

```bash
cd code
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

## Run an example

Each script is self-contained and writes a PNG into `outputs/`:

```bash
python examples/01_asymptotic_freeness.py
# ... or run them all:
for f in examples/*.py; do python "$f"; done
```

## What's here

| Example | Series post | The unexpected link it shows |
|---|---|---|
| `01_asymptotic_freeness.py` | 1 | Spectrum of `A + B` (randomly rotated) is the **arcsine** law (free convolution), **not** the classical 3-atom convolution. |
| `02_hessian_wishart_wigner.py` | 1 — Bridge I | The loss **Hessian** = **Wishart ⊞ Wigner**; negative eigenvalues (saddles) emerge as the loss/energy rises. |
| `03_jacobian_isometry.py` | 1 — Bridge II | The **Jacobian** is a matrix *product*; tuning the init (orthogonal + gain) gives **dynamical isometry** — singular values stay near 1 at any depth. |
| `04_heavy_tails.py` | 2 | Weight spectra drift from **Marchenko–Pastur** at init to a **heavy power-law tail**; the tail exponent is a generalization gauge (HT-SR). |
| `05_double_descent.py` | 3 | **Double descent**'s error spike sits exactly on the **MP edge** — the feature Gram's smallest eigenvalue → 0 at `p = n`. |
| `06_spikes_bbp.py` | 6 | A rank-1 signal escapes the noise bulk only past the **BBP threshold** — the sharp signal/noise split behind Hessian "spikes". |

## The `rmtdl` package

- `spectra.py` — sample Wigner / Wishart / Haar-orthogonal ensembles; empirical
  spectral density; closed-form **semicircle**, **Marchenko–Pastur**, **arcsine** laws.
- `free.py` — **free additive convolution** via the subordination fixed point
  (the numerical workhorse; no R-transform inversion needed).
- `plotting.py` — shared watercolor style + `save()` helper.

## Going further on a real model (optional)

The posts point at three drop-in tools for reproducing these effects on an actual
network (install from the commented lines in `requirements.txt`):

- **PyHessian** — full Hessian eigenvalue spectral density via Stochastic Lanczos
  Quadrature; compare a real net's bulk to `02`'s Wishart ⊞ Wigner fit.
- **WeightWatcher** — heavy-tailed self-regularization metrics on real trained
  layers; reproduces `04` on your own weights.
- **torch** — compute a real input–output Jacobian to pair with `03`.
