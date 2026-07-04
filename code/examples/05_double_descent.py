"""Post 3 -- double descent is a Marchenko-Pastur edge.

Ridgeless random-feature regression: draw p random features, fit n training
points by least norm. Sweep p/n across 1. The test error spikes exactly at the
interpolation threshold p = n -- and the *reason* is RMT: the feature Gram
matrix's smallest eigenvalue is governed by the Marchenko-Pastur edge
lam_minus = (1 - sqrt(p/n))^2, which -> 0 as p/n -> 1. A vanishing eigenvalue
means an exploding inverse, hence exploding variance. Past the threshold,
over-parameterization *improves* conditioning and the error falls again.

We plot test error and the empirical smallest eigenvalue of the (normalized)
feature Gram on the same p/n axis: the error spike sits right on the MP edge.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from rmtdl import plotting
from rmtdl.plotting import PALETTE, plt


def random_feature_experiment(n, p, d, noise, rng):
    """Ridgeless random-feature regression; returns (test_mse, min_eig)."""
    teacher = rng.standard_normal(d) / np.sqrt(d)
    W = rng.standard_normal((d, p)) / np.sqrt(d)        # random feature weights

    def features(X):
        return np.tanh(X @ W)

    X_tr = rng.standard_normal((n, d))
    y_tr = X_tr @ teacher + noise * rng.standard_normal(n)
    X_te = rng.standard_normal((2000, d))
    y_te = X_te @ teacher

    F_tr = features(X_tr)
    # Minimum-norm (ridgeless) least squares via the pseudo-inverse.
    beta = np.linalg.pinv(F_tr) @ y_tr
    pred = features(X_te) @ beta
    test_mse = np.mean((pred - y_te) ** 2)

    gram = (F_tr.T @ F_tr) / n if p <= n else (F_tr @ F_tr.T) / n
    min_eig = np.linalg.eigvalsh(gram)[0]
    return test_mse, max(min_eig, 0.0)


def main():
    rng = np.random.default_rng(4)
    n, d, noise = 300, 40, 0.3
    ratios = np.concatenate([np.linspace(0.2, 0.95, 12),
                             np.linspace(0.97, 1.03, 7),
                             np.linspace(1.1, 4.0, 14)])
    ps = np.maximum(1, (ratios * n).astype(int))

    errs, eigs = [], []
    for p in ps:
        e, lam = np.mean([random_feature_experiment(n, p, d, noise, rng)
                          for _ in range(5)], axis=0)
        errs.append(e)
        eigs.append(lam)

    plotting.use_watercolor_style()
    fig, ax = plt.subplots(figsize=(8, 4.8))
    x = ps / n
    ax.plot(x, errs, "o-", color=PALETTE["clay"], lw=2.2, label="test MSE (ridgeless)")
    ax.axvline(1.0, color=PALETTE["muted"], ls=":", lw=1.2,
               label="interpolation threshold  p = n")
    ax.set_yscale("log")
    ax.set_xlabel("over-parameterization  p / n")
    ax.set_ylabel("test MSE")
    ax.set_title("Double descent:  the error spike sits on the Marchenko-Pastur edge")

    ax2 = ax.twinx()
    ax2.plot(x, eigs, "s--", color=PALETTE["sea"], lw=1.8, alpha=0.9,
             label="min eigenvalue of feature Gram")
    ax2.set_ylabel("smallest eigenvalue  ->  0 at p = n", color=PALETTE["sea"])
    ax2.grid(False)

    lines = ax.get_lines()[:2] + ax2.get_lines()[:1]
    ax.legend(lines, [l.get_label() for l in lines], fontsize=9, loc="upper right")
    plotting.save(fig, "05_double_descent.png")


if __name__ == "__main__":
    main()
