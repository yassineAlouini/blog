"""Post 2 -- heavy tails know your generalization.

At initialization a weight matrix is essentially random and its eigenvalue
spectrum (of W W^T) is a clean Marchenko-Pastur bulk. As training proceeds the
spectrum develops a **heavy (power-law) tail**: rho(lambda) ~ lambda^(-(1+mu)).
Heavy-Tailed Self-Regularization (Martin & Mahoney) finds that the fitted tail
exponent correlates with test performance -- *without ever touching the test
set*. The `weightwatcher` package operationalizes this.

Here we contrast a random (MP) matrix with a synthetic 'trained' matrix carrying
a power-law tail, and estimate the tail exponent with a simple Hill estimator.
(Swap in a real trained layer's W to reproduce the effect on your own model.)
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from rmtdl import plotting, spectra
from rmtdl.plotting import PALETTE, plt


def hill_tail_index(values, tail_frac=0.1):
    """Hill estimator of the power-law **tail index** of a set of values.

    Mind the convention: this estimates the exponent of the eigenvalue
    *distribution*'s tail, P(lambda > x) ~ x^(-index). It is NOT the same number
    as the eigenvalue-vs-rank slope (that slope is the construction exponent
    alpha below), nor the HTSR alpha of Martin & Mahoney (who fit the spectral
    *density* rho(lambda) ~ lambda^(-alpha_HTSR), typically ~2-6). For a spectrum
    built as lambda_(i) ~ i^(-alpha) this index comes out near 1/alpha.
    """
    v = np.sort(values)[::-1]
    k = max(2, int(tail_frac * len(v)))
    top = v[:k]
    return 1.0 / np.mean(np.log(top / top[-1]))


def heavy_tailed_matrix(n, p, alpha, rng):
    """A p x p PSD matrix whose eigenvalues carry a power-law tail (a stylized
    'trained' weight Gram matrix)."""
    base = spectra.sample_wishart(n, p, rng=rng)          # MP bulk
    evals, evecs = np.linalg.eigh(base)
    # Reshape the spectrum to a power law lambda_i ~ i^(-alpha).
    ranks = np.arange(1, p + 1)[::-1]
    heavy = (ranks / p) ** (-alpha)
    heavy *= evals.sum() / heavy.sum()
    return (evecs * heavy) @ evecs.T


def main():
    rng = np.random.default_rng(3)
    n, p = 1200, 800
    gamma = p / n

    alpha_build = 1.6  # lambda_(i) ~ i^-alpha_build  -> rank-plot slope = -alpha_build
    rand = spectra.sample_wishart(n, p, rng=rng)
    trained = heavy_tailed_matrix(n, p, alpha=alpha_build, rng=rng)

    e_rand = np.linalg.eigvalsh(rand)
    e_trained = np.linalg.eigvalsh(trained)
    tail_index = hill_tail_index(e_trained, tail_frac=0.15)  # ~ 1 / alpha_build

    plotting.use_watercolor_style()
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 4.6))

    # (left) the spectra
    ax1.hist(e_rand, bins=70, density=True, color=PALETTE["lblue"],
             edgecolor="white", linewidth=0.3, label="random init  (MP bulk)")
    grid = np.linspace(e_rand.min(), e_rand.max(), 400)
    ax1.plot(grid, spectra.marchenko_pastur_pdf(grid, gamma), color=PALETTE["sea"],
             lw=2.2, label="Marchenko-Pastur")
    ax1.hist(e_trained, bins=200, density=True, color=PALETTE["clay"], alpha=0.45,
             label="'trained'  (heavy tail)")
    ax1.set_xlim(0, np.quantile(e_rand, 0.999) * 1.5)
    ax1.set_title("Init looks like Marchenko-Pastur; training grows a tail")
    ax1.set_xlabel("eigenvalue")
    ax1.set_ylabel("density")
    ax1.legend(fontsize=9)

    # (right) eigenvalue-vs-rank on log-log axes: the 'trained' spectrum is a
    # straight line of slope -alpha_build. NOTE this rank slope (alpha_build) and
    # the Hill *tail index* (~1/alpha_build) are two different exponents -- don't
    # conflate them.
    for e, color, lbl in [(e_rand, PALETTE["sea"], "random init"),
                          (e_trained, PALETTE["clay"], "'trained'")]:
        v = np.sort(e[e > 0])[::-1]
        ax2.loglog(np.arange(1, len(v) + 1), v, color=color, lw=2.0, label=lbl)
    ax2.set_title(f"eigenvalue vs rank  (rank slope alpha = {alpha_build:.1f}, "
                  f"Hill tail index = {tail_index:.2f})")
    ax2.set_xlabel("rank")
    ax2.set_ylabel("eigenvalue")
    ax2.legend(fontsize=9)

    fig.suptitle("Heavy-Tailed Self-Regularization:  the weight spectrum is a generalization gauge",
                 fontweight="bold", y=1.03)
    fig.tight_layout()
    plotting.save(fig, "04_heavy_tails.png")


if __name__ == "__main__":
    main()
