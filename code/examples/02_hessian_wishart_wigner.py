"""Post 1, Bridge I -- the loss Hessian as a *sum*: Wishart (+) Wigner.

Pennington & Bahri (ICML 2017) model the loss Hessian as H = H0 + H1, with H0 a
positive-semidefinite Gauss-Newton / Wishart term and H1 an indefinite
Wigner-like term, treated as *freely independent*. The spectrum of the sum is
then the free additive convolution MP (+) semicircle.

The 'energy' (loss value) controls the size of the indefinite Wigner part. We
sweep it and watch the spectrum morph from an all-positive, gapped, MP-like bowl
(low loss: a near-convex minimum) to one that spills across zero into **negative
eigenvalues** (high loss: a saddle). We overlay the free-convolution prediction
on the empirical spectrum of an actual H0 + H1 sample.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from rmtdl import free, plotting, spectra
from rmtdl.plotting import PALETTE, plt


def main():
    rng = np.random.default_rng(1)
    p = 1500            # Hessian dimension
    gamma = 0.6         # aspect ratio of the Wishart part (p / n)
    n = int(p / gamma)

    energies = [0.15, 0.5, 1.0]   # scale of the indefinite Wigner part
    grid = np.linspace(-2.0, 4.0, 700)

    plotting.use_watercolor_style()
    fig, axes = plt.subplots(1, len(energies), figsize=(13, 4.0), sharey=True)

    # Wishart (Gauss-Newton) part, fixed across the sweep.
    wishart = spectra.sample_wishart(n, p, rng=rng)
    G_mp = free.cauchy_from_pdf(
        lambda t: spectra.marchenko_pastur_pdf(t, gamma), spectra.mp_edges(gamma)
    )

    for ax, sigma in zip(axes, energies):
        wigner = spectra.sample_wigner(p, sigma=sigma, rng=rng)
        eig = np.linalg.eigvalsh(wishart + wigner)

        G_sc = free.cauchy_from_pdf(
            lambda t, s=sigma: spectra.semicircle_pdf(t, s), (-2 * sigma, 2 * sigma)
        )
        rho = free.free_additive_convolution(G_mp, G_sc, grid, eta=1.5e-2)

        n_neg = int(np.mean(eig < 0) * 100)
        ax.hist(eig, bins=70, density=True, color=PALETTE["lblue"],
                edgecolor="white", linewidth=0.3)
        ax.plot(grid, rho, color=PALETTE["clay"], lw=2.2)
        ax.axvline(0, color=PALETTE["muted"], lw=1.0, ls=":")
        ax.set_title(f"energy (sigma) = {sigma}\nnegative eigenvalues: {n_neg}%")
        ax.set_xlabel("eigenvalue")
    axes[0].set_ylabel("density")
    fig.suptitle("Bridge I:  Hessian = Wishart (+) Wigner  -- saddles emerge as loss rises",
                 fontweight="bold")
    fig.text(0.5, -0.02,
             "bars = eigenvalues of an actual H0+H1 sample;  clay curve = free convolution prediction",
             ha="center", fontsize=9, color=PALETTE["muted"])
    plotting.save(fig, "02_hessian_wishart_wigner.png")


if __name__ == "__main__":
    main()
