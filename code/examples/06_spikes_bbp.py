"""Post 6 -- signal vs noise: spikes and the BBP transition.

Add a rank-1 signal of strength theta to a Wigner noise bulk:
    M = Wigner(sigma=1)  +  theta * v v^T,   ||v|| = 1.
The bulk fills the semicircle on [-2, 2]. The signal produces a detached outlier
eigenvalue **only when theta exceeds the BBP threshold** theta_c = 1 (= sigma):
below it the signal hides inside the bulk; above it the top eigenvalue pops out
to theta + 1/theta. This sharp signal/noise threshold is exactly what's behind
the few 'outlier' eigenvalues of trained-network Hessians and Gram matrices
(whose count and structure track the label/class structure).
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from rmtdl import plotting, spectra
from rmtdl.plotting import PALETTE, plt


def top_eigs_vs_theta(n, thetas, rng):
    v = rng.standard_normal(n)
    v /= np.linalg.norm(v)
    noise = spectra.sample_wigner(n, sigma=1.0, rng=rng)
    tops = []
    for theta in thetas:
        M = noise + theta * np.outer(v, v)
        tops.append(np.linalg.eigvalsh(M)[-1])
    return np.array(tops)


def main():
    rng = np.random.default_rng(5)
    n = 1500
    thetas = np.linspace(0.0, 2.5, 26)
    tops = top_eigs_vs_theta(n, thetas, rng)

    # BBP prediction: outlier stays at the bulk edge (2) below theta_c = 1,
    # then detaches to theta + 1/theta.
    theta_c = 1.0
    pred = np.where(thetas <= theta_c, 2.0, thetas + 1.0 / np.maximum(thetas, 1e-9))

    plotting.use_watercolor_style()
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 4.6))

    # (left) two example spectra: below and above threshold.
    v = rng.standard_normal(n); v /= np.linalg.norm(v)
    noise = spectra.sample_wigner(n, sigma=1.0, rng=rng)
    grid = np.linspace(-2.6, 3.2, 400)
    for theta, color, lbl in [(0.6, PALETTE["blue"], "theta = 0.6  (below BBP: hidden)"),
                              (1.8, PALETTE["clay"], "theta = 1.8  (above BBP: outlier!)")]:
        eig = np.linalg.eigvalsh(noise + theta * np.outer(v, v))
        ax1.hist(eig, bins=80, density=True, histtype="step", lw=1.8,
                 color=color, label=lbl)
        ax1.axvline(eig[-1], color=color, ls=":", lw=1.2)
    ax1.plot(grid, spectra.semicircle_pdf(grid, 1.0), color=PALETTE["muted"],
             lw=1.6, label="semicircle bulk")
    ax1.set_title("a rank-1 signal only escapes the bulk past the threshold")
    ax1.set_xlabel("eigenvalue"); ax1.set_ylabel("density")
    ax1.legend(fontsize=8.5)

    # (right) top eigenvalue vs theta -- the BBP phase transition.
    ax2.plot(thetas, tops, "o", color=PALETTE["clay"], label="empirical top eigenvalue")
    ax2.plot(thetas, pred, color=PALETTE["sea"], lw=2.2, label="BBP prediction")
    ax2.axvline(theta_c, color=PALETTE["muted"], ls=":", lw=1.2, label="threshold theta_c = 1")
    ax2.axhline(2.0, color=PALETTE["muted"], ls="--", lw=0.8, alpha=0.7)
    ax2.set_title("BBP transition:  signal detaches from noise at theta_c")
    ax2.set_xlabel("signal strength  theta"); ax2.set_ylabel("largest eigenvalue")
    ax2.legend(fontsize=9)

    fig.suptitle("Spikes & BBP:  a sharp threshold separates learned signal from random noise",
                 fontweight="bold")
    plotting.save(fig, "06_spikes_bbp.png")


if __name__ == "__main__":
    main()
