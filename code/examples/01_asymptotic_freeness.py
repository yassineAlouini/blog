"""Post 1 -- "Free is not classical."

Take A = diag(+1,...,+1,-1,...,-1) (a symmetric Bernoulli spectrum) and B = a
Haar-rotated copy. Classically you'd expect the spectrum of A + B to be the
*classical* convolution of the two +/-1 laws: atoms at {-2, 0, +2} with weights
1/4, 1/2, 1/4. Instead the eigenvalues of A + B fill the **arcsine law** on
(-2, 2) -- the *free* convolution. The randomly rotated eigenbasis makes A and B
asymptotically *free*, not classically independent.

We overlay (i) the empirical histogram of A + B, (ii) the arcsine law, and
(iii) the free convolution computed from our own subordination routine -- they
agree; the classical 3-atom guess does not.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from rmtdl import free, plotting, spectra
from rmtdl.plotting import PALETTE, plt


def bernoulli_matrix(n, rng):
    """Symmetric matrix with spectrum exactly +/-1 (half each), random eigenbasis."""
    diag = np.ones(n)
    diag[: n // 2] = -1.0
    q = spectra.haar_orthogonal(n, rng)
    return (q * diag) @ q.T


def main():
    rng = np.random.default_rng(0)
    n = 2000

    a = bernoulli_matrix(n, rng)
    b = bernoulli_matrix(n, rng)
    eig = np.linalg.eigvalsh(a + b)

    grid = np.linspace(-2.4, 2.4, 600)

    # Free convolution of two two-atom measures, from our subordination solver.
    G = free.cauchy_from_atoms([-1.0, 1.0], [0.5, 0.5])
    rho_free = free.free_additive_convolution(G, G, grid, eta=1e-2)

    plotting.use_watercolor_style()
    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    ax.hist(eig, bins=80, density=True, color=PALETTE["lblue"],
            edgecolor="white", linewidth=0.3, label=f"eigenvalues of A+B  (N={n})")
    ax.plot(grid, spectra.arcsine_pdf(grid, 2.0), color=PALETTE["clay"],
            lw=2.4, label="arcsine law  (free convolution, closed form)")
    ax.plot(grid, rho_free, "--", color=PALETTE["sea"], lw=2.0,
            label="free convolution (our subordination solver)")
    # The classical (wrong) guess: atoms at -2, 0, +2.
    for loc, w in [(-2, 0.25), (0, 0.5), (2, 0.25)]:
        ax.annotate("", xy=(loc, w * 2.2), xytext=(loc, 0),
                    arrowprops=dict(arrowstyle="-|>", color=PALETTE["muted"], lw=1.4))
    ax.scatter([], [], marker="^", color=PALETTE["muted"],
               label="classical convolution guess (atoms -2,0,+2)  -- wrong")

    ax.set_title("Free is not classical:  spectrum of A + B is the arcsine law")
    ax.set_xlabel("eigenvalue")
    ax.set_ylabel("density")
    ax.set_ylim(0, 1.2)
    ax.legend(loc="upper center", fontsize=9)
    plotting.save(fig, "01_asymptotic_freeness.png")


if __name__ == "__main__":
    main()
