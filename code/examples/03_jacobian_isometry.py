"""Post 1, Bridge II -- the input-output Jacobian as a *product*, and dynamical
isometry.

The Jacobian of an L-layer network is a product  J = prod_l D_l W_l,  where W_l
are the weights and D_l = diag(phi'(preactivations)) is the nonlinearity's local
gain. Free *multiplicative* convolution (the S-transform) predicts the singular
spectrum of such a product. The headline consequence: with the right init you
can make the **entire** singular spectrum concentrate near 1 even for very deep
nets -- 'dynamical isometry' -- which is what lets 10,000-layer vanilla nets
train at all (Pennington, Schoenholz & Ganguli 2017).

We don't need the S-transform to *see* it: just build the product three ways and
look at the singular values vs depth.

  * Gaussian weights (linear): singular values spread exponentially -> the
    Jacobian's condition number explodes with depth (vanishing/exploding signal).
  * Orthogonal weights (linear): every singular value is exactly 1 -> perfect
    isometry at any depth.
  * Orthogonal weights + tanh (near the linear regime): a *smooth* nonlinearity
    keeps the spectrum concentrated near 1 -- (approximate) dynamical isometry.

A deliberate fourth curve makes the surprise sharp: **ReLU cannot achieve
dynamical isometry** even with orthogonal weights (Pennington, Schoenholz &
Ganguli 2018). The independent on/off masking drives the product to rank
collapse -- its singular values plunge by hundreds of orders of magnitude -- so
the choice of nonlinearity, not just the weights, is what unlocks trainable depth.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from rmtdl import plotting, spectra
from rmtdl.plotting import PALETTE, plt


def jacobian_singular_values(depth, n, kind, rng):
    # Forward-propagate a small signal so the nonlinearity's local gain
    # D_l = diag(phi'(preactivation)) is realistic, not made up.
    h = rng.standard_normal(n) * 0.3
    J = np.eye(n)
    for _ in range(depth):
        if kind == "gaussian":
            W = rng.standard_normal((n, n)) / np.sqrt(n)   # critical linear scaling
            D = np.ones(n)
        elif kind == "orthogonal":
            W = spectra.haar_orthogonal(n, rng)
            D = np.ones(n)
        elif kind == "orthogonal+tanh":
            W = spectra.haar_orthogonal(n, rng)            # gain 1
            pre = W @ h
            D = 1.0 - np.tanh(pre) ** 2                    # tanh'(pre)
            h = np.tanh(pre)
        elif kind == "orthogonal+relu":
            W = spectra.haar_orthogonal(n, rng) * np.sqrt(2.0)  # gain for ReLU
            pre = W @ h
            D = (pre > 0).astype(float)                    # ReLU'(pre): on/off
            h = np.maximum(pre, 0.0)
        else:
            raise ValueError(kind)
        J = (D[:, None] * W) @ J
    return np.linalg.svd(J, compute_uv=False)


def main():
    rng = np.random.default_rng(2)
    n = 256
    depths = [1, 2, 4, 8, 16, 32, 64]
    kinds = [("gaussian", PALETTE["clay"]),
             ("orthogonal", PALETTE["sea"]),
             ("orthogonal+tanh", PALETTE["blue"]),
             ("orthogonal+relu", PALETTE["muted"])]

    plotting.use_watercolor_style()
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 4.6))

    # (left) spread of the log-singular-values vs depth. This stays finite even
    # when the raw condition number overflows: for the Gaussian product the
    # singular values fan out ~linearly in depth (Lyapunov spread); orthogonal
    # stays pinned at 0 spread (perfect isometry); the ReLU init grows slowly.
    for kind, color in kinds:
        spreads = []
        for L in depths:
            s = jacobian_singular_values(L, n, kind, rng)
            log_s = np.log10(np.clip(s, 1e-300, None))
            spreads.append(log_s.std())
        ax1.plot(depths, spreads, "o-", color=color, lw=2.0, label=kind)
    ax1.set_title("Singular values fan out with depth -- unless you tune the init")
    ax1.set_xlabel("depth  L")
    ax1.set_ylabel("spread of log10(singular values)  (std)")
    ax1.legend(fontsize=9)

    # (right) the actual singular spectrum at a fixed deep depth (sorted, log y).
    # A line plot avoids the delta-at-1 histogram spike of the orthogonal case.
    L = 64
    for kind, color in kinds:
        s = np.sort(jacobian_singular_values(L, n, kind, rng))[::-1]
        ax2.semilogy(np.linspace(0, 1, len(s)), np.clip(s, 1e-300, None),
                     color=color, lw=2.0, label=kind)
    ax2.axhline(1.0, color=PALETTE["muted"], ls=":", lw=1.0, label="isometry (s = 1)")
    ax2.set_title(f"singular spectrum at depth L = {L}")
    ax2.set_xlabel("normalized index")
    ax2.set_ylabel("singular value (log scale)")
    ax2.legend(fontsize=9)

    fig.suptitle("Bridge II:  dynamical isometry -- keeping every Jacobian singular value near 1",
                 fontweight="bold", y=1.02)
    fig.tight_layout()
    plotting.save(fig, "03_jacobian_isometry.png")


if __name__ == "__main__":
    main()
