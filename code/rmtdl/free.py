"""Free additive convolution by the subordination fixed point.

Free probability replaces the classical convolution (which governs sums of
*independent scalars*) with the **free** convolution (which governs spectra of
sums of *freely independent matrices*, e.g. one matrix plus a randomly rotated
copy of another). The free additive convolution mu (+) nu is the limiting
eigenvalue distribution of A + B when A, B are asymptotically free.

We compute it through the Cauchy (Stieltjes) transform

    G_mu(z) = integral d mu(t) / (z - t),     Im z > 0,

and recover a density by the Stieltjes inversion

    rho(x) = -(1/pi) Im G(x + i*eta),     eta -> 0+.

The subordination theorem (Voiculescu; Biane; Belinschi-Bercovici) says there
exist analytic omega_1, omega_2 on the upper half-plane with

    G_{mu (+) nu}(z) = G_mu(omega_1(z)) = G_nu(omega_2(z)),

solving the coupled fixed point  (with h(w) = 1/G(w) - w)

    omega_1 = z + h_nu(omega_2),
    omega_2 = z + h_mu(omega_1).

That fixed-point iteration is the workhorse below: it sidesteps having to invert
the R-transform numerically, and converges for any z with Im z > 0.
"""

from __future__ import annotations

import numpy as np


# --------------------------------------------------------------------------- #
# Cauchy transforms                                                            #
# --------------------------------------------------------------------------- #
def cauchy_from_atoms(nodes, weights):
    """Cauchy transform of a discrete/atomic measure sum_i w_i delta_{nodes_i}.

    Pass a fine (nodes, weights) discretization to represent any measure -- this
    is the most robust route numerically, and it works for the exact two-atom
    measures in the 'free is not classical' demo too.
    """
    nodes = np.asarray(nodes, dtype=float)
    weights = np.asarray(weights, dtype=float)
    weights = weights / weights.sum()

    def G(z):
        z = np.asarray(z, dtype=complex)
        return np.sum(weights / (z[..., None] - nodes), axis=-1)

    return G


def cauchy_from_pdf(pdf, support, n_nodes: int = 4000):
    """Cauchy transform of a measure given by a density `pdf` on `support`."""
    a, b = support
    nodes = np.linspace(a, b, n_nodes)
    dx = nodes[1] - nodes[0]
    weights = np.clip(pdf(nodes), 0.0, None) * dx
    return cauchy_from_atoms(nodes, weights)


def cauchy_from_samples(eigs):
    """Cauchy transform of the empirical measure of `eigs` (an eigenvalue array)."""
    eigs = np.asarray(eigs, dtype=float)
    return cauchy_from_atoms(eigs, np.ones_like(eigs))


# --------------------------------------------------------------------------- #
# Free additive convolution                                                   #
# --------------------------------------------------------------------------- #
def free_additive_convolution(
    G_mu,
    G_nu,
    x,
    eta: float = 1e-2,
    iters: int = 5000,
    tol: float = 1e-12,
):
    """Density of mu (+) nu on the grid `x`, via the subordination fixed point.

    Parameters
    ----------
    G_mu, G_nu : callables z -> G(z)  (use the `cauchy_from_*` builders).
    x          : real grid to evaluate the resulting density on.
    eta        : imaginary regularizer; smaller = sharper but slower / noisier.
    """
    x = np.asarray(x, dtype=float)
    z = x + 1j * eta

    def h_mu(w):
        return 1.0 / G_mu(w) - w

    def h_nu(w):
        return 1.0 / G_nu(w) - w

    w1 = z.copy()
    w2 = z.copy()
    for _ in range(iters):
        w1_new = z + h_nu(w2)
        w2_new = z + h_mu(w1_new)
        # Keep the subordination functions in the upper half-plane.
        w1_new = np.where(w1_new.imag <= 0, w1_new.real + 1j * eta, w1_new)
        w2_new = np.where(w2_new.imag <= 0, w2_new.real + 1j * eta, w2_new)
        delta = max(np.max(np.abs(w1_new - w1)), np.max(np.abs(w2_new - w2)))
        w1, w2 = w1_new, w2_new
        if delta < tol:
            break

    G = G_mu(w1)
    rho = -G.imag / np.pi
    return np.clip(rho, 0.0, None)
