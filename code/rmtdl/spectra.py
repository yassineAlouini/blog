"""Sampling classic random-matrix ensembles, reference limiting laws, and the
empirical spectral density (ESD).

Conventions
-----------
* Wigner: symmetric N x N, off-diagonal variance sigma^2 / N
  -> eigenvalues fill the **semicircle** on [-2 sigma, 2 sigma].
* Wishart / sample covariance: S = (1/n) X^T X, with X an (n x p) matrix of
  iid N(0, sigma^2) entries. Aspect ratio gamma = p / n.
  -> eigenvalues of S follow the **Marchenko-Pastur** law.
"""

from __future__ import annotations

import numpy as np


# --------------------------------------------------------------------------- #
# Sampling                                                                     #
# --------------------------------------------------------------------------- #
def sample_wigner(n: int, sigma: float = 1.0, rng=None) -> np.ndarray:
    """Symmetric Wigner matrix; eigenvalues -> semicircle of radius 2*sigma."""
    rng = np.random.default_rng() if rng is None else rng
    g = rng.standard_normal((n, n))
    return (g + g.T) / np.sqrt(2.0 * n) * sigma


def sample_wishart(n: int, p: int, sigma: float = 1.0, rng=None) -> np.ndarray:
    """Sample-covariance matrix S = (1/n) X^T X, p x p. gamma = p / n."""
    rng = np.random.default_rng() if rng is None else rng
    x = rng.standard_normal((n, p)) * sigma
    return x.T @ x / n


def haar_orthogonal(n: int, rng=None) -> np.ndarray:
    """A Haar-distributed (uniformly random) orthogonal matrix."""
    rng = np.random.default_rng() if rng is None else rng
    a = rng.standard_normal((n, n))
    q, r = np.linalg.qr(a)
    # Fix the signs so Q is genuinely Haar-distributed.
    q *= np.sign(np.diag(r))
    return q


# --------------------------------------------------------------------------- #
# Empirical spectral density                                                   #
# --------------------------------------------------------------------------- #
def esd(matrix: np.ndarray, bins: int = 80, value_range=None):
    """Empirical spectral density of a symmetric matrix.

    Returns (bin_centers, density, eigenvalues).
    """
    eig = np.linalg.eigvalsh(matrix)
    density, edges = np.histogram(eig, bins=bins, range=value_range, density=True)
    centers = 0.5 * (edges[:-1] + edges[1:])
    return centers, density, eig


# --------------------------------------------------------------------------- #
# Reference limiting laws                                                       #
# --------------------------------------------------------------------------- #
def semicircle_pdf(x, sigma: float = 1.0):
    """Wigner semicircle density on [-2 sigma, 2 sigma] (the *free* Gaussian)."""
    x = np.asarray(x, dtype=float)
    radius = 2.0 * sigma
    out = np.zeros_like(x)
    inside = np.abs(x) < radius
    out[inside] = np.sqrt(radius**2 - x[inside] ** 2) / (2.0 * np.pi * sigma**2)
    return out


def marchenko_pastur_pdf(x, gamma: float, sigma: float = 1.0):
    """Marchenko-Pastur density (continuous part) for ratio gamma = p / n.

    Support [lam_minus, lam_plus]; when gamma > 1 there is also an atom of mass
    (1 - 1/gamma) at 0 (not returned here -- it is a delta, not a density).
    """
    x = np.asarray(x, dtype=float)
    lam_minus = sigma**2 * (1.0 - np.sqrt(gamma)) ** 2
    lam_plus = sigma**2 * (1.0 + np.sqrt(gamma)) ** 2
    out = np.zeros_like(x)
    inside = (x > lam_minus) & (x < lam_plus)
    out[inside] = np.sqrt((lam_plus - x[inside]) * (x[inside] - lam_minus)) / (
        2.0 * np.pi * gamma * sigma**2 * x[inside]
    )
    return out


def mp_edges(gamma: float, sigma: float = 1.0):
    """The Marchenko-Pastur support edges (lam_minus, lam_plus)."""
    return sigma**2 * (1.0 - np.sqrt(gamma)) ** 2, sigma**2 * (1.0 + np.sqrt(gamma)) ** 2


def arcsine_pdf(x, radius: float = 2.0):
    """Arcsine law on (-radius, radius).

    This is the *free* convolution of two symmetric Bernoulli(+/-1) laws -- the
    headline 'free is not classical' example (classical convolution would give a
    three-point distribution at {-2, 0, +2}).
    """
    x = np.asarray(x, dtype=float)
    out = np.zeros_like(x)
    inside = np.abs(x) < radius
    out[inside] = 1.0 / (np.pi * np.sqrt(radius**2 - x[inside] ** 2))
    return out
