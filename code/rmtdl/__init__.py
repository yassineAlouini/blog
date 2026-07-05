"""rmtdl — a tiny Random Matrix Theory toolkit for the *Random Matrices Meet
Deep Learning* blog series.

Everything is plain numpy/scipy so readers can run each example and *see* the
RMT <-> deep-learning connections for themselves.

Modules
-------
spectra   : sample classic ensembles (Wigner, Wishart) + reference laws
            (semicircle, Marchenko-Pastur, arcsine) + empirical spectral density.
free      : free additive convolution via the subordination fixed point
            (the numerical workhorse behind "Hessian = Wishart (+) Wigner").
plotting  : shared watercolor matplotlib style + a save helper.
"""

from . import spectra, free, plotting  # noqa: F401

__all__ = ["spectra", "free", "plotting"]
