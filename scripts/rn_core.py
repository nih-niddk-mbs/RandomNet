"""Shared helpers for RandomNet simulations."""

import os
from pathlib import Path

import numpy as np
from numpy.fft import fft, ifft

rng = np.random.default_rng(42)


def default_results_dir(*parts):
    """Return a writable results directory outside the repository.

    ``RANDOMNET_RESULTS_DIR`` has priority.  The portable fallback is
    ``~/randomnet-results``.
    """
    root = os.environ.get("RANDOMNET_RESULTS_DIR")
    if root is None:
        root = Path.home() / "randomnet-results"
    return str(Path(root).expanduser().joinpath(*parts))


def autocorr(x, max_lag):
    """Unbiased autocorrelation via FFT, normalised so C(0)=variance."""
    n = len(x)
    max_lag = min(max_lag, n - 1)  # Can't correlate beyond series length
    xc = x - x.mean()
    full = np.real(ifft(np.abs(fft(xc, n=2 * n)) ** 2))[:max_lag]
    nrm = n - np.arange(max_lag)  # unbiased normalisation
    return full / nrm


def make_weights(N, sigma, lam=1, rng=rng):
    """
    Draw NxN Gaussian weights with std sigma/sqrt(N).
    lam=1  -> row-sum corrected (W @ 1 = 0)
    lam=0  -> plain Gaussian
    """
    W = rng.normal(0, sigma / np.sqrt(N), (N, N))
    np.fill_diagonal(W, 0)          # zero diagonal first
    if lam:
        if N <= 1:
            return W
        # Correct only off-diagonal entries so the zero diagonal is preserved
        # while each row sum is exactly removed.
        offdiag = ~np.eye(N, dtype=bool)
        row_sums = W.sum(axis=1, keepdims=True)
        W[offdiag] -= np.repeat(row_sums / (N - 1), N, axis=1)[offdiag]
    return W
