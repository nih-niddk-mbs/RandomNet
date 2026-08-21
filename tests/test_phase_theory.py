import sys
from pathlib import Path

import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from rn_phase import theory_phase_autocorr  # noqa: E402


def test_phase_theory_obeys_same_spike_cusp():
    I = 1.0
    alpha = 1.0
    beta = 0.8
    sigma = 1.2 * 2.0 * np.pi
    dtau = 1e-4
    n_quad = 48

    _tau, covariance, _ = theory_phase_autocorr(
        I=I,
        alpha=alpha,
        beta=beta,
        sigma=sigma,
        tau_max=5 * dtau,
        dtau=dtau,
        n_quad=n_quad,
        q_method="hermite",
    )

    nodes, weights = np.polynomial.hermite.hermgauss(n_quad)
    u = np.sqrt(2.0 * covariance[0]) * nodes
    rate = alpha * np.clip(I + u, 0.0, None) ** (1.0 / alpha) / (2.0 * np.pi)
    mean_rate = np.dot(weights, rate) / np.sqrt(np.pi)
    expected_slope = -0.5 * beta**2 * sigma**2 * mean_rate
    measured_slope = (covariance[1] - covariance[0]) / dtau

    assert measured_slope == pytest.approx(expected_slope, rel=2e-3)


def test_inflated_initial_condition_solver_is_removed():
    with pytest.raises(ValueError, match="solver='cusp'"):
        theory_phase_autocorr(solver="inflated_ic", tau_max=0.2, dtau=0.1)
